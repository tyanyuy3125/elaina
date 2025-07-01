#pragma once

#include "core/common.h"
#include "cuda.h"
#include "core/device/buffer.h"
#include "core/device/cuda.h"
#include "core/device/context.h"
#include "core/device/atomic.h"
#include "core/interop.h"

#include "util/sampling.h"

#include "integrator/uniform/workqueue.h"
#include "integrator/guided/train.h"
#include "integrator/guided/guideditem.h"
#include "integrator/guided/guided.h"

#include "tiny-cuda-nn/common.h"

using precision_t = tcnn::network_precision_t;

ELAINA_NAMESPACE_BEGIN

enum class ELoss
{
	L1,
	L2,
	LogL1,
	RelL2,
	NumTypes
};

enum class EDivergence
{
	KL,
	ChiSquare,
	NumTypes
};

enum class EOutputActivation
{
	None = 0,
	ReLU,
	Logistic,
	Exponential,
	Softplus,
	NumTypes
};

constexpr auto ACTIVATION_LAMBDA = EOutputActivation::Exponential;
constexpr auto ACTIVATION_KAPPA = EOutputActivation::Exponential;
constexpr auto ACTIVATION_COORDINATES = EOutputActivation::None;
constexpr auto ACTIVATION_SELECTION_PROBABILITY = EOutputActivation::Logistic;

constexpr auto LR_LAMBDA = 1.f;
constexpr auto LR_KAPPA = 1.f;
constexpr auto LR_COORDINATES = 1.f;
constexpr auto LR_SELECTION_PROBABILITY = 1.f;

ELAINA_CALLABLE float network_to_params(float val, EOutputActivation activation)
{
	static constexpr float exp_clamp_min = -10, exp_clamp_max = 15;
	switch (activation)
	{
	case EOutputActivation::None:
		return val;
	case EOutputActivation::ReLU:
		return val > 0.0f ? val : 0.0f;
	case EOutputActivation::Logistic:
		return logistic(val);
	case EOutputActivation::Exponential:
		return expf(clamp(val, exp_clamp_min, exp_clamp_max));
	case EOutputActivation::Softplus:
		return logf((1 + expf(val)));
	default:
		assert(false);
	}
	return 0.0f;
}

ELAINA_CALLABLE float d_network_to_d_params(float val, EOutputActivation activation)
{
	static constexpr float exp_clamp_min = -10, exp_clamp_max = 15;
	switch (activation)
	{
	case EOutputActivation::None:
		return 1.0f;
	case EOutputActivation::ReLU:
		return val > 0.0f ? 1.0f : 0.0f;
	case EOutputActivation::Logistic:
	{
		float fval = logistic(val);
		return fval * (1 - fval);
	};
	case EOutputActivation::Exponential:
		return expf(clamp(val, exp_clamp_min, exp_clamp_max));
	case EOutputActivation::Softplus:
	{
		float fval = expf(val);
		return fval / (fval + 1);
	};
	default:
		assert(false);
	}
	return 0.0f;
}

ELAINA_CALLABLE float loss_and_derivatives(ELoss type, float pred, float target, float *derivative = nullptr)
{
	switch (type)
	{
	case ELoss::L1:
	{
		float diff = pred - target;
		if (derivative)
			*derivative = copysignf(1, diff);
		return fabs(diff);
	}
	case ELoss::L2:
	{
		float diff = pred - target;
		if (derivative)
			*derivative = 2 * diff;
		return pow2(diff);
	}
	case ELoss::LogL1:
	{
		float diff = pred - target;
		float divisor = fabs(diff) + 1;
		if (derivative)
			*derivative = copysignf(1 / divisor, diff);
		return log(divisor);
	}
	case ELoss::RelL2:
	{
		float diff = pred - target;
		float factor = 1 / (pow2(pred) + M_EPSILON);
		if (derivative)
			*derivative = 2 * diff * factor;
		return pow2(diff) * factor;
	}
	default:
		assert(false);
	}
	return 0;
}

template <typename VectorType, typename AABBType>
ELAINA_CALLABLE VectorType normalizeSpatialCoord(const VectorType &coord, AABBType aabb)
{
	using T = typename VectorType::Scalar;
	constexpr T inflation = static_cast<T>(0.005);
	aabb.inflate(aabb.diagonal().norm() * inflation);
	return VectorType{T(0.5)} + (coord - aabb.center()) / aabb.diagonal();
}

namespace common2d
{
	template <typename T>
	class DeviceBuffer
	{
	public:
		DeviceBuffer() = default;

		ELAINA_HOST DeviceBuffer(int n) : mMaxSize(n)
		{
			cudaMalloc(&mData, n * sizeof(T));
		}

		ELAINA_CALLABLE int push(const T &w)
		{
			int index = allocateEntry();
			DCHECK_LT(index, mMaxSize);
			(*this)[index % mMaxSize] = w;
			return index;
		}

		ELAINA_CALLABLE void clear()
		{
			mSize.store(0);
		}

		ELAINA_CALLABLE int size() const
		{
			return mSize.load();
		}

		ELAINA_CALLABLE T *data() { return mData; }

		ELAINA_CALLABLE T &operator[](int index)
		{
			DCHECK_LT(index, mMaxSize);
			return mData[index];
		}

		ELAINA_CALLABLE DeviceBuffer &operator=(const DeviceBuffer &w)
		{
			mSize.store(w.mSize);
			mMaxSize = w.mMaxSize;
			return *this;
		}

	protected:
		ELAINA_CALLABLE int allocateEntry()
		{
			return mSize.fetch_add(1);
		}

	private:
		atomic<int> mSize;
		T *mData;
		int mMaxSize{0};
	};

	class TrainBuffer
	{
	public:
		using GuidedInput = common2d::GuidedInput;
		using GuidedOutput = common2d::GuidedOutput;
		using WalkRecordItem = common2d::WalkRecordItem;

		TrainBuffer() = default;

		ELAINA_HOST TrainBuffer(int n) : mMaxSize(n)
		{
			cudaMalloc(&mInputs, n * sizeof(GuidedInput));
			cudaMalloc(&mOutputs, n * sizeof(GuidedOutput));
		}

		ELAINA_CALLABLE int push(const GuidedInput &input,
								 const GuidedOutput &output)
		{
			int index = allocateEntry();
			DCHECK_LT(index, mMaxSize);
			mInputs[index] = input;
			mOutputs[index] = output;
			return index;
		}

		ELAINA_CALLABLE void clear()
		{
			mSize.store(0);
		}

		ELAINA_CALLABLE int size() const
		{
#ifndef ELAINA_DEVICE_CODE
			CUDA_SYNC_CHECK();
			cudaDeviceSynchronize();
#endif
			return mSize.load();
		}

		ELAINA_CALLABLE void resize(int n)
		{
			if (mMaxSize)
			{
				cudaFree(mInputs);
				cudaFree(mOutputs);
			}
			cudaMalloc(&mInputs, n * sizeof(GuidedInput));
			cudaMalloc(&mOutputs, n * sizeof(GuidedOutput));
		}

		ELAINA_CALLABLE GuidedInput *inputs() const { return mInputs; }

		ELAINA_CALLABLE GuidedOutput *outputs() const { return mOutputs; }

		ELAINA_CALLABLE TrainBuffer &operator=(const TrainBuffer &w)
		{
			mSize.store(w.mSize);
			mMaxSize = w.mMaxSize;
			return *this;
		}

	private:
		ELAINA_CALLABLE int allocateEntry()
		{
			return mSize.fetch_add(1);
		}

		atomic<int> mSize;
		GuidedInput *mInputs;
		GuidedOutput *mOutputs;
		int mMaxSize{0};
	};
}

namespace common3d
{

	template <typename T>
	class DeviceBuffer
	{
	public:
		DeviceBuffer() = default;

		ELAINA_HOST DeviceBuffer(int n) : mMaxSize(n)
		{
			cudaMalloc(&mData, n * sizeof(T));
		}

		ELAINA_CALLABLE int push(const T &w)
		{
			int index = allocateEntry();
			DCHECK_LT(index, mMaxSize);
			(*this)[index % mMaxSize] = w;
			return index;
		}

		ELAINA_CALLABLE void clear()
		{
			mSize.store(0);
		}

		ELAINA_CALLABLE int size() const
		{
			return mSize.load();
		}

		ELAINA_CALLABLE T *data() { return mData; }

		ELAINA_CALLABLE T &operator[](int index)
		{
			DCHECK_LT(index, mMaxSize);
			return mData[index];
		}

		ELAINA_CALLABLE DeviceBuffer &operator=(const DeviceBuffer &w)
		{
			mSize.store(w.mSize);
			mMaxSize = w.mMaxSize;
			return *this;
		}

	protected:
		ELAINA_CALLABLE int allocateEntry()
		{
			return mSize.fetch_add(1);
		}

	private:
		atomic<int> mSize;
		T *mData;
		int mMaxSize{0};
	};

	class TrainBuffer
	{
	public:
		using GuidedInput = common3d::GuidedInput;
		using GuidedOutput = common3d::GuidedOutput;
		using WalkRecordItem = common3d::WalkRecordItem;

		TrainBuffer() = default;

		ELAINA_HOST TrainBuffer(int n) : mMaxSize(n)
		{
			cudaMalloc(&mInputs, n * sizeof(GuidedInput));
			cudaMalloc(&mOutputs, n * sizeof(GuidedOutput));
		}

		ELAINA_CALLABLE int push(const GuidedInput &input,
								 const GuidedOutput &output)
		{
			int index = allocateEntry();
			DCHECK_LT(index, mMaxSize);
			mInputs[index] = input;
			mOutputs[index] = output;
			return index;
		}

		ELAINA_CALLABLE void clear()
		{
			mSize.store(0);
		}

		ELAINA_CALLABLE int size() const
		{
#ifndef ELAINA_DEVICE_CODE
			CUDA_SYNC_CHECK();
			cudaDeviceSynchronize();
#endif
			return mSize.load();
		}

		ELAINA_CALLABLE void resize(int n)
		{
			if (mMaxSize)
			{
				cudaFree(mInputs);
				cudaFree(mOutputs);
			}
			cudaMalloc(&mInputs, n * sizeof(GuidedInput));
			cudaMalloc(&mOutputs, n * sizeof(GuidedOutput));
		}

		ELAINA_CALLABLE GuidedInput *inputs() const { return mInputs; }

		ELAINA_CALLABLE GuidedOutput *outputs() const { return mOutputs; }

		ELAINA_CALLABLE TrainBuffer &operator=(const TrainBuffer &w)
		{
			mSize.store(w.mSize);
			mMaxSize = w.mMaxSize;
			return *this;
		}

	private:
		ELAINA_CALLABLE int allocateEntry()
		{
			return mSize.fetch_add(1);
		}

		atomic<int> mSize;
		GuidedInput *mInputs;
		GuidedOutput *mOutputs;
		int mMaxSize{0};
	};
}

template <typename TrainBuffer, typename GuidedPixelStateBuffer, typename AABBType>
__global__ void generate_training_data(const size_t nElements,
									   uint trainPixelOffset, uint trainPixelStride,
									   TrainBuffer *trainBuffer,
									   GuidedPixelStateBuffer *guidedPixelStateBuffer,
									   AABBType *sceneAABB)
{
	// this costs about 0.5ms
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements)
		return;
	int pixelId = trainPixelOffset + tid * trainPixelStride;

	int depth = guidedPixelStateBuffer->curDepth[pixelId];
	for (int curDepth = 0; curDepth < depth; curDepth++)
	{
		typename TrainBuffer::GuidedInput input = {};
		typename TrainBuffer::GuidedOutput output = {};

		const typename TrainBuffer::WalkRecordItem &record = guidedPixelStateBuffer->records[curDepth][pixelId];
		input.pos = normalizeSpatialCoord(record.pos, *sceneAABB);

		Color solution = Color::Zero();
		for (int ch = 0; ch < Color::dim; ++ch)
		{
			if (std::abs(record.thp[ch]) > M_EPSILON)
			{
				solution[ch] = record.solution[ch] / record.thp[ch];
			}
		}
		output.solution = solution.abs();
		output.dir = record.dir;
		output.dirPdf = record.dirPdf;
		output.isOnNeumannBoundary = record.isOnNeumannBoundary;
		output.neumannBoundaryNormal = record.neumannBoundaryNormal;

		if (sceneAABB->contains(record.pos))
		{
			bool condition = !(input.pos.hasNaN() || output.dir.hasNaN() || isnan(output.dirPdf) || output.dirPdf == 0 || solution.hasNaN());
			if (condition)
			{
				trainBuffer->push(input, output);
			}
			else
			{
				printf("Find invalid training sample! (quite not expected...\n");
			}
		}
	}
}

template <typename OutShellPointQueue, typename AABBType>
__global__ void generate_inference_data(const size_t nElements,
										OutShellPointQueue *outShellPointQueue, float *data, AABBType *sceneAABB)
{
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= outShellPointQueue->size())
		return;
	static constexpr auto N_DIM_INPUT = std::is_same_v<AABBType, AABB3f> ? common3d::N_DIM_INPUT : common2d::N_DIM_INPUT;
	uint data_idx = i * N_DIM_INPUT;
	using pos_type = std::conditional_t<std::is_same_v<AABBType, AABB3f>, Vector3f, Vector2f>;
	using OutShellPointWorkItem = std::conditional_t<std::is_same_v<AABBType, AABB3f>, common3d::OutShellPointWorkItem, common2d::OutShellPointWorkItem>;
	pos_type normalized_pos = normalizeSpatialCoord(outShellPointQueue->operator[](i).operator OutShellPointWorkItem().point, *sceneAABB);
	*(pos_type *)&data[data_idx] = normalized_pos;
}

template <uint DIM, uint N>
class VMM;

template <typename GuidedOutput>
__global__ void compute_dL_doutput_divergence(
	const size_t nElements /* number of threads */,
	precision_t *outputPrediction /*[input] 4 x N_MIXTURE */,
	GuidedOutput *outputReference /*[input] 3(RGB), MC estimate */,
	precision_t *dL_doutput /*[output] 4 x N_MIXTURE */,
	float *likelihood /*[output] 1 */,
	float loss_scale /*[input] scale the loss so that it wont be too smol*/,
	EDivergence divergence_type /*[input] divergence type */)
{
	static constexpr auto IS_R3 = std::is_same_v<GuidedOutput, common3d::GuidedOutput>;
	static constexpr auto DIM = IS_R3 ? 3 : 2;
	static constexpr auto N_DIM_PADDED_OUTPUT = IS_R3 ? common3d::N_DIM_PADDED_OUTPUT : common2d::N_DIM_PADDED_OUTPUT;
	static constexpr auto NUM_VMF_COMPONENTS = IS_R3 ? common3d::NUM_VMF_COMPONENTS : common2d::NUM_VMF_COMPONENTS;
	static constexpr auto N_DIM_VMF = IS_R3 ? common3d::N_DIM_VMF : common2d::N_DIM_VMF;

	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements)
		return;

	loss_scale /= nElements;
	precision_t *data = outputPrediction + tid * N_DIM_PADDED_OUTPUT; // Raw data from the network
	precision_t *gradient_data = dL_doutput + tid * N_DIM_PADDED_OUTPUT;
	VMM<DIM, NUM_VMF_COMPONENTS> dist(data);

	auto wi = outputReference[tid].dir;
	float Li = outputReference[tid].solution.mean();
	float dirPdf = outputReference[tid].dirPdf + M_EPSILON;	// Reference PDF
	bool isOnNeumannBoundary = outputReference[tid].isOnNeumannBoundary; // Neumann boundary flag
	auto neumannBoundaryNormal = outputReference[tid].neumannBoundaryNormal; // Neumann boundary normal

	float guidePdf = dist.gradients_probability(wi, isOnNeumannBoundary, neumannBoundaryNormal, gradient_data) + M_EPSILON; // Model generated PDF
	float prefix = -Li / dirPdf / guidePdf * loss_scale;
	likelihood[tid] = -Li / dirPdf * logf(guidePdf);

	for (int sg = 0; sg < NUM_VMF_COMPONENTS; sg++)
	{
		precision_t *cur_params = data + sg * N_DIM_VMF;
		float lambda_r = cur_params[0], kappa_r = cur_params[1],
			  x_r = cur_params[2], y_r = cur_params[3];
		precision_t *cur_gradient = gradient_data + sg * N_DIM_VMF;
		cur_gradient[0] = LR_LAMBDA * prefix * (float)cur_gradient[0] * d_network_to_d_params(lambda_r, ACTIVATION_LAMBDA);
		cur_gradient[1] = LR_KAPPA * prefix * (float)cur_gradient[1] * d_network_to_d_params(kappa_r, ACTIVATION_KAPPA);
		cur_gradient[2] = LR_COORDINATES * prefix * (float)cur_gradient[2] * d_network_to_d_params(x_r, ACTIVATION_COORDINATES);
		cur_gradient[3] = LR_COORDINATES * prefix * (float)cur_gradient[3] * d_network_to_d_params(y_r, ACTIVATION_COORDINATES);
		if constexpr (IS_R3)
		{
			float z_r = cur_params[4];
			cur_gradient[4] = LR_COORDINATES * prefix * (float)cur_gradient[4] * d_network_to_d_params(z_r, ACTIVATION_COORDINATES);
		}
	}

	// Selection probability
	precision_t *sp_params = data + NUM_VMF_COMPONENTS * N_DIM_VMF;
	precision_t *sp_gradient = gradient_data + NUM_VMF_COMPONENTS * N_DIM_VMF;
	constexpr float e = 0.2f;
	float uniform_probability = uniformSampleSpherePDF<DIM>();
	if (isOnNeumannBoundary)
	{
		uniform_probability = uniformSampleHemispherePDF<DIM>();
	}
	sp_gradient[0] = LR_SELECTION_PROBABILITY * loss_scale * (-e) * Li * (guidePdf - uniform_probability) / pow2(dirPdf) * d_network_to_d_params(sp_params[0], ACTIVATION_SELECTION_PROBABILITY);
}

ELAINA_NAMESPACE_END