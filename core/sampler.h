#pragma once
#include "core/common.h"

#include "util/math_utils.h"
#include "taggedptr.h"
#include "util/hash.h"

ELAINA_NAMESPACE_BEGIN

class PCGSampler
{
#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL
public:
	PCGSampler() = default;

	ELAINA_CALLABLE void initialize() {};

	ELAINA_CALLABLE void setSeed(uint64_t initstate, uint64_t initseq = 1)
	{
		state = 0U;
		inc = (initseq << 1u) | 1u;
		nextUint();
		state += initstate;
		nextUint();
	}

	ELAINA_CALLABLE void setPixelSample(Vector2ui samplePixel, uint sampleIndex)
	{
		uint s0 = interleave_32bit(samplePixel);
		uint s1 = sampleIndex;
		setSeed(s0, s1);
	}

	// return u in [0, 1)
	ELAINA_CALLABLE float get1D() { return nextFloat(); }

	// return an independent 2D sampled vector in [0, 1)^2
	ELAINA_CALLABLE Vector2f get2D() { return {get1D(), get1D()}; }

	ELAINA_CALLABLE double get1D64() { return nextDouble(); }

	ELAINA_CALLABLE Vector2<double> get2D64() { return {get1D64(), get1D64()}; }

	ELAINA_CALLABLE void advance(int64_t delta = (1ll < 32))
	{
		uint64_t cur_mult = PCG32_MULT, cur_plus = inc, acc_mult = 1u, acc_plus = 0u;

		while (delta > 0)
		{
			if (delta & 1)
			{
				acc_mult *= cur_mult;
				acc_plus = acc_plus * cur_mult + cur_plus;
			}
			cur_plus = (cur_mult + 1) * cur_plus;
			cur_mult *= cur_mult;
			delta /= 2;
		}
		state = acc_mult * state + acc_plus;
	}

private:
	ELAINA_CALLABLE uint32_t nextUint()
	{
		uint64_t oldstate = state;
		state = oldstate * PCG32_MULT + inc;
		uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = (uint32_t)(oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
	}

	ELAINA_CALLABLE double nextDouble()
	{
		/* Trick from MTGP: generate an uniformly distributed
			double precision number in [1,2) and subtract 1. */
		union
		{
			uint64_t u;
			double d;
		} x;
		x.u = ((uint64_t)nextUint() << 20) | 0x3ff0000000000000ULL;
		return x.d - 1.0;
	}

	ELAINA_CALLABLE float nextFloat()
	{
		/* Trick from MTGP: generate an uniformly distributed
			single precision number in [1,2) and subtract 1. */
		union
		{
			uint32_t u;
			float f;
		} x;
		x.u = (nextUint() >> 9) | 0x3f800000u;
		return x.f - 1.0f;
	}

	uint64_t state; // RNG state.  All values are possible.
	uint64_t inc;	// Controls which RNG sequence (stream) is selected. Must
					// *always* be odd.
};

class Sampler : public TaggedPointer<PCGSampler>
{
public:
	using TaggedPointer::TaggedPointer;

	ELAINA_CALLABLE void setPixelSample(Vector2ui samplePixel, uint sampleIndex)
	{
		auto setPixelSample = [&](auto ptr) -> void
		{ return ptr->setPixelSample(samplePixel, sampleIndex); };
		return dispatch(setPixelSample);
	}

	// @returns uniform random variable in [0, 1)
	ELAINA_CALLABLE float get1D()
	{
		auto get1D = [&](auto ptr) -> float
		{ return ptr->get1D(); };
		return dispatch(get1D);
	};

	ELAINA_CALLABLE double get1D64()
	{
		auto get1D = [&](auto ptr) -> double
		{ return ptr->get1D64(); };
		return dispatch(get1D);
	}

	// @returns uniform independent 2D vector in [0, 1)^2
	ELAINA_CALLABLE Vector2f get2D()
	{
		auto get2D = [&](auto ptr) -> Vector2f
		{ return ptr->get2D(); };
		return dispatch(get2D);
	};

	ELAINA_CALLABLE Vector2<double> get2D64()
	{
		auto get2D = [&](auto ptr) -> Vector2<double>
		{ return ptr->get2D64(); };
		return dispatch(get2D);
	};
};

ELAINA_NAMESPACE_END