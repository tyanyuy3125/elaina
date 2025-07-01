#pragma once
#include <cuda.h>

#if defined(__CUDACC__) // Thrust can not get compiled on other compilers. Make them happy.
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#endif

#include "core/common.h"
#include "util/check.h"

ELAINA_NAMESPACE_BEGIN

class CUDABuffer
{
public:
	CUDABuffer() = default;
	CUDABuffer(std::size_t size) { resize(size); }
	~CUDABuffer() { free(); }

	ELAINA_CALLABLE void setPtr(void *ptr) { d_ptr = ptr; }
	ELAINA_CALLABLE CUdeviceptr data() const { return (CUdeviceptr)d_ptr; }
	ELAINA_CALLABLE std::size_t size() const { return sizeInBytes; }

	//! re-size buffer to given number of bytes
	void resize(std::size_t size)
	{
		if (sizeInBytes == size)
			return;
		if (d_ptr)
			free();
		this->sizeInBytes = size;
		CUDA_CHECK(cudaMalloc((void **)&d_ptr, sizeInBytes));
	}

	//! free allocated memory
	void free()
	{
		CUDA_CHECK(cudaFree(d_ptr));
		d_ptr = nullptr;
		sizeInBytes = 0;
	}

	template <typename T>
	void alloc_and_copy_from_host(const std::vector<T> &vt)
	{
		if (vt.size() == 0)
			return;
		resize(vt.size() * sizeof(T));
		copy_from_host((const T *)vt.data(), vt.size());
	}

	template <typename T>
	void alloc_and_copy_from_device(const std::vector<T> &vt)
	{
		if (vt.size() == 0)
			return;
		resize(vt.size() * sizeof(T));
		copy_from_device((const T *)vt.data(), vt.size());
	}

	template <typename T>
	void alloc_and_copy_from_host(const T *t, std::size_t count)
	{
		if (count == 0)
			return;
		resize(count * sizeof(T));
		copy_from_host((const T *)t, count);
	}

	template <typename T>
	void alloc_and_copy_from_device(const T *t, std::size_t count)
	{
		if (count == 0)
			return;
		resize(count * sizeof(T));
		copy_from_device((const T *)t, count);
	}

	template <typename T>
	void copy_from_host(const T *t, std::size_t count)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes >= count * sizeof(T));
		CUDA_CHECK(cudaMemcpy(d_ptr, (void *)t,
							  count * sizeof(T), cudaMemcpyHostToDevice));
	}

	template <typename T>
	void copy_to_host(T *t, std::size_t count)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes >= count * sizeof(T));
		CUDA_CHECK(cudaMemcpy((void *)t, d_ptr,
							  count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template <typename T>
	void copy_from_device(const T *t, std::size_t count)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes >= count * sizeof(T));
		CUDA_CHECK(cudaMemcpy(d_ptr, (void *)t,
							  count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	template <typename T>
	void copy_to_device(T *t, std::size_t count)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes >= count * sizeof(T));
		CUDA_CHECK(cudaMemcpy((void *)t, d_ptr,
							  count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

private:
	std::size_t sizeInBytes{0};
	void *d_ptr{nullptr};
};

template <typename T>
class TypedBuffer
{
public:
	ELAINA_CALLABLE TypedBuffer() = default;
	ELAINA_HOST TypedBuffer(size_t size) { resize(size); }
	ELAINA_CALLABLE ~TypedBuffer(){};
	/* copy constructor */
	TypedBuffer(const TypedBuffer &other)
	{
		alloc_and_copy_from_device(other.d_ptr, other.m_size);
	}
	/* move constructor */
	TypedBuffer(TypedBuffer &&other)
	{
		m_size = other.m_size;
		d_ptr = other.d_ptr;

		other.m_size = 0;
		other.d_ptr = nullptr;
	}
	/* copy assignment */
	TypedBuffer &operator=(const TypedBuffer &other)
	{
		clear();
		alloc_and_copy_from_device(other.d_ptr, other.m_size);
		return *this;
	}
	/* move assignment */
	TypedBuffer &operator=(TypedBuffer &&other)
	{
		clear();
		m_size = other.m_size;
		d_ptr = other.d_ptr;

		other.m_size = 0;
		other.d_ptr = nullptr;
		return *this;
	}

	ELAINA_CALLABLE T *data() const { return d_ptr; }

	ELAINA_CALLABLE const T &operator[](size_t index) const
	{
		DCHECK_LT(index, m_size);
		return d_ptr[index];
	}

	ELAINA_CALLABLE T &operator[](size_t index)
	{
		DCHECK_LT(index, m_size);
		return d_ptr[index];
	}

#if defined(__CUDACC__)
	template <typename F>
	ELAINA_HOST void for_each(F &&func)
	{
		thrust::transform(thrust::device, d_ptr, d_ptr + m_size, d_ptr,
						  [func] ELAINA_DEVICE(const T &val) mutable
						  { return func(val); });
	}
#else
	template <typename F>
	ELAINA_HOST void for_each(F &&func)
	{
		ELAINA_NOTIMPLEMENTED;
	}
#endif

	ELAINA_CALLABLE size_t size() const { return m_size; }

	ELAINA_CALLABLE size_t sizeInBytes() const { return m_size * sizeof(T); }

	inline void resize(size_t new_size)
	{
		if (m_size == new_size)
			return;
		clear();
		m_size = new_size;
		CUDA_CHECK(cudaMalloc((void **)&d_ptr, new_size * sizeof(T)));
	}

	inline void clear()
	{
		if (d_ptr)
			CUDA_CHECK(cudaFree(d_ptr));
		d_ptr = nullptr;
		m_size = 0;
	}

	void alloc_and_copy_from_host(const std::vector<T> &vt)
	{
		if (vt.size() == 0)
			return;
		resize(vt.size());
		copy_from_host((const T *)vt.data(), vt.size());
	}

	void alloc_and_copy_from_device(const std::vector<T> &vt)
	{
		if (vt.size() == 0)
			return;
		resize(vt.size());
		copy_from_device((const T *)vt.data(), vt.size());
	}

	void alloc_and_copy_from_host(const T *t, size_t count)
	{
		if (count == 0)
			return;
		resize(count);
		copy_from_host((const T *)t, count);
	}

	void alloc_and_copy_from_device(const T *t, size_t count)
	{
		if (count == 0)
			return;
		resize(count);
		copy_from_device((const T *)t, count);
	}

	void copy_from_host(const T *t, size_t count)
	{
		ELAINA_CHECK(d_ptr);
		ELAINA_CHECK(m_size >= count);
		CUDA_CHECK(cudaMemcpy(d_ptr, (void *)t,
							  count * sizeof(T), cudaMemcpyHostToDevice));
	}

	void copy_to_host(T *t, size_t count) const
	{
		ELAINA_CHECK(d_ptr);
		ELAINA_CHECK(m_size >= count);
		CUDA_CHECK(cudaMemcpy((void *)t, d_ptr,
							  count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	void copy_from_device(const T *t, size_t count)
	{
		ELAINA_CHECK(d_ptr);
		ELAINA_CHECK(m_size >= count);
		CUDA_CHECK(cudaMemcpy(d_ptr, (void *)t,
							  count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	void copy_to_device(T *t, size_t count) const
	{
		ELAINA_CHECK(d_ptr);
		ELAINA_CHECK(m_size >= count);
		CUDA_CHECK(cudaMemcpy((void *)t, d_ptr,
							  count * sizeof(T), cudaMemcpyDeviceToDevice));
	}

private:
	size_t m_size{0};
	T *d_ptr{nullptr};
};

ELAINA_NAMESPACE_END
