#pragma once
#include <map>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#endif

#include "common.h"
#include "util/check.h"
#include "file.h"
#include "logger.h"

ELAINA_NAMESPACE_BEGIN

class Image
{ // Not available on GPU
public:
	using SharedPtr = std::shared_ptr<Image>;

	enum class Format
	{
		NONE = 0,
		RGBAuchar,
		RGBAfloat,
	};

	Image() {};
	Image(Vector2i size, Format format = Format::RGBAuchar, bool srgb = false);
	~Image() {}

	bool loadImage(const fs::path &filepath, bool flip = false, bool srgb = false);
	bool saveImage(const fs::path &filepath);

	static bool isHdr(const string &filepath);
	static Image::SharedPtr createFromFile(const string &filepath, bool flip = false, bool srgb = false);
	bool isValid() const { return mFormat != Format::NONE && mSize[0] * mSize[1]; }
	bool isSrgb() const { return mSrgb; }
	Vector2i getSize() const { return mSize; }
	Format getFormat() const { return mFormat; }
	inline size_t getElementSize() const { return mFormat == Format::RGBAfloat ? sizeof(float) : sizeof(uchar); }
	int getChannels() const { return mChannels; }
	template <int DIM>
	inline void permuteChannels(const Vector<int, DIM> permutation);
	size_t getSizeInBytes() const { return mChannels * mSize[0] * mSize[1] * getElementSize(); }
	uchar *data() { return mData; }
	void reset(uchar *data) { mData = data; }

private:
	bool mSrgb{};
	Vector2i mSize = Vector2i::Zero();
	int mChannels{4};
	Format mFormat{};
	uchar *mData{};
};

#if defined(__CUDACC__)
template <int DIM>
void Image::permuteChannels(const Vector<int, DIM> permutation)
{
	if (!isValid())
		ELAINA_LOG(Error, "Load the image before do permutations");
	CHECK_LOG(4 == mChannels, "Only support channel == 4 currently!");
	CHECK_LOG(DIM <= mChannels, "Permutation do not match channel count!");
	size_t data_size = getElementSize();
	size_t n_pixels = mSize[0] * mSize[1];
	if (data_size == sizeof(float))
	{
		using PixelType = Array<float, 4>;
		auto *pixels = reinterpret_cast<PixelType *>(mData);
		thrust::transform(thrust::host, pixels, pixels + n_pixels, pixels, [=](PixelType pixel)
						  {
			PixelType res = pixel;
			for (int c = 0; c < DIM; c++) {
				res[c] = pixel[permutation[c]];
			}
			return res; });
	}
	else if (data_size == sizeof(char))
	{
		using PixelType = Array<char, 4>;
		auto *pixels = reinterpret_cast<PixelType *>(mData);
		thrust::transform(thrust::host, pixels, pixels + n_pixels, pixels, [=](PixelType pixel)
						  {
			PixelType res = pixel;
			for (int c = 0; c < DIM; c++) {
				res[c] = pixel[permutation[c]];
			}
			return res; });
	}
	else
	{
		ELAINA_LOG(Error, "Permute channels not implemented yet :-(");
	}
}
#else
template <int DIM>
void Image::permuteChannels(const Vector<int, DIM> permutation)
{
	ELAINA_NOTIMPLEMENTED;
}
#endif

ELAINA_NAMESPACE_END
