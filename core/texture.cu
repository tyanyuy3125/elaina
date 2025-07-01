#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT

#pragma nv_diag_suppress 550
#include "zlib.h" // needed by tinyexr
#include "stb_image.h"
#include "stb_image_write.h"
// marcos required by tinyexr.h is defined in util/image.cpp
#include "tinyexr.h"

#include <filesystem>
#include <cstdio>


#include "texture.h"
#include "logger.h"
#include "util/image.h"

ELAINA_NAMESPACE_BEGIN

Image::Image(Vector2i size, Format format, bool srgb) : mSrgb(srgb), mFormat(format), mSize(size) {
	mData = new uchar[size[0] * size[1] * 4 * getElementSize()];
}

bool Image::loadImage(const fs::path &filepath, bool flip, bool srgb) {
	Vector2i size;
	int channels;
	string filename = filepath.string();
	string format	= filepath.extension().string();
	uchar *data		= nullptr;
	stbi_set_flip_vertically_on_load(flip);
	if (IsEXR(filename.c_str()) == TINYEXR_SUCCESS) {
		char *errMsg = nullptr;
		// to do: if loadEXR always return RGBA data?
		// int res = LoadEXR((float**)&data, &size[0], &size[1], filename.c_str(), (const
		// char**)&errMsg);
		int res = tinyexr::load_exr((float **) &data, &size[0], &size[1], filename.c_str(), flip);

		if (res != TINYEXR_SUCCESS) {
			logError("Failed to load EXR image at " + filename);
			if (errMsg)
				logError(errMsg);
			return false;
		}
		mFormat = Format::RGBAfloat;
	} else if (stbi_is_hdr(filename.c_str())) {
		stbi_set_flip_vertically_on_load(flip);
		data =
			(uchar *) stbi_loadf(filename.c_str(), &size[0], &size[1], &channels, STBI_rgb_alpha);
		if (data == nullptr) {
			logError("Failed to load float hdr image at " + filename);
			return false;
		}
		mFormat = Format::RGBAfloat;
	} else if (format == ".pfm") {
		data = (uchar *) pfm::ReadImagePFM(filename, &size[0], &size[1]);
		if (data == nullptr) {
			logError("Failed to load PFM image at " + filename);
			return false;
		}
		mFormat = Format::RGBAfloat;
	} else { // formats other than exr...
		data = stbi_load(filename.c_str(), &size[0], &size[1], &channels, STBI_rgb_alpha);
		if (data == nullptr) {
			logError("Failed to load image at " + filename);
			return false;
		}
		mFormat = Format::RGBAuchar;
	}
	stbi_set_flip_vertically_on_load(false);
	int elementSize = getElementSize();
	mSrgb			= srgb;
	if (mData)
		delete[] mData;
	mData = data;
	mSize = size;
	logDebug("Loaded image " + to_string(size[0]) + "*" + to_string(size[1]));
	return true;
}

bool Image::saveImage(const fs::path &filepath) {
    fs::path absolutePath = fs::absolute(filepath);

    fs::path directory = absolutePath.parent_path();
    if (!fs::exists(directory)) {
        fs::create_directories(directory);
    }

	string extension = filepath.extension().string();
	uint nElements	 = mSize[0] * mSize[1] * 4;
	if (extension == ".png") {
		stbi_flip_vertically_on_write(true);
		if (mFormat == Format::RGBAuchar) {
			stbi_write_png(filepath.string().c_str(), mSize[0], mSize[1], 4, mData, 0);
		} else if (mFormat == Format::RGBAfloat) {
			uchar *data			= new uchar[nElements];
			float *internalData = reinterpret_cast<float *>(mData);
			std::transform(internalData, internalData + nElements, data,
						   [](float v) -> uchar { return clamp((int) (v * 255), 0, 255); });
			stbi_write_png(filepath.string().c_str(), mSize[0], mSize[1], 4, data, 0);
			delete[] data;
		}
		return true;
	} else if (extension == ".exr") {
		if (mFormat != Format::RGBAfloat) {
			logError("Image::saveImage Saving non-hdr image as hdr file...");
			return false;
		}
		tinyexr::save_exr(reinterpret_cast<float *>(mData), mSize[0], mSize[1], 4, 4,
						  filepath.string().c_str(), true);
	} else {
		logError("Image::saveImage Unknown image extension: " + extension);
		return false;
	}
	return false;
}

bool Image::isHdr(const string &filepath) {
	return (IsEXR(filepath.c_str()) == TINYEXR_SUCCESS) || stbi_is_hdr(filepath.c_str());
}

Image::SharedPtr Image::createFromFile(const string &filepath, bool flip, bool srgb) {
	Image::SharedPtr pImage = Image::SharedPtr(new Image());
	pImage->loadImage(filepath, flip, srgb);
	return pImage;
}

ELAINA_NAMESPACE_END