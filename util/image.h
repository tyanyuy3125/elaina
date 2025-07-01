#pragma once

#include "core/common.h"

#include <filesystem>

ELAINA_NAMESPACE_BEGIN

namespace tinyexr {
void save_exr(const float *data, int width, int height, int nChannels, int channelStride,
			  const char *outfilename, bool flip = true);
int load_exr(float **data, int *width, int *height, const char *filename, bool filp = true);
} // namespace tinyexr

namespace pfm {
/* out: data[rgb], res_x, res_y */
Color4f *ReadImagePFM(const std::string &filename, int *xres, int *yres);
} // namespace pfm


ELAINA_NAMESPACE_END