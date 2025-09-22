#pragma once
#include "core/common.h"
#include <memory>
// #include <Eigen/Dense>
#include "core/device/buffer.h"
#include "core/texture.h"

#if defined(__NVCC__)
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include "tonemapping.cuh"
#endif

ELAINA_NAMESPACE_BEGIN

class Film
{
public:
    using SharedPtr = std::shared_ptr<Film>;
    using Pixel = Color4f;
    using WeightedPixel = struct
    {
        Pixel pixel;
        float weight;
    };

    Film() = default;
    ~Film() = default;
    ELAINA_HOST Film(size_t res_x, size_t res_y)
    {
        m_size = {(int)res_x, (int)res_y};
        m_data.resize(res_x * res_y);
        reset();
    }

    ELAINA_HOST Film(const Vector2f size) : Film(size[0], size[1]) {}

    ELAINA_CALLABLE WeightedPixel *data() { return m_data.data(); }

    ELAINA_CALLABLE Vector2i size() { return m_size; }

    ELAINA_HOST void reset(const Pixel &value = {})
    {
        m_data.for_each([value] ELAINA_DEVICE(const WeightedPixel &c) -> WeightedPixel
                        { return {value, 0}; });
    }

    ELAINA_HOST void reset(const Pixel &value, const float weight)
    {
        m_data.for_each([value, weight] ELAINA_DEVICE(const WeightedPixel &c) -> WeightedPixel
                        { return {value, weight}; });
    }

    ELAINA_HOST TypedBuffer<WeightedPixel> &getInternalBuffer() { return m_data; }

    ELAINA_HOST void clear()
    {
        m_size = {};
        m_data.clear();
    }

    ELAINA_HOST void resize(const Vector2i &size)
    {
        m_size = size;
        m_data.resize(size[0] * size[1]);
        reset();
    }

    ELAINA_CALLABLE void put(const Pixel &pixel, const size_t offset)
    {
        m_data[offset].pixel += pixel;
        m_data[offset].weight += 1.f;
    };

    ELAINA_CALLABLE void put(const Pixel &pixel, const Vector2i &pos)
    {
        size_t idx = pos[0] + pos[1] * m_size[0];
        put(pixel, idx);
    }

    ELAINA_CALLABLE Pixel getPixel(const size_t offset)
    {
        const WeightedPixel &pixel = m_data[offset];
        return pixel.pixel / pixel.weight;
    }

    ELAINA_CALLABLE Pixel getPixel(const Vector2i &pos)
    {
        size_t idx = pos[0] + pos[1] * m_size[0];
        return getPixel(idx);
    }

#if defined(__NVCC__)
    ELAINA_HOST void save(const fs::path &filepath)
    {
        size_t n_pixels = m_size[0] * m_size[1];
        CUDABuffer tmp(n_pixels * sizeof(Pixel));
        Pixel *pixels_device = reinterpret_cast<Pixel *>(tmp.data());
        thrust::transform(thrust::device, m_data.data(), m_data.data() + n_pixels, pixels_device,
                          [] ELAINA_DEVICE(const WeightedPixel &d) -> Pixel
                          { return d.pixel / d.weight; });
        Image frame(m_size, Image::Format::RGBAfloat, false);
        tmp.copy_to_host(frame.data(), n_pixels * sizeof(Color4f));
        frame.saveImage(filepath);
    }

    ELAINA_HOST void saveEnergy(const fs::path &filepath, const ToneMapping tone)
    {
        size_t n_pixels = m_size[0] * m_size[1];
        thrust::device_vector<float> brightness(n_pixels);
        thrust::transform(thrust::device, m_data.data(), m_data.data() + n_pixels, brightness.begin(), 
                        [] ELAINA_DEVICE(const WeightedPixel &pixel) -> float 
                        { return pixel.pixel.matrix().dot(Vector4f(0.299f, 0.587f, 0.114f, 0.0f)); });
        auto minmax_iterators = thrust::minmax_element(thrust::device, brightness.begin(), brightness.end());
        float min_val = *(minmax_iterators.first);
        float max_val = *(minmax_iterators.second);
        float span = max_val - min_val;
        if (std::isnan(min_val) || std::isnan(max_val) || span == 0.0f)
        {
            ELAINA_LOG(Warning, "Invalid min/max values for tone mapping: min = %f, max = %f", min_val, max_val);
        }
        CUDABuffer tmp(n_pixels * sizeof(Pixel));
        Pixel *pixels_device = reinterpret_cast<Pixel *>(tmp.data());
        thrust::transform(thrust::device, brightness.begin(), brightness.end(), pixels_device, 
                        [min_val, span, tone] ELAINA_DEVICE(const float pixel) -> Pixel {
                            const float normalized_pixel = (pixel - min_val) / span;
                            switch(tone)
                            {
                                default:
                                case ToneMapping::NONE:
                                    return Color4f(Color3f(pixel), 1.0f);
                                case ToneMapping::NONE_NORMALIZED:
                                    return Color4f(Color3f(normalized_pixel), 1.0f);
                                case ToneMapping::MATLAB_JET:
                                    return Color4f(MatlabJet(normalized_pixel), 1.0f);
                                case ToneMapping::MATLAB_PARULA:
                                    return Color4f(MatlabParula(normalized_pixel), 1.0f);
                                case ToneMapping::IDL_RDBU:
                                    return Color4f(IDLRdBu(normalized_pixel), 1.0f);
                            } });
        Image frame(m_size, Image::Format::RGBAfloat, false);
        tmp.copy_to_host(frame.data(), n_pixels * sizeof(Color4f));
        frame.saveImage(filepath);
    }
#else
    ELAINA_HOST void save(const fs::path &filepath)
    {
        ELAINA_NOTIMPLEMENTED;
    }
#endif

private:
    TypedBuffer<WeightedPixel> m_data;
    Vector2i m_size{};
};

ELAINA_NAMESPACE_END
