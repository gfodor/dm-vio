/**
* This file is part of DSO.
*
* [License and copyright information]
*/

#pragma once

#include "util/NumType.h"
#include "algorithm"
#include <cstring> // Include for memset and memcpy

namespace dso
{

template<typename T>
class MinimalImage
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int w;        // Width of the image
    int h;        // Height of the image
    int stride;   // Number of elements per row (including any padding)
    T* data;

    /*
     * Creates a minimal image with its own memory.
     * The stride defaults to the width if not specified.
     */
    inline MinimalImage(int w_, int h_, int stride_ = -1) : w(w_), h(h_)
    {
        stride = (stride_ > 0) ? stride_ : w;
        data = new T[stride * h];
        ownData = true;
    }

    /*
     * Creates a minimal image wrapping around existing memory.
     * The stride defaults to the width if not specified.
     */
    inline MinimalImage(int w_, int h_, T* data_, int stride_ = -1) : w(w_), h(h_)
    {
        stride = (stride_ > 0) ? stride_ : w;
        data = data_;
        ownData = false;
    }

    // Copy constructor
    inline MinimalImage(const MinimalImage& other) : w(other.w), h(other.h), stride(other.stride)
    {
        if(other.ownData)
        {
            data = new T[stride * h];
            std::memcpy(data, other.data, sizeof(T) * stride * h);
            ownData = true;
        }
        else
        {
            data = other.data;
            ownData = false;
        }
    }

    // Move constructor
    inline MinimalImage(MinimalImage&& other) noexcept
        : w(other.w), h(other.h), stride(other.stride), data(other.data), ownData(other.ownData)
    {
        other.data = nullptr;
        other.ownData = false;
    }

    inline ~MinimalImage()
    {
        if(ownData && data)
            delete[] data;
    }

    // Assignment operator
    inline MinimalImage& operator=(const MinimalImage& other)
    {
        if(this != &other)
        {
            if(ownData && data)
                delete[] data;

            w = other.w;
            h = other.h;
            stride = other.stride;

            if(other.ownData)
            {
                data = new T[stride * h];
                std::memcpy(data, other.data, sizeof(T) * stride * h);
                ownData = true;
            }
            else
            {
                data = other.data;
                ownData = false;
            }
        }
        return *this;
    }

    // Move assignment operator
    inline MinimalImage& operator=(MinimalImage&& other) noexcept
    {
        if(this != &other)
        {
            if(ownData && data)
                delete[] data;

            w = other.w;
            h = other.h;
            stride = other.stride;
            data = other.data;
            ownData = other.ownData;

            other.data = nullptr;
            other.ownData = false;
        }
        return *this;
    }

    /*
     * Creates a clone of the image.
     * The clone will have the same stride as the original.
     */
    inline MinimalImage* getClone() const
    {
        MinimalImage* clone = new MinimalImage(w, h, stride);
        std::memcpy(clone->data, data, sizeof(T) * stride * h);
        return clone;
    }

    /*
     * Access pixel at (x, y).
     * Ensures that x and y are within bounds.
     */
    inline T& at(int x, int y)
    {
        // Optionally, add boundary checks here
        return data[x + y * stride];
    }

    inline const T& at(int x, int y) const
    {
        // Optionally, add boundary checks here
        return data[x + y * stride];
    }

    /*
     * Access pixel at linear index i.
     * This ignores the stride and assumes contiguous storage.
     * Use with caution if stride != w.
     */
    inline T& at(int i)
    {
        return data[i];
    }

    inline const T& at(int i) const
    {
        return data[i];
    }

    /*
     * Set all pixels to black (zero).
     */
    inline void setBlack()
    {
        std::memset(data, 0, sizeof(T) * stride * h);
    }

    /*
     * Set all pixels to a constant value.
     */
    inline void setConst(T val)
    {
        for(int y = 0; y < h; ++y)
        {
            for(int x = 0; x < w; ++x)
            {
                at(x, y) = val;
            }
        }
    }

    /*
     * Set a single pixel with floating-point coordinates (u, v).
     * Rounds to the nearest integer pixel.
     */
    inline void setPixel1(const float &u, const float &v, T val)
    {
        at(static_cast<int>(u + 0.5f), static_cast<int>(v + 0.5f)) = val;
    }

    /*
     * Set a 2x2 block of pixels around floating-point coordinates (u, v).
     */
    inline void setPixel4(const float &u, const float &v, T val)
    {
        int iu = static_cast<int>(u);
        int iv = static_cast<int>(v);
        at(iu + 1, iv + 1) = val;
        at(iu + 1, iv)     = val;
        at(iu, iv + 1)     = val;
        at(iu, iv)         = val;
    }

    /*
     * Set a 3x3 block of pixels around integer coordinates (u, v).
     */
    inline void setPixel9(const int &u, const int &v, T val)
    {
        for(int dy = -1; dy <= 1; ++dy)
        {
            for(int dx = -1; dx <= 1; ++dx)
            {
                at(u + dx, v + dy) = val;
            }
        }
    }

    /*
     * Set a circular pattern of pixels around integer coordinates (u, v).
     * This example sets a diamond shape with radius 3.
     */
    inline void setPixelCirc(const int &u, const int &v, T val)
    {
        for(int i = -3; i <= 3; ++i)
        {
            at(u + 3, v + i) = val;
            at(u - 3, v + i) = val;
            at(u + 2, v + i) = val;
            at(u - 2, v + i) = val;

            at(u + i, v - 3) = val;
            at(u + i, v + 3) = val;
            at(u + i, v - 2) = val;
            at(u + i, v + 2) = val;
        }
    }

private:
    bool ownData = false;
};

typedef Eigen::Matrix<unsigned char,3,1> Vec3b;
typedef MinimalImage<float> MinimalImageF;
typedef MinimalImage<Vec3f> MinimalImageF3;
typedef MinimalImage<unsigned char> MinimalImageB;
typedef MinimalImage<Vec3b> MinimalImageB3;
typedef MinimalImage<unsigned short> MinimalImageB16;

} // namespace dso

