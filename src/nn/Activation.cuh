#pragma once

#include "NNUtils.cuh"
#include "../core/math/MathUtils.h"

namespace NNano
{
    namespace Activation
    {
        // Constant activation function
        struct Linear
        {
            static __forceinline__ __host__ __device__ float F(const float x) { return x; }
            static __forceinline__ __host__ __device__ float dF(const float x) { return 1; }
            static __forceinline__ __host__ __device__ float InvF(const float x) { return x; }
        };

        // Leaky ReLU activation function
        struct LeakyReLU
        {
            static __forceinline__ __host__ __device__ float F(const float x)
            {
                return (x < 0.) ? (x * 1e-2f) : x;
            }

            static __forceinline__ __host__ __device__ float dF(const float x)
            {
                return (x < 0.) ? 1e-2f : 1.f;
            }

            static __forceinline__ __host__ __device__ float InvF(const float x)
            {
                return (x < 0.) ? (x * 1e2) : x;
            }
        };

        // Hyperbolic tangent activation function
        struct TanH
        {
            static __forceinline__ __host__ __device__ float F(const float x)
            {
                return 2 / (1.f + expf(-2 * x)) - 1;
            }

            static __forceinline__ __host__ __device__ float dF(const float x)
            {
                const float ef = expf(-2 * x);
                return (4 * ef) / sqr(1 + ef);
            }

            static __forceinline__ __host__ __device__ float InvF(const float x)
            {
                return 0.5 * logf((1 + x) / (1 - x));
            }
        };

        // Sigmoid activation function
        struct Sigmoid
        {
            static __forceinline__ __host__ __device__ float F(const float x)
            {
                return 1 / (1 + expf(-x));
            }

            static __forceinline__ __host__ __device__ float dF(const float x)
            {
                const float ef = expf(-x);
                return ef / sqr(1. + ef);
            }

            static __forceinline__ __host__ __device__ float InvF(const float x)
            {
                return logf(x / (1 - x));
            }
        };

        // Sinusoidal activation function
        struct Sine
        {
            static __forceinline__ __host__ __device__ float F(const float x)
            {
                return sinf(30. * x);
            }

            static __forceinline__ __host__ __device__ float dF(const float x)
            {
                return 30. * cosf(30. * x);
            }

            static __forceinline__ __host__ __device__ float InvF(const float x)
            {
                return asinf(x);
            }
        };

        // Sinusoidal activation function
        struct SineReLU
        {
            static __forceinline__ __host__ __device__ float F(const float x)
            {
                return sinf(30. * x) * ((x < 0.) ? 1e-2f : 1.f);
            }

            static __forceinline__ __host__ __device__ float dF(const float x)
            {
                return 30. * cosf(30. * x) * ((x < 0.) ? 1e-2f : 1.f);
            }
        };
    }
}