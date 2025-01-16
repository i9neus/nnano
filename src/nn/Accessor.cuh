#pragma once

#include "../core/cuda/CudaUtils.cuh"
#include "../core/math/MathUtils.h"

namespace NNano
{    
    template<typename InputSample, typename OutputSample>
    class Accessor
    {
    public:
        __host__ Accessor() = default;

        __host__ virtual size_t Size() const = 0;
        __host__ virtual InputSample Load(const int idx) = 0;
        __host__ virtual void Store(const int idx, const OutputSample& sample) = 0;
    };
}