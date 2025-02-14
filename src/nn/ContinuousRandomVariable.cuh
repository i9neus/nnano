#pragma once

#include "core/math/MathUtils.h"
#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include <random>

namespace NNano
{
    class ContinuousRandomVariable
    {
    protected:
        std::mt19937 m_mt;

        ContinuousRandomVariable() : m_mt(0) {}
        ContinuousRandomVariable(const uint32_t seed) : m_mt(seed) { }

    public:
        virtual float operator()() = 0;
    };
    
    // Continuous uniform distribution in the range [lower, upper)
    class UniformDistribution : public ContinuousRandomVariable
    {
    private:        
        std::uniform_real_distribution<float> m_rng;
        float m_lower, m_upper;

    public:
        UniformDistribution() : UniformDistribution(0, 1, 0) {}
        UniformDistribution(const float lower, const float upper, const uint32_t seed = 0) : ContinuousRandomVariable(seed), m_lower(lower), m_upper(upper) {}
        virtual float operator()() override final { return mix(m_lower, m_upper, m_rng(m_mt)); }
    };

    // Normal distribution with the option of mapping to log
    template<bool LogNormal>
    class NormalRandomDistributionImpl : public ContinuousRandomVariable
    {
    private:
        std::normal_distribution<float> m_rng;

    public:
        NormalRandomDistributionImpl() : NormalRandomDistributionImpl(0, 1, 0) {}
        NormalRandomDistributionImpl(const float mean, const float sigma, const uint32_t seed = 0) : ContinuousRandomVariable(seed), m_rng(mean, sigma) {}
        virtual float operator()() override final 
        { 
            return LogNormal ? std::exp(m_rng(m_mt)) : m_rng(m_mt);
        }
    };
    using NormalRandomDistribution = NormalRandomDistributionImpl<false>;
    using LogNormalRandomDistribution = NormalRandomDistributionImpl<true>;        

    // Always returns a constant Value
    template<int Value>
    class ConstantDistribution : public ContinuousRandomVariable
    {
    public:
        ConstantDistribution() : ContinuousRandomVariable(0) {}
        virtual float operator()() override final { return Value; }
    };

    using Ones = ConstantDistribution<1>;
    using Zeros = ConstantDistribution<0>;
}
