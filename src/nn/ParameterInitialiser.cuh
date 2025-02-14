#pragma once

#include "../core/cuda/CudaUtils.cuh"
#include "../core/utils/ConsoleUtils.h"
#include "NNUtils.cuh"
#include "../core/math/MathUtils.h"
#include <random>

namespace NNano
{
    namespace NN
    {
        class ParameterInitialiser
        {
        public:
            virtual float operator()(const int layerIdx, const int N, const int M) = 0;
        };

        template<typename TDistribution>
        class ParameterInitialiserBase : public ParameterInitialiser
        {
        protected:
            ParameterInitialiserBase(const uint32_t seed) : m_mt(seed) {}
            
            TDistribution m_rng;
            std::mt19937 m_mt;
        };


        class UniformXavierInitialiser : public ParameterInitialiserBase<std::uniform_real_distribution<float>>
        {
        public:
            UniformXavierInitialiser(const uint32_t seed = 0) : ParameterInitialiserBase(seed) {}

            virtual float operator()(const int layerIdx, const int N, const int M) override final
            {
                return mix(-1.f, 1.f, m_rng(m_mt)) * kRoot2 * std::sqrt(2.0f / (N + M));
            }
        };

        class SirenInitialiser : public ParameterInitialiserBase<std::uniform_real_distribution<float>>
        {
        public:
            SirenInitialiser(const uint32_t seed = 0) : ParameterInitialiserBase(seed) {}

            virtual float operator()(const int layerIdx, const int N, const int M) override final
            {
                constexpr float c = 6;
                constexpr float omega0 = 30.f;

                return mix(-1.f, 1.f, m_rng(m_mt)) * 
                    ((layerIdx == 0) ? 
                        (1. / N) : 
                        (std::sqrt(c / N) / omega0));
            }
        };
    }
}