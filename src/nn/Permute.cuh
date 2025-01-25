#pragma once

//#include "NNUtils.cuh"
#include "../core/cuda/CudaVector.cuh"
#include "../core/math/MathUtils.h"
#include <random>
#include <set>

namespace NNano
{
    __global__ static void FillSequentialKernel(int* data, const int dataSize)
    {
        if (kKernelIdx < dataSize)
        {
            data[kKernelIdx] = kKernelIdx;
        }
    }

    __global__ static void ShuffleKernel(int* dest, const int* src, const int dataSize, uint32_t offset)
    {
        if (kKernelIdx < dataSize)
        {
            dest[kKernelIdx] = (src[src[kKernelIdx]] + offset) % dataSize;
        }
    }

    __global__ static void BijectiveShuffleKernel(int* data, const int dataSize, const uint32_t W, const uint32_t Q, uint32_t offset)
    {
        const uint32_t k = kKernelIdx / Q;
        const uint32_t i = k & ((1 << W) - 1);
        const uint32_t j = RadicalInverse(i) >> (31 - W);
        const uint32_t p = (k & ~((1 << W) - 1)) << 1;

        const uint32_t r = kKernelIdx % Q;
        const int i0 = (p + i) * Q + r;
        const int i1 = (p + j) * Q + r;

        if (i0 < dataSize && i1 < dataSize)
        {
            Swap(data[(i0 + offset) % dataSize], data[(i1 + offset) % dataSize]);
        }
    }

    class Permutation
    {
    private:
        Cuda::Vector<int>                           m_computeIndices;
        std::vector<int>                            m_hostIndices;
        Cuda::Vector<int>                           m_swap;
        std::mt19937                                m_mt;
        std::uniform_int_distribution<int>          m_rng;
        cudaStream_t                                m_cudaStream;

    private:
        __host__ void Resize(const int newSize)
        {
            m_hostIndices.resize(newSize);
            m_computeIndices.Resize(newSize);
        }

    public:
        Permutation(const ComputeDevice targetDevice, cudaStream_t stream, const int size = 0, const uint32_t seed = 0) :
            m_mt(seed),
            m_hostIndices(size),
            m_computeIndices(targetDevice, size),
            m_swap(targetDevice, size),
            m_cudaStream(stream)
        {

        }

        __host__ inline size_t Size() const { return m_hostIndices.size(); }

        __host__ std::vector<int>& GetHostData()
        {
            m_hostIndices <<= m_computeIndices;
            return m_hostIndices;
        }

        __host__ void To(const ComputeDevice computeDevice) { m_computeIndices.To(computeDevice); }

        __host__ int* GetComputeData() { return m_computeIndices.GetComputeData(); }
        __host__ const int* GetComputeData() const { return m_computeIndices.GetComputeData(); }

        // Check that the each index in the vector maps to one and only one other element 
        __host__ void CheckBijective()
        {
            m_hostIndices <<= m_computeIndices;
            std::set<int> numbers;
            for (auto& i : m_hostIndices)
            {
                AssertFmt(numbers.find(i) == numbers.end(), "Map is not bijective!");
                numbers.emplace(i);
            }
            printf_green("Map is bijective!\n");
        }

        __host__ void Randomise(const int size)
        {
            Resize(size);
            
            // Fill array with sequential integers
            for (int i = 0; i < m_hostIndices.size(); ++i) { m_hostIndices[i] = i; }

            // Step through the array and place an element from the sorted part into a random position in the unsorted part
            for (int i = 0; i < m_hostIndices.size(); ++i)
            {
                const int j = (i < m_hostIndices.size() / 2) ? (i + m_rng(m_mt) % (m_hostIndices.size() - i)) : (m_rng(m_mt) % (1 + i));
                Swap(m_hostIndices[i], m_hostIndices[j]);
            }

            //CheckBijective(m_computeIndices);
            m_computeIndices <<= m_hostIndices;
        }

        __host__ void Sequential(const int size)
        {
            Resize(size);
            
            if (m_computeIndices.IsCUDA())
            {
                const int kNumBlocks = (m_computeIndices.Size() + 255) / 256;
                FillSequentialKernel << <kNumBlocks, 256, 0, m_cudaStream >> > (m_computeIndices.GetComputeData(), m_computeIndices.Size());
                IsOk(cudaGetLastError());
                IsOk(cudaStreamSynchronize(m_cudaStream));
            }
            else
            {
                for (int i = 0; i < m_computeIndices.Size(); ++i) { m_computeIndices[i] = i; }
            }
        }

        __host__ void Shuffle()
        {
            m_swap.Resize(m_computeIndices.Size());
            const int offset = m_rng(m_mt) % m_computeIndices.Size();

            if (m_computeIndices.IsCUDA())
            {
                const int kNumBlocks = (m_computeIndices.Size() + 255) / 256;
                ShuffleKernel << < kNumBlocks, 256, 0, m_cudaStream >> > (m_swap.GetComputeData(), m_computeIndices.GetComputeData(), m_computeIndices.Size(), offset);
                IsOk(cudaGetLastError());
                IsOk(cudaStreamSynchronize(m_cudaStream));
            }
            else
            {
                for (int i = 0; i < m_computeIndices.Size(); ++i)
                {
                    m_swap[i] = (m_computeIndices[m_computeIndices[i]] + offset) % m_computeIndices.Size();
                }
            }

            Swap(m_swap, m_computeIndices);

            //CheckBijective();
        }
    };
}