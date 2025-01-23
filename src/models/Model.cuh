#pragma once

#include "../core/cuda/CudaUtils.cuh"
#include "../core/cuda/CudaVector.cuh"

namespace NNano
{    
    template<typename InputSample, typename OutputSample>
    class DataAccessor
    {
    public:
        __host__ DataAccessor() = default;

        __host__ virtual int LoadTrainingSet(Cuda::Vector<InputSample>&, Cuda::Vector<OutputSample>&) = 0;
        __host__ virtual int LoadInferenceBatch(Cuda::Vector<InputSample>&, const int startIdx) = 0;
        __host__ virtual void StoreInferenceBatch(const Cuda::Vector<OutputSample>&, const int startIdx) = 0;
    };
    
    template<typename InputSample, typename OutputSample>
    class NNModel
    {
    protected:
        cudaStream_t                                    m_cudaStream;

    protected:
        __host__ NNModel(cudaStream_t stream) :
            m_cudaStream(stream) {}

    public:
        __host__ cudaStream_t GetCudaStream() const { return m_cudaStream; }

        __host__ virtual void PrepareTraining() = 0;
        __host__ virtual void TrainEpoch() = 0;

        __host__ virtual void Infer() = 0;
    };
}