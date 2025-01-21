#pragma once

#include "../core/cuda/CudaUtils.cuh"

namespace NNano
{    
    template<typename InputSample, typename OutputSample>
    class DataAccessor
    {
    public:
        __host__ DataAccessor() = default;

        __host__ virtual size_t Size() const = 0;
        __host__ virtual InputSample Load(const int idx) = 0;
        __host__ virtual void Store(const int idx, const OutputSample& sample) = 0;
    };
    
    template<typename InputSample, typename OutputSample>
    class ModelInterface
    {
    protected:
        __host__ ModelInterface() = default;

    public:
        __host__ virtual void ResetTraining() = 0;
        __host__ virtual void PrepareTraining(const std::vector<InputSample>& hostInputSamples, const std::vector<OutputSample>& hostTargetSamples) = 0;
        __host__ virtual void TrainEpoch() = 0;

        __host__ virtual void ResetInference() = 0;
        __host__ virtual void PrepareInference(const int inferBatchSize) = 0;
        __host__ virtual void Infer(DataAccessor<InputSample, OutputSample>& accessor) = 0;
    };
}