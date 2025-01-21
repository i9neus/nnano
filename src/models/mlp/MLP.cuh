#pragma once

#include "../../core/cuda/CudaObject.cuh"
#include "../../core/cuda/CudaVector.cuh"
#include "../../core/math/MathUtils.h"
#include "../../core/math/TensorOps.cuh"
#include "../../core/utils/HighResTimer.h"
#include "../../core/utils/IOUtils.h"
#include "../../nn/Permute.cuh"
#include "../../nn/Activation.cuh"
#include "../Model.cuh"

#include "MLP.cuh"
#include "Training.cuh"
#include "TrainingCPU.cuh"
#include "Inference.cuh"
#include "Optimiser.cuh"

#include <fstream>
#include <thread>
#include <chrono>
#include <memory>
#include <functional>
#include <vector>

namespace NNano
{
    template<typename Model, 
                typename ModelInitialiser, 
                typename ActivationFunction, 
                typename LossFunction, 
                typename OptimiserFunction, 
                int MiniBatchSize = 128, 
                ComputeDevice TargetDevice = ComputeDevice::kCUDA>
    class MLP : ModelInterface<Tensor1D<Model::kInputWidth>, Tensor1D<Model::kOutputWidth>>
    {
    public:
        using InputSample = Tensor1D<Model::kInputWidth>;
        using OutputSample = Tensor1D<Model::kOutputWidth>;
        using Evaluator = LinearSequentialEvaluator<TargetDevice, Model>;
        using Policy = MLPPolicy<TargetDevice, Model, Evaluator, HyperParameters<MiniBatchSize, ActivationFunction, LossFunction, OptimiserFunction>>;
        using Trainer = MLPTrainer<TargetDevice, Policy>;

        using ReadBatchFunctor = std::function<bool(std::vector<InputSample>&, const int)>;
        using WriteBatchFunctor = std::function<void(const std::vector<OutputSample>&, const int)>;

        enum : size_t
        {
            kMiniBatchSize = MiniBatchSize,
            kSharedMemorySafeMargin = 1024,
            kMaxMiniBatches = std::numeric_limits<int>::max()
        };

    private:            
        std::unique_ptr<Cuda::Vector<float>>            m_computeModelData;
        std::unique_ptr<Cuda::Vector<float>>            m_computeGradData;
        std::unique_ptr<Cuda::Vector<float>>            m_computeSampleLosses;
        std::unique_ptr<Cuda::Object<float>>            m_computeMiniBatchLoss;
        std::unique_ptr<Cuda::Vector<InputSample>>      m_computeTrainInputSamples;
        std::unique_ptr<Cuda::Vector<OutputSample>>     m_computeTrainOutputSamples;
        std::unique_ptr<Cuda::Vector<OutputSample>>     m_computeTrainTargetSamples;
        std::unique_ptr<Permutation>                    m_sampleIdxs;
         
        std::unique_ptr<Cuda::Vector<InputSample>>      m_computeInferInputSamples;
        std::unique_ptr<Cuda::Vector<OutputSample>>     m_computeInferOutputSamples;

        std::vector<InputSample>                        m_hostInputSamples;
        std::vector<OutputSample>                       m_hostOutputSamples; 

        TrainingKernelData<Policy>                      m_kernelData;
        std::vector<std::pair<int, float>>              m_epochLoss;
        std::vector<float>                              m_miniBatchLoss;
        double                                          m_totalTrainingTime;
        int                                             m_epochIdx;

        cudaStream_t                                    m_cudaStream;
        HighResTimer                                    m_kernelTimer;

    public:
        MLP() : 
            m_computeModelData(new Cuda::Vector<float>(TargetDevice))
        {
            IsOk(cudaStreamCreate(&m_cudaStream));
        }

        ~MLP()
        {
            cudaStreamDestroy(m_cudaStream);
        }

        virtual void ResetTraining() override final
        {
            m_computeGradData.reset();
            m_computeTrainInputSamples.reset();
            m_computeTrainTargetSamples.reset();
            m_computeSampleLosses.reset();
            m_computeMiniBatchLoss.reset();           

            m_epochLoss.clear();
            m_miniBatchLoss.clear();
            m_epochIdx = 0;
            m_totalTrainingTime = 0;
        }

        virtual void PrepareTraining(const std::vector<InputSample>& hostInputSamples, const std::vector<OutputSample>& hostTargetSamples) override final
        {
            AssertFmt(hostInputSamples.size() == hostTargetSamples.size(), "Size of input and target sample arrays does not match (%i != %i)", hostInputSamples.size(), hostTargetSamples.size());

            const size_t ctxSize = sizeof(TrainingCtx<Policy>);
            cudaDeviceProp prop;
            IsOk(cudaGetDeviceProperties(&prop, 0));
            AssertFmt(ctxSize < prop.sharedMemPerBlock - kSharedMemorySafeMargin, "Model context exceeds capacity of shared memory.");

            printf_red("TrainingCtx: %i bytes\n", ctxSize);
            printf_red("Model size: %i parameters\n", Model::kNumParams);

            m_computeGradData.reset(new Cuda::Vector<float>(TargetDevice, MiniBatchSize * Policy::Model::kNumParams, 0.f));
            m_computeTrainInputSamples.reset(new Cuda::Vector<InputSample>(TargetDevice, hostInputSamples.size()));
            m_computeTrainOutputSamples.reset(new Cuda::Vector<OutputSample>(TargetDevice, hostTargetSamples.size()));
            m_computeTrainTargetSamples.reset(new Cuda::Vector<OutputSample>(TargetDevice, hostTargetSamples.size()));
            m_computeSampleLosses.reset(new Cuda::Vector<float>(TargetDevice, MiniBatchSize));
            m_computeMiniBatchLoss.reset(new Cuda::Object<float>(TargetDevice));

            // Determininstically initialise the mini-batch weights and the optimiser 
            std::vector<float> hostModelData(Model::kNumParams);
            auto rng = ModelInitialiser();
            Model::Initialise(hostModelData, rng);

            *m_computeModelData <<= hostModelData;

            // Create and initialise the optimiser
            Cuda::Vector<float> computeOptimiserData(TargetDevice, Policy::Model::kNumParams * 2, 0.f);

            // Upload the samples
            *m_computeTrainInputSamples <<= hostInputSamples;
            *m_computeTrainTargetSamples <<= hostTargetSamples;

            // Create random indirection buffer
            m_sampleIdxs.reset(new Permutation(TargetDevice, m_cudaStream, hostInputSamples.size()));
            m_sampleIdxs->Randomise();
            //m_sampleIdxs->Sequential();

            // Initialise the kernel data structure
            m_kernelData.mlpModelData = m_computeModelData->GetComputeData();
            m_kernelData.mlpGradData = m_computeGradData->GetComputeData();
            m_kernelData.inputSamples = m_computeTrainInputSamples->GetComputeData();
            m_kernelData.outputSamples = m_computeTrainOutputSamples->GetComputeData();
            m_kernelData.targetSamples = m_computeTrainTargetSamples->GetComputeData();
            m_kernelData.optimiserData = computeOptimiserData.GetComputeData();
            m_kernelData.sampleIdxs = m_sampleIdxs->GetComputeData();
            m_kernelData.sampleLosses = m_computeSampleLosses->GetComputeData();
            m_kernelData.miniBatchLoss = m_computeMiniBatchLoss->GetComputeData();
            m_kernelData.batchSize = hostInputSamples.size();
        }

        virtual void TrainEpoch() override final
        {
            AssertFmt(m_computeGradData, "Training has not been initialised. Called PrepareTraining() first.");
                
            // Shuffle the indirection indices
            m_sampleIdxs->Shuffle();
            m_kernelTimer.Reset();
                
            float meanLoss = 0;
            int miniBatchIdx = 0;
            for (int sampleIdx = 0; sampleIdx < m_kernelData.batchSize && miniBatchIdx < kMaxMiniBatches; sampleIdx += MiniBatchSize, ++miniBatchIdx)
            {
                m_computeGradData->Fill(0.f);

                // Reset the kernel data (loss values, etc.) for the new epoch
                Trainer::PrepareNewEpoch(m_kernelData, m_cudaStream);

                // Estimate the gradients
                Trainer::EstimateGradients(m_kernelData, sampleIdx, m_cudaStream);

                // Optimiser step
                Optimiser<TargetDevice, Policy>::Descend(m_kernelData, m_epochIdx, sampleIdx, m_cudaStream);

                IsOk(cudaStreamSynchronize(m_cudaStream));
                const float loss = m_computeMiniBatchLoss->Download();
                m_miniBatchLoss.emplace_back(loss);
                meanLoss += loss;
            }
                
            // Record the epoch loss
            meanLoss /= std::ceil(m_kernelData.batchSize / float(MiniBatchSize));
            m_epochLoss.emplace_back(miniBatchIdx, meanLoss);
            m_totalTrainingTime += m_kernelTimer.Get();
            ++m_epochIdx;
        }

        virtual void PrepareInference(const int inferBatchSize) override final
        {
            m_computeInferInputSamples.reset(new Cuda::Vector<InputSample>(TargetDevice, inferBatchSize));
            m_computeInferOutputSamples.reset(new Cuda::Vector<OutputSample>(TargetDevice, inferBatchSize));
        }

        virtual void ResetInference() override final
        {
            m_computeInferInputSamples.reset();
            m_computeInferOutputSamples.reset();
        }

        virtual void Infer(DataAccessor<InputSample, OutputSample>& accessor) override final
        {
            AssertFmt(!m_computeModelData->IsEmpty(), "Model has not been initialised. Run a training cycle or load pre-trained weights first.");
            AssertFmt(m_computeInferInputSamples, "Inference has not been initialised. Call PrepareInference() first. ");

            std::vector<InputSample> hostInputSamples;
            std::vector<OutputSample> hostOutputSamples;
            hostInputSamples.reserve(MiniBatchSize);
            hostOutputSamples.reserve(MiniBatchSize);
                
            // Initialise the kernel data structure
            InferenceKernelData<Policy> kernelData;
            kernelData.mlpModelData = m_computeModelData->GetComputeData();

            // Process the samples in batches
            for (int sampleIdx = 0; sampleIdx < accessor.Size(); sampleIdx += m_computeInferInputSamples->Size())
            {          
                // Load a batch of samples from the accessor
                hostInputSamples.clear();
                for (int batchIdx = 0; batchIdx < MiniBatchSize && sampleIdx + batchIdx < accessor.Size(); ++batchIdx)
                {
                    hostInputSamples.push_back(accessor.Load(sampleIdx + batchIdx));
                }
                hostOutputSamples.resize(hostInputSamples.size());

                // Upload to the device
                *m_computeInferInputSamples <<= hostInputSamples;
                kernelData.batchSize = hostInputSamples.size();
                kernelData.inputSamples = m_computeInferInputSamples->GetComputeData();
                kernelData.outputSamples = m_computeInferOutputSamples->GetComputeData();

                // Run the inference pass
                MLPInferer<TargetDevice, Policy>::InferBatch(kernelData, m_cudaStream);

                // Store the output samples 
                hostOutputSamples <<= *m_computeInferOutputSamples;
                for (int batchIdx = 0; batchIdx < accessor.Size(); ++batchIdx)
                {
                    accessor.Store(sampleIdx + batchIdx, hostOutputSamples[batchIdx]);
                }
            }
        }
    }; 
}