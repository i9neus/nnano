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
    namespace NN
    {
        template<typename Model, typename ModelInitialiser, typename ActivationFunction, typename LossFunction, typename OptimiserFunction, int MiniBatchSize = 128, ComputeDevice TargetDevice = ComputeDevice::kCUDA>
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
            }

        private:            
            std::unique_ptr<Cuda::Vector<float>>            m_computeModelData;
            std::unique_ptr<Cuda::Vector<float>>            m_computeGradData;
            std::unique_ptr<Cuda::Vector<OutputSample>>     m_computeOutputSamples;
            std::unique_ptr<Cuda::Vector<float>>            m_computeSampleLosses;
            std::unique_ptr<Cuda::Object<float>>            m_computeMiniBatchLoss;
            std::unique_ptr<Cuda::Vector<InputSample>>      m_computeInputSamples;
            std::unique_ptr<Cuda::Vector<OutputSample>>     m_computeOutputSamples;

            std::vector<InputSample>                        m_hostInputSamples;
            std::vector<OutputSample>                       m_hostOutputSamples;

            TrainingKernelData<Policy>                      m_kernelData;
            std::vector<std::pair<int, float>>              m_epochLoss;
            std::vector<float>                              m_miniBatchLoss;
            double                                          m_totalTrainingTime;
            int                                             m_numEpochs;

            cudaStream_t                                    m_cudaStream;
            HighResTimer                                    m_kernelTimer;

        public:
            MLP() : 
                m_computeModelData(new Cuda::Vector<float>(TargetDevice)),
                m_totalTrainTime(0)
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
                m_computeOutputSamples.reset();
                m_computeSampleLosses.reset();
                m_computeMiniBatchLoss.reset();

                m_epochLoss.clear();
                m_miniBatchLoss.clear();
                m_numEpochs = 0;
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
                m_computeOutputSamples.reset(Cuda::Vector<OutputSample>(TargetDevice, hostInputSamples.size()));
                m_computeSampleLosses.reset(Cuda::Vector<float>(TargetDevice, MiniBatchSize));
                m_computeMiniBatchLoss.reset(Cuda::Object<float>(TargetDevice));

                // Determininstically initialise the mini-batch weights and the optimiser 
                std::vector<float> hostModelData(Model::kNumParams);
                auto rng = ModelInitialiser();
                Model::Initialise(hostModelData, rng);

                m_computeModelData <<= hostModelData;

                // Create and initialise the optimiser
                Cuda::Vector<float> computeOptimiserData(TargetDevice, Policy::Model::kNumParams * 2, 0.f);

                // Upload the samples
                computeInputSamples <<= hostInputSamples;
                computeTargetSamples <<= hostTargetSamples;
                computeTargetSamples.Resize(hostTargetSamples.size());

                // Create random indirection buffer
                Permutation sampleIdxs(TargetDevice, hostInputSamples.size());
                sampleIdxs.Randomise();
                //sampleIdxs.Sequential();

                // Initialise the kernel data structure
                m_kernelData.mlpModelData = m_computeModelData.GetComputeData();
                m_kernelData.mlpGradData = computeGradData.GetComputeData();
                m_kernelData.inputSamples = computeInputSamples.GetComputeData();
                m_kernelData.outputSamples = computeOutputSamples.GetComputeData();
                m_kernelData.targetSamples = computeTargetSamples.GetComputeData();
                m_kernelData.optimiserData = computeOptimiserData.GetComputeData();
                m_kernelData.sampleIdxs = sampleIdxs.GetComputeData();
                m_kernelData.sampleLosses = computeSampleLosses.GetComputeData();
                m_kernelData.miniBatchLoss = computeMiniBatchLoss.GetComputeData();
                m_kernelData.batchSize = hostInputSamples.size();
            }

            virtual void TrainEpoch() override final
            {
                AssertFmt(m_computeGradData, "Training has not been initialised. Called PrepareTraining() first.");
                
                // Shuffle the indirection indices
                sampleIdxs.Shuffle();
                m_kernelTimer.Reset();
                
                float meanLoss = 0;
                int miniBatchIdx = 0;
                for (int sampleIdx = 0; sampleIdx < kernelData.batchSize && miniBatchIdx < kMaxMiniBatches; sampleIdx += MiniBatchSize, ++miniBatchIdx)
                {
                    computeGradData.Fill(0.f);

                    // Reset the kernel data (loss values, etc.) for the new epoch
                    Trainer::PrepareNewEpoch(m_kernelData, m_cudaStream);

                    // Estimate the gradients
                    Trainer::EstimateGradients(m_kernelData, sampleIdx, m_cudaStream);

                    // Optimiser step
                    Optimiser<TargetDevice, Policy>::Descend(m_kernelData, epochIdx, sampleIdx, m_cudaStream);

                    IsOk(cudaStreamSynchronize(m_cudaStream));
                    const float loss = computeMiniBatchLoss.Download();
                    m_miniBatchLoss.emplace_back(loss);
                    meanLoss += loss;
                }
                
                // Record the epoch loss
                meanLoss /= std::ceil(kernelData.batchSize / float(MiniBatchSize));
                m_epochLoss.emplace_back(miniBatchIdx, meanLoss);
                m_totalTrainingTime += m_kernelTimer.Get();
                ++m_numEpochs;
            }

            virtual void PrepareInference(const int inferBatchSize) override final
            {
                m_computeInputSamples.reset(new Cuda::Vector<InputSample>(TargetDevice, inferBatchSize));
                m_computeOutputSamples.reset(new Cuda::Vector<OutputSample>(TargetDevice, inferBatchSize));
            }

            virtual void ResetInference() override final
            {
                m_computeInputSamples.reset();
                m_computeOutputSamples.reset();

                std::vector<InputSample> hostInputSamples;
                std::vector<OutputSample> hostOutputSamples;
                hostInputSamples.reserve(MiniBatchSize);
                hostOutputSamples.reserve(MiniBatchSize);
            }

            virtual void Infer(DataAccessor<InputSample, OutputSample>& accessor) override final
            {
                AssertFmt(!m_computeModelData->IsEmpty(), "Model has not been initialised. Run a training cycle or load pre-trained weights first.");
                AssertFmt(m_computeInputSamples, "Inference has not been initialised. Call PrepareInference() first. ");
                
                // Initialise the kernel data structure
                InferenceKernelData<Policy> kernelData;
                kernelData.mlpModelData = m_computeModelData.GetComputeData();

                // Process the samples in batches
                for (int sampleIdx = 0; sampleIdx < accessor.Size(); sampleIdx += m_computeInputSamples->Size())
                {          
                    // Load a batch of samples from the accessor
                    hostInputSamples.clear();
                    for (int batchIdx = 0; batchIdx < MiniBatchSize && sampleIdx + batchIdx < accessor.Size(); ++batchIdx)
                    {
                        hostInputSamples.push_back(accessor.Load(sampleIdx + batchIdx));
                    }
                    hostOutputSamples.resize(hostInputSamples.size());

                    // Upload to the device
                    computeInputSamples <<= hostInputSamples;
                    kernelData.batchSize = hostInputSamples.size();
                    kernelData.inputSamples = computeInputSamples.GetComputeData();
                    kernelData.outputSamples = computeOutputSamples.GetComputeData();

                    // Run the inference pass
                    MLPInferer<TargetDevice, Policy>::InferBatch(kernelData);

                    // Store the output samples 
                    hostOutputSamples <<= computeOutputSamples;
                    for (int batchIdx = 0; batchIdx < accessor.Size(); ++batchIdx)
                    {
                        accessor.Store(hostOutputSamples[batchIdx], sampleIdx + batchIdx);
                    }
                }
            }
        };
    }  
}