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
    struct MLPStats
    {
        int numEpochs;
        int numMiniBatches;
        float loss;
    };
    
    template<typename Model, 
                typename ModelInitialiser, 
                typename ActivationFunction, 
                typename LossFunction, 
                typename OptimiserFunction, 
                int MiniBatchSize = 128, 
                ComputeDevice TargetDevice = ComputeDevice::kCUDA>
    class MLP : public NNModel<Tensor1D<Model::kInputWidth>, Tensor1D<Model::kOutputWidth>>
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
        std::unique_ptr<Cuda::Vector<float>>            m_computeOptimiserData;
        std::unique_ptr<Cuda::Vector<float>>            m_computeGradData;
        std::unique_ptr<Cuda::Vector<float>>            m_computeSampleLosses;
        std::unique_ptr<Cuda::Object<float>>            m_computeMiniBatchLoss;
        std::shared_ptr<Cuda::Vector<InputSample>>      m_computeTrainInputSamples;
        std::unique_ptr<Cuda::Vector<OutputSample>>     m_computeTrainOutputSamples;
        std::shared_ptr<Cuda::Vector<OutputSample>>     m_computeTrainTargetSamples;
        std::unique_ptr<Permutation>                    m_sampleIdxs;

        std::unique_ptr<Cuda::Vector<InputSample>>      m_computeInferInputSamples;
        std::unique_ptr<Cuda::Vector<OutputSample>>     m_computeInferOutputSamples;
        int                                             m_inferenceBatchSize;
        int                                             m_inferenceSetSize;

        TrainingKernelData<Policy>                      m_kernelData;
        std::vector<std::pair<int, float>>              m_epochLoss;
        std::vector<float>                              m_miniBatchLoss;
        double                                          m_totalTrainingTime;
        int                                             m_epochIdx;
        int                                             m_miniBatchIdx;

        HighResTimer                                    m_kernelTimer;

        using AccessorType = DataAccessor<InputSample, OutputSample>;
        AccessorType& m_accessor;

    public:
        MLP(AccessorType& accessor, cudaStream_t stream = nullptr) :
            NNModel<InputSample, OutputSample>(stream),
            m_accessor(accessor)
        {
            // Model data
            m_computeModelData.reset(new Cuda::Vector<float>(TargetDevice));

            // Training data
            m_computeTrainInputSamples.reset(new Cuda::Vector<InputSample>(TargetDevice));
            m_computeTrainOutputSamples.reset(new Cuda::Vector<OutputSample>(TargetDevice));
            m_computeTrainTargetSamples.reset(new Cuda::Vector<OutputSample>(TargetDevice));
            m_computeSampleLosses.reset(new Cuda::Vector<float>(TargetDevice));
            m_computeMiniBatchLoss.reset(new Cuda::Object<float>(TargetDevice));
            m_computeGradData.reset(new Cuda::Vector<float>(TargetDevice));
            m_computeOptimiserData.reset(new Cuda::Vector<float>(TargetDevice));
            m_sampleIdxs.reset(new Permutation(TargetDevice, m_cudaStream));

            // Inference data
            m_computeInferInputSamples.reset(new Cuda::Vector<InputSample>(TargetDevice));
            m_computeInferOutputSamples.reset(new Cuda::Vector<OutputSample>(TargetDevice));
        }

        ~MLP() {}

        __host__ virtual std::pair< std::shared_ptr<Cuda::Vector<InputSample>>, std::shared_ptr<Cuda::Vector<OutputSample>>> GetTrainingDataObjects() override final
        {
            return { m_computeTrainInputSamples, m_computeTrainTargetSamples };
        }

        virtual void Initialise() override final
        {
            m_epochLoss.clear();
            m_miniBatchLoss.clear();
            m_epochIdx = 0;
            m_miniBatchIdx = 0;
            m_totalTrainingTime = 0;

            // Determininstically initialise the model weights using the specified initailiser
            std::vector<float> hostModelData(Model::kNumParams);
            auto rng = ModelInitialiser();
            Model::Initialise(hostModelData, rng);
            *m_computeModelData <<= hostModelData;

            // Check that the model fits in shared memory
            const size_t ctxSize = sizeof(TrainingCtx<Policy>);
            cudaDeviceProp prop;
            IsOk(cudaGetDeviceProperties(&prop, 0));
            AssertFmt(ctxSize < prop.sharedMemPerBlock - kSharedMemorySafeMargin, "Model context exceeds capacity of shared memory.");

            printf_red("TrainingCtx: %i bytes\n", ctxSize);
            printf_red("Model size: %i parameters\n", Model::kNumParams);

            // Zero init the gradient data
            m_computeSampleLosses->Resize(kMiniBatchSize);
            m_computeGradData->Resize(kMiniBatchSize * Policy::Model::kNumParams);
            m_computeGradData->Fill(0.f);

            // Initialise the optimiser
            m_computeOptimiserData->Resize(Policy::Model::kNumParams * 2);
            m_computeOptimiserData->Fill(0.f);

            // Populate the kernel data structure
            m_kernelData.mlpModelData = m_computeModelData->GetComputeData();
            m_kernelData.mlpGradData = m_computeGradData->GetComputeData();
            m_kernelData.optimiserData = m_computeOptimiserData->GetComputeData();
            m_kernelData.sampleLosses = m_computeSampleLosses->GetComputeData();
            m_kernelData.miniBatchLoss = m_computeMiniBatchLoss->GetComputeData();
        }

        // Called after intialising the training data but before training an epoch
        virtual void PrepareTraining() override final
        {
            const auto trainingSize = m_computeTrainInputSamples->Size();
            AssertFmt(trainingSize > 0, "No training data is specified. Set it with GetTrainingDataObjects().");
            AssertFmt(m_computeTrainInputSamples->Size() == m_computeTrainTargetSamples->Size(), "Size mismatch!");
            AssertFmt(m_computeTrainInputSamples->GetComputeDevice() == TargetDevice && m_computeTrainTargetSamples->GetComputeDevice() == TargetDevice, "Wrong compute device!");

            m_computeTrainOutputSamples->Resize(trainingSize);

            // If the training size hsa change, reinitialise the indirection buffer
            if (m_sampleIdxs->Size() != trainingSize)
            {
                m_sampleIdxs->Randomise(trainingSize);
                //m_sampleIdxs->Sequential(trainingSize);

                m_kernelData.sampleIdxs = m_sampleIdxs->GetComputeData();
            }
        }    

        // Shuffles the indirection buffer (called after each epoch)
        virtual void Shuffle() override final 
        {
            m_sampleIdxs->Shuffle();
        }

        virtual void TrainEpoch() override final
        {
            // Populate the kernel data structure
            m_kernelData.inputSamples = m_computeTrainInputSamples->GetComputeData();
            m_kernelData.outputSamples = m_computeTrainOutputSamples->GetComputeData();
            m_kernelData.targetSamples = m_computeTrainTargetSamples->GetComputeData();           
            m_kernelData.setSize = m_computeTrainInputSamples->Size();
            m_kernelData.Validate();

            AssertFmt(m_computeGradData, "Training has not been initialised. Called Prepare() first.");

            m_kernelTimer.Reset();

            float meanLoss = 0;
            for (int sampleIdx = 0; sampleIdx < m_kernelData.setSize && m_miniBatchIdx < kMaxMiniBatches; sampleIdx += kMiniBatchSize, ++m_miniBatchIdx)
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
            meanLoss /= std::ceil(m_kernelData.setSize / float(kMiniBatchSize));
            m_epochLoss.emplace_back(m_miniBatchIdx, meanLoss);
            m_totalTrainingTime += m_kernelTimer.Get();
            ++m_epochIdx;
        }

        virtual void Infer() override final
        {
            AssertFmt(!m_computeModelData->IsEmpty(), "Model has not been initialised. Run a training cycle or load pre-trained weights first.");
                
            // Initialise the kernel data structure
            InferenceKernelData<Policy> kernelData;
            kernelData.mlpModelData = m_computeModelData->GetComputeData();

            // Process the samples in batches
            int sampleIdx = 0;
            while(true)
            {                
                const int numLoaded = m_accessor.LoadInferenceBatch(*m_computeInferInputSamples, sampleIdx);
                if (numLoaded == 0) { break; }

                // Prepare the kernel data
                m_computeInferOutputSamples->Resize(numLoaded);
                kernelData.setSize = numLoaded;
                kernelData.inputSamples = m_computeInferInputSamples->GetComputeData();
                kernelData.outputSamples = m_computeInferOutputSamples->GetComputeData();

                // Run the inference pass
                MLPInferer<TargetDevice, Policy>::InferBatch(kernelData, m_cudaStream);
                
                // Store the output samples 
                m_accessor.StoreInferenceBatch(*m_computeInferOutputSamples, sampleIdx);
                sampleIdx += numLoaded;
            }
        }

        std::pair<Cuda::Vector<InputSample>*, Cuda::Vector<OutputSample>*> GetTrainingSet()
        {
            return { m_computeTrainInputSamples.get(), m_computeTrainTargetSamples.get() };
        }

        Cuda::Vector<InputSample>* GetInferenceSet()
        {
            return m_computeInferInputSamples.get();
        }

        MLPStats GetStats()
        {
            MLPStats stats;
            stats.numEpochs = m_epochIdx;
            stats.numMiniBatches = m_miniBatchIdx;
            stats.loss = m_epochLoss.empty() ? -kFltMax : m_epochLoss.back().second;
            
            return stats;
        }
    }; 
}