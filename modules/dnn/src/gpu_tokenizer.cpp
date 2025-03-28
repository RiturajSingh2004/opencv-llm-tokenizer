/**
 * @file gpu_tokenizer.cpp
 * @brief Implementation of GPU-accelerated tokenization
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements GPU acceleration for tokenization operations,
 * particularly focused on batch encoding for large text volumes.
 * It uses CUDA to parallelize tokenization processes for improved performance.
 */

#include "precomp.hpp"
#include <opencv2/dnn/gpu_tokenizer.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernels
namespace {

// CUDA kernel for copying token IDs to GPU memory
__global__ void copyTokensToGpuKernel(
    const int* h_tokens, int* d_tokens,
    const int* h_offsets, int batch_size, int max_length, int pad_token_id) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * max_length) {
        int batch_idx = idx / max_length;
        int token_idx = idx % max_length;
        
        // Calculate offset for this batch item
        int offset = batch_idx > 0 ? h_offsets[batch_idx - 1] : 0;
        int length = h_offsets[batch_idx] - offset;
        
        // Copy token if within range, otherwise use padding
        if (token_idx < length) {
            d_tokens[idx] = h_tokens[offset + token_idx];
        } else {
            d_tokens[idx] = pad_token_id;
        }
    }
}

} // anonymous namespace
#endif

namespace cv {
namespace dnn {

// ThreadPool for parallel CPU pre-processing
class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
    
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Implementation details
class GPUTokenizer::Impl {
public:
    Impl() {
        // Get number of threads for CPU parallel pre-processing
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 4;  // Default to 4 if detection fails
        }
        threadPool = std::make_unique<ThreadPool>(num_threads);
    }
    
    ~Impl() {
        // Free any allocated CUDA resources
    }
    
    bool checkCudaAvailability() const {
#ifdef HAVE_CUDA
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        return (error == cudaSuccess && deviceCount > 0);
#else
        return false;
#endif
    }
    
    std::vector<std::vector<int>> encodeBatchParallel(
        const Ptr<Tokenizer>& tokenizer, 
        const std::vector<std::string>& texts) const {
        
        size_t batch_size = texts.size();
        std::vector<std::vector<int>> result(batch_size);
        
        // Use thread pool for parallel encoding
        std::mutex result_mutex;
        std::atomic<size_t> completed(0);
        std::condition_variable cv;
        
        for (size_t i = 0; i < batch_size; ++i) {
            threadPool->enqueue([&, i]() {
                try {
                    auto tokens = tokenizer->encode(texts[i]);
                    
                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        result[i] = std::move(tokens);
                    }
                } catch (const cv::Exception& e) {
                    std::cerr << "Error encoding text: " << e.what() << std::endl;
                }
                
                completed++;
                cv.notify_one();
            });
        }
        
        // Wait for all tasks to complete
        std::unique_lock<std::mutex> lock(result_mutex);
        cv.wait(lock, [&completed, batch_size]() { return completed == batch_size; });
        
        return result;
    }
    
    cuda::GpuMat encodeBatchToGpuMatImpl(
        const Ptr<Tokenizer>& tokenizer,
        const std::vector<std::string>& texts,
        int max_length,
        bool padding,
        cuda::Stream& stream) const {
        
#ifdef HAVE_CUDA
        // First, encode all texts in parallel on CPU
        std::vector<std::vector<int>> tokenized = encodeBatchParallel(tokenizer, texts);
        
        size_t batch_size = texts.size();
        
        // Determine max_length if not provided
        if (max_length <= 0) {
            if (padding) {
                // Find the longest sequence
                for (const auto& tokens : tokenized) {
                    max_length = std::max(max_length, static_cast<int>(tokens.size()));
                }
            } else {
                // Just use the first sequence length
                max_length = tokenized.empty() ? 0 : static_cast<int>(tokenized[0].size());
            }
        }
        
        // No tokens case
        if (max_length == 0) {
            return cuda::GpuMat();
        }
        
        // Prepare for copying to GPU
        // 1. Flatten all tokens into a single vector
        // 2. Create offsets vector to track token boundaries
        std::vector<int> all_tokens;
        std::vector<int> offsets;
        size_t total_tokens = 0;
        
        for (const auto& tokens : tokenized) {
            size_t token_count = padding ? std::min(tokens.size(), static_cast<size_t>(max_length)) 
                                         : tokens.size();
            
            all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.begin() + token_count);
            total_tokens += token_count;
            offsets.push_back(static_cast<int>(total_tokens));
        }
        
        // Placeholder for padding token ID
        int pad_token_id = 0;
        
        // Create output GPU matrix
        cuda::GpuMat output(batch_size, max_length, CV_32S);
        
        // Copy tokens to GPU in batched format
        if (!all_tokens.empty()) {
            // Prepare data on device
            cuda::GpuMat d_tokens_flat(1, total_tokens, CV_32S);
            cuda::GpuMat d_offsets(1, offsets.size(), CV_32S);
            
            // Copy flattened tokens and offsets to device
            d_tokens_flat.upload(Mat(1, total_tokens, CV_32S, all_tokens.data()), stream);
            d_offsets.upload(Mat(1, offsets.size(), CV_32S, offsets.data()), stream);
            
            // Get raw pointers
            int* d_tokens_ptr = d_tokens_flat.ptr<int>();
            int* d_output_ptr = output.ptr<int>();
            int* d_offsets_ptr = d_offsets.ptr<int>();
            
            // Launch kernel to organize tokens in the output matrix
            cudaStream_t cudaStream = cuda::StreamAccessor::getStream(stream);
            
            int threadsPerBlock = 256;
            int blocksPerGrid = (batch_size * max_length + threadsPerBlock - 1) / threadsPerBlock;
            
            copyTokensToGpuKernel<<<blocksPerGrid, threadsPerBlock, 0, cudaStream>>>(
                d_tokens_ptr, d_output_ptr, d_offsets_ptr, batch_size, max_length, pad_token_id);
        }
        
        return output;
#else
        CV_Error(Error::StsNotImplemented, "CUDA support is not available");
        return cuda::GpuMat();
#endif
    }
    
    std::unique_ptr<ThreadPool> threadPool;
};

GPUTokenizer::GPUTokenizer(const Ptr<Tokenizer>& baseTokenizer) 
    : baseTokenizer_(baseTokenizer) {
    
    impl_ = makePtr<Impl>();
    
    // Check if CUDA is available
    if (!impl_->checkCudaAvailability()) {
        CV_LOG_WARNING(NULL, "CUDA is not available. GPUTokenizer will fall back to CPU processing.");
    }
}

GPUTokenizer::~GPUTokenizer() {
    // Destructor is needed for proper forward declaration
}

std::vector<std::vector<int>> GPUTokenizer::encodeBatch(
    const std::vector<std::string>& texts,
    cuda::Stream& stream) const {
    
    // If no CUDA or small batch, use CPU encoding directly
    if (!impl_->checkCudaAvailability() || texts.size() <= 4) {
        return baseTokenizer_->encodeBatch(texts);
    }
    
    return impl_->encodeBatchParallel(baseTokenizer_, texts);
}

cuda::GpuMat GPUTokenizer::encodeBatchToGpuMat(
    const std::vector<std::string>& texts,
    int max_length,
    bool padding,
    cuda::Stream& stream) const {
    
    return impl_->encodeBatchToGpuMatImpl(baseTokenizer_, texts, max_length, padding, stream);
}

Ptr<Tokenizer> GPUTokenizer::getBaseTokenizer() const {
    return baseTokenizer_;
}

bool GPUTokenizer::isAvailable() {
#ifdef HAVE_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
#else
    return false;
#endif
}

void GPUTokenizer::warmup() {
    // Perform a small encoding to warm up the GPU
    std::vector<std::string> warmup_texts = {"Hello", "World"};
    cuda::Stream stream;
    
    try {
        encodeBatchToGpuMat(warmup_texts, 0, false, stream);
        stream.waitForCompletion();
    } catch (...) {
        // Ignore any errors during warmup
    }
}

Ptr<GPUTokenizer> createGPUTokenizer(const Ptr<Tokenizer>& baseTokenizer) {
    return makePtr<GPUTokenizer>(baseTokenizer);
}

}} // namespace cv::dnn