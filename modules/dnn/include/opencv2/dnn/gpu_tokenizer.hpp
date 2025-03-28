#ifndef OPENCV_DNN_GPU_TOKENIZER_HPP
#define OPENCV_DNN_GPU_TOKENIZER_HPP

#include <opencv2/core.hpp>
#include <opencv2/dnn/tokenizer.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <vector>
#include <memory>

namespace cv {
namespace dnn {

/**
 * @brief GPU-accelerated tokenizer wrapper
 *
 * This class wraps around a standard tokenizer and provides GPU acceleration
 * for batch encoding operations. It supports CUDA for parallel processing
 * of large text batches.
 */
class CV_EXPORTS_W GPUTokenizer {
public:
    /**
     * @brief Constructor for GPU tokenizer
     * @param baseTokenizer Base CPU tokenizer to wrap
     */
    CV_WRAP GPUTokenizer(const Ptr<Tokenizer>& baseTokenizer);
    
    /**
     * @brief Destructor
     */
    virtual ~GPUTokenizer();
    
    /**
     * @brief Batch encode texts using GPU acceleration
     * @param texts Vector of texts to encode
     * @param stream CUDA stream to use (optional)
     * @return Vector of vectors of token IDs
     */
    CV_WRAP std::vector<std::vector<int>> encodeBatch(
        const std::vector<std::string>& texts,
        cuda::Stream& stream = cuda::Stream::Null()) const;
    
    /**
     * @brief Batch encode texts directly to GPU memory
     * @param texts Vector of texts to encode
     * @param max_length Maximum length for padding (0 for no padding)
     * @param padding Whether to pad sequences to max_length
     * @param stream CUDA stream to use (optional)
     * @return GpuMat containing tokens (shape: batch_size x max_length)
     */
    CV_WRAP cuda::GpuMat encodeBatchToGpuMat(
        const std::vector<std::string>& texts,
        int max_length = 0,
        bool padding = false,
        cuda::Stream& stream = cuda::Stream::Null()) const;
    
    /**
     * @brief Get the underlying CPU tokenizer
     * @return Base tokenizer
     */
    CV_WRAP Ptr<Tokenizer> getBaseTokenizer() const;
    
    /**
     * @brief Check if GPU acceleration is available
     * @return True if CUDA is available
     */
    CV_WRAP static bool isAvailable();
    
    /**
     * @brief Warm up the GPU for better performance
     * This function initializes GPU resources to avoid first-run overhead
     */
    CV_WRAP void warmup();

private:
    // Base CPU tokenizer
    Ptr<Tokenizer> baseTokenizer_;
    
    // CUDA-specific implementation details
    class Impl;
    Ptr<Impl> impl_;
};

/**
 * @brief Create a GPU-accelerated tokenizer
 * @param baseTokenizer Base CPU tokenizer to wrap
 * @return Smart pointer to the created GPU tokenizer
 */
CV_EXPORTS_W Ptr<GPUTokenizer> createGPUTokenizer(const Ptr<Tokenizer>& baseTokenizer);

}} // namespace cv::dnn

#endif // OPENCV_DNN_GPU_TOKENIZER_HPP