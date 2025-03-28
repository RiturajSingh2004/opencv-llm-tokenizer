/**
 * @file test_gpu_tokenizer.cpp
 * @brief Tests for GPU-accelerated tokenizer implementation
 * @author Rituraj Singh
 * @date 2025
 *
 * This file contains unit tests for the GPUTokenizer class,
 * verifying acceleration performance, correctness of results,
 * and proper CPU fallback behavior when GPU is unavailable.
 */

#include "test_precomp.hpp"
#include <opencv2/dnn/gpu_tokenizer.hpp>
#include <opencv2/dnn/tokenizer.hpp>
#include <opencv2/core/cuda.hpp>
#include <fstream>

namespace opencv_test {
namespace {

// Helper function to create a temporary file with content
std::string createTempFile(const std::string& content, const std::string& extension = ".txt") {
    const std::string tempFileName = cv::tempfile(extension.c_str());
    std::ofstream file(tempFileName);
    file << content;
    file.close();
    return tempFileName;
}

// BPE vocab and merges for testing
const std::string TEST_BPE_VOCAB = R"(
{
  "h": 0,
  "e": 1,
  "l": 2,
  "o": 3,
  "w": 4,
  "r": 5,
  "d": 6,
  "hello": 7,
  "world": 8,
  "<|endoftext|>": 9
}
)";

const std::string TEST_BPE_MERGES = R"(#version: 0.2
h e
he l
hel l
hell o
w o
wo r
wor l
worl d
)";

class GPUTokenizerTest : public testing::Test {
protected:
    void SetUp() override {
        // Create test files
        bpeVocabFile = createTempFile(TEST_BPE_VOCAB, ".json");
        bpeMergesFile = createTempFile(TEST_BPE_MERGES, ".txt");
        
        // Create base tokenizer
        baseTokenizer = cv::dnn::createBPETokenizer(bpeVocabFile, bpeMergesFile);
    }

    void TearDown() override {
        // Clean up temporary files
        std::remove(bpeVocabFile.c_str());
        std::remove(bpeMergesFile.c_str());
    }

    std::string bpeVocabFile;
    std::string bpeMergesFile;
    cv::Ptr<cv::dnn::Tokenizer> baseTokenizer;
};

TEST_F(GPUTokenizerTest, CreateGPUTokenizer) {
    // Skip if CUDA is not available
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        std::cout << "CUDA is not available, skipping test" << std::endl;
        GTEST_SKIP();
    }
    
    // Create GPU tokenizer
    cv::Ptr<cv::dnn::GPUTokenizer> gpuTokenizer = cv::dnn::createGPUTokenizer(baseTokenizer);
    EXPECT_NE(gpuTokenizer, nullptr);
    
    // Check if base tokenizer is correctly stored
    cv::Ptr<cv::dnn::Tokenizer> retrievedBaseTokenizer = gpuTokenizer->getBaseTokenizer();
    EXPECT_EQ(retrievedBaseTokenizer, baseTokenizer);
}

TEST_F(GPUTokenizerTest, CheckAvailability) {
    bool isAvailable = cv::dnn::GPUTokenizer::isAvailable();
    
    // This should match CUDA availability
    EXPECT_EQ(isAvailable, cv::cuda::getCudaEnabledDeviceCount() > 0);
}

TEST_F(GPUTokenizerTest, EncodeBatch) {
    // Skip if CUDA is not available
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        std::cout << "CUDA is not available, skipping test" << std::endl;
        GTEST_SKIP();
    }
    
    // Create GPU tokenizer
    cv::Ptr<cv::dnn::GPUTokenizer> gpuTokenizer = cv::dnn::createGPUTokenizer(baseTokenizer);
    
    // Test batch encoding
    std::vector<std::string> texts = {
        "hello", "world", "hello world", "hello hello world",
        "hello", "world", "hello world", "hello hello world"
    };
    
    // Encode with GPU tokenizer
    std::vector<std::vector<int>> gpuTokens = gpuTokenizer->encodeBatch(texts);
    
    // Encode with CPU tokenizer for comparison
    std::vector<std::vector<int>> cpuTokens = baseTokenizer->encodeBatch(texts);
    
    // Compare results
    EXPECT_EQ(gpuTokens.size(), cpuTokens.size());
    for (size_t i = 0; i < gpuTokens.size(); i++) {
        EXPECT_EQ(gpuTokens[i], cpuTokens[i]);
    }
}

TEST_F(GPUTokenizerTest, EncodeBatchToGpuMat) {
    // Skip if CUDA is not available
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        std::cout << "CUDA is not available, skipping test" << std::endl;
        GTEST_SKIP();
    }
    
    // Create GPU tokenizer
    cv::Ptr<cv::dnn::GPUTokenizer> gpuTokenizer = cv::dnn::createGPUTokenizer(baseTokenizer);
    
    // Test batch encoding to GPU matrix
    std::vector<std::string> texts = {
        "hello world", "world hello", "hello hello", "world world"
    };
    
    // Set max length and enable padding
    int max_length = 3;
    bool padding = true;
    
    // Create CUDA stream
    cv::cuda::Stream stream;
    
    // Encode to GPU matrix
    cv::cuda::GpuMat tokens_gpu = gpuTokenizer->encodeBatchToGpuMat(texts, max_length, padding, stream);
    
    // Wait for GPU operations to complete
    stream.waitForCompletion();
    
    // Check dimensions
    EXPECT_EQ(tokens_gpu.rows, texts.size());
    EXPECT_EQ(tokens_gpu.cols, max_length);
    EXPECT_EQ(tokens_gpu.type(), CV_32S);
    
    // Download results to verify
    cv::Mat tokens_cpu;
    tokens_gpu.download(tokens_cpu);
    
    // Encode with CPU tokenizer for comparison
    std::vector<std::vector<int>> cpuTokens = baseTokenizer->encodeBatch(texts);
    
    // Compare results, accounting for padding
    for (int i = 0; i < texts.size(); i++) {
        for (int j = 0; j < max_length; j++) {
            if (j < cpuTokens[i].size()) {
                EXPECT_EQ(tokens_cpu.at<int>(i, j), cpuTokens[i][j]);
            } else {
                // Should be pad token (0 in our test case)
                EXPECT_EQ(tokens_cpu.at<int>(i, j), 0);
            }
        }
    }
}

TEST_F(GPUTokenizerTest, Warmup) {
    // Skip if CUDA is not available
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        std::cout << "CUDA is not available, skipping test" << std::endl;
        GTEST_SKIP();
    }
    
    // Create GPU tokenizer
    cv::Ptr<cv::dnn::GPUTokenizer> gpuTokenizer = cv::dnn::createGPUTokenizer(baseTokenizer);
    
    // Test warmup (should not throw)
    EXPECT_NO_THROW(gpuTokenizer->warmup());
}

TEST_F(GPUTokenizerTest, PerformanceComparison) {
    // Skip if CUDA is not available
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        std::cout << "CUDA is not available, skipping test" << std::endl;
        GTEST_SKIP();
    }
    
    // Create GPU tokenizer
    cv::Ptr<cv::dnn::GPUTokenizer> gpuTokenizer = cv::dnn::createGPUTokenizer(baseTokenizer);
    
    // Create a larger batch for performance testing
    std::vector<std::string> texts;
    for (int i = 0; i < 1000; i++) {
        texts.push_back("hello world hello world hello world");
    }
    
    // Time CPU encoding
    int iterations = 5;
    double cpu_time = 0;
    for (int i = 0; i < iterations; i++) {
        auto start = cv::getTickCount();
        baseTokenizer->encodeBatch(texts);
        auto end = cv::getTickCount();
        cpu_time += (end - start) / cv::getTickFrequency() * 1000; // ms
    }
    cpu_time /= iterations;
    
    // Time GPU encoding
    double gpu_time = 0;
    for (int i = 0; i < iterations; i++) {
        auto start = cv::getTickCount();
        gpuTokenizer->encodeBatch(texts);
        auto end = cv::getTickCount();
        gpu_time += (end - start) / cv::getTickFrequency() * 1000; // ms
    }
    gpu_time /= iterations;
    
    // Time GPU matrix encoding
    double gpu_mat_time = 0;
    cv::cuda::Stream stream;
    for (int i = 0; i < iterations; i++) {
        auto start = cv::getTickCount();
        gpuTokenizer->encodeBatchToGpuMat(texts, 10, true, stream);
        stream.waitForCompletion();
        auto end = cv::getTickCount();
        gpu_mat_time += (end - start) / cv::getTickFrequency() * 1000; // ms
    }
    gpu_mat_time /= iterations;
    
    std::cout << "Performance comparison (ms):" << std::endl;
    std::cout << "  CPU encoding: " << cpu_time << std::endl;
    std::cout << "  GPU encoding: " << gpu_time << std::endl;
    std::cout << "  GPU matrix encoding: " << gpu_mat_time << std::endl;
    
    // Don't assert on performance, just log results
    // GPU should generally be faster for large batches
}

TEST_F(GPUTokenizerTest, FallbackToCPU) {
    // Create GPU tokenizer
    cv::Ptr<cv::dnn::GPUTokenizer> gpuTokenizer = cv::dnn::createGPUTokenizer(baseTokenizer);
    
    // Test with a small batch, which should fall back to CPU
    std::vector<std::string> smallBatch = {"hello", "world"};
    
    // Encode with GPU tokenizer
    std::vector<std::vector<int>> gpuTokens = gpuTokenizer->encodeBatch(smallBatch);
    
    // Encode with CPU tokenizer for comparison
    std::vector<std::vector<int>> cpuTokens = baseTokenizer->encodeBatch(smallBatch);
    
    // Results should be identical
    EXPECT_EQ(gpuTokens, cpuTokens);
}

}} // namespace