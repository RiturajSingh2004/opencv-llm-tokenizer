/**
 * @file tokenizer_comparison.cpp
 * @brief Comparative analysis tool for different tokenizer types
 * @author Rituraj Singh
 * @date 2025
 *
 * This application compares different tokenizer implementations
 * side-by-side, measuring efficiency, speed, and token distribution
 * characteristics across various text types and languages.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <opencv2/dnn/tokenizer.hpp>
#include <opencv2/dnn/gpu_tokenizer.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std::chrono;

const char* keys =
    "{ help h      | | Print help message }"
    "{ bpe         | | Path to BPE tokenizer files (vocab.json,merges.txt) }"
    "{ wordpiece   | | Path to WordPiece tokenizer file (vocab.txt) }"
    "{ tiktoken    | | TikToken encoding name (cl100k_base, p50k_base, etc.) }"
    "{ tiktoken_file | | Optional path to custom TikToken BPE file }"
    "{ corpus      | | Path to corpus file for benchmarking (if not provided, will use sample texts) }"
    "{ batch_size  | 100 | Batch size for benchmarking }"
    "{ iterations  | 5 | Number of iterations for benchmarking }"
    "{ gpu         | | Enable GPU acceleration if available }";

// Helper function to split string by comma
std::vector<std::string> splitByComma(const std::string& str) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }
    
    return result;
}

// Helper function to read corpus
std::vector<std::string> readCorpus(const std::string& filename, size_t maxLines = 10000) {
    std::vector<std::string> corpus;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open corpus file: " << filename << std::endl;
        return corpus;
    }
    
    std::string line;
    while (corpus.size() < maxLines && std::getline(file, line)) {
        if (!line.empty()) {
            corpus.push_back(line);
        }
    }
    
    return corpus;
}

// Generate a sample corpus if not provided
std::vector<std::string> generateSampleCorpus(size_t size = 1000) {
    std::vector<std::string> samples = {
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "OpenCV now supports large language model tokenizers.",
        "TikToken is an efficient tokenizer used by OpenAI models.",
        "GPU-accelerated tokenization can significantly improve batch processing performance.",
        "Tokenizers convert text into numerical token IDs that models can understand.",
        "Different tokenization algorithms include BPE, WordPiece, Unigram, and more.",
        "Preprocessing steps often include normalization, splitting, and subword tokenization.",
        "The cl100k_base encoding used in GPT-4 has a vocabulary size of over 100,000 tokens.",
        "Efficient tokenization is critical for the performance of NLP applications."
    };
    
    std::vector<std::string> corpus;
    for (size_t i = 0; i < size; i++) {
        corpus.push_back(samples[i % samples.size()]);
    }
    
    return corpus;
}

// Benchmark a tokenizer
struct BenchmarkResult {
    double avg_encoding_time_ms;
    double avg_batch_encoding_time_ms;
    double avg_tokens_per_text;
    double throughput_texts_per_second;
    double throughput_batch_texts_per_second;
};

BenchmarkResult benchmarkTokenizer(
    const Ptr<Tokenizer>& tokenizer,
    const std::vector<std::string>& corpus,
    int batch_size,
    int iterations
) {
    BenchmarkResult result = {};
    
    // Prepare batches
    std::vector<std::vector<std::string>> batches;
    for (size_t i = 0; i < corpus.size(); i += batch_size) {
        std::vector<std::string> batch;
        for (size_t j = 0; j < batch_size && i + j < corpus.size(); j++) {
            batch.push_back(corpus[i + j]);
        }
        batches.push_back(batch);
    }
    
    // Measure encoding time (individual texts)
    double total_encoding_time = 0;
    size_t total_tokens = 0;
    
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        
        for (const auto& text : corpus) {
            auto tokens = tokenizer->encode(text);
            
            if (i == 0) {
                total_tokens += tokens.size();
            }
        }
        
        auto end = high_resolution_clock::now();
        total_encoding_time += duration_cast<milliseconds>(end - start).count();
    }
    
    result.avg_encoding_time_ms = total_encoding_time / iterations;
    result.avg_tokens_per_text = static_cast<double>(total_tokens) / corpus.size();
    result.throughput_texts_per_second = corpus.size() * 1000.0 / result.avg_encoding_time_ms;
    
    // Measure batch encoding time
    double total_batch_encoding_time = 0;
    
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        
        for (const auto& batch : batches) {
            tokenizer->encodeBatch(batch);
        }
        
        auto end = high_resolution_clock::now();
        total_batch_encoding_time += duration_cast<milliseconds>(end - start).count();
    }
    
    result.avg_batch_encoding_time_ms = total_batch_encoding_time / iterations;
    result.throughput_batch_texts_per_second = corpus.size() * 1000.0 / result.avg_batch_encoding_time_ms;
    
    return result;
}

// Benchmark with GPU acceleration
BenchmarkResult benchmarkGPUTokenizer(
    const Ptr<GPUTokenizer>& gpuTokenizer,
    const std::vector<std::string>& corpus,
    int batch_size,
    int iterations
) {
    BenchmarkResult result = {};
    
    // Prepare batches
    std::vector<std::vector<std::string>> batches;
    for (size_t i = 0; i < corpus.size(); i += batch_size) {
        std::vector<std::string> batch;
        for (size_t j = 0; j < batch_size && i + j < corpus.size(); j++) {
            batch.push_back(corpus[i + j]);
        }
        batches.push_back(batch);
    }
    
    // Measure batch encoding time
    double total_batch_encoding_time = 0;
    
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        
        for (const auto& batch : batches) {
            gpuTokenizer->encodeBatch(batch);
        }
        
        auto end = high_resolution_clock::now();
        total_batch_encoding_time += duration_cast<milliseconds>(end - start).count();
    }
    
    result.avg_batch_encoding_time_ms = total_batch_encoding_time / iterations;
    result.throughput_batch_texts_per_second = corpus.size() * 1000.0 / result.avg_batch_encoding_time_ms;
    
    // Measure GPU matrix encoding time
    double total_gpu_mat_encoding_time = 0;
    cuda::Stream stream;
    
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        
        for (const auto& batch : batches) {
            gpuTokenizer->encodeBatchToGpuMat(batch, 0, false, stream);
            stream.waitForCompletion();
        }
        
        auto end = high_resolution_clock::now();
        total_gpu_mat_encoding_time += duration_cast<milliseconds>(end - start).count();
    }
    
    // Get token count for metrics
    size_t total_tokens = 0;
    for (const auto& text : corpus) {
        auto tokens = gpuTokenizer->getBaseTokenizer()->encode(text);
        total_tokens += tokens.size();
    }
    
    result.avg_encoding_time_ms = total_gpu_mat_encoding_time / iterations;
    result.avg_tokens_per_text = static_cast<double>(total_tokens) / corpus.size();
    result.throughput_texts_per_second = corpus.size() * 1000.0 / (total_gpu_mat_encoding_time / iterations);
    
    return result;
}

// Print benchmark results
void printResults(const std::string& tokenizer_name, const BenchmarkResult& result) {
    std::cout << "=== " << tokenizer_name << " ===" << std::endl;
    std::cout << "  Avg tokens per text: " << std::fixed << std::setprecision(2) << result.avg_tokens_per_text << std::endl;
    std::cout << "  Single text encoding:" << std::endl;
    std::cout << "    Avg time: " << std::fixed << std::setprecision(2) << result.avg_encoding_time_ms << " ms" << std::endl;
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2) << result.throughput_texts_per_second << " texts/sec" << std::endl;
    std::cout << "  Batch encoding:" << std::endl;
    std::cout << "    Avg time: " << std::fixed << std::setprecision(2) << result.avg_batch_encoding_time_ms << " ms" << std::endl;
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2) << result.throughput_batch_texts_per_second << " texts/sec" << std::endl;
    std::cout << "  Speedup from batching: " << std::fixed << std::setprecision(2) 
              << result.throughput_batch_texts_per_second / result.throughput_texts_per_second << "x" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV DNN Tokenizer Comparison");
    
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    
    // Load corpus
    std::vector<std::string> corpus;
    if (parser.has("corpus")) {
        std::string corpus_file = parser.get<std::string>("corpus");
        corpus = readCorpus(corpus_file);
    } else {
        corpus = generateSampleCorpus(1000);
    }
    
    if (corpus.empty()) {
        std::cerr << "Corpus is empty. Please provide a valid corpus file." << std::endl;
        return -1;
    }
    
    std::cout << "Loaded corpus with " << corpus.size() << " texts" << std::endl;
    
    // Get benchmark parameters
    int batch_size = parser.get<int>("batch_size");
    int iterations = parser.get<int>("iterations");
    bool use_gpu = parser.has("gpu") && GPUTokenizer::isAvailable();
    
    if (use_gpu) {
        std::cout << "GPU acceleration is available and enabled" << std::endl;
    } else if (parser.has("gpu")) {
        std::cout << "GPU acceleration requested but not available" << std::endl;
    }
    
    // Create tokenizers
    std::vector<std::pair<std::string, Ptr<Tokenizer>>> tokenizers;
    
    if (parser.has("bpe")) {
        std::vector<std::string> bpe_files = splitByComma(parser.get<std::string>("bpe"));
        if (bpe_files.size() >= 2) {
            std::string vocab_file = bpe_files[0];
            std::string merges_file = bpe_files[1];
            
            try {
                Ptr<Tokenizer> bpe_tokenizer = createBPETokenizer(vocab_file, merges_file);
                tokenizers.push_back({"BPE", bpe_tokenizer});
                std::cout << "Created BPE tokenizer with vocabulary size: " << bpe_tokenizer->getVocabSize() << std::endl;
            } catch (const Exception& e) {
                std::cerr << "Error creating BPE tokenizer: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "BPE tokenizer requires both vocab and merges files separated by comma" << std::endl;
        }
    }
    
    if (parser.has("wordpiece")) {
        std::string vocab_file = parser.get<std::string>("wordpiece");
        
        try {
            Ptr<Tokenizer> wordpiece_tokenizer = createWordPieceTokenizer(vocab_file);
            tokenizers.push_back({"WordPiece", wordpiece_tokenizer});
            std::cout << "Created WordPiece tokenizer with vocabulary size: " << wordpiece_tokenizer->getVocabSize() << std::endl;
        } catch (const Exception& e) {
            std::cerr << "Error creating WordPiece tokenizer: " << e.what() << std::endl;
        }
    }
    
    if (parser.has("tiktoken")) {
        std::string encoding_name = parser.get<std::string>("tiktoken");
        std::string tiktoken_file = parser.has("tiktoken_file") ? parser.get<std::string>("tiktoken_file") : "";
        
        try {
            Ptr<Tokenizer> tiktoken_tokenizer = createTikTokenTokenizer(encoding_name, tiktoken_file);
            tokenizers.push_back({"TikToken (" + encoding_name + ")", tiktoken_tokenizer});
            std::cout << "Created TikToken tokenizer with vocabulary size: " << tiktoken_tokenizer->getVocabSize() << std::endl;
        } catch (const Exception& e) {
            std::cerr << "Error creating TikToken tokenizer: " << e.what() << std::endl;
        }
    }
    
    if (tokenizers.empty()) {
        std::cerr << "No tokenizers created. Please provide at least one tokenizer." << std::endl;
        return -1;
    }
    
    // Run benchmarks
    std::cout << "\nRunning benchmarks with batch_size=" << batch_size 
              << ", iterations=" << iterations << std::endl << std::endl;
    
    for (const auto& [name, tokenizer] : tokenizers) {
        // CPU benchmark
        BenchmarkResult cpu_result = benchmarkTokenizer(tokenizer, corpus, batch_size, iterations);
        printResults(name + " (CPU)", cpu_result);
        
        // GPU benchmark
        if (use_gpu) {
            Ptr<GPUTokenizer> gpu_tokenizer = createGPUTokenizer(tokenizer);
            
            // Warm up
            gpu_tokenizer->warmup();
            
            BenchmarkResult gpu_result = benchmarkGPUTokenizer(gpu_tokenizer, corpus, batch_size, iterations);
            printResults(name + " (GPU)", gpu_result);
            
            std::cout << "  GPU vs CPU speedup: " << std::fixed << std::setprecision(2)
                      << gpu_result.throughput_batch_texts_per_second / cpu_result.throughput_batch_texts_per_second
                      << "x" << std::endl << std::endl;
        }
    }
    
    return 0;
}