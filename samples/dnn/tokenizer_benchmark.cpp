/**
 * @file tokenizer_benchmark.cpp
 * @brief Benchmark application for tokenizer performance evaluation
 * @author Rituraj Singh
 * @date 2025
 *
 * This application measures and compares performance metrics for
 * different tokenization algorithms and implementations, including
 * encoding speed, token efficiency, and batch processing throughput.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <opencv2/dnn/tokenizer.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std::chrono;

const char* keys =
    "{ help h      | | Print help message }"
    "{ @tokenizer  | | Tokenizer type (bpe or wordpiece) }"
    "{ @vocab      | | Path to vocabulary file }"
    "{ @merges     | | Path to merges file (only for BPE tokenizer) }"
    "{ corpus      | | Path to corpus file for benchmarking (if not provided, will use sample texts) }"
    "{ iterations  | 10 | Number of iterations for benchmarking }"
    "{ batch_size  | 100 | Batch size for batch encoding }"
    "{ compare     | | Compare with other tokenizers if available (requires Python with transformers or tiktoken) }";

// Function to read corpus from file
std::vector<std::string> readCorpus(const std::string& filePath, size_t maxLines = 1000) {
    std::vector<std::string> texts;
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        std::cerr << "Could not open corpus file: " << filePath << std::endl;
        return texts;
    }
    
    std::string line;
    while (texts.size() < maxLines && std::getline(file, line)) {
        if (!line.empty()) {
            texts.push_back(line);
        }
    }
    
    return texts;
}

// Function to generate sample texts for benchmarking
std::vector<std::string> generateSampleTexts(size_t count = 100) {
    std::vector<std::string> samples = {
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Artificial intelligence is the simulation of human intelligence by machines.",
        "Computer vision is a field of artificial intelligence that enables computers to derive meaningful information from digital images and videos.",
        "Natural language processing (NLP) is a field of artificial intelligence in which computers analyze, understand, and generate human language.",
        "OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library.",
        "The integration of LLM tokenizers into OpenCV provides a bridge between computer vision and natural language processing.",
        "The tokenization process breaks down text into smaller units called tokens, which can be words, subwords, or characters.",
        "Byte-Pair Encoding (BPE) is a data compression technique where the most common pair of consecutive bytes is replaced with a byte that does not occur in the data."
    };
    
    std::vector<std::string> result;
    result.reserve(count);
    
    for (size_t i = 0; i < count; i++) {
        result.push_back(samples[i % samples.size()]);
    }
    
    return result;
}

// Function to benchmark tokenizer
void benchmarkTokenizer(const Ptr<Tokenizer>& tokenizer, 
                        const std::vector<std::string>& texts,
                        int iterations,
                        int batchSize) {
    std::cout << "Benchmarking tokenizer with vocabulary size: " << tokenizer->getVocabSize() << std::endl;
    
    // Prepare batches
    std::vector<std::vector<std::string>> batches;
    for (size_t i = 0; i < texts.size(); i += batchSize) {
        std::vector<std::string> batch;
        for (size_t j = 0; j < batchSize && i + j < texts.size(); j++) {
            batch.push_back(texts[i + j]);
        }
        batches.push_back(batch);
    }
    
    // Benchmark single text encoding
    std::cout << "\nSingle text encoding benchmark:" << std::endl;
    {
        std::vector<double> times;
        size_t totalTokens = 0;
        
        for (int i = 0; i < iterations; i++) {
            auto start = high_resolution_clock::now();
            
            for (const auto& text : texts) {
                auto tokens = tokenizer->encode(text);
                if (i == 0) {
                    totalTokens += tokens.size();
                }
            }
            
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start);
            times.push_back(duration.count());
        }
        
        // Calculate statistics
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min = *std::min_element(times.begin(), times.end());
        double max = *std::max_element(times.begin(), times.end());
        
        double stdDev = 0.0;
        for (double time : times) {
            stdDev += (time - mean) * (time - mean);
        }
        stdDev = std::sqrt(stdDev / times.size());
        
        std::cout << "  Average time: " << mean << " ms" << std::endl;
        std::cout << "  Min time: " << min << " ms" << std::endl;
        std::cout << "  Max time: " << max << " ms" << std::endl;
        std::cout << "  Standard deviation: " << stdDev << " ms" << std::endl;
        std::cout << "  Throughput: " << texts.size() * 1000.0 / mean << " texts/second" << std::endl;
        std::cout << "  Total tokens: " << totalTokens << std::endl;
        std::cout << "  Tokens per text: " << totalTokens / static_cast<double>(texts.size()) << std::endl;
    }
    
    // Benchmark batch encoding
    std::cout << "\nBatch encoding benchmark:" << std::endl;
    {
        std::vector<double> times;
        
        for (int i = 0; i < iterations; i++) {
            auto start = high_resolution_clock::now();
            
            for (const auto& batch : batches) {
                tokenizer->encodeBatch(batch);
            }
            
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start);
            times.push_back(duration.count());
        }
        
        // Calculate statistics
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min = *std::min_element(times.begin(), times.end());
        double max = *std::max_element(times.begin(), times.end());
        
        double stdDev = 0.0;
        for (double time : times) {
            stdDev += (time - mean) * (time - mean);
        }
        stdDev = std::sqrt(stdDev / times.size());
        
        std::cout << "  Average time: " << mean << " ms" << std::endl;
        std::cout << "  Min time: " << min << " ms" << std::endl;
        std::cout << "  Max time: " << max << " ms" << std::endl;
        std::cout << "  Standard deviation: " << stdDev << " ms" << std::endl;
        std::cout << "  Throughput: " << texts.size() * 1000.0 / mean << " texts/second" << std::endl;
        std::cout << "  Batch size: " << batchSize << std::endl;
    }
    
    // Calculate total characters
    size_t totalChars = 0;
    for (const auto& text : texts) {
        totalChars += text.length();
    }
    
    std::cout << "\nOverall statistics:" << std::endl;
    std::cout << "  Total texts: " << texts.size() << std::endl;
    std::cout << "  Total characters: " << totalChars << std::endl;
    std::cout << "  Average text length: " << totalChars / static_cast<double>(texts.size()) << " characters" << std::endl;
}

// Function to compare with other tokenizers (if available)
void compareWithOtherTokenizers(const std::vector<std::string>& texts, 
                               const std::string& tokenizerType,
                               const std::string& vocabPath,
                               const std::string& mergesPath) {
    // Create a temporary Python script to run the comparison
    std::string tempScript = cv::tempfile(".py");
    std::ofstream scriptFile(tempScript);
    
    scriptFile << "import sys\n";
    scriptFile << "import time\n";
    scriptFile << "import json\n";
    
    // Try to import different tokenizer libraries
    scriptFile << "has_transformers = False\n";
    scriptFile << "has_tiktoken = False\n";
    scriptFile << "try:\n";
    scriptFile << "    from transformers import AutoTokenizer\n";
    scriptFile << "    has_transformers = True\n";
    scriptFile << "except ImportError:\n";
    scriptFile << "    pass\n";
    scriptFile << "try:\n";
    scriptFile << "    import tiktoken\n";
    scriptFile << "    has_tiktoken = True\n";
    scriptFile << "except ImportError:\n";
    scriptFile << "    pass\n";
    
    // Load test texts
    scriptFile << "texts = " << std::accumulate(
        texts.begin(), texts.end(), std::string("["),
        [](const std::string& a, const std::string& b) {
            return a + (a.back() == '[' ? "" : ", ") + "\"" + b + "\"";
        }
    ) << "]\n";
    
    // Benchmark function
    scriptFile << "def benchmark_tokenizer(name, tokenizer, func, iterations=5):\n";
    scriptFile << "    results = {'name': name}\n";
    scriptFile << "    # Single text encoding\n";
    scriptFile << "    times = []\n";
    scriptFile << "    total_tokens = 0\n";
    scriptFile << "    for i in range(iterations):\n";
    scriptFile << "        start = time.time()\n";
    scriptFile << "        for text in texts:\n";
    scriptFile << "            tokens = func(text)\n";
    scriptFile << "            if i == 0:\n";
    scriptFile << "                if isinstance(tokens, list):\n";
    scriptFile << "                    total_tokens += len(tokens)\n";
    scriptFile << "                else:\n";
    scriptFile << "                    total_tokens += len(tokens.input_ids[0])\n";
    scriptFile << "        end = time.time()\n";
    scriptFile << "        times.append((end - start) * 1000)  # Convert to ms\n";
    scriptFile << "    results['single_encode_avg_ms'] = sum(times) / len(times)\n";
    scriptFile << "    results['single_encode_min_ms'] = min(times)\n";
    scriptFile << "    results['single_encode_max_ms'] = max(times)\n";
    scriptFile << "    results['total_tokens'] = total_tokens\n";
    scriptFile << "    results['tokens_per_text'] = total_tokens / len(texts)\n";
    scriptFile << "    return results\n";
    
    // Run benchmarks for different tokenizers
    scriptFile << "results = []\n";
    
    // Hugging Face Transformers
    scriptFile << "if has_transformers:\n";
    scriptFile << "    try:\n";
    if (tokenizerType == "bpe") {
        scriptFile << "        model_name = 'gpt2'  # BPE tokenizer\n";
    } else {
        scriptFile << "        model_name = 'bert-base-uncased'  # WordPiece tokenizer\n";
    }
    scriptFile << "        tokenizer = AutoTokenizer.from_pretrained(model_name)\n";
    scriptFile << "        results.append(benchmark_tokenizer('Transformers: ' + model_name, tokenizer, tokenizer))\n";
    scriptFile << "    except Exception as e:\n";
    scriptFile << "        print(f'Error with Transformers: {e}')\n";
    
    // tiktoken
    scriptFile << "if has_tiktoken and '" << tokenizerType << "' == 'bpe':\n";
    scriptFile << "    try:\n";
    scriptFile << "        enc = tiktoken.get_encoding('cl100k_base')\n";
    scriptFile << "        results.append(benchmark_tokenizer('tiktoken: cl100k_base', enc, enc.encode))\n";
    scriptFile << "    except Exception as e:\n";
    scriptFile << "        print(f'Error with tiktoken: {e}')\n";
    
    // Print results
    scriptFile << "print(json.dumps(results))\n";
    
    scriptFile.close();
    
    // Run the Python script and capture output
    std::string command = "python " + tempScript;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error executing Python script for comparison" << std::endl;
        return;
    }
    
    char buffer[4096];
    std::string result = "";
    
    while (!feof(pipe)) {
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
    }
    
    pclose(pipe);
    
    // Remove the temporary script
    std::remove(tempScript.c_str());
    
    // Parse and display results
    try {
        std::cout << "\nComparison with other tokenizers:" << std::endl;
        
        // Simple JSON parsing for the benchmark results
        // In a real implementation, use a proper JSON parser
        size_t start = result.find('[');
        size_t end = result.rfind(']');
        
        if (start != std::string::npos && end != std::string::npos && start < end) {
            std::string jsonArray = result.substr(start, end - start + 1);
            
            // Very simple parsing - in a real implementation, use a proper JSON parser
            size_t pos = 0;
            while ((pos = jsonArray.find("{", pos)) != std::string::npos) {
                size_t endPos = jsonArray.find("}", pos);
                if (endPos == std::string::npos) break;
                
                std::string jsonObj = jsonArray.substr(pos, endPos - pos + 1);
                
                // Extract name
                size_t namePos = jsonObj.find("\"name\"");
                if (namePos != std::string::npos) {
                    size_t nameStart = jsonObj.find(":", namePos) + 1;
                    size_t nameEnd = jsonObj.find(",", nameStart);
                    if (nameEnd == std::string::npos) nameEnd = jsonObj.find("}", nameStart);
                    
                    std::string name = jsonObj.substr(nameStart, nameEnd - nameStart);
                    name.erase(std::remove(name.begin(), name.end(), '"'), name.end());
                    name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
                    
                    // Extract average time
                    size_t timePos = jsonObj.find("\"single_encode_avg_ms\"");
                    double avgTime = 0.0;
                    if (timePos != std::string::npos) {
                        size_t timeStart = jsonObj.find(":", timePos) + 1;
                        size_t timeEnd = jsonObj.find(",", timeStart);
                        if (timeEnd == std::string::npos) timeEnd = jsonObj.find("}", timeStart);
                        
                        avgTime = std::stod(jsonObj.substr(timeStart, timeEnd - timeStart));
                    }
                    
                    // Extract total tokens
                    size_t tokensPos = jsonObj.find("\"total_tokens\"");
                    int totalTokens = 0;
                    if (tokensPos != std::string::npos) {
                        size_t tokensStart = jsonObj.find(":", tokensPos) + 1;
                        size_t tokensEnd = jsonObj.find(",", tokensStart);
                        if (tokensEnd == std::string::npos) tokensEnd = jsonObj.find("}", tokensStart);
                        
                        totalTokens = std::stoi(jsonObj.substr(tokensStart, tokensEnd - tokensStart));
                    }
                    
                    std::cout << "  " << name << ":" << std::endl;
                    std::cout << "    Average time: " << avgTime << " ms" << std::endl;
                    std::cout << "    Total tokens: " << totalTokens << std::endl;
                    std::cout << "    Throughput: " << texts.size() * 1000.0 / avgTime << " texts/second" << std::endl;
                }
                
                pos = endPos + 1;
            }
        } else {
            std::cout << "  No comparison results available" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing comparison results: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV DNN Tokenizer Benchmark");
    
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    
    // Check if we have tokenizer type and vocab
    Ptr<Tokenizer> tokenizer;
    
    if (parser.has("@tokenizer") && parser.has("@vocab")) {
        std::string tokenizerType = parser.get<std::string>("@tokenizer");
        std::string vocabPath = parser.get<std::string>("@vocab");
        
        if (tokenizerType == "bpe") {
            // Check for merges file
            if (!parser.has("@merges")) {
                std::cerr << "Merges file must be specified for BPE tokenizer" << std::endl;
                parser.printMessage();
                return -1;
            }
            
            std::string mergesPath = parser.get<std::string>("@merges");
            std::cout << "Creating BPE tokenizer with vocab: " << vocabPath << " and merges: " << mergesPath << std::endl;
            
            try {
                tokenizer = createBPETokenizer(vocabPath, mergesPath);
            } catch (const cv::Exception& e) {
                std::cerr << "Error creating BPE tokenizer: " << e.what() << std::endl;
                return -1;
            }
        } else if (tokenizerType == "wordpiece") {
            std::cout << "Creating WordPiece tokenizer with vocab: " << vocabPath << std::endl;
            
            try {
                tokenizer = createWordPieceTokenizer(vocabPath);
            } catch (const cv::Exception& e) {
                std::cerr << "Error creating WordPiece tokenizer: " << e.what() << std::endl;
                return -1;
            }
        } else {
            std::cerr << "Unknown tokenizer type: " << tokenizerType << std::endl;
            parser.printMessage();
            return -1;
        }
    } else {
        std::cerr << "Tokenizer type and vocabulary file must be specified" << std::endl;
        parser.printMessage();
        return -1;
    }
    
    // Get benchmark parameters
    int iterations = parser.get<int>("iterations");
    int batchSize = parser.get<int>("batch_size");
    
    // Load or generate test texts
    std::vector<std::string> texts;
    
    if (parser.has("corpus")) {
        std::string corpusPath = parser.get<std::string>("corpus");
        std::cout << "Loading corpus from: " << corpusPath << std::endl;
        texts = readCorpus(corpusPath);
    } else {
        std::cout << "Using generated sample texts" << std::endl;
        texts = generateSampleTexts(1000);
    }
    
    if (texts.empty()) {
        std::cerr << "No texts available for benchmarking" << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << texts.size() << " texts for benchmarking" << std::endl;
    
    // Run benchmark
    benchmarkTokenizer(tokenizer, texts, iterations, batchSize);
    
    // Compare with other tokenizers if requested
    if (parser.has("compare")) {
        std::string tokenizerType = parser.get<std::string>("@tokenizer");
        std::string vocabPath = parser.get<std::string>("@vocab");
        std::string mergesPath = parser.has("@merges") ? parser.get<std::string>("@merges") : "";
        
        compareWithOtherTokenizers(texts, tokenizerType, vocabPath, mergesPath);
    }
    
    return 0;
}