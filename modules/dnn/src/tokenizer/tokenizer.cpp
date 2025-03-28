/**
 * @file tokenizer.cpp
 * @brief Implementation of base tokenizer and common utilities
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements the base Tokenizer class methods and common utilities
 * shared across different tokenizer implementations. It includes standard
 * implementations for batch processing, token sequence manipulation, and
 * serialization.
 */

#include "precomp.hpp"
#include <opencv2/dnn/tokenizer.hpp>
#include "tokenizer/bpe_tokenizer.hpp"
#include "tokenizer/wordpiece_tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>

namespace cv {
namespace dnn {

std::vector<TokenInfo> Tokenizer::encodeWithInfo(const std::string& text) const {
    // Default implementation that can be overridden by derived classes for more efficient implementation
    std::vector<int> tokens = encode(text);
    std::vector<TokenInfo> result;
    result.reserve(tokens.size());
    
    // Basic implementation that doesn't track positions accurately
    for (size_t i = 0; i < tokens.size(); i++) {
        TokenInfo info;
        info.id = tokens[i];
        info.text = getTokenText(tokens[i]);
        info.start = -1;  // Position tracking not implemented in base class
        info.end = -1;
        info.score = 1.0f;
        result.push_back(info);
    }
    
    return result;
}

std::vector<std::vector<int>> Tokenizer::encodeBatch(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> result;
    result.reserve(texts.size());
    
    for (const auto& text : texts) {
        result.push_back(encode(text));
    }
    
    return result;
}

void Tokenizer::exportTo(const std::string& filename, const std::string& format) const {
    if (format == "huggingface") {
        // Create directory if it doesn't exist
        namespace fs = std::filesystem;
        fs::path dirPath(filename);
        if (!fs::exists(dirPath)) {
            fs::create_directories(dirPath);
        }
        
        // Export special tokens
        std::ofstream specialTokensFile(dirPath / "special_tokens_map.json");
        specialTokensFile << "{\n";
        specialTokensFile << "  \"unk_token\": \"[UNK]\"\n";
        specialTokensFile << "}\n";
        specialTokensFile.close();
        
        // Export vocabulary (simple implementation - would need customization per tokenizer type)
        std::ofstream vocabFile(dirPath / "vocab.json");
        vocabFile << "{\n";
        // Tokenizer-specific implementation would be needed here
        vocabFile << "}\n";
        vocabFile.close();
        
        // Export tokenizer config
        std::ofstream configFile(dirPath / "tokenizer_config.json");
        configFile << "{\n";
        configFile << "  \"model_type\": \"custom\",\n";
        configFile << "  \"name\": \"" << this->getType() << "\",\n";
        configFile << "  \"vocab_size\": " << this->getVocabSize() << "\n";
        configFile << "}\n";
        configFile.close();
    }
    else if (format == "tiktoken") {
        // Basic implementation for tiktoken format
        std::ofstream outFile(filename);
        // Tokenizer-specific implementation would be needed here
        outFile.close();
    }
    else {
        CV_Error(Error::StsBadArg, "Unsupported export format: " + format);
    }
}

std::string Tokenizer::getTokenText(int tokenId) const {
    CV_Error(Error::StsNotImplemented, "getTokenText is not implemented in the base class");
    return "";
}

int Tokenizer::getTokenId(const std::string& tokenText) const {
    CV_Error(Error::StsNotImplemented, "getTokenId is not implemented in the base class");
    return -1;
}

std::vector<int> Tokenizer::truncate(const std::vector<int>& tokens, 
                                    int maxLength, 
                                    const std::string& strategy) const {
    if (static_cast<int>(tokens.size()) <= maxLength) {
        return tokens;
    }
    
    std::vector<int> result;
    if (strategy == "left") {
        // Keep rightmost tokens
        result.assign(tokens.end() - maxLength, tokens.end());
    }
    else if (strategy == "middle") {
        // Keep tokens from both ends
        int leftSize = maxLength / 2;
        int rightSize = maxLength - leftSize;
        result.insert(result.end(), tokens.begin(), tokens.begin() + leftSize);
        result.insert(result.end(), tokens.end() - rightSize, tokens.end());
    }
    else {
        // Default to "right" - keep leftmost tokens
        result.assign(tokens.begin(), tokens.begin() + maxLength);
    }
    
    return result;
}

std::vector<int> Tokenizer::mergeTokenSequences(const std::vector<int>& tokens1,
                                              const std::vector<int>& tokens2,
                                              const std::vector<int>& specialTokens) const {
    std::vector<int> result;
    result.reserve(tokens1.size() + tokens2.size() + specialTokens.size());
    
    // Add first sequence
    result.insert(result.end(), tokens1.begin(), tokens1.end());
    
    // Add special tokens
    result.insert(result.end(), specialTokens.begin(), specialTokens.end());
    
    // Add second sequence
    result.insert(result.end(), tokens2.begin(), tokens2.end());
    
    return result;
}

Ptr<Tokenizer> Tokenizer::load(const std::string& filename) {
    FileStorage fs(filename, FileStorage::READ);
    
    if (!fs.isOpened()) {
        CV_Error(Error::StsError, "Could not open tokenizer file: " + filename);
    }
    
    std::string tokenizer_type;
    fs["tokenizer_type"] >> tokenizer_type;
    
    if (tokenizer_type == "bpe") {
        std::string vocab_file, merges_file;
        fs["vocab_file"] >> vocab_file;
        fs["merges_file"] >> merges_file;
        
        // Check if preprocessing config is available
        TokenizerPreprocessConfig config;
        if (fs["preprocess_flags"].isInt()) {
            int flags;
            fs["preprocess_flags"] >> flags;
            config.flags = flags;
        }
        
        return createBPETokenizer(vocab_file, merges_file, config);
    } 
    else if (tokenizer_type == "wordpiece") {
        std::string vocab_file;
        fs["vocab_file"] >> vocab_file;
        
        // Check if preprocessing config is available
        TokenizerPreprocessConfig config;
        if (fs["preprocess_flags"].isInt()) {
            int flags;
            fs["preprocess_flags"] >> flags;
            config.flags = flags;
        }
        
        return createWordPieceTokenizer(vocab_file, config);
    } 
    else {
        CV_Error(Error::StsError, "Unknown tokenizer type: " + tokenizer_type);
    }
}

Ptr<Tokenizer> Tokenizer::importFrom(const std::string& directory, const std::string& format) {
    namespace fs = std::filesystem;
    fs::path dirPath(directory);
    
    if (!fs::exists(dirPath)) {
        CV_Error(Error::StsError, "Directory does not exist: " + directory);
    }
    
    if (format == "huggingface") {
        // Check for HuggingFace tokenizer files
        if (fs::exists(dirPath / "tokenizer_config.json")) {
            // Parse tokenizer config
            std::ifstream configFile(dirPath / "tokenizer_config.json");
            std::string configStr((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());
            configFile.close();
            
            // Simple check for tokenizer type (a proper implementation would use JSON parsing)
            if (configStr.find("\"model_type\":") != std::string::npos) {
                bool isBPE = configStr.find("\"model_type\": \"gpt") != std::string::npos || 
                             configStr.find("\"model_type\":\"gpt") != std::string::npos;
                bool isWordPiece = configStr.find("\"model_type\": \"bert") != std::string::npos ||
                                   configStr.find("\"model_type\":\"bert") != std::string::npos;
                
                // Check for vocabulary files
                std::string vocabPath;
                if (fs::exists(dirPath / "vocab.json")) {
                    vocabPath = (dirPath / "vocab.json").string();
                }
                else if (fs::exists(dirPath / "vocab.txt")) {
                    vocabPath = (dirPath / "vocab.txt").string();
                }
                else {
                    CV_Error(Error::StsError, "Could not find vocabulary file in directory: " + directory);
                }
                
                // Create appropriate tokenizer
                if (isBPE && fs::exists(dirPath / "merges.txt")) {
                    return createBPETokenizer(vocabPath, (dirPath / "merges.txt").string());
                }
                else if (isWordPiece) {
                    return createWordPieceTokenizer(vocabPath);
                }
            }
        }
    }
    else if (format == "tiktoken") {
        // Check for tiktoken files
        if (fs::exists(dirPath / "tiktoken.bpe")) {
            // Create a vocabulary and merges file for BPE tokenizer
            // This would require parsing the tiktoken.bpe file format
            // For now, we'll just provide a skeleton implementation
            std::string tempVocabFile = cv::tempfile(".json");
            std::string tempMergesFile = cv::tempfile(".txt");
            
            // Parse tiktoken file and generate vocab and merges files
            // ...
            
            return createBPETokenizer(tempVocabFile, tempMergesFile);
        }
    }
    
    CV_Error(Error::StsError, "Could not import tokenizer from directory: " + directory + " with format: " + format);
}

Ptr<Tokenizer> createBPETokenizer(const std::string& vocab_file, 
                                 const std::string& merges_file,
                                 const TokenizerPreprocessConfig& config) {
    return makePtr<BPETokenizer>(vocab_file, merges_file, config);
}

Ptr<Tokenizer> createWordPieceTokenizer(const std::string& vocab_file,
                                       const TokenizerPreprocessConfig& config) {
    return makePtr<WordPieceTokenizer>(vocab_file, config);
}

Ptr<Tokenizer> loadTokenizerFromDirectory(const std::string& dir_path) {
    // Check if directory contains configuration files for different tokenizer types
    namespace fs = std::filesystem;
    
    // Look for BPE tokenizer files
    if (fs::exists(fs::path(dir_path) / "vocab.json") && 
        fs::exists(fs::path(dir_path) / "merges.txt")) {
        return createBPETokenizer(
            (fs::path(dir_path) / "vocab.json").string(),
            (fs::path(dir_path) / "merges.txt").string()
        );
    }
    
    // Look for WordPiece tokenizer files
    if (fs::exists(fs::path(dir_path) / "vocab.txt")) {
        return createWordPieceTokenizer(
            (fs::path(dir_path) / "vocab.txt").string()
        );
    }
    
    // Try to detect HuggingFace tokenizer
    if (fs::exists(fs::path(dir_path) / "tokenizer_config.json")) {
        return Tokenizer::importFrom(dir_path, "huggingface");
    }
    
    // Try to detect tiktoken
    if (fs::exists(fs::path(dir_path) / "tiktoken.bpe")) {
        return Tokenizer::importFrom(dir_path, "tiktoken");
    }
    
    CV_Error(Error::StsError, "Could not find valid tokenizer files in directory: " + dir_path);
}

std::map<std::string, double> benchmarkTokenizer(
    const Ptr<Tokenizer>& tokenizer,
    const std::vector<std::string>& texts,
    int iterations
) {
    std::map<std::string, double> results;
    
    // Measure single-text encoding performance
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            for (const auto& text : texts) {
                tokenizer->encode(text);
            }
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        results["encode_ms_per_text"] = duration.count() / static_cast<double>(iterations * texts.size());
    }
    
    // Measure batch encoding performance
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            tokenizer->encodeBatch(texts);
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        results["batch_encode_ms"] = duration.count() / static_cast<double>(iterations);
    }
    
    // Calculate total characters processed
    size_t totalChars = 0;
    for (const auto& text : texts) {
        totalChars += text.length();
    }
    results["total_chars"] = static_cast<double>(totalChars);
    results["chars_per_second"] = (totalChars * iterations * 1000.0) / (results["batch_encode_ms"] * iterations);
    
    // Measure memory usage
    size_t vocabSize = tokenizer->getVocabSize();
    results["vocab_size"] = static_cast<double>(vocabSize);
    
    return results;
}

}} // namespace cv::dnn