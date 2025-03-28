/**
 * @file bpe_tokenizer.cpp
 * @brief Implementation of Byte-Pair Encoding tokenizer
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements the Byte-Pair Encoding algorithm for tokenization 
 * as used in GPT and similar language models. It handles vocabulary and
 * merge rules loading, text encoding/decoding, and token management.
 */

#include "precomp.hpp"
#include "bpe_tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <queue>

namespace cv {
namespace dnn {

BPETokenizer::BPETokenizer(const std::string& vocab_file, const std::string& merges_file) 
    : max_token_length_(0) {
    loadVocabulary(vocab_file);
    loadMerges(merges_file);
}

void BPETokenizer::loadVocabulary(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        CV_Error(Error::StsError, "Could not open vocabulary file: " + vocab_file);
    }
    
    // Parse JSON format vocabulary
    auto parseJson = [&]() {
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        
        // Very simple JSON parser for "token": id format
        // For production code, consider using a proper JSON parser
        std::regex token_pattern("\"(.*?)\"\\s*:\\s*(\\d+)");
        std::smatch matches;
        std::string::const_iterator search_start(content.cbegin());
        
        while (std::regex_search(search_start, content.cend(), matches, token_pattern)) {
            std::string token = matches[1].str();
            int id = std::stoi(matches[2].str());
            
            // Handle escape sequences
            std::string unescaped;
            for (size_t i = 0; i < token.size(); i++) {
                if (token[i] == '\\' && i + 1 < token.size()) {
                    switch (token[i + 1]) {
                        case 'n': unescaped += '\n'; break;
                        case 'r': unescaped += '\r'; break;
                        case 't': unescaped += '\t'; break;
                        case '\\': unescaped += '\\'; break;
                        case '"': unescaped += '"'; break;
                        case 'u': // Unicode escape
                            if (i + 5 < token.size()) {
                                // Simply use the escaped sequence for now
                                // A proper implementation would convert \uXXXX to UTF-8
                                unescaped += token.substr(i, 6);
                                i += 5;
                            }
                            break;
                        default: unescaped += token[i + 1];
                    }
                    i++;
                } else {
                    unescaped += token[i];
                }
            }
            
            token_to_id_[unescaped] = id;
            id_to_token_[id] = unescaped;
            max_token_length_ = std::max(max_token_length_, unescaped.length());
            
            search_start = matches.suffix().first;
        }
    };
    
    parseJson();
    
    if (token_to_id_.empty()) {
        CV_Error(Error::StsError, "Failed to load vocabulary or vocabulary is empty");
    }
}

void BPETokenizer::loadMerges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        CV_Error(Error::StsError, "Could not open merges file: " + merges_file);
    }
    
    std::string line;
    int priority = 0;
    
    // Skip first line if it's a version header
    std::getline(file, line);
    if (line.find("#version") != std::string::npos) {
        // This is a version line, skip it
    } else {
        // Not a version line, process it
        std::istringstream iss(line);
        std::string first, second;
        if (iss >> first >> second) {
            merges_[std::make_pair(first, second)] = priority++;
        }
    }
    
    // Process remaining lines
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string first, second;
        if (iss >> first >> second) {
            merges_[std::make_pair(first, second)] = priority++;
        }
    }
    
    if (merges_.empty()) {
        CV_Error(Error::StsError, "Failed to load merges or merges file is empty");
    }
}

std::vector<std::string> BPETokenizer::tokenize(const std::string& text) const {
    // Simple pattern to split on whitespace and punctuation
    std::regex pattern(R"(\s+|[^\w\s]+)");
    
    std::vector<std::string> tokens;
    std::sregex_token_iterator iter(text.begin(), text.end(), pattern, -1);
    std::sregex_token_iterator end;
    
    for (; iter != end; ++iter) {
        if (!iter->str().empty()) {
            tokens.push_back(iter->str());
        }
    }
    
    return tokens;
}

std::vector<std::string> BPETokenizer::bpe_encode(const std::string& token) const {
    // Initialize with characters
    std::vector<std::string> parts;
    for (char c : token) {
        parts.push_back(std::string(1, c));
    }
    
    if (parts.empty()) {
        return parts;
    }
    
    // Iteratively merge according to BPE rules
    bool changes = true;
    while (changes && parts.size() > 1) {
        changes = false;
        
        // Find the best merge
        int best_idx = -1;
        int best_priority = INT_MAX;
        
        for (size_t i = 0; i < parts.size() - 1; i++) {
            auto pair = std::make_pair(parts[i], parts[i + 1]);
            auto it = merges_.find(pair);
            
            if (it != merges_.end() && it->second < best_priority) {
                best_idx = static_cast<int>(i);
                best_priority = it->second;
            }
        }
        
        // Apply the best merge
        if (best_idx >= 0) {
            std::string merged = parts[best_idx] + parts[best_idx + 1];
            parts[best_idx] = merged;
            parts.erase(parts.begin() + best_idx + 1);
            changes = true;
        }
    }
    
    return parts;
}

std::vector<int> BPETokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens) const {
    std::vector<int> ids;
    ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Handle unknown tokens
            // For simplicity, we use individual characters or a special <unk> token if available
            for (const auto& part : token) {
                std::string char_str(1, part);
                auto char_it = token_to_id_.find(char_str);
                if (char_it != token_to_id_.end()) {
                    ids.push_back(char_it->second);
                } else if (token_to_id_.find("<unk>") != token_to_id_.end()) {
                    ids.push_back(token_to_id_.at("<unk>"));
                }
                // else skip the character
            }
        }
    }
    
    return ids;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    
    // Split text into tokens
    auto word_tokens = tokenize(text);
    
    // Apply BPE encoding to each token
    for (const auto& word : word_tokens) {
        auto bpe_tokens = bpe_encode(word);
        auto ids = convert_tokens_to_ids(bpe_tokens);
        result.insert(result.end(), ids.begin(), ids.end());
    }
    
    return result;
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    
    for (int token : tokens) {
        auto it = id_to_token_.find(token);
        if (it != id_to_token_.end()) {
            result += it->second;
        } else {
            // Handle unknown token IDs
            result += "<unk>";
        }
    }
    
    return result;
}

size_t BPETokenizer::getVocabSize() const {
    return token_to_id_.size();
}

void BPETokenizer::save(const std::string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    
    fs << "tokenizer_type" << "bpe";
    
    // Save token_to_id map
    fs << "vocabulary" << "{";
    for (const auto& pair : token_to_id_) {
        fs << pair.first << pair.second;
    }
    fs << "}";
    
    // Save merges
    fs << "merges" << "{";
    for (const auto& pair : merges_) {
        fs << pair.first.first + " " + pair.first.second << pair.second;
    }
    fs << "}";
    
    // Save max token length
    fs << "max_token_length" << static_cast<int>(max_token_length_);
}