/**
 * @file tiktoken_tokenizer.cpp
 * @brief Implementation of TikToken tokenizer
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements the TikToken tokenizer used by OpenAI models like GPT.
 * It provides efficient byte-level BPE tokenization with support for various 
 * encoding schemes like cl100k_base, p50k_base, etc.
 */

#include "precomp.hpp"
#include "tiktoken_tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <locale>
#include <codecvt>
#include <thread>
#include <base64.h> // For base64 encoding/decoding

namespace cv {
namespace dnn {

// Standard regex patterns used by TikToken
static const std::string GPT2_PATTERN = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
static const std::string CL100K_PATTERN = R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s)";

// Predefined encoding mappings
struct EncodingInfo {
    std::string pat_str;
    std::string tiktoken_file;
    std::map<std::string, int> special_tokens;
};

static const std::map<std::string, EncodingInfo> ENCODING_INFOS = {
    {"cl100k_base", {
        CL100K_PATTERN,
        "cl100k_base.tiktoken",
        {{"<|endoftext|>", 100257}, {"<|fim_prefix|>", 100258}, {"<|fim_middle|>", 100259}, 
         {"<|fim_suffix|>", 100260}, {"<|endofprompt|>", 100276}}
    }},
    {"p50k_base", {
        GPT2_PATTERN,
        "p50k_base.tiktoken",
        {{"<|endoftext|>", 50256}}
    }},
    {"p50k_edit", {
        GPT2_PATTERN,
        "p50k_base.tiktoken",
        {{"<|endoftext|>", 50256}, {"<|fim_prefix|>", 50281}, {"<|fim_middle|>", 50282}, 
         {"<|fim_suffix|>", 50283}}
    }},
    {"r50k_base", {
        GPT2_PATTERN,
        "r50k_base.tiktoken",
        {{"<|endoftext|>", 50256}}
    }},
    {"gpt2", {
        GPT2_PATTERN,
        "gpt2.tiktoken",
        {{"<|endoftext|>", 50256}}
    }}
};

TikTokenTokenizer::TikTokenTokenizer(
    const std::string& encoding_name,
    const std::string& bpe_ranks_path,
    const std::map<std::string, int>& special_tokens)
    : encoding_name_(encoding_name) {
    
    // Load predefined encoding if available
    auto it = ENCODING_INFOS.find(encoding_name);
    if (it != ENCODING_INFOS.end()) {
        pat_str_ = it->second.pat_str;
        
        if (!bpe_ranks_path.empty()) {
            // Use custom BPE ranks if provided
            load_bpe_ranks(bpe_ranks_path);
        } else {
            // Try to find predefined tiktoken file
            std::string tiktoken_file = findTikTokenFile(it->second.tiktoken_file);
            if (!tiktoken_file.empty()) {
                load_bpe_ranks(tiktoken_file);
            } else {
                CV_Error(Error::StsError, "Could not find predefined tiktoken file: " + it->second.tiktoken_file);
            }
        }
        
        // Merge predefined special tokens with custom ones
        special_tokens_ = it->second.special_tokens;
        for (const auto& pair : special_tokens) {
            special_tokens_[pair.first] = pair.second;
            special_token_ids_.insert(pair.second);
        }
    } else if (!bpe_ranks_path.empty()) {
        // Custom encoding with provided BPE ranks
        pat_str_ = GPT2_PATTERN;  // Default to GPT2 pattern
        load_bpe_ranks(bpe_ranks_path);
        
        // Use provided special tokens
        special_tokens_ = special_tokens;
        for (const auto& pair : special_tokens) {
            special_token_ids_.insert(pair.second);
        }
    } else {
        CV_Error(Error::StsError, 
            "Unknown encoding name: " + encoding_name + ". "
            "Must provide a valid encoding name or a custom BPE ranks file.");
    }
    
    // Generate token_byte_values_ for decoding
    for (const auto& pair : bpe_ranks_) {
        token_byte_values_[pair.second] = pair.first;
    }
    
    // Add special tokens to token_byte_values_
    for (const auto& pair : special_tokens_) {
        token_byte_values_[pair.second] = std::vector<uint8_t>(pair.first.begin(), pair.first.end());
    }
}

std::string TikTokenTokenizer::findTikTokenFile(const std::string& filename) {
    // Try common locations where tiktoken files might be stored
    
    // 1. Look in current directory
    if (std::ifstream(filename).good()) {
        return filename;
    }
    
    // 2. Look in OpenCV data directory
    std::string opencv_data = cv::utils::getOpenCVDataPath();
    std::string path = opencv_data + "/tiktoken/" + filename;
    if (std::ifstream(path).good()) {
        return path;
    }
    
    // 3. Try to find in a predefined local cache 
    // (similar to how tiktoken uses ~/.tiktoken)
    const char* home_dir = getenv("HOME");
    if (home_dir) {
        path = std::string(home_dir) + "/.tiktoken/" + filename;
        if (std::ifstream(path).good()) {
            return path;
        }
    }
    
    // 4. Try to look in temp directory
    const char* temp_dir = getenv("TEMP");
    if (temp_dir) {
        path = std::string(temp_dir) + "/tiktoken/" + filename;
        if (std::ifstream(path).good()) {
            return path;
        }
    }
    
    // Not found
    return "";
}

void TikTokenTokenizer::load_bpe_ranks(const std::string& bpe_ranks_path) {
    std::ifstream file(bpe_ranks_path);
    if (!file.is_open()) {
        CV_Error(Error::StsError, "Could not open BPE ranks file: " + bpe_ranks_path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token_bytes_b64;
        int rank;
        
        if (iss >> token_bytes_b64 >> rank) {
            try {
                // Decode base64 to get token bytes
                std::vector<uint8_t> token_bytes = base64_decode(token_bytes_b64);
                bpe_ranks_[token_bytes] = rank;
            } catch (const std::exception& e) {
                CV_Error(Error::StsError, 
                        "Failed to decode base64 token in BPE ranks file: " + std::string(e.what()));
            }
        }
    }
    
    if (bpe_ranks_.empty()) {
        CV_Error(Error::StsError, "Failed to load BPE ranks or file is empty");
    }
}

std::vector<uint8_t> base64_decode(const std::string& encoded) {
    if (encoded.empty()) {
        return {};
    }
    
    // Base64 decoding table
    const std::string base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    // Lookup table for base64 chars
    std::vector<int> t(256, -1);
    for (size_t i = 0; i < 64; i++) {
        t[base64_chars[i]] = i;
    }
    
    // Calculate output size
    size_t in_len = encoded.size();
    size_t i = 0;
    size_t padding = 0;
    
    // Skip padding characters
    if (in_len > 0 && encoded[in_len - 1] == '=') {
        padding++;
        if (in_len > 1 && encoded[in_len - 2] == '=') {
            padding++;
        }
    }
    
    std::vector<uint8_t> result;
    result.reserve(3 * in_len / 4 - padding);
    
    uint32_t val = 0;
    int valb = -8;
    
    for (i = 0; i < in_len; i++) {
        unsigned char c = encoded[i];
        if (c == '=') break;  // End of base64 content
        
        int d = t[c];
        if (d == -1) continue;  // Skip invalid characters
        
        val = (val << 6) + d;
        valb += 6;
        
        if (valb >= 0) {
            result.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    
    return result;
}

std::vector<std::string> TikTokenTokenizer::split_by_pattern(const std::string& text) const {
    std::vector<std::string> result;
    std::regex pattern(pat_str_, std::regex::ECMAScript);
    
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        result.push_back(i->str());
    }
    
    return result;
}

std::vector<int> TikTokenTokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    
    // Check for special tokens first
    for (const auto& [token, id] : special_tokens_) {
        size_t pos = 0;
        while ((pos = text.find(token, pos)) != std::string::npos) {
            result.push_back(id);
            pos += token.length();
        }
    }
    
    // If special tokens found, we've already tokenized the text
    if (!result.empty()) {
        return result;
    }
    
    // Split text by regex pattern
    auto words = split_by_pattern(text);
    
    // Process each split part
    for (const auto& word : words) {
        // Convert to UTF-8 bytes
        std::vector<uint8_t> bytes(word.begin(), word.end());
        
        // Apply byte pair encoding
        auto token_ids = byte_pair_encode(bytes);
        
        // Add to result
        result.insert(result.end(), token_ids.begin(), token_ids.end());
    }
    
    return result;
}

std::vector<int> TikTokenTokenizer::byte_pair_encode(const std::vector<uint8_t>& piece) const {
    if (piece.empty()) {
        return {};
    }
    
    // For single byte pieces, just return the rank
    if (piece.size() == 1) {
        auto it = bpe_ranks_.find(piece);
        if (it != bpe_ranks_.end()) {
            return {it->second};
        }
        return {};
    }
    
    // Initialize with individual bytes
    std::vector<std::vector<uint8_t>> parts;
    for (uint8_t b : piece) {
        parts.push_back({b});
    }
    
    // Iteratively merge according to BPE ranks
    while (parts.size() > 1) {
        // Find the pair with the lowest rank
        int best_rank = INT_MAX;
        int best_idx = -1;
        
        for (size_t i = 0; i < parts.size() - 1; i++) {
            std::vector<uint8_t> pair;
            pair.insert(pair.end(), parts[i].begin(), parts[i].end());
            pair.insert(pair.end(), parts[i + 1].begin(), parts[i + 1].end());
            
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        
        // No more merges possible
        if (best_idx == -1) {
            break;
        }
        
        // Merge the pair with lowest rank
        std::vector<uint8_t> merged;
        merged.insert(merged.end(), parts[best_idx].begin(), parts[best_idx].end());
        merged.insert(merged.end(), parts[best_idx + 1].begin(), parts[best_idx + 1].end());
        
        // Replace the pair with the merged token
        parts[best_idx] = merged;
        parts.erase(parts.begin() + best_idx + 1);
    }
    
    // Convert parts to token IDs
    std::vector<int> result;
    for (const auto& part : parts) {
        auto it = bpe_ranks_.find(part);
        if (it != bpe_ranks_.end()) {
            result.push_back(it->second);
        }
    }
    
    return result;
}

std::vector<TokenInfo> TikTokenTokenizer::encodeWithInfo(const std::string& text) const {
    std::vector<TokenInfo> result;
    
    // Split text by regex pattern
    auto words = split_by_pattern(text);
    
    size_t pos = 0;
    for (const auto& word : words) {
        // Convert to UTF-8 bytes
        std::vector<uint8_t> bytes(word.begin(), word.end());
        
        // Apply byte pair encoding
        auto token_ids = byte_pair_encode(bytes);
        
        // Create token info for each token
        for (size_t i = 0; i < token_ids.size(); i++) {
            TokenInfo info;
            info.id = token_ids[i];
            
            // Get token bytes
            std::vector<uint8_t> token_bytes = token_byte_values_.at(token_ids[i]);
            info.text = std::string(token_bytes.begin(), token_bytes.end());
            
            // For first token in a word, start from word position
            // For subsequent tokens, start after previous token
            if (i == 0) {
                info.start = pos;
            } else {
                info.start = result.back().end;
            }
            
            // End position is approximate
            info.end = info.start + info.text.length();
            
            // Calculate score (1.0 for now)
            info.score = 1.0f;
            
            result.push_back(info);
        }
        
        // Update position for next word
        pos += word.length();
    }
    
    return result;
}

std::vector<std::vector<int>> TikTokenTokenizer::encodeBatch(const std::vector<std::string>& texts) const {
    // Determine number of threads to use
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4;  // Default if detection fails
    }
    
    // Limit to a reasonable number
    num_threads = std::min(num_threads, 16U);
    
    // For small batches, don't bother with threading
    if (texts.size() < 4) {
        std::vector<std::vector<int>> result;
        for (const auto& text : texts) {
            result.push_back(encode(text));
        }
        return result;
    }
    
    // Prepare result vector
    std::vector<std::vector<int>> result(texts.size());
    
    // Calculate items per thread
    size_t items_per_thread = (texts.size() + num_threads - 1) / num_threads;
    
    // Create and run threads
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; i++) {
        size_t start_idx = i * items_per_thread;
        size_t end_idx = std::min(start_idx + items_per_thread, texts.size());
        
        if (start_idx >= texts.size()) {
            break;
        }
        
        threads.push_back(std::thread([this, &texts, &result, start_idx, end_idx]() {
            for (size_t j = start_idx; j < end_idx; j++) {
                result[j] = encode(texts[j]);
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    return result;
}

std::string TikTokenTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    
    for (int token : tokens) {
        // Check if this is a special token
        if (special_token_ids_.find(token) != special_token_ids_.end()) {
            // Get special token string
            for (const auto& [text, id] : special_tokens_) {
                if (id == token) {
                    result += text;
                    break;
                }
            }
        } else {
            // Regular token, get its bytes
            auto it = token_byte_values_.find(token);
            if (it != token_byte_values_.end()) {
                result.append(it->second.begin(), it->second.end());
            }
        }
    }
    
    return result;
}

size_t TikTokenTokenizer::getVocabSize() const {
    return bpe_ranks_.size() + special_tokens_.size();
}

void TikTokenTokenizer::save(const std::string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    
    fs << "tokenizer_type" << "tiktoken";
    fs << "encoding_name" << encoding_name_;
    fs << "pat_str" << pat_str_;
    
    // Save bpe_ranks
    fs << "bpe_ranks" << "{";
    for (const auto& [bytes, rank] : bpe_ranks_) {
        // Convert bytes to base64 for storage
        std::string bytes_b64 = base64_encode(bytes);
        fs << bytes_b64 << rank;
    }
    fs << "}";
    
    // Save special tokens
    fs << "special_tokens" << "{";
    for (const auto& [token, id] : special_tokens_) {
        fs << token << id;
    }
    fs << "}";
}

std::string TikTokenTokenizer::getTokenText(int tokenId) const {
    // Check if this is a special token
    if (special_token_ids_.find(tokenId) != special_token_ids_.end()) {
        // Get special token string
        for (const auto& [text, id] : special_tokens_) {
            if (id == tokenId) {
                return text;
            }
        }
    }
    
    // Regular token, get its bytes
    auto it = token_byte_values_.find(tokenId);
    if (it != token_byte_values_.end()) {
        return std::string(it->second.begin(), it->second.end());
    }
    
    return "";
}

int TikTokenTokenizer::getTokenId(const std::string& tokenText) const {
    // Check if it's a special token
    auto it = special_tokens_.find(tokenText);
    if (it != special_tokens_.end()) {
        return it->second;
    }
    
    // Convert to bytes and check in bpe_ranks
    std::vector<uint8_t> bytes(tokenText.begin(), tokenText.end());
    auto bpe_it = bpe_ranks_.find(bytes);
    if (bpe_it != bpe_ranks_.end()) {
        return bpe_it->second;
    }
    
    return -1;
}

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    static const char* b64_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    
    size_t i = 0;
    unsigned char array_3[3];
    unsigned char array_4[4];
    
    for (uint8_t byte : bytes) {
        array_3[i++] = byte;
        if (i == 3) {
            array_4[0] = (array_3[0] & 0xfc) >> 2;
            array_4[1] = ((array_3[0] & 0x03) << 4) + ((array_3[1] & 0xf0) >> 4);
            array_4[2] = ((array_3[1] & 0x0f) << 2) + ((array_3[2] & 0xc0) >> 6);
            array_4[3] = array_3[2] & 0x3f;
            
            for (i = 0; i < 4; i++) {
                result += b64_table[array_4[i]];
            }
            i = 0;
        }
    }
    
    if (i) {
        for (size_t j = i; j < 3; j++) {
            array_3[j] = '\0';
        }
        
        array_4[0] = (array_3[0] & 0xfc) >> 2;
        array_4[1] = ((array_3[0] & 0x03) << 4) + ((array_3[1] & 0xf0) >> 4);
        array_4[2] = ((array_3[1] & 0x0f) << 2) + ((array_3[2] & 0xc0) >> 6);
        
        for (size_t j = 0; j < i + 1; j++) {
            result += b64_table[array_4[j]];
        }
        
        while (i++ < 3) {
            result += '=';
        }
    }
    
    return result;
}

Ptr<Tokenizer> createTikTokenTokenizer(
    const std::string& encoding_name,
    const std::string& bpe_ranks_path,
    const std::map<std::string, int>& special_tokens) {
    
    return makePtr<TikTokenTokenizer>(encoding_name, bpe_ranks_path, special_tokens);
}

}} // namespace cv::dnn