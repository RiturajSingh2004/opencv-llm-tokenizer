/**
 * @file tokenizer_simd.cpp
 * @brief SIMD-accelerated tokenization routines
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements SIMD-optimized routines for text tokenization
 * to improve processing performance on modern CPUs. It includes optimized
 * string search, pattern matching, and batch processing operations.
 */

#include "precomp.hpp"
#include "tokenizer_simd.hpp"

#include <thread>
#include <algorithm>
#include <regex>
#include <atomic>
#include <mutex>

// Include SIMD headers based on architecture
#if CV_SSE4_2
#include <nmmintrin.h> // SSE4.2
#endif

#if CV_AVX2
#include <immintrin.h> // AVX2
#endif

namespace cv {
namespace dnn {

bool TokenizerSIMD::isAvailable() {
    // Check for SIMD support at runtime
    #if CV_SSE4_2
        return checkHardwareSupport(CV_CPU_SSE4_2);
    #elif CV_AVX2
        return checkHardwareSupport(CV_CPU_AVX2);
    #else
        return false;
    #endif
}

std::vector<size_t> TokenizerSIMD::findPattern(const std::string& text, const std::string& pattern) {
    std::vector<size_t> positions;
    
    if (pattern.empty() || text.empty() || pattern.length() > text.length()) {
        std::vector<std::string> TokenizerSIMD::splitFast(const std::string& text, const std::string& delimiters) {
    std::vector<std::string> tokens;
    
    if (text.empty()) {
        return tokens;
    }
    
    #if CV_SSE4_2
    if (checkHardwareSupport(CV_CPU_SSE4_2) && !delimiters.empty()) {
        const char* text_ptr = text.c_str();
        const size_t text_len = text.length();
        
        // Prepare delimiter comparison
        __m128i delim_mask = _mm_setzero_si128();
        for (char c : delimiters) {
            __m128i char_mask = _mm_set1_epi8(c);
            delim_mask = _mm_or_si128(delim_mask, char_mask);
        }
        
        size_t start = 0;
        while (start < text_len) {
            // Skip leading delimiters
            while (start < text_len && delimiters.find(text[start]) != std::string::npos) {
                start++;
            }
            
            if (start >= text_len) {
                break;
            }
            
            // Find the end of the token using SIMD
            size_t end = start;
            while (end < text_len) {
                // Process 16 bytes at a time with SSE
                if (end + 16 <= text_len) {
                    __m128i text_chunk = _mm_loadu_si128((__m128i*)(text_ptr + end));
                    __m128i result = _mm_cmpestrm(delim_mask, delimiters.length(), text_chunk, 16,
                                                 _SIDD_CMP_EQUAL_ANY | _SIDD_POSITIVE_POLARITY |
                                                 _SIDD_LEAST_SIGNIFICANT | _SIDD_BIT_MASK);
                    
                    int mask = _mm_extract_epi32(result, 0);
                    if (mask != 0) {
                        // Found a delimiter, get its position
                        end += __builtin_ctz(mask);
                        break;
                    }
                    
                    end += 16;
                } else {
                    // Process remaining bytes one by one
                    if (delimiters.find(text[end]) != std::string::npos) {
                        break;
                    }
                    end++;
                }
            }
            
            // Add token to the result
            if (end > start) {
                tokens.push_back(text.substr(start, end - start));
            }
            
            start = end + 1;
        }
        
        return tokens;
    }
    #endif
    
    // Fallback to standard algorithm
    size_t start = 0;
    size_t end = 0;
    
    while ((end = text.find_first_of(delimiters, start)) != std::string::npos) {
        if (end > start) {
            tokens.push_back(text.substr(start, end - start));
        }
        start = end + 1;
    }
    
    if (start < text.length()) {
        tokens.push_back(text.substr(start));
    }
    
    return tokens;
}

std::vector<std::vector<int>> TokenizerSIMD::batchEncodeFast(
    const std::vector<std::string>& texts,
    const std::unordered_map<std::string, int>& token_to_id,
    const std::string& regex_pattern,
    int num_threads) {
    
    std::vector<std::vector<int>> result(texts.size());
    
    // Determine number of threads to use
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 4;  // Default to 4 if we can't detect
        }
    }
    
    // Limit threads to the number of texts
    num_threads = std::min(num_threads, static_cast<int>(texts.size()));
    
    if (num_threads <= 1 || texts.size() < 2) {
        // Single-threaded processing
        std::regex pattern(regex_pattern);
        
        for (size_t i = 0; i < texts.size(); ++i) {
            const std::string& text = texts[i];
            std::vector<int> tokens;
            
            // Split text using regex
            std::sregex_iterator it(text.begin(), text.end(), pattern);
            std::sregex_iterator end;
            
            for (; it != end; ++it) {
                std::string piece = it->str();
                auto token_it = token_to_id.find(piece);
                if (token_it != token_to_id.end()) {
                    tokens.push_back(token_it->second);
                } else {
                    // Handle unknown tokens - this would depend on the specific tokenizer implementation
                    // For now, we'll just skip them
                }
            }
            
            result[i] = std::move(tokens);
        }
    } else {
        // Multi-threaded processing
        std::vector<std::thread> threads;
        std::mutex result_mutex;  // To protect result vector
        
        // Process texts in chunks
        int chunk_size = (texts.size() + num_threads - 1) / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            threads.push_back(std::thread([&, t]() {
                // Compute start and end indices for this thread
                size_t start_idx = t * chunk_size;
                size_t end_idx = std::min(start_idx + chunk_size, texts.size());
                
                // Compile regex once per thread
                std::regex pattern(regex_pattern);
                
                // Process assigned texts
                for (size_t i = start_idx; i < end_idx; ++i) {
                    const std::string& text = texts[i];
                    std::vector<int> tokens;
                    
                    // Split text using regex
                    std::sregex_iterator it(text.begin(), text.end(), pattern);
                    std::sregex_iterator end;
                    
                    for (; it != end; ++it) {
                        std::string piece = it->str();
                        auto token_it = token_to_id.find(piece);
                        if (token_it != token_to_id.end()) {
                            tokens.push_back(token_it->second);
                        }
                        // Handle unknown tokens as appropriate
                    }
                    
                    // Store result
                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        result[i] = std::move(tokens);
                    }
                }
            }));
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    return result;
}

std::vector<int> TokenizerSIMD::vocabularyLookupFast(
    const std::vector<std::string>& tokens,
    const std::unordered_map<std::string, int>& token_to_id,
    int unk_id) {
    
    std::vector<int> ids(tokens.size());
    
    #if CV_AVX2
    if (checkHardwareSupport(CV_CPU_AVX2) && tokens.size() >= 8) {
        // This is a conceptual implementation since direct string lookup using SIMD
        // is complex - in a real implementation we'd need more sophisticated approaches
        // like perfect hashing or bloom filters optimized for SIMD
        
        // For now, we'll use SIMD to parallelize the conversion of multiple tokens
        // by processing them in batches of 8
        
        // Process tokens in chunks of 8
        size_t i = 0;
        for (; i + 8 <= tokens.size(); i += 8) {
            // We'd use gather instructions here in a real implementation
            // but for demonstration, we'll just parallelize the lookups
            
            // In a sophisticated implementation, we'd use AVX2 gather instructions
            // to load multiple hash table entries in parallel
            
            for (size_t j = 0; j < 8; ++j) {
                auto it = token_to_id.find(tokens[i + j]);
                ids[i + j] = (it != token_to_id.end()) ? it->second : unk_id;
            }
        }
        
        // Process remaining tokens
        for (; i < tokens.size(); ++i) {
            auto it = token_to_id.find(tokens[i]);
            ids[i] = (it != token_to_id.end()) ? it->second : unk_id;
        }
        
        return ids;
    }
    #endif
    
    // Fallback implementation
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = token_to_id.find(tokens[i]);
        ids[i] = (it != token_to_id.end()) ? it->second : unk_id;
    }
    
    return ids;
}

std::vector<std::string> TokenizerSIMD::bytePairEncodeFast(
    const std::string& token,
    const std::unordered_map<std::pair<std::string, std::string>, int, PairHash>& merges) {
    
    if (token.empty()) {
        return {};
    }
    
    // Initialize with characters
    std::vector<std::string> parts;
    for (char c : token) {
        parts.push_back(std::string(1, c));
    }
    
    if (parts.size() <= 1) {
        return parts;
    }
    
    // SIMD can be used here to speed up the merging process
    // but the algorithm itself remains complex due to the dictionary lookups
    
    // For the BPE algorithm, we need to find the best merge at each step
    // which involves checking each adjacent pair of tokens
    
    bool changes = true;
    while (changes && parts.size() > 1) {
        changes = false;
        
        // Find the best pair to merge
        int best_idx = -1;
        int best_priority = std::numeric_limits<int>::max();
        
        // This loop is where SIMD could help, but it's complex due to the merge dictionary
        for (size_t i = 0; i < parts.size() - 1; ++i) {
            auto pair = std::make_pair(parts[i], parts[i + 1]);
            auto it = merges.find(pair);
            
            if (it != merges.end() && it->second < best_priority) {
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


    }
    
    #if CV_SSE4_2
    if (checkHardwareSupport(CV_CPU_SSE4_2)) {
        // SSE4.2 provides string processing instructions
        const char* text_ptr = text.c_str();
        const char* pattern_ptr = pattern.c_str();
        const size_t text_len = text.length();
        const size_t pattern_len = pattern.length();
        
        // For very small patterns, use SSE4.2's string comparison instructions
        if (pattern_len <= 16) {
            for (size_t i = 0; i <= text_len - pattern_len; ++i) {
                // _SIDD_CMP_EQUAL_ORDERED: compare for equality with the specific order
                // _SIDD_POSITIVE_POLARITY: match when equal
                // _SIDD_LEAST_SIGNIFICANT: return index of first match
                // _SIDD_BIT_MASK: return a bit mask of the matches
                
                int res;
                if (pattern_len <= 8) {
                    __m128i pattern_reg = _mm_loadu_si128((__m128i*)pattern_ptr);
                    __m128i text_reg = _mm_loadu_si128((__m128i*)(text_ptr + i));
                    
                    res = _mm_cmpestri(pattern_reg, pattern_len, text_reg, pattern_len,
                                      _SIDD_CMP_EQUAL_ORDERED | _SIDD_POSITIVE_POLARITY | 
                                      _SIDD_LEAST_SIGNIFICANT | _SIDD_BIT_MASK);
                    
                    if (res == 0) {  // Match found at position 0
                        positions.push_back(i);
                    }
                } else {
                    // For longer patterns, we need to compare in chunks
                    bool match = true;
                    for (size_t j = 0; j < pattern_len; j += 16) {
                        size_t chunk_size = std::min(size_t(16), pattern_len - j);
                        __m128i pattern_chunk = _mm_loadu_si128((__m128i*)(pattern_ptr + j));
                        __m128i text_chunk = _mm_loadu_si128((__m128i*)(text_ptr + i + j));
                        
                        res = _mm_cmpestri(pattern_chunk, chunk_size, text_chunk, chunk_size,
                                          _SIDD_CMP_EQUAL_ORDERED | _SIDD_POSITIVE_POLARITY | 
                                          _SIDD_LEAST_SIGNIFICANT | _SIDD_BIT_MASK);
                        
                        if (res != 0) {  // No match in this chunk
                            match = false;
                            break;
                        }
                    }
                    
                    if (match) {
                        positions.push_back(i);
                    }
                }