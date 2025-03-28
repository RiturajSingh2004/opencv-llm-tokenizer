/**
 * @file test_simd_tokenizer.cpp
 * @brief Tests for SIMD-optimized tokenization routines
 * @author Rituraj Singh
 * @date 2025
 *
 * This file contains unit tests for the SIMD-accelerated tokenization
 * functions, verifying performance improvements and result correctness
 * across various text patterns and token distributions.
 */

#include "test_precomp.hpp"
#include <opencv2/dnn/tokenizer.hpp>
#include "tokenizer/tokenizer_simd.hpp"

namespace opencv_test {
namespace {

// Test patterns for string search
const std::vector<std::pair<std::string, std::string>> SEARCH_PATTERNS = {
    {"hello world", "world"},
    {"abcdefghijklmnopqrstuvwxyz", "mnop"},
    {"tokenization is the process of splitting text into tokens", "process"},
    {"repetitive pattern: ababababababababab", "ababa"},
    {"unicode test: こんにちは世界", "世界"},
    {"special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?", "+=[]{}"}
};

// Test strings for splitting
const std::vector<std::pair<std::string, std::string>> SPLIT_TEXTS = {
    {"hello world test string", " "},
    {"comma,separated,values", ","},
    {"tabs\tare\talso\tdelimiters", "\t"},
    {"multiple   spaces   between   words", " "},
    {"mixed,delimiters with|various;characters", " ,|;"}
};

// Test data for batch processing
const std::vector<std::string> BATCH_TEXTS = {
    "First test text",
    "Second example for batch processing",
    "Another sample with different words",
    "Short text",
    "This is a longer text that contains more words and should result in more tokens when processed"
};

TEST(TokenizerSIMD, IsAvailable)
{
    // This test just checks if SIMD support detection works
    bool simd_available = cv::dnn::TokenizerSIMD::isAvailable();
    
    // We're not asserting the result because it depends on the CPU
    std::cout << "SIMD support " << (simd_available ? "is" : "is not") << " available on this CPU" << std::endl;
}

TEST(TokenizerSIMD, FindPattern)
{
    for (const auto& [text, pattern] : SEARCH_PATTERNS) {
        // Regular string find for comparison
        size_t std_pos = text.find(pattern);
        bool std_found = (std_pos != std::string::npos);
        
        // SIMD-accelerated find
        std::vector<size_t> simd_positions = cv::dnn::TokenizerSIMD::findPattern(text, pattern);
        bool simd_found = !simd_positions.empty();
        
        // Verify the results match
        EXPECT_EQ(std_found, simd_found) << "Failed on text: '" << text << "', pattern: '" << pattern << "'";
        
        if (std_found && simd_found) {
            EXPECT_EQ(std_pos, simd_positions[0]) << "Position mismatch for text: '" << text << "', pattern: '" << pattern << "'";
        }
    }
}

TEST(TokenizerSIMD, SplitFast)
{
    for (const auto& [text, delimiters] : SPLIT_TEXTS) {
        // Split using SIMD acceleration
        std::vector<std::string> simd_tokens = cv::dnn::TokenizerSIMD::splitFast(text, delimiters);
        
        // Manually split for comparison
        std::vector<std::string> expected_tokens;
        size_t start = 0;
        size_t end = 0;
        
        while ((end = text.find_first_of(delimiters, start)) != std::string::npos) {
            if (end > start) {
                expected_tokens.push_back(text.substr(start, end - start));
            }
            start = end + 1;
        }
        
        if (start < text.length()) {
            expected_tokens.push_back(text.substr(start));
        }
        
        // Verify the results match
        EXPECT_EQ(expected_tokens.size(), simd_tokens.size()) 
            << "Token count mismatch for text: '" << text << "', delimiters: '" << delimiters << "'";
        
        for (size_t i = 0; i < std::min(expected_tokens.size(), simd_tokens.size()); ++i) {
            EXPECT_EQ(expected_tokens[i], simd_tokens[i]) 
                << "Token mismatch at position " << i << " for text: '" << text << "', delimiters: '" << delimiters << "'";
        }
    }
}

TEST(TokenizerSIMD, BatchEncodeFast)
{
    // Create a simple token-to-id mapping for testing
    std::unordered_map<std::string, int> token_to_id;
    for (char c = 'a'; c <= 'z'; ++c) {
        std::string token(1, c);
        token_to_id[token] = c - 'a' + 1;
    }
    for (char c = 'A'; c <= 'Z'; ++c) {
        std::string token(1, c);
        token_to_id[token] = c - 'A' + 27;
    }
    // Add some common words
    token_to_id["the"] = 100;
    token_to_id["and"] = 101;
    token_to_id["for"] = 102;
    token_to_id["with"] = 103;
    token_to_id["is"] = 104;
    
    // Test regex pattern (split on whitespace)
    std::string regex_pattern = "\\s+";
    
    // Test batch encoding with different thread counts
    for (int thread_count : {1, 2, 4}) {
        std::vector<std::vector<int>> results = cv::dnn::TokenizerSIMD::batchEncodeFast(
            BATCH_TEXTS, token_to_id, regex_pattern, thread_count);
        
        // Verify we got results for each input text
        EXPECT_EQ(BATCH_TEXTS.size(), results.size());
        
        // Verify the results make sense (non-empty for non-empty inputs)
        for (size_t i = 0; i < results.size(); ++i) {
            if (!BATCH_TEXTS[i].empty()) {
                EXPECT_FALSE(results[i].empty()) << "Empty result for non-empty text at index " << i;
            }
        }
    }
}

TEST(TokenizerSIMD, VocabularyLookupFast)
{
    // Create a token-to-id mapping
    std::unordered_map<std::string, int> token_to_id;
    token_to_id["hello"] = 1;
    token_to_id["world"] = 2;
    token_to_id["test"] = 3;
    token_to_id["simd"] = 4;
    token_to_id["acceleration"] = 5;
    
    // Test tokens to look up
    std::vector<std::string> tokens = {"hello", "world", "unknown", "test", "simd", "another_unknown"};
    
    // Expected IDs (with unk_id = 0 for unknown tokens)
    std::vector<int> expected_ids = {1, 2, 0, 3, 4, 0};
    
    // Perform SIMD lookup
    std::vector<int> simd_ids = cv::dnn::TokenizerSIMD::vocabularyLookupFast(tokens, token_to_id, 0);
    
    // Verify the results
    ASSERT_EQ(expected_ids.size(), simd_ids.size());
    for (size_t i = 0; i < expected_ids.size(); ++i) {
        EXPECT_EQ(expected_ids[i], simd_ids[i]) << "Mismatch at position " << i;
    }
}

TEST(TokenizerSIMD, BytePairEncodeFast)
{
    // Create a simple BPE merges dictionary
    std::unordered_map<std::pair<std::string, std::string>, int, cv::dnn::PairHash> merges;
    merges[std::make_pair("h", "e")] = 1;
    merges[std::make_pair("he", "l")] = 2;
    merges[std::make_pair("hel", "l")] = 3;
    merges[std::make_pair("hell", "o")] = 4;
    merges[std::make_pair("w", "o")] = 5;
    merges[std::make_pair("wo", "r")] = 6;
    merges[std::make_pair("wor", "l")] = 7;
    merges[std::make_pair("worl", "d")] = 8;
    
    // Test BPE encoding
    std::string token = "hello";
    std::vector<std::string> expected_parts = {"hello"};
    std::vector<std::string> simd_parts = cv::dnn::TokenizerSIMD::bytePairEncodeFast(token, merges);
    
    ASSERT_EQ(expected_parts.size(), simd_parts.size());
    for (size_t i = 0; i < expected_parts.size(); ++i) {
        EXPECT_EQ(expected_parts[i], simd_parts[i]) << "Mismatch at position " << i;
    }
    
    // Test with token that doesn't merge completely
    token = "world!";
    expected_parts = {"world", "!"};
    simd_parts = cv::dnn::TokenizerSIMD::bytePairEncodeFast(token, merges);
    
    ASSERT_EQ(expected_parts.size(), simd_parts.size());
    for (size_t i = 0; i < expected_parts.size(); ++i) {
        EXPECT_EQ(expected_parts[i], simd_parts[i]) << "Mismatch at position " << i;
    }
}

TEST(TokenizerSIMD, PerformanceComparison)
{
    // This test compares the performance of SIMD-accelerated operations with regular operations
    // It's not a strict test but helps to verify that SIMD provides a speedup
    
    // Generate a large text for pattern search
    std::string large_text;
    large_text.reserve(1000000);
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    
    for (int i = 0; i < 100000; ++i) {
        large_text += alphabet[i % alphabet.size()];
    }
    
    // Pattern to search for
    std::string pattern = "WXYZ0123";
    
    // Measure standard string find
    int64 start = cv::getTickCount();
    size_t std_pos = large_text.find(pattern);
    int64 std_time = cv::getTickCount() - start;
    
    // Measure SIMD-accelerated find
    start = cv::getTickCount();
    std::vector<size_t> simd_positions = cv::dnn::TokenizerSIMD::findPattern(large_text, pattern);
    int64 simd_time = cv::getTickCount() - start;
    
    // We don't enforce timing requirements because SIMD availability varies,
    // but we log the results for information
    double std_time_ms = std_time * 1000.0 / cv::getTickFrequency();
    double simd_time_ms = simd_time * 1000.0 / cv::getTickFrequency();
    
    std::cout << "Pattern search performance:" << std::endl;
    std::cout << "  Standard: " << std_time_ms << " ms" << std::endl;
    std::cout << "  SIMD:     " << simd_time_ms << " ms" << std::endl;
    
    // Verify we found the same result
    if (std_pos != std::string::npos) {
        EXPECT_FALSE(simd_positions.empty());
        if (!simd_positions.empty()) {
            EXPECT_EQ(std_pos, simd_positions[0]);
        }
    }
}

}} // namespace