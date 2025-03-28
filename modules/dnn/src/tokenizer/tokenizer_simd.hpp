#ifndef OPENCV_DNN_TOKENIZER_SIMD_HPP
#define OPENCV_DNN_TOKENIZER_SIMD_HPP

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <unordered_map>

namespace cv {
namespace dnn {

/**
 * @brief SIMD-accelerated operations for tokenizers
 * 
 * This class provides SIMD-optimized implementations of common tokenizer operations
 * to improve performance, especially for large texts or batch processing.
 */
class CV_EXPORTS_W TokenizerSIMD {
public:
    /**
     * @brief Check if SIMD optimizations are available
     * @return True if SIMD optimizations are available
     */
    CV_WRAP static bool isAvailable();
    
    /**
     * @brief Fast search for a substring in a text using SIMD
     * @param text Text to search in
     * @param pattern Pattern to search for
     * @return Vector of positions where the pattern was found
     */
    CV_WRAP static std::vector<size_t> findPattern(const std::string& text, const std::string& pattern);
    
    /**
     * @brief Fast tokenization by splitting on delimiters using SIMD
     * @param text Text to split
     * @param delimiters String of delimiter characters
     * @return Vector of tokens
     */
    CV_WRAP static std::vector<std::string> splitFast(const std::string& text, const std::string& delimiters);
    
    /**
     * @brief Fast batch encoding of multiple texts using SIMD and multithreading
     * @param texts Vector of texts to encode
     * @param token_to_id Token to ID mapping
     * @param regex_pattern Regular expression pattern for tokenization
     * @param num_threads Number of threads to use (default: 0, which uses all available cores)
     * @return Vector of vectors of token IDs
     */
    CV_WRAP static std::vector<std::vector<int>> batchEncodeFast(
        const std::vector<std::string>& texts,
        const std::unordered_map<std::string, int>& token_to_id,
        const std::string& regex_pattern,
        int num_threads = 0);
    
    /**
     * @brief Fast vocabulary lookup using SIMD
     * @param tokens Vector of tokens to look up
     * @param token_to_id Token to ID mapping
     * @param unk_id Unknown token ID
     * @return Vector of token IDs
     */
    CV_WRAP static std::vector<int> vocabularyLookupFast(
        const std::vector<std::string>& tokens,
        const std::unordered_map<std::string, int>& token_to_id,
        int unk_id);
    
    /**
     * @brief SIMD-accelerated byte pair encoding
     * @param token Raw token to encode
     * @param merges Map of token pairs to their merge priorities
     * @return Encoded tokens
     */
    CV_WRAP static std::vector<std::string> bytePairEncodeFast(
        const std::string& token,
        const std::unordered_map<std::pair<std::string, std::string>, int, PairHash>& merges);
};

}} // namespace cv::dnn

#endif // OPENCV_DNN_TOKENIZER_SIMD_HPP