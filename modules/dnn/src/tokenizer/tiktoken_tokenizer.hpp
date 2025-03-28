#ifndef OPENCV_DNN_TIKTOKEN_TOKENIZER_HPP
#define OPENCV_DNN_TIKTOKEN_TOKENIZER_HPP

#include "tokenizer.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>

namespace cv {
namespace dnn {

/**
 * @brief Implementation of TikToken tokenizer
 * 
 * This tokenizer implements the TikToken encoding used by OpenAI models like GPT-3, GPT-4, etc.
 * It is based on the original implementation at https://github.com/openai/tiktoken
 * and supports efficient loading and encoding using byte-level operations.
 */
class CV_EXPORTS_W TikTokenTokenizer : public Tokenizer {
public:
    /**
     * @brief Constructor for TikToken tokenizer
     * @param encoding_name Name of the encoding to use (cl100k_base, p50k_base, r50k_base, etc.)
     * @param bpe_ranks_path Optional path to the BPE ranks file
     * @param special_tokens Optional special tokens mapping
     */
    CV_WRAP TikTokenTokenizer(
        const std::string& encoding_name,
        const std::string& bpe_ranks_path = "",
        const std::map<std::string, int>& special_tokens = {});
    
    /**
     * @brief Encode text into token IDs
     * @param text Input text to encode
     * @return Vector of token IDs
     */
    CV_WRAP std::vector<int> encode(const std::string& text) const override;
    
    /**
     * @brief Encode with detailed token information
     * @param text Input text to encode
     * @return Vector of token info structures
     */
    CV_WRAP std::vector<TokenInfo> encodeWithInfo(const std::string& text) const override;
    
    /**
     * @brief Batch encodes multiple strings into tokens
     * @param texts Vector of texts to encode
     * @return Vector of vectors of token IDs
     */
    CV_WRAP std::vector<std::vector<int>> encodeBatch(const std::vector<std::string>& texts) const override;
    
    /**
     * @brief Decode token IDs back to text
     * @param tokens Vector of token IDs
     * @return Decoded text
     */
    CV_WRAP std::string decode(const std::vector<int>& tokens) const override;
    
    /**
     * @brief Get vocabulary size
     * @return Number of tokens in vocabulary
     */
    CV_WRAP size_t getVocabSize() const override;
    
    /**
     * @brief Save tokenizer to a file
     * @param filename Path where to save tokenizer
     */
    CV_WRAP void save(const std::string& filename) const override;
    
    /**
     * @brief Get token text for a token ID
     * @param tokenId Token ID
     * @return Token text or empty string if not found
     */
    CV_WRAP std::string getTokenText(int tokenId) const override;
    
    /**
     * @brief Get token ID for a token text
     * @param tokenText Token text
     * @return Token ID or -1 if not found
     */
    CV_WRAP int getTokenId(const std::string& tokenText) const override;
    
    /**
     * @brief Get tokenizer type
     * @return Tokenizer type string
     */
    CV_WRAP std::string getType() const { return "tiktoken"; }
    
    /**
     * @brief Get encoding name
     * @return Encoding name
     */
    CV_WRAP std::string getEncodingName() const { return encoding_name_; }

private:
    // Name of the encoding
    std::string encoding_name_;
    
    // Pattern string for regex splitting
    std::string pat_str_;
    
    // Byte pair encoding ranks
    std::unordered_map<std::vector<uint8_t>, int, VectorHash> bpe_ranks_;
    
    // Special tokens mapping (string -> id)
    std::unordered_map<std::string, int> special_tokens_;
    
    // Token id to bytes mapping (for decoding)
    std::unordered_map<int, std::vector<uint8_t>> token_byte_values_;
    
    // Set of special token ids for quick lookup
    std::unordered_set<int> special_token_ids_;
    
    // Helper methods
    std::vector<std::string> split_by_pattern(const std::string& text) const;
    std::vector<int> byte_pair_encode(const std::vector<uint8_t>& piece) const;
    std::vector<uint8_t> token_to_bytes(int token) const;
    
    // Load predefined encodings
    void load_encoding(const std::string& encoding_name);
    void load_bpe_ranks(const std::string& bpe_ranks_path);
};

/**
 * @brief Create a TikToken tokenizer
 * @param encoding_name Name of the encoding to use
 * @param bpe_ranks_path Optional path to the BPE ranks file
 * @param special_tokens Optional special tokens mapping
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createTikTokenTokenizer(
    const std::string& encoding_name,
    const std::string& bpe_ranks_path = "",
    const std::map<std::string, int>& special_tokens = {});

// Utility struct for vector hashing
struct VectorHash {
    size_t operator()(const std::vector<uint8_t>& v) const {
        size_t hash = 0;
        for (auto b : v) {
            hash = hash * 31 + b;
        }
        return hash;
    }
};

}} // namespace cv::dnn

#endif // OPENCV_DNN_TIKTOKEN_TOKENIZER_HPP