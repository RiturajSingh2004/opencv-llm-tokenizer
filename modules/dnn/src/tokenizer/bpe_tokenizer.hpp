#ifndef OPENCV_DNN_BPE_TOKENIZER_HPP
#define OPENCV_DNN_BPE_TOKENIZER_HPP

#include "tokenizer.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <utility>
#include <memory>

namespace cv {
namespace dnn {

/**
 * @brief Implementation of Byte-Pair Encoding (BPE) tokenizer
 * 
 * This tokenizer implements the BPE algorithm as used in GPT models.
 * It supports loading pre-trained vocabularies and merge rules.
 */
class CV_EXPORTS_W BPETokenizer : public Tokenizer {
public:
    /**
     * @brief Constructor for BPE tokenizer
     * @param vocab_file Path to vocabulary file
     * @param merges_file Path to BPE merges file
     * @param config Preprocessing configuration
     */
    CV_WRAP BPETokenizer(const std::string& vocab_file, 
                        const std::string& merges_file,
                        const TokenizerPreprocessConfig& config = TokenizerPreprocessConfig());
    
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
     * @brief Export tokenizer to a different format
     * @param filename Path where to save tokenizer
     * @param format Format to export ("huggingface", "tiktoken", etc.)
     */
    CV_WRAP void exportTo(const std::string& filename, const std::string& format) const override;
    
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
    CV_WRAP std::string getType() const { return "bpe"; }
    
    /**
     * @brief Visualize tokenization for the given text
     * @param text Text to tokenize
     * @param colorized Whether to return colorized HTML output
     * @return Visualization string or HTML
     */
    CV_WRAP std::string visualizeTokenization(const std::string& text, bool colorized = false) const;

private:
    // Token to ID mapping
    std::unordered_map<std::string, int> token_to_id_;
    
    // ID to token mapping
    std::unordered_map<int, std::string> id_to_token_;
    
    // BPE merges priority
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges_;
    
    // Maximum token length in bytes
    size_t max_token_length_;
    
    // Preprocessing configuration
    TokenizerPreprocessConfig config_;
    
    // Helper methods
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<std::string> bpe_encode(const std::string& token) const;
    std::vector<int> convert_tokens_to_ids(const std::vector<std::string>& tokens) const;
    std::string preprocess_text(const std::string& text) const;
    
    // Load vocabulary and merges
    void loadVocabulary(const std::string& vocab_file);
    void loadMerges(const std::string& merges_file);
};

/**
 * @brief Create a BPE tokenizer
 * @param vocab_file Path to vocabulary file
 * @param merges_file Path to BPE merges file
 * @param config Preprocessing configuration
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createBPETokenizer(const std::string& vocab_file, 
                                              const std::string& merges_file,
                                              const TokenizerPreprocessConfig& config = TokenizerPreprocessConfig());

}} // namespace cv::dnn

#endif // OPENCV_DNN_BPE_TOKENIZER_HPP:#ifndef OPENCV_DNN_BPE_TOKENIZER_HPP
#define OPENCV_DNN_BPE_TOKENIZER_HPP

#include "tokenizer.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <utility>
#include <memory>

namespace cv {
namespace dnn {

/**
 * @brief Implementation of Byte-Pair Encoding (BPE) tokenizer
 * 
 * This tokenizer implements the BPE algorithm as used in GPT models.
 * It supports loading pre-trained vocabularies and merge rules.
 */
class CV_EXPORTS_W BPETokenizer : public Tokenizer {
public:
    /**
     * @brief Constructor for BPE tokenizer
     * @param vocab_file Path to vocabulary file
     * @param merges_file Path to BPE merges file
     */
    CV_WRAP BPETokenizer(const std::string& vocab_file, const std::string& merges_file);
    
    /**
     * @brief Encode text into token IDs
     * @param text Input text to encode
     * @return Vector of token IDs
     */
    CV_WRAP std::vector<int> encode(const std::string& text) const override;
    
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

private:
    // Token to ID mapping
    std::unordered_map<std::string, int> token_to_id_;
    
    // ID to token mapping
    std::unordered_map<int, std::string> id_to_token_;
    
    // BPE merges priority
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges_;
    
    // Maximum token length in bytes
    size_t max_token_length_;
    
    // Helper methods
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<std::string> bpe_encode(const std::string& token) const;
    std::vector<int> convert_tokens_to_ids(const std::vector<std::string>& tokens) const;
    
    // Load vocabulary and merges
    void loadVocabulary(const std::string& vocab_file);
    void loadMerges(const std::string& merges_file);
};

/**
 * @brief Create a BPE tokenizer
 * @param vocab_file Path to vocabulary file
 * @param merges_file Path to BPE merges file
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createBPETokenizer(const std::string& vocab_file, 
                                              const std::string& merges_file);

}} // namespace cv::dnn

#endif // OPENCV_DNN_BPE_TOKENIZER_HPP