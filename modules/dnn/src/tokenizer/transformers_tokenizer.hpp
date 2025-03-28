#ifndef OPENCV_DNN_TRANSFORMERS_TOKENIZER_HPP
#define OPENCV_DNN_TRANSFORMERS_TOKENIZER_HPP

#include "tokenizer.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace cv {
namespace dnn {

// Forward declaration of TransformersWrapper
class TransformersWrapper;

/**
 * @brief Tokenizer compatible with HuggingFace Transformers
 *
 * This tokenizer provides direct compatibility with HuggingFace Transformers models.
 * It can load pretrained tokenizers from the Transformers library and supports
 * all tokenization features like padding, truncation, and special token handling.
 */
class CV_EXPORTS_W TransformersTokenizer : public Tokenizer {
public:
    /**
     * @brief Constructor for Transformers tokenizer
     * @param model_name_or_path Model name or path to the model directory
     * @param use_fast Whether to use the fast tokenizer implementation if available
     */
    CV_WRAP TransformersTokenizer(const std::string& model_name_or_path, bool use_fast = true);
    
    /**
     * @brief Destructor
     */
    virtual ~TransformersTokenizer();
    
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
     * @brief Export tokenizer to a different format
     * @param filename Path where to save tokenizer
     * @param format Format to export (e.g., "tiktoken", "sentencepiece")
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
     * @brief Encode text for model input with additional options
     * @param text Input text
     * @param max_length Maximum sequence length
     * @param padding Whether to pad the sequence
     * @param truncation Whether to truncate the sequence
     * @param add_special_tokens Whether to add special tokens
     * @return Encoded tokens
     */
    CV_WRAP std::vector<int> encodeForModel(
        const std::string& text,
        int max_length = 512,
        bool padding = false,
        bool truncation = true,
        bool add_special_tokens = true) const;
    
    /**
     * @brief Batch encode texts for model input with additional options
     * @param texts Input texts
     * @param max_length Maximum sequence length
     * @param padding Whether to pad the sequence
     * @param truncation Whether to truncate the sequence
     * @param add_special_tokens Whether to add special tokens
     * @return Batch of encoded tokens
     */
    CV_WRAP std::vector<std::vector<int>> encodeBatchForModel(
        const std::vector<std::string>& texts,
        int max_length = 512,
        bool padding = false,
        bool truncation = true,
        bool add_special_tokens = true) const;
    
    /**
     * @brief Get model name or path
     * @return Model name or path
     */
    CV_WRAP std::string getModelNameOrPath() const;
    
    /**
     * @brief Get tokenizer type (e.g., "BertTokenizer", "GPT2Tokenizer")
     * @return Tokenizer type
     */
    CV_WRAP std::string getTokenizerType() const;
    
    /**
     * @brief Check if the tokenizer is using a fast implementation
     * @return True if using fast implementation
     */
    CV_WRAP bool isFast() const;

private:
    // PIMPL to hide Python/Transformers dependencies
    std::unique_ptr<TransformersWrapper> wrapper_;
    std::string model_name_or_path_;
    bool use_fast_;
};

/**
 * @brief Create a Transformers tokenizer
 * @param model_name_or_path Model name or path to the model directory
 * @param use_fast Whether to use the fast tokenizer implementation if available
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createTransformersTokenizer(
    const std::string& model_name_or_path,
    bool use_fast = true);

}} // namespace cv::dnn

#endif // OPENCV_DNN_TRANSFORMERS_TOKENIZER_HPP