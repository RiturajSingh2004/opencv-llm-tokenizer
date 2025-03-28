#ifndef OPENCV_DNN_SENTENCEPIECE_TOKENIZER_HPP
#define OPENCV_DNN_SENTENCEPIECE_TOKENIZER_HPP

#include "tokenizer.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace cv {
namespace dnn {

// Forward declaration of SentencePiece processor wrapper
class SPProcessorWrapper;

/**
 * @brief Implementation of SentencePiece tokenizer
 * 
 * This tokenizer provides direct compatibility with SentencePiece models.
 * It supports all tokenization modes of SentencePiece (unigram, BPE, char, word)
 * and can load models directly from .model files.
 */
class CV_EXPORTS_W SentencePieceTokenizer : public Tokenizer {
public:
    /**
     * @brief Constructor for SentencePiece tokenizer
     * @param model_path Path to SentencePiece model file (.model)
     */
    CV_WRAP SentencePieceTokenizer(const std::string& model_path);
    
    /**
     * @brief Destructor
     */
    virtual ~SentencePieceTokenizer();
    
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
     * @brief Get SentencePiece tokenizer type
     * @return SentencePiece model type (unigram, bpe, char, word)
     */
    CV_WRAP std::string getModelType() const;
    
    /**
     * @brief Get the normalized version of input text
     * @param text Text to normalize
     * @return Normalized text
     */
    CV_WRAP std::string normalizeText(const std::string& text) const;

private:
    // PIMPL to hide SentencePiece dependencies
    std::unique_ptr<SPProcessorWrapper> processor_;
    std::string model_path_;
};

/**
 * @brief Create a SentencePiece tokenizer
 * @param model_path Path to SentencePiece model file (.model)
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createSentencePieceTokenizer(const std::string& model_path);

}} // namespace cv::dnn

#endif // OPENCV_DNN_SENTENCEPIECE_TOKENIZER_HPP