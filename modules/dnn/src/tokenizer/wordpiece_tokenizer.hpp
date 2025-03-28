#ifndef OPENCV_DNN_WORDPIECE_TOKENIZER_HPP
#define OPENCV_DNN_WORDPIECE_TOKENIZER_HPP

#include "tokenizer.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <utility>
#include <memory>

namespace cv {
namespace dnn {

/**
 * @brief Implementation of WordPiece tokenizer
 * 
 * This tokenizer implements the WordPiece algorithm as used in BERT models.
 * It supports loading pre-trained vocabularies.
 */
class CV_EXPORTS_W WordPieceTokenizer : public Tokenizer {
public:
    /**
     * @brief Constructor for WordPiece tokenizer
     * @param vocab_file Path to vocabulary file
     * @param unk_token Token to use for unknown words (default: "[UNK]")
     * @param max_chars_per_word Maximum number of characters per word (default: 100)
     */
    CV_WRAP WordPieceTokenizer(const std::string& vocab_file,
                               const std::string& unk_token = "[UNK]",
                               int max_chars_per_word = 100);
    
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
    
    // Unknown token
    std::string unk_token_;
    
    // Unknown token ID
    int unk_token_id_;
    
    // Maximum number of characters per word
    int max_chars_per_word_;
    
    // Special tokens
    std::vector<std::string> special_tokens_;
    
    // Helper methods
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<std::string> wordpiece_tokenize(const std::string& word) const;
    std::vector<int> convert_tokens_to_ids(const std::vector<std::string>& tokens) const;
    
    // Load vocabulary
    void loadVocabulary(const std::string& vocab_file);
};

/**
 * @brief Create a WordPiece tokenizer
 * @param vocab_file Path to vocabulary file
 * @param unk_token Token to use for unknown words (default: "[UNK]")
 * @param max_chars_per_word Maximum number of characters per word (default: 100)
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createWordPieceTokenizer(const std::string& vocab_file,
                                                    const std::string& unk_token = "[UNK]",
                                                    int max_chars_per_word = 100);

}} // namespace cv::dnn

#endif // OPENCV_DNN_WORDPIECE_TOKENIZER_HPP