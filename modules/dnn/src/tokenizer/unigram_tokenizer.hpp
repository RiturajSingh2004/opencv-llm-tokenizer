#ifndef OPENCV_DNN_UNIGRAM_TOKENIZER_HPP
#define OPENCV_DNN_UNIGRAM_TOKENIZER_HPP

#include "tokenizer.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

namespace cv {
namespace dnn {

/**
 * @brief Implementation of Unigram Tokenizer as used in SentencePiece
 * 
 * This tokenizer implements the Unigram language model algorithm from SentencePiece,
 * which is used in models like T5, mBART, and XLM-RoBERTa.
 */
class CV_EXPORTS_W UnigramTokenizer : public Tokenizer {
public:
    /**
     * @brief Constructor for Unigram tokenizer
     * @param vocab_file Path to vocabulary file with scores
     * @param unk_id ID for unknown token (default: 0)
     * @param unk_piece String representation of unknown token (default: "<unk>")
     * @param score_threshold Score threshold for subword sampling (default: 0.0)
     * @param sample_alpha Alpha parameter for sampling subwords (default: 0.1)
     */
    CV_WRAP UnigramTokenizer(const std::string& vocab_file,
                             int unk_id = 0,
                             const std::string& unk_piece = "<unk>",
                             float score_threshold = 0.0f,
                             float sample_alpha = 0.1f);
    
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
    CV_WRAP std::string getType() const { return "unigram"; }
    
    /**
     * @brief Set sampling parameters
     * @param enable Enable subword sampling
     * @param alpha Alpha parameter for sampling subwords
     */
    CV_WRAP void setSampling(bool enable, float alpha = 0.1f);
    
    /**
     * @brief Get scores for tokens
     * @return Map of token IDs to scores
     */
    CV_WRAP std::unordered_map<int, float> getScores() const;

private:
    // Token to ID mapping
    std::unordered_map<std::string, int> token_to_id_;
    
    // ID to token mapping
    std::unordered_map<int, std::string> id_to_token_;
    
    // Scores for each token (log probabilities)
    std::unordered_map<int, float> scores_;
    
    // Unknown token ID and string
    int unk_id_;
    std::string unk_piece_;
    
    // Sampling parameters
    bool enable_sampling_;
    float score_threshold_;
    float sample_alpha_;
    
    // Helper methods
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<int> convert_tokens_to_ids(const std::vector<std::string>& tokens) const;
    std::string preprocess_text(const std::string& text) const;
    
    // Viterbi algorithm for finding the best segmentation
    std::vector<std::string> viterbi_segment(const std::string& text) const;
    
    // Calculate lattice for segmentation
    std::vector<std::pair<float, std::vector<std::string>>> calculate_lattice(const std::string& text) const;
    
    // Load vocabulary and scores
    void loadVocabulary(const std::string& vocab_file);
};

/**
 * @brief Create a Unigram tokenizer
 * @param vocab_file Path to vocabulary file with scores
 * @param unk_id ID for unknown token (default: 0)
 * @param unk_piece String representation of unknown token (default: "<unk>")
 * @param score_threshold Score threshold for subword sampling (default: 0.0)
 * @param sample_alpha Alpha parameter for sampling subwords (default: 0.1)
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createUnigramTokenizer(const std::string& vocab_file,
                                                  int unk_id = 0,
                                                  const std::string& unk_piece = "<unk>",
                                                  float score_threshold = 0.0f,
                                                  float sample_alpha = 0.1f);

}} // namespace cv::dnn

#endif // OPENCV_DNN_UNIGRAM_TOKENIZER_HPP