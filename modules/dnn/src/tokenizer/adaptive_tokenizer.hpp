#ifndef OPENCV_DNN_ADAPTIVE_TOKENIZER_HPP
#define OPENCV_DNN_ADAPTIVE_TOKENIZER_HPP

#include "tokenizer.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace cv {
namespace dnn {

/**
 * @brief Language detection result structure
 */
struct CV_EXPORTS_W_SIMPLE LanguageInfo {
    std::string language; //!< Detected language code (e.g., "en", "zh", "hi")
    std::string script;   //!< Detected script (e.g., "Latin", "Cyrillic", "Han")
    float confidence;     //!< Detection confidence (0-1)
};

/**
 * @brief Script-specific tokenization rules
 */
struct CV_EXPORTS_W_SIMPLE ScriptRules {
    std::string name;                 //!< Script name
    bool wordBoundaries;              //!< Whether script uses explicit word boundaries
    bool processAsWords;              //!< Process as whole words or individual characters
    std::string specialCharacters;    //!< Characters that need special handling
};

/**
 * @brief Adaptive tokenizer that changes strategy based on text content
 *
 * This tokenizer automatically detects language and script characteristics
 * and applies the most appropriate tokenization strategy. It's designed for
 * research on improving tokenization quality across diverse languages.
 */
class CV_EXPORTS_W AdaptiveTokenizer : public Tokenizer {
public:
    /**
     * @brief Constructor for adaptive tokenizer
     * @param bpe_tokenizer BPE tokenizer for use with alphabetic scripts
     * @param unigram_tokenizer Unigram tokenizer for use with syllabic scripts
     * @param char_tokenizer Character tokenizer for ideographic scripts
     */
    CV_WRAP AdaptiveTokenizer(
        const Ptr<Tokenizer>& bpe_tokenizer, 
        const Ptr<Tokenizer>& unigram_tokenizer,
        const Ptr<Tokenizer>& char_tokenizer = nullptr);
    
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
     * @brief Detect language and script of text
     * @param text Text to analyze
     * @return Language information
     */
    CV_WRAP LanguageInfo detectLanguage(const std::string& text) const;
    
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
     * @brief Generate encoding metrics for the last encoded text
     * @return Dictionary of metrics (efficiency, coverage, etc.)
     */
    CV_WRAP std::map<std::string, double> getTokenizationMetrics() const;
    
    /**
     * @brief Set script-specific rules
     * @param rules Map of script names to rules
     */
    CV_WRAP void setScriptRules(const std::map<std::string, ScriptRules>& rules);
    
    /**
     * @brief Register a tokenizer for a specific script
     * @param script Script name
     * @param tokenizer Tokenizer to use for this script
     */
    CV_WRAP void registerScriptTokenizer(const std::string& script, const Ptr<Tokenizer>& tokenizer);

private:
    // Component tokenizers
    Ptr<Tokenizer> bpe_tokenizer_;
    Ptr<Tokenizer> unigram_tokenizer_;
    Ptr<Tokenizer> char_tokenizer_;
    
    // Script-specific tokenizers
    std::map<std::string, Ptr<Tokenizer>> script_tokenizers_;
    
    // Script rules
    std::map<std::string, ScriptRules> script_rules_;
    
    // Cache for language detection
    mutable std::unordered_map<std::string, LanguageInfo> language_cache_;
    
    // Metrics from last encoding
    mutable std::map<std::string, double> last_metrics_;
    
    // Helper methods
    std::vector<std::string> splitByScript(const std::string& text) const;
    std::string getScriptName(char32_t codepoint) const;
    bool isIdeographic(const std::string& script) const;
    bool isSyllabic(const std::string& script) const;
    
    // Unicode utilities
    char32_t utf8ToCodepoint(const std::string& utf8_char) const;
    std::vector<char32_t> utf8ToCodepoints(const std::string& text) const;
    std::string codepointToUtf8(char32_t codepoint) const;
};

/**
 * @brief Create an adaptive tokenizer that selects strategies based on script
 * @param bpe_vocab_file Path to BPE vocabulary file
 * @param bpe_merges_file Path to BPE merges file
 * @param unigram_vocab_file Path to Unigram vocabulary file
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createAdaptiveTokenizer(
    const std::string& bpe_vocab_file,
    const std::string& bpe_merges_file,
    const std::string& unigram_vocab_file);

}} // namespace cv::dnn

#endif // OPENCV_DNN_ADAPTIVE_TOKENIZER_HPP