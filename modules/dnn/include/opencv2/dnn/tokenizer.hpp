#ifndef OPENCV_DNN_TOKENIZER_HPP
#define OPENCV_DNN_TOKENIZER_HPP

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <functional>

namespace cv {
namespace dnn {

/**
 * @brief Hash function for string pairs
 */
struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

/**
 * @brief Enum for tokenization preprocessing flags
 */
enum TokenizerPreprocessFlag {
    TOKENIZER_LOWERCASE = 1,        //!< Convert text to lowercase
    TOKENIZER_STRIP_ACCENTS = 2,    //!< Remove accents from characters
    TOKENIZER_NORMALIZE_SPACE = 4,  //!< Normalize whitespace
    TOKENIZER_HANDLE_CHINESE = 8,   //!< Special handling for Chinese characters
    TOKENIZER_STRIP_PUNCTUATION = 16 //!< Remove punctuation
};

/**
 * @brief Token info structure with extended token information
 */
struct CV_EXPORTS_W_SIMPLE TokenInfo {
    int id;           //!< Token ID
    std::string text; //!< Token text
    int start;        //!< Start position in original text
    int end;          //!< End position in original text
    float score;      //!< Token score or probability
};

/**
 * @brief Base class for all tokenizers
 */
class CV_EXPORTS_W Tokenizer {
public:
    /** @brief Virtual destructor */
    virtual ~Tokenizer() = default;

    /**
     * @brief Encodes a string into tokens
     * @param text Text to encode
     * @return Vector of token IDs
     */
    CV_WRAP virtual std::vector<int> encode(const std::string& text) const = 0;

    /**
     * @brief Encodes a string into tokens with detailed information
     * @param text Text to encode
     * @return Vector of TokenInfo structures
     */
    CV_WRAP virtual std::vector<TokenInfo> encodeWithInfo(const std::string& text) const;

    /**
     * @brief Batch encodes multiple strings into tokens
     * @param texts Vector of texts to encode
     * @return Vector of vectors of token IDs
     */
    CV_WRAP virtual std::vector<std::vector<int>> encodeBatch(const std::vector<std::string>& texts) const;

    /**
     * @brief Decodes a sequence of tokens back into text
     * @param tokens Vector of token IDs
     * @return Decoded text
     */
    CV_WRAP virtual std::string decode(const std::vector<int>& tokens) const = 0;

    /**
     * @brief Get the vocabulary size
     * @return Number of tokens in vocabulary
     */
    CV_WRAP virtual size_t getVocabSize() const = 0;

    /**
     * @brief Save tokenizer to a file
     * @param filename Path to save tokenizer
     */
    CV_WRAP virtual void save(const std::string& filename) const = 0;

    /**
     * @brief Export tokenizer to a different format
     * @param filename Path to save tokenizer
     * @param format Format to export ("huggingface", "tiktoken", etc.)
     */
    CV_WRAP virtual void exportTo(const std::string& filename, const std::string& format) const;

    /**
     * @brief Get token text for a token ID
     * @param tokenId Token ID
     * @return Token text or empty string if not found
     */
    CV_WRAP virtual std::string getTokenText(int tokenId) const;

    /**
     * @brief Get token ID for a token text
     * @param tokenText Token text
     * @return Token ID or -1 if not found
     */
    CV_WRAP virtual int getTokenId(const std::string& tokenText) const;
    
    /**
     * @brief Truncate tokens to a maximum length
     * @param tokens Vector of token IDs
     * @param maxLength Maximum length
     * @param strategy Truncation strategy ("left", "right", "middle")
     * @return Truncated tokens
     */
    CV_WRAP virtual std::vector<int> truncate(const std::vector<int>& tokens, 
                                             int maxLength, 
                                             const std::string& strategy = "right") const;

    /**
     * @brief Merge two token sequences with special tokens
     * @param tokens1 First token sequence
     * @param tokens2 Second token sequence
     * @param specialTokens Special tokens to insert between sequences
     * @return Merged token sequence
     */
    CV_WRAP virtual std::vector<int> mergeTokenSequences(const std::vector<int>& tokens1,
                                                        const std::vector<int>& tokens2,
                                                        const std::vector<int>& specialTokens = {}) const;

    /**
     * @brief Create a tokenizer from a saved file
     * @param filename Path to saved tokenizer
     * @return Created tokenizer
     */
    CV_WRAP static Ptr<Tokenizer> load(const std::string& filename);

    /**
     * @brief Import a tokenizer from a different format
     * @param directory Path to tokenizer directory or files
     * @param format Format to import from ("huggingface", "tiktoken", etc.)
     * @return Created tokenizer
     */
    CV_WRAP static Ptr<Tokenizer> importFrom(const std::string& directory, const std::string& format);
};

/**
 * @brief Configuration for tokenizer preprocessing
 */
struct CV_EXPORTS_W_SIMPLE TokenizerPreprocessConfig {
    CV_WRAP TokenizerPreprocessConfig() : flags(TOKENIZER_LOWERCASE | TOKENIZER_NORMALIZE_SPACE) {}
    
    int flags;                                   //!< Preprocessing flags
    std::vector<std::string> specialTokens;      //!< Special tokens that should not be split
    std::unordered_map<std::string, std::string> replacements; //!< Text replacements to apply
};

/**
 * @brief Factory function to create a BPE tokenizer
 * @param vocab_file Path to vocabulary file
 * @param merges_file Path to merges file
 * @param config Preprocessing configuration
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createBPETokenizer(const std::string& vocab_file, 
                                               const std::string& merges_file,
                                               const TokenizerPreprocessConfig& config = TokenizerPreprocessConfig());

/**
 * @brief Factory function to create a WordPiece tokenizer
 * @param vocab_file Path to vocabulary file
 * @param config Preprocessing configuration
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createWordPieceTokenizer(const std::string& vocab_file,
                                                    const TokenizerPreprocessConfig& config = TokenizerPreprocessConfig());

/**
 * @brief Factory function to create a Unigram tokenizer
 * @param vocab_file Path to vocabulary file
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

/**
 * @brief Factory function to create a SentencePiece tokenizer
 * @param model_path Path to SentencePiece model file (.model)
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createSentencePieceTokenizer(const std::string& model_path);

/**
 * @brief Factory function to create a TikToken tokenizer
 * @param encoding_name Name of the encoding to use
 * @param bpe_ranks_path Optional path to the BPE ranks file
 * @param special_tokens Optional special tokens mapping
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createTikTokenTokenizer(
    const std::string& encoding_name,
    const std::string& bpe_ranks_path = "",
    const std::map<std::string, int>& special_tokens = {});

/**
 * @brief Factory function to create a Transformers tokenizer
 * @param model_name_or_path Model name or path to the model directory
 * @param use_fast Whether to use the fast tokenizer implementation if available
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createTransformersTokenizer(
    const std::string& model_name_or_path,
    bool use_fast = true);

/**
 * @brief Factory function to create an adaptive tokenizer that selects strategies based on script
 * @param bpe_vocab_file Path to BPE vocabulary file
 * @param bpe_merges_file Path to BPE merges file
 * @param unigram_vocab_file Path to Unigram vocabulary file
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> createAdaptiveTokenizer(
    const std::string& bpe_vocab_file,
    const std::string& bpe_merges_file,
    const std::string& unigram_vocab_file);

/**
 * @brief Load a tokenizer model from a directory
 * @param dir_path Path to the directory containing tokenizer files
 * @return Smart pointer to the created tokenizer
 */
CV_EXPORTS_W Ptr<Tokenizer> loadTokenizerFromDirectory(const std::string& dir_path);

/**
 * @brief Function to benchmark tokenizer performance
 * @param tokenizer Tokenizer to benchmark
 * @param texts Texts to use for benchmarking
 * @param iterations Number of iterations for benchmarking
 * @return Dictionary with benchmark results
 */
CV_EXPORTS_W std::map<std::string, double> benchmarkTokenizer(
    const Ptr<Tokenizer>& tokenizer,
    const std::vector<std::string>& texts,
    int iterations = 100
);

}} // namespace cv::dnn

#endif // OPENCV_DNN_TOKENIZER_HPP