#ifndef OPENCV_DNN_TOKENIZATION_ANALYSIS_HPP
#define OPENCV_DNN_TOKENIZATION_ANALYSIS_HPP

#include "tokenizer.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace cv {
namespace dnn {

/**
 * @brief Analysis result for comparing tokenization strategies
 */
struct CV_EXPORTS_W_SIMPLE TokenizationAnalysisResult {
    std::string tokenizer_name;          //!< Name of the tokenizer
    double compression_ratio;            //!< Ratio of characters to tokens
    double encoding_time_ms;             //!< Time to encode in milliseconds
    double decoding_time_ms;             //!< Time to decode in milliseconds
    double vocabulary_coverage;          //!< Percentage of text covered by vocabulary (0-1)
    double unknown_token_percentage;     //!< Percentage of unknown tokens (0-1)
    int total_tokens;                    //!< Total number of tokens
    int unique_tokens;                   //!< Number of unique tokens used
    std::map<std::string, double> metrics; //!< Additional custom metrics
};

/**
 * @brief Framework for analyzing and comparing tokenization strategies
 *
 * This class provides tools for quantitative analysis of tokenizers, 
 * enabling research into tokenization quality and performance.
 */
class CV_EXPORTS_W TokenizationAnalysis {
public:
    /**
     * @brief Constructor
     */
    CV_WRAP TokenizationAnalysis();
    
    /**
     * @brief Register a tokenizer for analysis
     * @param name Name to identify this tokenizer
     * @param tokenizer Tokenizer instance
     */
    CV_WRAP void registerTokenizer(const std::string& name, const Ptr<Tokenizer>& tokenizer);
    
    /**
     * @brief Analyze performance and quality metrics for registered tokenizers
     * @param corpus Text corpus for analysis
     * @param num_iterations Number of iterations for timing measurements
     * @return Map of tokenizer name to analysis results
     */
    CV_WRAP std::map<std::string, TokenizationAnalysisResult> analyzeAll(
        const std::string& corpus, 
        int num_iterations = 5);
    
    /**
     * @brief Analyze a single tokenizer's performance
     * @param name Name of the tokenizer to analyze
     * @param corpus Text corpus for analysis
     * @param num_iterations Number of iterations for timing measurements
     * @return Analysis results
     */
    CV_WRAP TokenizationAnalysisResult analyze(
        const std::string& name,
        const std::string& corpus,
        int num_iterations = 5);
    
    /**
     * @brief Visualize token distribution
     * @param name Name of the tokenizer to visualize
     * @param corpus Text corpus for visualization
     * @param output_file Path to save visualization image (PNG)
     */
    CV_WRAP void visualizeTokenDistribution(
        const std::string& name,
        const std::string& corpus,
        const std::string& output_file);
    
    /**
     * @brief Compare token efficiency for different languages
     * @param languages Map of language codes to sample texts
     * @return Map of tokenizer name to language-specific analysis
     */
    CV_WRAP std::map<std::string, std::map<std::string, TokenizationAnalysisResult>> 
    compareLanguages(const std::map<std::string, std::string>& languages);

    /**
     * @brief Analyze tokenization bias by measuring token distribution across demographic text samples
     * @param demographic_samples Map of demographic group names to sample texts
     * @param output_file Path to save bias analysis report (CSV)
     */
    CV_WRAP void analyzeBias(
        const std::map<std::string, std::string>& demographic_samples,
        const std::string& output_file);
    
    /**
     * @brief Generate benchmark corpus with specific characteristics
     * @param options Map of options (e.g., "languages", "technical_content", "ambiguities")
     * @return Generated benchmark corpus
     */
    CV_WRAP static std::string generateBenchmarkCorpus(const std::map<std::string, std::string>& options);
    
    /**
     * @brief Export analysis results to CSV file
     * @param results Analysis results to export
     * @param output_file Path to save CSV file
     */
    CV_WRAP static void exportResultsToCSV(
        const std::map<std::string, TokenizationAnalysisResult>& results,
        const std::string& output_file);

private:
    // Registered tokenizers
    std::map<std::string, Ptr<Tokenizer>> tokenizers_;
    
    // Helper methods
    double calculateCompressionRatio(const std::string& text, const std::vector<int>& tokens);
    double calculateVocabularyCoverage(const Ptr<Tokenizer>& tokenizer, const std::string& text);
    std::map<int, int> getTokenFrequencies(const std::vector<int>& tokens);
    void generateTokenizationHeatmap(const std::string& text, const std::vector<int>& tokens, Mat& output);
};

}} // namespace cv::dnn

#endif // OPENCV_DNN_TOKENIZATION_ANALYSIS_HPP