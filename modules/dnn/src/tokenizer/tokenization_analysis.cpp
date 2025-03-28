/**
 * @file tokenization_analysis.cpp
 * @brief Implementation of tokenization analysis and benchmarking tools
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements analysis and benchmarking utilities for tokenizers,
 * including performance measurement, token distribution visualization,
 * and cross-language comparison tools for tokenization research.
 */

#include "precomp.hpp"
#include "tokenization_analysis.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <unordered_map>
#include <set>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace cv {
namespace dnn {

TokenizationAnalysis::TokenizationAnalysis() {
    // Nothing to initialize
}

void TokenizationAnalysis::registerTokenizer(const std::string& name, const Ptr<Tokenizer>& tokenizer) {
    if (!tokenizer) {
        CV_Error(Error::StsNullPtr, "Cannot register null tokenizer");
    }
    
    tokenizers_[name] = tokenizer;
}

std::map<std::string, TokenizationAnalysisResult> TokenizationAnalysis::analyzeAll(
    const std::string& corpus, 
    int num_iterations) {
    
    std::map<std::string, TokenizationAnalysisResult> results;
    
    for (const auto& [name, tokenizer] : tokenizers_) {
        results[name] = analyze(name, corpus, num_iterations);
    }
    
    return results;
}

TokenizationAnalysisResult TokenizationAnalysis::analyze(
    const std::string& name,
    const std::string& corpus,
    int num_iterations) {
    
    auto it = tokenizers_.find(name);
    if (it == tokenizers_.end()) {
        CV_Error(Error::StsBadArg, "Tokenizer not found: " + name);
    }
    
    Ptr<Tokenizer> tokenizer = it->second;
    TokenizationAnalysisResult result;
    result.tokenizer_name = name;
    
    // Measure encoding time
    double total_encoding_time = 0.0;
    std::vector<int> tokens;
    
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        tokens = tokenizer->encode(corpus);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        total_encoding_time += duration.count();
    }
    
    result.encoding_time_ms = total_encoding_time / num_iterations;
    
    // Measure decoding time
    double total_decoding_time = 0.0;
    std::string decoded;
    
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        decoded = tokenizer->decode(tokens);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        total_decoding_time += duration.count();
    }
    
    result.decoding_time_ms = total_decoding_time / num_iterations;
    
    // Calculate metrics
    result.total_tokens = static_cast<int>(tokens.size());
    result.compression_ratio = calculateCompressionRatio(corpus, tokens);
    
    // Calculate unique tokens
    std::set<int> unique_tokens(tokens.begin(), tokens.end());
    result.unique_tokens = static_cast<int>(unique_tokens.size());
    
    // Calculate vocabulary coverage
    result.vocabulary_coverage = calculateVocabularyCoverage(tokenizer, corpus);
    
    // Calculate frequency distribution metrics
    std::map<int, int> token_frequencies = getTokenFrequencies(tokens);
    
    // Calculate entropy of distribution as a measure of token utilization efficiency
    double entropy = 0.0;
    for (const auto& [token, freq] : token_frequencies) {
        double p = static_cast<double>(freq) / tokens.size();
        entropy -= p * std::log2(p);
    }
    result.metrics["entropy"] = entropy;
    
    // Calculate token length distribution
    std::map<int, int> token_length_counts;
    double avg_token_length = 0.0;
    
    for (int token : tokens) {
        std::string token_text = tokenizer->getTokenText(token);
        int length = static_cast<int>(token_text.length());
        token_length_counts[length]++;
        avg_token_length += length;
    }
    
    result.metrics["avg_token_length"] = avg_token_length / tokens.size();
    
    // Calculate unknown token percentage
    // This depends on tokenizer implementation, but we can check for common unknown token values
    // For example, check for empty token text or special unknown token IDs
    int unknown_count = 0;
    for (int token : tokens) {
        std::string token_text = tokenizer->getTokenText(token);
        if (token_text.empty() || token_text == "[UNK]" || token_text == "<unk>") {
            unknown_count++;
        }
    }
    
    result.unknown_token_percentage = static_cast<double>(unknown_count) / tokens.size();
    
    // Calculate additional metrics
    // Token type-token ratio (lexical diversity)
    result.metrics["type_token_ratio"] = static_cast<double>(unique_tokens.size()) / tokens.size();
    
    // Calculate token distribution skew (Zipfian distribution fit)
    // Sort tokens by frequency
    std::vector<std::pair<int, int>> sorted_freqs;
    for (const auto& [token, freq] : token_frequencies) {
        sorted_freqs.push_back({token, freq});
    }
    
    std::sort(sorted_freqs.begin(), sorted_freqs.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Calculate Zipf's law fit (log-log slope should be close to -1 for natural language)
    if (sorted_freqs.size() >= 5) {  // Need at least a few points for meaningful regression
        double sum_log_rank = 0.0;
        double sum_log_freq = 0.0;
        double sum_log_rank_squared = 0.0;
        double sum_log_rank_log_freq = 0.0;
        int n = std::min(static_cast<int>(sorted_freqs.size()), 100);  // Use top 100 tokens for calculation
        
        for (int i = 0; i < n; ++i) {
            double log_rank = std::log(i + 1);
            double log_freq = std::log(sorted_freqs[i].second);
            
            sum_log_rank += log_rank;
            sum_log_freq += log_freq;
            sum_log_rank_squared += log_rank * log_rank;
            sum_log_rank_log_freq += log_rank * log_freq;
        }
        
        // Calculate slope using linear regression
        double slope = (n * sum_log_rank_log_freq - sum_log_rank * sum_log_freq) / 
                       (n * sum_log_rank_squared - sum_log_rank * sum_log_rank);
        
        result.metrics["zipf_slope"] = slope;
    }
    
    return result;
}

void TokenizationAnalysis::visualizeTokenDistribution(
    const std::string& name,
    const std::string& corpus,
    const std::string& output_file) {
    
    auto it = tokenizers_.find(name);
    if (it == tokenizers_.end()) {
        CV_Error(Error::StsBadArg, "Tokenizer not found: " + name);
    }
    
    Ptr<Tokenizer> tokenizer = it->second;
    std::vector<int> tokens = tokenizer->encode(corpus);
    
    // Create token frequency map
    std::map<int, int> token_frequencies = getTokenFrequencies(tokens);
    
    // Sort tokens by frequency
    std::vector<std::pair<int, int>> sorted_freqs;
    for (const auto& [token, freq] : token_frequencies) {
        sorted_freqs.push_back({token, freq});
    }
    
    std::sort(sorted_freqs.begin(), sorted_freqs.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Create visualization
    // First, create histogram of top N tokens
    const int top_n = std::min(50, static_cast<int>(sorted_freqs.size()));
    Mat histogram(400, 800, CV_8UC3, Scalar(255, 255, 255));
    
    // Find max frequency for scaling
    int max_freq = sorted_freqs.empty() ? 1 : sorted_freqs[0].second;
    
    // Draw histogram bars
    int bar_width = histogram.cols / (top_n + 1);
    for (int i = 0; i < top_n; ++i) {
        int token = sorted_freqs[i].first;
        int freq = sorted_freqs[i].second;
        
        // Calculate bar height
        int bar_height = static_cast<int>(static_cast<double>(freq) / max_freq * 350);
        
        // Draw bar
        Point pt1(i * bar_width + 10, histogram.rows - 30);
        Point pt2((i + 1) * bar_width - 5, histogram.rows - 30 - bar_height);
        
        // Use different colors for different token types
        Scalar color;
        std::string token_text = tokenizer->getTokenText(token);
        
        if (token_text.empty()) {
            color = Scalar(0, 0, 255);  // Red for unknown tokens
        } else if (token_text.length() == 1) {
            color = Scalar(0, 255, 0);  // Green for single-character tokens
        } else if (token_text.find(" ") != std::string::npos) {
            color = Scalar(255, 0, 0);  // Blue for tokens containing spaces
        } else {
            color = Scalar(0, 128, 128);  // Yellow-green for other tokens
        }
        
        rectangle(histogram, pt1, pt2, color, FILLED);
        rectangle(histogram, pt1, pt2, Scalar(0, 0, 0), 1);
        
        // Add token text if space permits
        if (bar_width > 15) {
            std::string label = token_text.substr(0, 5);
            if (token_text.length() > 5) {
                label += "...";
            }
            putText(histogram, label, Point(i * bar_width + 10, histogram.rows - 10),
                    FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 0), 1);
        }
    }
    
    // Add title and legend
    putText(histogram, "Top " + std::to_string(top_n) + " Token Frequency Distribution - " + name,
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
    
    // Add legend
    rectangle(histogram, Point(650, 50), Point(670, 70), Scalar(0, 0, 255), FILLED);
    putText(histogram, "Unknown", Point(675, 65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    
    rectangle(histogram, Point(650, 80), Point(670, 100), Scalar(0, 255, 0), FILLED);
    putText(histogram, "Single char", Point(675, 95), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    
    rectangle(histogram, Point(650, 110), Point(670, 130), Scalar(255, 0, 0), FILLED);
    putText(histogram, "With space", Point(675, 125), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    
    rectangle(histogram, Point(650, 140), Point(670, 160), Scalar(0, 128, 128), FILLED);
    putText(histogram, "Other", Point(675, 155), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    
    // Also add a second visualization: token heatmap
    Mat heatmap;
    generateTokenizationHeatmap(corpus, tokens, heatmap);
    
    // Combine visualizations vertically
    Mat visualization;
    vconcat(histogram, heatmap, visualization);
    
    // Save to file
    imwrite(output_file, visualization);
}

std::map<std::string, std::map<std::string, TokenizationAnalysisResult>> 
TokenizationAnalysis::compareLanguages(const std::map<std::string, std::string>& languages) {
    std::map<std::string, std::map<std::string, TokenizationAnalysisResult>> results;
    
    for (const auto& [name, tokenizer] : tokenizers_) {
        std::map<std::string, TokenizationAnalysisResult> language_results;
        
        for (const auto& [lang_code, sample_text] : languages) {
            language_results[lang_code] = analyze(name, sample_text, 3);  // Use fewer iterations for multiple languages
        }
        
        results[name] = language_results;
    }
    
    return results;
}

void TokenizationAnalysis::analyzeBias(
    const std::map<std::string, std::string>& demographic_samples,
    const std::string& output_file) {
    
    // Initialize CSV output
    std::ofstream csv(output_file);
    if (!csv.is_open()) {
        CV_Error(Error::StsError, "Could not open output file for bias analysis: " + output_file);
    }
    
    // Write CSV header
    csv << "Tokenizer,Demographic,CompressionRatio,EncodingTimeMs,TotalTokens,UniqueTokens,";
    csv << "VocabularyCoverage,UnknownTokenPercentage,Entropy,TypeTokenRatio\n";
    
    // Analyze each demographic sample with each tokenizer
    for (const auto& [name, tokenizer] : tokenizers_) {
        for (const auto& [demographic, sample] : demographic_samples) {
            // Analyze the sample
            TokenizationAnalysisResult result = analyze(name, sample, 3);
            
            // Write results to CSV
            csv << name << "," << demographic << "," << result.compression_ratio << ",";
            csv << result.encoding_time_ms << "," << result.total_tokens << ",";
            csv << result.unique_tokens << "," << result.vocabulary_coverage << ",";
            csv << result.unknown_token_percentage << "," << result.metrics["entropy"] << ",";
            csv << result.metrics["type_token_ratio"] << "\n";
        }
    }
    
    csv.close();
}

std::string TokenizationAnalysis::generateBenchmarkCorpus(const std::map<std::string, std::string>& options) {
    std::stringstream corpus;
    
    // Include multilingual content if requested
    if (options.find("languages") != options.end()) {
        std::string langs = options.at("languages");
        if (langs.find("en") != std::string::npos) {
            corpus << "This is a sample English text for benchmarking tokenizers. ";
            corpus << "It includes various punctuation, numbers (123, 45.67), and special cases!\n";
        }
        if (langs.find("zh") != std::string::npos) {
            corpus << "这是一个用于基准测试分词器的中文样本文本。它包括各种标点符号，数字和特殊情况！\n";
        }
        if (langs.find("ar") != std::string::npos) {
            corpus << "هذا نص عربي عينة لاختبار أداء المحللات اللغوية. يتضمن علامات ترقيم مختلفة وأرقام وحالات خاصة!\n";
        }
        if (langs.find("hi") != std::string::npos) {
            corpus << "यह टोकनाइज़र के बेंचमार्किंग के लिए एक हिंदी नमूना पाठ है। इसमें विभिन्न विराम चिह्न, संख्याएँ और विशेष मामले शामिल हैं!\n";
        }
    }
    
    // Include technical content if requested
    if (options.find("technical_content") != options.end() && options.at("technical_content") == "true") {
        corpus << "# Programming Examples\n";
        corpus << "```python\n";
        corpus << "def tokenize(text):\n";
        corpus << "    return [token for token in text.split()]\n";
        corpus << "```\n\n";
        
        corpus << "## JSON Example\n";
        corpus << "```json\n";
        corpus << "{\n";
        corpus << "  \"name\": \"Tokenizer\",\n";
        corpus << "  \"version\": 1.0,\n";
        corpus << "  \"supported_languages\": [\"en\", \"fr\", \"de\"]\n";
        corpus << "}\n";
        corpus << "```\n\n";
        
        corpus << "Mathematical equations: E = mc^2, f(x) = ∫g(x)dx\n";
    }
    
    // Include ambiguities if requested
    if (options.find("ambiguities") != options.end() && options.at("ambiguities") == "true") {
        corpus << "Tokenization ambiguities:\n";
        corpus << "- 'don't' vs 'do n't'\n";
        corpus << "- 'New York-based' vs 'New-York-based'\n";
        corpus << "- 'co-operation' vs 'cooperation'\n";
        corpus << "- 'e-mail' vs 'email'\n";
        corpus << "- '2,000' vs '2000'\n";
    }
    
    // Include repeated patterns if requested
    if (options.find("repetition") != options.end() && options.at("repetition") == "true") {
        corpus << "Repetition patterns:\n";
        corpus << "abc abc abc abc abc abc abc abc abc abc\n";
        corpus << "a b c a b c a b c a b c a b c a b c\n";
        corpus << "abcdefghijklmnopqrstuvwxyz\n";
        corpus << "aaaaaaaaaabbbbbbbbbbcccccccccc\n";
    }
    
    return corpus.str();
}

void TokenizationAnalysis::exportResultsToCSV(
    const std::map<std::string, TokenizationAnalysisResult>& results,
    const std::string& output_file) {
    
    std::ofstream csv(output_file);
    if (!csv.is_open()) {
        CV_Error(Error::StsError, "Could not open output file: " + output_file);
    }
    
    // Write header
    csv << "Tokenizer,CompressionRatio,EncodingTimeMs,DecodingTimeMs,TotalTokens,";
    csv << "UniqueTokens,VocabularyCoverage,UnknownTokenPercentage";
    
    // Add headers for common metrics
    bool header_complete = false;
    for (const auto& [name, result] : results) {
        if (!header_complete && !result.metrics.empty()) {
            for (const auto& [metric_name, value] : result.metrics) {
                csv << "," << metric_name;
            }
            header_complete = true;
        }
        break;  // Only need to do this once
    }
    csv << "\n";
    
    // Write data for each tokenizer
    for (const auto& [name, result] : results) {
        csv << name << "," << result.compression_ratio << ",";
        csv << result.encoding_time_ms << "," << result.decoding_time_ms << ",";
        csv << result.total_tokens << "," << result.unique_tokens << ",";
        csv << result.vocabulary_coverage << "," << result.unknown_token_percentage;
        
        // Add metric values
        for (const auto& [metric_name, value] : result.metrics) {
            csv << "," << value;
        }
        csv << "\n";
    }
    
    csv.close();
}

double TokenizationAnalysis::calculateCompressionRatio(const std::string& text, const std::vector<int>& tokens) {
    return static_cast<double>(text.length()) / tokens.size();
}

double TokenizationAnalysis::calculateVocabularyCoverage(const Ptr<Tokenizer>& tokenizer, const std::string& text) {
    // A rough approximation - in a real implementation this would be more sophisticated
    std::vector<int> tokens = tokenizer->encode(text);
    int unknown_count = 0;
    
    for (int token : tokens) {
        std::string token_text = tokenizer->getTokenText(token);
        if (token_text.empty() || token_text == "[UNK]" || token_text == "<unk>") {
            unknown_count++;
        }
    }
    
    return 1.0 - static_cast<double>(unknown_count) / tokens.size();
}

std::map<int, int> TokenizationAnalysis::getTokenFrequencies(const std::vector<int>& tokens) {
    std::map<int, int> frequencies;
    for (int token : tokens) {
        frequencies[token]++;
    }
    return frequencies;
}

void TokenizationAnalysis::generateTokenizationHeatmap(
    const std::string& text, 
    const std::vector<int>& tokens, 
    Mat& output) {
    
    // Create a visualization of how the text is tokenized
    // We'll create a heatmap where each character is colored based on its token
    
    const int char_height = 20;
    const int char_width = 10;
    const int max_chars_per_line = 80;
    
    // Split text into lines for better visualization
    std::vector<std::string> lines;
    std::string current_line;
    
    for (char c : text) {
        if (c == '\n') {
            lines.push_back(current_line);
            current_line = "";
        } else {
            current_line += c;
            if (current_line.length() >= max_chars_per_line) {
                lines.push_back(current_line);
                current_line = "";
            }
        }
    }
    
    if (!current_line.empty()) {
        lines.push_back(current_line);
    }
    
    // Create output image
    const int height = static_cast<int>(lines.size()) * char_height + 50;  // Extra space for title
    const int width = max_chars_per_line * char_width + 20;  // Extra margin
    
    output = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
    
    // Generate random colors for tokens
    std::mt19937 rng(12345);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 255);
    
    std::map<int, Scalar> token_colors;
    
    // Add title
    putText(output, "Tokenization Heatmap", Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
    
    // We need to map tokens back to their positions in text
    // This is a simplified approach - in a real implementation,
    // we would track this during encoding
    
    // For now, just assign alternating colors to visualize token boundaries
    int current_token_idx = 0;
    int chars_in_current_token = 0;
    
    for (size_t line_idx = 0; line_idx < lines.size(); ++line_idx) {
        const std::string& line = lines[line_idx];
        
        for (size_t char_idx = 0; char_idx < line.length(); ++char_idx) {
            // Get color for this character based on its token
            Scalar color;
            
            if (current_token_idx < static_cast<int>(tokens.size())) {
                // Get or generate color for this token
                int token = tokens[current_token_idx];
                if (token_colors.find(token) == token_colors.end()) {
                    // Generate a new color for this token
                    token_colors[token] = Scalar(dist(rng), dist(rng), dist(rng));
                }
                
                color = token_colors[token];
                
                // Move to next token if needed
                chars_in_current_token++;
                if (chars_in_current_token >= 5) {  // Simplified - assume each token is ~5 chars
                    current_token_idx++;
                    chars_in_current_token = 0;
                }
            } else {
                color = Scalar(200, 200, 200);  // Grey for characters beyond tokens
            }
            
            // Draw character box with color
            Point pt1(static_cast<int>(char_idx) * char_width + 10, 
                      static_cast<int>(line_idx) * char_height + 50);
            Point pt2(pt1.x + char_width, pt1.y + char_height);
            
            rectangle(output, pt1, pt2, color, FILLED);
            rectangle(output, pt1, pt2, Scalar(0, 0, 0), 1);
            
            // Draw character
            std::string char_str(1, line[char_idx]);
            putText(output, char_str, Point(pt1.x + 2, pt1.y + char_height - 5),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
        }
    }
}

}} // namespace cv::dnn