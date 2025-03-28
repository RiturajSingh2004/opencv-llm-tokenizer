/**
 * @file unigram_tokenizer.cpp
 * @brief Implementation of Unigram language model tokenizer
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements the Unigram language model tokenizer as used
 * in SentencePiece and similar systems. It supports subword segmentation
 * based on a unigram language model with probability-based tokenization.
 */

#include "precomp.hpp"
#include "unigram_tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <queue>
#include <limits>
#include <random>
#include <cmath>

namespace cv {
namespace dnn {

UnigramTokenizer::UnigramTokenizer(const std::string& vocab_file,
                                   int unk_id,
                                   const std::string& unk_piece,
                                   float score_threshold,
                                   float sample_alpha)
    : unk_id_(unk_id),
      unk_piece_(unk_piece),
      enable_sampling_(sample_alpha > 0.0f),
      score_threshold_(score_threshold),
      sample_alpha_(sample_alpha) {
    
    loadVocabulary(vocab_file);
}

void UnigramTokenizer::loadVocabulary(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        CV_Error(Error::StsError, "Could not open vocabulary file: " + vocab_file);
    }
    
    std::string line;
    int id = 0;
    
    // Format: token\tscore
    while (std::getline(file, line)) {
        size_t pos = line.find('\t');
        if (pos == std::string::npos) {
            CV_Error(Error::StsError, "Invalid vocabulary format. Expected 'token\\tscore' format.");
        }
        
        std::string token = line.substr(0, pos);
        float score = std::stof(line.substr(pos + 1));
        
        // Store token and score
        token_to_id_[token] = id;
        id_to_token_[id] = token;
        scores_[id] = score;
        
        id++;
    }
    
    // Verify unk_id is valid
    if (id_to_token_.find(unk_id_) == id_to_token_.end()) {
        CV_Error(Error::StsError, "Unknown token ID is not in vocabulary.");
    }
    
    // Verify unk_piece matches
    if (id_to_token_[unk_id_] != unk_piece_) {
        CV_Warning(0, "Unknown token string does not match the one in vocabulary.");
        // Update unk_piece_ to match
        unk_piece_ = id_to_token_[unk_id_];
    }
    
    if (token_to_id_.empty()) {
        CV_Error(Error::StsError, "Failed to load vocabulary or vocabulary is empty");
    }
}

std::vector<std::string> UnigramTokenizer::viterbi_segment(const std::string& text) const {
    if (text.empty()) {
        return {};
    }
    
    const size_t len = text.size();
    
    // best_score[i] = best score for segmenting text[0:i]
    std::vector<float> best_score(len + 1, -std::numeric_limits<float>::max());
    // best_index[i] = starting index of last token in best segmentation of text[0:i]
    std::vector<int> best_index(len + 1, -1);
    
    best_score[0] = 0.0;  // Empty string has a score of 0
    
    // Dynamic programming to find best segmentation
    for (size_t i = 0; i < len; ++i) {
        if (best_score[i] == -std::numeric_limits<float>::max()) {
            continue;  // Unreachable position
        }
        
        // Consider subwords ending at position 'end'
        for (size_t end = i + 1; end <= len; ++end) {
            std::string subword = text.substr(i, end - i);
            
            // Check if subword is in vocabulary
            float score = -std::numeric_limits<float>::max();
            auto it = token_to_id_.find(subword);
            if (it != token_to_id_.end()) {
                score = scores_.at(it->second);
            } else if (unk_id_ >= 0) {
                // Use unknown token but with penalty
                score = scores_.at(unk_id_) - 8.0;  // Penalty factor
            }
            
            // Update best score if this is better
            if (best_score[i] + score > best_score[end]) {
                best_score[end] = best_score[i] + score;
                best_index[end] = i;
            }
        }
    }
    
    // Backtrack to get the tokens
    std::vector<std::string> tokens;
    size_t pos = len;
    while (pos > 0) {
        size_t start = best_index[pos];
        if (start < 0) {
            // If we can't backtrack (shouldn't happen with proper unknown handling)
            // Treat the remaining text as unknown
            tokens.push_back(unk_piece_);
            break;
        }
        
        std::string subword = text.substr(start, pos - start);
        if (token_to_id_.find(subword) == token_to_id_.end()) {
            subword = unk_piece_;
        }
        
        tokens.push_back(subword);
        pos = start;
    }
    
    // Tokens are in reverse order
    std::reverse(tokens.begin(), tokens.end());
    
    return tokens;
}

std::vector<std::pair<float, std::vector<std::string>>> UnigramTokenizer::calculate_lattice(const std::string& text) const {
    if (text.empty()) {
        return {};
    }
    
    struct SegmentationPath {
        float score;
        std::vector<std::string> tokens;
        
        // For priority queue (highest score first)
        bool operator<(const SegmentationPath& other) const {
            return score < other.score;
        }
    };
    
    // Find the top-k segmentations
    std::priority_queue<SegmentationPath> paths;
    
    // Initialize with the best segmentation from Viterbi
    std::vector<std::string> best_tokens = viterbi_segment(text);
    float best_score = 0.0;
    for (const auto& token : best_tokens) {
        auto it = token_to_id_.find(token);
        best_score += (it != token_to_id_.end()) ? scores_.at(it->second) : scores_.at(unk_id_);
    }
    
    paths.push({best_score, best_tokens});
    
    // If we're not sampling, just return the best segmentation
    if (!enable_sampling_ || sample_alpha_ <= 0.0f) {
        std::vector<std::pair<float, std::vector<std::string>>> result;
        result.push_back({best_score, best_tokens});
        return result;
    }
    
    // Define sampling parameters
    const int kMaxSamplePathSize = 100;
    std::vector<std::pair<float, std::vector<std::string>>> lattice;
    
    // Produce diverse segmentations by perturbing the path
    std::mt19937 gen(std::random_device{}());
    
    while (!paths.empty() && static_cast<int>(lattice.size()) < kMaxSamplePathSize) {
        SegmentationPath path = paths.top();
        paths.pop();
        
        lattice.push_back({path.score, path.tokens});
        
        // Stop if score is too low compared to the best
        if (path.score < best_score - score_threshold_) {
            break;
        }
        
        // Perturb the segmentation by trying to merge or split tokens
        for (size_t i = 0; i < path.tokens.size() - 1; ++i) {
            // Try to merge two consecutive tokens
            std::string merged = path.tokens[i] + path.tokens[i + 1];
            auto it_merged = token_to_id_.find(merged);
            
            if (it_merged != token_to_id_.end()) {
                std::vector<std::string> new_tokens = path.tokens;
                new_tokens.erase(new_tokens.begin() + i, new_tokens.begin() + i + 2);
                new_tokens.insert(new_tokens.begin() + i, merged);
                
                float new_score = 0.0;
                for (const auto& token : new_tokens) {
                    auto it = token_to_id_.find(token);
                    new_score += (it != token_to_id_.end()) ? scores_.at(it->second) : scores_.at(unk_id_);
                }
                
                if (new_score > path.score - score_threshold_) {
                    paths.push({new_score, new_tokens});
                }
            }
            
            // Try to split a token if it's longer than 1 character
            if (path.tokens[i].size() > 1) {
                for (size_t j = 1; j < path.tokens[i].size(); ++j) {
                    std::string first = path.tokens[i].substr(0, j);
                    std::string second = path.tokens[i].substr(j);
                    
                    auto it_first = token_to_id_.find(first);
                    auto it_second = token_to_id_.find(second);
                    
                    if (it_first != token_to_id_.end() && it_second != token_to_id_.end()) {
                        std::vector<std::string> new_tokens = path.tokens;
                        new_tokens.erase(new_tokens.begin() + i);
                        new_tokens.insert(new_tokens.begin() + i, second);
                        new_tokens.insert(new_tokens.begin() + i, first);
                        
                        float new_score = 0.0;
                        for (const auto& token : new_tokens) {
                            auto it = token_to_id_.find(token);
                            new_score += (it != token_to_id_.end()) ? scores_.at(it->second) : scores_.at(unk_id_);
                        }
                        
                        if (new_score > path.score - score_threshold_) {
                            paths.push({new_score, new_tokens});
                        }
                    }
                }
            }
        }
    }
    
    return lattice;
}

std::vector<std::string> UnigramTokenizer::tokenize(const std::string& text) const {
    // Basic preprocessing - in a full implementation, this would be more sophisticated
    std::string processed_text = preprocess_text(text);
    
    // If sampling is enabled, generate multiple segmentations and sample
    if (enable_sampling_ && sample_alpha_ > 0.0f) {
        auto lattice = calculate_lattice(processed_text);
        
        if (lattice.empty()) {
            return {};
        }
        
        // If there's only one segmentation, return it
        if (lattice.size() == 1) {
            return lattice[0].second;
        }
        
        // Sample a segmentation based on scores
        std::vector<float> probs;
        float max_score = lattice[0].first;  // Highest score
        
        // Calculate sampling probabilities using temperature (alpha)
        for (const auto& entry : lattice) {
            float score = entry.first;
            probs.push_back(std::exp((score - max_score) / sample_alpha_));
        }
        
        // Normalize to get a proper distribution
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (auto& p : probs) {
            p /= sum;
        }
        
        // Sample a segmentation
        std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        
        return lattice[dist(gen)].second;
    } else {
        // Otherwise, use Viterbi algorithm to find best segmentation
        return viterbi_segment(processed_text);
    }
}

std::string UnigramTokenizer::preprocess_text(const std::string& text) const {
    // Simple preprocessing - this would be more sophisticated in a full implementation
    // depending on the tokenizer configuration
    
    // For now, just perform simple normalization
    std::string result = text;
    
    // Convert to lowercase (optional, depending on vocabulary)
    // std::transform(result.begin(), result.end(), result.begin(),
    //               [](unsigned char c) { return std::tolower(c); });
    
    // Normalize whitespace
    std::regex whitespace_re("\\s+");
    result = std::regex_replace(result, whitespace_re, " ");
    
    return result;
}

std::vector<int> UnigramTokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens) const {
    std::vector<int> ids;
    ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unk_id_);
        }
    }
    
    return ids;
}

std::vector<int> UnigramTokenizer::encode(const std::string& text) const {
    // Tokenize the text
    auto tokens = tokenize(text);
    
    // Convert tokens to IDs
    return convert_tokens_to_ids(tokens);
}

std::vector<TokenInfo> UnigramTokenizer::encodeWithInfo(const std::string& text) const {
    // Tokenize and get detailed information
    auto tokens = tokenize(text);
    auto ids = convert_tokens_to_ids(tokens);
    
    std::vector<TokenInfo> result;
    result.reserve(tokens.size());
    
    size_t pos = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        TokenInfo info;
        info.id = ids[i];
        info.text = tokens[i];
        info.start = pos;
        pos += tokens[i].length();
        info.end = pos;
        
        // Calculate token score/probability
        if (info.id >= 0) {
            auto it = scores_.find(info.id);
            info.score = (it != scores_.end()) ? std::exp(it->second) : 0.0f;
        } else {
            info.score = 0.0f;
        }
        
        result.push_back(info);
    }
    
    return result;
}

std::string UnigramTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = id_to_token_.find(tokens[i]);
        if (it != id_to_token_.end()) {
            result += it->second;
        } else {
            result += unk_piece_;
        }
    }
    
    return result;
}

size_t UnigramTokenizer::getVocabSize() const {
    return token_to_id_.size();
}

void UnigramTokenizer::save(const std::string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    
    fs << "tokenizer_type" << "unigram";
    
    // Save token_to_id map and scores
    fs << "vocabulary" << "{";
    for (const auto& pair : token_to_id_) {
        fs << pair.first << pair.second;
    }
    fs << "}";
    
    // Save scores
    fs << "scores" << "{";
    for (const auto& pair : scores_) {
        fs << std::to_string(pair.first) << pair.second;
    }
    fs << "}";
    
    // Save configuration
    fs << "unk_id" << unk_id_;
    fs << "unk_piece" << unk_piece_;
    fs << "score_threshold" << score_threshold_;
    fs << "sample_alpha" << sample_alpha_;
    fs << "enable_sampling" << enable_sampling_;
}

std::string UnigramTokenizer::getTokenText(int tokenId) const {
    auto it = id_to_token_.find(tokenId);
    return (it != id_to_token_.end()) ? it->second : "";
}

int UnigramTokenizer::getTokenId(const std::string& tokenText) const {
    auto it = token_to_id_.find(tokenText);
    return (it != token_to_id_.end()) ? it->second : -1;
}

void UnigramTokenizer::setSampling(bool enable, float alpha) {
    enable_sampling_ = enable;
    if (alpha > 0.0f) {
        sample_alpha_ = alpha;
    }
}

std::unordered_map<int, float> UnigramTokenizer::getScores() const {
    return scores_;
}

Ptr<Tokenizer> createUnigramTokenizer(const std::string& vocab_file,
                                     int unk_id,
                                     const std::string& unk_piece,
                                     float score_threshold,
                                     float sample_alpha) {
    return makePtr<UnigramTokenizer>(vocab_file, unk_id, unk_piece, score_threshold, sample_alpha);
}

}} // namespace cv::dnn