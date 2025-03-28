/**
 * @file wordpiece_tokenizer.cpp
 * @brief Implementation of WordPiece tokenizer
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements the WordPiece tokenization algorithm as used in BERT
 * and similar transformer models. It handles subword tokenization with the
 * characteristic "##" prefix for word-internal tokens.
 */

#include "precomp.hpp"
#include "wordpiece_tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>

namespace cv {
namespace dnn {

WordPieceTokenizer::WordPieceTokenizer(const std::string& vocab_file,
                                       const std::string& unk_token,
                                       int max_chars_per_word)
    : unk_token_(unk_token),
      max_chars_per_word_(max_chars_per_word) {
    
    loadVocabulary(vocab_file);
    
    // Initialize special tokens that should not be split
    special_tokens_ = {
        "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]",
        "<s>", "</s>", "<pad>", "<mask>", "<unk>"
    };
    
    // Find unknown token ID
    auto it = token_to_id_.find(unk_token_);
    if (it != token_to_id_.end()) {
        unk_token_id_ = it->second;
    } else {
        // Default to 0 if unk_token not found
        unk_token_id_ = 0;
    }
}

void WordPieceTokenizer::loadVocabulary(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        CV_Error(Error::StsError, "Could not open vocabulary file: " + vocab_file);
    }
    
    // Read vocabulary file (one token per line)
    std::string token;
    int id = 0;
    
    while (std::getline(file, token)) {
        // Trim whitespace
        token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), token.end());
        
        if (!token.empty()) {
            token_to_id_[token] = id;
            id_to_token_[id] = token;
            id++;
        }
    }
    
    if (token_to_id_.empty()) {
        CV_Error(Error::StsError, "Failed to load vocabulary or vocabulary is empty");
    }
}

std::vector<std::string> WordPieceTokenizer::tokenize(const std::string& text) const {
    // Basic tokenization implementation
    // In a full implementation, you would include more sophisticated preprocessing
    
    // Convert to lowercase and split on whitespace
    std::string lowercase_text = text;
    std::transform(lowercase_text.begin(), lowercase_text.end(), lowercase_text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    // Tokenize on whitespace
    std::regex pattern(R"(\s+)");
    std::vector<std::string> tokens;
    
    std::sregex_token_iterator iter(lowercase_text.begin(), lowercase_text.end(), pattern, -1);
    std::sregex_token_iterator end;
    
    for (; iter != end; ++iter) {
        if (!iter->str().empty()) {
            tokens.push_back(iter->str());
        }
    }
    
    // Handle punctuation
    std::vector<std::string> result;
    std::regex punct_pattern(R"([^\w\s])");
    
    for (const auto& token : tokens) {
        // Check if this is a special token that shouldn't be split
        if (std::find(special_tokens_.begin(), special_tokens_.end(), token) != special_tokens_.end()) {
            result.push_back(token);
            continue;
        }
        
        // Split on punctuation
        std::string::const_iterator searchStart = token.begin();
        std::smatch matches;
        bool found_punct = false;
        
        while (std::regex_search(searchStart, token.end(), matches, punct_pattern)) {
            found_punct = true;
            
            // Add the text before the punctuation
            std::string before(token.begin(), matches.prefix().second);
            if (!before.empty()) {
                result.push_back(before);
            }
            
            // Add the punctuation as a separate token
            result.push_back(matches.str());
            
            searchStart = matches.suffix().first;
        }
        
        // Add the remaining text
        if (found_punct) {
            std::string remaining(searchStart, token.end());
            if (!remaining.empty()) {
                result.push_back(remaining);
            }
        } else {
            result.push_back(token);
        }
    }
    
    return result;
}

std::vector<std::string> WordPieceTokenizer::wordpiece_tokenize(const std::string& word) const {
    // Skip wordpiece tokenization for special tokens
    if (std::find(special_tokens_.begin(), special_tokens_.end(), word) != special_tokens_.end()) {
        return {word};
    }
    
    // Skip wordpiece tokenization for very long words
    if (word.length() > static_cast<size_t>(max_chars_per_word_)) {
        return {unk_token_};
    }
    
    // Try to find the word in the vocabulary
    if (token_to_id_.find(word) != token_to_id_.end()) {
        return {word};
    }
    
    // Initialize with the whole word as "unknown"
    std::vector<std::string> result;
    
    // Try to split into wordpieces
    bool is_bad = false;
    int start = 0;
    std::vector<std::string> sub_tokens;
    
    while (start < static_cast<int>(word.length())) {
        int end = static_cast<int>(word.length());
        std::string cur_substr;
        bool found_substr = false;
        
        while (start < end) {
            std::string substr = word.substr(start, end - start);
            
            // Add ## prefix except for the first subtoken
            if (start > 0) {
                substr = "##" + substr;
            }
            
            if (token_to_id_.find(substr) != token_to_id_.end()) {
                cur_substr = substr;
                found_substr = true;
                break;
            }
            
            end--;
        }
        
        if (!found_substr) {
            is_bad = true;
            break;
        }
        
        sub_tokens.push_back(cur_substr);
        start = end;
    }
    
    if (is_bad) {
        return {unk_token_};
    } else {
        return sub_tokens;
    }
}

std::vector<int> WordPieceTokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens) const {
    std::vector<int> ids;
    ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unk_token_id_);
        }
    }
    
    return ids;
}

std::vector<int> WordPieceTokenizer::encode(const std::string& text) const {
    // Basic tokenization
    auto tokens = tokenize(text);
    
    // WordPiece tokenization
    std::vector<std::string> word_pieces;
    
    for (const auto& token : tokens) {
        auto pieces = wordpiece_tokenize(token);
        word_pieces.insert(word_pieces.end(), pieces.begin(), pieces.end());
    }
    
    // Convert tokens to IDs
    return convert_tokens_to_ids(word_pieces);
}

std::string WordPieceTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    
    for (size_t i = 0; i < tokens.size(); i++) {
        auto it = id_to_token_.find(tokens[i]);
        if (it != id_to_token_.end()) {
            std::string token = it->second;
            
            // Remove ## prefix for wordpiece tokens
            if (token.substr(0, 2) == "##") {
                result += token.substr(2);
            } else if (i > 0 && !result.empty() && 
                       // Special handling for punctuation and special tokens
                       std::find(special_tokens_.begin(), special_tokens_.end(), token) == special_tokens_.end() &&
                       token.length() == 1 && std::ispunct(token[0])) {
                // Append punctuation without space
                result += token;
            } else if (i > 0) {
                // Add space before regular tokens
                result += " " + token;
            } else {
                // First token
                result += token;
            }
        } else {
            // Unknown token ID
            if (i > 0) {
                result += " " + unk_token_;
            } else {
                result += unk_token_;
            }
        }
    }
    
    return result;
}

size_t WordPieceTokenizer::getVocabSize() const {
    return token_to_id_.size();
}

void WordPieceTokenizer::save(const std::string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    
    fs << "tokenizer_type" << "wordpiece";
    
    // Save token_to_id map
    fs << "vocabulary" << "{";
    for (const auto& pair : token_to_id_) {
        fs << pair.first << pair.second;
    }
    fs << "}";
    
    // Save configuration
    fs << "unk_token" << unk_token_;
    fs << "max_chars_per_word" << max_chars_per_word_;
    
    // Save special tokens
    fs << "special_tokens" << "[";
    for (const auto& token : special_tokens_) {
        fs << token;
    }
    fs << "]";
}

Ptr<Tokenizer> createWordPieceTokenizer(const std::string& vocab_file,
                                       const std::string& unk_token,
                                       int max_chars_per_word) {
    return makePtr<WordPieceTokenizer>(vocab_file, unk_token, max_chars_per_word);
}

}} // namespace cv::dnn