/**
 * @file adaptive_tokenizer.cpp
 * @brief Implementation of adaptive tokenization strategies for multilingual text
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements the AdaptiveTokenizer class which dynamically selects the most
 * appropriate tokenization algorithm based on text language and script characteristics.
 * It supports BPE, Unigram, and character-based tokenization strategies.
 */

#include "precomp.hpp"
#include "adaptive_tokenizer.hpp"
#include "bpe_tokenizer.hpp"
#include "unigram_tokenizer.hpp"
#include "wordpiece_tokenizer.hpp"

#include <algorithm>
#include <codecvt>
#include <locale>
#include <regex>
#include <unordered_set>
#include <map>
#include <unicode/uchar.h>
#include <unicode/uscript.h>

namespace cv {
namespace dnn {

// Script categories for tokenization strategies
static const std::unordered_set<std::string> IDEOGRAPHIC_SCRIPTS = {
    "Han", "Hiragana", "Katakana", "Hangul"
};

static const std::unordered_set<std::string> SYLLABIC_SCRIPTS = {
    "Devanagari", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam", 
    "Thai", "Lao", "Khmer", "Myanmar"
};

static const std::map<std::string, ScriptRules> DEFAULT_SCRIPT_RULES = {
    {"Latin", ScriptRules{"Latin", true, true, ""}},
    {"Cyrillic", ScriptRules{"Cyrillic", true, true, ""}},
    {"Greek", ScriptRules{"Greek", true, true, ""}},
    {"Arabic", ScriptRules{"Arabic", true, true, ""}},
    {"Hebrew", ScriptRules{"Hebrew", true, true, ""}},
    {"Han", ScriptRules{"Han", false, false, ""}},
    {"Hiragana", ScriptRules{"Hiragana", false, true, ""}},
    {"Katakana", ScriptRules{"Katakana", false, true, ""}},
    {"Hangul", ScriptRules{"Hangul", false, true, ""}},
    {"Devanagari", ScriptRules{"Devanagari", false, true, ""}},
    {"Bengali", ScriptRules{"Bengali", false, true, ""}},
    {"Tamil", ScriptRules{"Tamil", false, true, ""}},
    {"Thai", ScriptRules{"Thai", false, false, ""}}
};

AdaptiveTokenizer::AdaptiveTokenizer(
    const Ptr<Tokenizer>& bpe_tokenizer,
    const Ptr<Tokenizer>& unigram_tokenizer,
    const Ptr<Tokenizer>& char_tokenizer)
    : bpe_tokenizer_(bpe_tokenizer),
      unigram_tokenizer_(unigram_tokenizer),
      char_tokenizer_(char_tokenizer),
      script_rules_(DEFAULT_SCRIPT_RULES) {
    
    // Validate that required tokenizers are provided
    if (!bpe_tokenizer_) {
        CV_Error(Error::StsNullPtr, "BPE tokenizer must be provided");
    }
    
    if (!unigram_tokenizer_) {
        CV_Error(Error::StsNullPtr, "Unigram tokenizer must be provided");
    }
    
    // If character tokenizer not provided, use unigram tokenizer
    if (!char_tokenizer_) {
        char_tokenizer_ = unigram_tokenizer_;
    }
}

std::vector<int> AdaptiveTokenizer::encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }
    
    // Reset metrics
    last_metrics_.clear();
    
    // Detect language and script
    LanguageInfo lang_info = detectLanguage(text);
    
    // Initialize token counts for metrics
    size_t total_tokens = 0;
    std::map<std::string, int> script_token_counts;
    
    // Start timer for performance metrics
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Split text into script segments
    std::vector<std::string> segments = splitByScript(text);
    
    // Encode each segment with the appropriate tokenizer
    std::vector<int> result;
    size_t chars_encoded = 0;
    
    for (const auto& segment : segments) {
        if (segment.empty()) {
            continue;
        }
        
        chars_encoded += segment.length();
        
        // Detect script of this segment
        char32_t first_char = utf8ToCodepoint(segment.substr(0, 4)); // Get first codepoint
        std::string script = getScriptName(first_char);
        
        // Get appropriate tokenizer for this script
        Ptr<Tokenizer> segment_tokenizer;
        
        // Check if we have a specific tokenizer for this script
        auto it = script_tokenizers_.find(script);
        if (it != script_tokenizers_.end()) {
            segment_tokenizer = it->second;
        } else if (isIdeographic(script)) {
            // Use character-based tokenization for ideographic scripts
            segment_tokenizer = char_tokenizer_;
        } else if (isSyllabic(script)) {
            // Use unigram tokenization for syllabic scripts
            segment_tokenizer = unigram_tokenizer_;
        } else {
            // Default to BPE for alphabetic scripts
            segment_tokenizer = bpe_tokenizer_;
        }
        
        // Encode segment
        std::vector<int> segment_tokens = segment_tokenizer->encode(segment);
        
        // Track token counts for metrics
        total_tokens += segment_tokens.size();
        script_token_counts[script] += segment_tokens.size();
        
        // Add to result
        result.insert(result.end(), segment_tokens.begin(), segment_tokens.end());
    }
    
    // Calculate execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Calculate metrics
    last_metrics_["total_tokens"] = static_cast<double>(total_tokens);
    last_metrics_["chars_per_token"] = static_cast<double>(chars_encoded) / total_tokens;
    last_metrics_["tokens_per_char"] = static_cast<double>(total_tokens) / chars_encoded;
    last_metrics_["encoding_time_us"] = static_cast<double>(duration.count());
    
    // Script distribution metrics
    for (const auto& [script, count] : script_token_counts) {
        std::string metric_name = "script_" + script + "_percentage";
        last_metrics_[metric_name] = static_cast<double>(count) * 100.0 / total_tokens;
    }
    
    return result;
}

std::vector<TokenInfo> AdaptiveTokenizer::encodeWithInfo(const std::string& text) const {
    if (text.empty()) {
        return {};
    }
    
    // Get token IDs first
    std::vector<int> tokens = encode(text);
    
    // Now reconstruct token info
    std::vector<TokenInfo> result;
    result.reserve(tokens.size());
    
    // We need to do a more complex analysis to properly track token positions
    // This is a simplified implementation
    
    size_t pos = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        TokenInfo info;
        info.id = tokens[i];
        info.text = getTokenText(tokens[i]);
        info.start = pos;
        
        // Skip empty tokens
        if (info.text.empty()) {
            info.end = pos;
            info.score = 0.0f;
            result.push_back(info);
            continue;
        }
        
        // Determine end position
        size_t len = info.text.length();
        if (pos + len <= text.length() && text.substr(pos, len) == info.text) {
            // Direct match - easy case
            pos += len;
        } else {
            // For complex cases (subword merges, etc.), use approximate positioning
            // In a real implementation, we'd track this more carefully during encoding
            pos = std::min(pos + len, text.length());
        }
        
        info.end = pos;
        
        // For score, use a placeholder - in a real implementation, we'd use
        // probabilities from the appropriate tokenizer model
        info.score = 1.0f;
        
        result.push_back(info);
    }
    
    return result;
}

std::string AdaptiveTokenizer::decode(const std::vector<int>& tokens) const {
    if (tokens.empty()) {
        return "";
    }
    
    // For decoding, we can use a simpler approach since we're just concatenating tokens
    // We'll use the union of all tokenizer vocabularies
    
    std::string result;
    for (int token : tokens) {
        // Try each tokenizer in order
        std::string token_text;
        
        // Check BPE tokenizer
        token_text = bpe_tokenizer_->getTokenText(token);
        if (!token_text.empty()) {
            result += token_text;
            continue;
        }
        
        // Check Unigram tokenizer
        token_text = unigram_tokenizer_->getTokenText(token);
        if (!token_text.empty()) {
            result += token_text;
            continue;
        }
        
        // Check Character tokenizer (if different from Unigram)
        if (char_tokenizer_ != unigram_tokenizer_) {
            token_text = char_tokenizer_->getTokenText(token);
            if (!token_text.empty()) {
                result += token_text;
                continue;
            }
        }
        
        // Check script-specific tokenizers
        bool found = false;
        for (const auto& [script, tokenizer] : script_tokenizers_) {
            token_text = tokenizer->getTokenText(token);
            if (!token_text.empty()) {
                result += token_text;
                found = true;
                break;
            }
        }
        
        // If we couldn't find the token, add a placeholder
        if (!found) {
            result += "[UNK]";
        }
    }
    
    return result;
}

size_t AdaptiveTokenizer::getVocabSize() const {
    // Calculate total vocabulary size across all tokenizers
    // Note: This may count duplicates if tokens appear in multiple tokenizers
    
    size_t total_size = 0;
    
    // Add each component tokenizer's vocabulary
    total_size += bpe_tokenizer_->getVocabSize();
    total_size += unigram_tokenizer_->getVocabSize();
    
    if (char_tokenizer_ != unigram_tokenizer_) {
        total_size += char_tokenizer_->getVocabSize();
    }
    
    // Add script-specific tokenizers
    for (const auto& [script, tokenizer] : script_tokenizers_) {
        total_size += tokenizer->getVocabSize();
    }
    
    return total_size;
}

void AdaptiveTokenizer::save(const std::string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    
    fs << "tokenizer_type" << "adaptive";
    
    // Save paths to component tokenizers
    // In a real implementation, we'd save each tokenizer to a subdirectory
    // and then reference them here
    
    // For now, we'll just save a placeholder
    fs << "component_tokenizers" << "{";
    fs << "bpe" << "bpe_tokenizer.yml";
    fs << "unigram" << "unigram_tokenizer.yml";
    if (char_tokenizer_ != unigram_tokenizer_) {
        fs << "char" << "char_tokenizer.yml";
    }
    fs << "}";
    
    // Save script-specific tokenizers
    fs << "script_tokenizers" << "{";
    for (const auto& [script, tokenizer] : script_tokenizers_) {
        fs << script << script + "_tokenizer.yml";
    }
    fs << "}";
    
    // Save script rules
    fs << "script_rules" << "{";
    for (const auto& [script, rules] : script_rules_) {
        fs << script << "{";
        fs << "name" << rules.name;
        fs << "wordBoundaries" << rules.wordBoundaries;
        fs << "processAsWords" << rules.processAsWords;
        fs << "specialCharacters" << rules.specialCharacters;
        fs << "}";
    }
    fs << "}";
}

LanguageInfo AdaptiveTokenizer::detectLanguage(const std::string& text) const {
    // Check cache first
    auto cache_it = language_cache_.find(text);
    if (cache_it != language_cache_.end()) {
        return cache_it->second;
    }
    
    // Simple language detection based on script prevalence
    std::map<std::string, int> script_counts;
    
    // Convert to codepoints and count scripts
    std::vector<char32_t> codepoints = utf8ToCodepoints(text);
    
    for (char32_t cp : codepoints) {
        std::string script = getScriptName(cp);
        script_counts[script]++;
    }
    
    // Find the dominant script
    std::string dominant_script;
    int max_count = 0;
    
    for (const auto& [script, count] : script_counts) {
        if (count > max_count) {
            max_count = count;
            dominant_script = script;
        }
    }
    
    // Map script to language (simplified approach)
    std::string language;
    float confidence = 0.0f;
    
    if (dominant_script == "Latin") {
        // For Latin script, we'd need more sophisticated analysis to determine
        // if it's English, Spanish, French, etc.
        // For now, default to English
        language = "en";
        
        // Calculate confidence based on proportion of Latin characters
        confidence = static_cast<float>(max_count) / codepoints.size();
    } else if (dominant_script == "Cyrillic") {
        language = "ru";  // Default to Russian for Cyrillic
        confidence = static_cast<float>(max_count) / codepoints.size();
    } else if (dominant_script == "Han") {
        // Could be Chinese, Japanese, or Korean
        // Check for presence of Hiragana/Katakana (Japanese) or Hangul (Korean)
        if (script_counts.find("Hiragana") != script_counts.end() || 
            script_counts.find("Katakana") != script_counts.end()) {
            language = "ja";
        } else if (script_counts.find("Hangul") != script_counts.end()) {
            language = "ko";
        } else {
            language = "zh";  // Default to Chinese
        }
        confidence = static_cast<float>(max_count) / codepoints.size();
    } else if (dominant_script == "Devanagari") {
        language = "hi";  // Hindi
        confidence = static_cast<float>(max_count) / codepoints.size();
    } else if (dominant_script == "Arabic") {
        language = "ar";
        confidence = static_cast<float>(max_count) / codepoints.size();
    } else if (dominant_script == "Bengali") {
        language = "bn";
        confidence = static_cast<float>(max_count) / codepoints.size();
    } else if (dominant_script == "Thai") {
        language = "th";
        confidence = static_cast<float>(max_count) / codepoints.size();
    } else {
        // Default to unknown
        language = "unknown";
        confidence = 0.5f;  // Medium confidence
    }
    
    // Create result
    LanguageInfo result;
    result.language = language;
    result.script = dominant_script;
    result.confidence = confidence;
    
    // Cache result
    language_cache_[text] = result;
    
    return result;
}

std::string AdaptiveTokenizer::getTokenText(int tokenId) const {
    // Try each tokenizer in order to find the token
    
    // Check BPE tokenizer
    std::string token_text = bpe_tokenizer_->getTokenText(tokenId);
    if (!token_text.empty()) {
        return token_text;
    }
    
    // Check Unigram tokenizer
    token_text = unigram_tokenizer_->getTokenText(tokenId);
    if (!token_text.empty()) {
        return token_text;
    }
    
    // Check Character tokenizer (if different from Unigram)
    if (char_tokenizer_ != unigram_tokenizer_) {
        token_text = char_tokenizer_->getTokenText(tokenId);
        if (!token_text.empty()) {
            return token_text;
        }
    }
    
    // Check script-specific tokenizers
    for (const auto& [script, tokenizer] : script_tokenizers_) {
        token_text = tokenizer->getTokenText(tokenId);
        if (!token_text.empty()) {
            return token_text;
        }
    }
    
    // Token not found
    return "";
}

int AdaptiveTokenizer::getTokenId(const std::string& tokenText) const {
    // Try each tokenizer in order to find the token
    
    // Check BPE tokenizer
    int token_id = bpe_tokenizer_->getTokenId(tokenText);
    if (token_id >= 0) {
        return token_id;
    }
    
    // Check Unigram tokenizer
    token_id = unigram_tokenizer_->getTokenId(tokenText);
    if (token_id >= 0) {
        return token_id;
    }
    
    // Check Character tokenizer (if different from Unigram)
    if (char_tokenizer_ != unigram_tokenizer_) {
        token_id = char_tokenizer_->getTokenId(tokenText);
        if (token_id >= 0) {
            return token_id;
        }
    }
    
    // Check script-specific tokenizers
    for (const auto& [script, tokenizer] : script_tokenizers_) {
        token_id = tokenizer->getTokenId(tokenText);
        if (token_id >= 0) {
            return token_id;
        }
    }
    
    // Token not found
    return -1;
}

std::map<std::string, double> AdaptiveTokenizer::getTokenizationMetrics() const {
    return last_metrics_;
}

void AdaptiveTokenizer::setScriptRules(const std::map<std::string, ScriptRules>& rules) {
    script_rules_ = rules;
}

void AdaptiveTokenizer::registerScriptTokenizer(const std::string& script, const Ptr<Tokenizer>& tokenizer) {
    if (!tokenizer) {
        CV_Error(Error::StsNullPtr, "Cannot register null tokenizer");
    }
    
    script_tokenizers_[script] = tokenizer;
}

std::vector<std::string> AdaptiveTokenizer::splitByScript(const std::string& text) const {
    if (text.empty()) {
        return {};
    }
    
    std::vector<std::string> segments;
    
    // Convert to codepoints for easier script detection
    std::vector<char32_t> codepoints = utf8ToCodepoints(text);
    
    // Handle empty text
    if (codepoints.empty()) {
        return {};
    }
    
    // Initialize with first codepoint
    std::string current_script = getScriptName(codepoints[0]);
    std::string current_segment;
    current_segment += codepointToUtf8(codepoints[0]);
    
    // Process remaining codepoints
    for (size_t i = 1; i < codepoints.size(); ++i) {
        std::string script = getScriptName(codepoints[i]);
        
        // Handle whitespace and punctuation - keep with the current segment
        if (script == "Common" || script == "Inherited") {
            current_segment += codepointToUtf8(codepoints[i]);
            continue;
        }
        
        // Check if script changes
        if (script != current_script && current_script != "Common" && current_script != "Inherited") {
            // Save current segment
            if (!current_segment.empty()) {
                segments.push_back(current_segment);
                current_segment.clear();
            }
            current_script = script;
        }
        
        // Add codepoint to current segment
        current_segment += codepointToUtf8(codepoints[i]);
    }
    
    // Add the last segment
    if (!current_segment.empty()) {
        segments.push_back(current_segment);
    }
    
    return segments;
}

std::string AdaptiveTokenizer::getScriptName(char32_t codepoint) const {
    // Use ICU to get script name
    UErrorCode errorCode = U_ZERO_ERROR;
    UScriptCode scriptCode = uscript_getScript(codepoint, &errorCode);
    
    if (U_FAILURE(errorCode)) {
        return "Unknown";
    }
    
    const char* script_name = uscript_getName(scriptCode);
    return script_name ? script_name : "Unknown";
}

bool AdaptiveTokenizer::isIdeographic(const std::string& script) const {
    return IDEOGRAPHIC_SCRIPTS.find(script) != IDEOGRAPHIC_SCRIPTS.end();
}

bool AdaptiveTokenizer::isSyllabic(const std::string& script) const {
    return SYLLABIC_SCRIPTS.find(script) != SYLLABIC_SCRIPTS.end();
}

char32_t AdaptiveTokenizer::utf8ToCodepoint(const std::string& utf8_char) const {
    // Simple UTF-8 to codepoint conversion
    if (utf8_char.empty()) {
        return 0;
    }
    
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(utf8_char.data());
    
    // Single byte (ASCII)
    if (bytes[0] < 0x80) {
        return bytes[0];
    }
    
    // 2-byte sequence
    if ((bytes[0] & 0xE0) == 0xC0 && utf8_char.length() >= 2) {
        return ((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F);
    }
    
    // 3-byte sequence
    if ((bytes[0] & 0xF0) == 0xE0 && utf8_char.length() >= 3) {
        return ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) | (bytes[2] & 0x3F);
    }
    
    // 4-byte sequence
    if ((bytes[0] & 0xF8) == 0xF0 && utf8_char.length() >= 4) {
        return ((bytes[0] & 0x07) << 18) | ((bytes[1] & 0x3F) << 12) | 
               ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
    }
    
    // Invalid UTF-8
    return 0xFFFD;  // Unicode replacement character
}

std::vector<char32_t> AdaptiveTokenizer::utf8ToCodepoints(const std::string& text) const {
    std::vector<char32_t> codepoints;
    
    // Reserve space for efficiency
    codepoints.reserve(text.length());
    
    for (size_t i = 0; i < text.length();) {
        // Get codepoint and its UTF-8 length
        char32_t cp = 0;
        size_t len = 0;
        
        unsigned char first_byte = static_cast<unsigned char>(text[i]);
        
        if (first_byte < 0x80) {
            // Single byte (ASCII)
            cp = first_byte;
            len = 1;
        } else if ((first_byte & 0xE0) == 0xC0 && i + 1 < text.length()) {
            // 2-byte sequence
            cp = ((first_byte & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            len = 2;
        } else if ((first_byte & 0xF0) == 0xE0 && i + 2 < text.length()) {
            // 3-byte sequence
            cp = ((first_byte & 0x0F) << 12) | 
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) | 
                 (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            len = 3;
        } else if ((first_byte & 0xF8) == 0xF0 && i + 3 < text.length()) {
            // 4-byte sequence
            cp = ((first_byte & 0x07) << 18) | 
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) | 
                 (static_cast<unsigned char>(text[i + 3]) & 0x3F);
            len = 4;
        } else {
            // Invalid UTF-8, use replacement character
            cp = 0xFFFD;
            len = 1;
        }
        
        codepoints.push_back(cp);
        i += len;
    }
    
    return codepoints;
}

std::string AdaptiveTokenizer::codepointToUtf8(char32_t codepoint) const {
    std::string result;
    
    // ASCII
    if (codepoint < 0x80) {
        result += static_cast<char>(codepoint);
    }
    // 2-byte sequence
    else if (codepoint < 0x800) {
        result += static_cast<char>(0xC0 | (codepoint >> 6));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    // 3-byte sequence
    else if (codepoint < 0x10000) {
        result += static_cast<char>(0xE0 | (codepoint >> 12));
        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    // 4-byte sequence
    else if (codepoint < 0x110000) {
        result += static_cast<char>(0xF0 | (codepoint >> 18));
        result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    // Invalid codepoint
    else {
        // Use replacement character
        result += "\xEF\xBF\xBD";
    }
    
    return result;
}

Ptr<Tokenizer> createAdaptiveTokenizer(
    const std::string& bpe_vocab_file,
    const std::string& bpe_merges_file,
    const std::string& unigram_vocab_file) {
    
    // Create component tokenizers
    Ptr<Tokenizer> bpe_tokenizer = createBPETokenizer(bpe_vocab_file, bpe_merges_file);
    Ptr<Tokenizer> unigram_tokenizer = createUnigramTokenizer(unigram_vocab_file);
    
    // Create adaptive tokenizer
    return makePtr<AdaptiveTokenizer>(bpe_tokenizer, unigram_tokenizer);
}

}} // namespace cv::dnn