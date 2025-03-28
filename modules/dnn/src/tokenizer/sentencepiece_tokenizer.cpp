/**
 * @file sentencepiece_tokenizer.cpp
 * @brief Implementation of SentencePiece tokenizer
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements the SentencePiece tokenizer wrapper for OpenCV,
 * providing integration with the SentencePiece library developed by Google.
 * It supports all SentencePiece model types including Unigram, BPE, char, and word.
 */

#include "precomp.hpp"
#include "sentencepiece_tokenizer.hpp"

// Include SentencePiece library
#include <sentencepiece_processor.h>

namespace cv {
namespace dnn {

// Wrapper class to hide SentencePiece implementation details
class SPProcessorWrapper {
public:
    SPProcessorWrapper(const std::string& model_path) {
        if (!processor_.Load(model_path).ok()) {
            CV_Error(Error::StsError, "Failed to load SentencePiece model from: " + model_path);
        }
    }
    
    std::vector<int> Encode(const std::string& text) const {
        std::vector<int> ids;
        processor_.Encode(text, &ids);
        return ids;
    }
    
    std::vector<std::pair<int, std::string>> EncodeWithPieces(const std::string& text) const {
        std::vector<std::string> pieces;
        std::vector<int> ids;
        processor_.Encode(text, &ids, &pieces);
        
        std::vector<std::pair<int, std::string>> result;
        for (size_t i = 0; i < ids.size(); i++) {
            result.push_back(std::make_pair(ids[i], pieces[i]));
        }
        return result;
    }
    
    std::vector<int> GetIds(const std::vector<std::string>& pieces) const {
        std::vector<int> ids;
        processor_.PieceToIds(pieces, &ids);
        return ids;
    }
    
    std::vector<std::string> GetPieces(const std::vector<int>& ids) const {
        std::vector<std::string> pieces;
        for (int id : ids) {
            pieces.push_back(processor_.IdToPiece(id));
        }
        return pieces;
    }
    
    std::string Decode(const std::vector<int>& ids) const {
        std::string text;
        processor_.Decode(ids, &text);
        return text;
    }
    
    std::string IdToPiece(int id) const {
        return processor_.IdToPiece(id);
    }
    
    int PieceToId(const std::string& piece) const {
        return processor_.PieceToId(piece);
    }
    
    int GetPieceSize() const {
        return processor_.GetPieceSize();
    }
    
    std::string model_type() const {
        if (processor_.model_type() == sentencepiece::ModelType::UNIGRAM) {
            return "unigram";
        } else if (processor_.model_type() == sentencepiece::ModelType::BPE) {
            return "bpe";
        } else if (processor_.model_type() == sentencepiece::ModelType::CHAR) {
            return "char";
        } else if (processor_.model_type() == sentencepiece::ModelType::WORD) {
            return "word";
        } else {
            return "unknown";
        }
    }
    
    std::string Normalize(const std::string& text) const {
        std::string normalized;
        processor_.Normalize(text, &normalized);
        return normalized;
    }
    
    // Multi-threaded batch encoding
    std::vector<std::vector<int>> EncodeBatch(const std::vector<std::string>& texts, int num_threads = -1) const {
        std::vector<std::vector<int>> batch_ids;
        processor_.EncodeSentences(texts, &batch_ids, num_threads);
        return batch_ids;
    }
    
private:
    sentencepiece::SentencePieceProcessor processor_;
};

// SentencePiece Tokenizer Implementation
SentencePieceTokenizer::SentencePieceTokenizer(const std::string& model_path)
    : model_path_(model_path) {
    processor_ = std::make_unique<SPProcessorWrapper>(model_path);
}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

std::vector<int> SentencePieceTokenizer::encode(const std::string& text) const {
    try {
        return processor_->Encode(text);
    } catch (const std::exception& e) {
        CV_Error(Error::StsError, "SentencePiece encoding error: " + std::string(e.what()));
    }
}

std::vector<TokenInfo> SentencePieceTokenizer::encodeWithInfo(const std::string& text) const {
    std::vector<TokenInfo> result;
    
    try {
        std::vector<std::pair<int, std::string>> encoded = processor_->EncodeWithPieces(text);
        
        // SentencePiece doesn't provide token positions, so we need to reconstruct them
        std::string processed_text = text;
        size_t current_pos = 0;
        
        for (const auto& [id, piece] : encoded) {
            TokenInfo info;
            info.id = id;
            info.text = piece;
            
            // Find the piece in the text
            size_t pos = processed_text.find(piece, current_pos);
            if (pos != std::string::npos) {
                info.start = static_cast<int>(pos);
                info.end = static_cast<int>(pos + piece.length());
                current_pos = pos + piece.length();
            } else {
                // If the exact piece is not found (can happen with normalization)
                // Just use the current position
                info.start = static_cast<int>(current_pos);
                current_pos += piece.length();
                info.end = static_cast<int>(current_pos);
            }
            
            // SentencePiece doesn't provide scores, so use a placeholder
            info.score = 1.0f;
            
            result.push_back(info);
        }
    } catch (const std::exception& e) {
        CV_Error(Error::StsError, "SentencePiece encoding error: " + std::string(e.what()));
    }
    
    return result;
}

std::vector<std::vector<int>> SentencePieceTokenizer::encodeBatch(const std::vector<std::string>& texts) const {
    try {
        // Use multi-threading if available
        return processor_->EncodeBatch(texts);
    } catch (const std::exception& e) {
        CV_Error(Error::StsError, "SentencePiece batch encoding error: " + std::string(e.what()));
    }
}

std::string SentencePieceTokenizer::decode(const std::vector<int>& tokens) const {
    try {
        return processor_->Decode(tokens);
    } catch (const std::exception& e) {
        CV_Error(Error::StsError, "SentencePiece decoding error: " + std::string(e.what()));
    }
}

size_t SentencePieceTokenizer::getVocabSize() const {
    return processor_->GetPieceSize();
}

void SentencePieceTokenizer::save(const std::string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    
    fs << "tokenizer_type" << "sentencepiece";
    fs << "model_path" << model_path_;
    fs << "model_type" << processor_->model_type();
    fs << "vocab_size" << static_cast<int>(getVocabSize());
}

std::string SentencePieceTokenizer::getTokenText(int tokenId) const {
    return processor_->IdToPiece(tokenId);
}

int SentencePieceTokenizer::getTokenId(const std::string& tokenText) const {
    return processor_->PieceToId(tokenText);
}

std::string SentencePieceTokenizer::getModelType() const {
    return processor_->model_type();
}

std::string SentencePieceTokenizer::normalizeText(const std::string& text) const {
    return processor_->Normalize(text);
}

Ptr<Tokenizer> createSentencePieceTokenizer(const std::string& model_path) {
    return makePtr<SentencePieceTokenizer>(model_path);
}

}} // namespace cv::dnn