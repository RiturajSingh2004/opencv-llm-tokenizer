/**
 * @file test_tokenizer.cpp
 * @brief Basic tests for tokenizer implementations
 * @author Rituraj Singh
 * @date 2025
 *
 * This file contains core unit tests for all tokenizer implementations,
 * verifying basic functionality like encoding, decoding, and serialization
 * across BPE, WordPiece, and other tokenization approaches.
 */

#include "test_precomp.hpp"
#include <opencv2/dnn/tokenizer.hpp>
#include <fstream>

namespace opencv_test {
namespace {

// Helper function to create a temporary file with content
std::string createTempFile(const std::string& content, const std::string& extension = ".txt") {
    const std::string tempFileName = cv::tempfile(extension.c_str());
    std::ofstream file(tempFileName);
    file << content;
    file.close();
    return tempFileName;
}

// BPE vocab and merges for testing
const std::string TEST_BPE_VOCAB = R"(
{
  "!": 0,
  "\"": 1,
  "#": 2,
  "$": 3,
  "%": 4,
  "&": 5,
  "'": 6,
  "(": 7,
  ")": 8,
  "*": 9,
  "+": 10,
  ",": 11,
  "-": 12,
  ".": 13,
  "/": 14,
  "0": 15,
  "1": 16,
  "2": 17,
  "3": 18,
  "4": 19,
  "5": 20,
  "6": 21,
  "7": 22,
  "8": 23,
  "9": 24,
  ":": 25,
  ";": 26,
  "<": 27,
  "=": 28,
  ">": 29,
  "?": 30,
  "@": 31,
  "A": 32,
  "B": 33,
  "C": 34,
  "D": 35,
  "E": 36,
  "F": 37,
  "G": 38,
  "H": 39,
  "I": 40,
  "J": 41,
  "K": 42,
  "L": 43,
  "M": 44,
  "N": 45,
  "O": 46,
  "P": 47,
  "Q": 48,
  "R": 49,
  "S": 50,
  "T": 51,
  "U": 52,
  "V": 53,
  "W": 54,
  "X": 55,
  "Y": 56,
  "Z": 57,
  "[": 58,
  "\\": 59,
  "]": 60,
  "^": 61,
  "_": 62,
  "`": 63,
  "a": 64,
  "b": 65,
  "c": 66,
  "d": 67,
  "e": 68,
  "f": 69,
  "g": 70,
  "h": 71,
  "i": 72,
  "j": 73,
  "k": 74,
  "l": 75,
  "m": 76,
  "n": 77,
  "o": 78,
  "p": 79,
  "q": 80,
  "r": 81,
  "s": 82,
  "t": 83,
  "u": 84,
  "v": 85,
  "w": 86,
  "x": 87,
  "y": 88,
  "z": 89,
  "{": 90,
  "|": 91,
  "}": 92,
  "~": 93,
  "hello": 94,
  "world": 95,
  "OpenCV": 96,
  "token": 97,
  "ize": 98,
  "tokenize": 99,
  "<|endoftext|>": 100
}
)";

const std::string TEST_BPE_MERGES = R"(#version: 0.2
t o
to k
tok e
toke n
token i
tokeni z
tokeniz e
tokenize r
h e
he l
hel l
hell o
hello w
hellow o
hellowo r
hellowor l
helloworl d
O p
Op e
Ope n
Open C
OpenC V
)";

// WordPiece vocab for testing
const std::string TEST_WORDPIECE_VOCAB = R"([PAD]
[UNK]
[CLS]
[SEP]
[MASK]
hello
world
open
cv
token
##ize
##r
)";

class TokenizerTest : public testing::Test {
protected:
    void SetUp() override {
        // Create test files
        bpeVocabFile = createTempFile(TEST_BPE_VOCAB, ".json");
        bpeMergesFile = createTempFile(TEST_BPE_MERGES, ".txt");
        wordpieceVocabFile = createTempFile(TEST_WORDPIECE_VOCAB, ".txt");
    }

    void TearDown() override {
        // Clean up temporary files
        std::remove(bpeVocabFile.c_str());
        std::remove(bpeMergesFile.c_str());
        std::remove(wordpieceVocabFile.c_str());
    }

    std::string bpeVocabFile;
    std::string bpeMergesFile;
    std::string wordpieceVocabFile;
};

TEST_F(TokenizerTest, CreateBPETokenizer) {
    cv::Ptr<cv::dnn::Tokenizer> tokenizer = cv::dnn::createBPETokenizer(bpeVocabFile, bpeMergesFile);
    EXPECT_NE(tokenizer, nullptr);
    EXPECT_GT(tokenizer->getVocabSize(), 0);
}

TEST_F(TokenizerTest, CreateWordPieceTokenizer) {
    cv::Ptr<cv::dnn::Tokenizer> tokenizer = cv::dnn::createWordPieceTokenizer(wordpieceVocabFile);
    EXPECT_NE(tokenizer, nullptr);
    EXPECT_GT(tokenizer->getVocabSize(), 0);
}

TEST_F(TokenizerTest, BPEEncodeDecode) {
    cv::Ptr<cv::dnn::Tokenizer> tokenizer = cv::dnn::createBPETokenizer(bpeVocabFile, bpeMergesFile);
    
    // Test with words that should be in the vocabulary
    std::string text = "hello world";
    std::vector<int> tokens = tokenizer->encode(text);
    EXPECT_FALSE(tokens.empty());
    
    // Check if we can decode back to text
    std::string decoded = tokenizer->decode(tokens);
    EXPECT_FALSE(decoded.empty());
    
    // The decoded text might not match exactly due to tokenization/detokenization
    // But we can check if our vocabulary words are preserved
    EXPECT_NE(decoded.find("hello"), std::string::npos);
    EXPECT_NE(decoded.find("world"), std::string::npos);
}

TEST_F(TokenizerTest, WordPieceEncodeDecode) {
    cv::Ptr<cv::dnn::Tokenizer> tokenizer = cv::dnn::createWordPieceTokenizer(wordpieceVocabFile);
    
    // Test with words that should be in the vocabulary
    std::string text = "hello world";
    std::vector<int> tokens = tokenizer->encode(text);
    EXPECT_FALSE(tokens.empty());
    
    // Check if we can decode back to text
    std::string decoded = tokenizer->decode(tokens);
    EXPECT_FALSE(decoded.empty());
    
    // The decoded text might not match exactly due to tokenization/detokenization
    // But we can check if our vocabulary words are preserved
    EXPECT_NE(decoded.find("hello"), std::string::npos);
    EXPECT_NE(decoded.find("world"), std::string::npos);
}

TEST_F(TokenizerTest, BPETokenizerSaveLoad) {
    cv::Ptr<cv::dnn::Tokenizer> tokenizer = cv::dnn::createBPETokenizer(bpeVocabFile, bpeMergesFile);
    
    // Test text
    std::string text = "hello world OpenCV";
    std::vector<int> expected_tokens = tokenizer->encode(text);
    
    // Save the tokenizer
    std::string saveFile = cv::tempfile(".yml");
    tokenizer->save(saveFile);
    
    // Load the tokenizer
    cv::Ptr<cv::dnn::Tokenizer> loaded_tokenizer = cv::dnn::Tokenizer::load(saveFile);
    EXPECT_NE(loaded_tokenizer, nullptr);
    
    // Check that encoding gives the same result
    std::vector<int> loaded_tokens = loaded_tokenizer->encode(text);
    EXPECT_EQ(expected_tokens, loaded_tokens);
    
    // Clean up
    std::remove(saveFile.c_str());
}

TEST_F(TokenizerTest, WordPieceTokenizerSaveLoad) {
    cv::Ptr<cv::dnn::Tokenizer> tokenizer = cv::dnn::createWordPieceTokenizer(wordpieceVocabFile);
    
    // Test text
    std::string text = "hello world tokenizer";
    std::vector<int> expected_tokens = tokenizer->encode(text);
    
    // Save the tokenizer
    std::string saveFile = cv::tempfile(".yml");
    tokenizer->save(saveFile);
    
    // Load the tokenizer
    cv::Ptr<cv::dnn::Tokenizer> loaded_tokenizer = cv::dnn::Tokenizer::load(saveFile);
    EXPECT_NE(loaded_tokenizer, nullptr);
    
    // Check that encoding gives the same result
    std::vector<int> loaded_tokens = loaded_tokenizer->encode(text);
    EXPECT_EQ(expected_tokens, loaded_tokens);
    
    // Clean up
    std::remove(saveFile.c_str());
}

TEST_F(TokenizerTest, BatchEncoding) {
    cv::Ptr<cv::dnn::Tokenizer> tokenizer = cv::dnn::createBPETokenizer(bpeVocabFile, bpeMergesFile);
    
    std::vector<std::string> texts = {"hello world", "OpenCV tokenize"};
    std::vector<std::vector<int>> batch_tokens = tokenizer->encodeBatch(texts);
    
    EXPECT_EQ(batch_tokens.size(), texts.size());
    
    // Check that batch encoding matches individual encoding
    for (size_t i = 0; i < texts.size(); i++) {
        std::vector<int> single_tokens = tokenizer->encode(texts[i]);
        EXPECT_EQ(batch_tokens[i], single_tokens);
    }
}

}} // namespace