/**
 * @file tokenizer_demo.cpp
 * @brief Demo application showcasing tokenizer functionality
 * @author Rituraj Singh
 * @date 2025
 *
 * This sample demonstrates basic usage of OpenCV's tokenizer implementations
 * including creation, text encoding/decoding, and save/load operations.
 * It serves as an introductory example for tokenizer usage.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/dnn/tokenizer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
    "{ help h      | | Print help message }"
    "{ @tokenizer  | | Tokenizer type (bpe or wordpiece) }"
    "{ @vocab      | | Path to vocabulary file }"
    "{ @merges     | | Path to merges file (only for BPE tokenizer) }"
    "{ @text       | | Text to tokenize (if not provided, will use sample texts) }"
    "{ display     | | Display tokens and their IDs }"
    "{ save s      | | Save the tokenizer to a file }"
    "{ load l      | | Load a tokenizer from a file }";

void printHelp() {
    std::cout << "This sample demonstrates the usage of tokenizers in OpenCV DNN module.\n"
              << "It can create BPE or WordPiece tokenizers, encode/decode text, and save/load tokenizers.\n\n"
              << "Usage examples:\n"
              << "  tokenizer_demo.exe bpe vocab.json merges.txt \"Hello world, this is a test!\"\n"
              << "  tokenizer_demo.exe wordpiece vocab.txt \"Hello world, this is a test!\"\n"
              << "  tokenizer_demo.exe bpe vocab.json merges.txt --save=tokenizer.yml\n"
              << "  tokenizer_demo.exe --load=tokenizer.yml \"Hello world, this is a test!\"\n";
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV DNN Tokenizer Demo");

    // Print help if requested
    if (parser.has("help")) {
        parser.printMessage();
        printHelp();
        return 0;
}
    }

    // Check if we're loading a pre-saved tokenizer
    cv::Ptr<cv::dnn::Tokenizer> tokenizer;
    if (parser.has("load")) {
        std::string load_path = parser.get<std::string>("load");
        std::cout << "Loading tokenizer from: " << load_path << std::endl;
        
        try {
            tokenizer = cv::dnn::Tokenizer::load(load_path);
            std::cout << "Tokenizer loaded successfully with vocabulary size: " 
                      << tokenizer->getVocabSize() << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error loading tokenizer: " << e.what() << std::endl;
            return -1;
        }
    } else {
        // Create a new tokenizer
        if (!parser.has("@tokenizer")) {
            std::cerr << "Tokenizer type must be specified (bpe or wordpiece)" << std::endl;
            parser.printMessage();
            return -1;
        }

        std::string tokenizer_type = parser.get<std::string>("@tokenizer");
        std::transform(tokenizer_type.begin(), tokenizer_type.end(), tokenizer_type.begin(),
                      [](unsigned char c){ return std::tolower(c); });

        if (!parser.has("@vocab")) {
            std::cerr << "Vocabulary file must be specified" << std::endl;
            parser.printMessage();
            return -1;
        }
        
        std::string vocab_path = parser.get<std::string>("@vocab");
        
        if (tokenizer_type == "bpe") {
            if (!parser.has("@merges")) {
                std::cerr << "Merges file must be specified for BPE tokenizer" << std::endl;
                parser.printMessage();
                return -1;
            }
            
            std::string merges_path = parser.get<std::string>("@merges");
            std::cout << "Creating BPE tokenizer with vocab: " << vocab_path 
                      << " and merges: " << merges_path << std::endl;
            
            try {
                tokenizer = cv::dnn::createBPETokenizer(vocab_path, merges_path);
                std::cout << "BPE tokenizer created successfully with vocabulary size: " 
                          << tokenizer->getVocabSize() << std::endl;
            } catch (const cv::Exception& e) {
                std::cerr << "Error creating BPE tokenizer: " << e.what() << std::endl;
                return -1;
            }
        } else if (tokenizer_type == "wordpiece") {
            std::cout << "Creating WordPiece tokenizer with vocab: " << vocab_path << std::endl;
            
            try {
                tokenizer = cv::dnn::createWordPieceTokenizer(vocab_path);
                std::cout << "WordPiece tokenizer created successfully with vocabulary size: " 
                          << tokenizer->getVocabSize() << std::endl;
            } catch (const cv::Exception& e) {
                std::cerr << "Error creating WordPiece tokenizer: " << e.what() << std::endl;
                return -1;
            }
        } else {
            std::cerr << "Unknown tokenizer type: " << tokenizer_type << std::endl;
            return -1;
        }
    }

    // Save the tokenizer if requested
    if (parser.has("save")) {
        std::string save_path = parser.get<std::string>("save");
        std::cout << "Saving tokenizer to: " << save_path << std::endl;
        
        try {
            tokenizer->save(save_path);
            std::cout << "Tokenizer saved successfully" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error saving tokenizer: " << e.what() << std::endl;
            return -1;
        }
    }

    // Process text if provided
    std::string text;
    if (parser.has("@text")) {
        text = parser.get<std::string>("@text");
    } else {
        // Use sample text if none provided
        text = "Hello world! This is the OpenCV tokenizer demo for LLMs.";
    }
    
    std::cout << "\nInput text: " << text << std::endl;
    
    // Encode the text
    std::vector<int> tokens;
    try {
        tokens = tokenizer->encode(text);
        std::cout << "Encoded into " << tokens.size() << " tokens" << std::endl;
        
        // Display tokens if requested
        if (parser.has("display")) {
            std::cout << "Tokens: ";
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << tokens[i] << " ";
            }
            std::cout << std::endl;
        }
        
        // Decode the tokens back to text
        std::string decoded = tokenizer->decode(tokens);
        std::cout << "Decoded text: " << decoded << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Error during tokenization: " << e.what() << std::endl;
        return -1;
    }

    return 0;