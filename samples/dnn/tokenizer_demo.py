#!/usr/bin/env python

'''
OpenCV DNN Tokenizer Demo Python Script

@file tokenizer_demo.py
@brief Python demonstration of OpenCV tokenizer functionality
@author Rituraj Singh
@date 2025

This sample demonstrates the usage of tokenizers in OpenCV DNN module.
It can create BPE or WordPiece tokenizers, encode/decode text, and save/load tokenizers.
This is the Python equivalent of the C++ tokenizer_demo.cpp example.
'''

import argparse
import os
import cv2 as cv
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='OpenCV DNN Tokenizer Demo',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('tokenizer_type', nargs='?', type=str, default=None,
                        help='Tokenizer type (bpe or wordpiece)')
    parser.add_argument('vocab', nargs='?', type=str, default=None,
                        help='Path to vocabulary file')
    parser.add_argument('merges', nargs='?', type=str, default=None,
                        help='Path to merges file (only for BPE tokenizer)')
    parser.add_argument('text', nargs='?', type=str, default=None,
                        help='Text to tokenize (if not provided, will use sample text)')
    parser.add_argument('--display', action='store_true',
                        help='Display tokens and their IDs')
    parser.add_argument('--save', type=str, default=None,
                        help='Save the tokenizer to a file')
    parser.add_argument('--load', type=str, default=None,
                        help='Load a tokenizer from a file')
    args = parser.parse_args()
    
    # Check if we're loading a pre-saved tokenizer
    tokenizer = None
    if args.load:
        print(f"Loading tokenizer from: {args.load}")
        
        try:
            tokenizer = cv.dnn.Tokenizer_load(args.load)
            print(f"Tokenizer loaded successfully with vocabulary size: {tokenizer.getVocabSize()}")
        except cv.error as e:
            print(f"Error loading tokenizer: {e}")
            return
    else:
        # Create a new tokenizer
        if not args.tokenizer_type:
            print("Tokenizer type must be specified (bpe or wordpiece)")
            parser.print_help()
            return
        
        tokenizer_type = args.tokenizer_type.lower()
        
        if not args.vocab:
            print("Vocabulary file must be specified")
            parser.print_help()
            return
        
        if tokenizer_type == "bpe":
            if not args.merges:
                print("Merges file must be specified for BPE tokenizer")
                parser.print_help()
                return
            
            print(f"Creating BPE tokenizer with vocab: {args.vocab} and merges: {args.merges}")
            
            try:
                tokenizer = cv.dnn.createBPETokenizer(args.vocab, args.merges)
                print(f"BPE tokenizer created successfully with vocabulary size: {tokenizer.getVocabSize()}")
            except cv.error as e:
                print(f"Error creating BPE tokenizer: {e}")
                return
        elif tokenizer_type == "wordpiece":
            print(f"Creating WordPiece tokenizer with vocab: {args.vocab}")
            
            try:
                tokenizer = cv.dnn.createWordPieceTokenizer(args.vocab)
                print(f"WordPiece tokenizer created successfully with vocabulary size: {tokenizer.getVocabSize()}")
            except cv.error as e:
                print(f"Error creating WordPiece tokenizer: {e}")
                return
        else:
            print(f"Unknown tokenizer type: {tokenizer_type}")
            return
    
    # Save the tokenizer if requested
    if args.save:
        print(f"Saving tokenizer to: {args.save}")
        
        try:
            tokenizer.save(args.save)
            print("Tokenizer saved successfully")
        except cv.error as e:
            print(f"Error saving tokenizer: {e}")
            return
    
    # Process text if provided
    text = args.text
    if not text:
        # Use sample text if none provided
        text = "Hello world! This is the OpenCV tokenizer demo for LLMs."
    
    print(f"\nInput text: {text}")
    
    # Encode the text
    try:
        tokens = tokenizer.encode(text)
        print(f"Encoded into {len(tokens)} tokens")
        
        # Display tokens if requested
        if args.display:
            print(f"Tokens: {tokens}")
        
        # Decode the tokens back to text
        decoded = tokenizer.decode(tokens)
        print(f"Decoded text: {decoded}")
    except cv.error as e:
        print(f"Error during tokenization: {e}")
        return

    # Example of batch encoding
    if args.display:
        texts = [
            "Hello world!",
            "OpenCV is great for computer vision.",
            "LLMs need tokenizers to process text."
        ]
        print("\nBatch encoding example:")
        for i, batch_text in enumerate(texts):
            print(f"\nText {i+1}: {batch_text}")
            
            # Individual encoding
            batch_tokens = tokenizer.encode(batch_text)
            print(f"Tokens: {batch_tokens}")

if __name__ == "__main__":
    main()