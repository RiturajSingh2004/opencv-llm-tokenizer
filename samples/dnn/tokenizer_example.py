#!/usr/bin/env python

'''
Example of using OpenCV DNN tokenizer bindings in Python

@file tokenizer_example.py
@brief Extended examples of OpenCV tokenizer API usage in Python
@author Rituraj Singh
@date 2025

This script demonstrates the complete Python API for OpenCV tokenizers,
including various tokenizer types, GPU acceleration, and performance benchmarking.
It serves as a practical reference for Python developers using OpenCV tokenizers.
'''

import cv2 as cv
import numpy as np
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='OpenCV DNN Tokenizer Python Example')
    parser.add_argument('--bpe_vocab', type=str, help='Path to BPE vocabulary file (JSON)')
    parser.add_argument('--bpe_merges', type=str, help='Path to BPE merges file')
    parser.add_argument('--wordpiece_vocab', type=str, help='Path to WordPiece vocabulary file')
    parser.add_argument('--unigram_vocab', type=str, help='Path to Unigram vocabulary file')
    parser.add_argument('--sentencepiece_model', type=str, help='Path to SentencePiece model file (.model)')
    parser.add_argument('--transformers_model', type=str, help='Name or path to HuggingFace Transformers model')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--text', type=str, default='Hello world! This is a test of OpenCV tokenizers.', 
                        help='Text to tokenize')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='Batch size for benchmarking')
    parser.add_argument('--iterations', type=int, default=5, 
                        help='Number of iterations for benchmarking')
    args = parser.parse_args()

    # Create tokenizers based on provided arguments
    tokenizers = []
    
    if args.bpe_vocab and args.bpe_merges:
        print(f"Creating BPE tokenizer with vocab: {args.bpe_vocab} and merges: {args.bpe_merges}")
        try:
            bpe_tokenizer = cv.dnn.createBPETokenizer(args.bpe_vocab, args.bpe_merges)
            tokenizers.append(("BPE", bpe_tokenizer))
            print(f"BPE tokenizer created with vocabulary size: {bpe_tokenizer.getVocabSize()}")
        except Exception as e:
            print(f"Error creating BPE tokenizer: {e}")
    
    if args.wordpiece_vocab:
        print(f"Creating WordPiece tokenizer with vocab: {args.wordpiece_vocab}")
        try:
            wp_tokenizer = cv.dnn.createWordPieceTokenizer(args.wordpiece_vocab)
            tokenizers.append(("WordPiece", wp_tokenizer))
            print(f"WordPiece tokenizer created with vocabulary size: {wp_tokenizer.getVocabSize()}")
        except Exception as e:
            print(f"Error creating WordPiece tokenizer: {e}")
    
    if args.unigram_vocab:
        print(f"Creating Unigram tokenizer with vocab: {args.unigram_vocab}")
        try:
            unigram_tokenizer = cv.dnn.createUnigramTokenizer(args.unigram_vocab)
            tokenizers.append(("Unigram", unigram_tokenizer))
            print(f"Unigram tokenizer created with vocabulary size: {unigram_tokenizer.getVocabSize()}")
        except Exception as e:
            print(f"Error creating Unigram tokenizer: {e}")
    
    if args.sentencepiece_model:
        print(f"Creating SentencePiece tokenizer with model: {args.sentencepiece_model}")
        try:
            sp_tokenizer = cv.dnn.createSentencePieceTokenizer(args.sentencepiece_model)
            tokenizers.append(("SentencePiece", sp_tokenizer))
            print(f"SentencePiece tokenizer created with vocabulary size: {sp_tokenizer.getVocabSize()}")
        except Exception as e:
            print(f"Error creating SentencePiece tokenizer: {e}")
    
    if args.transformers_model:
        print(f"Creating Transformers tokenizer with model: {args.transformers_model}")
        try:
            transformers_tokenizer = cv.dnn.createTransformersTokenizer(args.transformers_model)
            tokenizers.append(("Transformers", transformers_tokenizer))
            print(f"Transformers tokenizer created with vocabulary size: {transformers_tokenizer.getVocabSize()}")
        except Exception as e:
            print(f"Error creating Transformers tokenizer: {e}")
    
    if not tokenizers:
        print("Error: No tokenizers created. Please provide at least one tokenizer configuration.")
        return

    # Check GPU availability
    use_gpu = args.use_gpu
    if use_gpu:
        try:
            gpu_available = cv.dnn.GPUTokenizer.isAvailable()
            if not gpu_available:
                print("GPU acceleration requested but not available. Falling back to CPU.")
                use_gpu = False
            else:
                print("GPU acceleration is available and will be used.")
        except Exception as e:
            print(f"Error checking GPU availability: {e}")
            use_gpu = False

    # Process the input text with each tokenizer
    print(f"\nInput text: {args.text}")
    
    for name, tokenizer in tokenizers:
        print(f"\n=== {name} Tokenizer ===")
        
        # Encode
        try:
            start_time = time.time()
            tokens = tokenizer.encode(args.text)
            encode_time = (time.time() - start_time) * 1000
            
            print(f"Encoded into {len(tokens)} tokens in {encode_time:.2f} ms")
            print(f"Tokens: {tokens}")
            
            # Try to get token text for better visualization
            token_texts = []
            for token_id in tokens:
                try:
                    token_text = tokenizer.getTokenText(token_id)
                    token_texts.append(token_text)
                except:
                    token_texts.append(f"[{token_id}]")
            
            print(f"Token texts: {token_texts}")
            
            # Decode
            start_time = time.time()
            decoded = tokenizer.decode(tokens)
            decode_time = (time.time() - start_time) * 1000
            
            print(f"Decoded in {decode_time:.2f} ms")
            print(f"Decoded text: {decoded}")
            
            # Try encodeWithInfo for detailed token information
            try:
                token_info = tokenizer.encodeWithInfo(args.text)
                print(f"\nToken information:")
                for info in token_info:
                    print(f"  ID: {info['id']}, Text: '{info['text']}', "
                          f"Position: {info['start']}-{info['end']}, Score: {info['score']:.4f}")
            except Exception as e:
                print(f"encodeWithInfo not fully supported: {e}")
            
            # Check for TransformersTokenizer specific methods
            if name == "Transformers":
                try:
                    print("\nTrying Transformers-specific methods:")
                    
                    # encodeForModel
                    model_tokens = tokenizer.encodeForModel(
                        args.text, max_length=32, padding=True, truncation=True, add_special_tokens=True
                    )
                    print(f"  encodeForModel tokens: {model_tokens}")
                    
                    # encodeBatchForModel
                    batch_texts = [args.text, "This is another test."]
                    batch_model_tokens = tokenizer.encodeBatchForModel(
                        batch_texts, max_length=32, padding=True, truncation=True, add_special_tokens=True
                    )
                    print(f"  encodeBatchForModel result shape: {len(batch_model_tokens)} x {len(batch_model_tokens[0])}")
                except Exception as e:
                    print(f"  Transformers-specific methods not fully supported: {e}")
            
            # Create GPU tokenizer if requested
            if use_gpu:
                try:
                    gpu_tokenizer = cv.dnn.createGPUTokenizer(tokenizer)
                    gpu_tokenizer.warmup()
                    
                    # Prepare batch data for testing
                    batch_texts = [args.text] * args.batch_size
                    
                    # Benchmark CPU batch encoding
                    start_time = time.time()
                    for _ in range(args.iterations):
                        tokenizer.encodeBatch(batch_texts)
                    cpu_batch_time = (time.time() - start_time) * 1000 / args.iterations
                    
                    # Benchmark GPU batch encoding
                    start_time = time.time()
                    for _ in range(args.iterations):
                        gpu_tokenizer.encodeBatch(batch_texts)
                    gpu_batch_time = (time.time() - start_time) * 1000 / args.iterations
                    
                    print(f"\nBatch encoding performance (batch size: {args.batch_size}):")
                    print(f"  CPU: {cpu_batch_time:.2f} ms")
                    print(f"  GPU: {gpu_batch_time:.2f} ms")
                    print(f"  Speedup: {cpu_batch_time / gpu_batch_time:.2f}x")
                    
                    # Try GPU matrix encoding if available
                    try:
                        start_time = time.time()
                        for _ in range(args.iterations):
                            gpu_mat = gpu_tokenizer.encodeBatchToGpuMat(batch_texts)
                        gpu_mat_time = (time.time() - start_time) * 1000 / args.iterations
                        print(f"  GPU matrix: {gpu_mat_time:.2f} ms")
                        print(f"  Matrix shape: {gpu_mat.shape}")
                    except Exception as e:
                        print(f"GPU matrix encoding not fully supported: {e}")
                    
                except Exception as e:
                    print(f"Error with GPU tokenizer: {e}")
            
        except Exception as e:
            print(f"Error processing text: {e}")

    print("\nDone!")

if __name__ == "__main__":
    main()