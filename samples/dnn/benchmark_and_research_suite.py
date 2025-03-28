#!/usr/bin/env python

'''
Benchmark and Research Suite for OpenCV LLM Tokenizers

@file benchmark_and_research_suite.py
@brief Comprehensive benchmarking and research framework for tokenizers
@author Rituraj Singh
@date 2025

This script provides a comprehensive benchmarking and research framework
for tokenizers, supporting cross-framework comparisons, multilingual analysis,
and performance metrics. It can generate visualizations and reports for
tokenizer evaluation across multiple dimensions.
'''

import argparse
import os
import sys
import json
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from pathlib import Path

# Try to import optional dependencies for comparison
HAVE_HUGGINGFACE = False
HAVE_TIKTOKEN = False
HAVE_SENTENCEPIECE = False

try:
    from transformers import AutoTokenizer
    HAVE_HUGGINGFACE = True
except ImportError:
    pass

try:
    import tiktoken
    HAVE_TIKTOKEN = True
except ImportError:
    pass

try:
    import sentencepiece as spm
    HAVE_SENTENCEPIECE = True
except ImportError:
    pass

# Load language samples for multilingual benchmarking
LANGUAGE_SAMPLES = {
    "en": "This is a sample English text for benchmarking tokenizers. It includes various punctuation, numbers (123, 45.67), and special cases like URLs (https://example.com).",
    "fr": "Ceci est un exemple de texte français pour évaluer les performances des tokenizers. Il comprend diverses ponctuations, des nombres et des cas spéciaux.",
    "de": "Dies ist ein deutscher Beispieltext für Tokenizer-Benchmarking. Er enthält verschiedene Interpunktionen, Zahlen und Sonderfälle.",
    "es": "Este es un texto de muestra en español para comparar tokenizadores. Incluye varios signos de puntuación, números y casos especiales.",
    "zh": "这是一个用于基准测试分词器的中文样本文本。它包括各种标点符号，数字和特殊情况！",
    "ja": "これはトークナイザーのベンチマークのための日本語サンプルテキストです。さまざまな句読点、数字、特殊なケースが含まれています。",
    "ko": "이것은 토크나이저 벤치마킹을 위한 한국어 샘플 텍스트입니다. 다양한 구두점, 숫자 및 특수 사례가 포함되어 있습니다.",
    "ar": "هذا نص عربي عينة لاختبار أداء المحللات اللغوية. يتضمن علامات ترقيم مختلفة وأرقام وحالات خاصة!",
    "hi": "यह टोकनाइज़र के बेंचमार्किंग के लिए एक हिंदी नमूना पाठ है। इसमें विभिन्न विराम चिह्न, संख्याएँ और विशेष मामले शामिल हैं!",
    "th": "นี่เป็นข้อความตัวอย่างภาษาไทยสำหรับการทดสอบประสิทธิภาพของโทเค็นไนเซอร์ มันรวมถึงเครื่องหมายวรรคตอนต่างๆ ตัวเลข และกรณีพิเศษ!",
}

# Technical content samples
TECHNICAL_SAMPLES = {
    "code": '''
def tokenize(text):
    """Split text into tokens"""
    return text.split()
    
class Tokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
    
    def encode(self, text):
        return [ord(c) % self.vocab_size for c in text]
''',
    "math": r'''
Mathematical equation examples:
1. $E = mc^2$
2. $\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$
3. $f(x) = \frac{1}{1 + e^{-x}}$
4. $\nabla \times \vec{F} = 0$
''',
    "json": '''
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how does tokenization work?"},
    {"role": "assistant", "content": "Tokenization is the process of breaking text into smaller units called tokens."}
  ]
}
'''
}

class TokenizerWrapper:
    """
    Uniform interface for different tokenizer frameworks
    """
    def __init__(self, name, tokenizer, framework):
        self.name = name
        self.tokenizer = tokenizer
        self.framework = framework
    
    def encode(self, text):
        if self.framework == "opencv":
            return self.tokenizer.encode(text)
        elif self.framework == "huggingface":
            return self.tokenizer.encode(text)
        elif self.framework == "tiktoken":
            return self.tokenizer.encode(text)
        elif self.framework == "sentencepiece":
            return self.tokenizer.EncodeAsIds(text)
        return []
    
    def decode(self, tokens):
        if self.framework == "opencv":
            return self.tokenizer.decode(tokens)
        elif self.framework == "huggingface":
            return self.tokenizer.decode(tokens)
        elif self.framework == "tiktoken":
            return self.tokenizer.decode(tokens)
        elif self.framework == "sentencepiece":
            return self.tokenizer.DecodeIds(tokens)
        return ""
    
    def get_vocab_size(self):
        if self.framework == "opencv":
            return self.tokenizer.getVocabSize()
        elif self.framework == "huggingface":
            return len(self.tokenizer.get_vocab())
        elif self.framework == "tiktoken":
            # Tiktoken doesn't have a direct vocab size accessor, estimate from max token value
            return max(self.tokenizer.encode("test")) + 1
        elif self.framework == "sentencepiece":
            return self.tokenizer.GetPieceSize()
        return 0

def load_opencv_tokenizers(base_dir):
    """Load all available OpenCV tokenizers from the specified directory"""
    tokenizers = []
    
    # Check for BPE tokenizer files
    bpe_vocab = os.path.join(base_dir, "bpe_vocab.json")
    bpe_merges = os.path.join(base_dir, "bpe_merges.txt")
    if os.path.exists(bpe_vocab) and os.path.exists(bpe_merges):
        try:
            bpe_tokenizer = cv.dnn.createBPETokenizer(bpe_vocab, bpe_merges)
            tokenizers.append(TokenizerWrapper("opencv_bpe", bpe_tokenizer, "opencv"))
            print(f"Loaded OpenCV BPE tokenizer with vocabulary size: {bpe_tokenizer.getVocabSize()}")
        except Exception as e:
            print(f"Error loading OpenCV BPE tokenizer: {e}")
    
    # Check for WordPiece tokenizer files
    wordpiece_vocab = os.path.join(base_dir, "wordpiece_vocab.txt")
    if os.path.exists(wordpiece_vocab):
        try:
            wp_tokenizer = cv.dnn.createWordPieceTokenizer(wordpiece_vocab)
            tokenizers.append(TokenizerWrapper("opencv_wordpiece", wp_tokenizer, "opencv"))
            print(f"Loaded OpenCV WordPiece tokenizer with vocabulary size: {wp_tokenizer.getVocabSize()}")
        except Exception as e:
            print(f"Error loading OpenCV WordPiece tokenizer: {e}")
    
    # Check for Unigram tokenizer files
# Check for Unigram tokenizer files
    unigram_vocab = os.path.join(base_dir, "unigram_vocab.txt")
    if os.path.exists(unigram_vocab):
        try:
            unigram_tokenizer = cv.dnn.createUnigramTokenizer(unigram_vocab)
            tokenizers.append(TokenizerWrapper("opencv_unigram", unigram_tokenizer, "opencv"))
            print(f"Loaded OpenCV Unigram tokenizer with vocabulary size: {unigram_tokenizer.getVocabSize()}")
        except Exception as e:
            print(f"Error loading OpenCV Unigram tokenizer: {e}")
    
    # Check for Adaptive tokenizer
    if os.path.exists(bpe_vocab) and os.path.exists(bpe_merges) and os.path.exists(unigram_vocab):
        try:
            adaptive_tokenizer = cv.dnn.createAdaptiveTokenizer(bpe_vocab, bpe_merges, unigram_vocab)
            tokenizers.append(TokenizerWrapper("opencv_adaptive", adaptive_tokenizer, "opencv"))
            print(f"Loaded OpenCV Adaptive tokenizer")
        except Exception as e:
            print(f"Error loading OpenCV Adaptive tokenizer: {e}")
    
    # Check for saved tokenizers
    for filename in os.listdir(base_dir):
        if filename.endswith('.yml') or filename.endswith('.yaml'):
            try:
                saved_path = os.path.join(base_dir, filename)
                saved_tokenizer = cv.dnn.Tokenizer_load(saved_path)
                name = os.path.splitext(filename)[0]
                tokenizers.append(TokenizerWrapper(f"opencv_{name}", saved_tokenizer, "opencv"))
                print(f"Loaded saved OpenCV tokenizer '{name}'")
            except Exception as e:
                print(f"Error loading saved tokenizer {filename}: {e}")
    
    return tokenizers

def load_comparison_tokenizers():
    """Load tokenizers from other frameworks for comparison"""
    tokenizers = []
    
    # Hugging Face tokenizers
    if HAVE_HUGGINGFACE:
        models = [
            ("gpt2", "GPT-2"),
            ("bert-base-uncased", "BERT"),
            ("t5-base", "T5"),
            ("facebook/bart-base", "BART"),
            ("xlm-roberta-base", "XLM-RoBERTa")
        ]
        
        for model_name, display_name in models:
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizers.append(TokenizerWrapper(f"hf_{display_name}", hf_tokenizer, "huggingface"))
                print(f"Loaded HuggingFace {display_name} tokenizer")
            except Exception as e:
                print(f"Error loading HuggingFace {display_name} tokenizer: {e}")
    
    # tiktoken tokenizers
    if HAVE_TIKTOKEN:
        encodings = [
            ("cl100k_base", "GPT-4"),
            ("p50k_base", "GPT-3"),
            ("r50k_base", "Davinci"),
            ("gpt2", "GPT-2-tiktoken")
        ]
        
        for enc_name, display_name in encodings:
            try:
                tiktoken_enc = tiktoken.get_encoding(enc_name)
                tokenizers.append(TokenizerWrapper(f"tiktoken_{display_name}", tiktoken_enc, "tiktoken"))
                print(f"Loaded tiktoken {display_name} tokenizer")
            except Exception as e:
                print(f"Error loading tiktoken {display_name} tokenizer: {e}")
    
    # SentencePiece tokenizers
    if HAVE_SENTENCEPIECE:
        # Check for common SentencePiece model locations
        sp_models = []
        
        # Add standard locations where sentencepiece models might be found
        search_paths = [
            "./models",
            os.path.expanduser("~/.cache/sentencepiece"),
            os.path.expanduser("~/.cache/torch/transformers")
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                for filename in os.listdir(path):
                    if filename.endswith('.model'):
                        sp_models.append((os.path.join(path, filename), os.path.splitext(filename)[0]))
        
        for model_path, display_name in sp_models:
            try:
                sp_tokenizer = spm.SentencePieceProcessor()
                sp_tokenizer.Load(model_path)
                tokenizers.append(TokenizerWrapper(f"sp_{display_name}", sp_tokenizer, "sentencepiece"))
                print(f"Loaded SentencePiece tokenizer from {model_path}")
            except Exception as e:
                print(f"Error loading SentencePiece tokenizer {display_name}: {e}")
    
    return tokenizers

def benchmark_tokenizer(tokenizer, texts, iterations=5):
    """Benchmark a tokenizer's performance"""
    results = {
        "name": tokenizer.name,
        "framework": tokenizer.framework,
        "vocab_size": tokenizer.get_vocab_size(),
        "encoding_times_ms": [],
        "decoding_times_ms": [],
        "tokens_per_text": [],
        "chars_per_token": []
    }
    
    # Warm-up run
    for text in texts[:min(3, len(texts))]:
        tokenizer.encode(text)
    
    # Benchmark encoding
    for _ in range(iterations):
        start_time = time.time()
        tokens_list = [tokenizer.encode(text) for text in texts]
        encoding_time = (time.time() - start_time) * 1000  # Convert to ms
        results["encoding_times_ms"].append(encoding_time)
        
        # Calculate tokens per text
        total_tokens = sum(len(tokens) for tokens in tokens_list)
        results["tokens_per_text"].append(total_tokens / len(texts))
        
        # Calculate characters per token
        total_chars = sum(len(text) for text in texts)
        results["chars_per_token"].append(total_chars / total_tokens if total_tokens else 0)
        
        # Benchmark decoding (using the tokens from the first iteration)
        if len(results["decoding_times_ms"]) == 0:
            start_time = time.time()
            for tokens in tokens_list:
                tokenizer.decode(tokens)
            decoding_time = (time.time() - start_time) * 1000  # Convert to ms
            results["decoding_times_ms"].append(decoding_time)
    
    # Calculate averages
    results["avg_encoding_time_ms"] = sum(results["encoding_times_ms"]) / iterations
    results["avg_decoding_time_ms"] = sum(results["decoding_times_ms"]) / len(results["decoding_times_ms"])
    results["avg_tokens_per_text"] = sum(results["tokens_per_text"]) / iterations
    results["avg_chars_per_token"] = sum(results["chars_per_token"]) / iterations
    
    # Calculate throughput
    total_chars = sum(len(text) for text in texts)
    results["encoding_throughput_chars_per_second"] = (total_chars * 1000) / results["avg_encoding_time_ms"]
    
    # Additional metrics for the first encoding only
    first_tokens_list = [tokenizer.encode(text) for text in texts]
    
    # Token distribution
    all_tokens = [token for tokens in first_tokens_list for token in tokens]
    token_frequencies = {}
    for token in all_tokens:
        token_frequencies[token] = token_frequencies.get(token, 0) + 1
    
    # Calculate token frequency statistics
    results["unique_tokens"] = len(token_frequencies)
    results["token_frequency_entropy"] = 0
    
    # Calculate entropy
    for count in token_frequencies.values():
        p = count / len(all_tokens)
        results["token_frequency_entropy"] -= p * np.log2(p) if p > 0 else 0
    
    return results

def benchmark_language_diversity(tokenizers, language_samples, iterations=3):
    """Benchmark tokenizers across different languages"""
    results = {}
    
    for tokenizer in tokenizers:
        results[tokenizer.name] = {}
        
        for lang_code, text in language_samples.items():
            # Create a small corpus by repeating the sample text
            corpus = [text] * 10
            
            # Run benchmark
            lang_result = benchmark_tokenizer(tokenizer, corpus, iterations)
            
            # Store only the most relevant metrics for languages
            results[tokenizer.name][lang_code] = {
                "chars_per_token": lang_result["avg_chars_per_token"],
                "tokens_per_char": 1.0 / lang_result["avg_chars_per_token"] if lang_result["avg_chars_per_token"] else 0,
                "encoding_time_ms": lang_result["avg_encoding_time_ms"] / 10,  # Per sample
                "unique_tokens": lang_result["unique_tokens"]
            }
    
    return results

def benchmark_technical_content(tokenizers, technical_samples, iterations=3):
    """Benchmark tokenizers on technical content"""
    results = {}
    
    for tokenizer in tokenizers:
        results[tokenizer.name] = {}
        
        for content_type, text in technical_samples.items():
            # Run benchmark
            sample_result = benchmark_tokenizer(tokenizer, [text], iterations)
            
            # Store results
            results[tokenizer.name][content_type] = {
                "chars_per_token": sample_result["avg_chars_per_token"],
                "tokens": sample_result["avg_tokens_per_text"],
                "encoding_time_ms": sample_result["avg_encoding_time_ms"],
                "unique_token_percentage": sample_result["unique_tokens"] / sample_result["avg_tokens_per_text"] * 100 if sample_result["avg_tokens_per_text"] else 0
            }
    
    return results

def generate_charts(results, output_dir):
    """Generate visualization charts from benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall performance comparison
    plt.figure(figsize=(12, 8))
    
    # Sort tokenizers by encoding throughput
    tokenizers_sorted = sorted(results, key=lambda x: results[x]["encoding_throughput_chars_per_second"], reverse=True)
    
    # Prepare data for plotting
    names = [name for name in tokenizers_sorted]
    throughputs = [results[name]["encoding_throughput_chars_per_second"] / 1000 for name in tokenizers_sorted]  # Convert to thousands
    
    # Create bar chart
    colors = ['#3498db' if 'opencv' in name else '#e74c3c' for name in names]
    plt.bar(names, throughputs, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Throughput (Thousand chars/second)')
    plt.title('Tokenizer Encoding Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_performance.png'), dpi=300)
    plt.close()
    
    # 2. Efficiency comparison (chars per token)
    plt.figure(figsize=(12, 8))
    
    # Sort tokenizers by chars per token (higher is more efficient)
    tokenizers_sorted = sorted(results, key=lambda x: results[x]["avg_chars_per_token"], reverse=True)
    
    # Prepare data for plotting
    names = [name for name in tokenizers_sorted]
    efficiency = [results[name]["avg_chars_per_token"] for name in tokenizers_sorted]
    
    # Create bar chart
    colors = ['#3498db' if 'opencv' in name else '#e74c3c' for name in names]
    plt.bar(names, efficiency, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Characters per token (higher is more efficient)')
    plt.title('Tokenizer Efficiency Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Token distribution entropy
    plt.figure(figsize=(12, 8))
    
    # Sort tokenizers by token entropy
    tokenizers_sorted = sorted(results, key=lambda x: results[x]["token_frequency_entropy"], reverse=True)
    
    # Prepare data for plotting
    names = [name for name in tokenizers_sorted]
    entropy = [results[name]["token_frequency_entropy"] for name in tokenizers_sorted]
    
    # Create bar chart
    colors = ['#3498db' if 'opencv' in name else '#e74c3c' for name in names]
    plt.bar(names, entropy, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Token Frequency Entropy (higher is more diverse)')
    plt.title('Token Distribution Diversity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_entropy.png'), dpi=300)
    plt.close()
    
    return os.path.join(output_dir, 'overall_performance.png')

def generate_language_comparison_chart(language_results, output_dir):
    """Generate charts for language comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for easier plotting
    data = []
    for tokenizer_name, lang_results in language_results.items():
        for lang_code, metrics in lang_results.items():
            data.append({
                "tokenizer": tokenizer_name,
                "language": lang_code,
                "tokens_per_char": metrics["tokens_per_char"],
                "encoding_time_ms": metrics["encoding_time_ms"]
            })
    
    df = pd.DataFrame(data)
    
    # 1. Tokens per character across languages
    plt.figure(figsize=(14, 10))
    
    # Pivot data for heatmap
    pivot_data = df.pivot(index="tokenizer", columns="language", values="tokens_per_char")
    
    # Plot heatmap
    plt.imshow(pivot_data.values, cmap='viridis')
    plt.colorbar(label='Tokens per character (lower is more efficient)')
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns, rotation=45)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)
    plt.title('Tokenization Efficiency Across Languages')
    
    # Add values to cells
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            try:
                value = pivot_data.values[i, j]
                plt.text(j, i, f'{value:.2f}', ha='center', va='center', 
                         color='white' if value > 0.5 else 'black')
            except:
                pass
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_efficiency.png'), dpi=300)
    plt.close()
    
    # 2. Language efficiency radar chart for each tokenizer
    for tokenizer_name in language_results.keys():
        plt.figure(figsize=(10, 10))
        
        # Get data for this tokenizer
        tokenizer_data = {k: v["tokens_per_char"] for k, v in language_results[tokenizer_name].items()}
        
        # Prepare data for radar chart
        categories = list(tokenizer_data.keys())
        values = [tokenizer_data[cat] for cat in categories]
        
        # Number of variables
        N = len(categories)
        
        # Create angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add values
        values += values[:1]  # Close the loop
        
        # Create radar plot
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories)
        
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, alpha=0.1)
        
        # Add title
        plt.title(f'Language Efficiency: {tokenizer_name}')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'radar_{tokenizer_name}.png'), dpi=300)
        plt.close()
    
    return os.path.join(output_dir, 'language_efficiency.png')

def generate_technical_content_chart(technical_results, output_dir):
    """Generate charts for technical content comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for easier plotting
    data = []
    for tokenizer_name, content_results in technical_results.items():
        for content_type, metrics in content_results.items():
            data.append({
                "tokenizer": tokenizer_name,
                "content_type": content_type,
                "chars_per_token": metrics["chars_per_token"],
                "unique_token_percentage": metrics["unique_token_percentage"]
            })
    
    df = pd.DataFrame(data)
    
    # Technical content efficiency comparison
    plt.figure(figsize=(14, 10))
    
    # Pivot data for heatmap
    pivot_data = df.pivot(index="tokenizer", columns="content_type", values="chars_per_token")
    
    # Plot heatmap
    plt.imshow(pivot_data.values, cmap='plasma')
    plt.colorbar(label='Characters per token (higher is more efficient)')
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns, rotation=45)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)
    plt.title('Tokenization Efficiency on Technical Content')
    
    # Add values to cells
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            try:
                value = pivot_data.values[i, j]
                plt.text(j, i, f'{value:.2f}', ha='center', va='center', 
                         color='white' if value < 2.0 else 'black')
            except:
                pass
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'technical_content_efficiency.png'), dpi=300)
    plt.close()
    
    return os.path.join(output_dir, 'technical_content_efficiency.png')

def export_to_csv(results, output_file):
    """Export benchmark results to CSV file"""
    with open(output_file, 'w', newline='') as csvfile:
        # Determine all possible fields from results
        all_fields = set()
        for tokenizer_result in results.values():
            all_fields.update(tokenizer_result.keys())
        
        # Create CSV writer
        fieldnames = ['name', 'framework', 'vocab_size'] + sorted(list(all_fields - {'name', 'framework', 'vocab_size', 'encoding_times_ms', 'decoding_times_ms', 'tokens_per_text', 'chars_per_token'}))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for tokenizer_name, result in results.items():
            # Prepare row data
            row = {'name': tokenizer_name}
            for field in fieldnames:
                if field in result:
                    row[field] = result[field]
            
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='OpenCV LLM Tokenizer Benchmark and Research Suite')
    parser.add_argument('--tokenizer-dir', type=str, default='./tokenizers',
                        help='Directory containing tokenizer files')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Directory to save benchmark results')
    parser.add_argument('--corpus', type=str, default=None,
                        help='Path to custom corpus file for benchmarking')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations for benchmarking')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with other tokenizer frameworks')
    parser.add_argument('--languages', action='store_true',
                        help='Run multilingual benchmarking')
    parser.add_argument('--technical', action='store_true',
                        help='Run technical content benchmarking')
    parser.add_argument('--all', action='store_true',
                        help='Run all benchmark tests')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizers
    opencv_tokenizers = load_opencv_tokenizers(args.tokenizer_dir)
    
    if len(opencv_tokenizers) == 0:
        print(f"Error: No OpenCV tokenizers found in {args.tokenizer_dir}")
        return 1
    
    # Load comparison tokenizers if requested
    comparison_tokenizers = []
    if args.compare or args.all:
        comparison_tokenizers = load_comparison_tokenizers()
    
    all_tokenizers = opencv_tokenizers + comparison_tokenizers
    
    # Load or create benchmark corpus
    corpus = []
    if args.corpus:
        with open(args.corpus, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(corpus)} lines from corpus file")
    else:
        # Create a synthetic benchmark corpus
        import string
        
        # Basic English text
        corpus.extend([
            "This is a simple benchmark text for tokenizer testing.",
            "It contains various sentence structures and punctuation!",
            "Numbers like 123, 45.67, and 3.14159 are included.",
            "Special characters: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "URLs and emails: https://example.com and user@example.com"
        ])
        
        # Add longer paragraphs
        corpus.append("Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.")
        
        # Add technical content
        corpus.append("def tokenize(text):\n    return [token for token in text.split()]\n\nclass Tokenizer:\n    def __init__(self):\n        self.vocab = {}")
        
        # Add repetitive patterns
        corpus.append("a b c d e f g h i j k l m n o p q r s t u v w x y z")
        corpus.append("aaaaabbbbbcccccdddddeeeeefffff")
        
        print(f"Created synthetic benchmark corpus with {len(corpus)} entries")
    
    # Run main benchmark
    print("\nRunning main tokenizer benchmark...")
    results = {}
    
    for tokenizer in tqdm(all_tokenizers):
        try:
            result = benchmark_tokenizer(tokenizer, corpus, args.iterations)
            results[tokenizer.name] = result
        except Exception as e:
            print(f"Error benchmarking {tokenizer.name}: {e}")
    
    # Export results to CSV
    csv_path = os.path.join(args.output_dir, 'benchmark_results.csv')
    export_to_csv(results, csv_path)
    print(f"Exported benchmark results to {csv_path}")
    
    # Generate charts
    chart_path = generate_charts(results, args.output_dir)
    print(f"Generated performance charts in {args.output_dir}")
    
    # Run language diversity benchmark if requested
    if args.languages or args.all:
        print("\nRunning language diversity benchmark...")
        language_results = benchmark_language_diversity(all_tokenizers, LANGUAGE_SAMPLES)
        
        # Export language results
        lang_csv = os.path.join(args.output_dir, 'language_benchmark.csv')
        with open(lang_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Tokenizer', 'Language', 'Chars per Token', 'Tokens per Char', 'Encoding Time (ms)'])
            
            for tokenizer_name, lang_data in language_results.items():
                for lang, metrics in lang_data.items():
                    writer.writerow([
                        tokenizer_name, 
                        lang, 
                        metrics["chars_per_token"],
                        metrics["tokens_per_char"],
                        metrics["encoding_time_ms"]
                    ])
        
        # Generate language comparison charts
        lang_chart = generate_language_comparison_chart(language_results, os.path.join(args.output_dir, 'languages'))
        print(f"Generated language comparison charts in {os.path.join(args.output_dir, 'languages')}")
    
    # Run technical content benchmark if requested
    if args.technical or args.all:
        print("\nRunning technical content benchmark...")
        technical_results = benchmark_technical_content(all_tokenizers, TECHNICAL_SAMPLES)
        
        # Export technical results
        tech_csv = os.path.join(args.output_dir, 'technical_benchmark.csv')
        with open(tech_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Tokenizer', 'Content Type', 'Chars per Token', 'Total Tokens', 'Unique Token %'])
            
            for tokenizer_name, content_data in technical_results.items():
                for content_type, metrics in content_data.items():
                    writer.writerow([
                        tokenizer_name, 
                        content_type, 
                        metrics["chars_per_token"],
                        metrics["tokens"],
                        metrics["unique_token_percentage"]
                    ])
        
        # Generate technical comparison charts
        tech_chart = generate_technical_content_chart(technical_results, os.path.join(args.output_dir, 'technical'))
        print(f"Generated technical content comparison charts in {os.path.join(args.output_dir, 'technical')}")
    
    print("\nBenchmark complete! Results saved to", args.output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())