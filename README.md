# OpenCV Tokenizer Integration for LLM Support

This project implements a comprehensive tokenizer framework for Large Language Models (LLMs) in OpenCV's DNN module, enabling seamless text processing capabilities for multimodal AI applications.

## Overview

Tokenizers convert raw text into numerical token IDs that language models can process. This implementation provides native support for multiple tokenization algorithms used by popular LLMs, optimized for performance with SIMD instructions and GPU acceleration.

## Features

- **Complete Tokenization Ecosystem**:
  - Byte-Pair Encoding (BPE) - used by GPT models
  - WordPiece - used by BERT and other transformer models
  - Unigram - used by SentencePiece and T5 models
  - SentencePiece - direct integration with SentencePiece models
  - TikToken - compatible with OpenAI's tokenizers
  - Transformers - integration with Hugging Face Transformers
  - AdaptiveTokenizer - innovative multilingual tokenization

- **Performance Optimizations**:
  - SIMD acceleration for CPU operations
  - CUDA-based GPU acceleration for batch processing
  - Multi-threaded implementation for high throughput

- **Comprehensive API**:
  - Common interface for all tokenizer types
  - Detailed token information with positions and scores
  - Batch processing capabilities
  - Serialization and deserialization support

- **Multilingual Support**:
  - Script detection and adaptive tokenization strategies
  - Support for Latin, Cyrillic, CJK, and many other scripts
  - Language detection capabilities

- **Analysis Tools**:
  - Tokenization visualization
  - Performance benchmarking framework
  - Tokenization analysis metrics

## Code Structure

```
modules/dnn/
├── include/opencv2/dnn/
│   ├── tokenizer.hpp            # Main tokenizer interface
│   └── gpu_tokenizer.hpp        # GPU acceleration interface
├── src/
│   ├── tokenizer/
│   │   ├── tokenizer.cpp        # Base implementation
│   │   ├── bpe_tokenizer.cpp/hpp    # BPE implementation
│   │   ├── wordpiece_tokenizer.cpp/hpp  # WordPiece implementation
│   │   ├── unigram_tokenizer.cpp/hpp    # Unigram implementation
│   │   ├── sentencepiece_tokenizer.cpp/hpp  # SentencePiece implementation
│   │   ├── tiktoken_tokenizer.cpp/hpp  # TikToken implementation
│   │   ├── transformers_tokenizer.cpp/hpp  # Hugging Face integration
│   │   ├── adaptive_tokenizer.cpp/hpp   # Multi-script tokenizer
│   │   ├── tokenizer_simd.cpp/hpp   # SIMD optimizations
│   │   └── tokenization_analysis.cpp/hpp  # Analysis tools
│   └── gpu_tokenizer.cpp        # GPU acceleration implementation
├── python/
│   └── cv2_dnn_tokenizer.cpp    # Python bindings
├── tests/
│   ├── test_tokenizer.cpp        # Base tokenizer tests
│   ├── test_adaptive_tokenizer.cpp  # Adaptive tokenizer tests
│   ├── test_gpu_tokenizer.cpp    # GPU tokenizer tests
│   └── test_simd_tokenizer.cpp   # SIMD optimizations tests
```

## Usage Examples

### Basic Tokenization (C++)

```cpp
#include <opencv2/dnn/tokenizer.hpp>

// Create a BPE tokenizer
cv::Ptr<cv::dnn::Tokenizer> tokenizer = cv::dnn::createBPETokenizer("vocab.json", "merges.txt");

// Encode text to token IDs
std::string text = "Hello world, this is a test!";
std::vector<int> tokens = tokenizer->encode(text);

// Decode token IDs back to text
std::string decoded = tokenizer->decode(tokens);

// Get detailed token information
std::vector<cv::dnn::TokenInfo> token_info = tokenizer->encodeWithInfo(text);
```

### TikToken and SentencePiece Examples

```cpp
// Create a TikToken tokenizer
cv::Ptr<cv::dnn::Tokenizer> tiktoken = cv::dnn::createTikTokenTokenizer("cl100k_base");

// Create a SentencePiece tokenizer
cv::Ptr<cv::dnn::Tokenizer> sp_tokenizer = cv::dnn::createSentencePieceTokenizer("model.model");

// Create a Transformers tokenizer
cv::Ptr<cv::dnn::Tokenizer> hf_tokenizer = cv::dnn::createTransformersTokenizer("gpt2");
```

### AdaptiveTokenizer for Multilingual Text

```cpp
// Create an AdaptiveTokenizer that switches strategies based on script
cv::Ptr<cv::dnn::Tokenizer> adaptive = cv::dnn::createAdaptiveTokenizer(
    bpe_tokenizer,   // For Latin script
    unigram_tokenizer,  // For syllabic scripts
    char_tokenizer   // For ideographic scripts
);

// Encode multilingual text
std::string multilingual = "Hello 世界 and こんにちは";
std::vector<int> tokens = adaptive->encode(multilingual);
```

### Using GPU Acceleration

```cpp
// Create a GPU tokenizer based on a CPU tokenizer
cv::Ptr<cv::dnn::GPUTokenizer> gpu_tokenizer = cv::dnn::createGPUTokenizer(tokenizer);

// Prepare a batch of texts
std::vector<std::string> texts = {"Hello world", "Second example", "Third example"};

// Encode batch with GPU acceleration
std::vector<std::vector<int>> batch_tokens = gpu_tokenizer->encodeBatch(texts);

// Direct encoding to GPU memory
cv::cuda::GpuMat tokens_gpu = gpu_tokenizer->encodeBatchToGpuMat(texts);
```

### Python Example

```python
import cv2 as cv

# Create a WordPiece tokenizer
tokenizer = cv.dnn.createWordPieceTokenizer("vocab.txt")

# Encode text
text = "OpenCV now supports tokenizers for LLMs."
tokens = tokenizer.encode(text)
print(f"Encoded into {len(tokens)} tokens: {tokens}")

# Decode back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded text: {decoded}")

# Use TikToken
tiktoken = cv.dnn.createTikTokenTokenizer("cl100k_base")
tiktoken_tokens = tiktoken.encode("This is encoded with TikToken")
```

## Performance Considerations

The tokenizer implementation is optimized for both accuracy and speed:
- SIMD instructions accelerate CPU operations where available
- GPU acceleration provides significant speedup for batch processing
- Adaptive strategies improve efficiency for multilingual text

For specific performance metrics in your environment, use the included benchmark tools:
```bash
# C++ benchmark
./tokenizer_benchmark --tokenizer=bpe --vocab=vocab.json --merges=merges.txt --batch_size=128

# Python benchmark
python benchmark_and_research_suite.py --tokenizer-dir=./tokenizers --batch_size=128
```

## Sample Applications

The project includes several sample applications demonstrating tokenizer usage:

- `tokenizer_demo.cpp/.py`: Basic tokenizer usage examples
- `tokenizer_benchmark.cpp`: Performance benchmarking tool
- `tokenizer_visualization.py`: Interactive visualization of tokenization
- `tokenizer_comparison.cpp`: Compare different tokenizer implementations
- `ocr_llm_demo.cpp`: Integration of OCR with tokenizers and LLMs

## Building and Integration

This implementation is designed to be integrated directly into the OpenCV codebase. Simply include the source files in your OpenCV build.

### Requirements

- OpenCV 4.x
- CMake 3.10+
- C++14 compiler
- CUDA 10.0+ (optional, for GPU acceleration)
- Python 3.6+ (for Python bindings)

## License

This project is licensed under the same terms as OpenCV - Apache 2.0 License.

## Acknowledgments

- This work builds upon tokenization approaches from various LLM frameworks
- Inspired by implementations in Hugging Face Transformers, TikToken, and SentencePiece
- Special thanks to the OpenCV community

--

For more information, see the Doxygen ducumentation by running [Doxyfile](https://github.com/RiturajSingh2004/opencv-llm-tokenizer/blob/main/docs/Doxyfile)

---

*Developed for OpenMS as part of Google Summer of Code*
