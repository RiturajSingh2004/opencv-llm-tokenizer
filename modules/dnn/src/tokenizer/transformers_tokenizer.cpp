/**
 * @file transformers_tokenizer.cpp
 * @brief Implementation of Hugging Face Transformers-compatible tokenizer
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements integration with Hugging Face Transformers tokenizers,
 * providing a bridge between OpenCV and the popular transformer model ecosystem.
 * It supports loading pre-trained tokenizers and model-specific processing options.
 */

#include "precomp.hpp"
#include "transformers_tokenizer.hpp"

// For Python embedding
#include <Python.h>
#include <numpy/ndarrayobject.h>

namespace cv {
namespace dnn {

// Utility functions for Python error handling
static std::string getPythonError() {
    if (!PyErr_Occurred())
        return "Unknown error";
    
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
    
    std::string error_msg = "Python error: ";
    
    if (pvalue) {
        PyObject* str_exc_value = PyObject_Str(pvalue);
        if (str_exc_value) {
            const char* c_str_exc_value = PyUnicode_AsUTF8(str_exc_value);
            if (c_str_exc_value)
                error_msg += c_str_exc_value;
            Py_DECREF(str_exc_value);
        }
    }
    
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    
    return error_msg;
}

static void handlePyError(const std::string& context) {
    if (PyErr_Occurred()) {
        std::string error_msg = getPythonError();
        PyErr_Clear();
        CV_Error(Error::StsError, context + ": " + error_msg);
    }
}

// Wrapper class to hide Python/Transformers implementation details
class TransformersWrapper {
public:
    TransformersWrapper(const std::string& model_name_or_path, bool use_fast) {
        // Initialize Python interpreter if not already initialized
        if (!Py_IsInitialized()) {
            Py_Initialize();
            import_array(); // Initialize NumPy
        }
        
        try {
            // Import required modules
            PyObject* transformers_module = PyImport_ImportModule("transformers");
            if (!transformers_module) {
                throw std::runtime_error("Failed to import transformers module. Make sure it's installed.");
            }
            
            // Get AutoTokenizer class
            PyObject* auto_tokenizer_class = PyObject_GetAttrString(transformers_module, "AutoTokenizer");
            if (!auto_tokenizer_class) {
                Py_DECREF(transformers_module);
                throw std::runtime_error("Failed to get AutoTokenizer class.");
            }
            
            // Call from_pretrained method
            PyObject* from_pretrained = PyObject_GetAttrString(auto_tokenizer_class, "from_pretrained");
            Py_DECREF(auto_tokenizer_class);
            
            if (!from_pretrained) {
                Py_DECREF(transformers_module);
                throw std::runtime_error("Failed to get from_pretrained method.");
            }
            
            // Build args
            PyObject* args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, PyUnicode_FromString(model_name_or_path.c_str()));
            
            // Build kwargs
            PyObject* kwargs = PyDict_New();
            PyDict_SetItemString(kwargs, "use_fast", use_fast ? Py_True : Py_False);
            
            // Call method
            tokenizer_ = PyObject_Call(from_pretrained, args, kwargs);
            Py_DECREF(from_pretrained);
            Py_DECREF(args);
            Py_DECREF(kwargs);
            
            if (!tokenizer_) {
                Py_DECREF(transformers_module);
                throw std::runtime_error("Failed to load tokenizer from " + model_name_or_path);
            }
            
            // Get tokenizer type
            PyObject* type_obj = PyObject_GetAttrString(PyObject_Type(tokenizer_), "__name__");
            if (type_obj) {
                tokenizer_type_ = PyUnicode_AsUTF8(type_obj);
                Py_DECREF(type_obj);
            }
            
            // Get vocabulary
            PyObject* vocab = PyObject_GetAttrString(tokenizer_, "get_vocab");
            if (vocab) {
                PyObject* vocab_dict = PyObject_CallObject(vocab, NULL);
                if (vocab_dict) {
                    PyObject *key, *value;
                    Py_ssize_t pos = 0;
                    
                    while (PyDict_Next(vocab_dict, &pos, &key, &value)) {
                        const char* token = PyUnicode_AsUTF8(key);
                        int id = PyLong_AsLong(value);
                        
                        if (token) {
                            token_to_id_[token] = id;
                            id_to_token_[id] = token;
                        }
                    }
                    
                    Py_DECREF(vocab_dict);
                }
                Py_DECREF(vocab);
            }
            
            // Check if it's a fast tokenizer
            PyObject* is_fast_attr = PyObject_GetAttrString(tokenizer_, "is_fast");
            if (is_fast_attr) {
                is_fast_ = PyObject_IsTrue(is_fast_attr);
                Py_DECREF(is_fast_attr);
            }
            
            Py_DECREF(transformers_module);
        } catch (const std::exception& e) {
            if (PyErr_Occurred()) {
                PyObject *ptype, *pvalue, *ptraceback;
                PyErr_Fetch(&ptype, &pvalue, &ptraceback);
                
                if (pvalue) {
                    PyObject* pstr = PyObject_Str(pvalue);
                    if (pstr) {
                        const char* error_msg = PyUnicode_AsUTF8(pstr);
                        error_message_ = error_msg ? error_msg : e.what();
                        Py_DECREF(pstr);
                    } else {
                        error_message_ = e.what();
                    }
                    Py_DECREF(pvalue);
                } else {
                    error_message_ = e.what();
                }
                
                Py_XDECREF(ptype);
                Py_XDECREF(ptraceback);
            } else {
                error_message_ = e.what();
            }
            
            throw std::runtime_error("TransformersWrapper initialization error: " + error_message_);
        }
    }
    
    ~TransformersWrapper() {
        Py_XDECREF(tokenizer_);
    }
    
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> result;
        
        PyObject* py_text = PyUnicode_FromString(text.c_str());
        if (!py_text) {
            throw std::runtime_error("Failed to convert text to Python string.");
        }
        
        // Call encode method
        PyObject* encoded = PyObject_CallMethod(tokenizer_, "encode", "(O)", py_text);
        Py_DECREF(py_text);
        
        if (!encoded) {
            handlePyError("Failed to encode text");
            return result;
        }
        
        // Convert result to C++ vector
        Py_ssize_t size = PyList_Size(encoded);
        result.reserve(size);
        
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* item = PyList_GetItem(encoded, i);
            result.push_back(PyLong_AsLong(item));
        }
        
        Py_DECREF(encoded);
        return result;
    }
    
    std::vector<TokenInfo> encodeWithInfo(const std::string& text) const {
        std::vector<TokenInfo> result;
        
        // Call tokenize to get token info
        PyObject* py_text = PyUnicode_FromString(text.c_str());
        if (!py_text) {
            throw std::runtime_error("Failed to convert text to Python string.");
        }
        
        // Create kwargs with return_offsets_mapping=True
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "return_offsets_mapping", Py_True);
        PyDict_SetItemString(kwargs, "return_tensors", Py_None);
        
        PyObject* tokenized = PyObject_Call(
            PyObject_GetAttrString(tokenizer_, "encode_plus"),
            PyTuple_Pack(1, py_text),
            kwargs
        );
        
        Py_DECREF(py_text);
        Py_DECREF(kwargs);
        
        if (!tokenized) {
            handlePyError("Failed to tokenize text with offset mapping");
            return result;
        }
        
        // Extract input_ids
        PyObject* input_ids = PyDict_GetItemString(tokenized, "input_ids");
        if (!input_ids) {
            Py_DECREF(tokenized);
            throw std::runtime_error("No input_ids in tokenizer output");
        }
        
        // Extract offset_mapping
        PyObject* offset_mapping = PyDict_GetItemString(tokenized, "offset_mapping");
        bool has_offsets = offset_mapping && PyList_Check(offset_mapping);
        
        // Convert to TokenInfo objects
        Py_ssize_t size = PyList_Size(input_ids);
        result.reserve(size);
        
        for (Py_ssize_t i = 0; i < size; i++) {
            TokenInfo info;
            
            // Get token ID
            PyObject* token_id = PyList_GetItem(input_ids, i);
            info.id = PyLong_AsLong(token_id);
            
            // Get token text
            info.text = getTokenText(info.id);
            
            // Get offsets if available
            if (has_offsets) {
                PyObject* offset_tuple = PyList_GetItem(offset_mapping, i);
                if (PyTuple_Check(offset_tuple) && PyTuple_Size(offset_tuple) == 2) {
                    info.start = PyLong_AsLong(PyTuple_GetItem(offset_tuple, 0));
                    info.end = PyLong_AsLong(PyTuple_GetItem(offset_tuple, 1));
                } else {
                    // Default values if not a proper tuple
                    info.start = -1;
                    info.end = -1;
                }
            } else {
                // No offset information available
                info.start = -1;
                info.end = -1;
            }
            
            // Set default score
            info.score = 1.0f;
            
            result.push_back(info);
        }
        
        Py_DECREF(tokenized);
        return result;
    }
    
    std::vector<std::vector<int>> encodeBatch(const std::vector<std::string>& texts) const {
        std::vector<std::vector<int>> result;
        result.reserve(texts.size());
        
        // Convert texts to Python list
        PyObject* py_texts = PyList_New(texts.size());
        for (size_t i = 0; i < texts.size(); i++) {
            PyObject* py_text = PyUnicode_FromString(texts[i].c_str());
            if (!py_text) {
                Py_DECREF(py_texts);
                throw std::runtime_error("Failed to convert text to Python string.");
            }
            PyList_SetItem(py_texts, i, py_text);  // PyList_SetItem steals reference
        }
        
        // Call batch_encode_plus method
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "padding", Py_False);
        PyDict_SetItemString(kwargs, "return_tensors", Py_None);
        
        PyObject* batch_encoded = PyObject_Call(
            PyObject_GetAttrString(tokenizer_, "batch_encode_plus"),
            PyTuple_Pack(1, py_texts),
            kwargs
        );
        
        Py_DECREF(py_texts);
        Py_DECREF(kwargs);
        
        if (!batch_encoded) {
            handlePyError("Failed to batch encode texts");
            return result;
        }
        
        // Extract input_ids
        PyObject* input_ids = PyDict_GetItemString(batch_encoded, "input_ids");
        if (!input_ids || !PyList_Check(input_ids)) {
            Py_DECREF(batch_encoded);
            throw std::runtime_error("No valid input_ids in batch encoder output");
        }
        
        // Convert to vectors of ints
        Py_ssize_t batch_size = PyList_Size(input_ids);
        for (Py_ssize_t i = 0; i < batch_size; i++) {
            PyObject* tokens = PyList_GetItem(input_ids, i);
            if (!PyList_Check(tokens)) {
                continue;
            }
            
            std::vector<int> token_ids;
            Py_ssize_t num_tokens = PyList_Size(tokens);
            token_ids.reserve(num_tokens);
            
            for (Py_ssize_t j = 0; j < num_tokens; j++) {
                PyObject* token = PyList_GetItem(tokens, j);
                token_ids.push_back(PyLong_AsLong(token));
            }
            
            result.push_back(token_ids);
        }
        
        Py_DECREF(batch_encoded);
        return result;
    }
    
    std::string decode(const std::vector<int>& tokens) const {
        // Convert tokens to Python list
        PyObject* py_tokens = PyList_New(tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
            PyObject* py_token = PyLong_FromLong(tokens[i]);
            PyList_SetItem(py_tokens, i, py_token);  // PyList_SetItem steals reference
        }
        
        // Call decode method
        PyObject* decoded = PyObject_CallMethod(tokenizer_, "decode", "(O)", py_tokens);
        Py_DECREF(py_tokens);
        
        if (!decoded) {
            handlePyError("Failed to decode tokens");
            return "";
        }
        
        // Convert result to C++ string
        const char* decoded_str = PyUnicode_AsUTF8(decoded);
        std::string result = decoded_str ? decoded_str : "";
        
        Py_DECREF(decoded);
        return result;
    }
    
    std::string getTokenText(int tokenId) const {
        auto it = id_to_token_.find(tokenId);
        if (it != id_to_token_.end()) {
            return it->second;
        }
        
        // If not in cache, try to get from tokenizer
        PyObject* py_token_id = PyLong_FromLong(tokenId);
        PyObject* token_str = PyObject_CallMethod(tokenizer_, "convert_ids_to_tokens", "(O)", py_token_id);
        Py_DECREF(py_token_id);
        
        if (!token_str) {
            PyErr_Clear();
            return "";
        }
        
        const char* result_str = PyUnicode_AsUTF8(token_str);
        std::string result = result_str ? result_str : "";
        
        Py_DECREF(token_str);
        return result;
    }
    
    int getTokenId(const std::string& tokenText) const {
        auto it = token_to_id_.find(tokenText);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        
        // If not in cache, try to get from tokenizer
        PyObject* py_token_text = PyUnicode_FromString(tokenText.c_str());
        PyObject* token_id = PyObject_CallMethod(tokenizer_, "convert_tokens_to_ids", "(O)", py_token_text);
        Py_DECREF(py_token_text);
        
        if (!token_id) {
            PyErr_Clear();
            return -1;
        }
        
        int result = PyLong_AsLong(token_id);
        Py_DECREF(token_id);
        
        return result;
    }
    
    std::vector<int> encodeForModel(
        const std::string& text,
        int max_length = 512,
        bool padding = false,
        bool truncation = true,
        bool add_special_tokens = true) const {
        
        PyObject* py_text = PyUnicode_FromString(text.c_str());
        if (!py_text) {
            throw std::runtime_error("Failed to convert text to Python string.");
        }
        
        // Create kwargs
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "max_length", PyLong_FromLong(max_length));
        PyDict_SetItemString(kwargs, "padding", padding ? Py_True : Py_False);
        PyDict_SetItemString(kwargs, "truncation", truncation ? Py_True : Py_False);
        PyDict_SetItemString(kwargs, "add_special_tokens", add_special_tokens ? Py_True : Py_False);
        PyDict_SetItemString(kwargs, "return_tensors", Py_None);
        
        // Call encode_plus method
        PyObject* encoded = PyObject_Call(
            PyObject_GetAttrString(tokenizer_, "encode_plus"),
            PyTuple_Pack(1, py_text),
            kwargs
        );
        
        Py_DECREF(py_text);
        Py_DECREF(kwargs);
        
        if (!encoded) {
            handlePyError("Failed to encode text for model");
            return {};
        }
        
        // Extract input_ids
        PyObject* input_ids = PyDict_GetItemString(encoded, "input_ids");
        if (!input_ids || !PyList_Check(input_ids)) {
            Py_DECREF(encoded);
            throw std::runtime_error("No valid input_ids in encoder output");
        }
        
        // Convert to vector of ints
        std::vector<int> result;
        Py_ssize_t size = PyList_Size(input_ids);
        result.reserve(size);
        
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* token = PyList_GetItem(input_ids, i);
            result.push_back(PyLong_AsLong(token));
        }
        
        Py_DECREF(encoded);
        return result;
    }
    
    std::vector<std::vector<int>> encodeBatchForModel(
        const std::vector<std::string>& texts,
        int max_length = 512,
        bool padding = false,
        bool truncation = true,
        bool add_special_tokens = true) const {
        
        std::vector<std::vector<int>> result;
        result.reserve(texts.size());
        
        // Convert texts to Python list
        PyObject* py_texts = PyList_New(texts.size());
        for (size_t i = 0; i < texts.size(); i++) {
            PyObject* py_text = PyUnicode_FromString(texts[i].c_str());
            if (!py_text) {
                Py_DECREF(py_texts);
                throw std::runtime_error("Failed to convert text to Python string.");
            }
            PyList_SetItem(py_texts, i, py_text);  // PyList_SetItem steals reference
        }
        
        // Create kwargs
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "max_length", PyLong_FromLong(max_length));
        PyDict_SetItemString(kwargs, "padding", padding ? Py_True : Py_False);
        PyDict_SetItemString(kwargs, "truncation", truncation ? Py_True : Py_False);
        PyDict_SetItemString(kwargs, "add_special_tokens", add_special_tokens ? Py_True : Py_False);
        PyDict_SetItemString(kwargs, "return_tensors", Py_None);
        
        // Call batch_encode_plus method
        PyObject* batch_encoded = PyObject_Call(
            PyObject_GetAttrString(tokenizer_, "batch_encode_plus"),
            PyTuple_Pack(1, py_texts),
            kwargs
        );
        
        Py_DECREF(py_texts);
        Py_DECREF(kwargs);
        
        if (!batch_encoded) {
            handlePyError("Failed to batch encode texts for model");
            return result;
        }
        
        // Extract input_ids
        PyObject* input_ids = PyDict_GetItemString(batch_encoded, "input_ids");
        if (!input_ids || !PyList_Check(input_ids)) {
            Py_DECREF(batch_encoded);
            throw std::runtime_error("No valid input_ids in batch encoder output");
        }
        
        // Convert to vectors of ints
        Py_ssize_t batch_size = PyList_Size(input_ids);
        for (Py_ssize_t i = 0; i < batch_size; i++) {
            PyObject* tokens = PyList_GetItem(input_ids, i);
            if (!PyList_Check(tokens)) {
                continue;
            }
            
            std::vector<int> token_ids;
            Py_ssize_t num_tokens = PyList_Size(tokens);
            token_ids.reserve(num_tokens);
            
            for (Py_ssize_t j = 0; j < num_tokens; j++) {
                PyObject* token = PyList_GetItem(tokens, j);
                token_ids.push_back(PyLong_AsLong(token));
            }
            
            result.push_back(token_ids);
        }
        
        Py_DECREF(batch_encoded);
        return result;
    }
    
    size_t getVocabSize() const {
        return token_to_id_.size();
    }
    
    bool isFast() const {
        return is_fast_;
    }
    
    std::string getTokenizerType() const {
        return tokenizer_type_;
    }
    
private:
    PyObject* tokenizer_ = nullptr;
    std::string tokenizer_type_ = "Unknown";
    std::string error_message_;
    bool is_fast_ = false;
    
    // Caches for token<->id mapping
    mutable std::unordered_map<std::string, int> token_to_id_;
    mutable std::unordered_map<int, std::string> id_to_token_;
};

// TransformersTokenizer implementation
TransformersTokenizer::TransformersTokenizer(const std::string& model_name_or_path, bool use_fast)
    : model_name_or_path_(model_name_or_path), use_fast_(use_fast) {
    
    wrapper_ = std::make_unique<TransformersWrapper>(model_name_or_path, use_fast);
}

TransformersTokenizer::~TransformersTokenizer() = default;

std::vector<int> TransformersTokenizer::encode(const std::string& text) const {
    return wrapper_->encode(text);
}

std::vector<TokenInfo> TransformersTokenizer::encodeWithInfo(const std::string& text) const {
    return wrapper_->encodeWithInfo(text);
}

std::vector<std::vector<int>> TransformersTokenizer::encodeBatch(const std::vector<std::string>& texts) const {
    return wrapper_->encodeBatch(texts);
}

std::string TransformersTokenizer::decode(const std::vector<int>& tokens) const {
    return wrapper_->decode(tokens);
}

size_t TransformersTokenizer::getVocabSize() const {
    return wrapper_->getVocabSize();
}

void TransformersTokenizer::save(const std::string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    
    fs << "tokenizer_type" << "transformers";
    fs << "model_name_or_path" << model_name_or_path_;
    fs << "use_fast" << use_fast_;
    fs << "tokenizer_impl" << wrapper_->getTokenizerType();
    fs << "vocab_size" << static_cast<int>(wrapper_->getVocabSize());
}

void TransformersTokenizer::exportTo(const std::string& filename, const std::string& format) const {
    if (format == "tiktoken") {
        CV_Error(Error::StsNotImplemented, "Export to tiktoken format is not yet implemented for TransformersTokenizer");
    } else if (format == "sentencepiece") {
        CV_Error(Error::StsNotImplemented, "Export to sentencepiece format is not yet implemented for TransformersTokenizer");
    } else {
        CV_Error(Error::StsBadArg, "Unsupported export format: " + format);
    }
}

std::string TransformersTokenizer::getTokenText(int tokenId) const {
    return wrapper_->getTokenText(tokenId);
}

int TransformersTokenizer::getTokenId(const std::string& tokenText) const {
    return wrapper_->getTokenId(tokenText);
}

std::vector<int> TransformersTokenizer::encodeForModel(
    const std::string& text,
    int max_length,
    bool padding,
    bool truncation,
    bool add_special_tokens) const {
    
    return wrapper_->encodeForModel(text, max_length, padding, truncation, add_special_tokens);
}

std::vector<std::vector<int>> TransformersTokenizer::encodeBatchForModel(
    const std::vector<std::string>& texts,
    int max_length,
    bool padding,
    bool truncation,
    bool add_special_tokens) const {
    
    return wrapper_->encodeBatchForModel(texts, max_length, padding, truncation, add_special_tokens);
}

std::string TransformersTokenizer::getModelNameOrPath() const {
    return model_name_or_path_;
}

std::string TransformersTokenizer::getTokenizerType() const {
    return wrapper_->getTokenizerType();
}

bool TransformersTokenizer::isFast() const {
    return wrapper_->isFast();
}

Ptr<Tokenizer> createTransformersTokenizer(const std::string& model_name_or_path, bool use_fast) {
    return makePtr<TransformersTokenizer>(model_name_or_path, use_fast);
}

}} // namespace cv::dnn