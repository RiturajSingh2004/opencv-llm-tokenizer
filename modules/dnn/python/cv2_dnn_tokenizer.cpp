/**
 * @file cv2_dnn_tokenizer.cpp
 * @brief Python bindings for OpenCV tokenizer functionality
 * @author Rituraj Singh
 * @date 2025
 *
 * This file implements Python bindings for the OpenCV tokenizer classes and functions,
 * making the tokenizer functionality available through the cv2.dnn namespace in Python.
 * It provides wrapper methods for all tokenizer types and their associated operations.
 */

#include <opencv2/core.hpp>
#include <opencv2/core/utils/python.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn/tokenizer.hpp>
#include <opencv2/dnn/gpu_tokenizer.hpp>

// Structure definitions for Python C API
typedef struct {
    PyObject_HEAD
    Ptr<Tokenizer>* v;
} pyopencv_Tokenizer_t;

#ifdef HAVE_CUDA
typedef struct {
    PyObject_HEAD
    Ptr<GPUTokenizer>* v;
} pyopencv_GPUTokenizer_t;
#endif

// Forward declarations for helper functions used in different parts of the code
static PyObject* pyopencv_cuda_GpuMat_Instance(const cuda::GpuMat& m);

using namespace cv;
using namespace cv::dnn;

// Helper function to convert Python list to C++ vector of strings
static std::vector<std::string> pyListToStringVector(PyObject* py_list) {
    if (!py_list || !PyList_Check(py_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of strings");
        return {};
    }
    
    Py_ssize_t size = PyList_Size(py_list);
    std::vector<std::string> result;
    result.reserve(size);
    
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(py_list, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "List items must be strings");
            return {};
        }
        
        const char* str = PyUnicode_AsUTF8(item);
        if (str) {
            result.push_back(str);
        } else {
            PyErr_SetString(PyExc_TypeError, "Failed to convert string");
            return {};
        }
    }
    
    return result;
}

// Helper function to convert C++ TokenInfo to Python dict
static PyObject* tokenInfoToDict(const TokenInfo& info) {
    PyObject* dict = PyDict_New();
    PyDict_SetItemString(dict, "id", PyLong_FromLong(info.id));
    PyDict_SetItemString(dict, "text", PyUnicode_FromString(info.text.c_str()));
    PyDict_SetItemString(dict, "start", PyLong_FromLong(info.start));
    PyDict_SetItemString(dict, "end", PyLong_FromLong(info.end));
    PyDict_SetItemString(dict, "score", PyFloat_FromDouble(info.score));
    return dict;
}

// Tokenizer class
static PyTypeObject pyopencv_Tokenizer_Type;

// Tokenizer methods

// encode method
static PyObject* pyopencv_cv_dnn_Tokenizer_encode(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_text = NULL;
    
    const char* keywords[] = { "text", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:Tokenizer.encode", (char**)keywords, &pyobj_text))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    std::string text;
    if (PyUnicode_Check(pyobj_text)) {
        text = PyUnicode_AsUTF8(pyobj_text);
    } else {
        PyErr_SetString(PyExc_TypeError, "Text must be a string");
        return NULL;
    }
    
    try {
        std::vector<int> tokens = _self->encode(text);
        
        // Convert vector<int> to Python list
        PyObject* py_tokens = PyList_New(tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
            PyList_SetItem(py_tokens, i, PyLong_FromLong(tokens[i]));
        }
        
        return py_tokens;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// encodeWithInfo method
static PyObject* pyopencv_cv_dnn_Tokenizer_encodeWithInfo(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_text = NULL;
    
    const char* keywords[] = { "text", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:Tokenizer.encodeWithInfo", (char**)keywords, &pyobj_text))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    std::string text;
    if (PyUnicode_Check(pyobj_text)) {
        text = PyUnicode_AsUTF8(pyobj_text);
    } else {
        PyErr_SetString(PyExc_TypeError, "Text must be a string");
        return NULL;
    }
    
    try {
        std::vector<TokenInfo> tokenInfo = _self->encodeWithInfo(text);
        
        // Convert vector<TokenInfo> to Python list of dicts
        PyObject* py_token_info = PyList_New(tokenInfo.size());
        for (size_t i = 0; i < tokenInfo.size(); i++) {
            PyList_SetItem(py_token_info, i, tokenInfoToDict(tokenInfo[i]));
        }
        
        return py_token_info;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// encodeBatch method
static PyObject* pyopencv_cv_dnn_Tokenizer_encodeBatch(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_texts = NULL;
    
    const char* keywords[] = { "texts", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:Tokenizer.encodeBatch", (char**)keywords, &pyobj_texts))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    if (!PyList_Check(pyobj_texts)) {
        PyErr_SetString(PyExc_TypeError, "Texts must be a list of strings");
        return NULL;
    }
    
    std::vector<std::string> texts = pyListToStringVector(pyobj_texts);
    if (texts.empty() && PyList_Size(pyobj_texts) > 0) {
        // Error already set by pyListToStringVector
        return NULL;
    }
    
    try {
        std::vector<std::vector<int>> batch_tokens = _self->encodeBatch(texts);
        
        // Convert vector<vector<int>> to Python list of lists
        PyObject* py_batch_tokens = PyList_New(batch_tokens.size());
        for (size_t i = 0; i < batch_tokens.size(); i++) {
            const std::vector<int>& tokens = batch_tokens[i];
            PyObject* py_tokens = PyList_New(tokens.size());
            
            for (size_t j = 0; j < tokens.size(); j++) {
                PyList_SetItem(py_tokens, j, PyLong_FromLong(tokens[j]));
            }
            
            PyList_SetItem(py_batch_tokens, i, py_tokens);
        }
        
        return py_batch_tokens;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// decode method
static PyObject* pyopencv_cv_dnn_Tokenizer_decode(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_tokens = NULL;
    
    const char* keywords[] = { "tokens", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:Tokenizer.decode", (char**)keywords, &pyobj_tokens))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    if (!PyList_Check(pyobj_tokens)) {
        PyErr_SetString(PyExc_TypeError, "Tokens must be a list of integers");
        return NULL;
    }
    
    Py_ssize_t size = PyList_Size(pyobj_tokens);
    std::vector<int> tokens;
    tokens.reserve(size);
    
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(pyobj_tokens, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Tokens must be a list of integers");
            return NULL;
        }
        
        tokens.push_back(PyLong_AsLong(item));
    }
    
    try {
        std::string decoded = _self->decode(tokens);
        return PyUnicode_FromString(decoded.c_str());
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// getVocabSize method
static PyObject* pyopencv_cv_dnn_Tokenizer_getVocabSize(PyObject* self, PyObject* args, PyObject* kw) {
    const char* keywords[] = { NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, ":Tokenizer.getVocabSize", (char**)keywords))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    try {
        size_t vocabSize = _self->getVocabSize();
        return PyLong_FromSize_t(vocabSize);
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// getTokenText method
static PyObject* pyopencv_cv_dnn_Tokenizer_getTokenText(PyObject* self, PyObject* args, PyObject* kw) {
    int tokenId;
    
    const char* keywords[] = { "tokenId", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "i:Tokenizer.getTokenText", (char**)keywords, &tokenId))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    try {
        std::string text = _self->getTokenText(tokenId);
        return PyUnicode_FromString(text.c_str());
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// getTokenId method
static PyObject* pyopencv_cv_dnn_Tokenizer_getTokenId(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_text = NULL;
    
    const char* keywords[] = { "tokenText", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:Tokenizer.getTokenId", (char**)keywords, &pyobj_text))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    std::string tokenText;
    if (PyUnicode_Check(pyobj_text)) {
        tokenText = PyUnicode_AsUTF8(pyobj_text);
    } else {
        PyErr_SetString(PyExc_TypeError, "Token text must be a string");
        return NULL;
    }
    
    try {
        int tokenId = _self->getTokenId(tokenText);
        return PyLong_FromLong(tokenId);
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// save method
static PyObject* pyopencv_cv_dnn_Tokenizer_save(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_filename = NULL;
    
    const char* keywords[] = { "filename", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:Tokenizer.save", (char**)keywords, &pyobj_filename))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    std::string filename;
    if (PyUnicode_Check(pyobj_filename)) {
        filename = PyUnicode_AsUTF8(pyobj_filename);
    } else {
        PyErr_SetString(PyExc_TypeError, "Filename must be a string");
        return NULL;
    }
    
    try {
        _self->save(filename);
        Py_RETURN_NONE;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// Static methods

// load static method
static PyObject* pyopencv_cv_dnn_Tokenizer_load(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_filename = NULL;
    
    const char* keywords[] = { "filename", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:Tokenizer.load", (char**)keywords, &pyobj_filename))
        return NULL;
    
    std::string filename;
    if (PyUnicode_Check(pyobj_filename)) {
        filename = PyUnicode_AsUTF8(pyobj_filename);
    } else {
        PyErr_SetString(PyExc_TypeError, "Filename must be a string");
        return NULL;
    }
    
    try {
        Ptr<Tokenizer> tokenizer = Tokenizer::load(filename);
        
        // Create a Python wrapper for the tokenizer
        pyopencv_Tokenizer_t* py_tokenizer = PyObject_NEW(pyopencv_Tokenizer_t, &pyopencv_Tokenizer_Type);
        py_tokenizer->v = new Ptr<Tokenizer>(tokenizer);
        
        return (PyObject*)py_tokenizer;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// Factory functions

// createBPETokenizer function
static PyObject* pyopencv_cv_dnn_createBPETokenizer(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_vocab_file = NULL;
    PyObject* pyobj_merges_file = NULL;
    
    const char* keywords[] = { "vocab_file", "merges_file", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:createBPETokenizer", (char**)keywords, 
                                     &pyobj_vocab_file, &pyobj_merges_file))
        return NULL;
    
    std::string vocab_file, merges_file;
    
    if (PyUnicode_Check(pyobj_vocab_file)) {
        vocab_file = PyUnicode_AsUTF8(pyobj_vocab_file);
    } else {
        PyErr_SetString(PyExc_TypeError, "vocab_file must be a string");
        return NULL;
    }
    
    if (PyUnicode_Check(pyobj_merges_file)) {
        merges_file = PyUnicode_AsUTF8(pyobj_merges_file);
    } else {
        PyErr_SetString(PyExc_TypeError, "merges_file must be a string");
        return NULL;
    }
    
    try {
        Ptr<Tokenizer> tokenizer = createBPETokenizer(vocab_file, merges_file);
        
        // Create a Python wrapper for the tokenizer
        pyopencv_Tokenizer_t* py_tokenizer = PyObject_NEW(pyopencv_Tokenizer_t, &pyopencv_Tokenizer_Type);
        py_tokenizer->v = new Ptr<Tokenizer>(tokenizer);
        
        return (PyObject*)py_tokenizer;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// createWordPieceTokenizer function
static PyObject* pyopencv_cv_dnn_createWordPieceTokenizer(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_vocab_file = NULL;
    PyObject* pyobj_unk_token = NULL;
    int max_chars_per_word = 100;
    
    const char* keywords[] = { "vocab_file", "unk_token", "max_chars_per_word", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|Oi:createWordPieceTokenizer", (char**)keywords, 
                                     &pyobj_vocab_file, &pyobj_unk_token, &max_chars_per_word))
        return NULL;
    
    std::string vocab_file, unk_token = "[UNK]";
    
    if (PyUnicode_Check(pyobj_vocab_file)) {
        vocab_file = PyUnicode_AsUTF8(pyobj_vocab_file);
    } else {
        PyErr_SetString(PyExc_TypeError, "vocab_file must be a string");
        return NULL;
    }
    
    if (pyobj_unk_token && PyUnicode_Check(pyobj_unk_token)) {
        unk_token = PyUnicode_AsUTF8(pyobj_unk_token);
    } else if (pyobj_unk_token != NULL) {
        PyErr_SetString(PyExc_TypeError, "unk_token must be a string");
        return NULL;
    }
    
    try {
        Ptr<Tokenizer> tokenizer = createWordPieceTokenizer(vocab_file, unk_token, max_chars_per_word);
        
        // Create a Python wrapper for the tokenizer
        pyopencv_Tokenizer_t* py_tokenizer = PyObject_NEW(pyopencv_Tokenizer_t, &pyopencv_Tokenizer_Type);
        py_tokenizer->v = new Ptr<Tokenizer>(tokenizer);
        
        return (PyObject*)py_tokenizer;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// createUnigramTokenizer function
static PyObject* pyopencv_cv_dnn_createUnigramTokenizer(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_vocab_file = NULL;
    int unk_id = 0;
    PyObject* pyobj_unk_piece = NULL;
    float score_threshold = 0.0f;
    float sample_alpha = 0.1f;
    
    const char* keywords[] = { "vocab_file", "unk_id", "unk_piece", "score_threshold", "sample_alpha", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|iOff:createUnigramTokenizer", (char**)keywords, 
                                     &pyobj_vocab_file, &unk_id, &pyobj_unk_piece, &score_threshold, &sample_alpha))
        return NULL;
    
    std::string vocab_file, unk_piece = "<unk>";
    
    if (PyUnicode_Check(pyobj_vocab_file)) {
        vocab_file = PyUnicode_AsUTF8(pyobj_vocab_file);
    } else {
        PyErr_SetString(PyExc_TypeError, "vocab_file must be a string");
        return NULL;
    }
    
    if (pyobj_unk_piece && PyUnicode_Check(pyobj_unk_piece)) {
        unk_piece = PyUnicode_AsUTF8(pyobj_unk_piece);
    } else if (pyobj_unk_piece != NULL) {
        PyErr_SetString(PyExc_TypeError, "unk_piece must be a string");
        return NULL;
    }
    
    try {
        Ptr<Tokenizer> tokenizer = createUnigramTokenizer(vocab_file, unk_id, unk_piece, score_threshold, sample_alpha);
        
        // Create a Python wrapper for the tokenizer
        pyopencv_Tokenizer_t* py_tokenizer = PyObject_NEW(pyopencv_Tokenizer_t, &pyopencv_Tokenizer_Type);
        py_tokenizer->v = new Ptr<Tokenizer>(tokenizer);
        
        return (PyObject*)py_tokenizer;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// createAdaptiveTokenizer function
static PyObject* pyopencv_cv_dnn_createAdaptiveTokenizer(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_bpe_vocab_file = NULL;
    PyObject* pyobj_bpe_merges_file = NULL;
    PyObject* pyobj_unigram_vocab_file = NULL;
    
    const char* keywords[] = { "bpe_vocab_file", "bpe_merges_file", "unigram_vocab_file", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO:createAdaptiveTokenizer", (char**)keywords, 
                                     &pyobj_bpe_vocab_file, &pyobj_bpe_merges_file, &pyobj_unigram_vocab_file))
        return NULL;
    
    std::string bpe_vocab_file, bpe_merges_file, unigram_vocab_file;
    
    if (PyUnicode_Check(pyobj_bpe_vocab_file)) {
        bpe_vocab_file = PyUnicode_AsUTF8(pyobj_bpe_vocab_file);
    } else {
        PyErr_SetString(PyExc_TypeError, "bpe_vocab_file must be a string");
        return NULL;
    }
    
    if (PyUnicode_Check(pyobj_bpe_merges_file)) {
        bpe_merges_file = PyUnicode_AsUTF8(pyobj_bpe_merges_file);
    } else {
        PyErr_SetString(PyExc_TypeError, "bpe_merges_file must be a string");
        return NULL;
    }
    
    if (PyUnicode_Check(pyobj_unigram_vocab_file)) {
        unigram_vocab_file = PyUnicode_AsUTF8(pyobj_unigram_vocab_file);
    } else {
        PyErr_SetString(PyExc_TypeError, "unigram_vocab_file must be a string");
        return NULL;
    }
    
    try {
        Ptr<Tokenizer> tokenizer = createAdaptiveTokenizer(bpe_vocab_file, bpe_merges_file, unigram_vocab_file);
        
        // Create a Python wrapper for the tokenizer
        pyopencv_Tokenizer_t* py_tokenizer = PyObject_NEW(pyopencv_Tokenizer_t, &pyopencv_Tokenizer_Type);
        py_tokenizer->v = new Ptr<Tokenizer>(tokenizer);
        
        return (PyObject*)py_tokenizer;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// createTikTokenTokenizer function
static PyObject* pyopencv_cv_dnn_createTikTokenTokenizer(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_encoding_name = NULL;
    PyObject* pyobj_bpe_ranks_path = NULL;
    PyObject* pyobj_special_tokens = NULL;
    
    const char* keywords[] = { "encoding_name", "bpe_ranks_path", "special_tokens", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|OO:createTikTokenTokenizer", (char**)keywords, 
                                     &pyobj_encoding_name, &pyobj_bpe_ranks_path, &pyobj_special_tokens))
        return NULL;
    
    std::string encoding_name, bpe_ranks_path = "";
    std::map<std::string, int> special_tokens;
    
    if (PyUnicode_Check(pyobj_encoding_name)) {
        encoding_name = PyUnicode_AsUTF8(pyobj_encoding_name);
    } else {
        PyErr_SetString(PyExc_TypeError, "encoding_name must be a string");
        return NULL;
    }
    
    if (pyobj_bpe_ranks_path && PyUnicode_Check(pyobj_bpe_ranks_path)) {
        bpe_ranks_path = PyUnicode_AsUTF8(pyobj_bpe_ranks_path);
    } else if (pyobj_bpe_ranks_path != NULL && pyobj_bpe_ranks_path != Py_None) {
        PyErr_SetString(PyExc_TypeError, "bpe_ranks_path must be a string");
        return NULL;
    }
    
    if (pyobj_special_tokens && PyDict_Check(pyobj_special_tokens)) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        
        while (PyDict_Next(pyobj_special_tokens, &pos, &key, &value)) {
            if (!PyUnicode_Check(key) || !PyLong_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "special_tokens must be a dict of {string: int}");
                return NULL;
            }
            
            special_tokens[PyUnicode_AsUTF8(key)] = PyLong_AsLong(value);
        }
    } else if (pyobj_special_tokens != NULL && pyobj_special_tokens != Py_None) {
        PyErr_SetString(PyExc_TypeError, "special_tokens must be a dictionary");
        return NULL;
    }
    
    try {
        Ptr<Tokenizer> tokenizer = createTikTokenTokenizer(encoding_name, bpe_ranks_path, special_tokens);
        
        // Create a Python wrapper for the tokenizer
        pyopencv_Tokenizer_t* py_tokenizer = PyObject_NEW(pyopencv_Tokenizer_t, &pyopencv_Tokenizer_Type);
        py_tokenizer->v = new Ptr<Tokenizer>(tokenizer);
        
        return (PyObject*)py_tokenizer;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// createSentencePieceTokenizer function
static PyObject* pyopencv_cv_dnn_createSentencePieceTokenizer(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_model_path = NULL;
    
    const char* keywords[] = { "model_path", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:createSentencePieceTokenizer", (char**)keywords, &pyobj_model_path))
        return NULL;
    
    std::string model_path;
    
    if (PyUnicode_Check(pyobj_model_path)) {
        model_path = PyUnicode_AsUTF8(pyobj_model_path);
    } else {
        PyErr_SetString(PyExc_TypeError, "model_path must be a string");
        return NULL;
    }
    
    try {
        Ptr<Tokenizer> tokenizer = createSentencePieceTokenizer(model_path);
        
        // Create a Python wrapper for the tokenizer
        pyopencv_Tokenizer_t* py_tokenizer = PyObject_NEW(pyopencv_Tokenizer_t, &pyopencv_Tokenizer_Type);
        py_tokenizer->v = new Ptr<Tokenizer>(tokenizer);
        
        return (PyObject*)py_tokenizer;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// createTransformersTokenizer function
static PyObject* pyopencv_cv_dnn_createTransformersTokenizer(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_model_name_or_path = NULL;
    PyObject* pyobj_use_fast = NULL;
    
    const char* keywords[] = { "model_name_or_path", "use_fast", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O:createTransformersTokenizer", (char**)keywords, 
                                    &pyobj_model_name_or_path, &pyobj_use_fast))
        return NULL;
    
    std::string model_name_or_path;
    bool use_fast = true;
    
    if (PyUnicode_Check(pyobj_model_name_or_path)) {
        model_name_or_path = PyUnicode_AsUTF8(pyobj_model_name_or_path);
    } else {
        PyErr_SetString(PyExc_TypeError, "model_name_or_path must be a string");
        return NULL;
    }
    
    if (pyobj_use_fast && PyBool_Check(pyobj_use_fast)) {
        use_fast = (pyobj_use_fast == Py_True);
    } else if (pyobj_use_fast != NULL && pyobj_use_fast != Py_None) {
        PyErr_SetString(PyExc_TypeError, "use_fast must be a boolean");
        return NULL;
    }
    
    try {
        Ptr<Tokenizer> tokenizer = createTransformersTokenizer(model_name_or_path, use_fast);
        
        // Create a Python wrapper for the tokenizer
        pyopencv_Tokenizer_t* py_tokenizer = PyObject_NEW(pyopencv_Tokenizer_t, &pyopencv_Tokenizer_Type);
        py_tokenizer->v = new Ptr<Tokenizer>(tokenizer);
        
        return (PyObject*)py_tokenizer;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// Additional methods for TransformersTokenizer
// These would be added to the Tokenizer class since TransformersTokenizer inherits from it

// encodeForModel method
static PyObject* pyopencv_cv_dnn_TransformersTokenizer_encodeForModel(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_text = NULL;
    int max_length = 512;
    PyObject* pyobj_padding = NULL;
    PyObject* pyobj_truncation = NULL;
    PyObject* pyobj_add_special_tokens = NULL;
    
    const char* keywords[] = { "text", "max_length", "padding", "truncation", "add_special_tokens", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|iOOO:TransformersTokenizer.encodeForModel", (char**)keywords,
                                    &pyobj_text, &max_length, &pyobj_padding, &pyobj_truncation, &pyobj_add_special_tokens))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    // Check if this is a TransformersTokenizer
    TransformersTokenizer* transformersTokenizer = dynamic_cast<TransformersTokenizer*>(_self);
    if (!transformersTokenizer) {
        PyErr_SetString(PyExc_TypeError, "This method is only available for TransformersTokenizer");
        return NULL;
    }
    
    std::string text;
    if (PyUnicode_Check(pyobj_text)) {
        text = PyUnicode_AsUTF8(pyobj_text);
    } else {
        PyErr_SetString(PyExc_TypeError, "Text must be a string");
        return NULL;
    }
    
    bool padding = false;
    if (pyobj_padding && PyBool_Check(pyobj_padding)) {
        padding = (pyobj_padding == Py_True);
    } else if (pyobj_padding != NULL && pyobj_padding != Py_None) {
        PyErr_SetString(PyExc_TypeError, "padding must be a boolean");
        return NULL;
    }
    
    bool truncation = true;
    if (pyobj_truncation && PyBool_Check(pyobj_truncation)) {
        truncation = (pyobj_truncation == Py_True);
    } else if (pyobj_truncation != NULL && pyobj_truncation != Py_None) {
        PyErr_SetString(PyExc_TypeError, "truncation must be a boolean");
        return NULL;
    }
    
    bool add_special_tokens = true;
    if (pyobj_add_special_tokens && PyBool_Check(pyobj_add_special_tokens)) {
        add_special_tokens = (pyobj_add_special_tokens == Py_True);
    } else if (pyobj_add_special_tokens != NULL && pyobj_add_special_tokens != Py_None) {
        PyErr_SetString(PyExc_TypeError, "add_special_tokens must be a boolean");
        return NULL;
    }
    
    try {
        std::vector<int> tokens = transformersTokenizer->encodeForModel(
            text, max_length, padding, truncation, add_special_tokens);
        
        // Convert to Python list
        PyObject* py_tokens = PyList_New(tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
            PyList_SetItem(py_tokens, i, PyLong_FromLong(tokens[i]));
        }
        
        return py_tokens;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// encodeBatchForModel method
static PyObject* pyopencv_cv_dnn_TransformersTokenizer_encodeBatchForModel(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_texts = NULL;
    int max_length = 512;
    PyObject* pyobj_padding = NULL;
    PyObject* pyobj_truncation = NULL;
    PyObject* pyobj_add_special_tokens = NULL;
    
    const char* keywords[] = { "texts", "max_length", "padding", "truncation", "add_special_tokens", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|iOOO:TransformersTokenizer.encodeBatchForModel", (char**)keywords,
                                    &pyobj_texts, &max_length, &pyobj_padding, &pyobj_truncation, &pyobj_add_special_tokens))
        return NULL;
    
    Ptr<Tokenizer>* ptr = ((pyopencv_Tokenizer_t*)self)->v;
    Tokenizer* _self = ptr->get();
    
    // Check if this is a TransformersTokenizer
    TransformersTokenizer* transformersTokenizer = dynamic_cast<TransformersTokenizer*>(_self);
    if (!transformersTokenizer) {
        PyErr_SetString(PyExc_TypeError, "This method is only available for TransformersTokenizer");
        return NULL;
    }
    
    if (!PyList_Check(pyobj_texts)) {
        PyErr_SetString(PyExc_TypeError, "Texts must be a list of strings");
        return NULL;
    }
    
    std::vector<std::string> texts = pyListToStringVector(pyobj_texts);
    if (texts.empty() && PyList_Size(pyobj_texts) > 0) {
        // Error already set by pyListToStringVector
        return NULL;
    }
    
    bool padding = false;
    if (pyobj_padding && PyBool_Check(pyobj_padding)) {
        padding = (pyobj_padding == Py_True);
    } else if (pyobj_padding != NULL && pyobj_padding != Py_None) {
        PyErr_SetString(PyExc_TypeError, "padding must be a boolean");
        return NULL;
    }
    
    bool truncation = true;
    if (pyobj_truncation && PyBool_Check(pyobj_truncation)) {
        truncation = (pyobj_truncation == Py_True);
    } else if (pyobj_truncation != NULL && pyobj_truncation != Py_None) {
        PyErr_SetString(PyExc_TypeError, "truncation must be a boolean");
        return NULL;
    }
    
    bool add_special_tokens = true;
    if (pyobj_add_special_tokens && PyBool_Check(pyobj_add_special_tokens)) {
        add_special_tokens = (pyobj_add_special_tokens == Py_True);
    } else if (pyobj_add_special_tokens != NULL && pyobj_add_special_tokens != Py_None) {
        PyErr_SetString(PyExc_TypeError, "add_special_tokens must be a boolean");
        return NULL;
    }
    
    try {
        std::vector<std::vector<int>> batch_tokens = transformersTokenizer->encodeBatchForModel(
            texts, max_length, padding, truncation, add_special_tokens);
        
        // Convert vector<vector<int>> to Python list of lists
        PyObject* py_batch_tokens = PyList_New(batch_tokens.size());
        for (size_t i = 0; i < batch_tokens.size(); i++) {
            const std::vector<int>& tokens = batch_tokens[i];
            PyObject* py_tokens = PyList_New(tokens.size());
            
            for (size_t j = 0; j < tokens.size(); j++) {
                PyList_SetItem(py_tokens, j, PyLong_FromLong(tokens[j]));
            }
            
            PyList_SetItem(py_batch_tokens, i, py_tokens);
        }
        
        return py_batch_tokens;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// TokenizerPreprocessConfig struct for Python
static PyObject* pyopencv_cv_dnn_createTokenizerPreprocessConfig(PyObject* self, PyObject* args, PyObject* kw) {
    int flags = TOKENIZER_LOWERCASE | TOKENIZER_NORMALIZE_SPACE;
    PyObject* pyobj_special_tokens = NULL;
    PyObject* pyobj_replacements = NULL;
    
    const char* keywords[] = { "flags", "special_tokens", "replacements", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|iOO:createTokenizerPreprocessConfig", (char**)keywords,
                                     &flags, &pyobj_special_tokens, &pyobj_replacements))
        return NULL;
    
    TokenizerPreprocessConfig config;
    config.flags = flags;
    
    // Handle special tokens
    if (pyobj_special_tokens && PyList_Check(pyobj_special_tokens)) {
        Py_ssize_t size = PyList_Size(pyobj_special_tokens);
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* item = PyList_GetItem(pyobj_special_tokens, i);
            if (!PyUnicode_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "special_tokens must be a list of strings");
                return NULL;
            }
            config.specialTokens.push_back(PyUnicode_AsUTF8(item));
        }
    } else if (pyobj_special_tokens != NULL && pyobj_special_tokens != Py_None) {
        PyErr_SetString(PyExc_TypeError, "special_tokens must be a list");
        return NULL;
    }
    
    // Handle replacements
    if (pyobj_replacements && PyDict_Check(pyobj_replacements)) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        
        while (PyDict_Next(pyobj_replacements, &pos, &key, &value)) {
            if (!PyUnicode_Check(key) || !PyUnicode_Check(value)) {
                PyErr_SetString(PyExc_TypeError, "replacements must be a dict of {string: string}");
                return NULL;
            }
            
            config.replacements[PyUnicode_AsUTF8(key)] = PyUnicode_AsUTF8(value);
        }
    } else if (pyobj_replacements != NULL && pyobj_replacements != Py_None) {
        PyErr_SetString(PyExc_TypeError, "replacements must be a dictionary");
        return NULL;
    }
    
    // Here we would need to wrap the TokenizerPreprocessConfig in a Python object
    // For simplicity, we'll just return the flags for now
    return PyLong_FromLong(config.flags);
}

// TokenizationAnalysis class for benchmarking
// This would need to be implemented based on the TokenizationAnalysis class
// to provide Python bindings for the benchmarking functionality

// GPU Tokenizer support
#ifdef HAVE_CUDA

// GPUTokenizer class
static PyTypeObject pyopencv_GPUTokenizer_Type;

// GPUTokenizer methods

// encodeBatch method
static PyObject* pyopencv_cv_dnn_GPUTokenizer_encodeBatch(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_texts = NULL;
    PyObject* pyobj_stream = NULL;
    
    const char* keywords[] = { "texts", "stream", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O:GPUTokenizer.encodeBatch", (char**)keywords, 
                                     &pyobj_texts, &pyobj_stream))
        return NULL;
    
    Ptr<GPUTokenizer>* ptr = ((pyopencv_GPUTokenizer_t*)self)->v;
    GPUTokenizer* _self = ptr->get();
    
    if (!PyList_Check(pyobj_texts)) {
        PyErr_SetString(PyExc_TypeError, "Texts must be a list of strings");
        return NULL;
    }
    
    std::vector<std::string> texts = pyListToStringVector(pyobj_texts);
    if (texts.empty() && PyList_Size(pyobj_texts) > 0) {
        // Error already set by pyListToStringVector
        return NULL;
    }
    
    cuda::Stream stream = cuda::Stream::Null();
    
    if (pyobj_stream && pyobj_stream != Py_None) {
        // Get CUDA stream from Python object
        // This would need proper Stream handling code in OpenCV's Python bindings
        PyErr_SetString(PyExc_NotImplementedError, "Custom CUDA stream handling is not yet implemented");
        return NULL;
    }
    
    try {
        std::vector<std::vector<int>> batch_tokens = _self->encodeBatch(texts, stream);
        
        // Convert vector<vector<int>> to Python list of lists
        PyObject* py_batch_tokens = PyList_New(batch_tokens.size());
        for (size_t i = 0; i < batch_tokens.size(); i++) {
            const std::vector<int>& tokens = batch_tokens[i];
            PyObject* py_tokens = PyList_New(tokens.size());
            
            for (size_t j = 0; j < tokens.size(); j++) {
                PyList_SetItem(py_tokens, j, PyLong_FromLong(tokens[j]));
            }
            
            PyList_SetItem(py_batch_tokens, i, py_tokens);
        }
        
        return py_batch_tokens;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// encodeBatchToGpuMat method
// Note: This requires proper CUDA support in Python bindings
static PyObject* pyopencv_cv_dnn_GPUTokenizer_encodeBatchToGpuMat(PyObject* self, PyObject* args, PyObject* kw) {
    PyObject* pyobj_texts = NULL;
    int max_length = 0;
    PyObject* pyobj_padding = NULL;
    PyObject* pyobj_stream = NULL;
    
    const char* keywords[] = { "texts", "max_length", "padding", "stream", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|iOO:GPUTokenizer.encodeBatchToGpuMat", (char**)keywords,
                                     &pyobj_texts, &max_length, &pyobj_padding, &pyobj_stream))
        return NULL;
    
    Ptr<GPUTokenizer>* ptr = ((pyopencv_GPUTokenizer_t*)self)->v;
    GPUTokenizer* _self = ptr->get();
    
    if (!PyList_Check(pyobj_texts)) {
        PyErr_SetString(PyExc_TypeError, "Texts must be a list of strings");
        return NULL;
    }
    
    std::vector<std::string> texts = pyListToStringVector(pyobj_texts);
    if (texts.empty() && PyList_Size(pyobj_texts) > 0) {
        // Error already set by pyListToStringVector
        return NULL;
    }
    
    bool padding = false;
    if (pyobj_padding && PyBool_Check(pyobj_padding)) {
        padding = (pyobj_padding == Py_True);
    } else if (pyobj_padding != NULL && pyobj_padding != Py_None) {
        PyErr_SetString(PyExc_TypeError, "padding must be a boolean");
        return NULL;
    }
    
    cuda::Stream stream = cuda::Stream::Null();
    
    if (pyobj_stream && pyobj_stream != Py_None) {
        // Get CUDA stream from Python object
        // This would need proper Stream handling code in OpenCV's Python bindings
        PyErr_SetString(PyExc_NotImplementedError, "Custom CUDA stream handling is not yet implemented");
        return NULL;
    }
    
    try {
        cuda::GpuMat gpuMat = _self->encodeBatchToGpuMat(texts, max_length, padding, stream);
        
        // Wrap the GpuMat in a Python object
        // This requires OpenCV's Python bindings for cuda::GpuMat
        PyObject* py_gpu_mat = pyopencv_cuda_GpuMat_Instance(gpuMat);
        return py_gpu_mat;
    } catch (const cv::Exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}