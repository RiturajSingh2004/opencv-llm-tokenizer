#!/usr/bin/env python

'''
OpenCV DNN Tokenizer Visualization Tool

@file tokenizer_visualization.py
@brief Interactive visualization tool for exploring tokenization
@author Rituraj Singh
@date 2025

This script provides a visual interface for exploring tokenization in real-time.
It helps users understand how different tokenizers process text by displaying
colorized tokens and their properties in an interactive UI.
'''

import sys
import argparse
import random
import cv2 as cv
import numpy as np

def colorize_token(token_id, vocab_size):
    """Generate a color for a token based on its ID"""
    if vocab_size <= 1:
        h = 0
    else:
        h = (token_id * 137) % 360  # Use prime number to get good distribution
    
    s = 0.7 + 0.3 * (token_id % 3) / 3.0  # Vary saturation
    v = 0.8 + 0.2 * (token_id % 2) / 2.0  # Vary value
    
    # Convert HSV to RGB
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    r, g, b = int(255 * (r + m)), int(255 * (g + m)), int(255 * (b + m))
    return (b, g, r)  # OpenCV uses BGR

def draw_text_with_tokens(text, tokens, token_texts, canvas_size=(1200, 800)):
    """Draw text with colorized tokens"""
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    padding = 10
    line_height = 35
    
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    
    # Draw original text
    cv.putText(canvas, "Original Text:", (padding, padding + 30), 
               font, font_scale, (0, 0, 0), thickness)
    
    text_y = padding + 80
    text_x = padding
    wrapped_lines = []
    current_line = ""
    
    # Simple text wrapping
    words = text.split()
    for word in words:
        if len(current_line + word) * 15 < canvas_size[0] - 2 * padding:
            current_line += word + " "
        else:
            wrapped_lines.append(current_line)
            current_line = word + " "
    
    if current_line:
        wrapped_lines.append(current_line)
    
    for line in wrapped_lines:
        cv.putText(canvas, line, (text_x, text_y), 
                   font, font_scale, (0, 0, 0), thickness)
        text_y += line_height
    
    # Draw token information
    separator_y = text_y + 30
    cv.line(canvas, (padding, separator_y), (canvas_size[0] - padding, separator_y), 
            (200, 200, 200), 2)
    
    cv.putText(canvas, f"Tokenized ({len(tokens)} tokens):", 
               (padding, separator_y + 40), 
               font, font_scale, (0, 0, 0), thickness)
    
    token_y = separator_y + 80
    token_x = padding
    
    vocab_size = max(tokens) + 1
    
    # Display tokens with colors
    for i, (token, token_text) in enumerate(zip(tokens, token_texts)):
        color = colorize_token(token, vocab_size)
        
        # Check if we need to wrap to next line
        text_width = cv.getTextSize(token_text, font, font_scale, thickness)[0][0] + 10
        if token_x + text_width > canvas_size[0] - padding:
            token_x = padding
            token_y += line_height
        
        # Create token rectangle
        text_size = cv.getTextSize(token_text, font, font_scale, thickness)[0]
        cv.rectangle(canvas, 
                     (token_x, token_y - text_size[1] - 5),
                     (token_x + text_size[0] + 10, token_y + 5),
                     color, -1)
        
        # Add token text
        cv.putText(canvas, token_text, (token_x + 5, token_y), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Add token ID as small text below
        id_text = f"[{token}]"
        cv.putText(canvas, id_text, 
                   (token_x, token_y + 20), 
                   font, 0.4, (0, 0, 0), 1)
        
        token_x += text_size[0] + 20
    
    # Add a legend
    legend_y = canvas_size[1] - 70
    cv.putText(canvas, "Instructions:", (padding, legend_y), 
               font, font_scale, (0, 0, 100), thickness)
    
    cv.putText(canvas, "Type text to tokenize | ESC to exit | C to clear", 
               (padding, legend_y + 30), 
               font, 0.6, (100, 100, 100), 1)
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description='OpenCV DNN Tokenizer Visualization Tool')
    parser.add_argument('tokenizer_type', choices=['bpe', 'wordpiece'], 
                        help='Tokenizer type (bpe or wordpiece)')
    parser.add_argument('vocab', type=str, help='Path to vocabulary file')
    parser.add_argument('merges', type=str, nargs='?', 
                        help='Path to merges file (only for BPE tokenizer)')
    parser.add_argument('--sample', action='store_true', 
                        help='Start with a sample text')
    args = parser.parse_args()
    
    # Create tokenizer
    if args.tokenizer_type == 'bpe':
        if not args.merges:
            print("Error: Merges file must be provided for BPE tokenizer")
            return
        tokenizer = cv.dnn.createBPETokenizer(args.vocab, args.merges)
    else:
        tokenizer = cv.dnn.createWordPieceTokenizer(args.vocab)
    
    print(f"Created {args.tokenizer_type} tokenizer with vocabulary size: {tokenizer.getVocabSize()}")
    
    # Initial text
    sample_texts = [
        "Hello world! This is a tokenization visualization tool.",
        "OpenCV now supports LLM tokenizers for processing text.",
        "Tokenizers break text into smaller units called tokens.",
        "This tool helps you visualize how text is tokenized in real-time.",
        "Try typing different text to see how it's processed!"
    ]
    
    current_text = random.choice(sample_texts) if args.sample else ""
    
    # Create window
    window_name = "OpenCV Tokenizer Visualization"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1200, 800)
    
    while True:
        # Tokenize current text
        tokens = tokenizer.encode(current_text) if current_text else []
        
        # Get text representation of each token
        token_texts = []
        for token in tokens:
            try:
                token_text = tokenizer.getTokenText(token)
                if not token_text:  # Fallback if getTokenText is not implemented
                    if token_text == "":
                        token_text = "[EMPTY]"
                    else:
                        token_text = f"[{token}]"
            except:
                # If getTokenText fails, use the decoded representation
                if len(tokens) == 1:
                    token_text = tokenizer.decode([token])
                else:
                    # Try to isolate this token's text by decoding it alone
                    token_text = tokenizer.decode([token])
                    if not token_text:
                        token_text = f"[{token}]"
        
            token_texts.append(token_text)
        
        # Draw the visualization
        canvas = draw_text_with_tokens(current_text, tokens, token_texts)
        cv.imshow(window_name, canvas)
        
        # Handle keyboard input
        key = cv.waitKey(100) & 0xFF
        
        if key == 27:  # ESC key
            break
        elif key == 8:  # Backspace
            current_text = current_text[:-1]
        elif key == 99:  # 'c' key to clear
            current_text = ""
        elif key >= 32 and key < 127:  # Printable ASCII characters
            current_text += chr(key)
    
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()