"""
Text processing utilities for the Multi-Strategy NER system.

Contains functions for text normalization, tokenization, chunking, and fuzzy matching.
"""

import re
from typing import List
from ..config.settings import MAX_CHUNK_ITERATIONS, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE

def _fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Simple fuzzy matching using character overlap."""
    if not text1 or not text2:
        return False
    
    # Remove common words and punctuation
    common_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    text1_clean = ' '.join([w for w in text1.split() if w.lower() not in common_words])
    text2_clean = ' '.join([w for w in text2.split() if w.lower() not in common_words])
    
    if not text1_clean or not text2_clean:
        return False
    
    # Calculate character overlap
    set1 = set(text1_clean.lower())
    set2 = set(text2_clean.lower())
    
    if not set1 or not set2:
        return False
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    return similarity >= threshold

def normalize_surface(text: str) -> str:
    """Normalize text for consistent processing."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes and dashes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'–|—', '-', text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    """Simple tokenization for chunking."""
    return text.split()

def sentence_chunks(text: str, target_tokens: int, overlap_tokens: int, 
                   min_tokens: int = MIN_CHUNK_SIZE, max_tokens: int = MAX_CHUNK_SIZE) -> List[str]:
    """Create overlapping chunks based on token count."""
    tokens = tokenize(text)
    chunks = []
    
    if len(tokens) <= target_tokens:
        return [text]
    
    # Safety check: ensure overlap is less than target_tokens to prevent infinite loops
    if overlap_tokens >= target_tokens:
        overlap_tokens = max(1, target_tokens // 2)
    
    start = 0
    iteration_count = 0
    
    while start < len(tokens) and iteration_count < MAX_CHUNK_ITERATIONS:
        iteration_count += 1
        
        end = min(start + target_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        
        # Ensure minimum chunk size
        if len(chunk_tokens) >= min_tokens:
            chunk_text = " ".join(chunk_tokens)
            if len(chunk_text) <= max_tokens:
                chunks.append(chunk_text)
        
        # Move start position with overlap, ensuring we always advance
        new_start = end - overlap_tokens
        if new_start <= start:  # Safety check: ensure we're advancing
            new_start = start + 1
        
        start = new_start
        
        # Additional safety check
        if start >= len(tokens):
            break
    
    if iteration_count >= MAX_CHUNK_ITERATIONS:
        print(f"      [WARNING] Reached max iterations in sentence_chunks, forcing completion")
        # Force create at least one chunk
        if not chunks:
            chunks = [text]
    
    return chunks if chunks else [text]

def create_chunks_from_text(text: str, strategy: dict) -> List[str]:
    """Create chunks from text based on strategy configuration."""
    print(f"      [CHUNK] Creating chunks for {strategy['name']}")
    
    # Simple chunking by words
    words = text.split()
    chunks = []
    
    target_size = strategy["chunk_target"]
    overlap = strategy["chunk_overlap"]
    
    # Safety check: ensure overlap is less than target_size to prevent infinite loops
    if overlap >= target_size:
        print(f"      [WARNING] Overlap ({overlap}) >= target_size ({target_size}), reducing overlap to {target_size//2}")
        overlap = max(1, target_size // 2)
    
    # Safety check: ensure we have minimum words to process
    if len(words) < strategy["chunk_min"]:
        chunks = [text]
        print(f"      [CHUNK] Text too short, using single chunk")
    else:
        start = 0
        iteration_count = 0
        
        while start < len(words) and iteration_count < MAX_CHUNK_ITERATIONS:
            iteration_count += 1
            
            end = min(start + target_size, len(words))
            chunk_words = words[start:end]
            
            # Ensure minimum chunk size
            if len(chunk_words) >= strategy["chunk_min"]:
                chunk_text = " ".join(chunk_words)
                if len(chunk_text) <= strategy["chunk_max"]:
                    chunks.append(chunk_text)
                    print(f"      [CHUNK] Created chunk {len(chunks)}: {len(chunk_words)} words")
            
            # Move start position with overlap, ensuring we always advance
            new_start = end - overlap
            if new_start <= start:  # Safety check: ensure we're advancing
                new_start = start + 1
            
            start = new_start
            
            # Additional safety check
            if start >= len(words):
                break
        
        if iteration_count >= MAX_CHUNK_ITERATIONS:
            print(f"      [WARNING] Reached max iterations, forcing completion")
            # Force create at least one chunk
            if not chunks:
                chunks = [text]
    
    if not chunks:
        chunks = [text]
        print(f"      [CHUNK] No chunks created, using original text")
    
    print(f"      [CHUNK] Created {len(chunks)} chunks for {strategy['name']}")
    return chunks
