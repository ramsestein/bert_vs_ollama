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
    
    # Remove common words and punctuation (English + Spanish)
    common_words = {
        # English
        'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
        # Spanish
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'de', 'del', 
        'en', 'a', 'con', 'por', 'para', 'al'
    }
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

def normalize_surface(text: str, remove_accents: bool = False) -> str:
    """Normalize text for consistent processing.
    
    Args:
        text: Text to normalize
        remove_accents: If True, remove accents for fuzzy matching (useful for Spanish)
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes and dashes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'–|—', '-', text)
    
    # Optionally remove accents for Spanish matching
    if remove_accents:
        # Spanish accent normalization
        accent_map = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U',
            'ñ': 'n', 'Ñ': 'N'
        }
        for accented, plain in accent_map.items():
            text = text.replace(accented, plain)
    
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
        return [" ".join(tokens)]
    
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
            if len(chunk_tokens) <= max_tokens:
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
            chunks = [" ".join(tokens)]
    
    return chunks if chunks else [" ".join(tokens)]

def create_chunks_from_text(text: str, strategy: dict) -> List[str]:
    """Create chunks from text based on strategy configuration."""
    print(f"      [CHUNK] Creating chunks for {strategy['name']}")
    
    # Simple chunking by words
    words = text.split()
    chunks = []
    
    target_size = strategy["chunk_target"]
    overlap = strategy["chunk_overlap"]
    
    # 🔍 DEBUG LOGS
    print(f"      [CHUNK DEBUG] Text length: {len(text)} characters")
    print(f"      [CHUNK DEBUG] Total words: {len(words)}")
    print(f"      [CHUNK DEBUG] Target size: {target_size} words")
    print(f"      [CHUNK DEBUG] Overlap: {overlap} words")
    print(f"      [CHUNK DEBUG] Min chunk: {strategy['chunk_min']} words")
    print(f"      [CHUNK DEBUG] Max chunk: {strategy['chunk_max']} words")
    
    # Safety check: ensure overlap is less than target_size to prevent infinite loops
    if overlap >= target_size:
        print(f"      [WARNING] Overlap ({overlap}) >= target_size ({target_size}), reducing overlap to {target_size//2}")
        overlap = max(1, target_size // 2)
    
    # Safety check: ensure we have minimum words to process
    if len(words) < strategy["chunk_min"]:
        chunks = [" ".join(words)]
        print(f"      [CHUNK] ⚠️  Text too short ({len(words)} < {strategy['chunk_min']}), using single chunk")
    else:
        print(f"      [CHUNK DEBUG] ✅ Text has enough words ({len(words)} >= {strategy['chunk_min']}), creating multiple chunks...")
        start = 0
        iteration_count = 0
        
        while start < len(words) and iteration_count < MAX_CHUNK_ITERATIONS:
            iteration_count += 1
            
            end = min(start + target_size, len(words))
            chunk_words = words[start:end]
            
            # Ensure minimum chunk size
            if len(chunk_words) >= strategy["chunk_min"]:
                chunk_text = " ".join(chunk_words)
                if len(chunk_words) <= strategy["chunk_max"]:
                    chunks.append(chunk_text)
                    print(f"      [CHUNK] ✅ Created chunk {len(chunks)}: {len(chunk_words)} words (start={start}, end={end})")
                else:
                    print(f"      [CHUNK] ⚠️  Chunk too large ({len(chunk_words)} > {strategy['chunk_max']}), skipping")
            else:
                print(f"      [CHUNK] ⚠️  Chunk too small ({len(chunk_words)} < {strategy['chunk_min']}), skipping")
            
            # Move start position with overlap, ensuring we always advance
            new_start = end - overlap
            if new_start <= start:  # Safety check: ensure we're advancing
                new_start = start + 1
                print(f"      [CHUNK DEBUG] ⚠️  new_start <= start, forcing advance to {new_start}")
            
            print(f"      [CHUNK DEBUG] Moving to next chunk: old_start={start} -> new_start={new_start} (end={end}, overlap={overlap})")
            
            start = new_start
            
            # Additional safety check
            if start >= len(words):
                break
        
        if iteration_count >= MAX_CHUNK_ITERATIONS:
            print(f"      [WARNING] Reached max iterations, forcing completion")
            # Force create at least one chunk
            if not chunks:
                chunks = [" ".join(words)]
    
    if not chunks:
        chunks = [" ".join(words)]
        print(f"      [CHUNK] ⚠️  No chunks created, using original text as single chunk")
    
    print(f"      [CHUNK] ✅ FINAL RESULT: Created {len(chunks)} total chunks for {strategy['name']}")
    for i, chunk in enumerate(chunks, 1):
        print(f"      [CHUNK]    Chunk {i}: {len(chunk.split())} words, {len(chunk)} chars")
    
    return chunks
