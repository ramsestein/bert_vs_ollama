"""
Text processing utilities for the Multi-Strategy NER system.

Contains functions for text normalization, tokenization, chunking, and fuzzy matching.
"""

import re
from typing import List, Dict, Tuple
import unicodedata
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
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes and dashes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'–|—', '-', text)
    
    # Optionally remove accents for Spanish matching
    if remove_accents:
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    return text.strip()

def _strip_accents(text: str) -> str:
    """Remove accents/diacritics from text."""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')

def _normalize_with_positions(text: str) -> Tuple[str, List[int]]:
    """Lowercase and strip accents while tracking position mapping back to original.
    
    Returns:
        (normalized_text, pos_map) where pos_map[i] = index in original text 
        of the character that produced normalized char i.
    """
    text_lower = text.lower()
    normalized_chars = []
    pos_map = []
    
    for orig_idx, char in enumerate(text_lower):
        decomposed = unicodedata.normalize('NFD', char)
        for d_char in decomposed:
            if unicodedata.category(d_char) != 'Mn':
                normalized_chars.append(d_char)
                pos_map.append(orig_idx)
    
    return ''.join(normalized_chars), pos_map

def _find_text_in_original(original_text: str, search_term: str, 
                           norm_text: str = None, pos_map: List[int] = None) -> List[Dict[str, int]]:
    """Find all occurrences of search_term in original_text.
    
    Returns list of {"start": int, "end": int}. Tries case-insensitive first,
    then accent-insensitive with pre-computed normalized text/pos_map.
    """
    if not search_term:
        return []
    
    spans = []
    
    # Try direct case-insensitive search first (fast path)
    pattern = re.compile(r'\b' + re.escape(search_term) + r'\b', re.IGNORECASE)
    for match in pattern.finditer(original_text):
        spans.append({"start": match.start(), "end": match.end()})
    
    if spans:
        return spans
    
    # Fallback: accent-insensitive search
    if norm_text is None or pos_map is None:
        norm_text, pos_map = _normalize_with_positions(original_text)
    
    norm_search = _strip_accents(search_term.lower())
    if not norm_search:
        return []
    
    pattern = re.compile(r'\b' + re.escape(norm_search) + r'\b', re.IGNORECASE)
    for match in pattern.finditer(norm_text):
        orig_start = pos_map[match.start()]
        orig_end = pos_map[match.end() - 1] + 1
        spans.append({"start": orig_start, "end": orig_end})
    
    return spans

def find_entity_spans(original_text: str, entity_name: str, 
                      mentions: List[str] = None) -> List[Dict[str, int]]:
    """Find all occurrences of an entity in original_text.
    
    Searches for the entity_name itself AND any raw mentions (the text as
    originally detected by the LLM before fuzzy matching to the candidate name).
    
    Args:
        original_text: The full document text
        entity_name: The canonical entity name (user-defined candidate)
        mentions: List of raw text strings that were matched to this entity
                  (e.g. what the LLM actually found in the text)
    
    Returns:
        List of {"start": int, "end": int} character offsets, deduplicated.
    """
    if not original_text or not entity_name:
        return []
    
    # Pre-compute normalized text once for all searches
    norm_text, pos_map = _normalize_with_positions(original_text)
    
    # Collect all search terms: the canonical name + all raw mentions
    search_terms = {entity_name}
    if mentions:
        search_terms.update(m for m in mentions if m)
    
    # Find spans for each search term
    seen = set()  # (start, end) to deduplicate overlapping finds
    spans = []
    
    for term in search_terms:
        for span in _find_text_in_original(original_text, term, norm_text, pos_map):
            key = (span["start"], span["end"])
            if key not in seen:
                seen.add(key)
                spans.append(span)
    
    # Sort by position
    spans.sort(key=lambda s: s["start"])
    return spans

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
    
    # DEBUG LOGS
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
        print(f"      [CHUNK] [WARN] Text too short ({len(words)} < {strategy['chunk_min']}), using single chunk")
    else:
        print(f"      [CHUNK DEBUG] [OK] Text has enough words ({len(words)} >= {strategy['chunk_min']}), creating multiple chunks...")
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
                    print(f"      [CHUNK] [OK] Created chunk {len(chunks)}: {len(chunk_words)} words (start={start}, end={end})")
                else:
                    print(f"      [CHUNK] [WARN] Chunk too large ({len(chunk_words)} > {strategy['chunk_max']}), skipping")
            else:
                print(f"      [CHUNK] [WARN] Chunk too small ({len(chunk_words)} < {strategy['chunk_min']}), skipping")
            
            # Move start position with overlap, ensuring we always advance
            new_start = end - overlap
            if new_start <= start:  # Safety check: ensure we're advancing
                new_start = start + 1
                print(f"      [CHUNK DEBUG] [WARN] new_start <= start, forcing advance to {new_start}")
            
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
        print(f"      [CHUNK] [WARN] No chunks created, using original text as single chunk")
    
    print(f"      [CHUNK] [OK] FINAL RESULT: Created {len(chunks)} total chunks for {strategy['name']}")
    for i, chunk in enumerate(chunks, 1):
        print(f"      [CHUNK]    Chunk {i}: {len(chunk.split())} words, {len(chunk)} chars")
    
    return chunks
