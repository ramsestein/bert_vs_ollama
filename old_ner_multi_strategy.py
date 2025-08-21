#!/usr/bin/env python3
"""
Multi-Strategy NER System - Memory Efficient Version
Processes everything through files to minimize RAM usage

"""

import json
import re
import time
import threading
import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Any
import http.client
import hashlib
from datetime import datetime, timedelta

# ----------------------------
# Multi-Strategy Configuration
# ----------------------------

# Strategy 1: llama3.2:3b - Chunks Grandes (Máxima Sensibilidad)
STRATEGY_1 = {
    "name": "llama32_max_sensitivity",
    "model": "llama3.2:3b",
    "chunk_target": 100,
    "chunk_overlap": 40,
    "chunk_min": 50,
    "chunk_max": 150,
    "temperature": 0.1,
    "weight": 1.0
}

# Strategy 2: llama3.2:3b - Chunks Medianos (Balance)
STRATEGY_2 = {
    "name": "llama32_balanced",
    "model": "llama3.2:3b",
    "chunk_target": 60,
    "chunk_overlap": 30,
    "chunk_min": 30,
    "chunk_max": 90,
    "temperature": 0.3,
    "weight": 1.0
}

# Strategy 3: llama3.2:3b - Chunks Pequeños (Máxima Precisión)
STRATEGY_3 = {
    "name": "llama32_high_precision",
    "model": "llama3.2:3b",
    "chunk_target": 30,
    "chunk_overlap": 15,
    "chunk_min": 15,
    "chunk_max": 45,
    "temperature": 0.0,
    "weight": 1.0
}

# Strategy 4: qwen2.5:3b - Chunks Pequeños (Diversidad)
STRATEGY_4 = {
    "name": "qwen25_diversity",
    "model": "qwen2.5:3b",
    "chunk_target": 20,
    "chunk_overlap": 10,
    "chunk_min": 10,
    "chunk_max": 30,
    "temperature": 0.5,
    "weight": 0.5
}

# All strategies (4 for maximum recovery)
ALL_STRATEGIES = [STRATEGY_1, STRATEGY_2, STRATEGY_3, STRATEGY_4]

# Confidence thresholds optimized for 4 strategies
CONFIDENCE_THRESHOLDS = {
    "high": 0.9,      # Entity detected by 3+ strategies
    "medium": 0.7,    # Entity detected by 2+ strategies
    "low": 0.5,       # Entity detected by 1+ strategies
    "min_accept": 0.5 # Minimum threshold for acceptance
}

# File-based processing settings
TEMP_DIR = "temp_processing"
CHUNK_FILE_PREFIX = "chunks_"
STRATEGY_FILE_PREFIX = "strategy_"

# ----------------------------
# LLM Client with Caching
# ----------------------------

class LLMCache:
    def __init__(self, max_size=1000, ttl_hours=24):
        self.cache = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.lock = threading.Lock()
    
    def _generate_key(self, model: str, system_prompt: str, user_prompt: str) -> str:
        content = f"{model}:{system_prompt}:{user_prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> str:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if datetime.now() < entry['expiry']:
                    return entry['response']
                else:
                    del self.cache[key]
        return None
    
    def put(self, key: str, response: str):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entries
                oldest_keys = sorted(self.cache.keys(), 
                                   key=lambda k: self.cache[k]['expiry'])[:len(self.cache)//4]
                for old_key in oldest_keys:
                    del self.cache[old_key]
            
            self.cache[key] = {
                'response': response,
                'expiry': datetime.now() + timedelta(hours=self.ttl_hours)
            }

# Global cache instance
_llm_cache = LLMCache()

class OllamaClient:
    def __init__(self, host="localhost", port=11434, timeout=30):  # Reduced timeout to 30s
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn = None
        self.lock = threading.Lock()
    
    def _get_connection(self):
        if self.conn is None:
            print(f"      [DEBUG] Creating new connection to {self.host}:{self.port}")
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self.conn
    
    def _request(self, path: str, body: dict) -> str:
        conn = self._get_connection()
        payload = json.dumps(body)
        
        print(f"      [DEBUG] Sending request to {path}")
        print(f"      [DEBUG] Model: {body.get('model', 'unknown')}")
        print(f"      [DEBUG] Payload size: {len(payload)} chars")
        
        with self.lock:
            try:
                print(f"      [DEBUG] Making HTTP request...")
                conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
                
                print(f"      [DEBUG] Waiting for response...")
                resp = conn.getresponse()
                
                print(f"      [DEBUG] Response status: {resp.status}")
                print(f"      [DEBUG] Reading response...")
                raw = resp.read().decode("utf-8", errors="ignore")
                print(f"      [DEBUG] Response length: {len(raw)} chars")
                
                if resp.closed:
                    print(f"      [DEBUG] Response closed, cleaning up connection")
                    try:
                        conn.close()
                    except Exception as e:
                        print(f"      [DEBUG] Error closing connection: {e}")
                    self.conn = None
                
                return raw
                
            except Exception as e:
                print(f"      [DEBUG] Request failed: {e}")
                self.conn = None
                raise e
    
    def generate(self, model: str, system_prompt: str, user_prompt: str, options: dict) -> str:
        print(f"      [DEBUG] Starting generation for {model}")
        
        # Check cache first
        cache_key = _llm_cache._generate_key(model, system_prompt, user_prompt)
        cached_response = _llm_cache.get(cache_key)
        if cached_response:
            print(f"      [DEBUG] Cache hit for {model}")
            return cached_response
        
        print(f"      [DEBUG] Cache miss for {model}, generating new response")
        
        # Generate new response with simplified GPU optimization for stability
        body = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "keep_alive": "5m",  # Reduced for stability
            "options": {
                **options,
                "num_gpu": 1,           # Force GPU usage
                "num_thread": 2,        # Further reduced for stability
                "repeat_penalty": 1.1,  # Reduce repetition for speed
                "top_k": 40,            # Optimize sampling
                "num_predict": 32,      # Further reduced for speed
                "stop": ["\nUser:", "\nUSER:", "\nAssistant:", "\nASSISTANT:", "```", "```json"]
            }
        }
        
        try:
            print(f"      [DEBUG] Calling Ollama API for {model}")
            raw = self._request("/api/generate", body)
            
            print(f"      [DEBUG] Parsing response for {model}")
            try:
                data = json.loads(raw)
                response = data.get("response", raw)
                print(f"      [DEBUG] Successfully parsed response for {model}")
            except Exception as e:
                print(f"      [DEBUG] JSON parse failed for {model}: {e}")
                response = raw
            
            # Cache the response
            _llm_cache.put(cache_key, response)
            print(f"      [DEBUG] Cached response for {model}")
            return response
            
        except Exception as e:
            print(f"      [LLM_ERROR] {model}: {e}")
            # Return a safe fallback response
            error_response = '{"present": [], "evidence": {}}'
            _llm_cache.put(cache_key, error_response)
            return error_response

# Thread-local client storage
_thread_local = threading.local()

def get_thread_client():
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OllamaClient()
        _thread_local.client = client
    return client

# ----------------------------
# Text Processing Functions
# ----------------------------

def _fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Simple fuzzy matching using character overlap"""
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
    """Normalize text for consistent processing"""
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
    """Simple tokenization for chunking"""
    return text.split()

def sentence_chunks(text: str, target_tokens: int, overlap_tokens: int, 
                   min_tokens: int = 10, max_tokens: int = 150) -> List[str]:
    """Create overlapping chunks based on token count"""
    tokens = tokenize(text)
    chunks = []
    
    if len(tokens) <= target_tokens:
        return [text]
    
    # Safety check: ensure overlap is less than target_tokens to prevent infinite loops
    if overlap_tokens >= target_tokens:
        overlap_tokens = max(1, target_tokens // 2)
    
    start = 0
    iteration_count = 0
    max_iterations = len(tokens) * 2  # Safety limit
    
    while start < len(tokens) and iteration_count < max_iterations:
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
    
    if iteration_count >= max_iterations:
        print(f"      [WARNING] Reached max iterations in sentence_chunks, forcing completion")
        # Force create at least one chunk
        if not chunks:
            chunks = [text]
    
    return chunks if chunks else [text]

# ----------------------------
# File-based Processing Functions
# ----------------------------

def ensure_temp_dir():
    """Ensure temporary directory exists"""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"[INFO] Created temporary directory: {TEMP_DIR}")

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(TEMP_DIR)
        print(f"[INFO] Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        print(f"[WARNING] Could not clean up temp files: {e}")

def save_chunks_to_file(doc_id: str, chunks: List[str], strategy_name: str):
    """Save chunks to a temporary file"""
    filename = f"{CHUNK_FILE_PREFIX}{doc_id}_{strategy_name}.jsonl"
    filepath = os.path.join(TEMP_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i,
                "text": chunk,
                "length": len(chunk)
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
    
    print(f"      [FILE] Saved {len(chunks)} chunks to {filename}")
    return filepath

def load_chunks_from_file(filepath: str) -> List[str]:
    """Load chunks from a temporary file"""
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk_data = json.loads(line)
                chunks.append(chunk_data["text"])
    return chunks

def save_strategy_results(doc_id: str, strategy_name: str, entities: Set[str]):
    """Save strategy results to a temporary file"""
    filename = f"{STRATEGY_FILE_PREFIX}{doc_id}_{strategy_name}.json"
    filepath = os.path.join(TEMP_DIR, filename)
    
    results = {
        "strategy": strategy_name,
        "entities": list(entities),
        "count": len(entities),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"      [FILE] Saved strategy results to {filename}")
    return filepath

def load_strategy_results(filepath: str) -> Dict:
    """Load strategy results from a temporary file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def text_to_chunks_file(text: str, strategy: Dict, doc_id: str) -> str:
    """Convert text to chunks and save to file, return filepath"""
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
        max_iterations = len(words) * 2  # Safety limit
        
        while start < len(words) and iteration_count < max_iterations:
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
        
        if iteration_count >= max_iterations:
            print(f"      [WARNING] Reached max iterations, forcing completion")
            # Force create at least one chunk
            if not chunks:
                chunks = [text]
    
    if not chunks:
        chunks = [text]
        print(f"      [CHUNK] No chunks created, using original text")
    
    print(f"      [CHUNK] Created {len(chunks)} chunks for {strategy['name']}")
    
    # Save chunks to file and return filepath
    return save_chunks_to_file(doc_id, chunks, strategy['name'])

# ----------------------------
# Entity Detection Strategies
# ----------------------------

def regex_detection(text: str, entity_aliases: Dict[str, str]) -> Set[str]:
    """Strategy 0: Regex-based exact surface matching"""
    detected = set()
    text_norm = normalize_surface(text)
    
    # Build regex pattern for all aliases
    alias_patterns = []
    for alias, entity in entity_aliases.items():
        if alias.strip():
            # Escape special characters and create word boundary pattern
            escaped_alias = re.escape(alias.strip())
            pattern = rf'\b{escaped_alias}\b'
            alias_patterns.append((pattern, entity))
    
    # Find all matches
    for pattern, entity in alias_patterns:
        matches = re.finditer(pattern, text_norm, re.IGNORECASE)
        for match in matches:
            detected.add(entity)
    
    return detected

def llm_detection_strategy_file(text: str, strategy: Dict, entity_candidates: List[str], 
                               system_prompt: str, doc_id: str) -> str:
    """Run LLM-based detection for a specific strategy using files for memory efficiency"""
    try:
        print(f"      [DEBUG] Starting strategy: {strategy['name']}")
        print(f"      [DEBUG] Text length: {len(text)} chars")
        print(f"      [DEBUG] Entity candidates: {len(entity_candidates)}")
        
        # Validate strategy parameters
        if strategy["chunk_target"] <= 0 or strategy["chunk_overlap"] < 0:
            print(f"      [ERROR] Invalid strategy parameters: target={strategy['chunk_target']}, overlap={strategy['chunk_overlap']}")
            return save_strategy_results(doc_id, strategy['name'], set())
        
        # Step 1: Create chunks and save to file
        print(f"      [DEBUG] Creating chunks for {strategy['name']}...")
        chunks_filepath = text_to_chunks_file(text, strategy, doc_id)
        
        if not os.path.exists(chunks_filepath):
            print(f"      [ERROR] Chunks file not created for {strategy['name']}")
            return save_strategy_results(doc_id, strategy['name'], set())
        
        # Step 2: Load chunks one by one and process
        detected_entities = set()
        chunk_count = 0
        
        print(f"      [DEBUG] Processing chunks from file: {chunks_filepath}")
        
        with open(chunks_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    chunk_data = json.loads(line)
                    chunk = chunk_data["text"]
                    chunk_id = chunk_data["chunk_id"]
                    chunk_count += 1
                    
                    print(f"      [DEBUG] Processing chunk {chunk_id+1} for {strategy['name']}")
                    
                    if not chunk.strip():
                        print(f"      [DEBUG] Skipping empty chunk {chunk_id+1}")
                        continue
                    
                    print(f"      [DEBUG] Chunk {chunk_id+1} length: {len(chunk)} chars")
                    
                    # Build simplified prompt for stability
                    if strategy["model"] == "qwen2.5:3b":
                        # Special prompt for qwen2.5:3b to avoid reasoning
                        prompt = f"""TEXT: {chunk}

EXTRACT disease names. Return ONLY: ["disease1", "disease2"]"""
                    else:
                        # Standard prompt for other models
                        prompt = f"""Diseases in this text: {chunk}

Return ONLY a JSON list like: ["disease1", "disease2"]"""
                    
                    print(f"      [DEBUG] Built prompt for chunk {chunk_id+1}, calling LLM...")
                    
                    # Generate response with simplified options for stability
                    options = {
                        "temperature": strategy["temperature"],
                        "top_p": 0.9,
                        "num_predict": 32,  # Reduced for speed and stability
                        "num_gpu": 1,        # Force GPU usage
                        "num_thread": 2,     # Reduced for stability
                        "repeat_penalty": 1.1, # Reduce repetition for speed
                        "top_k": 40,         # Optimize sampling
                    }
                    
                    # ADVANCED RETRY SYSTEM with confidence scoring
                    max_retries = 3
                    present = []
                    retry_reason = "none"
                    final_attempt = 0
                    
                    for attempt in range(max_retries):
                        try:
                            print(f"      [DEBUG] Attempt {attempt+1}/{max_retries} for chunk {chunk_id+1}")
                            
                            client = get_thread_client()
                            response = client.generate(strategy["model"], system_prompt, prompt, options)
                            print(f"      [DEBUG] Got response for chunk {chunk_id+1}, length: {len(response)}")
                            
                            # Parse response - try multiple formats
                            print(f"      [DEBUG] Parsing response for chunk {chunk_id+1} (attempt {attempt+1})")
                            
                            # Try to find JSON array first (our expected format)
                            json_match = re.search(r'\[.*\]', response, re.DOTALL)
                            if json_match:
                                try:
                                    result = json.loads(json_match.group())
                                    if isinstance(result, list):
                                        present = result
                                        final_attempt = attempt + 1
                                        print(f"      [DEBUG] ✓ SUCCESS! Found JSON array with {len(present)} items: {present}")
                                        break  # Success, exit retry loop
                                    else:
                                        print(f"      [DEBUG] JSON array is not a list, retrying...")
                                        present = []
                                        retry_reason = "invalid_json_structure"
                                except json.JSONDecodeError:
                                    print(f"      [DEBUG] Failed to parse JSON array, retrying...")
                                    present = []
                                    retry_reason = "json_parse_error"
                            else:
                                # Try to find JSON object with "present" field
                                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                                if json_match:
                                    try:
                                        result = json.loads(json_match.group())
                                        present = result.get("present", [])
                                        if isinstance(present, list):
                                            final_attempt = attempt + 1
                                            print(f"      [DEBUG] ✓ SUCCESS! Found JSON object with 'present' field: {len(present)} items: {present}")
                                            break  # Success, exit retry loop
                                        else:
                                            print(f"      [DEBUG] 'present' field is not a list, retrying...")
                                            present = []
                                            retry_reason = "invalid_present_field"
                                    except json.JSONDecodeError:
                                        print(f"      [DEBUG] Failed to parse JSON object, retrying...")
                                        present = []
                                        retry_reason = "json_parse_error"
                                else:
                                    # No JSON found, retry
                                    print(f"      [DEBUG] No JSON found in attempt {attempt+1}, retrying...")
                                    print(f"      [DEBUG] Raw response: {response[:200]}...")
                                    present = []
                                    retry_reason = "no_json_found"
                            
                            # If we get here, parsing failed, try again
                            if attempt < max_retries - 1:
                                print(f"      [DEBUG] Parsing failed, waiting before retry...")
                                import time
                                time.sleep(1)  # Brief pause between retries
                            
                        except Exception as e:
                            print(f"      [CHUNK_ERROR] {strategy['name']} chunk {chunk_id+1} attempt {attempt+1}: {e}")
                            if attempt < max_retries - 1:
                                print(f"      [DEBUG] Will retry...")
                                import time
                                time.sleep(1)
                            else:
                                print(f"      [DEBUG] All retries exhausted for chunk {chunk_id+1}")
                                break
                    
                    # ADDITIONAL RETRY: If we got empty entities, try one more time
                    if not present and retry_reason != "none":
                        print(f"      [DEBUG] Empty entities detected, trying one more time for chunk {chunk_id+1}")
                        try:
                            # Modify prompt slightly to encourage entity detection
                            enhanced_prompt = f"""TEXT: {chunk}

EXTRACT disease names. If you find any diseases, return them as: ["disease1", "disease2"]
If you find NO diseases, return: []"""
                            
                            client = get_thread_client()
                            response = client.generate(strategy["model"], system_prompt, enhanced_prompt, options)
                            
                            # Try to parse again
                            json_match = re.search(r'\[.*\]', response, re.DOTALL)
                            if json_match:
                                try:
                                    result = json.loads(json_match.group())
                                    if isinstance(result, list):
                                        present = result
                                        final_attempt = 4  # Mark as 4th attempt (empty retry)
                                        print(f"      [DEBUG] ✓ SUCCESS on empty retry! Found {len(present)} items: {present}")
                                except json.JSONDecodeError:
                                    print(f"      [DEBUG] Empty retry failed to parse JSON")
                        except Exception as e:
                            print(f"      [DEBUG] Empty retry failed: {e}")
                    
                    # Store retry information for confidence scoring
                    chunk_retry_info = {
                        "final_attempt": final_attempt,
                        "retry_reason": retry_reason,
                        "entities_found": len(present) if present else 0
                    }
                    
                    # If all retries failed, try to extract from plain text as last resort
                    if not present and retry_reason != "none":
                        print(f"      [DEBUG] All retries failed, trying plain text extraction as last resort...")
                        # Look for disease-like patterns in the last response
                        disease_patterns = [
                            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|cancer|tumor|anemia|deficiency|mutation|gene)\b',
                            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:G\d+PD|BRCA\d+|ATM|LCAT)\b',
                            r'\b(?:G6PD|BRCA1|BRCA2|ATM|LCAT)\b'
                        ]
                        
                        for pattern in disease_patterns:
                            matches = re.findall(pattern, response, re.IGNORECASE)
                            for match in matches:
                                if match.lower() in [c.lower() for c in entity_candidates]:
                                    present.append(match)
                                    print(f"      [DEBUG] Extracted entity from text: {match}")
                        
                        if present:
                            print(f"      [DEBUG] Extracted {len(present)} entities from plain text as fallback")
                        else:
                            print(f"      [DEBUG] No entities found even with plain text extraction")
                    
                    # Process extracted entities
                    if isinstance(present, list) and present:
                        print(f"      [DEBUG] Processing {len(present)} entities: {present}")
                        print(f"      [DEBUG] Entity candidates: {entity_candidates}")
                        
                        for entity in present:
                            if entity and isinstance(entity, str):
                                # Check if entity matches any candidate (case-insensitive)
                                entity_lower = entity.lower().strip()
                                print(f"      [DEBUG] Checking entity: '{entity}' (lower: '{entity_lower}')")
                                
                                for candidate in entity_candidates:
                                    candidate_lower = candidate.lower().strip()
                                    print(f"      [DEBUG] Comparing with candidate: '{candidate}' (lower: '{candidate_lower}')")
                                    
                                    if candidate_lower == entity_lower:
                                        detected_entities.add(candidate)  # Use original candidate text
                                        print(f"      [DEBUG] ✓ MATCH! Found entity: {candidate} (matched: {entity})")
                                        break
                                    elif entity_lower in candidate_lower or candidate_lower in entity_lower:
                                        print(f"      [DEBUG] ~ PARTIAL MATCH: '{entity_lower}' vs '{candidate_lower}'")
                                        # Add partial matches with lower confidence
                                        detected_entities.add(candidate)
                                        print(f"      [DEBUG] ✓ PARTIAL MATCH! Added: {candidate}")
                                        break
                                    # Add fuzzy matching for similar terms
                                    elif _fuzzy_match(entity_lower, candidate_lower):
                                        print(f"      [DEBUG] ~ FUZZY MATCH: '{entity_lower}' vs '{candidate_lower}'")
                                        detected_entities.add(candidate)
                                        print(f"      [DEBUG] ✓ FUZZY MATCH! Added: {candidate}")
                                        break
                                    else:
                                        print(f"      [DEBUG] ✗ NO MATCH: '{entity_lower}' vs '{candidate_lower}'")
                        
                        print(f"      [DEBUG] Parsed {len(present)} entities from chunk {chunk_id+1}, {len(detected_entities)} matched candidates")
                    else:
                        print(f"      [DEBUG] No valid entities found in chunk {chunk_id+1} after {max_retries} attempts")
                        
                except json.JSONDecodeError as e:
                    print(f"      [JSON_ERROR] Failed to parse chunk line {line_num+1}: {e}")
                    continue
        
        print(f"      [DEBUG] Strategy {strategy['name']} completed with {len(detected_entities)} entities from {chunk_count} chunks")
        
        # Step 3: Save results to file
        results_filepath = save_strategy_results(doc_id, strategy['name'], detected_entities)
        
        # Step 4: Clean up chunks file to free memory
        try:
            os.remove(chunks_filepath)
            print(f"      [FILE] Cleaned up chunks file for {strategy['name']}")
        except Exception as e:
            print(f"      [WARNING] Could not clean up chunks file: {e}")
        
        return results_filepath
        
    except Exception as e:
        print(f"    [ERROR] Strategy {strategy['name']} failed: {e}")
        # Return empty results filepath
        empty_results = save_strategy_results(doc_id, strategy['name'], set())
        return empty_results

# ----------------------------
# Multi-Strategy Entity Detection
# ----------------------------

def run_multi_strategy_detection(text: str, entity_candidates: List[str], 
                                strategies: List[Dict], doc_id: str) -> Dict[str, Any]:
    """Run all detection strategies in parallel and combine results using files for memory efficiency"""
    
    # Strategy 0: Regex detection (baseline) - runs instantly
    print(f"  [STRATEGY] Running regex detection...")
    regex_entities = regex_detection(text, {c: c for c in entity_candidates})
    print(f"    [REGEX] Found {len(regex_entities)} entities")
    
    # System prompt for LLM strategies
    if any(s["model"] == "qwen3:4b" for s in strategies):
        # Special system prompt for qwen3:4b
        system_prompt = """You are a disease extractor. Extract disease names from biomedical text.

CRITICAL: Do NOT use reasoning or thinking. Return ONLY a JSON list of disease names.
Example: ["disease1", "disease2"]"""
    else:
        # Standard system prompt for other models
        system_prompt = """You are a biomedical entity extractor. Extract disease names and medical conditions from the text.

RULES:
1. Only extract entities that are diseases/conditions
2. Be conservative - if unsure, don't extract
3. Return entities in valid JSON format
4. Evidence must be the EXACT literal mention from the text

Output ONLY valid JSON with no explanations."""
    
    # Run exactly 4 strategies in parallel for maximum entity recovery
    strategy_results = {}
    strategy_filepaths = {}
    
    def run_strategy(strategy):
        try:
            print(f"  [STRATEGY] Starting {strategy['name']}...")
            results_filepath = llm_detection_strategy_file(text, strategy, entity_candidates, system_prompt, doc_id)
            print(f"    [{strategy['name']}] Completed and saved to file")
            return strategy['name'], results_filepath
        except Exception as e:
            print(f"    [ERROR] Strategy {strategy['name']} failed: {e}")
            return strategy['name'], None
    
    # Execute exactly 4 strategies in parallel for maximum recovery
    print(f"  [PARALLEL] Launching 4 strategies for maximum entity recovery...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # Exactly 4 workers
        # Submit both strategies simultaneously
        future_to_strategy = {
            executor.submit(run_strategy, strategy): strategy 
            for strategy in strategies
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_strategy):
            strategy_name, results_filepath = future.result()
            if results_filepath:
                strategy_filepaths[strategy_name] = results_filepath
                print(f"    [COMPLETED] {strategy_name} finished and saved")
            else:
                print(f"    [FAILED] {strategy_name} failed to complete")
    
    # Load results from files and combine
    all_detections = {"regex": regex_entities}
    
    for strategy_name, filepath in strategy_filepaths.items():
        try:
            results = load_strategy_results(filepath)
            all_detections[strategy_name] = set(results["entities"])
            print(f"    [LOADED] {strategy_name}: {len(results['entities'])} entities")
        except Exception as e:
            print(f"    [LOAD_ERROR] {strategy_name}: {e}")
            all_detections[strategy_name] = set()
    
    # ADVANCED CONFIDENCE SCORING with strategy weights
    entity_confidence = {}
    entity_strategies = defaultdict(list)
    
    for strategy_name, detected_entities in all_detections.items():
        strategy_weight = 1.0
        if strategy_name != "regex":
            strategy_weight = next(s["weight"] for s in strategies if s["name"] == strategy_name)
        
        for entity in detected_entities:
            if entity not in entity_confidence:
                entity_confidence[entity] = 0.0
                entity_strategies[entity] = []
            
            # Base confidence from strategy weight
            base_confidence = strategy_weight
            
            # Apply retry penalties if available
            # Note: retry_info is not currently implemented in the entity detection system
            # This section is reserved for future implementation of retry-based confidence scoring
            
            entity_confidence[entity] += base_confidence
            entity_strategies[entity].append(strategy_name)
    
    # Apply advanced confidence rules
    for entity in entity_confidence:
        # Rule 1: Regex detection gives maximum confidence
        if "regex" in entity_strategies[entity]:
            entity_confidence[entity] = min(1.0, entity_confidence[entity] * 1.5)
        
        # Rule 2: Multiple strategy detection increases confidence
        strategy_count = len(entity_strategies[entity])
        if strategy_count > 1:
            entity_confidence[entity] = min(1.0, entity_confidence[entity] * (1.0 + 0.2 * (strategy_count - 1)))
        
        # Rule 3: LLM-only detection gets penalty
        if "regex" not in entity_strategies[entity]:
            entity_confidence[entity] *= 0.8
        
        # Rule 4: Normalize to [0, 1] range
        entity_confidence[entity] = max(0.0, min(1.0, entity_confidence[entity]))
    
    # Filter entities based on confidence thresholds
    accepted_entities = []
    for entity, confidence in entity_confidence.items():
        if confidence >= CONFIDENCE_THRESHOLDS["min_accept"]:
            accepted_entities.append({
                "entity": entity,
                "confidence": confidence,
                "strategies": entity_strategies[entity],
                "strategy_count": len(entity_strategies[entity])
            })
    
    # Sort by confidence
    accepted_entities.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Clean up strategy result files to free memory
    for filepath in strategy_filepaths.values():
        try:
            os.remove(filepath)
            print(f"      [FILE] Cleaned up strategy results file")
        except Exception as e:
            print(f"      [WARNING] Could not clean up strategy file: {e}")
    
    return {
        "all_detections": {k: list(v) for k, v in all_detections.items()},
        "entity_confidence": entity_confidence,
        "entity_strategies": dict(entity_strategies),
        "accepted_entities": accepted_entities,
        "total_detected": len(entity_confidence),
        "total_accepted": len(accepted_entities)
    }

# ----------------------------
# Main Processing Function
# ----------------------------

def process_document(pmid: str, text: str, entity_candidates: List[str], 
                    strategies: List[Dict]) -> Dict[str, Any]:
    """Process a single document with multi-strategy detection"""
    print(f"[PROCESSING] PMID={pmid} | text_length={len(text)} | candidates={len(entity_candidates)}")
    
    t0 = time.time()
    
    # Run multi-strategy detection
    results = run_multi_strategy_detection(text, entity_candidates, strategies, pmid)
    
    # Prepare output
    output = {
        "PMID": pmid,
        "Texto": text,
        "Entidad": [{"texto": item["entity"], "tipo": "SpecificDisease", 
                     "confidence": item["confidence"], "strategies": item["strategies"]} 
                    for item in results["accepted_entities"]],
        "_multi_strategy": {
            "all_detections": results["all_detections"],
            "entity_confidence": results["entity_confidence"],
            "entity_strategies": results["entity_strategies"],
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
            "strategies_used": [s["name"] for s in strategies]
        },
        "_latency_sec": round(time.time() - t0, 3)
    }
    
    # Check if we have any entities detected
    if results['entity_confidence']:
        confidence_range = f"[{min(results['entity_confidence'].values()):.3f}, {max(results['entity_confidence'].values()):.3f}]"
    else:
        confidence_range = "[0.000, 0.000]"
    
    print(f"[COMPLETED] PMID={pmid} | accepted={len(results['accepted_entities'])} | confidence_range={confidence_range}")
    
    return output

# ----------------------------
# Main CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy NER System - Memory Efficient")
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL file with text and entities to search")
    parser.add_argument("--benchmark_jsonl", required=False, default=None, help="Benchmark JSONL file for evaluation (optional - without it, no metrics are calculated)")
    parser.add_argument("--out_pred", default="results_multi_strategy.jsonl", help="Output file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents")
    parser.add_argument("--strategies", nargs="+", default=["all"], 
                       help="Strategies to use (or 'all' for all strategies)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Minimum confidence threshold for acceptance")
    # Optional overrides per strategy
    parser.add_argument("--s1_target", type=int, default=None, help="Override STRATEGY_1 chunk_target")
    parser.add_argument("--s1_overlap", type=int, default=None, help="Override STRATEGY_1 chunk_overlap")
    parser.add_argument("--s1_min", type=int, default=None, help="Override STRATEGY_1 chunk_min")
    parser.add_argument("--s1_max", type=int, default=None, help="Override STRATEGY_1 chunk_max")
    parser.add_argument("--s1_temp", type=float, default=None, help="Override STRATEGY_1 temperature")
    parser.add_argument("--s2_target", type=int, default=None, help="Override STRATEGY_2 chunk_target")
    parser.add_argument("--s2_overlap", type=int, default=None, help="Override STRATEGY_2 chunk_overlap")
    parser.add_argument("--s2_min", type=int, default=None, help="Override STRATEGY_2 chunk_min")
    parser.add_argument("--s2_max", type=int, default=None, help="Override STRATEGY_2 chunk_max")
    parser.add_argument("--s2_temp", type=float, default=None, help="Override STRATEGY_2 temperature")
    
    args = parser.parse_args()
    
    # Update confidence threshold if provided
    CONFIDENCE_THRESHOLDS["min_accept"] = args.confidence_threshold
    
    # Select strategies
    if "all" in args.strategies:
        strategies = ALL_STRATEGIES
    else:
        strategies = [s for s in ALL_STRATEGIES if s["name"] in args.strategies]

    # Apply overrides
    name_to_strategy = {s["name"]: s for s in ALL_STRATEGIES}
    # STRATEGY_1 is expected to be ALL_STRATEGIES[0]
    s1 = ALL_STRATEGIES[0]
    if args.s1_target is not None:
        s1["chunk_target"] = args.s1_target
        # Auto-adjust min/max if not explicitly provided
        if args.s1_min is None:
            s1["chunk_min"] = max(5, int(s1["chunk_target"] * 0.5))
        if args.s1_max is None:
            s1["chunk_max"] = max(s1["chunk_target"], int(s1["chunk_target"] * 2))
    if args.s1_overlap is not None:
        s1["chunk_overlap"] = args.s1_overlap
    if args.s1_min is not None:
        s1["chunk_min"] = args.s1_min
    if args.s1_max is not None:
        s1["chunk_max"] = args.s1_max
    if args.s1_temp is not None:
        s1["temperature"] = args.s1_temp

    # STRATEGY_2 is expected to be ALL_STRATEGIES[1]
    if len(ALL_STRATEGIES) > 1:
        s2 = ALL_STRATEGIES[1]
        if args.s2_target is not None:
            s2["chunk_target"] = args.s2_target
            if args.s2_min is None:
                s2["chunk_min"] = max(5, int(s2["chunk_target"] * 0.5))
            if args.s2_max is None:
                s2["chunk_max"] = max(s2["chunk_target"], int(s2["chunk_target"] * 2))
        if args.s2_overlap is not None:
            s2["chunk_overlap"] = args.s2_overlap
        if args.s2_min is not None:
            s2["chunk_min"] = args.s2_min
        if args.s2_max is not None:
            s2["chunk_max"] = args.s2_max
        if args.s2_temp is not None:
            s2["temperature"] = args.s2_temp
    
    print(f"[CONFIG] Using {len(strategies)} strategies:")
    for s in strategies:
        print(f"  - {s['name']}: {s['model']} | chunks={s['chunk_target']}t | temp={s['temperature']}")
    
    print(f"[CONFIG] Confidence threshold: {CONFIDENCE_THRESHOLDS['min_accept']}")
    print(f"[CONFIG] Input file: {args.input_jsonl}")
    if args.benchmark_jsonl:
        print(f"[CONFIG] Benchmark file: {args.benchmark_jsonl}")
    else:
        print(f"[CONFIG] No benchmark file provided - metrics will not be calculated")
    
    # Ensure temporary directory exists
    ensure_temp_dir()
    
    # Load documents one by one to minimize memory usage
    print(f"[INFO] Processing documents with minimal memory usage...")
    
    results = []
    doc_count = 0
    
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            
            if args.limit > 0 and doc_count >= args.limit:
                break
            
            try:
                doc = json.loads(line)
                pmid = str(doc.get("PMID", f"doc_{line_num}"))
                text = doc.get("Texto", "")
                
                if not text.strip():
                    continue
                
                doc_count += 1
                print(f"\n[PROGRESS] {doc_count} (line {line_num+1})")
                print(f"[PROCESSING] PMID={pmid} | text_length={len(text)}")
                
                # Extract entity candidates from this document only
                entity_candidates = []
                entities = doc.get("Entidad", [])
                for ent in entities:
                    if isinstance(ent, dict) and "texto" in ent:
                        entity_candidates.append(ent["texto"])
                
                print(f"[INFO] Found {len(entity_candidates)} entity candidates in this document")
                
                # Process this document
                result = process_document(pmid, text, entity_candidates, strategies)
                results.append(result)
                
                print(f"[COMPLETED] Document {doc_count} processed successfully")
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"[ERROR] Failed to process line {line_num+1}: {e}")
                continue
    
    # Save results
    with open(args.out_pred, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Print summary
    print(f"\n[SUMMARY] Processed {len(results)} documents")
    print(f"[SUMMARY] Results saved to {args.out_pred}")
    
    total_entities = sum(len(r["Entidad"]) for r in results)
    avg_confidence = 0
    if results:
        all_confidences = []
        for r in results:
            for ent in r["Entidad"]:
                all_confidences.append(ent["confidence"])
        avg_confidence = sum(all_confidences) / len(all_confidences)
    
    print(f"[SUMMARY] Total entities detected: {total_entities}")
    print(f"[SUMMARY] Average confidence: {avg_confidence:.3f}")
    
    # Strategy performance analysis
    strategy_counts = defaultdict(int)
    for r in results:
        for ent in r["Entidad"]:
            for strategy in ent["strategies"]:
                strategy_counts[strategy] += 1
    
    print(f"\n[STRATEGY PERFORMANCE]")
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy}: {count} entities")
    
    # Clean up temporary files
    cleanup_temp_files()

if __name__ == "__main__":
    main()
