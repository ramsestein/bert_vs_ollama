#!/usr/bin/env python3
"""
llama_ner_optimized.py

Optimized verifier for NER over NCBI develop:
- Prefilter lexical + alias generation
- Fast surface matching via regex (longest-first alternation)
- Chunking ~40 tokens with overlap 20 favoring sentence boundaries
- Batch verification (candidate lists) per chunk using Ollama
- Per-thread persistent Ollama client
- Self-consistency re-try for doubtful cases
- Output JSONL with predictions and strict mention metrics

Usage example:
python llama_ner_optimized.py --develop_jsonl ./datasets/ncbi_develop.jsonl --limit 5 --model llama3.2:3b --n_workers 4
"""
import argparse
import json
import re
import http.client
import time
import threading
from typing import List, Dict, Any, Tuple, Set
from collections import Counter, deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Config defaults
# ----------------------------
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_WORKERS = 4
TIMEOUT_S = 120

GEN_OPTIONS = {
    "temperature": DEFAULT_TEMPERATURE,
    "top_p": 0.9,
    "num_predict": 512,
    "num_ctx": 1536,
    "cache_prompt": True,
}

DEFAULT_BATCH_SIZE = 12

# ----------------------------
# Enhanced negation/speculation detection
# ----------------------------
NEGATION_PATTERNS = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bsin\b", r"\bdenies\b", 
    r"\brule out\b", r"\bdescarta\b", r"\bnegativo para\b", r"\babsence of\b",
    r"\bfree of\b", r"\bclear of\b", r"\bnormal\b", r"\bhealthy\b"
]

SPECULATION_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bpossible\b", r"\bpossibly\b", 
    r"\bprobable\b", r"\blikely\b", r"\bsuggests\b", r"\bpotential\b",
    r"\bsuspected\b", r"\bquestionable\b", r"\buncertain\b"
]

# ----------------------------
# Noise term blacklist
# ----------------------------
NOISE_TERMS = {
    "was", "atm protein", "p53", "brca", "atm", "p53", "brca1", "brca2"
}

# ----------------------------
# Generic term patterns
# ----------------------------
GENERIC_TERMS = {
    "cancer", "tumour", "tumor", "disease", "syndrome", "deficiency"
}

# ----------------------------
# Optimized chunking parameters
# ----------------------------
CHUNK_TARGET_TOKENS = 80  # Increased from 60
CHUNK_OVERLAP_TOKENS = 25  # Reduced from 40 for better performance
CHUNK_MIN_TOKENS = 15
CHUNK_MAX_TOKENS = 120

# ----------------------------
# LLM Cache for performance
# ----------------------------
LLM_CACHE_SIZE = 1000
CACHE_TTL_HOURS = 24

# ----------------------------
# Self-consistency settings
# ----------------------------
SELF_CONSISTENCY_TEMP = 0.3

# ----------------------------
# Double-check strategy
# ----------------------------
DOUBLE_CHECK_ENABLED = True
DOUBLE_CHECK_STRICT = False  # Changed from True to False
DOUBLE_CHECK_SMART = True  # New: only check doubtful entities

# ----------------------------
# Entity filtering thresholds
# ----------------------------
MIN_OCCURRENCES = 1
RELIABILITY_THRESH = 0.2

# ----------------------------
# Confidence scoring
# ----------------------------
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.6
CONFIDENCE_LOW = 0.4

# ----------------------------
# Utilities: text normalization & tokenization
# ----------------------------
_ws_re = re.compile(r"\s+")
_token_re = re.compile(r"\S+")
_sentence_split_re = re.compile(r'(?<=[.!?])\s+')

# LLM Response Cache
class LLMCache:
    def __init__(self, max_size=LLM_CACHE_SIZE, ttl_hours=CACHE_TTL_HOURS):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.lock = threading.Lock()
    
    def _cleanup_expired(self):
        current_time = time.time()
        expired_keys = [k for k, (_, timestamp) in self.cache.items() 
                       if current_time - timestamp > self.ttl_seconds]
        for k in expired_keys:
            del self.cache[k]
    
    def get(self, key: str):
        with self.lock:
            self._cleanup_expired()
            if key in self.cache:
                response, timestamp = self.cache[key]
                if time.time() - timestamp <= self.ttl_seconds:
                    return response
                else:
                    del self.cache[key]
            return None
    
    def put(self, key: str, response: str):
        with self.lock:
            self._cleanup_expired()
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (response, time.time())
    
    def _generate_key(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """Generate cache key from prompt components"""
        import hashlib
        content = f"{model}:{system_prompt}:{user_prompt}"
        return hashlib.md5(content.encode()).hexdigest()

# Global cache instance
_llm_cache = LLMCache()

def normalize_surface(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

def clean_mention(s: str) -> str:
    s = normalize_surface(s)
    s = s.replace("##", "")
    s = re.sub(r'(^[\W_]+|[\W_]+$)', '', s)
    return s.strip()

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _token_re.findall(text)

def tokens_set(text: str) -> Set[str]:
    return set([t.lower() for t in tokenize(text)])

# ----------------------------
# Load develop and build global entity list with aliases
# ----------------------------
def load_develop_entities(path: str) -> List[Tuple[str, str]]:
    entities = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for ent in obj.get("Entidad", []) or []:
                txt = clean_mention(ent.get("texto", ""))
                typ = ent.get("tipo", "") or "SpecificDisease"
                if txt:
                    entities.add((txt, typ))
    return sorted(list(entities), key=lambda x: (x[1], x[0].lower()))

# ----------------------------
# Alias generation (simple deterministic rules)
# ----------------------------
_us_uk_map = {"tumour": "tumor", "tumors": "tumours", "anaemia": "anemia", "oestrogen": "estrogen"}

def generate_aliases(entity: str) -> Set[str]:
    """
    Generate deterministic surface variants:
    - remove hyphens vs hyphenated
    - expand parentheses abbreviations: keep both forms
    - plural/singular heuristics (naive)
    - US/UK alternate forms
    - remove commas, slashes, and other punctuation
    - more aggressive normalization
    """
    e = normalize_surface(entity)
    aliases = set([e, e.lower()])
    
    # remove hyphens/spaces variants
    aliases.add(e.replace("-", " "))
    aliases.add(e.replace(" - ", " "))
    aliases.add(e.replace(" ", "-"))
    
    # remove punctuation variants (more aggressive)
    aliases.add(e.replace(",", ""))
    aliases.add(e.replace("/", " "))
    aliases.add(e.replace("/", "-"))
    aliases.add(e.replace("\\", " "))
    aliases.add(e.replace("\\", "-"))
    aliases.add(e.replace(":", " "))
    aliases.add(e.replace(";", " "))
    
    # combinations of punctuation removal
    aliases.add(e.replace(",", "").replace("/", " "))
    aliases.add(e.replace(",", "").replace("/", "-"))
    aliases.add(e.replace(",", "").replace(":", " "))
    
    # parentheses -> add inside part if seems an abbreviation
    m = re.search(r"\(([^)]+)\)", e)
    if m:
        inside = m.group(1).strip()
        without = re.sub(r"\s*\([^)]+\)", "", e).strip()
        aliases.add(without)
        aliases.add(inside)
        # combine inside + without if inside likely abbreviation
        aliases.add(f"{inside} {without}")
        aliases.add(f"{without} ({inside})")
    
    # simple pluralization (naive)
    if e.endswith("y"):
        aliases.add(e[:-1] + "ies")
    if not e.endswith("s"):
        aliases.add(e + "s")
    if e.endswith("s"):
        aliases.add(e[:-1])
    
    # US/UK map
    for uk, us in _us_uk_map.items():
        if uk in e.lower():
            aliases.add(re.sub(uk, us, e, flags=re.I))
        if us in e.lower():
            aliases.add(re.sub(us, uk, e, flags=re.I))
    
    # Greek letter names common expansions (alpha,beta...)
    greeks = {"α":"alpha","β":"beta","γ":"gamma","delta":"δ"}
    for gsym, gname in greeks.items():
        if gsym in e:
            aliases.add(e.replace(gsym, gname))
        if gname in e:
            aliases.add(e.replace(gname, gsym))
    
    # clean and keep reasonable length aliases
    clean_aliases = set()
    for a in aliases:
        a2 = clean_mention(a)
        if 1 <= len(a2) <= 200:
            clean_aliases.add(a2)
    
    return clean_aliases

# ----------------------------
# Build inverted index token -> entities and a regex matcher (longest-first)
# ----------------------------
def build_indexes(entities_with_type: List[Tuple[str,str]]):
    # entity -> aliases
    entity_aliases = {}
    for ent, typ in entities_with_type:
        entity_aliases[ent] = sorted(generate_aliases(ent), key=lambda x: -len(x))
    # inverted token index
    token_index = defaultdict(set)  # token -> set(entity)
    entity_tokens = {}
    for ent, typ in entities_with_type:
        tokens = set()
        for alias in entity_aliases[ent]:
            for t in tokenize(alias):
                tokens.add(t.lower())
        entity_tokens[ent] = tokens
        for t in tokens:
            token_index[t].add(ent)
    # build regex patterns for fast surface matching (escape, longest-first)
    # We'll create per-entity joined alternation for exact matching: but to keep memory manageable
    # Build a global pattern of all aliases sorted by length desc, with capture group to map back
    alias_to_entity = {}
    aliases_all = []
    for ent, aliases in entity_aliases.items():
        for a in aliases:
            a_esc = re.escape(a)
            aliases_all.append((len(a), a_esc, ent, a))
            alias_to_entity[a] = ent
    aliases_all.sort(reverse=True, key=lambda x: x[0])  # longest first
    # chunk patterns into manageable groups to avoid huge single regex (but here one is okay)
    pattern_parts = [p[1] for p in aliases_all]
    # join with word boundaries where appropriate
    # Use lookaround to avoid partial matches inside words:
    # límites por no-palabra en lugar de \b para no romper "G6PD-deficiency", etc.
    global_pattern = r"(?i)(?<!\w)(" + "|".join(pattern_parts) + r")(?!\w)"
    try:
        global_re = re.compile(global_pattern)
    except re.error:
        # fallback sin límites
        global_pattern = r"(?i)(" + "|".join(pattern_parts) + r")"
        global_re = re.compile(global_pattern)

    # map alias lower->canonical entity
    alias_lower_to_entity = {p[3].lower(): p[2] for p in aliases_all}
    return {
        "entity_aliases": entity_aliases,
        "token_index": dict(token_index),
        "entity_tokens": entity_tokens,
        "global_re": global_re,
        "alias_to_entity": alias_to_entity,
        "alias_lower_to_entity": alias_lower_to_entity
    }

# ----------------------------
# Sentence-aware chunking: build chunks with intelligent overlap and size optimization
# ----------------------------
def sentence_chunks(text: str, target_tokens=CHUNK_TARGET_TOKENS, 
                   overlap_tokens=CHUNK_OVERLAP_TOKENS) -> List[str]:
    """
    Optimized chunking with intelligent overlap and size management.
    - Uses sentence boundaries when possible
    - Ensures minimum overlap for entity continuity
    - Balances chunk size for optimal LLM performance
    """
    text = normalize_surface(text)
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(_sentence_split_re, text)
    if not sentences:
        return []
    
    # Tokenize sentences and get lengths
    sent_toks = [tokenize(s) for s in sentences]
    sent_lens = [len(s) for s in sent_toks]
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        # Build chunk starting from sentence i
        cur_tokens = 0
        j = i
        
        # Accumulate sentences until we reach target size or have at least one sentence
        while j < len(sentences) and (cur_tokens < target_tokens or j == i):
            cur_tokens += sent_lens[j]
            j += 1
        
        # Create chunk from sentences i..j-1
        chunk_text = " ".join(sentences[i:j]).strip()
        if chunk_text and cur_tokens >= CHUNK_MIN_TOKENS:
            chunks.append(chunk_text)
        
        # Calculate next starting position with intelligent overlap
        if j >= len(sentences):
            break
            
        # Find overlap point: we want to ensure at least overlap_tokens tokens overlap
        overlap_remaining = overlap_tokens
        k = i
        
        # Move forward until we've covered the overlap
        while k < j and overlap_remaining > 0:
            overlap_remaining -= sent_lens[k]
            k += 1
        
        # Ensure we don't go backwards and maintain minimum overlap
        if k <= i:
            k = i + 1
        
        # Ensure we don't exceed maximum chunk size in next iteration
        next_chunk_size = sum(sent_lens[k:j])
        if next_chunk_size > CHUNK_MAX_TOKENS:
            # Find a better split point
            temp_size = 0
            temp_k = k
            while temp_k < j and temp_size < CHUNK_MAX_TOKENS:
                temp_size += sent_lens[temp_k]
                temp_k += 1
            k = temp_k
        
        i = k
    
    # Fallback: if no chunks were created, use simple sliding window
    if not chunks:
        toks = tokenize(text)
        i = 0
        step = max(1, target_tokens - overlap_tokens)
        
        while i < len(toks):
            j = min(len(toks), i + target_tokens)
            chunk_text = " ".join(toks[i:j])
            if len(toks[i:j]) >= CHUNK_MIN_TOKENS:
                chunks.append(chunk_text)
            
            if j >= len(toks):
                break
            i += step
    
    # Post-process chunks: merge very small chunks and ensure reasonable sizes
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = len(tokenize(chunk))
        if chunk_tokens < CHUNK_MIN_TOKENS and final_chunks:
            # Merge with previous chunk if too small
            final_chunks[-1] = final_chunks[-1] + " " + chunk
        else:
            final_chunks.append(chunk)
    
    return final_chunks

# ----------------------------
# Ollama client with thread-local reuse
# ----------------------------
_thread_local = threading.local()

class OllamaClient:
    def __init__(self, host="localhost", port=11434, timeout=TIMEOUT_S):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        self.lock = threading.Lock()

    def _request(self, path: str, body: dict) -> str:
        payload = json.dumps(body)
        with self.lock:
            self.conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
            resp = self.conn.getresponse()
            raw = resp.read().decode("utf-8", errors="ignore")
            if resp.closed:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            return raw

    def pull(self, model: str):
        body = {"name": model, "stream": False}
        return self._request("/api/pull", body)

    def generate(self, model: str, system_prompt: str, user_prompt: str, options: dict) -> str:
        # Check cache first
        cache_key = _llm_cache._generate_key(model, system_prompt, user_prompt)
        cached_response = _llm_cache.get(cache_key)
        if cached_response:
            return cached_response
        
        # Generate new response
        body = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "keep_alive": "15m",
            "options": options,
            "stop": ["\nUser:", "\nUSER:", "\nAssistant:", "\nASSISTANT:"]
        }
        
        try:
            raw = self._request("/api/generate", body)
            try:
                data = json.loads(raw)
                response = data.get("response", raw)
            except Exception:
                response = raw
            
            # Cache the response
            _llm_cache.put(cache_key, response)
            return response
            
        except Exception as e:
            # Return a safe fallback response
            error_response = '{"present": [], "evidence": {}}'
            _llm_cache.put(cache_key, error_response)
            return error_response

def get_thread_client():
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OllamaClient()
        _thread_local.client = client
    return client

# ----------------------------
# Prompts (English, safe JSON, braces escaped)
# ----------------------------
SYSTEM_VERIFY_BATCH = (
    "You are a STRICT, factual, literal verifier. Output ONLY valid JSON. "
    "Task: Given a short biomedical fragment and a SMALL candidate list of DISEASE strings, "
    "return which of those candidates are EXPLICITLY and LITERALLY mentioned in the fragment. "
    "Be VERY conservative - only mark entities that you are 100% certain appear in the text."
)

SYSTEM_DOUBLE_CHECK = (
    "You are a PRECISE biomedical entity verifier. Output ONLY valid JSON. "
    "Task: Given a biomedical text fragment and a SINGLE disease entity, "
    "determine if this entity is EXPLICITLY and LITERALLY mentioned in the text. "
    "Be EXTREMELY strict - only confirm if you are 100% certain the entity appears exactly as stated."
)

# Dynamic prompt construction without empty placeholders
def build_verify_prompt(candidates: List[str], chunk: str) -> str:
    cand_block = "\n".join(f"- {c}" for c in candidates if c.strip())
    return f"""
Return ONLY this JSON (and nothing else):
{{
  "present": ["<entity1>", "..."],
  "evidence": {{
    "<entity1>": "<literal mention from the fragment>",
    "..." : "..."
  }}
}}

CRITICAL RULES:
- Be EXTREMELY conservative - only mark entities that are 100% explicitly mentioned
- Do NOT infer, guess, or assume - if you're not 100% sure, leave it out
- Evidence must be the EXACT literal substring from the fragment
- Partial matches or similar terms are NOT valid
- If none match, use "present": [] and "evidence": {{}}

EXAMPLES:
- "breast cancer" in text "The patient has breast cancer" → ACCEPT
- "diabetes" in text "The patient has type 2 diabetes" → ACCEPT  
- "cancer" in text "The patient has breast cancer" → REJECT (not exact match)
- "leukemia" in text "The patient has cancer" → REJECT (not mentioned)

CANDIDATE_ENTITIES:
{cand_block}

FRAGMENT:
<<< {chunk} >>>
""".strip()

def build_double_check_prompt(entity: str, chunk: str) -> str:
    return f"""
Return ONLY this JSON (and nothing else):
{{
  "is_present": true/false,
  "confidence": "high/medium/low",
  "evidence": "<exact literal mention from the fragment or empty string>",
  "reason": "<brief explanation>"
}}

CRITICAL RULES:
- Be EXTREMELY strict - only confirm if the entity is 100% explicitly mentioned
- Do NOT infer, guess, or assume - if you're not 100% sure, mark as false
- Evidence must be the EXACT literal substring from the fragment
- Partial matches or similar terms are NOT valid
- If the entity is not mentioned, use "is_present": false

ENTITY TO VERIFY: {entity}

FRAGMENT:
<<< {chunk} >>>
""".strip()

# ----------------------------
# Parse response robustly
# ----------------------------
def parse_verify_batch(text: str) -> Tuple[List[str], Dict[str,str]]:
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return [], {}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return [], {}
    present = obj.get("present", [])
    evidence = obj.get("evidence", {})
    if not isinstance(present, list):
        present = []
    if not isinstance(evidence, dict):
        evidence = {}
    present = [str(x) for x in present]
    evidence = {str(k): str(v) for k,v in evidence.items()}
    return present, evidence

def parse_double_check(text: str) -> Tuple[bool, str, str]:
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return False, "", "parse_error"
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return False, "", "parse_error"
    
    is_present = obj.get("is_present", False)
    confidence = obj.get("confidence", "low")
    evidence = obj.get("evidence", "")
    reason = obj.get("reason", "")
    
    if not isinstance(is_present, bool):
        is_present = False
    if not isinstance(evidence, str):
        evidence = ""
    if not isinstance(reason, str):
        reason = ""
    
    return is_present, evidence, reason

# ----------------------------
# Prefilter: candidate set per chunk using token-index & regex exact matches
# ----------------------------
def candidates_for_chunk(chunk: str, idxs: Dict, min_tokens_match=1) -> Tuple[Set[str], Set[str]]:
    """
    Optimized candidate selection with intelligent filtering:
    - Fast exact surface matching via regex
    - Smart token-based prefiltering with Jaccard similarity
    - Entity length and frequency considerations
    - Deduplication and ranking
    """
    chunk_norm = normalize_surface(chunk)
    chunk_tokens = set([t.lower() for t in tokenize(chunk_norm)])
    
    # Surface matching with global regex (fast exact matches)
    global_re = idxs["global_re"]
    alias_lower_to_entity = idxs["alias_lower_to_entity"]
    exact_set = set()
    
    for m in global_re.finditer(chunk_norm):
        found = m.group(1)
        if not found:
            continue
        key = found.lower()
        ent = alias_lower_to_entity.get(key)
        if ent:
            exact_set.add(ent)
    
    # Token-based prefiltering with improved heuristics
    token_index = idxs["token_index"]
    entity_tokens = idxs["entity_tokens"]
    
    # Collect potential candidates based on token overlap
    candidate_scores = {}
    
    for t in chunk_tokens:
        if len(t) < 2:  # Skip very short tokens
            continue
        for ent in token_index.get(t, ()):
            if ent not in candidate_scores:
                candidate_scores[ent] = {"tokens": set(), "score": 0}
            candidate_scores[ent]["tokens"].add(t)
            candidate_scores[ent]["score"] += 1
    
    # Score candidates based on multiple factors
    final_candidates = set()
    
    for ent, score_info in candidate_scores.items():
        ent_toks = entity_tokens.get(ent, set())
        if not ent_toks:
            continue
        
        # Calculate overlap and Jaccard similarity
        overlap = len(score_info["tokens"] & ent_toks)
        total_tokens = len(ent_toks | chunk_tokens)
        jaccard = overlap / max(1, total_tokens)
        
        # Scoring criteria:
        # 1. High token overlap (≥3 tokens)
        # 2. Good Jaccard similarity (≥0.5)
        # 3. Entity length consideration (prefer reasonable lengths)
        # 4. Token frequency in chunk
        
        score = 0
        if overlap >= 3:
            score += 3
        elif overlap >= 2:
            score += 2
        elif overlap >= 1:
            score += 1
        
        if jaccard >= 0.7:
            score += 3
        elif jaccard >= 0.5:
            score += 2
        elif jaccard >= 0.3:
            score += 1
        
        # Bonus for entities with reasonable length (not too short, not too long)
        ent_len = len(ent.split())
        if 2 <= ent_len <= 8:
            score += 1
        
        # Bonus for high-frequency tokens in chunk
        if score_info["score"] >= 2:
            score += 1
        
        # Accept candidates with good scores
        if score >= 3 or (overlap >= 2 and jaccard >= 0.4):
            final_candidates.add(ent)
    
    # Always include exact matches
    final_candidates |= exact_set
    
    # Limit candidates to prevent overwhelming the LLM
    if len(final_candidates) > 50:
        # Keep exact matches and top-scoring candidates
        exact_and_top = list(exact_set)
        remaining = [ent for ent in final_candidates if ent not in exact_set]
        # Sort by score and take top
        remaining_sorted = sorted(remaining, 
                                key=lambda x: candidate_scores.get(x, {}).get("score", 0), 
                                reverse=True)
        final_candidates = set(exact_and_top + remaining_sorted[:47])  # 50 - 3 exact
    
    return exact_set, final_candidates

# ----------------------------
# Run pipeline for one record
# ----------------------------
def run_for_record(pmid: str, text: str, global_entities: List[Tuple[str,str]], idxs: Dict,
                   model: str, batch_size: int, min_occ: int, rel_thresh: float,
                   self_consistency: bool, gen_options_base: Dict, double_check: bool = True,
                   verify_model: str = None, verify_sc_n: int = 3, verify_options: Dict = None) -> Dict[str, Any]:
    t0 = time.time()
    text = normalize_surface(text)
    
    # Use optimized chunking parameters
    chunks = sentence_chunks(text, target_tokens=CHUNK_TARGET_TOKENS, 
                           overlap_tokens=CHUNK_OVERLAP_TOKENS)
    n_chunks = len(chunks)
    
    counts = Counter()
    evidence_store = defaultdict(list)
    exact_hits = Counter()
    seen_checks = Counter()
    
    # Precompute entity list and mapping
    entity_list = [e for e,_ in global_entities]
    entity_types = {e:t for e,t in global_entities}
    
    client = get_thread_client()
    
    # Process chunks with improved candidate selection
    for chunk_idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        exact_matches, token_candidates = candidates_for_chunk(chunk, idxs)
        
        # Debug info for this chunk
        if len(exact_matches) > 0 or len(token_candidates) > 0:
            chunk_tokens = len(tokenize(chunk))
            print(f"  [CHUNK] PMID={pmid} | chunk={chunk_idx+1}/{n_chunks} | "
                  f"tokens={chunk_tokens} | exact={len(exact_matches)} | candidates={len(token_candidates)}")
        
        # Process exact matches immediately
        for ent in exact_matches:
            counts[ent] += 1
            exact_hits[ent] += 1
            evidence_store[ent].append(("exact_surface", chunk))
        
        # Process remaining candidates with LLM
        to_verify = sorted(list(token_candidates - exact_matches))
        if not to_verify:
            continue
            
        # Batch processing with improved error handling
        for i in range(0, len(to_verify), batch_size):
            batch = to_verify[i:i+batch_size]
            prompt = build_verify_prompt(batch, chunk)
            
            print(f"    [BATCH] PMID={pmid} | chunk={chunk_idx+1} | "
                  f"batch_size={len(batch)} | entities={batch[:3]}...")
            
            # Track seen entities
            for ent in batch:
                seen_checks[ent] += 1 
            
            try:
                # Use dedicated verification model/options if provided
                v_model = verify_model or model
                v_opts = verify_options or gen_options_base

                # Self-consistency with majority voting
                votes = Counter()
                last_evidence = {}
                runs = max(1, int(verify_sc_n) if self_consistency else 1)
                for _run in range(runs):
                    resp = client.generate(v_model, SYSTEM_VERIFY_BATCH, prompt, v_opts)
                    present_r, evidence_r = parse_verify_batch(resp)
                    for ent in present_r:
                        if ent in batch:
                            votes[ent] += 1
                            if ent not in last_evidence:
                                last_evidence[ent] = evidence_r.get(ent, "")

                # Majority threshold
                need = (runs // 2) + 1
                present = [ent for ent, c in votes.items() if c >= need]

                print(f"      [LLM] PMID={pmid} | runs={runs} | need={need} | found={len(present)} | entities={present[:3]}...")

                # Record results
                for ent in present:
                    if ent in batch:
                        counts[ent] += 1
                        evidence_store[ent].append(("llm", last_evidence.get(ent, "")))
                        # also store the chunk context for negation/speculation analysis
                        evidence_store[ent].append(("llm_chunk", chunk))
                        
            except Exception as e:
                print(f"      [ERROR] PMID={pmid} | chunk={chunk_idx+1} | batch processing failed: {e}")
                continue
    
    # Decide final mentions with improved filtering
    final = []
    # Helper gates
    def _is_noise_term(e: str) -> bool:
        return e.strip().lower() in NOISE_TERMS

    def _is_generic_unmodified(e: str) -> bool:
        e_norm = e.strip().lower()
        return (e_norm in GENERIC_TERMS) or (len(e.split()) == 1 and e_norm in GENERIC_TERMS)

    def _negated_or_speculative(ent_text: str, contexts: List[str]) -> bool:
        ent_l = ent_text.lower()
        
        for ctx in contexts:
            ctx_l = ctx.lower()
            # Check for negation patterns in proximity
            for pattern in NEGATION_PATTERNS:
                if re.search(pattern + r"[^.]{0,50}" + re.escape(ent_l), ctx_l) or \
                   re.search(re.escape(ent_l) + r"[^.]{0,50}" + pattern, ctx_l):
                    return True
            
            # Check for speculation patterns in proximity
            for pattern in SPECULATION_PATTERNS:
                if re.search(pattern + r"[^.]{0,50}" + re.escape(ent_l), ctx_l) or \
                   re.search(re.escape(ent_l) + r"[^.]{0,50}" + pattern, ctx_l):
                    return True
        return False

    def _calculate_confidence_score(ent: str, cnt: int, exact_count: int, 
                                  reliability: float, unique_chunks: int) -> float:
        """Calculate confidence score for entity acceptance decision"""
        score = 0.0
        
        # Base score from exact matches
        if exact_count >= 2:
            score += 0.8
        elif exact_count == 1:
            score += 0.6
        
        # Reliability bonus
        if reliability >= CONFIDENCE_HIGH:
            score += 0.2
        elif reliability >= CONFIDENCE_MEDIUM:
            score += 0.1
        
        # Chunk diversity bonus
        if unique_chunks >= 3:
            score += 0.2
        elif unique_chunks >= 2:
            score += 0.1
        
        # Entity length bonus (prefer specific terms)
        if len(ent.split()) >= 3:
            score += 0.1
        
        return min(1.0, score)

    def _should_double_check(ent: str, confidence: float, exact_count: int) -> bool:
        """Smart decision on whether to double-check an entity"""
        if not DOUBLE_CHECK_SMART:
            return True
        
        # High confidence entities don't need double-check
        if confidence >= CONFIDENCE_HIGH and exact_count >= 1:
            return False
        
        # Low confidence or no exact matches need verification
        if confidence < CONFIDENCE_MEDIUM or exact_count == 0:
            return True
        
        # Generic terms always need verification
        if _is_generic_unmodified(ent):
            return True
        
        return False

    for ent, cnt in counts.items():
        # Calculate reliability: how many times we hit vs how many times we checked
        denom_seen = max(1, seen_checks.get(ent, 0))
        reliability = cnt / denom_seen
        
        # Additional confidence metrics
        exact_count = exact_hits.get(ent, 0)
        llm_count = cnt - exact_count
        
        # Count unique chunks where this entity was found
        unique_chunks = set()
        for evidence_type, chunk_text in evidence_store[ent]:
            unique_chunks.add(chunk_text[:100])  # Use first 100 chars as chunk identifier
        
        # Calculate confidence score
        confidence = _calculate_confidence_score(ent, cnt, exact_count, reliability, len(unique_chunks))
        
        # Acceptance criteria with improved logic:
        # (a) 2+ exact matches (high confidence)
        # (b) 1 exact match + meets thresholds
        # (c) High confidence without exact matches (multiple chunks + high reliability)
        accepted = False
        reason = ""
        
        if exact_count >= 2:
            accepted = True
            reason = "2+ exact matches"
        elif exact_count >= 1 and cnt >= min_occ and reliability >= rel_thresh:
            accepted = True
            reason = "1 exact + meets thresholds"
        elif exact_count == 0 and cnt >= (min_occ + 1) and reliability >= (rel_thresh + 0.15):
            accepted = True
            reason = "high confidence no exact"
        
        # Additional filtering: require evidence from multiple chunks for high confidence
        if accepted:
            # For entities without exact matches, require evidence from multiple chunks
            if exact_count == 0 and len(unique_chunks) < 2:
                accepted = False
                reason = f"rejected: only {len(unique_chunks)} chunk(s) without exact matches"
                print(f"  [REJECT] PMID={pmid} | entity={ent} | count={cnt} | reliability={reliability:.3f} | "
                      f"exact_hits={exact_count} | chunks={len(unique_chunks)} | reason={reason}")
            else:
                # Additional gates: noise and generic terms (unless strong evidence)
                if _is_noise_term(ent):
                    accepted = False
                    reason = "noise term"
                    print(f"  [REJECT] PMID={pmid} | entity={ent} | reason={reason}")
                elif _is_generic_unmodified(ent) and exact_count == 0:
                    accepted = False
                    reason = "generic term without modifier"
                    print(f"  [REJECT] PMID={pmid} | entity={ent} | reason={reason}")
                else:
                    # Negation/speculation gate using available contexts
                    contexts = [ctx for typ, ctx in evidence_store[ent] if typ in ("exact_surface", "llm_chunk")]
                    if contexts and _negated_or_speculative(ent, contexts):
                        accepted = False
                        reason = "negated/speculative context"
                        print(f"  [REJECT] PMID={pmid} | entity={ent} | reason={reason}")
                    else:
                        final.append(ent)
                        print(f"  [ACCEPT] PMID={pmid} | entity={ent} | count={cnt} | reliability={reliability:.3f} | "
                              f"exact_hits={exact_count} | chunks={len(unique_chunks)} | confidence={confidence:.3f} | reason={reason}")
        else:
            print(f"  [REJECT] PMID={pmid} | entity={ent} | count={cnt} | reliability={reliability:.3f} | "
                  f"exact_hits={exact_count} | min_occ={min_occ} | rel_thresh={rel_thresh}")
    
    # No postprocessing needed for now
    # final = postprocess_entities_textonly(final)
    
    # SMART DOUBLE CHECK STRATEGY: Only verify doubtful entities
    if double_check and final:
        print(f"[DOUBLE_CHECK] PMID={pmid} | Starting smart double-check for {len(final)} entities")
        double_check_final = []
        
        for entity in final:
            # Calculate confidence for this entity
            entity_count = counts.get(entity, 0)
            entity_exact = exact_hits.get(entity, 0)
            entity_reliability = entity_count / max(1, seen_checks.get(entity, 0))
            entity_chunks = len(set([ctx[:100] for typ, ctx in evidence_store[entity]]))
            entity_confidence = _calculate_confidence_score(entity, entity_count, entity_exact, 
                                                         entity_reliability, entity_chunks)
            
            # Decide if this entity needs double-check
            needs_check = _should_double_check(entity, entity_confidence, entity_exact)
            
            if not needs_check:
                # High confidence entity - accept without double-check
                double_check_final.append(entity)
                print(f"  [DOUBLE_CHECK] PMID={pmid} | entity={entity} | SKIP | confidence={entity_confidence:.3f} | exact_hits={entity_exact}")
                continue
            
            # Perform double-check for doubtful entities
            print(f"  [DOUBLE_CHECK] PMID={pmid} | entity={entity} | VERIFY | confidence={entity_confidence:.3f}")
            
            # Check if entity appears in any chunk
            entity_found = False
            total_confidence = 0
            total_checks = 0
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                # Build double-check prompt for this entity
                prompt = build_double_check_prompt(entity, chunk)
                
                try:
                    resp = client.generate(model, SYSTEM_DOUBLE_CHECK, prompt, gen_options_base)
                    is_present, evidence, reason = parse_double_check(resp)
                    
                    if is_present:
                        entity_found = True
                        total_confidence += 1
                    total_checks += 1
                    
                    # Debug info
                    print(f"    [DOUBLE_CHECK] PMID={pmid} | entity={entity} | chunk={chunk[:50]}... | present={is_present} | reason={reason}")
                    
                except Exception as e:
                    print(f"    [DOUBLE_CHECK] PMID={pmid} | entity={entity} | error: {e}")
                    total_checks += 1
            
            # Entity is confirmed if found in at least one chunk
            if entity_found:
                # In strict mode, require higher confidence
                if DOUBLE_CHECK_STRICT and total_confidence < 1:
                    print(f"  [DOUBLE_CHECK] PMID={pmid} | entity={entity} | REJECTED | strict mode requires 1+ confirmations | confidence={total_confidence}/{total_checks}")
                else:
                    double_check_final.append(entity)
                    print(f"  [DOUBLE_CHECK] PMID={pmid} | entity={entity} | CONFIRMED | confidence={total_confidence}/{total_checks}")
            else:
                print(f"  [DOUBLE_CHECK] PMID={pmid} | entity={entity} | REJECTED | not found in any chunk")
        
        final = double_check_final
        print(f"[DOUBLE_CHECK] PMID={pmid} | Final entities after smart double-check: {len(final)}")
    
    print(f"[DEBUG] PMID={pmid} | chunks={n_chunks} | exact_total={sum(exact_hits.values())} | "
          f"verified_entities={len(seen_checks)} | candidates_seen={sum(seen_checks.values())} | "
          f"final_entities={len(final)} | total_entities_processed={len(counts)}")
    rec = {
        "PMID": pmid,
        "Texto": text,
        "Entidad": [{"texto": m, "tipo": entity_types.get(m, "SpecificDisease")} for m in final],
        "_debug": {
            "n_chunks": n_chunks,
            "counts_top10": dict(sorted(counts.items(), key=lambda x:-x[1])[:10]),
            "evidence_examples": {k: evidence_store[k][:3] for k in list(evidence_store)[:8]},
            "exact_hits": dict(exact_hits),
            "seen_checks": dict(seen_checks),
            "min_occurrences": min_occ,
            "reliability_thresh": rel_thresh,
            "double_check_enabled": double_check
        },
        "_latency_sec": round(time.time() - t0, 3)
    }
    return rec

# ----------------------------
# I/O and metrics
# ----------------------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, recs: List[Dict[str,Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_for_match(text: str) -> str:
    """
    Normalize entity mention for case-insensitive, punctuation-insensitive matching.
    """
    if not text:
        return ""
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_strict(gold_records: List[Dict[str,Any]], pred_records: List[Dict[str,Any]]):
    from collections import Counter
    gold_by_id = {str(r.get("PMID")): r for r in gold_records}
    pred_by_id = {str(r.get("PMID")): r for r in pred_records}
    total_tp = total_pred = total_gold = 0
    for pmid, grec in gold_by_id.items():
        prec = pred_by_id.get(pmid, {"Entidad":[]})
        gold_mentions = [norm_for_match(e.get("texto","")) for e in (grec.get("Entidad") or []) if e.get("texto")]
        pred_mentions = [norm_for_match(e.get("texto","")) for e in (prec.get("Entidad") or []) if e.get("texto")]
        c_gold = Counter(gold_mentions)
        c_pred = Counter(pred_mentions)
        c_tp = c_gold & c_pred
        total_tp += sum(c_tp.values())
        total_gold += sum(c_gold.values())
        total_pred += sum(c_pred.values())
    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, total_pred, total_gold, total_tp

# ----------------------------
# Heartbeat
# ----------------------------
def start_heartbeat(total_jobs: int, interval=4, qps_window=50):
    state = {"done":0,"errors":0,"start":time.time(),"lat_hist":deque(maxlen=qps_window),"stop":False}
    def beat():
        while not state["stop"]:
            time.sleep(interval)
            done = state["done"]; errs = state["errors"]
            elapsed = time.time() - state["start"]
            avg_lat = sum(state["lat_hist"])/len(state["lat_hist"]) if state["lat_hist"] else 0.0
            qps = done/elapsed if elapsed>0 else 0.0
            eta = (elapsed/done*(total_jobs-done)) if done>0 else 0.0
            print(f"[HEARTBEAT] {done}/{total_jobs} ({100*done/total_jobs:.1f}%) | elapsed={elapsed:.1f}s | ETA={eta:.1f}s | QPS={qps:.2f} | avg_lat={avg_lat:.2f}s | errors={errs}")
    th = threading.Thread(target=beat, daemon=True)
    th.start()
    return state

# ----------------------------
# Main CLI
# ----------------------------
def main():
    # Global declarations
    global CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS
    global LLM_CACHE_SIZE, _llm_cache
    
    ap = argparse.ArgumentParser(description="Optimized batch-verify NER with intelligent chunking and caching")
    ap.add_argument("--develop_jsonl", required=True)
    ap.add_argument("--out_pred", default="results_llama_optimized.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--n_workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--min_occurrences", type=int, default=MIN_OCCURRENCES)
    ap.add_argument("--reliability_thresh", type=float, default=RELIABILITY_THRESH)
    ap.add_argument("--self_consistency", action="store_true")
    ap.add_argument("--double_check", action="store_true", default=DOUBLE_CHECK_ENABLED)
    
    # New optimization parameters
    ap.add_argument("--chunk_target_tokens", type=int, default=CHUNK_TARGET_TOKENS,
                   help="Target tokens per chunk (default: 60)")
    ap.add_argument("--chunk_overlap_tokens", type=int, default=CHUNK_OVERLAP_TOKENS,
                   help="Overlap tokens between chunks (default: 30)")
    ap.add_argument("--chunk_min_tokens", type=int, default=CHUNK_MIN_TOKENS,
                   help="Minimum tokens per chunk (default: 20)")
    ap.add_argument("--chunk_max_tokens", type=int, default=CHUNK_MAX_TOKENS,
                   help="Maximum tokens per chunk (default: 120)")
    ap.add_argument("--enable_cache", action="store_true", default=True,
                   help="Enable LLM response caching (default: True)")
    ap.add_argument("--cache_size", type=int, default=LLM_CACHE_SIZE,
                   help="LLM cache size (default: 1000)")
    
    args = ap.parse_args()
    
    # Update global parameters based on command line arguments
    CHUNK_TARGET_TOKENS = args.chunk_target_tokens
    CHUNK_OVERLAP_TOKENS = args.chunk_overlap_tokens
    CHUNK_MIN_TOKENS = args.chunk_min_tokens
    CHUNK_MAX_TOKENS = args.chunk_max_tokens
    
    if args.enable_cache:
        LLM_CACHE_SIZE = args.cache_size
        _llm_cache = LLMCache(max_size=LLM_CACHE_SIZE, ttl_hours=CACHE_TTL_HOURS)
    else:
        _llm_cache = None

    print(f"[INFO] model={args.model} temp={args.temperature} workers={args.n_workers} batch_size={args.batch_size}")
    print(f"[INFO] min_occurrences={args.min_occurrences} reliability_thresh={args.reliability_thresh} self_consistency={args.self_consistency}")
    print(f"[INFO] chunking: target={CHUNK_TARGET_TOKENS} overlap={CHUNK_OVERLAP_TOKENS} min={CHUNK_MIN_TOKENS} max={CHUNK_MAX_TOKENS}")
    print(f"[INFO] caching: enabled={args.enable_cache} size={args.cache_size}")
    
    all_records = list(load_jsonl(args.develop_jsonl))
    gold = all_records[:args.limit] if args.limit and args.limit>0 else all_records
    print(f"[INFO] Loaded {len(gold)} records (will process).")

    # Build global entity list and indexes
    global_entities = load_develop_entities(args.develop_jsonl)
    print(f"[INFO] Global entity vocabulary size: {len(global_entities)}")
    idxs = build_indexes(global_entities)
    
    # Count total aliases
    total_aliases = sum(len(aliases) for aliases in idxs["entity_aliases"].values())
    regex_pattern = idxs["global_re"].pattern
    print(f"[INFO] Built token index and regex matcher. Total aliases: {total_aliases}")
    print(f"[INFO] Regex pattern length: {len(regex_pattern)} characters")

    # Warm-up one client
    client = get_thread_client()
    print("[WARMUP] pulling model...")
    try:
        client.pull(args.model)
    except Exception as e:
        print(f"[WARN] pull failed or already present: {e}")
    print("[WARMUP] running small warm-up...")
    try:
        dummy_frag = "No diseases here."
        # create a safe small prompt using dynamic function
        test_candidates = ["breast cancer", "G6PD deficiency", "hemolytic anemia", "ovarian cancer", "Addison's disease"]
        dprompt = build_verify_prompt(test_candidates, dummy_frag)
        _ = client.generate(args.model, SYSTEM_VERIFY_BATCH, dprompt, GEN_OPTIONS)
        print("[WARMUP] OK")
    except Exception as e:
        print(f"[WARN] warm-up failed: {e}")

    # set generation options
    gen_opts = dict(GEN_OPTIONS)
    gen_opts["temperature"] = float(args.temperature)
    # Force deterministic verification
    verify_opts = dict(gen_opts)
    verify_opts["temperature"] = 0.0

    jobs = [(str(r.get("PMID")), r.get("Texto","")) for r in gold]
    state = start_heartbeat(total_jobs=len(jobs), interval=4)

    preds = []
    def worker(job):
        pmid, text = job
        try:
            print(f"[WORKER] Starting PMID={pmid}")
            rec = run_for_record(pmid, text, global_entities, idxs, args.model,
                                 batch_size=args.batch_size, min_occ=args.min_occurrences,
                                 rel_thresh=args.reliability_thresh,
                                 self_consistency=args.self_consistency,
                                 gen_options_base=gen_opts,
                                 double_check=args.double_check,
                                 verify_model="qwen3:4b",
                                 verify_sc_n=3,
                                 verify_options=verify_opts)
            state["lat_hist"].append(rec.get("_latency_sec", 0.0))
            print(f"[WORKER] Completed PMID={pmid} | entities={len(rec.get('Entidad', []))}")
            return rec
        except Exception as e:
            print(f"[WORKER] Error PMID={pmid}: {e}")
            return {"PMID": pmid, "Texto": text, "Entidad": [], "_error": str(e)}

    print("[INFO] Starting parallel inference...")
    if args.limit and args.limit > 0:
        print(f"[INFO] Processing first document as example...")
        first_job = jobs[0]
        first_result = worker(first_job)
        print(f"[INFO] First document completed. Final entities: {len(first_result.get('Entidad', []))}")
        # Add first result to predictions
        preds.append(first_result)
        # Remove first job from jobs list to avoid double processing
        jobs = jobs[1:]
    
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(worker, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            pmid = futures[fut]
            try:
                rec = fut.result()
                preds.append(rec)
            except Exception as e:
                print(f"[WARN] exception for PMID={pmid}: {e}")
                preds.append({"PMID": pmid, "Texto":"", "Entidad":[], "_error": str(e)})
                state["errors"] += 1
            state["done"] = i
            if i % 5 == 0:
                print(f"[PROGRESS] Completed {i}/{len(jobs)} ({100*i/len(jobs):.1f}%)")

    state["stop"] = True
    elapsed = time.time() - t0
    print(f"[INFO] Finished in {elapsed:.1f}s. Writing {len(preds)} predictions to {args.out_pred}")
    preds_sorted = sorted(preds, key=lambda x: str(x.get("PMID")))
    write_jsonl(args.out_pred, preds_sorted)

    print("[INFO] Computing strict metrics...")
    p, r, f1, npred, ngold, ntp = evaluate_strict(gold, preds_sorted)
    print("\n== Metrics (mention string-match; case-insensitive) ==")
    print(f"Predictions: {npred}")
    print(f"Gold:        {ngold}")
    print(f"TP:          {ntp}")
    print(f"Precision:   {p:.4f}")
    print(f"Recall:      {r:.4f}")
    print(f"F1:          {f1:.4f}")
    
    # Additional summary information
    if preds:
        total_chunks = sum(r.get("_debug", {}).get("n_chunks", 0) for r in preds)
        total_exact = sum(sum(r.get("_debug", {}).get("exact_hits", {}).values()) for r in preds)
        total_verified = sum(len(r.get("_debug", {}).get("seen_checks", {})) for r in preds)
        
        # Cache statistics
        cache_stats = ""
        if _llm_cache:
            cache_hits = len(_llm_cache.cache)
            cache_stats = f" | Cache entries: {cache_hits}"
        
        print(f"\n== Pipeline Summary ==")
        print(f"Total chunks processed: {total_chunks}")
        print(f"Total exact matches: {total_exact}")
        print(f"Total entities verified: {total_verified}")
        print(f"Average chunks per document: {total_chunks/len(preds):.1f}")
        print(f"Chunking config: {CHUNK_TARGET_TOKENS}t + {CHUNK_OVERLAP_TOKENS}t overlap{cache_stats}")
        
        # Performance metrics
        if preds:
            avg_latency = sum(r.get("_latency_sec", 0) for r in preds) / len(preds)
            total_time = sum(r.get("_latency_sec", 0) for r in preds)
            print(f"Average latency per document: {avg_latency:.2f}s")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Throughput: {len(preds)/total_time:.2f} docs/sec")
    
    print("[INFO] Done.")

if __name__ == "__main__":
    main()