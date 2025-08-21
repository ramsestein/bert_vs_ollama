"""
LLM Client with caching for the Multi-Strategy NER system.

Provides OllamaClient for API communication and LLMCache for response caching.
"""

import json
import threading
import http.client
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from ..config.settings import (
    DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT,
    CACHE_MAX_SIZE, CACHE_TTL_HOURS
)

class LLMCache:
    """Thread-safe cache for LLM responses with TTL."""
    
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl_hours: int = CACHE_TTL_HOURS):
        self.cache = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.lock = threading.Lock()
    
    def _generate_key(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """Generate a unique cache key for a request."""
        content = f"{model}:{system_prompt}:{user_prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """Get a cached response if it exists and is not expired."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if datetime.now() < entry['expiry']:
                    return entry['response']
                else:
                    del self.cache[key]
        return None
    
    def put(self, key: str, response: str):
        """Store a response in the cache with TTL."""
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
    """HTTP client for communicating with Ollama API."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: int = DEFAULT_TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn = None
        self.lock = threading.Lock()
    
    def _get_connection(self):
        """Get or create HTTP connection."""
        if self.conn is None:
            print(f"      [DEBUG] Creating new connection to {self.host}:{self.port}")
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self.conn
    
    def _request(self, path: str, body: dict) -> str:
        """Make HTTP request to Ollama API."""
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
        """Generate response from Ollama model with caching."""
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

def get_thread_client() -> OllamaClient:
    """Get thread-local Ollama client instance."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OllamaClient()
        _thread_local.client = client
    return client
