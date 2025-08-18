#!/usr/bin/env python3
"""
Optimized NER with DSPy prompt optimization
Uses DSPy to automatically optimize prompts for better entity detection
"""

import json
import re
import time
import argparse
import threading
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# DSPy imports
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from dspy.primitives import Example

# ----------------------------
# DSPy Signature for NER
# ----------------------------
class NERSignature(dspy.Signature):
    """Signature for Named Entity Recognition with optimized prompts"""
    
    text = dspy.InputField(desc="Biomedical text to analyze")
    entities = dspy.OutputField(desc="List of disease entities found in the text", 
                               prefix="Entities: ")

# ----------------------------
# DSPy Module for NER
# ----------------------------
class OptimizedNER(dspy.Module):
    """Optimized NER module using DSPy"""
    
    def __init__(self):
        super().__init__()
        self.ner = dspy.ChainOfThought("text -> entities")
    
    def forward(self, text):
        return self.ner(text=text)

# ----------------------------
# Ollama Client
# ----------------------------
class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 120
    
    def _request(self, endpoint: str, body: dict) -> str:
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=body)
        response.raise_for_status()
        return response.text
    
    def generate(self, model: str, system_prompt: str, user_prompt: str, options: dict) -> str:
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
                return data.get("response", "")
            except json.JSONDecodeError:
                return raw
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

# ----------------------------
# Text Processing
# ----------------------------
def normalize_surface(text: str) -> str:
    """Normalize text surface"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    """Simple tokenization"""
    if not text:
        return []
    return re.findall(r'\S+', text)

def sentence_chunks(text: str, target_tokens=60, overlap_tokens=30) -> List[str]:
    """Create sentence-aware chunks with overlap"""
    text = normalize_surface(text)
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return []
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenize(sentence))
        
        if current_tokens + sentence_tokens <= target_tokens:
            current_chunk += sentence + " "
            current_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if chunks and overlap_tokens > 0:
                # Get last few sentences for overlap
                overlap_text = ""
                overlap_count = 0
                for s in reversed(chunks[-1].split('.')):
                    if overlap_count >= overlap_tokens:
                        break
                    overlap_text = s + '. ' + overlap_text
                    overlap_count += len(tokenize(s))
                    if overlap_count >= overlap_tokens:
                        break
                current_chunk = overlap_text + sentence + " "
                current_tokens = len(tokenize(current_chunk))
            else:
                current_chunk = sentence + " "
                current_tokens = sentence_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# ----------------------------
# DSPy Prompt Optimization
# ----------------------------
def create_training_examples() -> List[Example]:
    """Create training examples for DSPy optimization"""
    examples = [
        Example(
            text="BRCA1 is secreted and exhibits properties of a granin. Germline mutations in BRCA1 are responsible for most cases of inherited breast and ovarian cancer.",
            entities="breast cancer, ovarian cancer, inherited breast cancer"
        ),
        Example(
            text="Myotonic dystrophy (DM) is associated with a (CTG) n trinucleotide repeat expansion in the 3-untranslated region of a protein kinase-encoding gene.",
            entities="Myotonic dystrophy, DM"
        ),
        Example(
            text="Type II human complement C2 deficiency is characterized by a selective block in C2 secretion.",
            entities="Type II human complement C2 deficiency, Type II C2 deficiency"
        ),
        Example(
            text="Wiskott-Aldrich syndrome protein, a novel effector for the GTPase CDC42Hs, is implicated in actin polymerization.",
            entities="Wiskott-Aldrich syndrome, WAS"
        ),
        Example(
            text="X-linked adrenoleukodystrophy (ALD) is a genetic disease associated with demyelination of the central nervous system.",
            entities="X-linked adrenoleukodystrophy, ALD"
        )
    ]
    return examples

def optimize_prompts_with_dspy(model_name: str, client: OllamaClient) -> OptimizedNER:
    """Use DSPy to optimize prompts for NER"""
    
    # Configure DSPy to use Ollama
    dspy.settings.configure(lm=OllamaClient())
    
    # Create training examples
    examples = create_training_examples()
    
    # Create the NER module
    ner_module = OptimizedNER()
    
    # Use BootstrapFewShot to optimize prompts
    teleprompter = BootstrapFewShot(metric=None, max_bootstrapped_demos=3, max_labeled_demos=5)
    
    # Compile the module with examples
    compiled_ner = teleprompter.compile(ner_module, trainset=examples)
    
    return compiled_ner

# ----------------------------
# Main NER Processing
# ----------------------------
def process_chunk_with_dspy(chunk: str, ner_module: OptimizedNER, model: str, 
                           client: OllamaClient, options: dict) -> List[str]:
    """Process a chunk using optimized DSPy prompts"""
    
    try:
        # Use DSPy module to get entities
        result = ner_module(text=chunk)
        entities_text = result.entities
        
        # Parse entities from DSPy output
        if entities_text and entities_text != "None":
            # Extract entities from the response
            entities = [e.strip() for e in entities_text.split(',') if e.strip()]
            return entities
        else:
            # Fallback to direct LLM call if DSPy fails
            system_prompt = "You are a biomedical entity extractor. Extract disease names from the text."
            user_prompt = f"Extract disease entities from this text: {chunk}"
            
            response = client.generate(model, system_prompt, user_prompt, options)
            
            # Simple parsing of response
            entities = []
            lines = response.split('\n')
            for line in lines:
                if ':' in line or ',' in line:
                    parts = re.split(r'[:,]', line)
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 2:
                            entities.append(part)
            
            return entities[:10]  # Limit to 10 entities
            
    except Exception as e:
        print(f"Error in DSPy processing: {e}")
        # Fallback to simple extraction
        return []

def run_for_record_dspy(pmid: str, text: str, ner_module: OptimizedNER, 
                        model: str, client: OllamaClient, options: dict) -> Dict[str, Any]:
    """Process a single record using DSPy-optimized prompts"""
    
    t0 = time.time()
    text = normalize_surface(text)
    
    # Create chunks
    chunks = sentence_chunks(text, target_tokens=60, overlap_tokens=30)
    n_chunks = len(chunks)
    
    all_entities = []
    entity_counts = Counter()
    
    # Process each chunk with DSPy
    for chunk_idx, chunk in enumerate(chunks):
        try:
            entities = process_chunk_with_dspy(chunk, ner_module, model, client, options)
            
            for entity in entities:
                entity_counts[entity] += 1
                all_entities.append(entity)
                
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {e}")
            continue
    
    # Filter entities by frequency
    final_entities = []
    for entity, count in entity_counts.most_common(20):
        if count >= 1:  # Accept entities that appear at least once
            final_entities.append({
                "texto": entity,
                "tipo": "SpecificDisease"  # Default type
            })
    
    latency = time.time() - t0
    
    return {
        "PMID": pmid,
        "Texto": text,
        "Entidad": final_entities,
        "_debug": {
            "n_chunks": n_chunks,
            "total_entities_found": len(all_entities),
            "unique_entities": len(entity_counts),
            "entity_counts": dict(entity_counts.most_common(10))
        },
        "_latency_sec": latency
    }

# ----------------------------
# Main Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="DSPy-optimized NER with prompt optimization")
    parser.add_argument("--develop_jsonl", required=True)
    parser.add_argument("--out_pred", default="results_dspy_optimized.jsonl")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    
    args = parser.parse_args()
    
    print(f"[INFO] Starting DSPy-optimized NER")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Workers: {args.n_workers}")
    
    # Initialize Ollama client
    client = OllamaClient()
    
    # Generate options
    options = {
        "temperature": args.temperature,
        "top_p": 0.9,
        "num_predict": 512,
        "num_ctx": 1536,
        "cache_prompt": True,
    }
    
    # Optimize prompts with DSPy
    print("[INFO] Optimizing prompts with DSPy...")
    ner_module = optimize_prompts_with_dspy(args.model, client)
    print("[INFO] Prompt optimization complete!")
    
    # Load data
    with open(args.develop_jsonl, 'r') as f:
        all_records = [json.loads(line) for line in f]
    
    gold = all_records[:args.limit] if args.limit and args.limit > 0 else all_records
    print(f"[INFO] Loaded {len(gold)} records to process")
    
    # Process records
    results = []
    
    if args.n_workers == 1:
        # Sequential processing
        for i, record in enumerate(gold):
            print(f"[INFO] Processing record {i+1}/{len(gold)}")
            result = run_for_record_dspy(
                record["PMID"], record["Texto"], ner_module, 
                args.model, client, options
            )
            results.append(result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
            futures = []
            for record in gold:
                future = executor.submit(
                    run_for_record_dspy,
                    record["PMID"], record["Texto"], ner_module,
                    args.model, client, options
                )
                futures.append(future)
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[INFO] Completed record {i+1}/{len(gold)}")
                except Exception as e:
                    print(f"[ERROR] Failed to process record: {e}")
    
    # Save results
    with open(args.out_pred, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"[INFO] Results saved to {args.out_pred}")
    print(f"[INFO] Total records processed: {len(results)}")
    
    # Summary statistics
    total_entities = sum(len(r['Entidad']) for r in results)
    total_chunks = sum(r.get('_debug', {}).get('n_chunks', 0) for r in results)
    avg_latency = sum(r.get('_latency_sec', 0) for r in results) / len(results)
    
    print(f"\n=== DSPy Optimization Summary ===")
    print(f"Total entities detected: {total_entities}")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Average latency per document: {avg_latency:.2f} seconds")
    print(f"Average entities per document: {total_entities/len(results):.2f}")

if __name__ == "__main__":
    main()
