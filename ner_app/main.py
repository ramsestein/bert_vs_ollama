#!/usr/bin/env python3
"""
Multi-Strategy NER System - Refactored Version

Main entry point for the NER application with modular architecture.
"""

import json
import time
import gc
from collections import defaultdict
from typing import List, Dict, Any

from .core.file_manager import ensure_temp_dir, cleanup_temp_files
from .strategies.multi_strategy import run_multi_strategy_detection
from .utils.cli_parser import parse_arguments, configure_strategies, print_configuration, validate_arguments
from .config.thresholds import update_confidence_thresholds

def process_document(pmid: str, text: str, entity_candidates: List[str], 
                    strategies: List[Dict]) -> Dict[str, Any]:
    """Process a single document with multi-strategy detection."""
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
            "confidence_thresholds": results.get("confidence_thresholds", {}),
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

def load_documents(input_file: str, limit: int = 0) -> List[Dict[str, Any]]:
    """Load documents from input file with optional limit."""
    documents = []
    doc_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            
            if limit > 0 and doc_count >= limit:
                break
            
            try:
                doc = json.loads(line)
                pmid = str(doc.get("PMID", f"doc_{line_num}"))
                text = doc.get("Texto", "")
                
                if not text.strip():
                    continue
                
                # Extract entity candidates from this document
                entity_candidates = []
                entities = doc.get("Entidad", [])
                for ent in entities:
                    if isinstance(ent, dict) and "texto" in ent:
                        entity_candidates.append(ent["texto"])
                
                documents.append({
                    "pmid": pmid,
                    "text": text,
                    "entity_candidates": entity_candidates,
                    "line_num": line_num + 1
                })
                
                doc_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to process line {line_num+1}: {e}")
                continue
    
    return documents

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to output file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"[INFO] Results saved to {output_file}")

def print_summary(results: List[Dict[str, Any]], strategies: List[Dict]):
    """Print processing summary."""
    print(f"\n[SUMMARY] Processed {len(results)} documents")
    
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
    
    # Confidence distribution
    confidence_levels = defaultdict(int)
    for r in results:
        for ent in r["Entidad"]:
            conf = ent["confidence"]
            if conf >= 0.9:
                confidence_levels["high"] += 1
            elif conf >= 0.7:
                confidence_levels["medium"] += 1
            elif conf >= 0.5:
                confidence_levels["low"] += 1
            else:
                confidence_levels["below_threshold"] += 1
    
    print(f"\n[CONFIDENCE DISTRIBUTION]")
    for level, count in confidence_levels.items():
        print(f"  {level}: {count} entities")

def main():
    """Main entry point for the NER application."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        if not validate_arguments(args):
            print("[ERROR] Invalid arguments provided")
            return 1
        
        # Configure strategies and thresholds
        strategies = configure_strategies(args)
        update_confidence_thresholds(args.confidence_threshold)
        
        # Print configuration
        print_configuration(args, strategies)
        
        # Setup temporary directory
        ensure_temp_dir()
        
        # Load documents
        print(f"[INFO] Loading documents from {args.input_jsonl}...")
        documents = load_documents(args.input_jsonl, args.limit)
        print(f"[INFO] Loaded {len(documents)} documents")
        
        if not documents:
            print("[ERROR] No valid documents found")
            return 1
        
        # Process documents
        print(f"[INFO] Processing documents with minimal memory usage...")
        results = []
        
        for i, doc in enumerate(documents, 1):
            print(f"\n[PROGRESS] {i}/{len(documents)} (line {doc['line_num']})")
            print(f"[PROCESSING] PMID={doc['pmid']} | text_length={len(doc['text'])}")
            print(f"[INFO] Found {len(doc['entity_candidates'])} entity candidates in this document")
            
            # Process this document
            result = process_document(doc['pmid'], doc['text'], doc['entity_candidates'], strategies)
            results.append(result)
            
            print(f"[COMPLETED] Document {i} processed successfully")
            
            # Force garbage collection to free memory
            gc.collect()
        
        # Save results
        save_results(results, args.out_pred)
        
        # Print summary
        print_summary(results, strategies)
        
        # Clean up temporary files
        cleanup_temp_files()
        
        print(f"\n[SUCCESS] Processing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Processing stopped by user")
        cleanup_temp_files()
        return 130
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        cleanup_temp_files()
        return 1

if __name__ == "__main__":
    exit(main())
