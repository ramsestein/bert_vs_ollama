"""
Multi-strategy orchestrator for the NER system.

Combines results from multiple detection strategies and applies confidence scoring.
"""

import os
from collections import defaultdict
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..strategies.regex_strategy import regex_detection
from ..strategies.llm_strategy import llm_detection_strategy_file
from ..core.file_manager import load_strategy_results, cleanup_strategy_files
from ..config.thresholds import CONFIDENCE_THRESHOLDS, get_confidence_rules
from ..config.settings import MAX_WORKERS, get_system_prompts

def run_multi_strategy_detection(text: str, entity_candidates: List[str], 
                                strategies: List[Dict], doc_id: str) -> Dict[str, Any]:
    """Run all detection strategies in parallel and combine results using files for memory efficiency"""
    
    # Strategy 0: Regex detection (baseline) - runs instantly
    print(f"  [STRATEGY] Running regex detection...")
    regex_entities = regex_detection(text, {c: c for c in entity_candidates})
    print(f"    [REGEX] Found {len(regex_entities)} entities")
    
    # System prompt for LLM strategies
    system_prompts = get_system_prompts()
    if any(s["model"] == "qwen2.5:3b" for s in strategies):
        system_prompt = system_prompts["qwen2.5:3b"]
    else:
        system_prompt = system_prompts["default"]
    
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
    print(f"  [PARALLEL] Launching {len(strategies)} strategies for maximum entity recovery...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all strategies simultaneously
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
    confidence_rules = get_confidence_rules()
    
    for entity in entity_confidence:
        # Rule 1: Regex detection gives maximum confidence
        if "regex" in entity_strategies[entity]:
            entity_confidence[entity] = min(1.0, entity_confidence[entity] * confidence_rules["regex_multiplier"])
        
        # Rule 2: Multiple strategy detection increases confidence
        strategy_count = len(entity_strategies[entity])
        if strategy_count > 1:
            entity_confidence[entity] = min(1.0, entity_confidence[entity] * (1.0 + confidence_rules["multi_strategy_bonus"] * (strategy_count - 1)))
        
        # Rule 3: LLM-only detection gets penalty
        if "regex" not in entity_strategies[entity]:
            entity_confidence[entity] *= confidence_rules["llm_only_penalty"]
        
        # Rule 4: Normalize to [0, 1] range
        entity_confidence[entity] = max(confidence_rules["min_confidence"], 
                                     min(confidence_rules["max_confidence"], entity_confidence[entity]))
    
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
    cleanup_strategy_files(strategy_filepaths)
    
    return {
        "all_detections": {k: list(v) for k, v in all_detections.items()},
        "entity_confidence": entity_confidence,
        "entity_strategies": dict(entity_strategies),
        "accepted_entities": accepted_entities,
        "total_detected": len(entity_confidence),
        "total_accepted": len(accepted_entities)
    }
