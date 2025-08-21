"""
Command-line interface parser for the Multi-Strategy NER system.

Handles argument parsing, validation, and strategy configuration.
"""

import argparse
from typing import List, Dict
from ..config.strategies import ALL_STRATEGIES, get_strategy_by_name
from ..config.thresholds import update_confidence_thresholds

def parse_arguments():
    """Parse command-line arguments for the NER system."""
    parser = argparse.ArgumentParser(description="Multi-Strategy NER System - Refactored Version")
    
    # Required arguments
    parser.add_argument("--input_jsonl", required=True, 
                       help="Input JSONL file with text and entities to search")
    parser.add_argument("--benchmark_jsonl", required=False, default=None,
                       help="Benchmark JSONL file for evaluation (optional - without it, no metrics are calculated)")
    
    # Optional arguments
    parser.add_argument("--out_pred", default="results_multi_strategy.jsonl", 
                       help="Output file")
    parser.add_argument("--limit", type=int, default=0, 
                       help="Limit number of documents (0 = all)")
    parser.add_argument("--strategies", nargs="+", default=["all"], 
                       help="Strategies to use (or 'all' for all strategies)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Minimum confidence threshold for acceptance")
    
    # Strategy overrides
    parser.add_argument("--s1_target", type=int, default=None, 
                       help="Override STRATEGY_1 chunk_target")
    parser.add_argument("--s1_overlap", type=int, default=None, 
                       help="Override STRATEGY_1 chunk_overlap")
    parser.add_argument("--s1_min", type=int, default=None, 
                       help="Override STRATEGY_1 chunk_min")
    parser.add_argument("--s1_max", type=int, default=None, 
                       help="Override STRATEGY_1 chunk_max")
    parser.add_argument("--s1_temp", type=float, default=None, 
                       help="Override STRATEGY_1 temperature")
    parser.add_argument("--s2_target", type=int, default=None, 
                       help="Override STRATEGY_2 chunk_target")
    parser.add_argument("--s2_overlap", type=int, default=None, 
                       help="Override STRATEGY_2 chunk_overlap")
    parser.add_argument("--s2_min", type=int, default=None, 
                       help="Override STRATEGY_2 chunk_min")
    parser.add_argument("--s2_max", type=int, default=None, 
                       help="Override STRATEGY_2 chunk_max")
    parser.add_argument("--s2_temp", type=float, default=None, 
                       help="Override STRATEGY_2 temperature")
    
    return parser.parse_args()

def configure_strategies(args) -> List[Dict]:
    """Configure strategies based on command-line arguments."""
    # Select strategies
    if "all" in args.strategies:
        strategies = ALL_STRATEGIES.copy()
    else:
        strategies = [s for s in ALL_STRATEGIES if s["name"] in args.strategies]
    
    # Apply overrides
    name_to_strategy = {s["name"]: s for s in strategies}
    
    # STRATEGY_1 overrides
    if len(strategies) > 0:
        s1 = strategies[0]
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

    # STRATEGY_2 overrides
    if len(strategies) > 1:
        s2 = strategies[1]
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
    
    return strategies

def print_configuration(args, strategies: List[Dict]):
    """Print the current configuration."""
    print(f"[CONFIG] Using {len(strategies)} strategies:")
    for s in strategies:
        print(f"  - {s['name']}: {s['model']} | chunks={s['chunk_target']}t | temp={s['temperature']}")
    
    print(f"[CONFIG] Confidence threshold: {args.confidence_threshold}")
    print(f"[CONFIG] Input file: {args.input_jsonl}")
    if args.benchmark_jsonl:
        print(f"[CONFIG] Benchmark file: {args.benchmark_jsonl}")
    else:
        print(f"[CONFIG] No benchmark file provided - metrics will not be calculated")

def validate_arguments(args) -> bool:
    """Validate command-line arguments."""
    # Check if input file exists
    import os
    if not os.path.exists(args.input_jsonl):
        print(f"[ERROR] Input file not found: {args.input_jsonl}")
        return False
    
    # Check if benchmark file exists (only if provided)
    if args.benchmark_jsonl and not os.path.exists(args.benchmark_jsonl):
        print(f"[ERROR] Benchmark file not found: {args.benchmark_jsonl}")
        return False
    
    # Validate confidence threshold
    if not (0.0 <= args.confidence_threshold <= 1.0):
        print(f"[ERROR] Confidence threshold must be between 0.0 and 1.0")
        return False
    
    # Validate limit
    if args.limit < 0:
        print(f"[ERROR] Limit must be non-negative")
        return False
    
    return True
