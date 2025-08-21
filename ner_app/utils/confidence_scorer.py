"""
Confidence scoring utilities for the Multi-Strategy NER system.

Provides tools for calculating and applying confidence scores to detected entities.
"""

from typing import Dict, List, Any
from ..config.thresholds import CONFIDENCE_THRESHOLDS, get_confidence_rules

class ConfidenceScorer:
    """Handles confidence scoring for detected entities."""
    
    def __init__(self):
        self.confidence_rules = get_confidence_rules()
    
    def calculate_confidence(self, entity: str, strategies: List[str], 
                           strategy_weights: Dict[str, float]) -> float:
        """Calculate confidence score for an entity based on detection strategies."""
        base_confidence = 0.0
        
        # Add base confidence from each strategy
        for strategy in strategies:
            weight = strategy_weights.get(strategy, 1.0)
            base_confidence += weight
        
        # Apply confidence rules
        confidence = base_confidence
        
        # Rule 1: Regex detection gives maximum confidence
        if "regex" in strategies:
            confidence *= self.confidence_rules["regex_multiplier"]
        
        # Rule 2: Multiple strategy detection increases confidence
        strategy_count = len(strategies)
        if strategy_count > 1:
            confidence *= (1.0 + self.confidence_rules["multi_strategy_bonus"] * (strategy_count - 1))
        
        # Rule 3: LLM-only detection gets penalty
        if "regex" not in strategies:
            confidence *= self.confidence_rules["llm_only_penalty"]
        
        # Rule 4: Normalize to [0, 1] range
        confidence = max(self.confidence_rules["min_confidence"], 
                        min(self.confidence_rules["max_confidence"], confidence))
        
        return confidence
    
    def filter_entities_by_confidence(self, entity_confidence: Dict[str, float]) -> List[Dict[str, Any]]:
        """Filter entities based on confidence thresholds."""
        accepted_entities = []
        
        for entity, confidence in entity_confidence.items():
            if confidence >= CONFIDENCE_THRESHOLDS["min_accept"]:
                accepted_entities.append({
                    "entity": entity,
                    "confidence": confidence,
                    "confidence_level": self.get_confidence_level(confidence)
                })
        
        # Sort by confidence (highest first)
        accepted_entities.sort(key=lambda x: x["confidence"], reverse=True)
        
        return accepted_entities
    
    def get_confidence_level(self, confidence_score: float) -> str:
        """Get the confidence level string based on score."""
        if confidence_score >= CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif confidence_score >= CONFIDENCE_THRESHOLDS["medium"]:
            return "medium"
        elif confidence_score >= CONFIDENCE_THRESHOLDS["low"]:
            return "low"
        else:
            return "below_threshold"
    
    def get_confidence_summary(self, entity_confidence: Dict[str, float]) -> Dict[str, Any]:
        """Get summary statistics for confidence scores."""
        if not entity_confidence:
            return {
                "total_entities": 0,
                "average_confidence": 0.0,
                "confidence_range": [0.0, 0.0],
                "by_level": {"high": 0, "medium": 0, "low": 0, "below_threshold": 0}
            }
        
        confidences = list(entity_confidence.values())
        total_entities = len(confidences)
        average_confidence = sum(confidences) / total_entities
        confidence_range = [min(confidences), max(confidences)]
        
        # Count by confidence level
        by_level = {"high": 0, "medium": 0, "low": 0, "below_threshold": 0}
        for confidence in confidences:
            level = self.get_confidence_level(confidence)
            by_level[level] += 1
        
        return {
            "total_entities": total_entities,
            "average_confidence": average_confidence,
            "confidence_range": confidence_range,
            "by_level": by_level
        }
