"""
Confidence thresholds and scoring rules for the Multi-Strategy NER system.

Defines how confidence scores are calculated and applied for entity acceptance.
"""

# Confidence thresholds optimized for 4 strategies
CONFIDENCE_THRESHOLDS = {
    "high": 0.9,      # Entity detected by 3+ strategies
    "medium": 0.7,    # Entity detected by 2+ strategies
    "low": 0.5,       # Entity detected by 1+ strategies
    "min_accept": 0.5 # Minimum threshold for acceptance
}

def update_confidence_thresholds(new_min_accept: float):
    """Update the minimum acceptance threshold."""
    if 0.0 <= new_min_accept <= 1.0:
        CONFIDENCE_THRESHOLDS["min_accept"] = new_min_accept
        return True
    return False

def get_confidence_level(confidence_score: float) -> str:
    """Get the confidence level string based on score."""
    if confidence_score >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif confidence_score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    elif confidence_score >= CONFIDENCE_THRESHOLDS["low"]:
        return "low"
    else:
        return "below_threshold"

def is_entity_accepted(confidence_score: float) -> bool:
    """Check if an entity should be accepted based on confidence."""
    return confidence_score >= CONFIDENCE_THRESHOLDS["min_accept"]

def get_confidence_rules():
    """Get the current confidence scoring rules."""
    return {
        "regex_multiplier": 1.5,           # Regex detection boost
        "multi_strategy_bonus": 0.2,       # Bonus per additional strategy
        "llm_only_penalty": 0.8,           # Penalty for LLM-only detection
        "max_confidence": 1.0,             # Maximum possible confidence
        "min_confidence": 0.0              # Minimum possible confidence
    }
