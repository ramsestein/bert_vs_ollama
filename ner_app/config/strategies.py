"""
Strategy configurations for the Multi-Strategy NER system.

Defines the 4 main strategies with their specific parameters for
chunking, temperature, and model selection.
"""

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

def get_strategy_by_name(name: str):
    """Get a strategy configuration by name."""
    for strategy in ALL_STRATEGIES:
        if strategy["name"] == name:
            return strategy
    return None

def get_strategies_by_model(model: str):
    """Get all strategies that use a specific model."""
    return [s for s in ALL_STRATEGIES if s["model"] == model]

def validate_strategy(strategy: dict) -> bool:
    """Validate that a strategy has all required parameters."""
    required_keys = ["name", "model", "chunk_target", "chunk_overlap", 
                    "chunk_min", "chunk_max", "temperature", "weight"]
    
    for key in required_keys:
        if key not in strategy:
            return False
    
    # Validate numeric parameters
    if strategy["chunk_target"] <= 0:
        return False
    if strategy["chunk_overlap"] < 0:
        return False
    if strategy["chunk_min"] <= 0:
        return False
    if strategy["chunk_max"] <= strategy["chunk_min"]:
        return False
    if not (0.0 <= strategy["temperature"] <= 1.0):
        return False
    if strategy["weight"] <= 0:
        return False
    
    return True
