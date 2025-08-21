"""
NER Multi-Strategy Application Package

A refactored, modular implementation of the multi-strategy NER system
for biomedical entity recognition using local language models.
"""

__version__ = "2.0.0"
__author__ = "NER Development Team"

from .core.llm_client import OllamaClient, LLMCache
from .strategies.multi_strategy import run_multi_strategy_detection
from .config.strategies import ALL_STRATEGIES
from .config.thresholds import CONFIDENCE_THRESHOLDS

__all__ = [
    'OllamaClient',
    'LLMCache', 
    'run_multi_strategy_detection',
    'ALL_STRATEGIES',
    'CONFIDENCE_THRESHOLDS'
]
