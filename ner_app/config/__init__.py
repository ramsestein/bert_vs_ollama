"""
Configuration module for NER Multi-Strategy system.

Contains strategy definitions, confidence thresholds, and system settings.
"""

from .strategies import ALL_STRATEGIES, STRATEGY_1, STRATEGY_2, STRATEGY_3, STRATEGY_4
from .thresholds import CONFIDENCE_THRESHOLDS
from .settings import TEMP_DIR, CHUNK_FILE_PREFIX, STRATEGY_FILE_PREFIX

__all__ = [
    'ALL_STRATEGIES',
    'STRATEGY_1',
    'STRATEGY_2', 
    'STRATEGY_3',
    'STRATEGY_4',
    'CONFIDENCE_THRESHOLDS',
    'TEMP_DIR',
    'CHUNK_FILE_PREFIX',
    'STRATEGY_FILE_PREFIX'
]
