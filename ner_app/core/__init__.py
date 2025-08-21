"""
Core functionality module for NER Multi-Strategy system.

Contains LLM client, text processing, and file management components.
"""

from .llm_client import OllamaClient, LLMCache
from .text_processor import normalize_surface, tokenize, sentence_chunks, _fuzzy_match
from .file_manager import (
    ensure_temp_dir, cleanup_temp_files, save_chunks_to_file,
    load_chunks_from_file, save_strategy_results, load_strategy_results
)

__all__ = [
    'OllamaClient',
    'LLMCache',
    'normalize_surface',
    'tokenize', 
    'sentence_chunks',
    '_fuzzy_match',
    'ensure_temp_dir',
    'cleanup_temp_files',
    'save_chunks_to_file',
    'load_chunks_from_file',
    'save_strategy_results',
    'load_strategy_results'
]
