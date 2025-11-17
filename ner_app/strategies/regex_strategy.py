"""
Regex-based entity detection strategy for the Multi-Strategy NER system.

Provides exact surface matching using regular expressions as a baseline strategy.
"""

import re
from typing import Dict, Set
from ..core.text_processor import normalize_surface

def regex_detection(text: str, entity_aliases: Dict[str, str]) -> Set[str]:
    """Strategy 0: Regex-based exact surface matching with Spanish support"""
    detected = set()
    text_norm = normalize_surface(text)
    text_no_accents = normalize_surface(text, remove_accents=True)
    
    # Build regex pattern for all aliases
    alias_patterns = []
    for alias, entity in entity_aliases.items():
        if alias.strip():
            # Escape special characters and create word boundary pattern
            escaped_alias = re.escape(alias.strip())
            pattern = rf'\b{escaped_alias}\b'
            alias_patterns.append((pattern, entity, alias.strip()))
    
    # Find all matches (with and without accents for Spanish)
    for pattern, entity, original_alias in alias_patterns:
        # Try exact match first
        matches = re.finditer(pattern, text_norm, re.IGNORECASE)
        for match in matches:
            detected.add(entity)
        
        # Try accent-insensitive match for Spanish
        pattern_no_accents = re.escape(normalize_surface(original_alias, remove_accents=True))
        pattern_no_accents = rf'\b{pattern_no_accents}\b'
        matches_no_accents = re.finditer(pattern_no_accents, text_no_accents, re.IGNORECASE)
        for match in matches_no_accents:
            detected.add(entity)
    
    return detected
