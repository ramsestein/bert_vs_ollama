"""
Regex-based entity detection strategy for the Multi-Strategy NER system.

Provides exact surface matching using regular expressions as a baseline strategy.
"""

import re
from typing import Dict, Set
from ..core.text_processor import normalize_surface

def regex_detection(text: str, entity_aliases: Dict[str, str]) -> Set[str]:
    """Strategy 0: Regex-based exact surface matching"""
    detected = set()
    text_norm = normalize_surface(text)
    
    # Build regex pattern for all aliases
    alias_patterns = []
    for alias, entity in entity_aliases.items():
        if alias.strip():
            # Escape special characters and create word boundary pattern
            escaped_alias = re.escape(alias.strip())
            pattern = rf'\b{escaped_alias}\b'
            alias_patterns.append((pattern, entity))
    
    # Find all matches
    for pattern, entity in alias_patterns:
        matches = re.finditer(pattern, text_norm, re.IGNORECASE)
        for match in matches:
            detected.add(entity)
    
    return detected
