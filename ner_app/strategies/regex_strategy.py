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
    # Use accent-insensitive matching as a single-pass:
    # normalize both text and aliases to a no-accent form and run literal (escaped) regex
    text_no_accents = normalize_surface(text, remove_accents=True)

    # Build regex pattern for all aliases using their no-accent form
    alias_patterns = []
    for alias, entity in entity_aliases.items():
        if alias and alias.strip():
            alias_no_acc = normalize_surface(alias.strip(), remove_accents=True)
            escaped_alias = re.escape(alias_no_acc)
            pattern = rf'\b{escaped_alias}\b'
            alias_patterns.append((pattern, entity, alias.strip(), alias_no_acc))

    # Find all matches over the no-accent text and map back to canonical entity
    for pattern, entity, original_alias, alias_no_acc in alias_patterns:
        matches = re.finditer(pattern, text_no_accents, re.IGNORECASE)
        for match in matches:
            detected.add(entity)
    
    return detected
