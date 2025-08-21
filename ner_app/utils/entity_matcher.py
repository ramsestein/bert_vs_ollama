"""
Entity matching utilities for the Multi-Strategy NER system.

Provides tools for matching detected entities with candidate entities.
"""

from typing import List, Dict, Set, Any
from ..core.text_processor import _fuzzy_match

class EntityMatcher:
    """Handles entity matching and validation."""
    
    def __init__(self, fuzzy_threshold: float = 0.8):
        self.fuzzy_threshold = fuzzy_threshold
    
    def match_entities(self, detected_entities: List[str], 
                      candidate_entities: List[str]) -> Dict[str, List[str]]:
        """Match detected entities with candidate entities using multiple strategies."""
        matches = {
            "exact": [],
            "partial": [],
            "fuzzy": [],
            "unmatched": []
        }
        
        for detected in detected_entities:
            if not detected or not isinstance(detected, str):
                continue
            
            detected_lower = detected.lower().strip()
            best_match = None
            match_type = None
            
            for candidate in candidate_entities:
                candidate_lower = candidate.lower().strip()
                
                # Exact match
                if candidate_lower == detected_lower:
                    best_match = candidate
                    match_type = "exact"
                    break
                
                # Partial match (one contains the other)
                elif detected_lower in candidate_lower or candidate_lower in detected_lower:
                    if not best_match or match_type != "exact":
                        best_match = candidate
                        match_type = "partial"
                
                # Fuzzy match
                elif _fuzzy_match(detected_lower, candidate_lower, self.fuzzy_threshold):
                    if not best_match or match_type not in ["exact", "partial"]:
                        best_match = candidate
                        match_type = "fuzzy"
            
            # Categorize the match
            if best_match:
                if match_type == "exact":
                    matches["exact"].append((detected, best_match))
                elif match_type == "partial":
                    matches["partial"].append((detected, best_match))
                elif match_type == "fuzzy":
                    matches["fuzzy"].append((detected, best_match))
            else:
                matches["unmatched"].append(detected)
        
        return matches
    
    def validate_entity_candidates(self, candidates: List[str]) -> Dict[str, Any]:
        """Validate entity candidates for processing."""
        validation = {
            "valid": [],
            "invalid": [],
            "duplicates": [],
            "total": len(candidates)
        }
        
        seen = set()
        
        for candidate in candidates:
            if not candidate or not isinstance(candidate, str):
                validation["invalid"].append(candidate)
                continue
            
            candidate_clean = candidate.strip()
            if not candidate_clean:
                validation["invalid"].append(candidate)
                continue
            
            candidate_lower = candidate_clean.lower()
            if candidate_lower in seen:
                validation["duplicates"].append(candidate)
            else:
                seen.add(candidate_lower)
                validation["valid"].append(candidate_clean)
        
        return validation
    
    def get_matching_statistics(self, matches: Dict[str, List]) -> Dict[str, Any]:
        """Get statistics about entity matching."""
        total_detected = sum(len(match_list) for match_list in matches.values())
        
        return {
            "total_detected": total_detected,
            "exact_matches": len(matches["exact"]),
            "partial_matches": len(matches["partial"]),
            "fuzzy_matches": len(matches["fuzzy"]),
            "unmatched": len(matches["unmatched"]),
            "match_rate": (total_detected - len(matches["unmatched"])) / total_detected if total_detected > 0 else 0.0
        }
    
    def suggest_candidate_improvements(self, matches: Dict[str, List], 
                                    candidates: List[str]) -> List[str]:
        """Suggest improvements to candidate entities based on matching results."""
        suggestions = []
        
        # Check for unmatched detected entities that might be valid
        for unmatched in matches["unmatched"]:
            suggestions.append(f"Consider adding '{unmatched}' as a candidate entity")
        
        # Check for potential typos or variations
        for detected, candidate in matches["fuzzy"]:
            if detected.lower() != candidate.lower():
                suggestions.append(f"'{detected}' and '{candidate}' are similar - consider standardizing")
        
        # Check for duplicates in candidates
        seen_candidates = set()
        for candidate in candidates:
            candidate_lower = candidate.lower().strip()
            if candidate_lower in seen_candidates:
                suggestions.append(f"Duplicate candidate: '{candidate}'")
            seen_candidates.add(candidate_lower)
        
        return suggestions
