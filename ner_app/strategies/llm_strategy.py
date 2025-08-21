"""
LLM-based entity detection strategy for the Multi-Strategy NER system.

Provides intelligent entity detection using language models with retry logic and fallback mechanisms.
"""

import json
import re
import time
from typing import Dict, List, Set
from ..core.llm_client import get_thread_client
from ..core.file_manager import save_strategy_results, text_to_chunks_file
from ..core.text_processor import _fuzzy_match
from ..config.settings import MAX_LLM_RETRIES, RETRY_DELAY_SECONDS, get_system_prompts

def llm_detection_strategy_file(text: str, strategy: Dict, entity_candidates: List[str], 
                               system_prompt: str, doc_id: str) -> str:
    """Run LLM-based detection for a specific strategy using files for memory efficiency"""
    try:
        print(f"      [DEBUG] Starting strategy: {strategy['name']}")
        print(f"      [DEBUG] Text length: {len(text)} chars")
        print(f"      [DEBUG] Entity candidates: {len(entity_candidates)}")
        
        # Validate strategy parameters
        if strategy["chunk_target"] <= 0 or strategy["chunk_overlap"] < 0:
            print(f"      [ERROR] Invalid strategy parameters: target={strategy['chunk_target']}, overlap={strategy['chunk_overlap']}")
            return save_strategy_results(doc_id, strategy['name'], set())
        
        # Step 1: Create chunks and save to file
        print(f"      [DEBUG] Creating chunks for {strategy['name']}...")
        chunks_filepath = text_to_chunks_file(text, strategy, doc_id)
        
        if not chunks_filepath or not chunks_filepath.strip():
            print(f"      [ERROR] Chunks file not created for {strategy['name']}")
            return save_strategy_results(doc_id, strategy['name'], set())
        
        # Step 2: Load chunks one by one and process
        detected_entities = set()
        chunk_count = 0
        
        print(f"      [DEBUG] Processing chunks from file: {chunks_filepath}")
        
        with open(chunks_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    chunk_data = json.loads(line)
                    chunk = chunk_data["text"]
                    chunk_id = chunk_data["chunk_id"]
                    chunk_count += 1
                    
                    print(f"      [DEBUG] Processing chunk {chunk_id+1} for {strategy['name']}")
                    
                    if not chunk.strip():
                        print(f"      [DEBUG] Skipping empty chunk {chunk_id+1}")
                        continue
                    
                    print(f"      [DEBUG] Chunk {chunk_id+1} length: {len(chunk)} chars")
                    
                    # Build simplified prompt for stability
                    if strategy["model"] == "qwen2.5:3b":
                        # Special prompt for qwen2.5:3b to avoid reasoning
                        prompt = f"""TEXT: {chunk}

EXTRACT disease names. Return ONLY: ["disease1", "disease2"]"""
                    else:
                        # Standard prompt for other models
                        prompt = f"""Diseases in this text: {chunk}

Return ONLY a JSON list like: ["disease1", "disease2"]"""
                    
                    print(f"      [DEBUG] Built prompt for chunk {chunk_id+1}, calling LLM...")
                    
                    # Generate response with simplified options for stability
                    options = {
                        "temperature": strategy["temperature"],
                        "top_p": 0.9,
                        "num_predict": 32,  # Reduced for speed and stability
                        "num_gpu": 1,        # Force GPU usage
                        "num_thread": 2,     # Reduced for stability
                        "repeat_penalty": 1.1, # Reduce repetition for speed
                        "top_k": 40,         # Optimize sampling
                    }
                    
                    # ADVANCED RETRY SYSTEM with confidence scoring
                    max_retries = MAX_LLM_RETRIES
                    present = []
                    retry_reason = "none"
                    final_attempt = 0
                    
                    for attempt in range(max_retries):
                        try:
                            print(f"      [DEBUG] Attempt {attempt+1}/{max_retries} for chunk {chunk_id+1}")
                            
                            client = get_thread_client()
                            response = client.generate(strategy["model"], system_prompt, prompt, options)
                            print(f"      [DEBUG] Got response for chunk {chunk_id+1}, length: {len(response)}")
                            
                            # Parse response - try multiple formats
                            print(f"      [DEBUG] Parsing response for chunk {chunk_id+1} (attempt {attempt+1})")
                            
                            # Try to find JSON array first (our expected format)
                            json_match = re.search(r'\[.*\]', response, re.DOTALL)
                            if json_match:
                                try:
                                    result = json.loads(json_match.group())
                                    if isinstance(result, list):
                                        present = result
                                        final_attempt = attempt + 1
                                        print(f"      [DEBUG] ✓ SUCCESS! Found JSON array with {len(present)} items: {present}")
                                        break  # Success, exit retry loop
                                    else:
                                        print(f"      [DEBUG] JSON array is not a list, retrying...")
                                        present = []
                                        retry_reason = "invalid_json_structure"
                                except json.JSONDecodeError:
                                    print(f"      [DEBUG] Failed to parse JSON array, retrying...")
                                    present = []
                                    retry_reason = "json_parse_error"
                            else:
                                # Try to find JSON object with "present" field
                                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                                if json_match:
                                    try:
                                        result = json.loads(json_match.group())
                                        present = result.get("present", [])
                                        if isinstance(present, list):
                                            final_attempt = attempt + 1
                                            print(f"      [DEBUG] ✓ SUCCESS! Found JSON object with 'present' field: {len(present)} items: {present}")
                                            break  # Success, exit retry loop
                                        else:
                                            print(f"      [DEBUG] 'present' field is not a list, retrying...")
                                            present = []
                                            retry_reason = "invalid_present_field"
                                    except json.JSONDecodeError:
                                        print(f"      [DEBUG] Failed to parse JSON object, retrying...")
                                        present = []
                                        retry_reason = "json_parse_error"
                                else:
                                    # No JSON found, retry
                                    print(f"      [DEBUG] No JSON found in attempt {attempt+1}, retrying...")
                                    print(f"      [DEBUG] Raw response: {response[:200]}...")
                                    present = []
                                    retry_reason = "no_json_found"
                            
                            # If we get here, parsing failed, try again
                            if attempt < max_retries - 1:
                                print(f"      [DEBUG] Parsing failed, waiting before retry...")
                                time.sleep(RETRY_DELAY_SECONDS)  # Brief pause between retries
                            
                        except Exception as e:
                            print(f"      [CHUNK_ERROR] {strategy['name']} chunk {chunk_id+1} attempt {attempt+1}: {e}")
                            if attempt < max_retries - 1:
                                print(f"      [DEBUG] Will retry...")
                                time.sleep(RETRY_DELAY_SECONDS)
                            else:
                                print(f"      [DEBUG] All retries exhausted for chunk {chunk_id+1}")
                                break
                    
                    # ADDITIONAL RETRY: If we got empty entities, try one more time
                    if not present and retry_reason != "none":
                        print(f"      [DEBUG] Empty entities detected, trying one more time for chunk {chunk_id+1}")
                        try:
                            # Modify prompt slightly to encourage entity detection
                            enhanced_prompt = f"""TEXT: {chunk}

EXTRACT disease names. If you find any diseases, return them as: ["disease1", "disease2"]
If you find NO diseases, return: []"""
                            
                            client = get_thread_client()
                            response = client.generate(strategy["model"], system_prompt, enhanced_prompt, options)
                            
                            # Try to parse again
                            json_match = re.search(r'\[.*\]', response, re.DOTALL)
                            if json_match:
                                try:
                                    result = json.loads(json_match.group())
                                    if isinstance(result, list):
                                        present = result
                                        final_attempt = 4  # Mark as 4th attempt (empty retry)
                                        print(f"      [DEBUG] ✓ SUCCESS on empty retry! Found {len(present)} items: {present}")
                                except json.JSONDecodeError:
                                    print(f"      [DEBUG] Empty retry failed to parse JSON")
                        except Exception as e:
                            print(f"      [DEBUG] Empty retry failed: {e}")
                    
                    # If all retries failed, try to extract from plain text as last resort
                    if not present and retry_reason != "none":
                        print(f"      [DEBUG] All retries failed, trying plain text extraction as last resort...")
                        # Look for disease-like patterns in the last response
                        disease_patterns = [
                            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|cancer|tumor|anemia|deficiency|mutation|gene)\b',
                            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:G\d+PD|BRCA\d+|ATM|LCAT)\b',
                            r'\b(?:G6PD|BRCA1|BRCA2|ATM|LCAT)\b'
                        ]
                        
                        for pattern in disease_patterns:
                            matches = re.findall(pattern, response, re.IGNORECASE)
                            for match in matches:
                                if match.lower() in [c.lower() for c in entity_candidates]:
                                    present.append(match)
                                    print(f"      [DEBUG] Extracted entity from text: {match}")
                        
                        if present:
                            print(f"      [DEBUG] Extracted {len(present)} entities from plain text as fallback")
                        else:
                            print(f"      [DEBUG] No entities found even with plain text extraction")
                    
                    # Process extracted entities
                    if isinstance(present, list) and present:
                        print(f"      [DEBUG] Processing {len(present)} entities: {present}")
                        print(f"      [DEBUG] Entity candidates: {entity_candidates}")
                        
                        for entity in present:
                            if entity and isinstance(entity, str):
                                # Check if entity matches any candidate (case-insensitive)
                                entity_lower = entity.lower().strip()
                                print(f"      [DEBUG] Checking entity: '{entity}' (lower: '{entity_lower}')")
                                
                                for candidate in entity_candidates:
                                    candidate_lower = candidate.lower().strip()
                                    print(f"      [DEBUG] Comparing with candidate: '{candidate}' (lower: '{candidate_lower}')")
                                    
                                    if candidate_lower == entity_lower:
                                        detected_entities.add(candidate)  # Use original candidate text
                                        print(f"      [DEBUG] ✓ MATCH! Found entity: {candidate} (matched: {entity})")
                                        break
                                    elif entity_lower in candidate_lower or candidate_lower in entity_lower:
                                        print(f"      [DEBUG] ~ PARTIAL MATCH: '{entity_lower}' vs '{candidate_lower}'")
                                        # Add partial matches with lower confidence
                                        detected_entities.add(candidate)
                                        print(f"      [DEBUG] ✓ PARTIAL MATCH! Added: {candidate}")
                                        break
                                    # Add fuzzy matching for similar terms
                                    elif _fuzzy_match(entity_lower, candidate_lower):
                                        print(f"      [DEBUG] ~ FUZZY MATCH: '{entity_lower}' vs '{candidate_lower}'")
                                        detected_entities.add(candidate)
                                        print(f"      [DEBUG] ✓ FUZZY MATCH! Added: {candidate}")
                                        break
                                    else:
                                        print(f"      [DEBUG] ✗ NO MATCH: '{entity_lower}' vs '{candidate_lower}'")
                        
                        print(f"      [DEBUG] Parsed {len(present)} entities from chunk {chunk_id+1}, {len(detected_entities)} matched candidates")
                    else:
                        print(f"      [DEBUG] No valid entities found in chunk {chunk_id+1} after {max_retries} attempts")
                        
                except json.JSONDecodeError as e:
                    print(f"      [JSON_ERROR] Failed to parse chunk line {line_num+1}: {e}")
                    continue
        
        print(f"      [DEBUG] Strategy {strategy['name']} completed with {len(detected_entities)} entities from {chunk_count} chunks")
        
        # Step 3: Save results to file
        results_filepath = save_strategy_results(doc_id, strategy['name'], detected_entities)
        
        # Step 4: Clean up chunks file to free memory
        try:
            import os
            os.remove(chunks_filepath)
            print(f"      [FILE] Cleaned up chunks file for {strategy['name']}")
        except Exception as e:
            print(f"      [WARNING] Could not clean up chunks file: {e}")
        
        return results_filepath
        
    except Exception as e:
        print(f"    [ERROR] Strategy {strategy['name']} failed: {e}")
        # Return empty results filepath
        empty_results = save_strategy_results(doc_id, strategy['name'], set())
        return empty_results
