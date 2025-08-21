"""
File management utilities for the Multi-Strategy NER system.

Handles temporary file creation, cleanup, and file-based processing for memory efficiency.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Set
from ..config.settings import TEMP_DIR, get_temp_dir, get_chunk_file_path, get_strategy_file_path

def ensure_temp_dir():
    """Ensure temporary directory exists."""
    temp_dir = get_temp_dir()
    print(f"[INFO] Using temporary directory: {temp_dir}")

def cleanup_temp_files():
    """Clean up temporary files."""
    try:
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(TEMP_DIR)
            print(f"[INFO] Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        print(f"[WARNING] Could not clean up temp files: {e}")

def save_chunks_to_file(doc_id: str, chunks: List[str], strategy_name: str) -> str:
    """Save chunks to a temporary file."""
    filepath = get_chunk_file_path(doc_id, strategy_name)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i,
                "text": chunk,
                "length": len(chunk)
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
    
    print(f"      [FILE] Saved {len(chunks)} chunks to {os.path.basename(filepath)}")
    return filepath

def load_chunks_from_file(filepath: str) -> List[str]:
    """Load chunks from a temporary file."""
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk_data = json.loads(line)
                chunks.append(chunk_data["text"])
    return chunks

def save_strategy_results(doc_id: str, strategy_name: str, entities: Set[str]) -> str:
    """Save strategy results to a temporary file."""
    filepath = get_strategy_file_path(doc_id, strategy_name)
    
    results = {
        "strategy": strategy_name,
        "entities": list(entities),
        "count": len(entities),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"      [FILE] Saved strategy results to {os.path.basename(filepath)}")
    return filepath

def load_strategy_results(filepath: str) -> Dict:
    """Load strategy results from a temporary file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def text_to_chunks_file(text: str, strategy: Dict, doc_id: str) -> str:
    """Convert text to chunks and save to file, return filepath."""
    from .text_processor import create_chunks_from_text
    
    print(f"      [CHUNK] Creating chunks for {strategy['name']}")
    
    # Create chunks using the text processor
    chunks = create_chunks_from_text(text, strategy)
    
    # Save chunks to file and return filepath
    return save_chunks_to_file(doc_id, chunks, strategy['name'])

def cleanup_strategy_files(strategy_filepaths: Dict[str, str]):
    """Clean up strategy result files to free memory."""
    for strategy_name, filepath in strategy_filepaths.items():
        try:
            os.remove(filepath)
            print(f"      [FILE] Cleaned up strategy results file for {strategy_name}")
        except Exception as e:
            print(f"      [WARNING] Could not clean up strategy file for {strategy_name}: {e}")

def cleanup_chunk_files(chunk_filepaths: Dict[str, str]):
    """Clean up chunk files to free memory."""
    for strategy_name, filepath in chunk_filepaths.items():
        try:
            os.remove(filepath)
            print(f"      [FILE] Cleaned up chunks file for {strategy_name}")
        except Exception as e:
            print(f"      [WARNING] Could not clean up chunks file for {strategy_name}: {e}")

def get_temp_file_info() -> Dict:
    """Get information about temporary files."""
    if not os.path.exists(TEMP_DIR):
        return {"exists": False, "file_count": 0, "total_size": 0}
    
    files = os.listdir(TEMP_DIR)
    total_size = 0
    
    for file in files:
        file_path = os.path.join(TEMP_DIR, file)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    
    return {
        "exists": True,
        "file_count": len(files),
        "total_size": total_size,
        "files": files
    }
