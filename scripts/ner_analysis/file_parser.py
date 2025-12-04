#!/usr/bin/env python3
"""
File Parser Module
Extracts parameters from result filenames and reads result files
"""

import json
import re
from typing import Dict, Optional


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Extrae parámetros del nombre del archivo de forma flexible
    
    Formatos soportados:
    - results_{modelo}_{dataset}_chunk{n}_ov{n}.jsonl
    - results_{modelo}_{dataset}_chunk{n}_ov{n}_temp{n}.jsonl
    - results_{modelo}_{dataset}_chunk{n}_ov{n}_temp{n}_conf{n}.jsonl
    
    Args:
        filename: Nombre del archivo a parsear
    
    Returns:
        Dict con parámetros extraídos o None si no coincide con el patrón
    """
    pattern = r"results_([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)_chunk(\d+)_ov(\d+)(?:_temp([\d.]+))?(?:_conf([\d.]+))?\.jsonl"
    match = re.search(pattern, filename)
    
    if not match:
        return None
    
    params = {
        'model': match.group(1),
        'dataset': match.group(2),
        'chunk': int(match.group(3)),
        'overlap': int(match.group(4)),
    }
    
    # Temperatura opcional
    if match.group(5):
        params['temperature'] = float(match.group(5))
    
    # Confidence opcional
    if match.group(6):
        params['confidence'] = float(match.group(6))
    
    return params


def read_results(filepath: str) -> Optional[Dict]:
    """
    Lee y analiza un archivo de resultados JSONL
    
    Args:
        filepath: Ruta al archivo de resultados
    
    Returns:
        Dict con estadísticas agregadas o None si hay error
    """
    docs = 0
    total_entities = 0
    confidences = []
    entities_set = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                docs += 1
                data = json.loads(line)
                ents = data.get("Entidad", [])
                total_entities += len(ents)
                
                for ent in ents:
                    confidences.append(ent.get("confidence", 0.0))
                    texto = ent.get("texto", "").strip().lower()
                    if texto:
                        entities_set.add(texto)
                        
    except Exception as e:
        print(f"[ERROR] Leyendo {filepath}: {e}")
        return None
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "docs": docs,
        "entities": total_entities,
        "avg_conf": avg_conf,
        "entities_set": entities_set,
        "min_conf": min(confidences) if confidences else 0.0,
        "max_conf": max(confidences) if confidences else 0.0,
    }
