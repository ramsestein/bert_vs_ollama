"""
General system settings and constants for the Multi-Strategy NER system.

Contains file paths, processing settings, and system-wide constants.
"""

import os

# File-based processing settings
TEMP_DIR = "temp_processing"
CHUNK_FILE_PREFIX = "chunks_"
STRATEGY_FILE_PREFIX = "strategy_"

# LLM client settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 11434
DEFAULT_TIMEOUT = 600  # seconds (increased to handle slow model responses)

# Caching settings
CACHE_MAX_SIZE = 1000
CACHE_TTL_HOURS = 24

# Processing settings
MAX_CHUNK_ITERATIONS = 1000  # Safety limit for chunking
MIN_CHUNK_SIZE = 5           # Minimum words per chunk
MAX_CHUNK_SIZE = 200         # Maximum words per chunk

# Retry settings
MAX_LLM_RETRIES = 3
RETRY_DELAY_SECONDS = 1

# Threading settings
MAX_WORKERS = 4  # Maximum parallel strategies

# Language settings
DEFAULT_LANGUAGE = "en"  # "en" for English, "es" for Spanish

def get_temp_dir() -> str:
    """Get the temporary directory path, creating it if necessary."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    return TEMP_DIR

def get_chunk_file_path(doc_id: str, strategy_name: str) -> str:
    """Generate chunk file path for a specific document and strategy."""
    filename = f"{CHUNK_FILE_PREFIX}{doc_id}_{strategy_name}.jsonl"
    return os.path.join(TEMP_DIR, filename)

def get_strategy_file_path(doc_id: str, strategy_name: str) -> str:
    """Generate strategy results file path for a specific document and strategy."""
    filename = f"{STRATEGY_FILE_PREFIX}{doc_id}_{strategy_name}.json"
    return os.path.join(TEMP_DIR, filename)

def get_system_prompts(language="en"):
    """Get system prompts for different models.
    
    Args:
        language: "en" for English, "es" for Spanish
    """
    if language == "es":
        return {
            "qwen2.5:3b": """Eres un extractor de enfermedades. Extrae nombres de enfermedades de texto médico.

CRÍTICO: NO uses razonamiento. Devuelve SOLO una lista JSON de nombres de enfermedades.
Ejemplo: ["enfermedad1", "enfermedad2"]""",
            
            "default": """Eres un extractor de entidades médicas. Extrae nombres de enfermedades y condiciones médicas del texto.

REGLAS:
1. Solo extrae entidades que sean enfermedades/condiciones
2. Sé conservador - si no estás seguro, no extraigas
3. Devuelve entidades en formato JSON válido
4. La evidencia debe ser la mención EXACTA del texto

Devuelve SOLO JSON válido sin explicaciones."""
        }
    else:  # English
        return {
            "qwen2.5:3b": """You are a disease extractor. Extract disease names from biomedical text.

CRITICAL: Do NOT use reasoning or thinking. Return ONLY a JSON list of disease names.
Example: ["disease1", "disease2"]""",
            
            "default": """You are a biomedical entity extractor. Extract disease names and medical conditions from the text.

RULES:
1. Only extract entities that are diseases/conditions
2. Be conservative - if unsure, don't extract
3. Return entities in valid JSON format
4. Evidence must be the EXACT literal mention from the text

Output ONLY valid JSON with no explanations."""
        }
