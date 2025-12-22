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
            "qwen2.5:3b": """Eres un extractor de entidades diagnósticas. Extrae únicamente nombres de diagnósticos mencionados en el texto.

CRÍTICO:
- NO uses razonamiento.
- NO añadas explicaciones.
- NO añadas comentarios.
- Devuelve SOLO una lista JSON válida de nombres de diagnósticos.

Ejemplo de salida: ["diagnóstico1", "diagnóstico2"]""",
            
            "default": """Eres un extractor experto de entidades diagnósticas biomédicas. Tu única tarea es identificar y devolver nombres de diagnósticos presentes en el texto.

REGLAS:
1. Solo extrae entidades que sean diagnósticos o condiciones médicas.
2. Sé conservador: si no estás seguro, no extraigas nada.
3. Las entidades deben aparecer EXACTAMENTE como en el texto (mismo literal).
4. Devuelve el resultado exclusivamente como JSON válido.
5. NO incluyas explicaciones, notas ni texto adicional fuera del JSON.

Devuelve SOLO JSON."""
        }
    else:  # English
        return {
            "qwen2.5:3b": """You are a disease extractor. Extract only disease names mentioned in the text.

CRITICAL:
- Do NOT use reasoning.
- Do NOT add explanations.
- Do NOT add comments.
- Return ONLY a valid JSON list of disease names.

Example output: ["disease1", "disease2"]""",
            
            "default": """You are an expert biomedical entity extractor. Your only task is to identify and return disease names and medical conditions present in the text.

RULES:
1. Extract only entities that are diseases or medical conditions.
2. Be conservative: if you are unsure, do not extract anything.
3. Entities must appear EXACTLY as written in the text.
4. Return the output exclusively as valid JSON.
5. Do NOT include explanations, notes, or any text outside the JSON.

Output ONLY JSON."""
        }
