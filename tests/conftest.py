"""
Configuración global para todos los tests del sistema NER Multi-Strategy.

Este archivo se ejecuta automáticamente por pytest y contiene fixtures
compartidas entre todos los tests.
"""

import pytest
import os
import json
import tempfile
import shutil
from typing import Dict, List, Any

# Fixture para datos de prueba sintéticos
@pytest.fixture
def sample_documents():
    """Documentos de prueba sintéticos para testing."""
    return [
        {
            "PMID": "test_001",
            "Texto": "The patient was diagnosed with diabetes mellitus type 2 and hypertension. Treatment includes metformin for glucose control.",
            "Entidad": [
                {"texto": "diabetes mellitus", "tipo": "SpecificDisease"},
                {"texto": "hypertension", "tipo": "SpecificDisease"}
            ]
        },
        {
            "PMID": "test_002", 
            "Texto": "Patient presents with chest pain and shortness of breath. ECG shows signs of atrial fibrillation.",
            "Entidad": [
                {"texto": "chest pain", "tipo": "SpecificDisease"},
                {"texto": "atrial fibrillation", "tipo": "SpecificDisease"}
            ]
        },
        {
            "PMID": "test_003",
            "Texto": "Blood work reveals anemia and vitamin D deficiency. Patient reports fatigue and weakness.",
            "Entidad": [
                {"texto": "anemia", "tipo": "SpecificDisease"},
                {"texto": "vitamin D deficiency", "tipo": "SpecificDisease"}
            ]
        }
    ]

@pytest.fixture
def sample_entity_candidates():
    """Lista de candidatos de entidades para testing."""
    return [
        "diabetes mellitus",
        "hypertension", 
        "chest pain",
        "atrial fibrillation",
        "anemia",
        "vitamin D deficiency",
        "heart failure",
        "pneumonia"
    ]

@pytest.fixture
def sample_strategies():
    """Estrategias de prueba simplificadas."""
    return [
        {
            "name": "test_strategy_1",
            "model": "llama3.2:3b",
            "chunk_target": 50,
            "chunk_overlap": 10,
            "chunk_min": 20,
            "chunk_max": 80,
            "temperature": 0.1,
            "weight": 1.0
        },
        {
            "name": "test_strategy_2", 
            "model": "qwen2.5:3b",
            "chunk_target": 30,
            "chunk_overlap": 5,
            "chunk_min": 15,
            "chunk_max": 50,
            "temperature": 0.3,
            "weight": 0.8
        }
    ]

@pytest.fixture
def temp_jsonl_file(sample_documents):
    """Crea un archivo JSONL temporal para testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        for doc in sample_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)

@pytest.fixture
def temp_dir():
    """Crea un directorio temporal para testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def mock_ollama_response():
    """Respuesta mock del servidor Ollama."""
    return {
        "model": "llama3.2:3b",
        "response": '["diabetes mellitus", "hypertension"]',
        "done": True
    }

@pytest.fixture
def sample_confidence_thresholds():
    """Umbrales de confianza para testing."""
    return {
        "high": 0.9,
        "medium": 0.7,
        "low": 0.5,
        "min_accept": 0.5
    }

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configuración inicial del entorno de testing."""
    # Configurar variables de entorno para testing
    os.environ['TESTING'] = 'true'
    os.environ['OLLAMA_HOST'] = 'localhost'
    os.environ['OLLAMA_PORT'] = '11434'
    
    yield
    
    # Cleanup del entorno
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

@pytest.fixture
def disable_llm_calls(monkeypatch):
    """Desactiva llamadas reales al LLM durante testing."""
    def mock_generate(self, model, system_prompt, user_prompt, options):
        # Respuesta mock basada en el prompt
        if "diabetes" in user_prompt.lower():
            return '["diabetes mellitus"]'
        elif "hypertension" in user_prompt.lower():
            return '["hypertension"]'
        else:
            return '[]'
    
    # Importar aquí para evitar problemas circulares
    from ner_app.core.llm_client import OllamaClient
    monkeypatch.setattr(OllamaClient, 'generate', mock_generate)

@pytest.fixture
def sample_detection_results():
    """Resultados de detección de muestra para testing."""
    return {
        "all_detections": {
            "regex": ["diabetes mellitus", "hypertension"],
            "llama32_balanced": ["diabetes mellitus"],
            "qwen25_diversity": ["hypertension"]
        },
        "entity_confidence": {
            "diabetes mellitus": 0.85,
            "hypertension": 0.72
        },
        "entity_strategies": {
            "diabetes mellitus": ["regex", "llama32_balanced"],
            "hypertension": ["regex", "qwen25_diversity"]
        },
        "accepted_entities": [
            {
                "entity": "diabetes mellitus",
                "confidence": 0.85,
                "strategies": ["regex", "llama32_balanced"],
                "strategy_count": 2
            },
            {
                "entity": "hypertension", 
                "confidence": 0.72,
                "strategies": ["regex", "qwen25_diversity"],
                "strategy_count": 2
            }
        ],
        "total_detected": 2,
        "total_accepted": 2
    }
