"""
Tests unitarios para los módulos core del sistema NER.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from ner_app.core.text_processor import (
    _fuzzy_match, normalize_surface, tokenize, sentence_chunks, create_chunks_from_text
)
from ner_app.core.file_manager import (
    ensure_temp_dir, cleanup_temp_files, save_chunks_to_file,
    load_chunks_from_file, save_strategy_results, load_strategy_results,
    text_to_chunks_file
)
from ner_app.core.llm_client import LLMCache, OllamaClient, get_thread_client

class TestTextProcessor:
    """Tests para el procesador de texto."""
    
    def test_fuzzy_match(self):
        """Test matching difuso."""
        # Match exacto
        assert _fuzzy_match("diabetes", "diabetes") == True
        
        # Match similar
        assert _fuzzy_match("diabetes mellitus", "diabetes") == True
        
        # No match
        assert _fuzzy_match("diabetes", "hypertension") == False
        
        # Casos edge
        assert _fuzzy_match("", "diabetes") == False
        assert _fuzzy_match("diabetes", "") == False
        assert _fuzzy_match("", "") == False
    
    def test_normalize_surface(self):
        """Test normalización de texto."""
        # Espacios múltiples
        assert normalize_surface("  hello   world  ") == "hello world"
        
        # Comillas
        assert normalize_surface('"quoted text"') == '"quoted text"'
        
        # Guiones
        assert normalize_surface("text—with—dashes") == "text-with-dashes"
        
        # Texto vacío
        assert normalize_surface("") == ""
        assert normalize_surface(None) == ""
    
    def test_tokenize(self):
        """Test tokenización."""
        tokens = tokenize("hello world test")
        assert tokens == ["hello", "world", "test"]
        
        # Texto vacío
        tokens = tokenize("")
        assert tokens == [""]
    
    def test_sentence_chunks(self):
        """Test creación de chunks."""
        text = "This is a test sentence with multiple words for chunking"
        chunks = sentence_chunks(text, target_tokens=5, overlap_tokens=2)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        # Texto corto
        short_text = "Short text"
        chunks = sentence_chunks(short_text, target_tokens=10, overlap_tokens=2)
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_create_chunks_from_text(self):
        """Test creación de chunks con estrategia."""
        strategy = {
            "name": "test_strategy",
            "chunk_target": 5,
            "chunk_overlap": 2,
            "chunk_min": 3,
            "chunk_max": 10
        }
        
        text = "This is a test sentence with multiple words"
        chunks = create_chunks_from_text(text, strategy)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

class TestFileManager:
    """Tests para el gestor de archivos."""
    
    def test_save_and_load_chunks(self, temp_dir):
        """Test guardar y cargar chunks."""
        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        
        # Guardar chunks
        filepath = save_chunks_to_file("test_doc", chunks, "test_strategy")
        assert os.path.exists(filepath)
        
        # Cargar chunks
        loaded_chunks = load_chunks_from_file(filepath)
        assert loaded_chunks == chunks
        
        # Limpiar
        os.remove(filepath)
    
    def test_save_and_load_strategy_results(self, temp_dir):
        """Test guardar y cargar resultados de estrategia."""
        entities = {"diabetes", "hypertension"}
        
        # Guardar resultados
        filepath = save_strategy_results("test_doc", "test_strategy", entities)
        assert os.path.exists(filepath)
        
        # Cargar resultados
        loaded_results = load_strategy_results(filepath)
        assert "strategy" in loaded_results
        assert "entities" in loaded_results
        assert "count" in loaded_results
        assert set(loaded_results["entities"]) == entities
        
        # Limpiar
        os.remove(filepath)
    
    @patch('ner_app.core.file_manager.create_chunks_from_text')
    def test_text_to_chunks_file(self, mock_create_chunks, temp_dir):
        """Test conversión de texto a archivo de chunks."""
        mock_create_chunks.return_value = ["chunk1", "chunk2"]
        
        strategy = {"name": "test_strategy"}
        text = "test text"
        
        filepath = text_to_chunks_file(text, strategy, "test_doc")
        assert os.path.exists(filepath)
        
        # Verificar que se llamó la función de chunks
        mock_create_chunks.assert_called_once_with(text, strategy)
        
        # Limpiar
        os.remove(filepath)

class TestLLMClient:
    """Tests para el cliente LLM."""
    
    def test_llm_cache_basic_operations(self):
        """Test operaciones básicas del caché LLM."""
        cache = LLMCache(max_size=10, ttl_hours=1)
        
        # Test put/get
        key = cache._generate_key("model", "system", "user")
        cache.put(key, "test_response")
        
        response = cache.get(key)
        assert response == "test_response"
        
        # Test key inexistente
        fake_key = "fake_key"
        response = cache.get(fake_key)
        assert response is None
    
    def test_llm_cache_key_generation(self):
        """Test generación de claves del caché."""
        cache = LLMCache()
        
        key1 = cache._generate_key("model1", "system1", "user1")
        key2 = cache._generate_key("model1", "system1", "user1")
        key3 = cache._generate_key("model2", "system1", "user1")
        
        # Mismos parámetros = misma clave
        assert key1 == key2
        
        # Parámetros diferentes = claves diferentes
        assert key1 != key3
        
        # Las claves deben ser strings
        assert isinstance(key1, str)
        assert len(key1) > 0
    
    @patch('http.client.HTTPConnection')
    def test_ollama_client_initialization(self, mock_http):
        """Test inicialización del cliente Ollama."""
        client = OllamaClient(host="localhost", port=11434, timeout=30)
        
        assert client.host == "localhost"
        assert client.port == 11434
        assert client.timeout == 30
        assert client.conn is None
    
    @patch('http.client.HTTPConnection')
    def test_ollama_client_request_error_handling(self, mock_http):
        """Test manejo de errores en requests del cliente Ollama."""
        # Simular error de conexión
        mock_conn = Mock()
        mock_conn.request.side_effect = Exception("Connection error")
        mock_http.return_value = mock_conn
        
        client = OllamaClient()
        
        # El método _request debe manejar errores gracefully
        with pytest.raises(Exception):
            client._request("/test", {"test": "data"})
    
    def test_get_thread_client(self):
        """Test obtención de cliente por thread."""
        client1 = get_thread_client()
        client2 = get_thread_client()
        
        # Mismo thread = mismo cliente
        assert client1 is client2
        assert isinstance(client1, OllamaClient)
