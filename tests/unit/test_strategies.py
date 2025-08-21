"""
Tests unitarios para los módulos de estrategias del sistema NER.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from ner_app.strategies.regex_strategy import regex_detection
from ner_app.strategies.multi_strategy import run_multi_strategy_detection
from ner_app.config.strategies import STRATEGY_1, STRATEGY_2

class TestRegexStrategy:
    """Tests para la estrategia de regex."""
    
    def test_regex_detection_exact_match(self):
        """Test detección exacta con regex."""
        text = "The patient has diabetes mellitus and hypertension."
        entity_aliases = {
            "diabetes mellitus": "diabetes mellitus",
            "hypertension": "hypertension",
            "heart failure": "heart failure"
        }
        
        detected = regex_detection(text, entity_aliases)
        
        assert "diabetes mellitus" in detected
        assert "hypertension" in detected
        assert "heart failure" not in detected
    
    def test_regex_detection_case_insensitive(self):
        """Test detección insensible a mayúsculas."""
        text = "Patient diagnosed with DIABETES MELLITUS and Hypertension."
        entity_aliases = {
            "diabetes mellitus": "diabetes mellitus",
            "hypertension": "hypertension"
        }
        
        detected = regex_detection(text, entity_aliases)
        
        assert "diabetes mellitus" in detected
        assert "hypertension" in detected
    
    def test_regex_detection_word_boundaries(self):
        """Test detección con límites de palabra."""
        text = "The patient has diabetes but not pre-diabetes."
        entity_aliases = {
            "diabetes": "diabetes",
            "pre-diabetes": "pre-diabetes"
        }
        
        detected = regex_detection(text, entity_aliases)
        
        assert "diabetes" in detected
        # "pre-diabetes" no debe detectarse como "diabetes"
        assert len(detected) == 1
    
    def test_regex_detection_empty_input(self):
        """Test detección con entrada vacía."""
        # Texto vacío
        detected = regex_detection("", {"diabetes": "diabetes"})
        assert len(detected) == 0
        
        # Aliases vacíos
        detected = regex_detection("Patient has diabetes", {})
        assert len(detected) == 0
        
        # Alias vacío
        detected = regex_detection("Patient has diabetes", {"": "diabetes"})
        assert len(detected) == 0

class TestMultiStrategy:
    """Tests para el orquestador multi-estrategia."""
    
    @patch('ner_app.strategies.multi_strategy.regex_detection')
    @patch('ner_app.strategies.multi_strategy.llm_detection_strategy_file')
    @patch('ner_app.strategies.multi_strategy.load_strategy_results')
    @patch('ner_app.strategies.multi_strategy.cleanup_strategy_files')
    def test_run_multi_strategy_detection_basic(
        self, mock_cleanup, mock_load_results, mock_llm_strategy, mock_regex
    ):
        """Test básico del orquestador multi-estrategia."""
        # Configurar mocks
        mock_regex.return_value = {"diabetes mellitus"}
        mock_llm_strategy.return_value = "/fake/path"
        mock_load_results.return_value = {
            "strategy": "test_strategy",
            "entities": ["hypertension"],
            "count": 1
        }
        
        text = "Patient has diabetes mellitus and hypertension"
        entity_candidates = ["diabetes mellitus", "hypertension"]
        strategies = [STRATEGY_1]
        doc_id = "test_001"
        
        results = run_multi_strategy_detection(text, entity_candidates, strategies, doc_id)
        
        # Verificar estructura de resultados
        assert "all_detections" in results
        assert "entity_confidence" in results
        assert "entity_strategies" in results
        assert "accepted_entities" in results
        assert "total_detected" in results
        assert "total_accepted" in results
        
        # Verificar que se llamaron las funciones
        mock_regex.assert_called_once()
        mock_llm_strategy.assert_called_once()
        mock_cleanup.assert_called_once()
    
    @patch('ner_app.strategies.multi_strategy.regex_detection')
    @patch('ner_app.strategies.multi_strategy.llm_detection_strategy_file')
    @patch('ner_app.strategies.multi_strategy.load_strategy_results')
    def test_confidence_scoring(self, mock_load_results, mock_llm_strategy, mock_regex):
        """Test sistema de puntuación de confianza."""
        # Configurar mocks para detectar la misma entidad con múltiples estrategias
        mock_regex.return_value = {"diabetes mellitus"}
        mock_llm_strategy.return_value = "/fake/path"
        mock_load_results.return_value = {
            "strategy": "test_strategy",
            "entities": ["diabetes mellitus"],
            "count": 1
        }
        
        text = "Patient has diabetes mellitus"
        entity_candidates = ["diabetes mellitus"]
        strategies = [STRATEGY_1, STRATEGY_2]
        doc_id = "test_001"
        
        results = run_multi_strategy_detection(text, entity_candidates, strategies, doc_id)
        
        # La entidad debe tener alta confianza por múltiples detecciones
        assert "diabetes mellitus" in results["entity_confidence"]
        confidence = results["entity_confidence"]["diabetes mellitus"]
        assert confidence > 0.5  # Debe estar por encima del mínimo
        
        # Debe estar en la lista de aceptadas
        accepted_entities = [item["entity"] for item in results["accepted_entities"]]
        assert "diabetes mellitus" in accepted_entities
    
    @patch('ner_app.strategies.multi_strategy.regex_detection')
    @patch('ner_app.strategies.multi_strategy.llm_detection_strategy_file')
    @patch('ner_app.strategies.multi_strategy.load_strategy_results')
    def test_empty_detection(self, mock_load_results, mock_llm_strategy, mock_regex):
        """Test cuando no se detectan entidades."""
        # Configurar mocks para no detectar nada
        mock_regex.return_value = set()
        mock_llm_strategy.return_value = "/fake/path"
        mock_load_results.return_value = {
            "strategy": "test_strategy",
            "entities": [],
            "count": 0
        }
        
        text = "Patient has no specific conditions mentioned"
        entity_candidates = ["diabetes mellitus", "hypertension"]
        strategies = [STRATEGY_1]
        doc_id = "test_001"
        
        results = run_multi_strategy_detection(text, entity_candidates, strategies, doc_id)
        
        # No debe haber entidades detectadas
        assert results["total_detected"] == 0
        assert results["total_accepted"] == 0
        assert len(results["accepted_entities"]) == 0
        assert len(results["entity_confidence"]) == 0
    
    @patch('ner_app.strategies.multi_strategy.regex_detection')
    @patch('ner_app.strategies.multi_strategy.llm_detection_strategy_file')
    def test_strategy_failure_handling(self, mock_llm_strategy, mock_regex):
        """Test manejo de fallas en estrategias."""
        # Configurar mocks
        mock_regex.return_value = {"diabetes mellitus"}
        mock_llm_strategy.return_value = None  # Simular falla
        
        text = "Patient has diabetes mellitus"
        entity_candidates = ["diabetes mellitus"]
        strategies = [STRATEGY_1]
        doc_id = "test_001"
        
        # No debe lanzar excepción
        results = run_multi_strategy_detection(text, entity_candidates, strategies, doc_id)
        
        # Debe seguir funcionando con las estrategias que funcionan
        assert "all_detections" in results
        assert "regex" in results["all_detections"]
    
    @patch('ner_app.strategies.multi_strategy.get_system_prompts')
    def test_system_prompt_selection(self, mock_get_prompts):
        """Test selección de prompt del sistema."""
        mock_get_prompts.return_value = {
            "qwen2.5:3b": "qwen prompt",
            "default": "default prompt"
        }
        
        # Test con estrategia qwen
        qwen_strategy = {
            "name": "qwen_test",
            "model": "qwen2.5:3b",
            "chunk_target": 50,
            "chunk_overlap": 10,
            "chunk_min": 20,
            "chunk_max": 80,
            "temperature": 0.1,
            "weight": 1.0
        }
        
        with patch('ner_app.strategies.multi_strategy.regex_detection'), \
             patch('ner_app.strategies.multi_strategy.llm_detection_strategy_file') as mock_llm:
            
            mock_llm.return_value = "/fake/path"
            
            run_multi_strategy_detection(
                "test text", 
                ["diabetes"], 
                [qwen_strategy], 
                "test_001"
            )
            
            # Verificar que se seleccionó el prompt correcto
            mock_get_prompts.assert_called_once()
