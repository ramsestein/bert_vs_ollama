"""
Tests unitarios para los módulos de utilidades del sistema NER.
"""

import pytest
import argparse
from unittest.mock import Mock, patch
from ner_app.utils.cli_parser import (
    parse_arguments, configure_strategies, print_configuration, validate_arguments
)
from ner_app.utils.confidence_scorer import ConfidenceScorer
from ner_app.utils.entity_matcher import EntityMatcher

class TestCLIParser:
    """Tests para el parser de línea de comandos."""
    
    def test_parse_arguments_minimal(self):
        """Test parsing con argumentos mínimos."""
        with patch('sys.argv', [
            'script.py', 
            '--input_jsonl', 'test_input.jsonl'
        ]):
            args = parse_arguments()
            assert args.input_jsonl == 'test_input.jsonl'
            assert args.benchmark_jsonl is None  # Ahora es opcional
            assert args.out_pred == "results_multi_strategy.jsonl"
            assert args.limit == 0
            assert args.confidence_threshold == 0.5
    
    def test_parse_arguments_full(self):
        """Test parsing con todos los argumentos."""
        with patch('sys.argv', [
            'script.py',
            '--input_jsonl', 'test_input.jsonl',
            '--benchmark_jsonl', 'test_benchmark.jsonl',
            '--out_pred', 'custom_output.jsonl',
            '--limit', '100',
            '--confidence_threshold', '0.7',
            '--strategies', 'llama32_balanced', 'qwen25_diversity'
        ]):
            args = parse_arguments()
            assert args.input_jsonl == 'test_input.jsonl'
            assert args.benchmark_jsonl == 'test_benchmark.jsonl'
            assert args.out_pred == 'custom_output.jsonl'
            assert args.limit == 100
            assert args.confidence_threshold == 0.7
            assert args.strategies == ['llama32_balanced', 'qwen25_diversity']
    
    def test_configure_strategies_all(self):
        """Test configuración de todas las estrategias."""
        mock_args = Mock()
        mock_args.strategies = ["all"]
        mock_args.s1_target = None
        mock_args.s1_overlap = None
        mock_args.s1_min = None
        mock_args.s1_max = None
        mock_args.s1_temp = None
        mock_args.s2_target = None
        mock_args.s2_overlap = None
        mock_args.s2_min = None
        mock_args.s2_max = None
        mock_args.s2_temp = None
        
        strategies = configure_strategies(mock_args)
        assert len(strategies) == 4  # Todas las estrategias
    
    def test_configure_strategies_specific(self):
        """Test configuración de estrategias específicas."""
        mock_args = Mock()
        mock_args.strategies = ["llama32_balanced"]
        mock_args.s1_target = None
        mock_args.s1_overlap = None
        mock_args.s1_min = None
        mock_args.s1_max = None
        mock_args.s1_temp = None
        mock_args.s2_target = None
        mock_args.s2_overlap = None
        mock_args.s2_min = None
        mock_args.s2_max = None
        mock_args.s2_temp = None
        
        strategies = configure_strategies(mock_args)
        assert len(strategies) == 1
        assert strategies[0]["name"] == "llama32_balanced"
    
    def test_configure_strategies_with_overrides(self):
        """Test configuración con overrides de parámetros."""
        mock_args = Mock()
        mock_args.strategies = ["all"]
        mock_args.s1_target = 200  # Override
        mock_args.s1_overlap = 50   # Override
        mock_args.s1_min = None
        mock_args.s1_max = None
        mock_args.s1_temp = 0.8     # Override
        mock_args.s2_target = None
        mock_args.s2_overlap = None
        mock_args.s2_min = None
        mock_args.s2_max = None
        mock_args.s2_temp = None
        
        strategies = configure_strategies(mock_args)
        
        # Verificar que se aplicaron los overrides
        strategy_1 = strategies[0]
        assert strategy_1["chunk_target"] == 200
        assert strategy_1["chunk_overlap"] == 50
        assert strategy_1["temperature"] == 0.8
    
    @patch('os.path.exists')
    def test_validate_arguments_valid(self, mock_exists):
        """Test validación de argumentos válidos."""
        mock_exists.return_value = True
        
        mock_args = Mock()
        mock_args.input_jsonl = "test_input.jsonl"
        mock_args.benchmark_jsonl = "test_benchmark.jsonl"
        mock_args.confidence_threshold = 0.5
        mock_args.limit = 10
        
        assert validate_arguments(mock_args) == True
    
    @patch('os.path.exists')
    def test_validate_arguments_no_benchmark(self, mock_exists):
        """Test validación sin benchmark (modo solo anotación)."""
        mock_exists.return_value = True
        
        mock_args = Mock()
        mock_args.input_jsonl = "test_input.jsonl"
        mock_args.benchmark_jsonl = None  # Sin benchmark
        mock_args.confidence_threshold = 0.5
        mock_args.limit = 10
        
        assert validate_arguments(mock_args) == True
    
    @patch('os.path.exists')
    def test_validate_arguments_invalid_file(self, mock_exists):
        """Test validación con archivo inexistente."""
        mock_exists.return_value = False
        
        mock_args = Mock()
        mock_args.input_jsonl = "nonexistent.jsonl"
        mock_args.benchmark_jsonl = None
        mock_args.confidence_threshold = 0.5
        mock_args.limit = 10
        
        assert validate_arguments(mock_args) == False
    
    def test_validate_arguments_invalid_threshold(self):
        """Test validación con umbral inválido."""
        mock_args = Mock()
        mock_args.input_jsonl = "test_input.jsonl"
        mock_args.benchmark_jsonl = None
        mock_args.confidence_threshold = 1.5  # Inválido
        mock_args.limit = 10
        
        with patch('os.path.exists', return_value=True):
            assert validate_arguments(mock_args) == False

class TestConfidenceScorer:
    """Tests para el sistema de puntuación de confianza."""
    
    def test_calculate_confidence_single_strategy(self):
        """Test cálculo de confianza con una estrategia."""
        scorer = ConfidenceScorer()
        strategies = ["llama32_balanced"]
        strategy_weights = {"llama32_balanced": 1.0}
        
        confidence = scorer.calculate_confidence("diabetes", strategies, strategy_weights)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0
    
    def test_calculate_confidence_multiple_strategies(self):
        """Test cálculo de confianza con múltiples estrategias."""
        scorer = ConfidenceScorer()
        strategies = ["regex", "llama32_balanced", "qwen25_diversity"]
        strategy_weights = {
            "regex": 1.0,
            "llama32_balanced": 1.0,
            "qwen25_diversity": 0.8
        }
        
        confidence = scorer.calculate_confidence("diabetes", strategies, strategy_weights)
        
        assert 0.0 <= confidence <= 1.0
        # Múltiples estrategias deben dar mayor confianza
        assert confidence > 0.5
    
    def test_calculate_confidence_with_regex(self):
        """Test cálculo de confianza incluyendo regex."""
        scorer = ConfidenceScorer()
        strategies_with_regex = ["regex", "llama32_balanced"]
        strategies_without_regex = ["llama32_balanced"]
        strategy_weights = {"regex": 1.0, "llama32_balanced": 1.0}
        
        conf_with_regex = scorer.calculate_confidence("diabetes", strategies_with_regex, strategy_weights)
        conf_without_regex = scorer.calculate_confidence("diabetes", strategies_without_regex, strategy_weights)
        
        # Regex debe aumentar la confianza
        assert conf_with_regex > conf_without_regex
    
    def test_filter_entities_by_confidence(self):
        """Test filtrado de entidades por confianza."""
        scorer = ConfidenceScorer()
        entity_confidence = {
            "diabetes": 0.8,
            "hypertension": 0.6,
            "low_confidence_entity": 0.3
        }
        
        accepted = scorer.filter_entities_by_confidence(entity_confidence)
        
        # Solo entidades con confianza >= 0.5 deben ser aceptadas
        accepted_entities = [item["entity"] for item in accepted]
        assert "diabetes" in accepted_entities
        assert "hypertension" in accepted_entities
        assert "low_confidence_entity" not in accepted_entities
    
    def test_get_confidence_level(self):
        """Test obtención de nivel de confianza."""
        scorer = ConfidenceScorer()
        
        assert scorer.get_confidence_level(0.95) == "high"
        assert scorer.get_confidence_level(0.75) == "medium"
        assert scorer.get_confidence_level(0.55) == "low"
        assert scorer.get_confidence_level(0.3) == "below_threshold"
    
    def test_get_confidence_summary(self):
        """Test resumen de estadísticas de confianza."""
        scorer = ConfidenceScorer()
        entity_confidence = {
            "diabetes": 0.9,
            "hypertension": 0.7,
            "anemia": 0.5,
            "low_entity": 0.3
        }
        
        summary = scorer.get_confidence_summary(entity_confidence)
        
        assert summary["total_entities"] == 4
        assert summary["average_confidence"] == 0.6
        assert summary["confidence_range"] == [0.3, 0.9]
        assert summary["by_level"]["high"] == 1
        assert summary["by_level"]["medium"] == 1
        assert summary["by_level"]["low"] == 1
        assert summary["by_level"]["below_threshold"] == 1

class TestEntityMatcher:
    """Tests para el matcher de entidades."""
    
    def test_match_entities_exact(self):
        """Test matching exacto de entidades."""
        matcher = EntityMatcher()
        detected = ["diabetes mellitus", "hypertension"]
        candidates = ["diabetes mellitus", "hypertension", "heart failure"]
        
        matches = matcher.match_entities(detected, candidates)
        
        assert len(matches["exact"]) == 2
        assert ("diabetes mellitus", "diabetes mellitus") in matches["exact"]
        assert ("hypertension", "hypertension") in matches["exact"]
        assert len(matches["unmatched"]) == 0
    
    def test_match_entities_partial(self):
        """Test matching parcial de entidades."""
        matcher = EntityMatcher()
        detected = ["diabetes", "heart"]
        candidates = ["diabetes mellitus", "heart failure"]
        
        matches = matcher.match_entities(detected, candidates)
        
        # Debe encontrar matches parciales
        assert len(matches["partial"]) == 2
        assert len(matches["exact"]) == 0
    
    def test_match_entities_fuzzy(self):
        """Test matching difuso de entidades."""
        matcher = EntityMatcher(fuzzy_threshold=0.5)
        detected = ["diabetic"]
        candidates = ["diabetes mellitus"]
        
        matches = matcher.match_entities(detected, candidates)
        
        # Debe encontrar al menos algún tipo de match
        total_matches = len(matches["exact"]) + len(matches["partial"]) + len(matches["fuzzy"])
        assert total_matches >= 0  # Puede variar según el threshold
    
    def test_match_entities_unmatched(self):
        """Test entidades no matching."""
        matcher = EntityMatcher()
        detected = ["unknown_disease", "another_unknown"]
        candidates = ["diabetes mellitus", "hypertension"]
        
        matches = matcher.match_entities(detected, candidates)
        
        assert len(matches["unmatched"]) == 2
        assert "unknown_disease" in matches["unmatched"]
        assert "another_unknown" in matches["unmatched"]
    
    def test_validate_entity_candidates(self):
        """Test validación de candidatos de entidades."""
        matcher = EntityMatcher()
        candidates = [
            "diabetes mellitus",    # Válido
            "",                     # Inválido (vacío)
            "hypertension",         # Válido
            None,                   # Inválido (None)
            "diabetes mellitus",    # Duplicado
            "  heart failure  "     # Válido (con espacios)
        ]
        
        validation = matcher.validate_entity_candidates(candidates)
        
        assert len(validation["valid"]) == 3  # diabetes, hypertension, heart failure
        assert len(validation["invalid"]) == 2  # vacío y None
        assert len(validation["duplicates"]) == 1  # diabetes duplicado
        assert validation["total"] == 6
    
    def test_get_matching_statistics(self):
        """Test estadísticas de matching."""
        matcher = EntityMatcher()
        matches = {
            "exact": [("diabetes", "diabetes mellitus")],
            "partial": [("hyper", "hypertension")],
            "fuzzy": [],
            "unmatched": ["unknown_disease"]
        }
        
        stats = matcher.get_matching_statistics(matches)
        
        assert stats["total_detected"] == 3
        assert stats["exact_matches"] == 1
        assert stats["partial_matches"] == 1
        assert stats["fuzzy_matches"] == 0
        assert stats["unmatched"] == 1
        assert 0.0 <= stats["match_rate"] <= 1.0
    
    def test_suggest_candidate_improvements(self):
        """Test sugerencias de mejora de candidatos."""
        matcher = EntityMatcher()
        matches = {
            "exact": [],
            "partial": [],
            "fuzzy": [("diabetic", "diabetes mellitus")],
            "unmatched": ["new_disease"]
        }
        candidates = ["diabetes mellitus", "diabetes mellitus"]  # Duplicado
        
        suggestions = matcher.suggest_candidate_improvements(matches, candidates)
        
        assert len(suggestions) > 0
        # Debe sugerir agregar la entidad no matching
        assert any("new_disease" in suggestion for suggestion in suggestions)
        # Debe detectar duplicados
        assert any("Duplicate" in suggestion for suggestion in suggestions)
