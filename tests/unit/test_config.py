"""
Tests unitarios para los módulos de configuración del sistema NER.
"""

import pytest
from ner_app.config.strategies import (
    ALL_STRATEGIES, STRATEGY_1, STRATEGY_2, STRATEGY_3, STRATEGY_4,
    get_strategy_by_name, get_strategies_by_model, validate_strategy
)
from ner_app.config.thresholds import (
    CONFIDENCE_THRESHOLDS, update_confidence_thresholds,
    get_confidence_level, is_entity_accepted, get_confidence_rules
)
from ner_app.config.settings import (
    TEMP_DIR, get_temp_dir, get_chunk_file_path, get_strategy_file_path,
    get_system_prompts
)

class TestStrategies:
    """Tests para el módulo de estrategias."""
    
    def test_all_strategies_loaded(self):
        """Test que verifica que todas las estrategias están cargadas."""
        assert len(ALL_STRATEGIES) == 4
        assert STRATEGY_1 in ALL_STRATEGIES
        assert STRATEGY_2 in ALL_STRATEGIES
        assert STRATEGY_3 in ALL_STRATEGIES
        assert STRATEGY_4 in ALL_STRATEGIES
    
    def test_strategy_structure(self):
        """Test que verifica la estructura de las estrategias."""
        required_keys = [
            "name", "model", "chunk_target", "chunk_overlap",
            "chunk_min", "chunk_max", "temperature", "weight"
        ]
        
        for strategy in ALL_STRATEGIES:
            for key in required_keys:
                assert key in strategy
                assert strategy[key] is not None
    
    def test_get_strategy_by_name(self):
        """Test para obtener estrategia por nombre."""
        strategy = get_strategy_by_name("llama32_max_sensitivity")
        assert strategy is not None
        assert strategy["name"] == "llama32_max_sensitivity"
        
        # Test estrategia inexistente
        strategy = get_strategy_by_name("nonexistent")
        assert strategy is None
    
    def test_get_strategies_by_model(self):
        """Test para obtener estrategias por modelo."""
        llama_strategies = get_strategies_by_model("llama3.2:3b")
        assert len(llama_strategies) == 3  # 3 estrategias llama
        
        qwen_strategies = get_strategies_by_model("qwen2.5:3b")
        assert len(qwen_strategies) == 1  # 1 estrategia qwen
    
    def test_validate_strategy_valid(self):
        """Test validación de estrategia válida."""
        for strategy in ALL_STRATEGIES:
            assert validate_strategy(strategy) == True
    
    def test_validate_strategy_invalid(self):
        """Test validación de estrategia inválida."""
        invalid_strategy = {
            "name": "test",
            "model": "test",
            "chunk_target": -1,  # Inválido
            "chunk_overlap": 0,
            "chunk_min": 10,
            "chunk_max": 50,
            "temperature": 0.5,
            "weight": 1.0
        }
        assert validate_strategy(invalid_strategy) == False

class TestThresholds:
    """Tests para el módulo de umbrales."""
    
    def test_confidence_thresholds_structure(self):
        """Test estructura de umbrales de confianza."""
        required_keys = ["high", "medium", "low", "min_accept"]
        for key in required_keys:
            assert key in CONFIDENCE_THRESHOLDS
            assert isinstance(CONFIDENCE_THRESHOLDS[key], (int, float))
    
    def test_update_confidence_thresholds(self):
        """Test actualización de umbrales."""
        original_value = CONFIDENCE_THRESHOLDS["min_accept"]
        
        # Test actualización válida
        assert update_confidence_thresholds(0.7) == True
        assert CONFIDENCE_THRESHOLDS["min_accept"] == 0.7
        
        # Test actualización inválida
        assert update_confidence_thresholds(1.5) == False
        assert CONFIDENCE_THRESHOLDS["min_accept"] == 0.7  # No cambió
        
        # Restaurar valor original
        CONFIDENCE_THRESHOLDS["min_accept"] = original_value
    
    def test_get_confidence_level(self):
        """Test obtención de nivel de confianza."""
        assert get_confidence_level(0.95) == "high"
        assert get_confidence_level(0.75) == "medium"
        assert get_confidence_level(0.55) == "low"
        assert get_confidence_level(0.3) == "below_threshold"
    
    def test_is_entity_accepted(self):
        """Test aceptación de entidades."""
        assert is_entity_accepted(0.6) == True  # Por encima del mínimo
        assert is_entity_accepted(0.4) == False  # Por debajo del mínimo
    
    def test_get_confidence_rules(self):
        """Test obtención de reglas de confianza."""
        rules = get_confidence_rules()
        expected_keys = [
            "regex_multiplier", "multi_strategy_bonus", "llm_only_penalty",
            "max_confidence", "min_confidence"
        ]
        for key in expected_keys:
            assert key in rules

class TestSettings:
    """Tests para el módulo de configuración general."""
    
    def test_temp_dir_constants(self):
        """Test constantes de directorio temporal."""
        assert TEMP_DIR == "temp_processing"
        assert isinstance(TEMP_DIR, str)
    
    def test_get_temp_dir(self):
        """Test obtención de directorio temporal."""
        temp_dir = get_temp_dir()
        assert isinstance(temp_dir, str)
        assert len(temp_dir) > 0
    
    def test_get_chunk_file_path(self):
        """Test generación de ruta de archivo de chunks."""
        path = get_chunk_file_path("test_doc", "test_strategy")
        assert "chunks_test_doc_test_strategy.jsonl" in path
        assert path.endswith(".jsonl")
    
    def test_get_strategy_file_path(self):
        """Test generación de ruta de archivo de estrategia."""
        path = get_strategy_file_path("test_doc", "test_strategy")
        assert "strategy_test_doc_test_strategy.json" in path
        assert path.endswith(".json")
    
    def test_get_system_prompts(self):
        """Test obtención de prompts del sistema."""
        prompts = get_system_prompts()
        assert "qwen2.5:3b" in prompts
        assert "default" in prompts
        assert isinstance(prompts["qwen2.5:3b"], str)
        assert isinstance(prompts["default"], str)
        assert len(prompts["qwen2.5:3b"]) > 0
        assert len(prompts["default"]) > 0
