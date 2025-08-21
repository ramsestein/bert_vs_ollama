"""
Tests de integración para el sistema NER refactorizado.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, Mock
from ner_app.main import main, process_document, load_documents, save_results
from ner_app.utils.cli_parser import parse_arguments, configure_strategies
from ner_app.config.strategies import ALL_STRATEGIES

class TestRefactoredSystemIntegration:
    """Tests de integración para el sistema refactorizado."""
    
    def test_load_documents_with_limit(self, temp_jsonl_file):
        """Test carga de documentos con límite."""
        documents = load_documents(temp_jsonl_file, limit=2)
        
        assert len(documents) == 2
        assert all("pmid" in doc for doc in documents)
        assert all("text" in doc for doc in documents)
        assert all("entity_candidates" in doc for doc in documents)
    
    def test_load_documents_no_limit(self, temp_jsonl_file):
        """Test carga de documentos sin límite."""
        documents = load_documents(temp_jsonl_file, limit=0)
        
        assert len(documents) == 3  # Todos los documentos del fixture
    
    def test_save_results(self, temp_dir, sample_detection_results):
        """Test guardado de resultados."""
        output_file = os.path.join(temp_dir, "test_output.jsonl")
        
        # Crear resultados de prueba
        results = [
            {
                "PMID": "test_001",
                "Texto": "Test text",
                "Entidad": [{"texto": "diabetes", "tipo": "SpecificDisease", "confidence": 0.8, "strategies": ["regex"]}],
                "_multi_strategy": sample_detection_results,
                "_latency_sec": 1.5
            }
        ]
        
        save_results(results, output_file)
        
        # Verificar que se guardó correctamente
        assert os.path.exists(output_file)
        
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_data = [json.loads(line) for line in f if line.strip()]
        
        assert len(saved_data) == 1
        assert saved_data[0]["PMID"] == "test_001"
    
    @patch('ner_app.strategies.multi_strategy.llm_detection_strategy_file')
    def test_process_document_integration(self, mock_llm_strategy, sample_strategies, disable_llm_calls):
        """Test procesamiento de documento completo."""
        mock_llm_strategy.return_value = "/fake/path"
        
        # Mock load_strategy_results
        with patch('ner_app.strategies.multi_strategy.load_strategy_results') as mock_load:
            mock_load.return_value = {
                "strategy": "test_strategy",
                "entities": ["diabetes mellitus"],
                "count": 1
            }
            
            result = process_document(
                pmid="test_001",
                text="Patient has diabetes mellitus and hypertension",
                entity_candidates=["diabetes mellitus", "hypertension"],
                strategies=sample_strategies
            )
            
            # Verificar estructura del resultado
            assert "PMID" in result
            assert "Texto" in result
            assert "Entidad" in result
            assert "_multi_strategy" in result
            assert "_latency_sec" in result
            
            assert result["PMID"] == "test_001"
            assert isinstance(result["Entidad"], list)
            assert isinstance(result["_latency_sec"], (int, float))
    
    @patch('sys.argv')
    @patch('ner_app.main.load_documents')
    @patch('ner_app.main.process_document')
    @patch('ner_app.main.save_results')
    def test_main_function_with_benchmark(
        self, mock_save, mock_process, mock_load, mock_argv, 
        temp_jsonl_file, sample_detection_results
    ):
        """Test función main completa con benchmark."""
        # Configurar argumentos simulados
        mock_argv.__getitem__ = Mock(side_effect=[
            'script.py',
            '--input_jsonl', temp_jsonl_file,
            '--benchmark_jsonl', temp_jsonl_file,
            '--out_pred', 'test_output.jsonl',
            '--limit', '2'
        ])
        
        # Configurar mocks
        mock_load.return_value = [
            {
                "pmid": "test_001",
                "text": "Test text",
                "entity_candidates": ["diabetes"],
                "line_num": 1
            }
        ]
        
        mock_process.return_value = {
            "PMID": "test_001",
            "Texto": "Test text",
            "Entidad": [{"texto": "diabetes", "tipo": "SpecificDisease"}],
            "_multi_strategy": sample_detection_results,
            "_latency_sec": 1.0
        }
        
        # Ejecutar main
        with patch('ner_app.main.parse_arguments') as mock_parse:
            mock_args = Mock()
            mock_args.input_jsonl = temp_jsonl_file
            mock_args.benchmark_jsonl = temp_jsonl_file
            mock_args.out_pred = 'test_output.jsonl'
            mock_args.limit = 2
            mock_args.confidence_threshold = 0.5
            mock_parse.return_value = mock_args
            
            with patch('ner_app.main.validate_arguments', return_value=True), \
                 patch('ner_app.main.configure_strategies', return_value=ALL_STRATEGIES), \
                 patch('ner_app.main.update_confidence_thresholds'), \
                 patch('ner_app.main.print_configuration'), \
                 patch('ner_app.main.ensure_temp_dir'), \
                 patch('ner_app.main.cleanup_temp_files'), \
                 patch('ner_app.main.print_summary'):
                
                result = main()
                
                assert result == 0  # Éxito
                mock_load.assert_called_once()
                mock_process.assert_called_once()
                mock_save.assert_called_once()
    
    @patch('sys.argv')
    @patch('ner_app.main.load_documents')
    @patch('ner_app.main.process_document')
    @patch('ner_app.main.save_results')
    def test_main_function_without_benchmark(
        self, mock_save, mock_process, mock_load, mock_argv, 
        temp_jsonl_file, sample_detection_results
    ):
        """Test función main sin benchmark (solo anotación)."""
        # Configurar argumentos simulados
        mock_argv.__getitem__ = Mock(side_effect=[
            'script.py',
            '--input_jsonl', temp_jsonl_file,
            '--out_pred', 'test_output.jsonl',
            '--limit', '1'
        ])
        
        # Configurar mocks
        mock_load.return_value = [
            {
                "pmid": "test_001",
                "text": "Test text",
                "entity_candidates": ["diabetes"],
                "line_num": 1
            }
        ]
        
        mock_process.return_value = {
            "PMID": "test_001",
            "Texto": "Test text",
            "Entidad": [{"texto": "diabetes", "tipo": "SpecificDisease"}],
            "_multi_strategy": sample_detection_results,
            "_latency_sec": 1.0
        }
        
        # Ejecutar main sin benchmark
        with patch('ner_app.main.parse_arguments') as mock_parse:
            mock_args = Mock()
            mock_args.input_jsonl = temp_jsonl_file
            mock_args.benchmark_jsonl = None  # Sin benchmark
            mock_args.out_pred = 'test_output.jsonl'
            mock_args.limit = 1
            mock_args.confidence_threshold = 0.5
            mock_parse.return_value = mock_args
            
            with patch('ner_app.main.validate_arguments', return_value=True), \
                 patch('ner_app.main.configure_strategies', return_value=ALL_STRATEGIES), \
                 patch('ner_app.main.update_confidence_thresholds'), \
                 patch('ner_app.main.print_configuration'), \
                 patch('ner_app.main.ensure_temp_dir'), \
                 patch('ner_app.main.cleanup_temp_files'), \
                 patch('ner_app.main.print_summary'):
                
                result = main()
                
                assert result == 0  # Éxito
                mock_load.assert_called_once()
                mock_process.assert_called_once()
                mock_save.assert_called_once()
    
    def test_error_handling(self, temp_jsonl_file):
        """Test manejo de errores en el sistema."""
        # Test con archivo inexistente
        documents = load_documents("nonexistent_file.jsonl", limit=10)
        assert len(documents) == 0  # Debe manejar el error gracefully
    
    def test_empty_input_handling(self):
        """Test manejo de entrada vacía."""
        # Crear archivo temporal vacío
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_file = f.name
        
        try:
            documents = load_documents(temp_file, limit=10)
            assert len(documents) == 0
        finally:
            os.remove(temp_file)
    
    def test_malformed_json_handling(self):
        """Test manejo de JSON malformado."""
        # Crear archivo con JSON malformado
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid json"}\n')
            temp_file = f.name
        
        try:
            documents = load_documents(temp_file, limit=10)
            # Debe cargar solo las líneas válidas
            assert len(documents) <= 2
        finally:
            os.remove(temp_file)
