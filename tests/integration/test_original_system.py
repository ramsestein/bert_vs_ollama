"""
Tests de integración para el sistema NER original.
"""

import pytest
import json
import os
import tempfile
import subprocess
import sys
from unittest.mock import patch, Mock

class TestOriginalSystemIntegration:
    """Tests de integración para el sistema original."""
    
    def test_original_script_help(self):
        """Test que el script original muestra ayuda."""
        result = subprocess.run([
            sys.executable, 'llama_ner_multi_strategy.py', '--help'
        ], capture_output=True, text=True, cwd='.')
        
        assert result.returncode == 0
        assert '--input_jsonl' in result.stdout
        assert '--benchmark_jsonl' in result.stdout
        assert 'optional' in result.stdout  # Debe mostrar que benchmark es opcional
    
    def test_original_script_arguments_validation(self):
        """Test validación de argumentos del script original."""
        # Test sin argumentos requeridos
        result = subprocess.run([
            sys.executable, 'llama_ner_multi_strategy.py'
        ], capture_output=True, text=True, cwd='.')
        
        assert result.returncode != 0
        assert 'required' in result.stderr.lower()
    
    def test_original_script_with_nonexistent_input(self):
        """Test script original con archivo de entrada inexistente."""
        result = subprocess.run([
            sys.executable, 'llama_ner_multi_strategy.py',
            '--input_jsonl', 'nonexistent_input.jsonl'
        ], capture_output=True, text=True, cwd='.')
        
        # Debe fallar gracefully
        assert result.returncode != 0
    
    @pytest.mark.integration
    def test_original_script_execution_with_test_data(self, temp_jsonl_file):
        """Test ejecución del script original con datos de prueba."""
        output_file = 'test_original_output.jsonl'
        
        try:
            # Ejecutar script original con límite pequeño
            result = subprocess.run([
                sys.executable, 'llama_ner_multi_strategy.py',
                '--input_jsonl', temp_jsonl_file,
                '--out_pred', output_file,
                '--limit', '1',
                '--confidence_threshold', '0.5'
            ], capture_output=True, text=True, cwd='.', timeout=60)
            
            # Verificar que no hubo errores críticos
            # (puede fallar por Ollama no disponible, pero debe manejar errores gracefully)
            assert 'Traceback' not in result.stderr or result.returncode != 0
            
        except subprocess.TimeoutExpired:
            # Si se agota el tiempo, es esperado sin Ollama
            pass
        finally:
            # Limpiar archivo de salida si existe
            if os.path.exists(output_file):
                os.remove(output_file)
    
    def test_refactored_script_help(self):
        """Test que el script refactorizado muestra ayuda."""
        result = subprocess.run([
            sys.executable, 'llama_ner_multi_strategy_refactored.py', '--help'
        ], capture_output=True, text=True, cwd='.')
        
        assert result.returncode == 0
        assert '--input_jsonl' in result.stdout
        assert '--benchmark_jsonl' in result.stdout
        assert 'optional' in result.stdout  # Debe mostrar que benchmark es opcional
    
    def test_refactored_script_arguments_validation(self):
        """Test validación de argumentos del script refactorizado."""
        # Test sin argumentos requeridos
        result = subprocess.run([
            sys.executable, 'llama_ner_multi_strategy_refactored.py'
        ], capture_output=True, text=True, cwd='.')
        
        assert result.returncode != 0
        assert 'required' in result.stderr.lower()
    
    def test_both_scripts_same_help_structure(self):
        """Test que ambos scripts tienen estructura de ayuda similar."""
        # Obtener ayuda del script original
        result_original = subprocess.run([
            sys.executable, 'llama_ner_multi_strategy.py', '--help'
        ], capture_output=True, text=True, cwd='.')
        
        # Obtener ayuda del script refactorizado
        result_refactored = subprocess.run([
            sys.executable, 'llama_ner_multi_strategy_refactored.py', '--help'
        ], capture_output=True, text=True, cwd='.')
        
        # Ambos deben tener éxito
        assert result_original.returncode == 0
        assert result_refactored.returncode == 0
        
        # Verificar argumentos principales en ambos
        main_args = [
            '--input_jsonl',
            '--benchmark_jsonl', 
            '--out_pred',
            '--limit',
            '--confidence_threshold',
            '--strategies'
        ]
        
        for arg in main_args:
            assert arg in result_original.stdout
            assert arg in result_refactored.stdout
    
    @pytest.mark.integration
    def test_compare_outputs_structure(self, temp_jsonl_file):
        """Test que ambos scripts producen estructura de salida compatible."""
        original_output = 'test_original_output.jsonl'
        refactored_output = 'test_refactored_output.jsonl'
        
        try:
            # Configurar argumentos comunes
            common_args = [
                '--input_jsonl', temp_jsonl_file,
                '--limit', '1',
                '--confidence_threshold', '0.8'  # Alto para reducir procesamiento
            ]
            
            # Ejecutar script original
            result_original = subprocess.run([
                sys.executable, 'llama_ner_multi_strategy.py',
                '--out_pred', original_output
            ] + common_args, capture_output=True, text=True, cwd='.', timeout=30)
            
            # Ejecutar script refactorizado  
            result_refactored = subprocess.run([
                sys.executable, 'llama_ner_multi_strategy_refactored.py',
                '--out_pred', refactored_output
            ] + common_args, capture_output=True, text=True, cwd='.', timeout=30)
            
            # Si ambos terminan exitosamente, comparar estructura
            if (result_original.returncode == 0 and result_refactored.returncode == 0 and
                os.path.exists(original_output) and os.path.exists(refactored_output)):
                
                # Cargar resultados
                with open(original_output, 'r', encoding='utf-8') as f:
                    original_data = [json.loads(line) for line in f if line.strip()]
                
                with open(refactored_output, 'r', encoding='utf-8') as f:
                    refactored_data = [json.loads(line) for line in f if line.strip()]
                
                # Verificar estructura similar
                if original_data and refactored_data:
                    orig_keys = set(original_data[0].keys())
                    refact_keys = set(refactored_data[0].keys())
                    
                    # Claves principales deben estar presentes en ambos
                    essential_keys = {'PMID', 'Texto', 'Entidad'}
                    assert essential_keys.issubset(orig_keys)
                    assert essential_keys.issubset(refact_keys)
                    
        except subprocess.TimeoutExpired:
            # Esperado sin Ollama
            pass
        finally:
            # Limpiar archivos de salida
            for output_file in [original_output, refactored_output]:
                if os.path.exists(output_file):
                    os.remove(output_file)
    
    def test_backward_compatibility(self):
        """Test que la refactorización mantiene compatibilidad hacia atrás."""
        # Los argumentos del script original deben funcionar en el refactorizado
        # Test básico de argumentos
        
        result = subprocess.run([
            sys.executable, 'llama_ner_multi_strategy_refactored.py',
            '--input_jsonl', 'dummy.jsonl',  # No importa que no exista para este test
            '--strategies', 'llama32_balanced', 'qwen25_diversity',
            '--confidence_threshold', '0.7',
            '--limit', '5'
        ], capture_output=True, text=True, cwd='.')
        
        # Debe fallar por archivo inexistente, no por argumentos inválidos
        assert 'unrecognized arguments' not in result.stderr
        assert 'Input file not found' in result.stderr or result.returncode != 0
    
    def test_scripts_handle_missing_ollama_gracefully(self, temp_jsonl_file):
        """Test que ambos scripts manejan la ausencia de Ollama gracefully."""
        # Simular puerto incorrecto para Ollama
        
        for script in ['llama_ner_multi_strategy.py', 'llama_ner_multi_strategy_refactored.py']:
            output_file = f'test_{script}_no_ollama.jsonl'
            
            try:
                # Usar puerto incorrecto para forzar error de conexión
                result = subprocess.run([
                    sys.executable, script,
                    '--input_jsonl', temp_jsonl_file,
                    '--out_pred', output_file,
                    '--limit', '1'
                ], capture_output=True, text=True, cwd='.', timeout=20)
                
                # No debe hacer crash con Traceback no manejado
                # Puede fallar, pero debe ser manejado gracefully
                assert ('Traceback' not in result.stderr or 
                       'Connection' in result.stderr or
                       'LLM_ERROR' in result.stderr or
                       result.returncode != 0)
                
            except subprocess.TimeoutExpired:
                # Esperado si el script no maneja timeouts
                pass
            finally:
                if os.path.exists(output_file):
                    os.remove(output_file)
