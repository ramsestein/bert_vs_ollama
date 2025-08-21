#!/usr/bin/env python3
"""
Script para ejecutar todos los tests del sistema NER Multi-Strategy.

Proporciona opciones para ejecutar diferentes tipos de tests y generar reportes.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Agregar el directorio raíz al path para imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

def run_command(command, description=""):
    """Ejecuta un comando y retorna el resultado."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Ejecutando: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            cwd=ROOT_DIR,
            timeout=300  # 5 minutos timeout
        )
        
        if result.stdout:
            print(f"\n📊 STDOUT:\n{result.stdout}")
        
        if result.stderr and result.returncode != 0:
            print(f"\n❌ STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print(f"\n✅ {description} - ÉXITO")
        else:
            print(f"\n❌ {description} - FALLÓ (código: {result.returncode})")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} - TIMEOUT (>5 minutos)")
        return False
    except Exception as e:
        print(f"\n💥 {description} - ERROR: {e}")
        return False

def check_pytest():
    """Verifica que pytest esté instalado."""
    try:
        import pytest
        print(f"✅ pytest encontrado: versión {pytest.__version__}")
        return True
    except ImportError:
        print("❌ pytest no encontrado. Instalando...")
        return run_command([sys.executable, "-m", "pip", "install", "pytest"], "Instalando pytest")

def run_unit_tests(verbose=False):
    """Ejecuta tests unitarios."""
    cmd = [sys.executable, "-m", "pytest", "tests/unit/"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short", "-x"])  # Stop on first failure
    
    return run_command(cmd, "TESTS UNITARIOS")

def run_integration_tests(verbose=False):
    """Ejecuta tests de integración."""
    cmd = [sys.executable, "-m", "pytest", "tests/integration/"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short"])
    
    return run_command(cmd, "TESTS DE INTEGRACIÓN")

def run_all_tests(verbose=False):
    """Ejecuta todos los tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--tb=short"])
    
    return run_command(cmd, "TODOS LOS TESTS")

def run_coverage_tests():
    """Ejecuta tests con reporte de cobertura."""
    # Verificar que coverage esté instalado
    try:
        import coverage
        print(f"✅ coverage encontrado: versión {coverage.__version__}")
    except ImportError:
        print("❌ coverage no encontrado. Instalando...")
        if not run_command([sys.executable, "-m", "pip", "install", "coverage"], "Instalando coverage"):
            return False
    
    # Ejecutar tests con coverage
    cmd = [
        sys.executable, "-m", "coverage", "run", "--source=ner_app", 
        "-m", "pytest", "tests/", "--tb=short"
    ]
    
    success = run_command(cmd, "TESTS CON COBERTURA")
    
    if success:
        # Generar reporte de cobertura
        run_command([sys.executable, "-m", "coverage", "report"], "REPORTE DE COBERTURA")
        run_command([sys.executable, "-m", "coverage", "html"], "REPORTE HTML DE COBERTURA")
        print("\n📊 Reporte HTML generado en: htmlcov/index.html")
    
    return success

def test_script_help():
    """Prueba que los scripts muestren ayuda correctamente."""
    scripts = [
        "llama_ner_multi_strategy.py",
        "llama_ner_multi_strategy_refactored.py"
    ]
    
    all_success = True
    
    for script in scripts:
        success = run_command(
            [sys.executable, script, "--help"], 
            f"TEST AYUDA - {script}"
        )
        all_success = all_success and success
    
    return all_success

def validate_test_data():
    """Valida que los datos de prueba sean correctos."""
    test_files = [
        "tests/data/test_input.jsonl",
        "tests/data/test_benchmark.jsonl"
    ]
    
    print(f"\n{'='*60}")
    print("🔍 VALIDANDO DATOS DE PRUEBA")
    print(f"{'='*60}")
    
    all_valid = True
    
    for test_file in test_files:
        file_path = ROOT_DIR / test_file
        if not file_path.exists():
            print(f"❌ Archivo no encontrado: {test_file}")
            all_valid = False
            continue
        
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        json.loads(line)  # Validar JSON
                        line_count += 1
            
            print(f"✅ {test_file}: {line_count} líneas válidas")
            
        except json.JSONDecodeError as e:
            print(f"❌ {test_file}: Error JSON en línea {line_num}: {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {test_file}: Error: {e}")
            all_valid = False
    
    return all_valid

def main():
    parser = argparse.ArgumentParser(description="Ejecutor de tests para NER Multi-Strategy")
    parser.add_argument("--type", choices=["unit", "integration", "all", "coverage"], 
                       default="all", help="Tipo de tests a ejecutar")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Salida verbose")
    parser.add_argument("--validate-data", action="store_true",
                       help="Solo validar datos de prueba")
    parser.add_argument("--test-scripts", action="store_true",
                       help="Solo probar scripts principales")
    
    args = parser.parse_args()
    
    print("🧪 EJECUTOR DE TESTS - SISTEMA NER MULTI-STRATEGY")
    print(f"Directorio de trabajo: {ROOT_DIR}")
    
    # Validar prerequisitos
    if not check_pytest():
        print("❌ No se pudo instalar pytest")
        return 1
    
    # Validar datos de prueba
    if not validate_test_data():
        print("❌ Datos de prueba inválidos")
        if not args.validate_data:
            return 1
    
    if args.validate_data:
        print("✅ Validación de datos completada")
        return 0
    
    # Probar scripts principales
    if args.test_scripts:
        success = test_script_help()
        return 0 if success else 1
    
    # Ejecutar tests según el tipo
    success = False
    
    if args.type == "unit":
        success = run_unit_tests(args.verbose)
    elif args.type == "integration":
        success = run_integration_tests(args.verbose)
    elif args.type == "coverage":
        success = run_coverage_tests()
    else:  # all
        print("\n🔄 EJECUTANDO TODOS LOS TESTS...")
        unit_success = run_unit_tests(args.verbose)
        integration_success = run_integration_tests(args.verbose)
        script_success = test_script_help()
        
        success = unit_success and integration_success and script_success
        
        print(f"\n{'='*60}")
        print("📊 RESUMEN FINAL")
        print(f"{'='*60}")
        print(f"Tests unitarios: {'✅ ÉXITO' if unit_success else '❌ FALLÓ'}")
        print(f"Tests integración: {'✅ ÉXITO' if integration_success else '❌ FALLÓ'}")
        print(f"Tests scripts: {'✅ ÉXITO' if script_success else '❌ FALLÓ'}")
        print(f"RESULTADO GENERAL: {'✅ ÉXITO' if success else '❌ FALLÓ'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
