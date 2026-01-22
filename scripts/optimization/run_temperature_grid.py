#!/usr/bin/env python3
"""
Grid Search de Temperaturas y Confidence Thresholds

Ejecuta grid search optimizando temperatura y confidence threshold
manteniendo chunks y overlaps fijos.

Uso:
    python scripts/run_temperature_grid.py --model qwen --dataset n2c2
    python scripts/run_temperature_grid.py --model gemma --dataset ncbi --chunk 60 --overlap 20
"""

import argparse
import itertools
import subprocess
import sys
import os
import time
from datetime import datetime
from typing import List, Tuple

# Configuración de datasets
DATASET_CONFIG = {
    "n2c2": {
        "input_file": "datasets/n2c2_test_input.jsonl",
        "reference_file": "datasets/n2c2_test.jsonl",
        "display_name": "n2c2_test"
    },
    "ncbi": {
        "input_file": "datasets/ncbi_develop_input.jsonl",
        "reference_file": "datasets/ncbi_develop.jsonl",
        "display_name": "ncbi_develop"
    }
}

# Mapeo de nombres de modelos a IDs de Ollama
MODEL_ID_MAP = {
    "qwen": "qwen2.5:3b",
    "gemma": "gemma3:4b",
    "gemma3": "gemma3:4b",
    "llama": "llama3.2:3b"
}

# Valores por defecto para temperaturas y thresholds
DEFAULT_TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_CONFIDENCE = [0.1, 0.2, 0.3, 0.4, 0.5]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Grid Search de temperaturas y confidence thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_ID_MAP.keys()),
        help="Modelo a evaluar (qwen, gemma, gemma3, llama)"
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIG.keys()),
        help="Dataset a utilizar (n2c2, ncbi)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Número de documentos a procesar (default: 20)"
    )
    
    parser.add_argument(
        "--chunk",
        type=int,
        default=60,
        help="Tamaño de chunk fijo (default: 60)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=20,
        help="Overlap fijo (default: 20)"
    )
    
    parser.add_argument(
        "--custom-temps",
        type=str,
        help="Temperaturas personalizadas separadas por comas (ej: 0.3,0.5,0.7)"
    )
    
    parser.add_argument(
        "--custom-confidence",
        type=str,
        help="Confidence thresholds personalizados separados por comas (ej: 0.2,0.3,0.4)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directorio personalizado para resultados"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Archivo de log personalizado"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Saltar evaluación de métricas (más rápido)"
    )
    
    return parser.parse_args()


def get_grid_parameters(args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    """Obtiene los parámetros del grid search"""
    # Temperaturas
    if args.custom_temps:
        temperatures = [float(t.strip()) for t in args.custom_temps.split(',')]
    else:
        temperatures = DEFAULT_TEMPERATURES
    
    # Confidence thresholds
    if args.custom_confidence:
        confidence_thresholds = [float(c.strip()) for c in args.custom_confidence.split(',')]
    else:
        confidence_thresholds = DEFAULT_CONFIDENCE
    
    return temperatures, confidence_thresholds


def run_ner_command(model: str, dataset: str, chunk: int, overlap: int,
                   temperature: float, confidence: float, output_path: str, 
                   args: argparse.Namespace, log_handle) -> Tuple[subprocess.CompletedProcess, float]:
    """Ejecutar comando NER con los parámetros especificados
    
    Returns:
        Tuple[CompletedProcess, float]: Resultado del comando y tiempo de ejecución en segundos
    """
    
    dataset_info = DATASET_CONFIG[dataset]
    model_id = MODEL_ID_MAP.get(model.lower(), model)
    
    cmd = [
        sys.executable,
        "-m",
        "ner_app.main",
        "--input_jsonl",
        dataset_info["input_file"],
        "--benchmark_jsonl",
        dataset_info["reference_file"],
        "--out_pred",
        output_path,
        "--limit",
        str(args.limit),
        "--confidence_threshold",
        str(confidence),
        "--model",
        model.lower(),
        "--model-id",
        model_id,
        "--s1_target",
        str(chunk),
        "--s1_overlap",
        str(overlap),
        "--s1_temp",
        str(temperature),
    ]
    
    # Medir tiempo de ejecución
    start_time = time.time()
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    execution_time = time.time() - start_time
    
    # Loggear output completo del procesamiento NER
    if result.stdout:
        log_handle.write("\n" + "="*80 + "\n")
        log_handle.write("OUTPUT COMPLETO DEL PROCESAMIENTO NER:\n")
        log_handle.write("="*80 + "\n")
        log_handle.write(result.stdout)
        log_handle.write("\n" + "="*80 + "\n")
        log_handle.flush()
    
    return result, execution_time


def evaluate_performance(result_file: str, reference_file: str) -> None:
    """Evalúa el rendimiento de una configuración específica"""
    try:
        cmd = [
            sys.executable,
            "scripts/evaluate_ner_performance.py",
            "--predictions",
            result_file,
            "--reference",
            reference_file
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        
        # Extraer métricas clave
        for line in output.split('\n'):
            if "Precisión:" in line or "Precision:" in line:
                precision = line.split(":")[1].strip().split()[0]
                print(f"      [METRICS] P={precision}")
                break
                
    except subprocess.CalledProcessError:
        print(f"      [WARNING] No se pudo evaluar rendimiento")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Configurar logging a archivo y terminal
    if args.log_file:
        log_file = args.log_file
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"temp_grid_{args.model}_{args.dataset}_{timestamp}.log"
    
    # Abrir archivo de log
    log_handle = open(log_file, 'w', encoding='utf-8')
    
    def log_print(message):
        """Imprime en terminal y escribe en log"""
        print(message)
        log_handle.write(message + '\n')
        log_handle.flush()
    
    # Obtener configuración
    dataset_info = DATASET_CONFIG[args.dataset]
    model_id = MODEL_ID_MAP.get(args.model.lower(), args.model)
    
    # Obtener parámetros del grid
    temperatures, confidence_thresholds = get_grid_parameters(args)
    
    # Generar combinaciones
    combos = list(itertools.product(temperatures, confidence_thresholds))
    
    # Crear directorio para resultados
    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = f"{args.model}_temp_grid_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Banner inicial
    log_print(f"=" * 80)
    log_print(f"Temperature Grid Search {args.model.upper()} para {dataset_info['display_name']}")
    log_print(f"=" * 80)
    log_print(f"Resultados en: {results_dir}")
    log_print(f"Dataset: {dataset_info['display_name']}")
    log_print(f"Configuraciones: {len(combos)}")
    log_print(f"Documentos por ejecucion: {args.limit}")
    log_print(f"Chunk fijo: {args.chunk}, Overlap fijo: {args.overlap}")
    log_print(f"Log guardado en: {log_file}")
    log_print("")
    log_print(f"Parametros del grid:")
    log_print(f"  - Temperaturas: {temperatures}")
    log_print(f"  - Confidence thresholds: {confidence_thresholds}")
    log_print(f"=" * 80)
    log_print("")
    
    # Ejecutar cada combinación
    successful_runs = 0
    failed_runs = 0
    execution_times = []
    
    for i, (temp, conf) in enumerate(combos, 1):
        log_print(f"[{i:3d}/{len(combos)}] temp={temp:4.2f}, conf={conf:4.2f}")
        
        # Nombre del archivo de salida
        out_file = f"results_{args.model}_{args.dataset}_chunk{args.chunk}_ov{args.overlap}_temp{temp:.2f}_conf{conf:.2f}.jsonl"
        out_path = os.path.join(results_dir, out_file)
        
        log_print(f"   Ejecutando: {out_file}")
        
        try:
            result, exec_time = run_ner_command(
                args.model, args.dataset, args.chunk, args.overlap,
                temp, conf, out_path, args, log_handle
            )
            
            log_print(f"   [OK] Exito: {out_file}")
            log_print(f"   [TIME] Procesamiento NER: {exec_time:.2f} segundos ({exec_time/60:.2f} minutos)")
            
            execution_times.append(exec_time)
            successful_runs += 1
            
            # Evaluar rendimiento si no se saltea
            if not args.skip_evaluation:
                evaluate_performance(out_path, dataset_info["reference_file"])
                
        except subprocess.CalledProcessError as e:
            log_print(f"   [ERROR] Error: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                log_print(f"   [OUTPUT] Salida: {e.stdout[:500]}")
            failed_runs += 1
        
        log_print(f"   [TIME] Tiempo total configuracion: {exec_time:.2f} segundos ({exec_time/60:.2f} minutos)")
        log_print("")
    
    # Resumen final
    log_print("")
    log_print("=" * 80)
    log_print(f"TEMPERATURE GRID SEARCH {args.model.upper()} COMPLETADO")
    log_print("=" * 80)
    log_print(f"Ejecuciones exitosas: {successful_runs}")
    log_print(f"Ejecuciones fallidas: {failed_runs}")
    log_print(f"Resultados guardados en: {results_dir}")
    log_print(f"Log completo en: {log_file}")
    log_print("")
    
    # Estadísticas de tiempo
    if execution_times:
        total_time = sum(execution_times)
        avg_time = total_time / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        log_print("ESTADISTICAS DE TIEMPO DE EJECUCION:")
        log_print("=" * 80)
        log_print(f"Tiempo total NER: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
        log_print(f"Tiempo promedio por configuracion: {avg_time:.2f} segundos ({avg_time/60:.2f} minutos)")
        log_print(f"Tiempo minimo: {min_time:.2f} segundos")
        log_print(f"Tiempo maximo: {max_time:.2f} segundos")
        log_print("")
    
    log_print("Para analizar resultados:")
    log_print(f"   python scripts/analyze_temperature_grid.py {results_dir} --reference {dataset_info['reference_file']}")
    
    log_handle.close()


if __name__ == "__main__":
    main()
