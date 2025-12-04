#!/usr/bin/env python3
"""
Grid Search Unificado para Optimización de Parámetros NER

Permite ejecutar grid search de hiperparámetros para cualquier combinación de:
- Modelo (qwen, llama, gemma)
- Dataset (n2c2, ncbi)
- Parámetros (chunk_target, overlap, temperature)

Uso:
    python scripts/run_grid_search.py --model qwen --dataset n2c2
    python scripts/run_grid_search.py --model llama --dataset ncbi --limit 10
    python scripts/run_grid_search.py --model gemma --dataset n2c2 --custom-chunks 30,50,70
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
    "gemma": "gemma2:9b",
    "gemma3": "gemma2:9b",
    "llama": "llama3.2:3b"
}

# Valores por defecto para grid search
DEFAULT_CHUNK_TARGETS = [20, 40, 60, 80, 100]
DEFAULT_OVERLAPS = [10, 20, 30, 40, 50]
DEFAULT_TEMPERATURES = [0.5]  # Se puede extender para grid de temperatura


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Grid Search unificado para optimización de parámetros NER",
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
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Umbral de confianza (default: 0.3)"
    )
    
    parser.add_argument(
        "--custom-chunks",
        type=str,
        help="Lista personalizada de chunk_targets separados por comas (ej: 30,50,70)"
    )
    
    parser.add_argument(
        "--custom-overlaps",
        type=str,
        help="Lista personalizada de overlaps separados por comas (ej: 15,25,35)"
    )
    
    parser.add_argument(
        "--custom-temps",
        type=str,
        help="Lista personalizada de temperaturas separadas por comas (ej: 0.3,0.5,0.7)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directorio de salida personalizado (default: auto-generado)"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Omitir evaluación automática después de cada ejecución"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Archivo de log personalizado (default: grid_search_TIMESTAMP.log)"
    )
    
    return parser.parse_args()


def get_grid_parameters(args: argparse.Namespace) -> Tuple[List[int], List[int], List[float]]:
    """Obtener parámetros del grid search"""
    # Chunk targets
    if args.custom_chunks:
        chunk_targets = [int(x.strip()) for x in args.custom_chunks.split(',')]
    else:
        chunk_targets = DEFAULT_CHUNK_TARGETS
    
    # Overlaps
    if args.custom_overlaps:
        overlaps = [int(x.strip()) for x in args.custom_overlaps.split(',')]
    else:
        overlaps = DEFAULT_OVERLAPS
    
    # Temperatures
    if args.custom_temps:
        temperatures = [float(x.strip()) for x in args.custom_temps.split(',')]
    else:
        temperatures = DEFAULT_TEMPERATURES
    
    return chunk_targets, overlaps, temperatures


def generate_combinations(chunk_targets: List[int], overlaps: List[int], 
                         temperatures: List[float]) -> List[Tuple]:
    """Generar todas las combinaciones válidas de parámetros"""
    # Filtrar combinaciones donde overlap < chunk_target
    valid_combos = []
    for t, o, temp in itertools.product(chunk_targets, overlaps, temperatures):
        if o < t:
            valid_combos.append((t, o, temp))
    
    return valid_combos


def run_ner_command(model: str, dataset: str, chunk_target: int, overlap: int, 
                   temperature: float, output_path: str, args: argparse.Namespace, 
                   log_handle) -> Tuple[subprocess.CompletedProcess, float]:
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
        str(args.confidence_threshold),
        "--model",
        model.lower(),
        "--model-id",
        model_id,
        "--s1_target",
        str(chunk_target),
        "--s1_overlap",
        str(overlap),
        "--s1_temp",
        str(temperature),
    ]
    
    # Medir tiempo de ejecución
    start_time = time.time()
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
    execution_time = time.time() - start_time
    
    # Loggear output completo del procesamiento NER (incluye respuestas del LLM)
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
            if "Precisión:" in line:
                precision = line.split(":")[1].strip().split()[0]
                print(f"      📊 P={precision}")
                break
                
    except subprocess.CalledProcessError:
        print(f"      ⚠️  No se pudo evaluar rendimiento")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Configurar logging a archivo y terminal
    if args.log_file:
        log_file = args.log_file
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"grid_search_{args.model}_{args.dataset}_{timestamp}.log"
    
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
    chunk_targets, overlaps, temperatures = get_grid_parameters(args)
    
    # Generar combinaciones
    combos = generate_combinations(chunk_targets, overlaps, temperatures)
    
    # Crear directorio para resultados
    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = f"{args.model}_grid_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Banner inicial
    log_print(f"=" * 80)
    log_print(f"Grid Search {args.model.upper()} para {dataset_info['display_name']}")
    log_print(f"=" * 80)
    log_print(f"Resultados en: {results_dir}")
    log_print(f"Dataset: {dataset_info['display_name']}")
    log_print(f"Configuraciones: {len(combos)}")
    log_print(f"Documentos por ejecucion: {args.limit}")
    log_print(f"Umbral de confianza: {args.confidence_threshold}")
    log_print(f"Log guardado en: {log_file}")
    log_print("")
    log_print(f"Parametros del grid:")
    log_print(f"  - Chunk targets: {chunk_targets}")
    log_print(f"  - Overlaps: {overlaps}")
    log_print(f"  - Temperatures: {temperatures}")
    log_print(f"=" * 80)
    log_print("")

    successful_runs = 0
    failed_runs = 0
    execution_times = []  # Para estadísticas de tiempo

    for i, (t, o, temp) in enumerate(combos, 1):
        log_print(f"[{i:2d}/{len(combos)}] chunk={t:3d}, overlap={o:3d}, temp={temp:.2f}")
        
        out_file = f"results_{args.model}_{args.dataset}_chunk{t}_ov{o}_temp{temp:.2f}.jsonl"
        out_path = os.path.join(results_dir, out_file)
        
        log_print(f"   Ejecutando: {out_file}")
        
        # Tiempo de inicio para esta configuración
        config_start_time = time.time()
        
        try:
            result, ner_time = run_ner_command(args.model, args.dataset, t, o, temp, out_path, args, log_handle)
            log_print(f"   [OK] Exito: {out_file}")
            log_print(f"   [TIME] Procesamiento NER: {ner_time:.2f} segundos ({ner_time/60:.2f} minutos)")
            successful_runs += 1
            execution_times.append({
                "config": f"chunk={t}, overlap={o}, temp={temp}",
                "ner_time": ner_time
            })
            
            # Evaluar rendimiento si no se desactiva
            if not args.skip_evaluation:
                # Capturar y loggear la evaluación
                try:
                    # Usar rutas absolutas para evitar problemas
                    abs_out_path = os.path.abspath(out_path)
                    abs_ref_path = os.path.abspath(dataset_info["reference_file"])
                    
                    cmd = [
                        sys.executable,
                        "scripts/evaluate_ner_performance.py",
                        "--predictions",
                        abs_out_path,
                        "--reference",
                        abs_ref_path
                    ]
                    
                    eval_result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
                    eval_output = eval_result.stdout
                    
                    # Extraer y loggear métricas principales
                    metrics_found = False
                    for line in eval_output.split('\n'):
                        line_clean = line.strip()
                        # Capturar las 3 métricas principales (con diferentes variantes de encoding)
                        if (line_clean.startswith("Precisi") or  # Captura "Precisión" con cualquier variante
                            line_clean.startswith("Recall:") or 
                            line_clean.startswith("F1-Score:")):
                            log_print(f"      [METRICS] {line_clean}")
                            metrics_found = True
                    
                    if not metrics_found:
                        log_print(f"      [WARNING] No se pudieron extraer metricas del output")
                            
                except subprocess.CalledProcessError as eval_error:
                    log_print(f"      [WARNING] Error al evaluar rendimiento: {eval_error}")
                    if eval_error.stderr:
                        log_print(f"      [STDERR] {eval_error.stderr[:300]}")
                except Exception as eval_error:
                    log_print(f"      [WARNING] Excepcion al evaluar: {eval_error}")
            
            # Tiempo total para esta configuración (NER + evaluación)
            config_total_time = time.time() - config_start_time
            log_print(f"   [TIME] Tiempo total configuracion: {config_total_time:.2f} segundos ({config_total_time/60:.2f} minutos)")
            
        except subprocess.CalledProcessError as e:
            log_print(f"   [ERROR] Error: {e}")
            if e.stdout:
                log_print(f"   [OUTPUT] Salida: {e.stdout[:200]}")
            if e.stderr:
                log_print(f"   [STDERR] Error: {e.stderr[:200]}")
            failed_runs += 1
        
        log_print("")
    
    # Resumen final
    log_print("")
    log_print("=" * 80)
    log_print(f"GRID SEARCH {args.model.upper()} COMPLETADO")
    log_print("=" * 80)
    log_print(f"Ejecuciones exitosas: {successful_runs}")
    log_print(f"Ejecuciones fallidas: {failed_runs}")
    log_print(f"Resultados guardados en: {results_dir}")
    log_print(f"Log completo en: {log_file}")
    log_print("")
    
    # Estadísticas de tiempo
    if execution_times:
        log_print("ESTADISTICAS DE TIEMPO DE EJECUCION:")
        log_print("=" * 80)
        total_ner_time = sum(t["ner_time"] for t in execution_times)
        avg_ner_time = total_ner_time / len(execution_times)
        min_ner_time = min(t["ner_time"] for t in execution_times)
        max_ner_time = max(t["ner_time"] for t in execution_times)
        
        log_print(f"Tiempo total NER: {total_ner_time:.2f} segundos ({total_ner_time/60:.2f} minutos)")
        log_print(f"Tiempo promedio por configuracion: {avg_ner_time:.2f} segundos ({avg_ner_time/60:.2f} minutos)")
        log_print(f"Tiempo minimo: {min_ner_time:.2f} segundos")
        log_print(f"Tiempo maximo: {max_ner_time:.2f} segundos")
        log_print("")
        log_print("Detalle por configuracion:")
        for t_info in execution_times:
            log_print(f"  {t_info['config']:40} -> {t_info['ner_time']:.2f}s ({t_info['ner_time']/60:.2f}m)")
        log_print("")
    
    log_print("Para analizar resultados:")
    log_print(f"   python scripts/analyze_grid.py {results_dir}")
    
    # Cerrar archivo de log
    log_handle.close()


if __name__ == "__main__":
    main()
