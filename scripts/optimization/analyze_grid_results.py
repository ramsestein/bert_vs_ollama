#!/usr/bin/env python3
"""
Análisis comparativo general de resultados de grid search
Funciona con cualquier dataset y modelo, analiza chunks, overlaps, temperaturas y confidence
"""

import argparse
import glob
import os
import sys

# Importar módulo compartido desde el directorio padre
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ner_analysis import (
    parse_filename,
    read_results,
    evaluate_performance,
    aggregate_by_parameter,
    print_section
)


def analyze_by_parameter(results, param_name, display_name):
    """Analiza resultados agrupados por un parámetro específico"""
    aggregated = aggregate_by_parameter(results, param_name)
    
    if not aggregated:
        return
    
    print(f"\n[ANALISIS POR {display_name.upper()}]")
    for value in sorted(aggregated.keys()):
        data = aggregated[value]
        avg_entities = sum(data['entities']) / len(data['entities'])
        avg_conf = sum(data['avg_conf']) / len(data['avg_conf'])
        avg_precision = sum(data['precision']) / len(data['precision']) if data['precision'] else 0
        avg_recall = sum(data['recall']) / len(data['recall']) if data['recall'] else 0
        avg_f1 = sum(data['f1_score']) / len(data['f1_score']) if data['f1_score'] else 0
        
        print(f"  {param_name}={value:>6}: entidades={avg_entities:6.2f}, conf={avg_conf:5.3f}, P={avg_precision:5.3f}, R={avg_recall:5.3f}, F1={avg_f1:5.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="Análisis comparativo general de resultados de grid search"
    )
    parser.add_argument(
        "results_dir",
        help="Directorio con los archivos de resultados"
    )
    parser.add_argument(
        "--reference",
        help="Archivo de referencia para evaluar métricas (opcional)",
        default=None
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Número de configuraciones top a mostrar (default: 10)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"[ERROR] El directorio {args.results_dir} no existe")
        sys.exit(1)
    
    print_section(f"ANALISIS DE RESULTADOS: {args.results_dir}", "=")
    
    # Buscar archivos de resultados
    pattern = os.path.join(args.results_dir, "results_*.jsonl")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"[ERROR] No se encontraron archivos de resultados en {args.results_dir}")
        sys.exit(1)
    
    print(f"\n[INFO] Encontrados {len(result_files)} archivos de resultados")
    
    # Analizar cada archivo
    all_results = []
    
    for filepath in result_files:
        filename = os.path.basename(filepath)
        params = parse_filename(filename)
        
        if not params:
            print(f"[WARN] No se pudieron parsear parámetros de: {filename}")
            continue
        
        print(f"[PROCESSING] {filename}")
        
        # Leer resultados
        results = read_results(filepath)
        if not results:
            continue
        
        # Evaluar rendimiento si hay archivo de referencia
        metrics = {}
        if args.reference:
            metrics = evaluate_performance(filepath, args.reference)
        
        # Combinar información
        result_data = {
            'filename': filename,
            'docs': results['docs'],
            'entities': results['entities'],
            'avg_conf': results['avg_conf'],
            'min_conf': results['min_conf'],
            'max_conf': results['max_conf'],
            **params,
            **metrics
        }
        
        all_results.append(result_data)
        
        # Mostrar métricas básicas
        if metrics:
            print(f"   [METRICS] P={metrics.get('precision', 0):.3f}, R={metrics.get('recall', 0):.3f}, F1={metrics.get('f1_score', 0):.3f}")
        print(f"   [INFO] Entidades: {results['entities']}, Confianza: {results['avg_conf']:.3f}")
    
    if not all_results:
        print("[ERROR] No se pudieron analizar resultados válidos")
        return
    
    print(f"\n[SUCCESS] Analizados {len(all_results)} archivos correctamente")
    
    # Detectar qué parámetros están presentes
    has_temperature = any('temperature' in r for r in all_results)
    has_confidence = any('confidence' in r for r in all_results)
    models = set(r['model'] for r in all_results)
    datasets = set(r['dataset'] for r in all_results)
    
    print_section("RESUMEN DE CONFIGURACIONES")
    print(f"Modelos: {', '.join(sorted(models))}")
    print(f"Datasets: {', '.join(sorted(datasets))}")
    print(f"Parámetros variables: chunk, overlap", end="")
    if has_temperature:
        print(", temperature", end="")
    if has_confidence:
        print(", confidence", end="")
    print()
    
    # Análisis por modelo y dataset
    for model in sorted(models):
        for dataset in sorted(datasets):
            filtered = [r for r in all_results if r['model'] == model and r['dataset'] == dataset]
            if not filtered:
                continue
            
            print_section(f"MODELO: {model.upper()} | DATASET: {dataset.upper()}", "-")
            
            # Análisis por chunk
            analyze_by_parameter(filtered, 'chunk', 'Chunk Target')
            
            # Análisis por overlap
            analyze_by_parameter(filtered, 'overlap', 'Overlap')
            
            # Análisis por temperatura (si existe)
            if has_temperature:
                analyze_by_parameter(filtered, 'temperature', 'Temperature')
            
            # Análisis por confidence (si existe)
            if has_confidence:
                analyze_by_parameter(filtered, 'confidence', 'Confidence Threshold')
    
    # Top configuraciones por F1-Score
    print_section(f"TOP {args.top} CONFIGURACIONES POR F1-SCORE")
    valid_results = [r for r in all_results if 'f1_score' in r and r['f1_score'] > 0]
    
    if valid_results:
        top_f1 = sorted(valid_results, key=lambda x: x['f1_score'], reverse=True)[:args.top]
        print(f"\n{'#':<3} {'Modelo':<10} {'Dataset':<10} {'Chunk':<6} {'Overlap':<7} {'Temp':<6} {'Conf':<6} {'F1':<6} {'P':<6} {'R':<6}")
        print("-" * 90)
        
        for i, result in enumerate(top_f1, 1):
            temp_str = f"{result['temperature']:.1f}" if 'temperature' in result else "N/A"
            conf_str = f"{result['confidence']:.1f}" if 'confidence' in result else "N/A"
            
            print(f"{i:<3} {result['model']:<10} {result['dataset']:<10} "
                  f"{result['chunk']:<6} {result['overlap']:<7} "
                  f"{temp_str:<6} {conf_str:<6} "
                  f"{result['f1_score']:.3f}  {result.get('precision', 0):.3f}  {result.get('recall', 0):.3f}")
    else:
        print("\n[INFO] No hay métricas de F1 disponibles (use --reference para calcularlas)")
    
    # Top configuraciones por entidades detectadas
    print_section(f"TOP {args.top} CONFIGURACIONES POR ENTIDADES DETECTADAS")
    top_entities = sorted(all_results, key=lambda x: x['entities'], reverse=True)[:args.top]
    
    print(f"\n{'#':<3} {'Modelo':<10} {'Dataset':<10} {'Chunk':<6} {'Overlap':<7} {'Temp':<6} {'Conf':<6} {'Ents':<6} {'AvgConf':<8}")
    print("-" * 90)
    
    for i, result in enumerate(top_entities, 1):
        temp_str = f"{result['temperature']:.1f}" if 'temperature' in result else "N/A"
        conf_str = f"{result['confidence']:.1f}" if 'confidence' in result else "N/A"
        
        print(f"{i:<3} {result['model']:<10} {result['dataset']:<10} "
              f"{result['chunk']:<6} {result['overlap']:<7} "
              f"{temp_str:<6} {conf_str:<6} "
              f"{result['entities']:<6} {result['avg_conf']:.3f}")
    
    # Guardar resultados en CSV
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        
        # Ordenar columnas de forma lógica
        base_cols = ['filename', 'model', 'dataset', 'chunk', 'overlap']
        optional_cols = []
        if has_temperature:
            optional_cols.append('temperature')
        if has_confidence:
            optional_cols.append('confidence')
        
        metric_cols = ['precision', 'recall', 'f1_score', 'tp', 'fp', 'fn']
        other_cols = ['entities', 'avg_conf', 'min_conf', 'max_conf', 'docs']
        
        # Filtrar solo columnas que existen
        all_cols = base_cols + optional_cols + metric_cols + other_cols
        existing_cols = [col for col in all_cols if col in df.columns]
        
        df = df[existing_cols]
        csv_file = os.path.join(args.results_dir, "grid_search_analysis.csv")
        df.to_csv(csv_file, index=False)
        print_section(f"RESULTADOS GUARDADOS", "-")
        print(f"CSV: {csv_file}")
    except ImportError:
        print("\n[WARN] pandas no disponible, no se guardó CSV")

if __name__ == "__main__":
    main()
