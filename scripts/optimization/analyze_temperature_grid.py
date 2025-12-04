#!/usr/bin/env python3
"""
Análisis de resultados del grid search con enfoque en temperaturas
Refactorizado para usar módulo ner_analysis compartido
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
    print_section,
    get_top_configurations,
    print_parameter_analysis
)


def main():
    parser = argparse.ArgumentParser(
        description="Analiza resultados del grid search con enfoque en temperaturas"
    )
    parser.add_argument(
        "results_dir",
        help="Directorio con resultados del grid search"
    )
    parser.add_argument(
        "--reference",
        help="Archivo JSONL de referencia para evaluación (opcional)",
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
    
    if args.reference and not os.path.exists(args.reference):
        print(f"[ERROR] El archivo de referencia {args.reference} no existe")
        sys.exit(1)
    
    print_section(f"ANALISIS DE RESULTADOS: {args.results_dir}", "=")
    
    # Buscar archivos de resultados
    pattern = os.path.join(args.results_dir, "results_*_*_chunk*_ov*_temp*.jsonl")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"[ERROR] No se encontraron archivos de resultados en {args.results_dir}")
        sys.exit(1)
    
    print(f"\n[FOUND] {len(result_files)} archivos de resultados")
    
    # Analizar cada archivo
    all_results = []
    
    for filepath in result_files:
        filename = os.path.basename(filepath)
        params = parse_filename(filename)
        
        if not params:
            print(f"[WARNING] No se pudieron parsear parámetros de: {filename}")
            continue
        
        print(f"[ANALYZING] {filename}")
        
        # Leer resultados
        results = read_results(filepath)
        if not results:
            continue
        
        # Evaluar rendimiento si hay referencia
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
        print(f"   [STATS] Entidades: {results['entities']}, Confianza: {results['avg_conf']:.3f}")
    
    if not all_results:
        print("[ERROR] No se pudieron analizar resultados válidos")
        return
    
    print(f"\n[SUCCESS] Analizados {len(all_results)} archivos correctamente")
    
    # Detectar parámetros presentes
    has_temperature = any('temperature' in r for r in all_results)
    has_confidence = any('confidence' in r for r in all_results)
    models = set(r['model'] for r in all_results)
    datasets = set(r['dataset'] for r in all_results)
    
    print_section("RESUMEN DE CONFIGURACIONES")
    print(f"Modelos: {', '.join(sorted(models))}")
    print(f"Datasets: {', '.join(sorted(datasets))}")
    
    # Análisis por temperatura
    if has_temperature:
        print_section("ANALISIS POR TEMPERATURA", "-")
        temp_aggregated = aggregate_by_parameter(all_results, 'temperature')
        print_parameter_analysis('temperature', 'Temperature', temp_aggregated)
    
    # Análisis por chunk
    print_section("ANALISIS POR CHUNK", "-")
    chunk_aggregated = aggregate_by_parameter(all_results, 'chunk')
    print_parameter_analysis('chunk', 'Chunk Target', chunk_aggregated)
    
    # Análisis por overlap
    print_section("ANALISIS POR OVERLAP", "-")
    overlap_aggregated = aggregate_by_parameter(all_results, 'overlap')
    print_parameter_analysis('overlap', 'Overlap', overlap_aggregated)
    
    # Análisis por confidence si existe
    if has_confidence:
        print_section("ANALISIS POR CONFIDENCE", "-")
        conf_aggregated = aggregate_by_parameter(all_results, 'confidence')
        print_parameter_analysis('confidence', 'Confidence Threshold', conf_aggregated)
    
    # Top configuraciones
    if args.reference:
        print_section(f"TOP {args.top} CONFIGURACIONES POR F1-SCORE")
        top_f1 = get_top_configurations(all_results, 'f1_score', args.top)
        
        if top_f1:
            print(f"\n{'#':<3} {'Modelo':<10} {'Dataset':<10} {'Chunk':<6} {'Overlap':<7} {'Temp':<6} {'Conf':<6} {'F1':<6} {'P':<6} {'R':<6}")
            print("-" * 90)
            
            for i, result in enumerate(top_f1, 1):
                temp_str = f"{result.get('temperature', 0):.1f}" if 'temperature' in result else "N/A"
                conf_str = f"{result.get('confidence', 0):.1f}" if 'confidence' in result else "N/A"
                
                print(f"{i:<3} {result['model']:<10} {result['dataset']:<10} "
                      f"{result['chunk']:<6} {result['overlap']:<7} "
                      f"{temp_str:<6} {conf_str:<6} "
                      f"{result['f1_score']:.3f}  {result.get('precision', 0):.3f}  {result.get('recall', 0):.3f}")
        else:
            print("\n[INFO] No hay métricas de F1 disponibles")
    
    # Top por entidades
    print_section(f"TOP {args.top} CONFIGURACIONES POR ENTIDADES DETECTADAS")
    top_entities = get_top_configurations(all_results, 'entities', args.top)
    
    print(f"\n{'#':<3} {'Modelo':<10} {'Dataset':<10} {'Chunk':<6} {'Overlap':<7} {'Temp':<6} {'Conf':<6} {'Ents':<6} {'AvgConf':<8}")
    print("-" * 90)
    
    for i, result in enumerate(top_entities, 1):
        temp_str = f"{result.get('temperature', 0):.1f}" if 'temperature' in result else "N/A"
        conf_str = f"{result.get('confidence', 0):.1f}" if 'confidence' in result else "N/A"
        
        print(f"{i:<3} {result['model']:<10} {result['dataset']:<10} "
              f"{result['chunk']:<6} {result['overlap']:<7} "
              f"{temp_str:<6} {conf_str:<6} "
              f"{result['entities']:<6} {result['avg_conf']:.3f}")
    
    # Guardar CSV
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        
        base_cols = ['filename', 'model', 'dataset', 'chunk', 'overlap']
        optional_cols = []
        if has_temperature:
            optional_cols.append('temperature')
        if has_confidence:
            optional_cols.append('confidence')
        
        metric_cols = ['precision', 'recall', 'f1_score', 'tp', 'fp', 'fn']
        other_cols = ['entities', 'avg_conf', 'min_conf', 'max_conf', 'docs']
        
        all_cols = base_cols + optional_cols + metric_cols + other_cols
        existing_cols = [col for col in all_cols if col in df.columns]
        
        df = df[existing_cols]
        csv_file = os.path.join(args.results_dir, "temperature_grid_analysis.csv")
        df.to_csv(csv_file, index=False)
        print_section("RESULTADOS GUARDADOS", "-")
        print(f"CSV: {csv_file}")
    except ImportError:
        print("\n[WARN] pandas no disponible, no se guardó CSV")


if __name__ == "__main__":
    main()
