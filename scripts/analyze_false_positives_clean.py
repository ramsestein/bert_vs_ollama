#!/usr/bin/env python3
"""
Script para analizar falsos positivos del resultado limpio de n2c2
"""

import json
import os

def analyze_false_positives(predictions_file, benchmark_file):
    """Analiza los falsos positivos comparando predicciones con benchmark"""
    print(f"=== ANÁLISIS DE FALSOS POSITIVOS ===\n")
    
    # Cargar archivos
    print(f"[INFO] Cargando predicciones: {predictions_file}")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    
    print(f"[INFO] Cargando benchmark: {benchmark_file}")
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark = [json.loads(line) for line in f if line.strip()]
    
    print(f"[INFO] Predicciones cargadas: {len(predictions)}")
    print(f"[INFO] Benchmark cargado: {len(benchmark)}")
    
    # Crear diccionario de benchmark por PMID
    benchmark_dict = {}
    for doc in benchmark:
        pmid = doc.get("PMID", "unknown")
        entities = set()
        for ent in doc.get("Entidad", []):
            if isinstance(ent, dict) and "texto" in ent:
                entities.add(ent["texto"].lower().strip())
        benchmark_dict[pmid] = entities
    
    # Analizar falsos positivos
    false_positives = []
    total_predictions = 0
    total_benchmark = 0
    
    for pred_doc in predictions:
        pmid = pred_doc.get("PMID", "unknown")
        predicted_entities = set()
        
        # Obtener entidades predichas
        for ent in pred_doc.get("Entidad", []):
            if isinstance(ent, dict) and "texto" in ent:
                predicted_entities.add(ent["texto"].lower().strip())
        
        total_predictions += len(predicted_entities)
        
        # Obtener entidades del benchmark
        benchmark_entities = benchmark_dict.get(pmid, set())
        total_benchmark += len(benchmark_entities)
        
        # Identificar falsos positivos
        for pred_entity in predicted_entities:
            if pred_entity not in benchmark_entities:
                false_positives.append({
                    "PMID": pmid,
                    "predicted_entity": pred_entity,
                    "benchmark_entities": list(benchmark_entities),
                    "confidence": next((ent.get("confidence", 0.0) for ent in pred_doc.get("Entidad", []) 
                                      if ent.get("texto", "").lower().strip() == pred_entity), 0.0),
                    "strategies": next((ent.get("strategies", []) for ent in pred_doc.get("Entidad", []) 
                                      if ent.get("texto", "").lower().strip() == pred_entity), [])
                })
    
    # Mostrar resultados
    print(f"\n=== RESULTADOS ===\n")
    print(f"Total entidades predichas: {total_predictions}")
    print(f"Total entidades en benchmark: {total_benchmark}")
    print(f"Total falsos positivos: {len(false_positives)}")
    
    if false_positives:
        print(f"\n=== FALSOS POSITIVOS DETALLADOS ===\n")
        
        for i, fp in enumerate(false_positives, 1):
            print(f"**Caso {i}:**")
            print(f"  PMID: {fp['PMID']}")
            print(f"  Entidad predicha: '{fp['predicted_entity']}'")
            print(f"  Confianza: {fp['confidence']:.3f}")
            print(f"  Estrategias: {', '.join(fp['strategies'])}")
            print(f"  Entidades en benchmark: {fp['benchmark_entities']}")
            print()
        
        # Análisis por estrategia
        strategy_fps = {}
        for fp in false_positives:
            for strategy in fp['strategies']:
                if strategy not in strategy_fps:
                    strategy_fps[strategy] = []
                strategy_fps[strategy].append(fp)
        
        print(f"=== ANÁLISIS POR ESTRATEGIA ===\n")
        for strategy, fps in strategy_fps.items():
            print(f"{strategy}: {len(fps)} falsos positivos")
        
        # Guardar resultados en JSON
        output_file = "false_positives_clean_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_predictions": total_predictions,
                    "total_benchmark": total_benchmark,
                    "total_false_positives": len(false_positives)
                },
                "false_positives": false_positives,
                "by_strategy": {strategy: len(fps) for strategy, fps in strategy_fps.items()}
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] Análisis guardado en: {output_file}")
        
    else:
        print(f"\n✅ ¡No se encontraron falsos positivos!")
    
    return false_positives

def main():
    """Función principal"""
    predictions_file = "results_n2c2_clean_100.jsonl"
    benchmark_file = "datasets/n2c2_test.jsonl"
    
    if not os.path.exists(predictions_file):
        print(f"[ERROR] Archivo de predicciones no encontrado: {predictions_file}")
        return
    
    if not os.path.exists(benchmark_file):
        print(f"[ERROR] Archivo de benchmark no encontrado: {benchmark_file}")
        return
    
    analyze_false_positives(predictions_file, benchmark_file)

if __name__ == "__main__":
    main()
