#!/usr/bin/env python3
"""
Script para analizar falsos negativos del resultado limpio de n2c2
"""

import json
import os

def analyze_false_negatives(predictions_file, benchmark_file):
    """Analiza los falsos negativos comparando benchmark con predicciones"""
    print(f"=== ANÁLISIS DE FALSOS NEGATIVOS (100 documentos procesados) ===\n")
    
    # Cargar archivos
    print(f"[INFO] Cargando predicciones: {predictions_file}")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    
    print(f"[INFO] Cargando benchmark: {benchmark_file}")
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark = [json.loads(line) for line in f if line.strip()]
    
    print(f"[INFO] Predicciones cargadas: {len(predictions)}")
    print(f"[INFO] Benchmark cargado: {len(benchmark)}")
    
    # Obtener PMIDs de los documentos procesados
    processed_pmids = set()
    for doc in predictions:
        pmid = doc.get("PMID", "unknown")
        processed_pmids.add(pmid)
    
    print(f"[INFO] PMIDs procesados: {len(processed_pmids)}")
    
    # Crear diccionario de predicciones por PMID
    predictions_dict = {}
    for doc in predictions:
        pmid = doc.get("PMID", "unknown")
        entities = set()
        for ent in doc.get("Entidad", []):
            if isinstance(ent, dict) and "texto" in ent:
                entities.add(ent["texto"].lower().strip())
        predictions_dict[pmid] = entities
    
    # Filtrar benchmark para solo incluir documentos procesados
    filtered_benchmark = []
    for doc in benchmark:
        pmid = doc.get("PMID", "unknown")
        if pmid in processed_pmids:
            filtered_benchmark.append(doc)
    
    print(f"[INFO] Benchmark filtrado (solo PMIDs procesados): {len(filtered_benchmark)}")
    
    # Analizar falsos negativos
    false_negatives = []
    total_benchmark_entities = 0
    total_detected_entities = 0
    
    for doc in filtered_benchmark:
        pmid = doc.get("PMID", "unknown")
        benchmark_entities = set()
        
        # Obtener entidades del benchmark
        for ent in doc.get("Entidad", []):
            if isinstance(ent, dict) and "texto" in ent:
                benchmark_entities.add(ent["texto"].lower().strip())
        
        total_benchmark_entities += len(benchmark_entities)
        
        # Obtener entidades predichas para este PMID
        predicted_entities = predictions_dict.get(pmid, set())
        total_detected_entities += len(predicted_entities)
        
        # Identificar falsos negativos
        for benchmark_entity in benchmark_entities:
            if benchmark_entity not in predicted_entities:
                false_negatives.append({
                    "PMID": pmid,
                    "benchmark_entity": benchmark_entity,
                    "predicted_entities": list(predicted_entities),
                    "text": doc.get("Texto", "")[:200] + "..." if len(doc.get("Texto", "")) > 200 else doc.get("Texto", "")
                })
    
    # Mostrar resultados
    print(f"\n=== RESULTADOS (100 documentos procesados) ===\n")
    print(f"Total entidades en benchmark (filtrado): {total_benchmark_entities}")
    print(f"Total entidades detectadas: {total_detected_entities}")
    print(f"Total falsos negativos: {len(false_negatives)}")
    
    if false_negatives:
        print(f"\n=== FALSOS NEGATIVOS DETALLADOS ===\n")
        
        for i, fn in enumerate(false_negatives, 1):
            print(f"**Caso {i}:**")
            print(f"  PMID: {fn['PMID']}")
            print(f"  Entidad en benchmark: '{fn['benchmark_entity']}'")
            print(f"  Entidades predichas: {fn['predicted_entities']}")
            print(f"  Texto (primeros 200 chars): {fn['text']}")
            print()
        
        # Análisis por PMID
        pmid_fns = {}
        for fn in false_negatives:
            pmid = fn['PMID']
            if pmid not in pmid_fns:
                pmid_fns[pmid] = []
            pmid_fns[pmid].append(fn)
        
        print(f"=== FALSOS NEGATIVOS POR PMID ===\n")
        for pmid, fns in pmid_fns.items():
            print(f"PMID {pmid}: {len(fns)} entidades no detectadas")
            for fn in fns:
                print(f"  - '{fn['benchmark_entity']}'")
            print()
        
        # Guardar resultados en JSON
        output_file = "false_negatives_clean_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_benchmark_entities": total_benchmark_entities,
                    "total_detected_entities": total_detected_entities,
                    "total_false_negatives": len(false_negatives),
                    "pmids_processed": len(processed_pmids),
                    "pmids_with_benchmark": len(filtered_benchmark)
                },
                "false_negatives": false_negatives,
                "by_pmid": {pmid: len(fns) for pmid, fns in pmid_fns.items()}
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] Análisis guardado en: {output_file}")
        
    else:
        print(f"\n✅ ¡No se encontraron falsos negativos!")
        print(f"La máquina detectó todas las entidades del benchmark en los documentos procesados.")
    
    return false_negatives

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
    
    analyze_false_negatives(predictions_file, benchmark_file)

if __name__ == "__main__":
    main()
