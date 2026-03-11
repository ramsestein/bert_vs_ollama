#!/usr/bin/env python3
"""
Script para analizar falsos negativos comparando predicciones con benchmark
Versión basada en códigos ICD10 en lugar de coincidencias exactas de texto
"""

import json
import os
import re
import argparse
from collections import defaultdict
from typing import Dict, Set

# Diccionario de entidades: mapea textos a códigos ICD10
ENTITIES = {
    "I10": [  # Hipertensión arterial
        "hta",
        "hipertensión arterial",
        "hipertensión"
    ],
    
    "E78.5": [  # Dislipemia
        "dislipemia",
        "dlp"
    ],
    
    "Z87.891": [  # Exfumador
        "exfumador",
        "ex-fumador"
    ],
    
    "E11.9": [  # Diabetes mellitus tipo 2
        "dm2",
        "diabetes mellitus tipo 2",
        "diabetes mellitus",
        "dm"
    ],
    
    "F17.210": [  # Fumador
        "fumador",
        "tabaquismo"
    ],
    
    "Z79.01": [  # Anticoagulado
        "anticoagulado",
        "anticoagulante",
        "sintrom"
    ],
    
    "I25.10": [  # Cardiopatía isquémica
        "cardiopatía isquémica",
        "enfermedad coronaria",
        "eac"
    ],
    
    "Z79.82": [  # AAS
        "aas",
        "aspirina",
        "adiro"
    ],
    
    "N17.9": [  # Insuficiencia renal aguda
        "insuficiencia renal aguda",
        "ira",
        "aki"
    ],
    
    "I48.91": [  # Fibrilación auricular
        "fibrilación auricular",
        "fa",
        "acxfa"
    ]
}

# Nombres descriptivos para códigos ICD10
ICD10_NAMES = {
    'I10': 'Hipertensión arterial',
    'E78.5': 'Dislipemia',
    #'Z87.891': 'Exfumador',
    'E11.9': 'Diabetes mellitus tipo 2',
    #'F17.210': 'Fumador',
    'Z79.01': 'Anticoagulado',
    'I25.10': 'Cardiopatía isquémica',
    'Z79.82': 'AAS',
    'N17.9': 'Insuficiencia renal aguda',
    'I48.91': 'Fibrilación auricular'
}

def normalize_text(text: str) -> str:
    """Normaliza texto para comparación consistente"""
    if not text:
        return ""
    # Convertir a minúsculas y normalizar espacios
    text = re.sub(r'\s+', ' ', text.lower().strip())
    # Normalizar caracteres especiales
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'–|—', '-', text)
    return text

def build_text_to_icd10_map() -> Dict[str, str]:
    """Construye un mapa de texto normalizado -> código ICD10"""
    text_to_code = {}
    for icd10_code, variants in ENTITIES.items():
        for variant in variants:
            normalized = normalize_text(variant)
            text_to_code[normalized] = icd10_code
    return text_to_code

def map_text_to_icd10(text: str, text_to_code_map: Dict[str, str]) -> str:
    """Mapea un texto a su código ICD10, retorna vacío si no se encuentra"""
    normalized = normalize_text(text)
    return text_to_code_map.get(normalized, "")

def analyze_false_negatives_icd10(predictions_file, benchmark_file, output_file="false_negatives_analysis_icd10.json"):
    """Analiza los falsos negativos comparando por código ICD10"""
    print(f"=== ANÁLISIS DE FALSOS NEGATIVOS (ICD10) ===\n")
    
    # Construir mapa de texto -> ICD10
    text_to_code_map = build_text_to_icd10_map()
    print(f"[INFO] Mapa de texto a ICD10: {len(text_to_code_map)} variantes")
    
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
    
    # Crear diccionario de predicciones por PMID (conjunto de códigos ICD10)
    predictions_dict = {}
    for doc in predictions:
        pmid = doc.get("PMID", "unknown")
        icd10_codes = set()
        
        for ent in doc.get("Entidad", []):
            if not isinstance(ent, dict):
                continue
            
            # Solo considerar entidades extraídas por regex o con código explícito
            strategies = ent.get("strategies", []) or []
            if "codigo" not in ent and "regex" not in strategies:
                continue
            
            # Obtener código
            code = ""
            if "codigo" in ent:
                code = ent.get("codigo", "").strip()
            elif "texto" in ent:
                texto = ent.get("texto", "")
                code = map_text_to_icd10(texto, text_to_code_map)
            
            if code:
                icd10_codes.add(code)
        
        predictions_dict[pmid] = icd10_codes
    
    # Filtrar benchmark para solo incluir documentos procesados
    filtered_benchmark = []
    for doc in benchmark:
        pmid = doc.get("PMID", "unknown")
        if pmid in processed_pmids:
            filtered_benchmark.append(doc)
    
    print(f"[INFO] Benchmark filtrado (solo PMIDs procesados): {len(filtered_benchmark)}")
    
    # Analizar falsos negativos
    false_negatives = []
    total_benchmark_codes = 0
    total_detected_codes = 0
    
    for doc in filtered_benchmark:
        pmid = doc.get("PMID", "unknown")
        benchmark_codes = set()
        benchmark_entities_by_code = defaultdict(list)
        
        # Obtener códigos del benchmark
        for ent in doc.get("Entidad", []):
            if isinstance(ent, dict):
                # Priorizar campo "codigo" si existe
                if "codigo" in ent:
                    code = ent.get("codigo", "").strip()
                    if code:
                        benchmark_codes.add(code)
                        benchmark_entities_by_code[code].append(ent.get("texto", ""))
                # Si no, mapear desde "texto"
                elif "texto" in ent:
                    texto = ent.get("texto", "")
                    code = map_text_to_icd10(texto, text_to_code_map)
                    if code:
                        benchmark_codes.add(code)
                        benchmark_entities_by_code[code].append(texto)
        
        total_benchmark_codes += len(benchmark_codes)
        
        # Obtener códigos predichos para este PMID
        predicted_codes = predictions_dict.get(pmid, set())
        total_detected_codes += len(predicted_codes)
        
        # Identificar falsos negativos (en benchmark pero no predichos)
        fn_codes = benchmark_codes - predicted_codes
        
        for code in fn_codes:
            entities_in_benchmark = benchmark_entities_by_code[code]
            false_negatives.append({
                "PMID": pmid,
                "benchmark_code": code,
                "benchmark_code_name": ICD10_NAMES.get(code, "Desconocido"),
                "benchmark_entities": entities_in_benchmark,
                "predicted_codes": sorted(list(predicted_codes)),
                "predicted_names": [ICD10_NAMES.get(c, "Desconocido") for c in sorted(predicted_codes)],
                "text_preview": doc.get("Texto", "")[:200] + "..." if len(doc.get("Texto", "")) > 200 else doc.get("Texto", "")
            })
    
    # Mostrar resultados
    print(f"\n=== RESULTADOS ({len(processed_pmids)} documentos procesados) ===\n")
    print(f"Total códigos en benchmark (filtrado): {total_benchmark_codes}")
    print(f"Total códigos detectados: {total_detected_codes}")
    print(f"Total falsos negativos: {len(false_negatives)}")
    
    if false_negatives:
        print(f"\n=== FALSOS NEGATIVOS DETALLADOS ===\n")
        
        for i, fn in enumerate(false_negatives, 1):
            print(f"**Caso {i}:**")
            print(f"  PMID: {fn['PMID']}")
            print(f"  Código en benchmark: {fn['benchmark_code']} ({fn['benchmark_code_name']})")
            print(f"  Entidades en benchmark: {fn['benchmark_entities']}")
            print(f"  Códigos predichos: {fn['predicted_codes']}")
            print(f"  Nombres predichos: {fn['predicted_names']}")
            print(f"  Texto (primeros 200 chars): {fn['text_preview']}")
            print()
        
        # Análisis por código ICD10
        code_fns = defaultdict(list)
        for fn in false_negatives:
            code_fns[fn['benchmark_code']].append(fn)
        
        print(f"\n=== FALSOS NEGATIVOS POR CÓDIGO ICD10 ===\n")
        for code in sorted(code_fns.keys()):
            fns = code_fns[code]
            name = ICD10_NAMES.get(code, "Desconocido")
            print(f"{code} ({name}): {len(fns)} no detectados")
        
        # Análisis por PMID
        pmid_fns = defaultdict(list)
        for fn in false_negatives:
            pmid_fns[fn['PMID']].append(fn)
        
        print(f"\n=== FALSOS NEGATIVOS POR PMID ===\n")
        for pmid in sorted(pmid_fns.keys()):
            fns = pmid_fns[pmid]
            print(f"PMID {pmid}: {len(fns)} códigos no detectados")
            for fn in fns:
                print(f"  - {fn['benchmark_code']} ({fn['benchmark_code_name']})")
            print()
        
        # Guardar resultados en JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_benchmark_codes": total_benchmark_codes,
                    "total_detected_codes": total_detected_codes,
                    "total_false_negatives": len(false_negatives),
                    "pmids_processed": len(processed_pmids),
                    "pmids_with_benchmark": len(filtered_benchmark)
                },
                "false_negatives": false_negatives,
                "by_code": {code: len(fns) for code, fns in code_fns.items()},
                "by_pmid": {pmid: len(fns) for pmid, fns in pmid_fns.items()}
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] Análisis guardado en: {output_file}")
        
    else:
        print(f"\n✅ ¡No se encontraron falsos negativos!")
        print(f"La máquina detectó todos los códigos ICD10 del benchmark en los documentos procesados.")
    
    return false_negatives

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Analiza falsos negativos comparando predicciones con benchmark (basado en códigos ICD10)"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Archivo JSONL con predicciones del modelo"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Archivo JSONL con entidades de referencia (ground truth)"
    )
    parser.add_argument(
        "--output",
        default="false_negatives_analysis_icd10.json",
        help="Archivo de salida para el análisis (default: false_negatives_analysis_icd10.json)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions):
        print(f"[ERROR] Archivo de predicciones no encontrado: {args.predictions}")
        return
    
    if not os.path.exists(args.benchmark):
        print(f"[ERROR] Archivo de benchmark no encontrado: {args.benchmark}")
        return
    
    analyze_false_negatives_icd10(args.predictions, args.benchmark, args.output)

if __name__ == "__main__":
    main()
