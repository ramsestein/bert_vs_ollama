#!/usr/bin/env python3
"""
Script para analizar falsos positivos comparando predicciones con benchmark
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

def analyze_false_positives_icd10(predictions_file, benchmark_file, output_file="false_positives_analysis_icd10.json"):
    """Analiza los falsos positivos comparando por código ICD10"""
    print(f"=== ANÁLISIS DE FALSOS POSITIVOS (ICD10) ===\n")
    
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
    
    # Crear diccionario de benchmark por PMID (conjunto de códigos ICD10)
    benchmark_dict = {}
    for doc in benchmark:
        pmid = doc.get("PMID", "unknown")
        icd10_codes = set()
        for ent in doc.get("Entidad", []):
            if isinstance(ent, dict):
                # Priorizar campo "codigo" si existe
                if "codigo" in ent:
                    code = ent.get("codigo", "").strip()
                    if code:
                        icd10_codes.add(code)
                # Si no, mapear desde "texto"
                elif "texto" in ent:
                    texto = ent.get("texto", "")
                    code = map_text_to_icd10(texto, text_to_code_map)
                    if code:
                        icd10_codes.add(code)
        benchmark_dict[pmid] = icd10_codes
    
    # === ANÁLISIS: Lista detallada de entidades + agrupación por documento ===
    false_positives = []
    fp_by_document = []
    total_fp_codes = 0  # Total de códigos FP (agrupados por documento)
    fp_by_code_global = defaultdict(int)  # Contador global por código
    total_predictions = 0
    total_benchmark = 0
    unmapped_predictions = defaultdict(int)
    
    for pred_doc in predictions:
        pmid = pred_doc.get("PMID", "unknown")
        predicted_codes = set()
        predicted_entities_by_code = defaultdict(list)
        
        # Obtener códigos ICD10 predichos
        for ent in pred_doc.get("Entidad", []):
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
                if not code:
                    unmapped_predictions[texto] += 1
            
            if code:
                predicted_codes.add(code)
                predicted_entities_by_code[code].append({
                    "texto": ent.get("texto", ""),
                    "confidence": ent.get("confidence", 0.0),
                    "strategies": ent.get("strategies", [])
                })
        
        total_predictions += len(predicted_codes)
        
        # Obtener códigos del benchmark
        benchmark_codes = benchmark_dict.get(pmid, set())
        total_benchmark += len(benchmark_codes)
        
        # Identificar falsos positivos (predichos pero no en benchmark)
        fp_codes = predicted_codes - benchmark_codes
        
        # === LISTA DETALLADA: Añadir cada entidad textual como FP ===
        for code in fp_codes:
            entities_info = predicted_entities_by_code[code]
            for ent_info in entities_info:
                false_positives.append({
                    "PMID": pmid,
                    "predicted_code": code,
                    "predicted_code_name": ICD10_NAMES.get(code, "Desconocido"),
                    "predicted_entity": ent_info["texto"],
                    "benchmark_codes": sorted(list(benchmark_codes)),
                    "benchmark_names": [ICD10_NAMES.get(c, "Desconocido") for c in sorted(benchmark_codes)],
                    "confidence": ent_info["confidence"],
                    "strategies": ent_info["strategies"]
                })
        
        # === AGRUPACIÓN POR DOCUMENTO: Contar códigos únicos FP ===
        num_fp = len(fp_codes)
        total_fp_codes += num_fp
        
        for code in fp_codes:
            fp_by_code_global[code] += 1
        
        if num_fp > 0:
            fp_by_document.append({
                "PMID": pmid,
                "fp_count": num_fp,
                "fp_codes": sorted(list(fp_codes)),
                "fp_codes_names": [ICD10_NAMES.get(c, "Desconocido") for c in sorted(fp_codes)],
                "predicted_codes": sorted(list(predicted_codes)),
                "benchmark_codes": sorted(list(benchmark_codes))
            })
    
    # Mostrar resultados
    print(f"\n=== RESULTADOS ===\n")
    print(f"Total códigos predichos: {total_predictions}")
    print(f"Total códigos en benchmark: {total_benchmark}")
    print(f"Total falsos positivos (entidades individuales): {len(false_positives)}")
    print(f"Total falsos positivos (códigos agrupados): {total_fp_codes}")
    print(f"Total entidades sin mapeo: {sum(unmapped_predictions.values())}")
    
    if unmapped_predictions:
        print(f"\n=== ENTIDADES SIN MAPEO A ICD10 ===\n")
        for text, count in sorted(unmapped_predictions.items(), key=lambda x: -x[1]):
            print(f"  '{text}': {count} veces")
    
    if false_positives:
        print(f"\n=== FALSOS POSITIVOS DETALLADOS ===\n")
        
        for i, fp in enumerate(false_positives[:10], 1):  # Mostrar primeros 10
            print(f"**Caso {i}:**")
            print(f"  PMID: {fp['PMID']}")
            print(f"  Código predicho: {fp['predicted_code']} ({fp['predicted_code_name']})")
            print(f"  Entidad predicha: '{fp['predicted_entity']}'")
            print(f"  Confianza: {fp['confidence']:.3f}")
            print(f"  Estrategias: {', '.join(fp['strategies'])}")
            print(f"  Códigos en benchmark: {fp['benchmark_codes']}")
            print(f"  Nombres benchmark: {fp['benchmark_names']}")
            print()
        
        if len(false_positives) > 10:
            print(f"... y {len(false_positives) - 10} casos más\n")
        
        # Análisis por código ICD10
        code_fps = defaultdict(list)
        for fp in false_positives:
            code_fps[fp['predicted_code']].append(fp)
        
        print(f"\n=== ANÁLISIS POR CÓDIGO ICD10 ===\n")
        for code in sorted(code_fps.keys()):
            fps = code_fps[code]
            name = ICD10_NAMES.get(code, "Desconocido")
            print(f"{code} ({name}): {len(fps)} entidades FP")
        
        # Análisis por estrategia
        strategy_fps = defaultdict(list)
        for fp in false_positives:
            for strategy in fp['strategies']:
                strategy_fps[strategy].append(fp)
        
        print(f"\n=== ANÁLISIS POR ESTRATEGIA ===\n")
        for strategy in sorted(strategy_fps.keys()):
            fps = strategy_fps[strategy]
            print(f"{strategy}: {len(fps)} entidades FP")
        
        print(f"\n=== AGRUPACIÓN POR DOCUMENTO ===")
        print(f"(Cuenta cada código ICD10 una vez por documento)\n")
        print(f"Total documentos con FP: {len(fp_by_document)}")
        print(f"Total FP (códigos agrupados): {total_fp_codes}")
        print(f"\nDesglose por código:")
        for code in sorted(fp_by_code_global.keys()):
            count = fp_by_code_global[code]
            name = ICD10_NAMES.get(code, "Desconocido")
            print(f"  {code} ({name}): {count} documentos")
        
        # Guardar resultados en JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_predictions": total_predictions,
                    "total_benchmark": total_benchmark,
                    "total_false_positives": len(false_positives),
                    "total_false_positives_grouped": total_fp_codes,
                    "total_unmapped": sum(unmapped_predictions.values())
                },
                "false_positives": false_positives,
                "by_code": {code: len(fps) for code, fps in code_fps.items()},
                "by_strategy": {strategy: len(fps) for strategy, fps in strategy_fps.items()},
                "by_document": fp_by_document,
                "by_code_grouped": dict(fp_by_code_global),
                "unmapped_predictions": dict(unmapped_predictions)
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] Análisis guardado en: {output_file}")
        print(f"\n[VALIDACIÓN] 'total_false_positives_grouped' ({total_fp_codes}) debe coincidir")
        print(f"             con 'overall.fp' en ner_evaluation_results")
        
    else:
        print(f"\n✅ ¡No se encontraron falsos positivos!")
    
    return false_positives

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Analiza falsos positivos comparando predicciones con benchmark (basado en códigos ICD10)"
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
        default="false_positives_analysis_icd10.json",
        help="Archivo de salida para el análisis (default: false_positives_analysis_icd10.json)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions):
        print(f"[ERROR] Archivo de predicciones no encontrado: {args.predictions}")
        return
    
    if not os.path.exists(args.benchmark):
        print(f"[ERROR] Archivo de benchmark no encontrado: {args.benchmark}")
        return
    
    analyze_false_positives_icd10(args.predictions, args.benchmark, args.output)

if __name__ == "__main__":
    main()