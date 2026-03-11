#!/usr/bin/env python3
"""
NER Performance Evaluator - ICD10 Based
Calcula precisión, recall y F1 comparando por código ICD10 en lugar de texto exacto
Mapea automáticamente textos predichos a códigos ICD10 usando el diccionario de entidades
"""

import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple

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

def evaluate_ner_performance_icd10(predictions_file: str, reference_file: str, require_all_targets: bool = False) -> Dict:
    """Evalúa el rendimiento NER comparando por código ICD10"""
    
    print(f"[NER EVAL ICD10] Evaluando: {predictions_file}")
    print(f"[NER EVAL ICD10] Referencia: {reference_file}")
    
    # Construir mapa de texto -> ICD10
    text_to_code_map = build_text_to_icd10_map()
    print(f"[INFO] Mapa de texto a ICD10: {len(text_to_code_map)} variantes")
    
    # Cargar predicciones
    predictions = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    print(f"[LOADED] Predicciones: {len(predictions)} documentos")
    
    # Cargar referencias
    references = []
    with open(reference_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                references.append(json.loads(line))
    
    print(f"[LOADED] Referencias: {len(references)} documentos")
    
    # Crear diccionario de referencias por PMID (conjunto de códigos ICD10)
    ref_by_pmid = {}
    for ref in references:
        pmid = str(ref.get("PMID", ""))
        if pmid:
            icd10_codes = set()
            for ent in ref.get("Entidad", []):
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
            ref_by_pmid[pmid] = icd10_codes
    
    # Si se solicita, filtrar referencias para mantener solo documentos que
    # contengan todos los códigos objetivo (evita evaluar documentos incompletos)
    if require_all_targets:
        target_codes = set(ENTITIES.keys())
        before = len(ref_by_pmid)
        ref_by_pmid = {pmid: codes for pmid, codes in ref_by_pmid.items() if codes.issuperset(target_codes)}
        after = len(ref_by_pmid)
        print(f"[INFO] Filtrado por documentos que contienen todos los códigos target: {before} -> {after}")

    print(f"[INDEXED] Referencias por PMID: {len(ref_by_pmid)} documentos")
    
    # Evaluar cada predicción
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    
    detailed_results = []
    
    # Contadores por código ICD10
    icd10_tp = defaultdict(int)
    icd10_fp = defaultdict(int)
    icd10_fn = defaultdict(int)
    
    # Contadores de entidades sin mapeo
    unmapped_predictions = defaultdict(int)
    
    for pred in predictions:
        pmid = str(pred.get("PMID", ""))
        
        # Extraer códigos ICD10 predichos (mapeando desde texto)
        predicted_codes = set()
        predicted_entities_by_code = defaultdict(list)
        unmapped_in_doc = []
        
        for ent in pred.get("Entidad", []):
            if not isinstance(ent, dict):
                continue

            # Only consider entities that were extracted by the regex step
            # (the cleaned input) or that already carry an explicit `codigo`.
            # This avoids counting model-only detections (e.g. 'fa') as FP when
            # the input was cleaned by regex and shouldn't contain them.
            strategies = ent.get("strategies", []) or []
            if "codigo" not in ent and "regex" not in strategies:
                # skip model-only detections
                continue

            # Intentar obtener código directamente
            code = ""
            if "codigo" in ent:
                code = ent.get("codigo", "").strip()
            # Si no, mapear desde texto
            elif "texto" in ent:
                texto = ent.get("texto", "")
                code = map_text_to_icd10(texto, text_to_code_map)
                if not code:
                    unmapped_in_doc.append(texto)
                    unmapped_predictions[texto] += 1

            if code:
                predicted_codes.add(code)
                predicted_entities_by_code[code].append(ent.get("texto", ""))
        
        # Obtener códigos ICD10 de referencia
        reference_codes = ref_by_pmid.get(pmid, set())
        
        if not reference_codes:
            print(f"[WARNING] No se encontraron referencias para PMID {pmid}")
            # Si hay predicciones pero no referencias, contar como FP
            fp = len(predicted_codes)
            total_fp += fp
            for code in predicted_codes:
                icd10_fp[code] += 1
            
            doc_result = {
                "pmid": pmid,
                "predicted_codes": list(predicted_codes),
                "reference_codes": list(reference_codes),
                "unmapped_predictions": unmapped_in_doc,
                "tp": 0,
                "fp": fp,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0
            }
            detailed_results.append(doc_result)
            continue
        
        # Calcular métricas para este documento
        tp_codes = predicted_codes.intersection(reference_codes)
        fp_codes = predicted_codes - reference_codes
        fn_codes = reference_codes - predicted_codes
        
        tp = len(tp_codes)
        fp = len(fp_codes)
        fn = len(fn_codes)
        
        # Actualizar contadores por código
        for code in tp_codes:
            icd10_tp[code] += 1
        for code in fp_codes:
            icd10_fp[code] += 1
        for code in fn_codes:
            icd10_fn[code] += 1
        
        # Acumular totales
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Guardar resultados detallados
        doc_result = {
            "pmid": pmid,
            "predicted_codes": sorted(list(predicted_codes)),
            "reference_codes": sorted(list(reference_codes)),
            "tp_codes": sorted(list(tp_codes)),
            "fp_codes": sorted(list(fp_codes)),
            "fn_codes": sorted(list(fn_codes)),
            "unmapped_predictions": unmapped_in_doc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "predicted_entities_by_code": {k: v for k, v in predicted_entities_by_code.items()}
        }
        detailed_results.append(doc_result)
        
        print(f"[DOC] PMID {pmid}: TP={tp}, FP={fp}, FN={fn}, P={doc_result['precision']:.3f}, R={doc_result['recall']:.3f}")
        if unmapped_in_doc:
            print(f"      Unmapped: {unmapped_in_doc}")
    
    # Calcular métricas globales
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calcular métricas por código ICD10
    icd10_metrics = {}
    for code in sorted(set(list(icd10_tp.keys()) + list(icd10_fp.keys()) + list(icd10_fn.keys()))):
        tp = icd10_tp[code]
        fp = icd10_fp[code]
        fn = icd10_fn[code]
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        icd10_metrics[code] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    # Análisis por estrategia
    strategy_analysis = defaultdict(lambda: {"tp": 0, "fp": 0})
    
    for pred in predictions:
        pmid = str(pred.get("PMID", ""))
        reference_codes = ref_by_pmid.get(pmid, set())
        
        if not reference_codes:
            continue
        
        # Analizar cada estrategia
        for ent in pred.get("Entidad", []):
            if isinstance(ent, dict) and "strategies" in ent:
                strategies = ent.get("strategies", [])
                
                # Obtener código (desde campo o mapeado)
                code = ""
                if "codigo" in ent:
                    code = ent.get("codigo", "").strip()
                elif "texto" in ent:
                    texto = ent.get("texto", "")
                    code = map_text_to_icd10(texto, text_to_code_map)
                
                if not code:
                    continue
                
                # Verificar si es TP o FP
                is_tp = code in reference_codes
                
                for strategy in strategies:
                    if is_tp:
                        strategy_analysis[strategy]["tp"] += 1
                    else:
                        strategy_analysis[strategy]["fp"] += 1
    
    # Calcular métricas por estrategia
    strategy_metrics = {}
    for strategy, counts in strategy_analysis.items():
        tp = counts["tp"]
        fp = counts["fp"]
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        strategy_metrics[strategy] = {
            "precision": p,
            "tp": tp,
            "fp": fp
        }
    
    # Resultados finales
    results = {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn
        },
        "icd10_metrics": icd10_metrics,
        "strategy_metrics": strategy_metrics,
        "unmapped_predictions": dict(unmapped_predictions),
        "detailed_results": detailed_results,
        "summary": {
            "total_documents": len(detailed_results),
            "total_predicted_codes": sum(len(r["predicted_codes"]) for r in detailed_results),
            "total_reference_codes": sum(len(r["reference_codes"]) for r in detailed_results),
            "unique_predicted_codes": len(set(code for r in detailed_results for code in r["predicted_codes"])),
            "unique_reference_codes": len(set(code for r in detailed_results for code in r["reference_codes"])),
            "total_unmapped": sum(unmapped_predictions.values())
        }
    }
    
    return results

def print_results(results: Dict):
    """Imprime los resultados de la evaluación"""
    
    print("\n" + "="*70)
    print("RESULTADOS DE EVALUACION NER - BASADO EN CODIGOS ICD10")
    print("="*70)
    
    overall = results["overall"]
    summary = results["summary"]
    
    print(f"\nMETRICAS GLOBALES:")
    print(f"   Precisión: {overall['precision']:.3f} ({overall['precision']*100:.1f}%)")
    print(f"   Recall:    {overall['recall']:.3f} ({overall['recall']*100:.1f}%)")
    print(f"   F1-Score:  {overall['f1']:.3f} ({overall['f1']*100:.1f}%)")
    
    print(f"\nCONTEO DE CODIGOS ICD10:")
    print(f"   True Positives (TP):  {overall['tp']}")
    print(f"   False Positives (FP): {overall['fp']}")
    print(f"   False Negatives (FN): {overall['fn']}")
    
    print(f"\nRESUMEN DEL DATASET:")
    print(f"   Documentos procesados: {summary['total_documents']}")
    print(f"   Códigos predichos (total): {summary['total_predicted_codes']}")
    print(f"   Códigos de referencia (total): {summary['total_reference_codes']}")
    print(f"   Códigos únicos predichos: {summary['unique_predicted_codes']}")
    print(f"   Códigos únicos en referencia: {summary['unique_reference_codes']}")
    
    # Nombres descriptivos para códigos ICD10
    icd10_names = {
        'I10': 'Hipertensión arterial',
        'E78.5': 'Dislipemia',
        'Z87.891': 'Exfumador',
        'E11.9': 'Diabetes mellitus tipo 2',
        'F17.210': 'Fumador',
        'Z79.01': 'Anticoagulado',
        'I25.10': 'Cardiopatía isquémica',
        'Z79.82': 'AAS',
        'N17.9': 'Insuficiencia renal aguda',
        'I48.91': 'Fibrilación auricular'
    }
    
    print(f"\nRENDIMIENTO POR CODIGO ICD10:")
    print(f"{'Código':<10} {'Diagnóstico':<30} {'P':>6} {'R':>6} {'F1':>6} | {'TP':>3} {'FP':>3} {'FN':>3}")
    print("-" * 80)
    
    for code in sorted(results["icd10_metrics"].keys()):
        metrics = results["icd10_metrics"][code]
        name = icd10_names.get(code, 'Desconocido')
        p = metrics["precision"]
        r = metrics["recall"]
        f = metrics["f1"]
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        
        print(f"{code:<10} {name:<30} {p:>6.3f} {r:>6.3f} {f:>6.3f} | {tp:>3} {fp:>3} {fn:>3}")
    
    print(f"\nRENDIMIENTO POR ESTRATEGIA:")
    for strategy, metrics in results["strategy_metrics"].items():
        p = metrics["precision"]
        tp = metrics["tp"]
        fp = metrics["fp"]
        print(f"   {strategy:30}: P={p:.3f} ({p*100:.1f}%) | TP={tp}, FP={fp}")
    
    print(f"\nANALISIS DETALLADO (primeros 5 documentos):")
    for i, doc in enumerate(results["detailed_results"][:5]):
        print(f"\n   {i+1}. PMID {doc['pmid']}: P={doc['precision']:.3f}, R={doc['recall']:.3f}")
        print(f"      Predichos: {doc['predicted_codes']}")
        print(f"      Referencia: {doc['reference_codes']}")
        if doc['tp_codes']:
            print(f"      ✓ TP: {doc['tp_codes']}")
        if doc['fp_codes']:
            print(f"      ✗ FP: {doc['fp_codes']}")
        if doc['fn_codes']:
            print(f"      ✗ FN: {doc['fn_codes']}")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluador de rendimiento NER basado en códigos ICD10")
    parser.add_argument("--predictions", required=True, help="Archivo de predicciones JSONL")
    parser.add_argument("--reference", required=True, help="Archivo de referencia JSONL")
    parser.add_argument("--output", default="ner_evaluation_results_icd10.json", 
                       help="Archivo de salida para resultados JSON")
    parser.add_argument("--require-all-targets", action="store_true",
                        help="Evaluar sólo documentos cuya referencia contiene todos los códigos target")
    
    args = parser.parse_args()
    
    try:
        # Evaluar rendimiento
        results = evaluate_ner_performance_icd10(args.predictions, args.reference, args.require_all_targets)
        
        # Imprimir resultados
        print_results(results)
        
        # Guardar resultados en archivo
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[SAVED] Resultados guardados en: {args.output}")
        
    except Exception as e:
        print(f"[ERROR] Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
