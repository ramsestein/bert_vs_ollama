#!/usr/bin/env python3
"""
NER Performance Evaluator
Calcula precisión, recall y F1 para el sistema llama_ner_multi_strategy
"""

import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple

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

def fuzzy_match(predicted: str, reference: str, threshold: float = 0.8) -> bool:
    """Matching fuzzy entre entidad predicha y referencia"""
    pred_norm = normalize_text(predicted)
    ref_norm = normalize_text(reference)
    
    if pred_norm == ref_norm:
        return True
    
    # Verificar si una está contenida en la otra
    if pred_norm in ref_norm or ref_norm in pred_norm:
        return True
    
    # Calcular similitud de caracteres
    pred_chars = set(pred_norm)
    ref_chars = set(ref_norm)
    
    if not pred_chars or not ref_chars:
        return False
    
    intersection = len(pred_chars.intersection(ref_chars))
    union = len(pred_chars.union(ref_chars))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    return similarity >= threshold

def evaluate_ner_performance(predictions_file: str, reference_file: str = None) -> Dict:
    """Evalúa el rendimiento NER comparando predicciones con referencias"""
    
    print(f"📊 EVALUANDO RENDIMIENTO NER: {predictions_file}")
    
    # Cargar predicciones
    predictions = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    print(f"📁 Predicciones cargadas: {len(predictions)} documentos")
    
    # Si no hay archivo de referencia, usar las entidades del mismo archivo
    if not reference_file:
        reference_file = predictions_file
        print("ℹ️  Usando entidades del mismo archivo como referencia")
    
    # Cargar referencias
    references = []
    with open(reference_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                references.append(json.loads(line))
    
    print(f"📁 Referencias cargadas: {len(references)} documentos")
    
    # Crear diccionario de referencias por PMID
    ref_by_pmid = {}
    for ref in references:
        pmid = str(ref.get("PMID", ""))
        if pmid:
            entities = []
            for ent in ref.get("Entidad", []):
                if isinstance(ent, dict) and "texto" in ent:
                    entities.append(normalize_text(ent["texto"]))
            ref_by_pmid[pmid] = entities
    
    print(f"🔍 Referencias indexadas por PMID: {len(ref_by_pmid)} documentos")
    
    # Evaluar cada predicción
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    
    detailed_results = []
    
    for pred in predictions:
        pmid = str(pred.get("PMID", ""))
        predicted_entities = []
        
        # Extraer entidades predichas
        for ent in pred.get("Entidad", []):
            if isinstance(ent, dict) and "texto" in ent:
                predicted_entities.append(normalize_text(ent["texto"]))
        
        # Obtener entidades de referencia
        reference_entities = ref_by_pmid.get(pmid, [])
        
        if not reference_entities:
            print(f"⚠️  No se encontraron referencias para PMID {pmid}")
            continue
        
        # Calcular métricas para este documento
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        matched_predictions = set()
        matched_references = set()
        
        # Encontrar matches
        for pred_ent in predicted_entities:
            matched = False
            for i, ref_ent in enumerate(reference_entities):
                if i not in matched_references and fuzzy_match(pred_ent, ref_ent):
                    tp += 1
                    matched_predictions.add(pred_ent)
                    matched_references.add(i)
                    matched = True
                    break
            
            if not matched:
                fp += 1
        
        # False Negatives (referencias no encontradas)
        fn = len(reference_entities) - len(matched_references)
        
        # Acumular totales
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Guardar resultados detallados
        doc_result = {
            "pmid": pmid,
            "predicted": predicted_entities,
            "reference": reference_entities,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0
        }
        detailed_results.append(doc_result)
        
        print(f"📄 PMID {pmid}: TP={tp}, FP={fp}, FN={fn}, P={doc_result['precision']:.3f}, R={doc_result['recall']:.3f}")
    
    # Calcular métricas globales
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Análisis por estrategia
    strategy_analysis = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for pred in predictions:
        pmid = str(pred.get("PMID", ""))
        reference_entities = ref_by_pmid.get(pmid, [])
        
        if not reference_entities:
            continue
        
        # Analizar cada estrategia
        for ent in pred.get("Entidad", []):
            if isinstance(ent, dict) and "strategies" in ent:
                strategies = ent.get("strategies", [])
                entity_text = normalize_text(ent.get("texto", ""))
                
                # Verificar si es TP o FP
                is_tp = any(fuzzy_match(entity_text, ref_ent) for ref_ent in reference_entities)
                
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
        "strategy_metrics": strategy_metrics,
        "detailed_results": detailed_results,
        "summary": {
            "total_documents": len(detailed_results),
            "total_predictions": sum(len(r["predicted"]) for r in detailed_results),
            "total_references": sum(len(r["reference"]) for r in detailed_results)
        }
    }
    
    return results

def print_results(results: Dict):
    """Imprime los resultados de la evaluación"""
    
    print("\n" + "="*60)
    print("🏆 RESULTADOS DE EVALUACIÓN NER")
    print("="*60)
    
    overall = results["overall"]
    summary = results["summary"]
    
    print(f"\n📊 MÉTRICAS GLOBALES:")
    print(f"   Precisión: {overall['precision']:.3f} ({overall['precision']*100:.1f}%)")
    print(f"   Recall:    {overall['recall']:.3f} ({overall['recall']*100:.1f}%)")
    print(f"   F1-Score:  {overall['f1']:.3f} ({overall['f1']*100:.1f}%)")
    
    print(f"\n📈 CONTEO DE ENTIDADES:")
    print(f"   True Positives (TP):  {overall['tp']}")
    print(f"   False Positives (FP): {overall['fp']}")
    print(f"   False Negatives (FN): {overall['fn']}")
    
    print(f"\n📁 RESUMEN DEL DATASET:")
    print(f"   Documentos procesados: {summary['total_documents']}")
    print(f"   Entidades predichas:   {summary['total_predictions']}")
    print(f"   Entidades de referencia: {summary['total_references']}")
    
    print(f"\n🎯 RENDIMIENTO POR ESTRATEGIA:")
    for strategy, metrics in results["strategy_metrics"].items():
        p = metrics["precision"]
        tp = metrics["tp"]
        fp = metrics["fp"]
        print(f"   {strategy:20}: P={p:.3f} ({p*100:.1f}%) | TP={tp}, FP={fp}")
    
    print(f"\n📊 ANÁLISIS DETALLADO:")
    print("   Los primeros 5 documentos:")
    for i, doc in enumerate(results["detailed_results"][:5]):
        print(f"   {i+1}. PMID {doc['pmid']}: P={doc['precision']:.3f}, R={doc['recall']:.3f}")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluador de rendimiento NER")
    parser.add_argument("--predictions", required=True, help="Archivo de predicciones JSONL")
    parser.add_argument("--reference", help="Archivo de referencia JSONL (opcional)")
    
    args = parser.parse_args()
    
    try:
        # Evaluar rendimiento
        results = evaluate_ner_performance(args.predictions, args.reference)
        
        # Imprimir resultados
        print_results(results)
        
        # Guardar resultados en archivo
        output_file = "ner_evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Resultados guardados en: {output_file}")
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
