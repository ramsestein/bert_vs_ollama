#!/usr/bin/env python3
import json

def analyze_document(pmid, predictions_file, reference_file):
    """Analiza un documento específico para entender los errores"""
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DETALLADO: PMID {pmid}")
    print(f"{'='*60}")
    
    # Cargar datos
    with open(predictions_file, 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    
    with open(reference_file, 'r') as f:
        references = [json.loads(line) for line in f if line.strip()]
    
    # Encontrar documento
    pred_doc = next((d for d in predictions if d['PMID'] == pmid), None)
    ref_doc = next((d for d in references if d['PMID'] == pmid), None)
    
    if not pred_doc or not ref_doc:
        print(f"Documento no encontrado: {pmid}")
        return
    
    print(f"Texto: {pred_doc['Texto'][:200]}...")
    print()
    
    # Entidades predichas
    pred_entities = [e['texto'] for e in pred_doc.get('Entidad', [])]
    ref_entities = [e['texto'] for e in ref_doc.get('Entidad', [])]
    
    print("ENTIDADES PREDICHAS:")
    for i, ent in enumerate(pred_entities):
        print(f"  {i+1}. {ent}")
    
    print("\nENTIDADES DE REFERENCIA:")
    for i, ent in enumerate(ref_entities):
        print(f"  {i+1}. {ent}")
    
    print(f"\nTotal predichas: {len(pred_entities)}")
    print(f"Total referencia: {len(ref_entities)}")
    
    # Análisis de matching
    print("\nANÁLISIS DE MATCHING:")
    matched_pred = set()
    matched_ref = set()
    
    for i, pred_ent in enumerate(pred_entities):
        matched = False
        for j, ref_ent in enumerate(ref_entities):
            if j not in matched_ref and pred_ent.lower() == ref_ent.lower():
                print(f"  ✓ MATCH: '{pred_ent}' = '{ref_ent}'")
                matched_pred.add(i)
                matched_ref.add(j)
                matched = True
                break
        
        if not matched:
            print(f"  ✗ NO MATCH: '{pred_ent}'")
    
    # False Negatives
    print("\nFALSE NEGATIVES (referencias no encontradas):")
    for j, ref_ent in enumerate(ref_entities):
        if j not in matched_ref:
            print(f"  ✗ NO ENCONTRADA: '{ref_ent}'")
    
    # False Positives
    print("\nFALSE POSITIVES (predicciones incorrectas):")
    for i, pred_ent in enumerate(pred_entities):
        if i not in matched_pred:
            print(f"  ✗ INCORRECTA: '{pred_ent}'")

def main():
    # Documentos con errores identificados
    error_pmids = ['9083764', '9090524', '8637912']
    
    print("ANÁLISIS DE ERRORES EN DETALLE")
    print("Analizando los 3 documentos que tienen errores...")
    
    for pmid in error_pmids:
        analyze_document(pmid, 'results_full_develop.jsonl', './datasets/ncbi_develop.jsonl')

if __name__ == "__main__":
    main()
