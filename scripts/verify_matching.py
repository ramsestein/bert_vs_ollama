#!/usr/bin/env python3
import json
import re

def normalize_text(text):
    """Normaliza texto para comparación consistente"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'–|—', '-', text)
    return text

def fuzzy_match(predicted, reference, threshold=0.8):
    """Matching fuzzy entre entidad predicha y referencia"""
    pred_norm = normalize_text(predicted)
    ref_norm = normalize_text(reference)
    
    if pred_norm == ref_norm:
        return True
    
    if pred_norm in ref_norm or ref_norm in pred_norm:
        return True
    
    pred_chars = set(pred_norm)
    ref_chars = set(ref_norm)
    
    if not pred_chars or not ref_chars:
        return False
    
    intersection = len(pred_chars.intersection(ref_chars))  # CORREGIDO: ref_chars en lugar de pred_chars
    union = len(pred_chars.union(ref_chars))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    return similarity >= threshold

def analyze_matching_issues():
    """Analiza problemas de matching en los documentos con errores"""
    
    # Cargar datos
    with open('results_full_develop.jsonl', 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    
    with open('./datasets/ncbi_develop.jsonl', 'r') as f:
        references = [json.loads(line) for line in f if line.strip()]
    
    error_pmids = ['9083764', '9090524', '8637912']
    
    for pmid in error_pmids:
        print(f"\n{'='*60}")
        print(f"VERIFICACIÓN DE MATCHING: PMID {pmid}")
        print(f"{'='*60}")
        
        pred_doc = next((d for d in predictions if d['PMID'] == pmid), None)
        ref_doc = next((d for d in references if d['PMID'] == pmid), None)
        
        if not pred_doc or not ref_doc:
            continue
        
        pred_entities = [e['texto'] for e in pred_doc.get('Entidad', [])]
        ref_entities = [e['texto'] for e in ref_doc.get('Entidad', [])]
        
        print(f"Predichas ({len(pred_entities)}): {pred_entities}")
        print(f"Referencia ({len(ref_entities)}): {ref_entities}")
        
        # Verificar matching exacto
        print("\nMATCHING EXACTO:")
        matched_pred = set()
        matched_ref = set()
        
        for i, pred_ent in enumerate(pred_entities):
            for j, ref_ent in enumerate(ref_entities):
                if j not in matched_ref and pred_ent == ref_ent:
                    print(f"  ✓ EXACTO: '{pred_ent}' = '{ref_ent}'")
                    matched_pred.add(i)
                    matched_ref.add(j)
                    break
        
        # Verificar matching case-insensitive
        print("\nMATCHING CASE-INSENSITIVE:")
        for i, pred_ent in enumerate(pred_entities):
            if i in matched_pred:
                continue
            for j, ref_ent in enumerate(ref_entities):
                if j not in matched_ref and pred_ent.lower() == ref_ent.lower():
                    print(f"  ✓ CASE-INSENSITIVE: '{pred_ent}' = '{ref_ent}'")
                    matched_pred.add(i)
                    matched_ref.add(j)
                    break
        
        # Verificar matching fuzzy
        print("\nMATCHING FUZZY:")
        for i, pred_ent in enumerate(pred_entities):
            if i in matched_pred:
                continue
            for j, ref_ent in enumerate(ref_entities):
                if j not in matched_ref and fuzzy_match(pred_ent, ref_ent):
                    print(f"  ✓ FUZZY: '{pred_ent}' ≈ '{ref_ent}'")
                    matched_pred.add(i)
                    matched_ref.add(j)
                    break
        
        # Entidades no matcheadas
        print("\nENTIDADES NO MATCHEADAS:")
        for i, pred_ent in enumerate(pred_entities):
            if i not in matched_pred:
                print(f"  ✗ Predicha no matcheada: '{pred_ent}'")
        
        for j, ref_ent in enumerate(ref_entities):
            if j not in matched_ref:
                print(f"  ✗ Referencia no matcheada: '{ref_ent}'")

if __name__ == "__main__":
    analyze_matching_issues()
