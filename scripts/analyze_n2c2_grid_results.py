#!/usr/bin/env python3
"""
Análisis de resultados del grid search de n2c2
Analiza chunks y overlaps para Llama y Qwen
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

def parse_filename(filename):
    """Extrae parámetros del nombre del archivo"""
    # Formato: results_llama_n2c2_chunk{chunk}_ov{overlap}.jsonl
    # o: results_qwen_n2c2_chunk{chunk}_ov{overlap}.jsonl
    pattern = r"results_(llama|qwen)_n2c2_chunk(\d+)_ov(\d+)\.jsonl"
    match = re.search(pattern, filename)
    if match:
        return {
            'model': match.group(1),
            'chunk': int(match.group(2)),
            'overlap': int(match.group(3))
        }
    return None

def read_results(filepath):
    """Lee y analiza un archivo de resultados"""
    docs = 0
    total_entities = 0
    confidences = []
    entities_set = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                docs += 1
                data = json.loads(line)
                ents = data.get("Entidad", [])
                total_entities += len(ents)
                for ent in ents:
                    confidences.append(ent.get("confidence", 0.0))
                    texto = ent.get("texto", "").strip().lower()
                    if texto:
                        entities_set.add(texto)
    except Exception as e:
        print(f"[ERROR] Leyendo {filepath}: {e}")
        return None
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "docs": docs,
        "entities": total_entities,
        "avg_conf": avg_conf,
        "entities_set": entities_set,
        "min_conf": min(confidences) if confidences else 0.0,
        "max_conf": max(confidences) if confidences else 0.0,
    }

def evaluate_performance(result_file, reference_file):
    """Evalúa el rendimiento usando el evaluador NER"""
    try:
        import subprocess
        
        cmd = [
            sys.executable,
            "scripts/evaluate_ner_performance.py",
            "--predictions",
            result_file,
            "--reference",
            reference_file
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        
        # Extraer métricas
        metrics = {}
        for line in output.split('\n'):
            if "Precisión:" in line:
                metrics['precision'] = float(line.split(":")[1].strip().split()[0])
            elif "Recall:" in line:
                metrics['recall'] = float(line.split(":")[1].strip().split()[0])
            elif "F1-Score:" in line:
                metrics['f1_score'] = float(line.split(":")[1].strip().split()[0])
            elif "True Positives (TP):" in line:
                metrics['tp'] = int(line.split(":")[1].strip())
            elif "False Positives (FP):" in line:
                metrics['fp'] = int(line.split(":")[1].strip())
            elif "False Negatives (FN):" in line:
                metrics['fn'] = int(line.split(":")[1].strip())
        
        return metrics
        
    except Exception as e:
        print(f"[ERROR] Evaluando {result_file}: {e}")
        return {}

def main():
    if len(sys.argv) != 2:
        print("Uso: python analyze_n2c2_grid_results.py <directorio_resultados>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"❌ Error: El directorio {results_dir} no existe")
        sys.exit(1)
    
    print(f"🔍 Analizando resultados del grid search en: {results_dir}")
    print("=" * 70)
    
    # Buscar archivos de resultados (tanto llama como qwen)
    pattern = os.path.join(results_dir, "results_*_n2c2_chunk*_ov*.jsonl")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"❌ No se encontraron archivos de resultados en {results_dir}")
        sys.exit(1)
    
    print(f"📁 Encontrados {len(result_files)} archivos de resultados")
    print()
    
    # Analizar cada archivo
    all_results = []
    reference_file = "datasets/n2c2_developt.jsonl"
    
    for filepath in result_files:
        filename = os.path.basename(filepath)
        params = parse_filename(filename)
        
        if not params:
            print(f"⚠️  No se pudieron parsear parámetros de: {filename}")
            continue
        
        print(f"📊 Analizando: {filename}")
        
        # Leer resultados
        results = read_results(filepath)
        if not results:
            continue
        
        # Evaluar rendimiento
        metrics = evaluate_performance(filepath, reference_file)
        
        # Combinar información
        result_data = {
            'filename': filename,
            'model': params['model'],
            'chunk': params['chunk'],
            'overlap': params['overlap'],
            'docs': results['docs'],
            'entities': results['entities'],
            'avg_conf': results['avg_conf'],
            'min_conf': results['min_conf'],
            'max_conf': results['max_conf'],
            **metrics
        }
        
        all_results.append(result_data)
        
        # Mostrar métricas básicas
        if metrics:
            print(f"   📈 P={metrics.get('precision', 0):.3f}, R={metrics.get('recall', 0):.3f}, F1={metrics.get('f1_score', 0):.3f}")
        print(f"   📊 Entidades: {results['entities']}, Confianza: {results['avg_conf']:.3f}")
        print()
    
    if not all_results:
        print("❌ No se pudieron analizar resultados válidos")
        return
    
    # Análisis por modelo
    print("🎯 ANÁLISIS POR MODELO")
    print("=" * 50)
    
    for model in ['llama', 'qwen']:
        model_results = [r for r in all_results if r['model'] == model]
        if not model_results:
            continue
            
        print(f"\n🤖 MODELO: {model.upper()}")
        print("-" * 30)
        
        # Análisis por chunk
        print(f"📏 ANÁLISIS POR CHUNK TARGET:")
        chunk_analysis = defaultdict(lambda: {'entities': [], 'avg_conf': [], 'precision': [], 'recall': [], 'f1_score': []})
        
        for result in model_results:
            chunk_analysis[result['chunk']]['entities'].append(result['entities'])
            chunk_analysis[result['chunk']]['avg_conf'].append(result['avg_conf'])
            if 'precision' in result:
                chunk_analysis[result['chunk']]['precision'].append(result['precision'])
            if 'recall' in result:
                chunk_analysis[result['chunk']]['recall'].append(result['recall'])
            if 'f1_score' in result:
                chunk_analysis[result['chunk']]['f1_score'].append(result['f1_score'])
        
        for chunk in sorted(chunk_analysis.keys()):
            data = chunk_analysis[chunk]
            avg_entities = sum(data['entities']) / len(data['entities'])
            avg_conf = sum(data['avg_conf']) / len(data['avg_conf'])
            avg_precision = sum(data['precision']) / len(data['precision']) if data['precision'] else 0
            avg_recall = sum(data['recall']) / len(data['recall']) if data['recall'] else 0
            avg_f1 = sum(data['f1_score']) / len(data['f1_score']) if data['f1_score'] else 0
            
            print(f"  chunk={chunk:3d}: entidades={avg_entities:6.2f}, conf={avg_conf:5.3f}, P={avg_precision:5.3f}, R={avg_recall:5.3f}, F1={avg_f1:5.3f}")
        
        # Análisis por overlap
        print(f"\n🔗 ANÁLISIS POR OVERLAP:")
        overlap_analysis = defaultdict(lambda: {'entities': [], 'avg_conf': [], 'precision': [], 'recall': [], 'f1_score': []})
        
        for result in model_results:
            overlap_analysis[result['overlap']]['entities'].append(result['entities'])
            overlap_analysis[result['overlap']]['avg_conf'].append(result['avg_conf'])
            if 'precision' in result:
                overlap_analysis[result['overlap']]['precision'].append(result['precision'])
            if 'recall' in result:
                overlap_analysis[result['overlap']]['recall'].append(result['recall'])
            if 'f1_score' in result:
                overlap_analysis[result['overlap']]['f1_score'].append(result['f1_score'])
        
        for overlap in sorted(overlap_analysis.keys()):
            data = overlap_analysis[overlap]
            avg_entities = sum(data['entities']) / len(data['entities'])
            avg_conf = sum(data['avg_conf']) / len(data['avg_conf'])
            avg_precision = sum(data['precision']) / len(data['precision']) if data['precision'] else 0
            avg_recall = sum(data['recall']) / len(data['recall']) if data['recall'] else 0
            avg_f1 = sum(data['f1_score']) / len(data['f1_score']) if data['f1_score'] else 0
            
            print(f"  overlap={overlap:3d}: entidades={avg_entities:6.2f}, conf={avg_conf:5.3f}, P={avg_precision:5.3f}, R={avg_recall:5.3f}, F1={avg_f1:5.3f}")
    
    # Top configuraciones por F1-Score
    print("\n🏆 TOP 10 CONFIGURACIONES POR F1-SCORE:")
    print("-" * 80)
    valid_results = [r for r in all_results if 'f1_score' in r and r['f1_score'] > 0]
    if valid_results:
        top_f1 = sorted(valid_results, key=lambda x: x['f1_score'], reverse=True)[:10]
        for i, result in enumerate(top_f1, 1):
            print(f"{i:2d}. {result['model']:6s} | chunk={result['chunk']:3d}, ov={result['overlap']:3d} | F1={result['f1_score']:5.3f}, P={result.get('precision', 0):5.3f}, R={result.get('recall', 0):5.3f}")
    else:
        print("No hay métricas de F1 disponibles")
    
    # Top configuraciones por entidades detectadas
    print("\n📊 TOP 10 CONFIGURACIONES POR ENTIDADES DETECTADAS:")
    print("-" * 80)
    top_entities = sorted(all_results, key=lambda x: x['entities'], reverse=True)[:10]
    for i, result in enumerate(top_entities, 1):
        print(f"{i:2d}. {result['model']:6s} | chunk={result['chunk']:3d}, ov={result['overlap']:3d} | entidades={result['entities']:3d}, conf={result['avg_conf']:5.3f}")
    
    # Guardar resultados en CSV
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(results_dir, "grid_search_analysis.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Análisis completo guardado en: {csv_file}")
    except ImportError:
        print("\n⚠️  pandas no disponible, no se guardó CSV")

if __name__ == "__main__":
    main()
