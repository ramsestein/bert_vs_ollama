#!/usr/bin/env python3
"""
Grid Search de Gemma3 para NCBI develop
Optimiza chunks y overlaps usando gemma3_optimized
"""

import itertools
import subprocess
import sys
import os
from datetime import datetime

def main():
    chunk_targets = [20, 40, 60, 80, 100]
    overlaps = [10, 20, 30, 40, 50]

    # Crear directorio para resultados
    results_dir = f"gemma_grid_ncbi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    combos = [(t, o) for t in chunk_targets for o in overlaps if o < t]
    print(f"🚀 Grid Search Gemma3 para NCBI: {len(combos)} configuraciones")
    print(f"📁 Resultados en: {results_dir}")
    print(f"📊 Dataset: ncbi_develop")
    print()

    successful_runs = 0
    failed_runs = 0

    for i, (t, o) in enumerate(combos, 1):
        print(f"🔄 [{i:2d}/{len(combos)}] chunk={t:3d}, overlap={o:3d}")
        
        out_file = f"results_gemma_ncbi_chunk{t}_ov{o}.jsonl"
        out_path = os.path.join(results_dir, out_file)
        
        cmd = [
            sys.executable,
            "-m",
            "ner_app.main",
            "--input_jsonl",
            "datasets/ncbi_develop_input.jsonl",
            "--benchmark_jsonl",
            "datasets/ncbi_develop.jsonl",
            "--out_pred",
            out_path,
            "--limit",
            "20",  # Usar 20 documentos para evaluación
            "--confidence_threshold",
            "0.3",
            "--strategies",
            "gemma3_max_sensitivity",  # Use the actual strategy name from strategies.py
            "--s1_target",
            str(t),
            "--s1_overlap",
            str(o),
            "--s1_temp",
            "0.5",
        ]
        
        print(f"   📝 Ejecutando: {out_file}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ Éxito: {out_file}")
            successful_runs += 1
            
            # Evaluar rendimiento inmediatamente
            evaluate_performance(out_path)
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error: {e}")
            print(f"   📄 Salida: {e.stdout}")
            print(f"   ⚠️  Error: {e.stderr}")
            failed_runs += 1
        
        print()
    
    # Resumen final
    print("=" * 60)
    print("🏁 GRID SEARCH GEMMA3 COMPLETADO")
    print("=" * 60)
    print(f"✅ Ejecuciones exitosas: {successful_runs}")
    print(f"❌ Ejecuciones fallidas: {failed_runs}")
    print(f"📁 Resultados guardados en: {results_dir}")
    print()
    print("📊 Para analizar resultados:")
    print(f"   python scripts/analyze_grid.py {results_dir}")

def evaluate_performance(result_file):
    """Evalúa el rendimiento de una configuración específica"""
    try:
        cmd = [
            sys.executable,
            "scripts/evaluate_ner_performance.py",
            "--predictions",
            result_file,
            "--reference",
            "datasets/ncbi_develop.jsonl"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        
        # Extraer métricas clave
        for line in output.split('\n'):
            if "Precisión:" in line:
                precision = line.split(":")[1].strip().split()[0]
                print(f"      📊 P={precision}")
                break
                
    except subprocess.CalledProcessError:
        print(f"      ⚠️  No se pudo evaluar rendimiento")

if __name__ == "__main__":
    main()
