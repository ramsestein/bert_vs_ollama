#!/usr/bin/env python3
"""
Grid Search de Temperaturas para n2c2_developt
Optimiza temperatura manteniendo chunks y overlaps por defecto
"""

import itertools
import subprocess
import sys
import os
from datetime import datetime

def main():
    # Configuración del grid search de temperaturas
    temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Chunks y overlaps por defecto (que ya sabemos que funcionan)
    default_chunk = 100
    default_overlap = 20

    # Crear directorio para resultados
    results_dir = f"temperature_grid_n2c2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    # Generar todas las combinaciones
    combos = list(itertools.product(temperatures, confidence_thresholds))
    print(f"🌡️  Grid Search de Temperaturas para n2c2: {len(combos)} configuraciones")
    print(f"📁 Resultados en: {results_dir}")
    print(f"📊 Dataset: n2c2_developt")
    print(f"🔧 Chunk: {default_chunk}, Overlap: {default_overlap} (por defecto)")
    print()

    successful_runs = 0
    failed_runs = 0

    for i, (temp, conf_thresh) in enumerate(combos, 1):
        print(f"🔄 [{i:2d}/{len(combos)}] temp={temp:3.1f}, conf={conf_thresh:3.1f}")

        # Nombre del archivo de salida
        out_file = f"results_temp{temp:03.1f}_conf{conf_thresh:03.1f}.jsonl"
        out_path = os.path.join(results_dir, out_file)

        # Comando para ejecutar
        cmd = [
            sys.executable,
            "scripts/llama_ner_multi_strategy.py",
            "--input_jsonl",
            "datasets/n2c2_developt_input.jsonl",
            "--benchmark_jsonl",
            "datasets/n2c2_developt.jsonl",
            "--out_pred",
            out_path,
            "--limit",
            "20",  # Usar 20 documentos para evaluación
            "--confidence_threshold",
            str(conf_thresh),
            "--strategies",
            "llama32_optimized",  # Usar Llama como modelo principal
            "--s1_target",
            str(default_chunk),
            "--s1_overlap",
            str(default_overlap),
            "--s1_temp",
            str(temp),
        ]

        print(f"   📝 Ejecutando: {out_file}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ Éxito: {out_file}")
            successful_runs += 1

            # Evaluar rendimiento inmediatamente
            evaluate_performance(out_path, temp, conf_thresh)

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error: {e}")
            failed_runs += 1

        print()

    # Resumen final
    print("=" * 60)
    print("🌡️  GRID SEARCH DE TEMPERATURAS COMPLETADO")
    print("=" * 60)
    print(f"✅ Ejecuciones exitosas: {successful_runs}")
    print(f"❌ Ejecuciones fallidas: {failed_runs}")
    print(f"📁 Resultados guardados en: {results_dir}")
    print()
    print("📊 Para analizar resultados:")
    print(f"   python scripts/analyze_temperature_grid.py {results_dir}")

def evaluate_performance(result_file, temp, conf_thresh):
    """Evalúa el rendimiento de una configuración específica"""
    try:
        cmd = [
            sys.executable,
            "scripts/evaluate_ner_performance.py",
            "--predictions",
            result_file,
            "--reference",
            "datasets/n2c2_developt.jsonl"
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
