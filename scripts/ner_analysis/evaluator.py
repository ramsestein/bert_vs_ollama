#!/usr/bin/env python3
"""
Evaluator Module
Wrapper for evaluate_ner_performance.py to calculate metrics
"""

import os
import subprocess
import sys
from typing import Dict


def evaluate_performance(result_file: str, reference_file: str) -> Dict:
    """
    Evalúa el rendimiento NER usando el script evaluate_ner_performance.py
    
    Args:
        result_file: Archivo con predicciones
        reference_file: Archivo con referencias (ground truth)
    
    Returns:
        Dict con métricas (precision, recall, f1_score, tp, fp, fn) o dict vacío si hay error
    """
    if not reference_file or not os.path.exists(reference_file):
        return {}
    
    try:
        # Get the absolute path to the evaluation script
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_script = os.path.join(script_dir, "evaluation", "evaluate_ner_performance.py")
        
        cmd = [
            sys.executable,
            eval_script,
            "--predictions",
            result_file,
            "--reference",
            reference_file
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        
        # Extraer métricas del output
        metrics = {}
        for line in output.split('\n'):
            if "Precisión:" in line or "Precision:" in line:
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
