#!/usr/bin/env python3
import json

# Cargar resultados de evaluación
with open('ner_evaluation_results.json', 'r') as f:
    data = json.load(f)

print("=== REVISIÓN DE ERRORES ===")
print(f"Total documentos evaluados: {len(data['detailed_results'])}")

# Encontrar documentos con errores
errors = [d for d in data['detailed_results'] if d['fp'] > 0 or d['fn'] > 0]
print(f"Documentos con errores: {len(errors)}")

print("\n=== DETALLE DE ERRORES ===")
for doc in errors:
    print(f"PMID {doc['pmid']}:")
    print(f"  TP={doc['tp']}, FP={doc['fp']}, FN={doc['fn']}")
    print(f"  Precisión={doc['precision']:.3f}, Recall={doc['recall']:.3f}")
    print(f"  Predichas: {doc['predicted']}")
    print(f"  Referencia: {doc['reference']}")
    print()

# Verificar métricas globales
overall = data['overall']
print("=== MÉTRICAS GLOBALES ===")
print(f"Precisión: {overall['precision']:.3f} ({overall['precision']*100:.1f}%)")
print(f"Recall: {overall['recall']:.3f} ({overall['recall']*100:.1f}%)")
print(f"F1: {overall['f1']:.3f} ({overall['f1']*100:.1f}%)")
print(f"TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']}")

# Verificar que los números coincidan
total_predicted = sum(len(d['predicted']) for d in data['detailed_results'])
total_reference = sum(len(d['reference']) for d in data['detailed_results'])
print(f"\nTotal predichas: {total_predicted}")
print(f"Total referencia: {total_reference}")
print(f"TP + FP = {overall['tp'] + overall['fp']} (debería ser {total_predicted})")
print(f"TP + FN = {overall['tp'] + overall['fn']} (debería ser {total_reference})")
