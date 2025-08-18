import json
from collections import Counter

# Analizar resultados de qwen3:4b
with open('results_qwen4b_balance.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

print(f"=== ANÁLISIS DE RESULTADOS QWEN3:4B ===")
print(f"Total documentos procesados: {len(results)}")
print(f"Total entidades detectadas: {sum(len(r['Entidad']) for r in results)}")
print(f"Promedio entidades por documento: {sum(len(r['Entidad']) for r in results)/len(results):.2f}")

total_chunks = sum(r.get('_debug', {}).get('n_chunks', 0) for r in results)
print(f"Total chunks procesados: {total_chunks}")
print(f"Promedio chunks por documento: {total_chunks/len(results):.2f}")

total_exact = sum(sum(r.get('_debug', {}).get('exact_hits', {}).values()) for r in results)
print(f"Total exact matches: {total_exact}")

total_verified = sum(len(r.get('_debug', {}).get('seen_checks', {})) for r in results)
print(f"Total entidades verificadas: {total_verified}")

# Análisis de tipos de entidades
entity_types = Counter()
for r in results:
    for ent in r['Entidad']:
        entity_types[ent['tipo']] += 1

print(f"\n=== TIPOS DE ENTIDADES ===")
for tipo, count in entity_types.most_common():
    print(f"{tipo}: {count}")

# Análisis de latencia
latencies = [r.get('_latency_sec', 0) for r in results]
avg_latency = sum(latencies) / len(latencies)
print(f"\n=== RENDIMIENTO ===")
print(f"Latencia promedio por documento: {avg_latency:.2f} segundos")
print(f"Latencia total: {sum(latencies):.2f} segundos")

# Análisis de debug info
print(f"\n=== DEBUG INFO ===")
for i, r in enumerate(results[:5]):  # Primeros 5 documentos
    debug = r.get('_debug', {})
    print(f"Doc {i+1}: chunks={debug.get('n_chunks', 0)}, exact={sum(debug.get('exact_hits', {}).values())}, verified={len(debug.get('seen_checks', {}))}, final={len(r.get('Entidad', []))}")
