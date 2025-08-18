import json
from collections import Counter

# Cargar resultados
with open('results_llama_optimized.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

print(f"=== ANÁLISIS DE RESULTADOS ===")
print(f"Total documentos procesados: {len(results)}")
print(f"Total entidades detectadas: {sum(len(r['Entidad']) for r in results)}")
print(f"Promedio entidades por documento: {sum(len(r['Entidad']) for r in results)/len(results):.2f}")

# Estadísticas de chunks
total_chunks = sum(r.get('_debug', {}).get('n_chunks', 0) for r in results)
print(f"Total chunks procesados: {total_chunks}")
print(f"Promedio chunks por documento: {total_chunks/len(results):.2f}")

# Estadísticas de cache
cache_entries = 0
for r in results:
    if '_debug' in r and 'cache_stats' in r['_debug']:
        cache_entries = r['_debug']['cache_stats']
        break

print(f"Entradas en cache: {cache_entries}")

# Tipos de entidades más comunes
entity_types = Counter()
for r in results:
    for ent in r['Entidad']:
        entity_types[ent['tipo']] += 1

print(f"\n=== TIPOS DE ENTIDADES ===")
for tipo, count in entity_types.most_common():
    print(f"{tipo}: {count}")

# Latencia promedio
latencies = [r.get('_latency_sec', 0) for r in results]
avg_latency = sum(latencies) / len(latencies)
print(f"\n=== RENDIMIENTO ===")
print(f"Latencia promedio por documento: {avg_latency:.2f} segundos")
print(f"Latencia total: {sum(latencies):.2f} segundos")
