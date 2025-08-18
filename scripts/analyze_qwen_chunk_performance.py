import glob
import json
import os
import re
from collections import defaultdict


def read_results(filepath):
    """Read and analyze results from a single file"""
    docs = 0
    total_entities = 0
    confidences = []
    entities_set = set()
    qwen_entities = 0
    avg_latency = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            latencies = []
            for line in f:
                if not line.strip():
                    continue
                docs += 1
                data = json.loads(line)
                ents = data.get("Entidad", [])
                total_entities += len(ents)
                
                # Get latency
                latency = data.get("_latency_sec", 0)
                latencies.append(latency)
                
                for ent in ents:
                    confidences.append(ent.get("confidence", 0.0))
                    texto = ent.get("texto", "").strip().lower()
                    if texto:
                        entities_set.add(texto)
                        # Count qwen entities
                        strategies = ent.get("strategies", [])
                        if any("qwen" in s.lower() for s in strategies):
                            qwen_entities += 1
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            
    except Exception as e:
        print(f"[ERROR] Reading {filepath}: {e}")
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "docs": docs,
        "entities": total_entities,
        "qwen_entities": qwen_entities,
        "avg_conf": avg_conf,
        "entities_set": entities_set,
        "avg_latency": avg_latency
    }


def parse_cfg_from_name(name):
    """Extract chunk_target and overlap from filename"""
    m = re.search(r"chunk(\d+)_ov(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def main():
    # Find all qwen result files
    result_files = sorted(glob.glob("results_qwen_chunk*_ov*.jsonl"))
    
    if not result_files:
        print("❌ No se encontraron archivos de resultados de qwen")
        return

    print("🔬 ANÁLISIS DE RENDIMIENTO DE CHUNKS - qwen2.5:3b")
    print("=" * 60)
    
    # Analyze all results
    all_results = []
    for fp in result_files:
        base = os.path.basename(fp)
        chunk, ov = parse_cfg_from_name(base)
        if chunk is None:
            continue
            
        res = read_results(fp)
        all_results.append({
            "file": base,
            "chunk_target": chunk,
            "chunk_overlap": ov,
            "docs": res["docs"],
            "entities": res["entities"],
            "qwen_entities": res["qwen_entities"],
            "avg_conf": res["avg_conf"],
            "avg_latency": res["avg_latency"],
            "unique_entities": len(res["entities_set"])
        })
    
    # Sort by qwen_entities desc, then avg_conf desc
    all_results.sort(key=lambda x: (x["qwen_entities"], x["avg_conf"]), reverse=True)
    
    # Print detailed results
    print("\n📊 RESULTADOS DETALLADOS POR CONFIGURACIÓN:")
    print("chunk_target | chunk_overlap | docs | total_entities | qwen_entities | avg_conf | avg_latency | unique_entities")
    print("-" * 100)
    
    for r in all_results:
        print(f"{r['chunk_target']:11d} | {r['chunk_overlap']:13d} | {r['docs']:4d} | {r['entities']:14d} | {r['qwen_entities']:12d} | {r['avg_conf']:8.3f} | {r['avg_latency']:11.3f} | {r['unique_entities']:15d}")
    
    # Find best configurations
    print("\n🏆 MEJORES CONFIGURACIONES:")
    
    # Best by entities detected
    best_entities = max(all_results, key=lambda x: x["qwen_entities"])
    print(f"🥇 Mejor por entidades detectadas:")
    print(f"   chunk_target={best_entities['chunk_target']}, chunk_overlap={best_entities['chunk_overlap']}")
    print(f"   Entidades: {best_entities['qwen_entities']}, Confianza: {best_entities['avg_conf']:.3f}")
    
    # Best by confidence
    best_conf = max(all_results, key=lambda x: x["avg_conf"])
    print(f"\n🥈 Mejor por confianza:")
    print(f"   chunk_target={best_conf['chunk_target']}, chunk_overlap={best_conf['chunk_overlap']}")
    print(f"   Entidades: {best_conf['qwen_entities']}, Confianza: {best_conf['avg_conf']:.3f}")
    
    # Best by speed
    best_speed = min(all_results, key=lambda x: x["avg_latency"])
    print(f"\n🥉 Mejor por velocidad:")
    print(f"   Latencia: {best_speed['avg_latency']:.3f}s, Entidades: {best_speed['qwen_entities']}")
    
    # Overall statistics
    print("\n📈 ESTADÍSTICAS GENERALES:")
    total_qwen_entities = sum(r["qwen_entities"] for r in all_results)
    avg_qwen_entities = total_qwen_entities / len(all_results)
    avg_conf_overall = sum(r["avg_conf"] for r in all_results) / len(all_results)
    avg_latency_overall = sum(r["avg_latency"] for r in all_results) / len(all_results)
    
    print(f"   Total entidades detectadas por qwen: {total_qwen_entities}")
    print(f"   Promedio entidades por configuración: {avg_qwen_entities:.2f}")
    print(f"   Promedio confianza general: {avg_conf_overall:.3f}")
    print(f"   Promedio latencia general: {avg_latency_overall:.3f}s")
    
    # Analysis by chunk size
    print("\n🔍 ANÁLISIS POR TAMAÑO DE CHUNK:")
    chunk_sizes = defaultdict(list)
    for r in all_results:
        chunk_sizes[r["chunk_target"]].append(r)
    
    for chunk_size in sorted(chunk_sizes.keys()):
        configs = chunk_sizes[chunk_size]
        avg_entities = sum(c["qwen_entities"] for c in configs) / len(configs)
        avg_conf = sum(c["avg_conf"] for c in configs) / len(configs)
        print(f"   chunk_target={chunk_size:3d}: {len(configs):2d} configs, avg_entities={avg_entities:.2f}, avg_conf={avg_conf:.3f}")
    
    print("\n✅ Análisis de qwen2.5:3b completado!")


if __name__ == "__main__":
    main()
