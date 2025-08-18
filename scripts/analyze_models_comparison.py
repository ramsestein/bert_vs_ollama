import glob
import json
import os
import re
from collections import defaultdict


def read_results(filepath):
    docs = 0
    total_entities = 0
    confidences = []
    entities_set = set()
    llama_entities = 0
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
                        # Count llama entities
                        strategies = ent.get("strategies", [])
                        if any("llama" in s.lower() for s in strategies):
                            llama_entities += 1
    except Exception as e:
        print(f"[ERROR] Reading {filepath}: {e}")
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "docs": docs,
        "entities": total_entities,
        "llama_entities": llama_entities,
        "avg_conf": avg_conf,
        "entities_set": entities_set,
    }


def parse_cfg_from_name(name):
    # Expect names like results_llama_chunk40_ov20.jsonl or results_qwen_chunk40_ov20.jsonl
    m = re.search(r"(llama|qwen).*chunk(\d+)_ov(\d+)", name)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


def main():
    # Find all result files
    llama_files = sorted(glob.glob("results_llama_chunk*_ov*.jsonl"))
    qwen_files = sorted(glob.glob("results_qwen_chunk*_ov*.jsonl"))
    
    if not llama_files and not qwen_files:
        print("No se encontraron archivos de resultados")
        return

    print("=== COMPARACIÓN DE MODELOS: qwen2.5:3b vs llama3.2:3b ===\n")

    # Analyze llama results
    if llama_files:
        print("📊 RESULTADOS DE llama3.2:3b:")
        llama_summaries = []
        for fp in llama_files:
            base = os.path.basename(fp)
            model, chunk, ov = parse_cfg_from_name(base)
            res = read_results(fp)
            llama_summaries.append({
                "file": base,
                "chunk": chunk,
                "overlap": ov,
                "docs": res["docs"],
                "entities": res["entities"],
                "llama_entities": res["llama_entities"],
                "avg_conf": res["avg_conf"],
            })
        
        # Sort by entities desc, then avg_conf desc
        llama_summaries.sort(key=lambda x: (x["entities"], x["avg_conf"]), reverse=True)
        
        print("file, chunk, overlap, docs, total_entities, llama_entities, avg_conf")
        for s in llama_summaries:
            print(f"{s['file']}, {s['chunk']}, {s['overlap']}, {s['docs']}, {s['entities']}, {s['llama_entities']}, {s['avg_conf']:.3f}")

    print()

    # Analyze qwen results
    if qwen_files:
        print("📊 RESULTADOS DE qwen2.5:3b:")
        qwen_summaries = []
        for fp in qwen_files:
            base = os.path.basename(fp)
            model, chunk, ov = parse_cfg_from_name(base)
            res = read_results(fp)
            qwen_summaries.append({
                "file": base,
                "chunk": chunk,
                "overlap": ov,
                "docs": res["docs"],
                "entities": res["entities"],
                "llama_entities": res["llama_entities"],
                "avg_conf": res["avg_conf"],
            })
        
        # Sort by entities desc, then avg_conf desc
        qwen_summaries.sort(key=lambda x: (x["entities"], x["avg_conf"]), reverse=True)
        
        print("file, chunk, overlap, docs, total_entities, llama_entities, avg_conf")
        for s in qwen_summaries:
            print(f"{s['file']}, {s['chunk']}, {s['overlap']}, {s['docs']}, {s['entities']}, {s['llama_entities']}, {s['avg_conf']:.3f}")

    print()

    # Compare best configurations
    if llama_files and qwen_files:
        print("🏆 COMPARACIÓN DE MEJORES CONFIGURACIONES:")
        
        # Best llama config
        best_llama = max(llama_summaries, key=lambda x: (x["llama_entities"], x["avg_conf"]))
        print(f"llama3.2:3b - Mejor: chunk={best_llama['chunk']}, overlap={best_llama['overlap']}")
        print(f"  Entidades detectadas: {best_llama['llama_entities']}")
        print(f"  Confianza promedio: {best_llama['avg_conf']:.3f}")
        
        # Best qwen config
        best_qwen = max(qwen_summaries, key=lambda x: (x["llama_entities"], x["avg_conf"]))
        print(f"qwen2.5:3b - Mejor: chunk={best_qwen['chunk']}, overlap={best_qwen['overlap']}")
        print(f"  Entidades detectadas: {best_qwen['llama_entities']}")
        print(f"  Confianza promedio: {best_qwen['avg_conf']:.3f}")
        
        print()
        
        # Overall comparison
        llama_avg_entities = sum(s["llama_entities"] for s in llama_summaries) / len(llama_summaries)
        qwen_avg_entities = sum(s["llama_entities"] for s in qwen_summaries) / len(qwen_summaries)
        
        llama_avg_conf = sum(s["avg_conf"] for s in llama_summaries) / len(llama_summaries)
        qwen_avg_conf = sum(s["avg_conf"] for s in qwen_summaries) / len(qwen_summaries)
        
        print("📈 COMPARACIÓN GENERAL:")
        print(f"llama3.2:3b - Promedio entidades: {llama_avg_entities:.2f}, Promedio confianza: {llama_avg_conf:.3f}")
        print(f"qwen2.5:3b - Promedio entidades: {qwen_avg_entities:.2f}, Promedio confianza: {qwen_avg_conf:.3f}")
        
        if qwen_avg_entities == 0:
            print("✅ llama3.2:3b detecta TODAS las entidades (qwen2.5:3b no detectó ninguna)")
        elif llama_avg_entities > qwen_avg_entities:
            print(f"✅ llama3.2:3b detecta {(llama_avg_entities/qwen_avg_entities-1)*100:.1f}% más entidades")
        else:
            print(f"✅ qwen2.5:3b detecta {(qwen_avg_entities/llama_avg_entities-1)*100:.1f}% más entidades")
            
        if llama_avg_conf > qwen_avg_conf:
            print(f"✅ llama3.2:3b tiene {(llama_avg_conf/qwen_avg_conf-1)*100:.1f}% más confianza")
        else:
            print(f"✅ qwen2.5:3b tiene {(qwen_avg_conf/llama_avg_conf-1)*100:.1f}% más confianza")


if __name__ == "__main__":
    main()
