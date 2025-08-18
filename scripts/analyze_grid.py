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
        print(f"[ERROR] Reading {filepath}: {e}")
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "docs": docs,
        "entities": total_entities,
        "avg_conf": avg_conf,
        "entities_set": entities_set,
    }


def parse_cfg_from_name(name):
    # Expect names like results_qwen_chunk40_ov20.jsonl
    m = re.search(r"chunk(\d+)_ov(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def main():
    files = sorted(glob.glob("results_qwen_chunk*_ov*.jsonl"))
    if not files:
        print("No se encontraron archivos de resultados 'results_qwen_chunk*_ov*.jsonl'")
        return

    summaries = []
    union_all = set()
    per_file_sets = {}

    for fp in files:
        base = os.path.basename(fp)
        chunk, ov = parse_cfg_from_name(base)
        res = read_results(fp)
        summaries.append({
            "file": base,
            "chunk": chunk,
            "overlap": ov,
            "docs": res["docs"],
            "entities": res["entities"],
            "avg_conf": res["avg_conf"],
        })
        per_file_sets[base] = res["entities_set"]
        union_all |= res["entities_set"]

    # Sort by entities desc, then avg_conf desc
    summaries.sort(key=lambda x: (x["entities"], x["avg_conf"]), reverse=True)

    print("\n=== RESUMEN POR CONFIGURACIÓN ===")
    print("file, chunk, overlap, docs, entities, avg_conf")
    for s in summaries:
        print(f"{s['file']}, {s['chunk']}, {s['overlap']}, {s['docs']}, {s['entities']}, {s['avg_conf']:.3f}")

    # Unicidad por configuración
    print("\n=== ENTIDADES ÚNICAS POR CONFIGURACIÓN (muestras) ===")
    for base, eset in per_file_sets.items():
        others_union = set().union(*[v for k, v in per_file_sets.items() if k != base]) if len(per_file_sets) > 1 else set()
        unique = eset - others_union
        sample = list(sorted(unique))[:5]
        print(f"{base}: únicas={len(unique)} | ejemplo={sample}")

    # Agregados por chunk y por overlap
    by_chunk = defaultdict(lambda: {"entities": 0, "files": 0})
    by_overlap = defaultdict(lambda: {"entities": 0, "files": 0})
    for s in summaries:
        by_chunk[s["chunk"]]["entities"] += s["entities"]
        by_chunk[s["chunk"]]["files"] += 1
        by_overlap[s["overlap"]]["entities"] += s["entities"]
        by_overlap[s["overlap"]]["files"] += 1

    print("\n=== PROMEDIO ENTIDADES POR CHUNK TARGET ===")
    for ch in sorted(by_chunk):
        avg = by_chunk[ch]["entities"] / max(1, by_chunk[ch]["files"])
        print(f"chunk={ch}: avg_entities={avg:.2f}")

    print("\n=== PROMEDIO ENTIDADES POR OVERLAP ===")
    for ov in sorted(by_overlap):
        avg = by_overlap[ov]["entities"] / max(1, by_overlap[ov]["files"])
        print(f"overlap={ov}: avg_entities={avg:.2f}")


if __name__ == "__main__":
    main()


