# infer_ncbi.py
# Inferencia NER (NCBI) + validación contra JSONL develop
# - Usa GPU (si disponible) con fp16
# - Reutiliza pipelines (sin recargar modelo cada vez)
# - Limpia '##' en menciones y normaliza guiones/espacios

import json
import sys
import argparse
import re
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# ====== Configuración de dispositivo (GPU si disponible) ======
USE_CUDA = torch.cuda.is_available()
DEVICE_IDX = 0 if USE_CUDA else -1
DTYPE = torch.float16 if USE_CUDA else torch.float32

# Heurísticas para recortar FPs
MIN_SCORE = 0.8      # umbral de confianza mínimo
MIN_LEN   = 5         # longitud mínima de mención (caracteres) tras limpieza
MAX_TOKS  = 6         # máximo de tokens en una mención
GENERIC_BLACKLIST = {
    "disease","disorder","defect","abnormality","abnormalities","syndrome",
    "malignancy","cancer","tumor","tumour","condition",
    "chronic","acute","severe","mild","familial","hereditary","genetic",
    "bilateral","congenital","recessive","dominant","enzyme","deficiency","red"
}
STOP_ADJ = {"familial","hereditary","chronic","acute","severe","mild",
            "autosomal","dominant","recessive","genetic","inherited",
            "bilateral","congenital"}


DEFAULT_MODEL = "./biobert_ncbi_ner"  # o tu carpeta local
MAX_LENGTH = 652

# ----------------------
# Limpieza y normalización de menciones
# ----------------------
_punct_edge = re.compile(r"(^[\W_]+|[\W_]+$)", flags=re.UNICODE)
_multi_space = re.compile(r"\s+", flags=re.UNICODE)

def clean_mention(s: str) -> str:
    """Quita ## de subtokens, normaliza espacios/guiones y recorta puntuación de bordes."""
    if s is None:
        return ""
    x = s.replace("##", "")
    # normalización de guiones y espacios
    x = x.replace(" – ", "-").replace(" — ", "-")
    x = x.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    x = _multi_space.sub(" ", x)
    x = _punct_edge.sub("", x)
    return x.strip()

def head_of(mention: str) -> str:
    toks = [t for t in re.findall(r"\w+", mention.lower()) if t not in STOP_ADJ]
    return toks[-1] if toks else mention.lower()

def tok_count(s: str) -> int:
    return 0 if not s else len(re.findall(r"\w+|[^\w\s]", s))

def norm_for_match(s: str) -> str:
    """Normalización para matching (case-insensitive)."""
    return clean_mention(s).lower()

# ----------------------
# Mapeo de etiquetas del modelo -> BIO
# ----------------------
def map_label_to_bio(label_str: str) -> str:
    s = str(label_str).strip()
    su = s.upper()
    if su == "O":
        return "O"
    if s in ("B-Disease", "I-Disease"):
        return s
    if su in ("B-DISEASE", "I-DISEASE"):
        return f"{su[0]}-Disease"
    if su.startswith("LABEL_"):
        # interpreta 0=O, 1=B, 2=I (patrón común en checkpoints)
        try:
            idx = int(su.split("_")[1])
        except Exception:
            idx = None
        return {0: "O", 1: "B-Disease", 2: "I-Disease"}.get(idx, "O")
    return "O"

# ----------------------
# Agregación y conversión a JSON final
# ----------------------
def to_entities_for_final_json(spans):
    ents = []
    for e in spans:
        bio = map_label_to_bio(e.get("entity_group"))
        if not (bio.startswith("B-") or bio.startswith("I-")):
            continue
        score = float(e.get("score", 0.0))
        if score < MIN_SCORE:
            continue
        mention = clean_mention(e.get("word", ""))
        if not mention or len(mention) < MIN_LEN:
            continue
        # descarta menciones de 1 token “demasiado genéricas” (p. ej. 'enzyme', 'deficiency', 'red')
        if mention.lower() in GENERIC_BLACKLIST:
            continue
        # límites por número de tokens (evita spans kilométricos)
        if tok_count(mention) > MAX_TOKS:
            continue
        ents.append({"texto": mention, "tipo": "SpecificDisease"})
    return ents


def aggregate_tokens_to_spans(tokens):
    """
    tokens: salida del pipeline con aggregation_strategy=None (token a token).
    Une tokens contiguos no-O en un solo span (limpia '##' al concatenar).
    """
    spans, cur = [], None
    for t in tokens:
        lab = t.get("entity")
        bio = map_label_to_bio(lab)
        is_ent = bio.startswith(("B-", "I-"))
        w = clean_mention(t.get("word", ""))
        if not is_ent:
            if cur is not None:
                spans.append(cur); cur = None
            continue
        if cur is None:
            cur = {
                "entity_group": lab,
                "score_sum": float(t.get("score", 0.0)),
                "score_n": 1,
                "word": w,
                "start": int(t.get("start", 0)),
                "end": int(t.get("end", 0)),
            }
        else:
            cur["end"] = int(t.get("end", cur["end"]))
            cur["word"] += w
            cur["score_sum"] += float(t.get("score", 0.0))
            cur["score_n"] += 1
    if cur is not None:
        spans.append(cur)

    out = []
    for s in spans:
        out.append({
            "entity_group": s["entity_group"],
            "score": s["score_sum"] / max(1, s["score_n"]),
            "word": s["word"],
            "start": s["start"],
            "end": s["end"],
        })
    return out

# ----------------------
# Inferir un texto -> registro final (reutiliza pipelines)
# ----------------------
def infer_record(pmid: str, text: str, ner_simple, ner_tokens):
    # Primer intento: spans agregados
    try:
        spans = ner_simple(text, truncation=True, max_length=MAX_LENGTH)
    except TypeError:
        spans = ner_simple(text)

    entidades = to_entities_for_final_json(spans)
    if not entidades:
        # FallBack: token a token y reagrupar
        try:
            toks = ner_tokens(text, truncation=True, max_length=MAX_LENGTH)
        except TypeError:
            toks = ner_tokens(text)
        spans2 = aggregate_tokens_to_spans(toks)
        entidades = to_entities_for_final_json(spans2)

    rec = {"PMID": pmid, "Texto": text, "Entidad": entidades}
    # NMS textual a posteriori
    if rec["Entidad"]:
        mencs = [x["texto"] for x in rec["Entidad"]]
        mencs = suppress_substrings(dedup_list(mencs))
        rec["Entidad"] = [{"texto": m, "tipo": "SpecificDisease"} for m in mencs]
    return rec


# ----------------------
# Evaluación contra JSONL develop
# ----------------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def mentions_from_record(rec):
    """Extrae menciones (gold) de un registro develop: normalizadas para matching; ignora tipo."""
    ents = rec.get("Entidad", []) or []
    return [norm_for_match(e.get("texto", "")) for e in ents if e.get("texto")]

def evaluate_against_develop(dev_jsonl: str, ner_simple, ner_tokens, out_jsonl: str, dump_diffs: str = None):
    """
    Evalúa por mención (case-insensitive), multiconjunto.
    Además de la métrica STRICT (exact string), calcula RELAXED por head léxica.
    """
    total_tp = total_pred = total_gold = 0
    total_tp_rel = total_pred_rel = total_gold_rel = 0
    diffs = []  # dump opcional

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for rec in load_jsonl(dev_jsonl):
            pmid = str(rec.get("PMID", "NA"))
            text = rec.get("Texto", "")
            gold_mentions = [m for m in mentions_from_record(rec) if m]

            # inferimos (reutilizando pipelines)
            our = infer_record(pmid, text, ner_simple, ner_tokens)
            our_mentions = [norm_for_match(e["texto"]) for e in our.get("Entidad", [])]

            # guardar predicciones nuestras (JSONL)
            fout.write(json.dumps(our, ensure_ascii=False) + "\n")

            # ===== STRICT =====
            c_pred = Counter(our_mentions)
            c_gold = Counter(gold_mentions)
            c_tp   = c_pred & c_gold
            tp     = sum(c_tp.values())
            total_tp   += tp
            total_pred += sum(c_pred.values())
            total_gold += sum(c_gold.values())

            # ===== RELAXED (por head) =====
            hp = Counter(map(head_of, our_mentions))
            hg = Counter(map(head_of, gold_mentions))
            c_tp_rel = hp & hg
            tp_rel   = sum(c_tp_rel.values())
            total_tp_rel   += tp_rel
            total_pred_rel += sum(hp.values())
            total_gold_rel += sum(hg.values())

            if dump_diffs is not None:
                fps = list((c_pred - c_tp).elements())
                fns = list((c_gold - c_tp).elements())
                diffs.append({
                    "PMID": pmid,
                    "tp": list(c_tp.elements())[:200],
                    "fp": fps[:200],
                    "fn": fns[:200],
                })

    # STRICT
    precision = total_tp / total_pred if total_pred else 0.0
    recall    = total_tp / total_gold if total_gold else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # RELAXED
    precision_rel = total_tp_rel / total_pred_rel if total_pred_rel else 0.0
    recall_rel    = total_tp_rel / total_gold_rel if total_gold_rel else 0.0
    f1_rel        = 2 * precision_rel * recall_rel / (precision_rel + recall_rel) if (precision_rel + recall_rel) else 0.0

    if dump_diffs is not None:
        with open(dump_diffs, "w", encoding="utf-8") as fd:
            for d in diffs:
                fd.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Devuelve ambas métricas + conteos strict (por compatibilidad)
    return (precision, recall, f1, total_pred, total_gold, total_tp,
            precision_rel, recall_rel, f1_rel)


def dedup_list(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def suppress_substrings(mentions):
    # conserva solo las menciones más largas cuando una contiene completamente a otra
    ms = sorted(mentions, key=len, reverse=True)
    keep = []
    for m in ms:
        if any(m != k and m in k for k in keep):
            continue
        keep.append(m)
    return keep

# ----------------------
# CLI
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="Inferencia NER (NCBI) y validación contra JSONL develop")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Modo 1: inferir un solo texto -> JSONL final
    p_inf = sub.add_parser("infer", help="Inferir un solo texto y guardar JSON final (JSONL)")
    p_inf.add_argument("--pmid", required=True, help="Identificador del documento (PMID)")
    p_inf.add_argument("--text", help="Texto a procesar; si se omite, se pedirá por input()")
    p_inf.add_argument("--model", default=DEFAULT_MODEL, help="Modelo HF o carpeta local")
    p_inf.add_argument("--out", default="inference.jsonl", help="Salida JSONL (append)")

    # Modo 2: evaluar contra develop JSONL
    p_eval = sub.add_parser("eval", help="Evaluar contra JSONL develop (PMID, Texto, Entidad[])")
    p_eval.add_argument("--develop_jsonl", required=True, help="Ruta a ncbi_develop.jsonl")
    p_eval.add_argument("--model", default=DEFAULT_MODEL, help="Modelo HF o carpeta local")
    p_eval.add_argument("--out_pred", default="inference_on_develop.jsonl", help="Inferencias nuestras (JSONL)")
    p_eval.add_argument("--dump_diffs", default=None, help="Opcional: JSONL con TP/FP/FN por PMID")

    args = ap.parse_args()

    # ====== Carga ÚNICA del modelo/tokenizer y creación de pipelines (GPU si hay) ======
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, torch_dtype=DTYPE)
    if USE_CUDA:
        model = model.to("cuda").eval()

    ner_simple = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=DEVICE_IDX
    )
    ner_tokens = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=None,
        device=DEVICE_IDX
    )

    if args.cmd == "infer":
        pmid = args.pmid
        text = args.text if args.text is not None else input("Introduce el texto a procesar: ").strip()

        rec = infer_record(pmid, text, ner_simple, ner_tokens)

        print("\n=== JSON final ===")
        print(json.dumps(rec, indent=2, ensure_ascii=False))

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\n✅ Guardado en {args.out}")

    elif args.cmd == "eval":
        res = evaluate_against_develop(
            dev_jsonl=args.develop_jsonl,
            ner_simple=ner_simple,
            ner_tokens=ner_tokens,
            out_jsonl=args.out_pred,
            dump_diffs=args.dump_diffs
        )
        (p, r, f1, npred, ngold, ntp, p_rel, r_rel, f1_rel) = res
        
        print("\n== Métricas STRICT (match exacto por mención; case-insensitive) ==")
        print(f"Predicciones: {npred}")
        print(f"Gold:        {ngold}")
        print(f"TP:          {ntp}")
        print(f"Precision:   {p:.4f}")
        print(f"Recall:      {r:.4f}")
        print(f"F1:          {f1:.4f}")
        
        print("\n== Métricas RELAXED (head match; case-insensitive) ==")
        print(f"Precision:   {p_rel:.4f}")
        print(f"Recall:      {r_rel:.4f}")
        print(f"F1:          {f1_rel:.4f}")
        
        if args.dump_diffs:
            print(f"\nDetalles por PMID en: {args.dump_diffs}")


if __name__ == "__main__":
    main()


# python infer_ncbi.py eval --develop_jsonl ./datasets/ncbi_develop.jsonl --out_pred results_develop_bert_ner.jsonl --dump_diffs diffs_results_develop_bert_ner.jsonl

