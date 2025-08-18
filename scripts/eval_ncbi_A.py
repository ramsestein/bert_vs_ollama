# eval_ncbi_all_in_one.py
# Eval NER para NCBI-disease:
#  A) PubTator -> BIO + seqeval (oficial/comparable a papers; colapsado a 'Disease' por defecto)

import argparse
import csv
import os
import re
import json
from typing import List, Dict, Tuple
from collections import Counter
from tqdm.auto import tqdm
import string
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

GENERIC_BLACKLIST = {
    "disease", "disorder", "defect", "abnormality", "abnormalities",
    "illness", "syndrome", "malignancy", "tumor", "tumour", "cancer"
}

_punct_edge = re.compile(r"(^[\W_]+|[\W_]+$)")  # borra puntuación en bordes
_multi_space = re.compile(r"\s+")
_PUNCT = set(string.punctuation)


# ----------------------------
# Utilidades comunes
# ----------------------------

def auto_device_str(user_choice: str = None) -> str:
    if user_choice in ("cpu", "cuda"):
        return user_choice
    return "cuda" if torch.cuda.is_available() else "cpu"

def dedup_and_suppress_subspans(mentions: List[str]) -> List[str]:
    # Ordena por longitud descendente y elimina subcadenas redundantes
    mentions = sorted(mentions, key=len, reverse=True)
    keep = []
    for m in mentions:
        if any(m in k for k in keep):
            continue
        keep.append(m)
    return keep


PUBTATOR_T_RE = re.compile(r"^(\d+)\|t\|(.*)$")
PUBTATOR_A_RE = re.compile(r"^(\d+)\|a\|(.*)$")

def diagnostico_truncado(pubtator_txt: str, model_dir: str, max_length: int = 512):
    from transformers import AutoTokenizer
    import re
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def whitespace_tokens(text: str):
        return [m.group(0) for m in re.finditer(r"\S+", text)]

    docs = parse_pubtator(pubtator_txt)
    over, total, max_len = 0, 0, 0
    for d in docs:
        toks = whitespace_tokens(d["text"])
        enc = tokenizer(toks, is_split_into_words=True, truncation=False, return_tensors=None)
        n_subtoks = len(enc["input_ids"])
        max_len = max(max_len, n_subtoks)
        total += 1
        if n_subtoks > max_length:
            over += 1
    rate = 100.0 * over / total if total else 0.0
    print(f"[Diag] Docs: {total} | >{max_length} subtokens: {over} ({rate:.1f}%) | max subtokens: {max_len}")


def predict_word_labels_windowed(model, tokenizer, tokens, device, max_length=512, stride=256):
    """
    Etiqueta PALABRAS con ventana deslizante sobre la secuencia de subtokens.
    Ensambla las predicciones palabra a palabra usando word_ids.
    """
    # Mapa de palabra -> primera etiqueta vista (B/I/O) en la primera ventana que la cubre
    word_labels = [None] * len(tokens)
    start_word = 0
    while start_word < len(tokens):
        end_word = min(len(tokens), start_word + stride)  # stride en palabras
        enc = tokenizer(
            tokens[start_word:end_word],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**inputs)
            preds = out.logits.argmax(dim=-1)[0].cpu().tolist()
        id2label = model.config.id2label
        wids = enc.word_ids(0)
        # asignar etiqueta al primer subtoken de cada palabra de la ventana
        seen = set()
        for j, w in enumerate(wids):
            if w is None or w in seen:
                continue
            seen.add(w)
            global_w = start_word + w
            if word_labels[global_w] is None:
                word_labels[global_w] = id2label[int(preds[j])]
        # avanzar ventana
        if end_word == len(tokens):
            break
        start_word = end_word - (stride // 2)  # solape 50%
        if start_word <= 0:
            start_word = end_word
    # Rellena None -> "O"
    return [lbl if lbl is not None else "O" for lbl in word_labels]


def parse_pubtator(path: str) -> List[Dict]:
    """
    Lee PubTator (.txt) y devuelve:
      [{"pmid": str, "text": str, "entities": [(start,end,label), ...]}, ...]
    """
    docs = []
    cur = {"pmid": None, "title": [], "abstract": [], "entities": []}

    def flush():
        nonlocal cur
        if cur["pmid"] is not None:
            text = " ".join([*cur["title"], *cur["abstract"]]).strip()
            docs.append({
                "pmid": cur["pmid"],
                "text": text,
                "entities": cur["entities"]
            })
        cur = {"pmid": None, "title": [], "abstract": [], "entities": []}

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                flush()
                continue

            m_t = PUBTATOR_T_RE.match(line)
            if m_t:
                pmid, title = m_t.groups()
                if cur["pmid"] is not None and pmid != cur["pmid"]:
                    flush()
                cur["pmid"] = pmid
                cur["title"].append(title)
                continue

            m_a = PUBTATOR_A_RE.match(line)
            if m_a:
                pmid, abstract = m_a.groups()
                if cur["pmid"] is not None and pmid != cur["pmid"]:
                    flush()
                    cur["pmid"] = pmid
                elif cur["pmid"] is None:
                    cur["pmid"] = pmid
                cur["abstract"].append(abstract)
                continue

            # anotación: pmid \t start \t end \t mention \t type \t id
            parts = line.split("\t")
            if len(parts) >= 6 and parts[0].isdigit():
                pmid, start, end, mention, etype = parts[0], int(parts[1]), int(parts[2]), parts[3], parts[4]
                if cur["pmid"] is not None and pmid != cur["pmid"]:
                    flush()
                    cur["pmid"] = pmid
                elif cur["pmid"] is None:
                    cur["pmid"] = pmid
                cur["entities"].append((start, end, etype))
                continue

        flush()
    return docs

def collapse_to_disease(entities: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    # Colapsa todas las etiquetas a 'Disease' (práctica común en benchmarks del corpus NCBI)
    return [(s, e, "Disease") for (s, e, _t) in entities]

def whitespace_tokenize_with_offsets(text: str) -> List[Tuple[str,int,int]]:
    """
    Tokeniza por espacios preservando offsets [start,end) por token.
    Devuelve lista de (token, start_char, end_char).
    """
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def normalize_bio_labels(labels, id2label=None, label2id=None):
    """
    Normaliza cualquier variante de labels del modelo a BIO con tipo 'Disease':
    - O, 0, LABEL_0 -> O
    - LABEL_1/LABEL_2 (cuando hay 3 labels) -> B-Disease / I-Disease
    - B-*/I-* (con -, _, mayúsculas) -> B-Disease / I-Disease
    - labels que contengan 'disease' -> mapear prefijo B/I adecuado
    Todo lo demás -> O
    """
    out = []
    # Detecta patrón LABEL_i -> asume 3 clases: 0=O, 1=B, 2=I
    three_class = False
    if id2label:
        # si todos son LABEL_* y hay 3
        if all(str(v).upper().startswith("LABEL_") for v in id2label.values()) and len(id2label) == 3:
            three_class = True

    for lab in labels:
        s = str(lab).strip()
        su = s.upper()

        # casos obvios de O
        if su in {"O", "0", "LABEL_0"}:
            out.append("O")
            continue

        # Mapeo 3-clases por índice
        if three_class and su.startswith("LABEL_"):
            try:
                idx = int(su.split("_")[1])
            except:
                idx = None
            if idx == 1:
                out.append("B-Disease"); continue
            if idx == 2:
                out.append("I-Disease"); continue
            out.append("O"); continue

        # Formatos B-*/I-* o B_*/I_* (cualquier tipo) -> B/I-Disease
        m = re.match(r"^([BI])[-_](.+)$", s, flags=re.IGNORECASE)
        if m:
            prefix = m.group(1).upper()
            out.append(f"{prefix}-Disease"); continue

        # Labels que contienen 'disease' (ej. 'B-DISEASE', 'I_DISEASE')
        if "DISEASE" in su:
            if su.startswith("B") or su.endswith("B"):
                out.append("B-Disease"); continue
            if su.startswith("I") or su.endswith("I"):
                out.append("I-Disease"); continue
            # si no sabemos el prefijo pero contiene DISEASE, aprox B-
            out.append("B-Disease"); continue

        # fallback
        out.append("O")
    return out


def _is_punct(tok: str) -> bool:
    return all(ch in _PUNCT for ch in tok)

def spans_to_bio(tokens_offsets, entities):
    """
    entities: lista de (start, end, etype) con etype en {SpecificDisease, DiseaseClass, CompositeMention, Modifier}
    Devuelve BIO por token respetando el tipo original: B-<etype>/I-<etype>.
    Criterio: cuenta como dentro si hay solape estricto de offsets.
    """
    tags = ["O"] * len(tokens_offsets)
    for (s, e, etype) in entities:
        first = True
        for i, (tok, ts, te) in enumerate(tokens_offsets):
            if _is_punct(tok):
                continue
            if max(ts, s) < min(te, e):  # hay solape
                prefix = "B-" if first else "I-"
                tags[i] = prefix + etype
                first = False
    return tags


def predict_word_labels(model, tokenizer, tokens: List[str], device: str, max_length: int = 512) -> List[str]:
    """
    Predice etiquetas por PALABRA usando tokenizer con is_split_into_words=True.
    Toma la etiqueta del PRIMER subtoken de cada palabra.
    """
    # OJO: mantenemos BatchEncoding para poder usar .word_ids()
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Copiamos a device sin pisar 'enc'
    inputs = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits  # (1, seq_len, num_labels)
        preds = logits.argmax(dim=-1)[0].cpu().tolist()


    word_ids = enc.word_ids(0)  # ahora sí funciona
    id2label = model.config.id2label

    word_labels = []
    prev_w = None
    for j, w in enumerate(word_ids):
        if w is None:
            continue
        if w != prev_w:  # primer subtoken de la palabra
            word_labels.append(id2label[int(preds[j])])
        prev_w = w
    return word_labels


def run_option_a(pubtator_txt: str, model_dir: str, collapse: bool, device: str, max_length: int):
    device = auto_device_str(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()

    docs = parse_pubtator(pubtator_txt)
    y_true, y_pred = [], []

    for doc in tqdm(docs, desc="[A] Evaluando (PubTator->BIO)"):
        text = doc["text"]
        ents = doc["entities"]
        if collapse:
            ents = collapse_to_disease(ents)

        toks_offs = whitespace_tokenize_with_offsets(text)
        tokens = [t for (t, s, e) in toks_offs]
        gold = spans_to_bio(toks_offs, ents)

        pred = predict_word_labels_windowed(model, tokenizer, tokens, device=device, max_length=max_length, stride=256)
        pred = normalize_bio_labels(pred, id2label=model.config.id2label, label2id=model.config.label2id)

        L = min(len(gold), len(pred))
        y_true.append(gold[:L])
        y_pred.append(pred[:L])

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("\n== [A] Métricas (BIO, seqeval, etiqueta colapsada='Disease'={}) ==".format(collapse))
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")
    print("\n-- Informe por clase --")
    print(classification_report(y_true, y_pred, digits=4))
    # Guardar JSON
    results = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "collapse": collapse
    }
    with open("./metrics/results_ncbi.json", "w", encoding="utf-8") as fjson:
        json.dump(results, fjson, indent=2)
    
    # Guardar CSV (append si ya existe)
    csv_path = "./metrics/results_ncbi.csv"
    header = ["precision", "recall", "f1", "collapse"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(results)

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluación NER NCBI-disease en dos modos: A) PubTator->BIO+seqeval  B) JSONL sin offsets (string-match)")
    # Parámetros Opción A
    ap.add_argument("--pubtator_txt", type=str, help="Ruta a PubTator (.txt), p.ej. ./datasets/NCBItestset_corpus.txt")
    ap.add_argument("--model_dir", type=str, required=True, help="Ruta del modelo HF (local o hub), p.ej. ./biobert_ncbi_ner")
    ap.add_argument("--device", type=str, default='cpu', help="'cuda' o 'cpu'")
    ap.add_argument("--no_collapse", action="store_true", help="No colapsar clases a 'Disease' en Opción A")
    ap.add_argument("--max_length", type=int, default=512, help="Máx. longitud para tokenizador en Opción A")

    args = ap.parse_args()

    run_option_a(
            pubtator_txt=args.pubtator_txt,
            model_dir=args.model_dir,
            collapse=not args.no_collapse,
            device=args.device,
            max_length=args.max_length
        )

if __name__ == "__main__":
    main()
