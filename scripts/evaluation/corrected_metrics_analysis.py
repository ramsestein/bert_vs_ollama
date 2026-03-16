#!/usr/bin/env python3
"""
Corrected NER Metrics Analysis
================================
Analiza FP y FN con corrección manual (NER_analysis.jsonl), generando:

1. Métricas originales (baseline de comparación)
2. Métricas corregidas (sin agrupación por código)
3. Métricas corregidas agrupadas por código ICD10
4. Métricas 1-3 excluyendo entidades de fumador/exfumador (F17.210 / Z87.891)
5. Análisis exhaustivo de errores por estrategia/modelo:
   - Cuántos FP fueron detectados SOLO por regex
   - Cuántos FP por combinación de modelos
   - Qué modelos generan más FP
   - Distribución de FP por código y por modelo
   - FP que se convierten en TP con corrección manual
   - FN que son verdaderos errores vs FN aceptables
"""

import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FINAL_METRICS_DIR = BASE_DIR / "final_metrics"

MANUAL_ANALYSIS_FILE  = FINAL_METRICS_DIR / "NER_analysis.jsonl"
FP_ANALYSIS_FILE      = FINAL_METRICS_DIR / "false_positives_analysis_icd10_final_all.json"
FN_ANALYSIS_FILE      = FINAL_METRICS_DIR / "false_negatives_analysis_icd10_final_all.json"
EVAL_RESULTS_FILE     = FINAL_METRICS_DIR / "ner_evaluation_results_icd10_final_all.json"
PREDICTIONS_FILE      = FINAL_METRICS_DIR / "spanish_clinical_filtered_predictions_final_all.jsonl"
OUTPUT_FILE           = FINAL_METRICS_DIR / "corrected_metrics_analysis.json"

# Códigos relacionados con tabaco (para análisis sin fumador/exfumador)
SMOKING_CODES = {"F17.210", "Z87.891"}

# Entidades de texto relacionadas con tabaco
SMOKING_ENTITIES = {
    "fumador", "fumadora", "ex-fumador", "ex fumador", "exfumador",
    "exfumadora", "ex-fumadora", "ex fumadora", "tabaquismo",
    "ex - fumador", "ex - fumadora"
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Normaliza texto para comparación."""
    return re.sub(r'\s+', ' ', text.lower().strip()) if text else ""


def extract_code_from_entity_str(entity_str: str) -> str:
    """Extrae el código ICD10 de cadenas tipo 'texto (CÓDIGO)'."""
    m = re.search(r'\(([A-Z0-9.]+)\)\s*$', entity_str)
    return m.group(1) if m else ""


def extract_text_from_entity_str(entity_str: str) -> str:
    """Extrae el texto de cadenas tipo 'texto (CÓDIGO)'."""
    return re.sub(r'\s*\([A-Z0-9.]+\)\s*$', '', entity_str).strip()


def compute_metrics(tp: int, fp: int, fn: int) -> Dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_manual_analysis(path: Path) -> Dict:
    """
    Carga NER_analysis.jsonl.
    Devuelve dos estructuras:
      fp_corrections[(pmid, entity_text, icd_code)] -> is_error (bool)
      fn_corrections[(pmid, entity_text, icd_code)] -> is_error (bool)
    """
    fp_corrections = {}
    fn_corrections = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            pmid        = rec.get("pmid", "")
            entity_str  = rec.get("predicted_entity", "")
            error_type  = rec.get("error_type", "")
            is_error    = rec.get("is_error", True)

            text = extract_text_from_entity_str(entity_str)
            code = extract_code_from_entity_str(entity_str)

            key = (pmid, normalize(text), code)

            if error_type == "FP":
                fp_corrections[key] = is_error
            elif error_type == "FN":
                fn_corrections[key] = is_error

    return fp_corrections, fn_corrections


def load_fp_analysis(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("false_positives", [])


def load_fn_analysis(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("false_negatives", [])


def load_eval_results(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: Path) -> List[Dict]:
    preds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    preds.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# CORRECTED METRICS (entity-level, ungrouped)
# ──────────────────────────────────────────────────────────────────────────────

def compute_entity_tp_baseline(eval_orig: Dict, exclude_smoking: bool = False) -> int:
    """
    Calcula el nº de TPs a nivel de entidad (sin agrupar por código) a partir
    de los detailed_results del fichero de evaluación original.

    Para cada documento, por cada código que fue TP, cuenta cuántas entidades
    de texto distintas se predijeron para ese código (todas son TP a nivel
    de entidad, aunque a nivel de código-documento cuenten como 1 solo TP).

    Ejemplo: si 'fa', 'fibrilación auricular' y 'acxfa' se predicen en el mismo
    documento y I48.91 está en referencia → 3 TPs a nivel entidad, 1 TP agrupado.
    """
    entity_tp = 0
    for doc in eval_orig.get("detailed_results", []):
        tp_codes = set(doc.get("tp_codes", []))
        entities_by_code = doc.get("predicted_entities_by_code", {})
        for code in tp_codes:
            if exclude_smoking and code in SMOKING_CODES:
                continue
            entity_tp += len(entities_by_code.get(code, []))
    return entity_tp


def build_corrected_counts_ungrouped(
    fp_list: List[Dict],
    fn_list: List[Dict],
    fp_corrections: Dict,
    fn_corrections: Dict,
    original_eval: Dict,
    exclude_smoking: bool = False
) -> Tuple[int, int, int]:
    """
    Calcula TP, FP, FN corregidos a nivel de entidad (sin agrupar por código ICD10).

    Sin agrupar significa: si en un documento se predicen 'fa' y 'acxfa' y ambas
    son FP para I48.91, se cuentan como 2 FPs independientes (no 1).
    Igualmente, si 'fa' y 'fibrilación auricular' son TP, son 2 TPs separados.

    Base de cálculo:
    - FP_base  = nº de entidades FP en el análisis (nivel texto, sin colapsar)
    - TP_base  = nº de entidades TP a nivel texto (desde detailed_results)
    - FN_base  = nº de códigos perdidos (igual que el nivel agrupado)

    Lógica de corrección manual:
    - FP con is_error=False  --> pasa a ser TP (corrección positiva)
    - FP con is_error=True   --> sigue siendo FP
    - FN con is_error=False  --> se elimina del cómputo de FN
    - FN con is_error=True   --> sigue siendo FN
    """
    # Ajustes basados en correcciones manuales
    fp_converted_to_tp = 0   # FPs que en realidad son correctos
    fp_confirmed       = 0   # FPs confirmados como errores

    fn_real_errors     = 0   # FNs confirmados como errores reales
    fn_acceptable      = 0   # FNs que se aceptan (no son error grave)

    for fp in fp_list:
        code   = fp.get("predicted_code", "")
        entity = fp.get("predicted_entity", "")
        pmid   = fp.get("PMID", "")

        if exclude_smoking and code in SMOKING_CODES:
            continue

        key = (pmid, normalize(entity), code)
        if key in fp_corrections:
            if not fp_corrections[key]:
                fp_converted_to_tp += 1
            else:
                fp_confirmed += 1
        else:
            # Sin corrección manual → mantenemos como FP
            fp_confirmed += 1

    for fn in fn_list:
        code   = fn.get("benchmark_code", "")
        entities = fn.get("benchmark_entities", [])
        pmid   = fn.get("PMID", "")

        if exclude_smoking and code in SMOKING_CODES:
            continue

        # Intentamos buscar en correcciones. Puede haber varias entidades de benchmark.
        found = False
        is_real_error = True
        for ent in entities:
            key = (pmid, normalize(ent), code)
            # También intentamos con el pmid con prefijo "0" extra como en NER_analysis
            key2 = ("0" + pmid, normalize(ent), code)
            if key in fn_corrections:
                found = True
                is_real_error = fn_corrections[key]
                break
            elif key2 in fn_corrections:
                found = True
                is_real_error = fn_corrections[key2]
                break

        if not found:
            # Sin corrección → asumimos error real
            is_real_error = True

        if is_real_error:
            fn_real_errors += 1
        else:
            fn_acceptable += 1

    # Cálculo final a nivel entidad (sin agrupar por código)
    # Base TP = nº de entidades que el sistema predijo correctamente (nivel texto)
    # Esto puede ser > TP agrupado porque el mismo código puede aparecer con varias
    # formas textuales en un documento (p.ej. 'fa' + 'fibrilación auricular' → 2 TP entidad, 1 TP agrupado)
    base_tp_entity = compute_entity_tp_baseline(original_eval, exclude_smoking)

    # Base FP = nº de entidades FP a nivel texto (fp_confirmed + fp_converted_to_tp)
    # Este es el total de FPs en el fichero de análisis (sin los de smoking si exclude)
    # fp_confirmed + fp_converted_to_tp = todas las FP entidades procesadas

    corrected_tp = base_tp_entity + fp_converted_to_tp
    corrected_fp = fp_confirmed
    corrected_fn = fn_real_errors

    return corrected_tp, corrected_fp, corrected_fn


# ──────────────────────────────────────────────────────────────────────────────
# CORRECTED METRICS GROUPED BY ICD10 CODE
# ──────────────────────────────────────────────────────────────────────────────

def build_corrected_counts_by_code(
    fp_list: List[Dict],
    fn_list: List[Dict],
    fp_corrections: Dict,
    fn_corrections: Dict,
    original_eval: Dict,
    exclude_smoking: bool = False
) -> Dict[str, Dict]:
    """
    Calcula TP, FP, FN corregidos agrupados por código ICD10.

    Agrupado significa: si para un PMID+código hay múltiples entidades FP,
    todas se colapsan en un solo evento (TP o FP a nivel de código-documento).
    Esto es coherente con la métrica original que ya agrupa por código/documento.

    La corrección manual a nivel de entidad se traslada al nivel de código:
    - Si CUALQUIER entidad del mismo código+PMID es marcada como is_error=False
      (es decir, "no era FP real"), el código-documento se convierte en TP
      (a menos que TODAS las entidades sean errores reales).
    """
    # Reconstruir métricas originales por código desde el fichero de evaluación
    original_by_code = original_eval.get("icd10_metrics", {})

    # Agrupar FPs por (pmid, code)
    fp_by_pmid_code: Dict = defaultdict(list)
    for fp in fp_list:
        code = fp.get("predicted_code", "")
        pmid = fp.get("PMID", "")
        if exclude_smoking and code in SMOKING_CODES:
            continue
        fp_by_pmid_code[(pmid, code)].append(fp)

    # Para cada grupo (pmid, code), determinar si alguna entidad fue corregida a TP
    fp_adjustments_by_code: Dict[str, int] = defaultdict(int)  # código -> nº de conversiones FP→TP
    fp_confirmed_by_code:   Dict[str, int] = defaultdict(int)  # código -> FP confirmados

    for (pmid, code), fps in fp_by_pmid_code.items():
        # Si al menos una entidad en el grupo es is_error=False, el código-doc pasa a TP
        any_not_error = False
        for fp in fps:
            entity = fp.get("predicted_entity", "")
            key = (pmid, normalize(entity), code)
            if key in fp_corrections and not fp_corrections[key]:
                any_not_error = True
                break

        if any_not_error:
            fp_adjustments_by_code[code] += 1  # convierte 1 FP (código-doc) a TP
        else:
            fp_confirmed_by_code[code] += 1

    # Agrupar FNs por (pmid, code)
    fn_real_by_code:       Dict[str, int] = defaultdict(int)
    fn_acceptable_by_code: Dict[str, int] = defaultdict(int)

    for fn in fn_list:
        code     = fn.get("benchmark_code", "")
        entities = fn.get("benchmark_entities", [])
        pmid     = fn.get("PMID", "")

        if exclude_smoking and code in SMOKING_CODES:
            continue

        # Buscar en correcciones
        found         = False
        is_real_error = True
        for ent in entities:
            key  = (pmid,       normalize(ent), code)
            key2 = ("0" + pmid, normalize(ent), code)
            if key in fn_corrections:
                found = True
                is_real_error = fn_corrections[key]
                break
            elif key2 in fn_corrections:
                found = True
                is_real_error = fn_corrections[key2]
                break

        if not found:
            is_real_error = True

        if is_real_error:
            fn_real_by_code[code] += 1
        else:
            fn_acceptable_by_code[code] += 1

    # Construir métricas corregidas por código
    all_codes = set(original_by_code.keys())
    if not exclude_smoking:
        all_codes |= {code for pmid, code in fp_by_pmid_code.keys()}  # por si hay códigos nuevos (raro)

    result = {}
    for code in sorted(all_codes):
        if exclude_smoking and code in SMOKING_CODES:
            continue

        orig = original_by_code.get(code, {"tp": 0, "fp": 0, "fn": 0})
        orig_tp = orig["tp"]
        orig_fp = orig["fp"]
        orig_fn = orig["fn"]

        converted = fp_adjustments_by_code.get(code, 0)
        fn_real   = fn_real_by_code.get(code, 0)

        corr_tp = orig_tp + converted
        corr_fp = orig_fp - converted
        corr_fn = fn_real

        metrics = compute_metrics(corr_tp, corr_fp, corr_fn)
        metrics["original_tp"] = orig_tp
        metrics["original_fp"] = orig_fp
        metrics["original_fn"] = orig_fn
        metrics["fp_converted_to_tp"] = converted
        metrics["fn_real_errors"]      = fn_real
        metrics["fn_acceptable"]       = fn_acceptable_by_code.get(code, 0)
        result[code] = metrics

    return result


# ──────────────────────────────────────────────────────────────────────────────
# STRATEGY / MODEL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_strategy_errors(
    fp_list: List[Dict],
    fn_list: List[Dict],
    fp_corrections: Dict,
    fn_corrections: Dict,
    predictions: List[Dict],
    exclude_smoking: bool = False
) -> Dict:
    """
    Análisis exhaustivo de errores por estrategia/modelo.

    Para los FP analiza:
    - Cuántos FP detectó SOLO regex (ningún modelo LLM)
    - Cuántos FP detectó SOLO modelos LLM (no regex)
    - Cuántos FP detectaron varios modelos
    - Por modelo: total FP, FP confirmados, FP convertidos a TP
    - Distribución de FP por código y modelo
    - FP que SON errores reales vs los que son "falsos FP" (corregidos)

    Para los FN analiza:
    - Cuántos FN son errores reales
    - Caracterización de las entidades no detectadas
    """

    # Construir mapa de estrategias por entidad desde predictions
    # estructura: strategies_map[(pmid, normalize(texto))] -> set de estrategias
    strategies_map = {}
    for pred in predictions:
        pmid = str(pred.get("PMID", ""))
        for ent in pred.get("Entidad", []):
            if not isinstance(ent, dict):
                continue
            texto      = normalize(ent.get("texto", ""))
            strategies = set(ent.get("strategies", []))
            key = (pmid, texto)
            if key not in strategies_map:
                strategies_map[key] = strategies
            else:
                strategies_map[key] |= strategies

    ALL_MODELS = {"regex", "gemma3_balanced", "gemma3_high_precision",
                  "gemma3_max_sensitivity", "qwen25_diversity"}

    # ── FP analysis ──────────────────────────────────────────────────────────
    fp_only_regex        = []   # FP detectado SOLO por regex
    fp_only_llm          = []   # FP detectado SOLO por LLMs (sin regex)
    fp_multi_model       = []   # FP detectado por regex + ≥1 LLM
    fp_by_model          = defaultdict(list)   # model -> lista de FPs
    fp_real_by_model     = defaultdict(int)    # model -> nº FP reales
    fp_corrected_by_model= defaultdict(int)    # model -> nº FP convertidos a TP
    fp_by_code_model     = defaultdict(lambda: defaultdict(int))  # code -> model -> count
    fp_detail_records    = []

    for fp in fp_list:
        code   = fp.get("predicted_code", "")
        entity = fp.get("predicted_entity", "")
        pmid   = fp.get("PMID", "")

        if exclude_smoking and code in SMOKING_CODES:
            continue

        key       = (pmid, normalize(entity), code)
        strat_key = (pmid, normalize(entity))
        strategies_used = strategies_map.get(strat_key, set(fp.get("strategies", [])))

        # Corrección manual
        is_real_fp = fp_corrections.get(key, True)  # sin corrección = error real

        # Clasificar por quién lo detectó
        has_regex = "regex" in strategies_used
        llm_strats = strategies_used - {"regex"}
        has_llm   = len(llm_strats) > 0

        if has_regex and not has_llm:
            detection_type = "solo_regex"
            fp_only_regex.append(fp)
        elif not has_regex and has_llm:
            detection_type = "solo_llm"
            fp_only_llm.append(fp)
        elif has_regex and has_llm:
            detection_type = "regex_y_llm"
            fp_multi_model.append(fp)
        else:
            detection_type = "desconocido"

        record = {
            "pmid": pmid,
            "entity": entity,
            "code": code,
            "is_real_fp": is_real_fp,
            "strategies": sorted(strategies_used),
            "detection_type": detection_type
        }
        fp_detail_records.append(record)

        for strat in strategies_used:
            fp_by_model[strat].append(fp)
            fp_by_code_model[code][strat] += 1
            if is_real_fp:
                fp_real_by_model[strat] += 1
            else:
                fp_corrected_by_model[strat] += 1

    # Estadísticas globales de FP por tipo de detección
    def summarize_fp_group(fp_group, description):
        total = len(fp_group)
        real_errors = 0
        corrected   = 0
        by_code     = defaultdict(int)
        for fp in fp_group:
            code   = fp.get("predicted_code", "")
            entity = fp.get("predicted_entity", "")
            pmid   = fp.get("PMID", "")
            key    = (pmid, normalize(entity), code)
            is_real = fp_corrections.get(key, True)
            if is_real:
                real_errors += 1
            else:
                corrected += 1
            by_code[code] += 1
        return {
            "description": description,
            "total_fp": total,
            "real_errors": real_errors,
            "corrected_to_tp": corrected,
            "by_code": dict(sorted(by_code.items(), key=lambda x: x[1], reverse=True))
        }

    fp_solo_regex_summary  = summarize_fp_group(fp_only_regex,  "FP detectados SOLO por regex")
    fp_solo_llm_summary    = summarize_fp_group(fp_only_llm,    "FP detectados SOLO por modelos LLM")
    fp_regex_llm_summary   = summarize_fp_group(fp_multi_model, "FP detectados por regex + ≥1 LLM")

    # Métricas por modelo
    model_stats = {}
    for model in sorted(ALL_MODELS):
        fps    = fp_by_model.get(model, [])
        total  = len(fps)
        real   = fp_real_by_model.get(model, 0)
        corr   = fp_corrected_by_model.get(model, 0)
        by_cod = {c: fp_by_code_model[c].get(model, 0)
                  for c in fp_by_code_model if fp_by_code_model[c].get(model, 0) > 0}
        model_stats[model] = {
            "total_fp": total,
            "real_fp_errors": real,
            "fp_corrected_to_tp": corr,
            "fp_error_rate": round(real / total, 4) if total > 0 else 0.0,
            "by_code": dict(sorted(by_cod.items(), key=lambda x: x[1], reverse=True))
        }

    # Ranking de modelos por nº de FP reales que generaron
    ranking_by_real_fp = sorted(
        [(m, s["real_fp_errors"]) for m, s in model_stats.items()],
        key=lambda x: x[1], reverse=True
    )

    # Análisis de solapamiento: cuántos FP son compartidos entre N modelos
    overlap_distribution = Counter()
    for rec in fp_detail_records:
        n = len(rec["strategies"])
        overlap_distribution[n] += 1

    # ── FN analysis ──────────────────────────────────────────────────────────
    fn_real_errors_list   = []
    fn_acceptable_list    = []

    fn_by_code            = defaultdict(lambda: {"real": 0, "acceptable": 0})
    fn_entity_examples    = defaultdict(list)  # code -> [entity_text, ...]

    for fn in fn_list:
        code     = fn.get("benchmark_code", "")
        entities = fn.get("benchmark_entities", [])
        pmid     = fn.get("PMID", "")

        if exclude_smoking and code in SMOKING_CODES:
            continue

        found         = False
        is_real_error = True
        matched_entity = entities[0] if entities else ""

        for ent in entities:
            key  = (pmid,       normalize(ent), code)
            key2 = ("0" + pmid, normalize(ent), code)
            if key in fn_corrections:
                found = True
                is_real_error = fn_corrections[key]
                matched_entity = ent
                break
            elif key2 in fn_corrections:
                found = True
                is_real_error = fn_corrections[key2]
                matched_entity = ent
                break

        if not found:
            is_real_error = True

        record = {
            "pmid": pmid,
            "code": code,
            "entity": matched_entity,
            "is_real_error": is_real_error
        }

        if is_real_error:
            fn_real_errors_list.append(record)
            fn_by_code[code]["real"] += 1
        else:
            fn_acceptable_list.append(record)
            fn_by_code[code]["acceptable"] += 1

        fn_entity_examples[code].append(matched_entity)

    fn_summary_by_code = {
        code: {
            "real_errors":  v["real"],
            "acceptable":   v["acceptable"],
            "total":        v["real"] + v["acceptable"],
            "entity_examples": list(set(fn_entity_examples[code]))[:10]
        }
        for code, v in sorted(fn_by_code.items())
    }

    # ── Análisis específico: ¿qué combinaciones de modelos generan FP? ──────
    combination_counter = Counter()
    combination_real_error = defaultdict(int)
    for rec in fp_detail_records:
        combo = tuple(sorted(rec["strategies"]))
        combination_counter[combo] += 1
        if rec["is_real_fp"]:
            combination_real_error[combo] += 1

    top_combinations = [
        {
            "strategies": list(combo),
            "total_fp": count,
            "real_errors": combination_real_error[combo],
            "corrected": count - combination_real_error[combo]
        }
        for combo, count in combination_counter.most_common(20)
    ]

    return {
        "fp_by_detection_type": {
            "solo_regex":   fp_solo_regex_summary,
            "solo_llm":     fp_solo_llm_summary,
            "regex_y_llm":  fp_regex_llm_summary,
        },
        "fp_by_model": model_stats,
        "model_ranking_by_real_fp": [
            {"model": m, "real_fp_errors": n}
            for m, n in ranking_by_real_fp
        ],
        "fp_overlap_distribution": {
            f"{n}_strategies": count
            for n, count in sorted(overlap_distribution.items())
        },
        "top_fp_combinations": top_combinations,
        "fn_analysis": {
            "total_fn": len(fn_real_errors_list) + len(fn_acceptable_list),
            "real_errors": len(fn_real_errors_list),
            "acceptable_fn": len(fn_acceptable_list),
            "by_code": fn_summary_by_code,
        },
        "fp_detail_records": fp_detail_records,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def generate_plots(
    orig_metrics: Dict,
    orig_metrics_nsmk: Dict,
    corr_metrics: Dict,
    corr_metrics_nsmk: Dict,
    corr_grouped_overall: Dict,
    corr_grouped_overall_nsmk: Dict,
    corr_by_code: Dict,
    eval_orig: Dict,
    strategy_analysis: Dict,
    output_dir: Path,
):
    """Genera y guarda gráficos del análisis de métricas corregidas."""
    import matplotlib
    matplotlib.use("Agg")  # backend no interactivo
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Paleta
    C_ORIG  = "#E74C3C"
    C_UNGR  = "#3498DB"
    C_GRP   = "#2ECC71"
    C_REAL  = "#E74C3C"
    C_CORR  = "#2ECC71"
    C_FP    = "#F39C12"

    # ── 1. P / R / F1 por escenario (todos los códigos | sin fumador) ─────
    scenarios = ["Original", "Corr.\nsin agrupar", "Corr.\nagrupado"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metrics_set, title in zip(
        axes,
        [
            (orig_metrics, corr_metrics, corr_grouped_overall),
            (orig_metrics_nsmk, corr_metrics_nsmk, corr_grouped_overall_nsmk),
        ],
        ["Todos los códigos", "Sin fumador / exfumador"],
    ):
        x = np.arange(3)
        w = 0.25
        p_vals = [m["precision"] for m in metrics_set]
        r_vals = [m["recall"]    for m in metrics_set]
        f_vals = [m["f1"]        for m in metrics_set]

        for vals, offset, color, label in [
            (p_vals, -w, "#3498DB", "Precisión"),
            (r_vals,  0, "#E74C3C", "Recall"),
            (f_vals,  w, "#2ECC71", "F1"),
        ]:
            bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Precisión / Recall / F1 por escenario de evaluación", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(plots_dir / "1_prf1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. F1 por código: original vs corregido agrupado ─────────────────
    codes = sorted(corr_by_code.keys())
    x = np.arange(len(codes))
    w = 0.35
    orig_f1 = [eval_orig["icd10_metrics"].get(c, {}).get("f1", 0) for c in codes]
    corr_f1 = [corr_by_code[c]["f1"] for c in codes]

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w / 2, orig_f1, w, label="Original", color=C_ORIG, alpha=0.85)
    b2 = ax.bar(x + w / 2, corr_f1, w, label="Corregido (agrupado)", color=C_GRP, alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=30, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("F1-Score")
    ax.set_title("F1 por código ICD10: Original vs. Corregido (agrupado)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "2_f1_by_code.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2b. Recall por código: original vs corregido agrupado ──────────────
    orig_r = [eval_orig["icd10_metrics"].get(c, {}).get("recall", 0) for c in codes]
    corr_r = [corr_by_code[c]["recall"] for c in codes]

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w / 2, orig_r, w, label="Original", color=C_ORIG, alpha=0.85)
    b2 = ax.bar(x + w / 2, corr_r, w, label="Corregido (agrupado)", color=C_GRP, alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=30, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Recall")
    ax.set_title("Recall por código ICD10: Original vs. Corregido (agrupado)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "2b_recall_by_code.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Precisión: original vs corregida por código (horizontal) ───────
    codes_sp = sorted(eval_orig["icd10_metrics"].keys(),
                      key=lambda c: eval_orig["icd10_metrics"][c]["precision"])
    orig_p  = [eval_orig["icd10_metrics"][c]["precision"] for c in codes_sp]
    corr_p  = [corr_by_code.get(c, {}).get("precision", 0) for c in codes_sp]
    y = np.arange(len(codes_sp))
    h = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - h / 2, orig_p, h, label="Precisión original", color=C_ORIG, alpha=0.85)
    ax.barh(y + h / 2, corr_p, h, label="Precisión corregida", color=C_GRP, alpha=0.85)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Umbral 0.5")
    ax.set_yticks(y)
    ax.set_yticklabels(codes_sp)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Precisión")
    ax.set_title("Precisión por código: Original vs. Corregida", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "3_precision_by_code.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. FP por código: original vs corregido (stacked) ────────────────
    codes_all = sorted(eval_orig["icd10_metrics"].keys())
    orig_fp_v = [eval_orig["icd10_metrics"][c]["fp"] for c in codes_all]
    corr_fp_v = [corr_by_code.get(c, {}).get("fp", 0) for c in codes_all]
    conv_fp_v = [orig_fp_v[i] - corr_fp_v[i] for i in range(len(codes_all))]  # convertidos a TP

    x = np.arange(len(codes_all))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, orig_fp_v, w, label="FP original", color=C_ORIG, alpha=0.85)
    b_rem = ax.bar(x + w / 2, corr_fp_v, w, label="FP corregido (real)", color=C_FP, alpha=0.85)
    ax.bar(x + w / 2, conv_fp_v, w, bottom=corr_fp_v, label="→ convertidos a TP", color=C_GRP, alpha=0.6)

    for i, (orig, corr) in enumerate(zip(orig_fp_v, corr_fp_v)):
        ax.text(i - w / 2, orig + 0.2, str(orig), ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, orig_fp_v[i] + 0.2, str(corr), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(codes_all, rotation=30, ha="right")
    ax.set_ylabel("Nº de FP")
    ax.set_title("FP por código: Original vs. Corregido", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "4_fp_by_code.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. FP por modelo: real vs corregido a TP ─────────────────────────
    model_data = strategy_analysis["fp_by_model"]
    models = sorted(model_data.keys())
    real_fp_m = [model_data[m]["real_fp_errors"]       for m in models]
    corr_fp_m = [model_data[m]["fp_corrected_to_tp"]   for m in models]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, real_fp_m, w, label="FP reales (errores)", color=C_REAL, alpha=0.85)
    b2 = ax.bar(x + w / 2, corr_fp_m, w, label="FP corregidos → TP", color=C_CORR, alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=9,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("gemma3_", "g3_").replace("qwen25_", "q25_") for m in models],
        rotation=20, ha="right",
    )
    ax.set_ylabel("Nº de FP")
    ax.set_title("FP por modelo: reales vs. corregidos a TP", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "5_fp_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 6. Tipo de detección: solo_regex / regex+LLM / solo_LLM ─────────
    det = strategy_analysis["fp_by_detection_type"]
    det_keys   = ["solo_regex", "regex_y_llm", "solo_llm"]
    det_labels = ["Solo regex", "Regex + LLM", "Solo LLM"]
    real_det   = [det[k]["real_errors"]      for k in det_keys]
    corr_det   = [det[k]["corrected_to_tp"]  for k in det_keys]

    x = np.arange(len(det_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, real_det, w, label="FP reales", color=C_REAL, alpha=0.85)
    b2 = ax.bar(x + w / 2, corr_det, w, label="Corregidos → TP", color=C_CORR, alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=10,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(det_labels)
    ax.set_ylabel("Nº de FP")
    ax.set_title("FP por tipo de detección: reales vs. corregidos", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "6_fp_by_detection_type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Solapamiento (mejorado): solo_regex / regex + N modelos / solo LLM ─
    # Construir distribución mejorada a partir de los registros detallados
    fp_recs = strategy_analysis.get("fp_detail_records", [])
    improved_counts = defaultdict(int)
    for rec in fp_recs:
        strategies = rec.get("strategies", [])
        sset = set(strategies)
        if "regex" in sset:
            llms = sorted(list(sset - {"regex"}))
            n = len(llms)
            if n == 0:
                label = "Solo regex"
            elif n == 1:
                label = "Regex + 1 modelo"
            else:
                label = f"Regex + {n} modelos"
        else:
            # detección sin regex
            if len(sset) == 0:
                label = "Desconocido"
            else:
                label = "Solo LLM"
        improved_counts[label] += 1

    labels_order = ["Solo regex"] + [f"Regex + {i} modelos" for i in range(1, 6)] + ["Solo LLM"]
    labels = [l for l in labels_order if improved_counts.get(l, 0) > 0]
    counts = [improved_counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color="#8E44AD", alpha=0.85, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Tipo de detección (regex y/o LLMs)")
    ax.set_ylabel("Nº de FPs")
    ax.set_title("Distribución de FPs por tipo de consenso (mejorado)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "7_fp_overlap_distribution_improved.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 8. FN por código ─────────────────────────────────────────────────
    fn_data = strategy_analysis["fn_analysis"]["by_code"]
    fn_codes   = sorted(fn_data.keys())
    fn_real_v  = [fn_data[c]["real_errors"] for c in fn_codes]
    fn_accep_v = [fn_data[c]["acceptable"]  for c in fn_codes]

    x = np.arange(len(fn_codes))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, fn_real_v,  label="FN reales (error real)", color=C_REAL, alpha=0.85)
    ax.bar(x, fn_accep_v, bottom=fn_real_v, label="FN aceptables (no críticos)", color="#BDC3C7", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(fn_codes, rotation=30, ha="right")
    ax.set_ylabel("Nº de FN")
    ax.set_title("FN por código ICD10 (tras revisión manual)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "8_fn_by_code.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 9. Tasa de error real por modelo (% FPs que son errores reales) ───
    model_stats = strategy_analysis["fp_by_model"]
    models_s = sorted(model_stats.keys(), key=lambda m: model_stats[m]["fp_error_rate"], reverse=True)
    error_rates = [model_stats[m]["fp_error_rate"] * 100 for m in models_s]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [m.replace("gemma3_", "g3_").replace("qwen25_", "q25_") for m in models_s],
        error_rates,
        color=[C_REAL if r > 20 else C_FP if r > 15 else C_GRP for r in error_rates],
        alpha=0.85,
        edgecolor="white",
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax.axhline(20, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Umbral 20%")
    ax.set_ylabel("% FPs que son errores reales")
    ax.set_title("Tasa de error real por modelo/estrategia", fontweight="bold")
    ax.set_ylim(0, max(error_rates) * 1.2 + 5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(plots_dir / "9_error_rate_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Gráficos guardados en: {plots_dir}")
    return plots_dir


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_metrics(label: str, m: Dict):
    print(f"  {label}")
    print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"    Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")


def main():
    # Forzar UTF-8 en stdout para evitar errores de codificación en Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Cargando ficheros...")

    fp_corrections, fn_corrections = load_manual_analysis(MANUAL_ANALYSIS_FILE)
    print(f"  Correcciones FP cargadas: {len(fp_corrections)}")
    print(f"  Correcciones FN cargadas: {len(fn_corrections)}")

    fp_list     = load_fp_analysis(FP_ANALYSIS_FILE)
    fn_list     = load_fn_analysis(FN_ANALYSIS_FILE)
    eval_orig   = load_eval_results(EVAL_RESULTS_FILE)
    predictions = load_predictions(PREDICTIONS_FILE)
    print(f"  FPs en análisis: {len(fp_list)}")
    print(f"  FNs en análisis: {len(fn_list)}")
    print(f"  Predicciones: {len(predictions)} documentos")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Métricas ORIGINALES (de referencia)
    # ──────────────────────────────────────────────────────────────────────────
    orig_overall = eval_orig["overall"]
    orig_metrics = compute_metrics(
        orig_overall["tp"], orig_overall["fp"], orig_overall["fn"]
    )

    # Métricas originales sin fumador
    orig_tp_nsmk = orig_overall["tp"] - sum(
        v["tp"] for k, v in eval_orig["icd10_metrics"].items() if k in SMOKING_CODES)
    orig_fp_nsmk = orig_overall["fp"] - sum(
        v["fp"] for k, v in eval_orig["icd10_metrics"].items() if k in SMOKING_CODES)
    orig_fn_nsmk = orig_overall["fn"] - sum(
        v["fn"] for k, v in eval_orig["icd10_metrics"].items() if k in SMOKING_CODES)
    orig_metrics_nsmk = compute_metrics(orig_tp_nsmk, orig_fp_nsmk, orig_fn_nsmk)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Métricas CORREGIDAS sin agrupar (todos los códigos)
    # ──────────────────────────────────────────────────────────────────────────
    corr_tp, corr_fp, corr_fn = build_corrected_counts_ungrouped(
        fp_list, fn_list, fp_corrections, fn_corrections, eval_orig,
        exclude_smoking=False
    )
    corr_metrics = compute_metrics(corr_tp, corr_fp, corr_fn)

    # ──────────────────────────────────────────────────────────────────────────
    # 2b. Métricas CORREGIDAS sin agrupar SIN fumador
    # ──────────────────────────────────────────────────────────────────────────
    corr_tp_nsmk, corr_fp_nsmk, corr_fn_nsmk = build_corrected_counts_ungrouped(
        fp_list, fn_list, fp_corrections, fn_corrections, eval_orig,
        exclude_smoking=True
    )
    corr_metrics_nsmk = compute_metrics(corr_tp_nsmk, corr_fp_nsmk, corr_fn_nsmk)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Métricas CORREGIDAS agrupadas por código
    # ──────────────────────────────────────────────────────────────────────────
    corr_by_code = build_corrected_counts_by_code(
        fp_list, fn_list, fp_corrections, fn_corrections, eval_orig,
        exclude_smoking=False
    )
    # Agregar para obtener global
    g_tp = sum(v["tp"] for v in corr_by_code.values())
    g_fp = sum(v["fp"] for v in corr_by_code.values())
    g_fn = sum(v["fn"] for v in corr_by_code.values())
    corr_grouped_overall = compute_metrics(g_tp, g_fp, g_fn)

    # ──────────────────────────────────────────────────────────────────────────
    # 3b. Métricas CORREGIDAS agrupadas por código SIN fumador
    # ──────────────────────────────────────────────────────────────────────────
    corr_by_code_nsmk = build_corrected_counts_by_code(
        fp_list, fn_list, fp_corrections, fn_corrections, eval_orig,
        exclude_smoking=True
    )
    g_tp_nsmk = sum(v["tp"] for v in corr_by_code_nsmk.values())
    g_fp_nsmk = sum(v["fp"] for v in corr_by_code_nsmk.values())
    g_fn_nsmk = sum(v["fn"] for v in corr_by_code_nsmk.values())
    corr_grouped_overall_nsmk = compute_metrics(g_tp_nsmk, g_fp_nsmk, g_fn_nsmk)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Análisis de estrategias/modelos
    # ──────────────────────────────────────────────────────────────────────────
    strategy_analysis = analyze_strategy_errors(
        fp_list, fn_list, fp_corrections, fn_corrections, predictions,
        exclude_smoking=False
    )
    strategy_analysis_nsmk = analyze_strategy_errors(
        fp_list, fn_list, fp_corrections, fn_corrections, predictions,
        exclude_smoking=True
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CONSOLA: Resumen
    # ──────────────────────────────────────────────────────────────────────────
    print_section("MÉTRICAS ORIGINALES")
    print_metrics("Todos los códigos", orig_metrics)
    print_metrics("Sin fumador/exfumador", orig_metrics_nsmk)

    print_section("MÉTRICAS CORREGIDAS (sin agrupar por código)")
    print_metrics("Todos los códigos", corr_metrics)
    print_metrics("Sin fumador/exfumador", corr_metrics_nsmk)

    print_section("MÉTRICAS CORREGIDAS AGRUPADAS POR CÓDIGO - Global")
    print_metrics("Todos los códigos", corr_grouped_overall)
    print_metrics("Sin fumador/exfumador", corr_grouped_overall_nsmk)

    print_section("DETALLE POR CÓDIGO - Corregido agrupado (todos)")
    for code, m in sorted(corr_by_code.items()):
        orig = eval_orig["icd10_metrics"].get(code, {})
        print(f"  {code}")
        print(f"    Original:  TP={orig.get('tp',0)}  FP={orig.get('fp',0)}  FN={orig.get('fn',0)}"
              f"  P={orig.get('precision',0):.3f}  R={orig.get('recall',0):.3f}  F1={orig.get('f1',0):.3f}")
        print(f"    Corregido: TP={m['tp']}  FP={m['fp']}  FN={m['fn']}"
              f"  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
        print(f"    Δ: FP_conv→TP={m['fp_converted_to_tp']}  FN_reales={m['fn_real_errors']}  FN_aceptables={m['fn_acceptable']}")

    print_section("ANÁLISIS DE ESTRATEGIAS / MODELOS (todos los códigos)")
    sa = strategy_analysis
    print(f"\n  FP por tipo de detección:")
    for dtype, info in sa["fp_by_detection_type"].items():
        print(f"    {dtype}: total={info['total_fp']}  errores_reales={info['real_errors']}"
              f"  corregidos→TP={info['corrected_to_tp']}")
        print(f"      Por código: {info['by_code']}")

    print(f"\n  FP por modelo (total FPs en los que participó cada modelo):")
    for model, stats in sorted(sa["fp_by_model"].items(),
                                key=lambda x: x[1]["total_fp"], reverse=True):
        print(f"    {model}: total_FP={stats['total_fp']}"
              f"  errores_reales={stats['real_fp_errors']}"
              f"  corregidos→TP={stats['fp_corrected_to_tp']}"
              f"  tasa_error={stats['fp_error_rate']:.2%}")
        print(f"      Por código: {stats['by_code']}")

    print(f"\n  Ranking modelos por FP reales generados:")
    for entry in sa["model_ranking_by_real_fp"]:
        print(f"    {entry['model']}: {entry['real_fp_errors']} FP reales")

    print(f"\n  Distribución de FPs por nº de estrategias que lo detectaron:")
    for k, v in sorted(sa["fp_overlap_distribution"].items()):
        print(f"    {k}: {v} FPs")

    print(f"\n  Top combinaciones de estrategias que generan FP:")
    for combo in sa["top_fp_combinations"][:10]:
        print(f"    {' + '.join(combo['strategies'])}: "
              f"{combo['total_fp']} FP  ({combo['real_errors']} reales, {combo['corrected']} corregidos)")

    print(f"\n  Análisis de FN:")
    fna = sa["fn_analysis"]
    print(f"    Total FN: {fna['total_fn']}")
    print(f"    FN errores reales: {fna['real_errors']}")
    print(f"    FN aceptables (no son error grave): {fna['acceptable_fn']}")
    print(f"    Por código:")
    for code, info in fna["by_code"].items():
        print(f"      {code}: real={info['real_errors']}  aceptable={info['acceptable']}  "
              f"total={info['total']}  ejemplos={info.get('entity_examples', [])[:5]}")

    # ──────────────────────────────────────────────────────────────────────────
    # GUARDAR RESULTADOS
    # ──────────────────────────────────────────────────────────────────────────
    output = {
        "original_metrics": {
            "all_codes": orig_metrics,
            "excluding_smoking": orig_metrics_nsmk,
            "by_code": eval_orig["icd10_metrics"]
        },
        "corrected_metrics_ungrouped": {
            "all_codes": corr_metrics,
            "excluding_smoking": corr_metrics_nsmk,
        },
        "corrected_metrics_grouped_by_code": {
            "all_codes": {
                "overall": corr_grouped_overall,
                "by_code": corr_by_code
            },
            "excluding_smoking": {
                "overall": corr_grouped_overall_nsmk,
                "by_code": corr_by_code_nsmk
            }
        },
        "strategy_error_analysis": {
            "all_codes": {
                "fp_by_detection_type":    sa["fp_by_detection_type"],
                "fp_by_model":             sa["fp_by_model"],
                "model_ranking_by_real_fp":sa["model_ranking_by_real_fp"],
                "fp_overlap_distribution": sa["fp_overlap_distribution"],
                "top_fp_combinations":     sa["top_fp_combinations"],
                "fn_analysis":             sa["fn_analysis"],
            },
            "excluding_smoking": {
                "fp_by_detection_type":    strategy_analysis_nsmk["fp_by_detection_type"],
                "fp_by_model":             strategy_analysis_nsmk["fp_by_model"],
                "model_ranking_by_real_fp":strategy_analysis_nsmk["model_ranking_by_real_fp"],
                "fp_overlap_distribution": strategy_analysis_nsmk["fp_overlap_distribution"],
                "top_fp_combinations":     strategy_analysis_nsmk["top_fp_combinations"],
                "fn_analysis":             strategy_analysis_nsmk["fn_analysis"],
            }
        },
        "summary_comparison": {
            "scenario": ["original", "corr_ungrouped", "corr_grouped"],
            "all_codes": {
                "precision": [orig_metrics["precision"], corr_metrics["precision"], corr_grouped_overall["precision"]],
                "recall":    [orig_metrics["recall"],    corr_metrics["recall"],    corr_grouped_overall["recall"]],
                "f1":        [orig_metrics["f1"],        corr_metrics["f1"],        corr_grouped_overall["f1"]],
                "tp":        [orig_metrics["tp"],        corr_metrics["tp"],        corr_grouped_overall["tp"]],
                "fp":        [orig_metrics["fp"],        corr_metrics["fp"],        corr_grouped_overall["fp"]],
                "fn":        [orig_metrics["fn"],        corr_metrics["fn"],        corr_grouped_overall["fn"]],
            },
            "excluding_smoking": {
                "precision": [orig_metrics_nsmk["precision"], corr_metrics_nsmk["precision"], corr_grouped_overall_nsmk["precision"]],
                "recall":    [orig_metrics_nsmk["recall"],    corr_metrics_nsmk["recall"],    corr_grouped_overall_nsmk["recall"]],
                "f1":        [orig_metrics_nsmk["f1"],        corr_metrics_nsmk["f1"],        corr_grouped_overall_nsmk["f1"]],
                "tp":        [orig_metrics_nsmk["tp"],        corr_metrics_nsmk["tp"],        corr_grouped_overall_nsmk["tp"]],
                "fp":        [orig_metrics_nsmk["fp"],        corr_metrics_nsmk["fp"],        corr_grouped_overall_nsmk["fp"]],
                "fn":        [orig_metrics_nsmk["fn"],        corr_metrics_nsmk["fn"],        corr_grouped_overall_nsmk["fn"]],
            }
        },
        "meta": {
            "total_fp_in_analysis": len(fp_list),
            "total_fn_in_analysis": len(fn_list),
            "fp_manual_corrections_loaded": len(fp_corrections),
            "fn_manual_corrections_loaded": len(fn_corrections),
            "documents_analyzed": eval_orig.get("summary", {}).get("total_documents", 0)
        }
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print()
    print(f"\n✓ Resultados guardados en: {OUTPUT_FILE}")

    # ──────────────────────────────────────────────────────────────────────────
    # IMPRESIÓN FINAL COMPACTA DE COMPARATIVA
    # ──────────────────────────────────────────────────────────────────────────
    print_section("RESUMEN COMPARATIVO FINAL")
    rows = [
        ("Original (todos)",           orig_metrics),
        ("Corregido sin agrupar",       corr_metrics),
        ("Corregido agrupado x código", corr_grouped_overall),
    ]
    print(f"  {'Escenario':<35} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 70)
    for label, m in rows:
        print(f"  {label:<35} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f}"
              f" {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    print()
    print("  --- Excluyendo fumador/exfumador ---")
    rows_nsmk = [
        ("Original sin fumador",               orig_metrics_nsmk),
        ("Corregido sin agrupar, sin fumador",  corr_metrics_nsmk),
        ("Corregido agrupado, sin fumador",     corr_grouped_overall_nsmk),
    ]
    print(f"  {'Escenario':<40} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 75)
    for label, m in rows_nsmk:
        print(f"  {label:<40} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f}"
              f" {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    # ──────────────────────────────────────────────────────────────────────────
    # GRÁFICOS
    # ──────────────────────────────────────────────────────────────────────────
    print_section("GENERANDO GRÁFICOS")
    try:
        generate_plots(
            orig_metrics, orig_metrics_nsmk,
            corr_metrics, corr_metrics_nsmk,
            corr_grouped_overall, corr_grouped_overall_nsmk,
            corr_by_code, eval_orig, strategy_analysis,
            FINAL_METRICS_DIR,
        )
    except Exception as e:
        print(f"  [WARNING] No se pudieron generar los gráficos: {e}")


if __name__ == "__main__":
    main()
