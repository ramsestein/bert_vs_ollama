#!/usr/bin/env python3
"""
NER Results Viewer

Generates a self-contained HTML file to compare NER predictions against ground truth.

Usage:
    python scripts/ner_viewer.py \
        --predictions metrics/ncbi/ncbi_test_predictions.jsonl \
        --benchmark datasets/ncbi_test.jsonl \
        --output ner_viewer.html

Colors:
    Green  = True Positive  (predicted and in ground truth)
    Red    = False Positive (predicted but not in ground truth)
    Orange = False Negative (in ground truth but not predicted)
"""

import json
import argparse
import re
from pathlib import Path


def load_jsonl(path):
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                docs[doc["PMID"]] = doc
    return docs


def normalize(text):
    """Normalize entity text for matching (strip trailing punctuation, lowercase)."""
    return re.sub(r"[,\.;]+$", "", text.strip()).lower()


def find_spans(text, entity_text):
    """Find all start/end positions of entity_text in text (case-insensitive)."""
    spans = []
    pattern = re.escape(entity_text.rstrip(","))
    for m in re.finditer(pattern, text, re.IGNORECASE):
        spans.append((m.start(), m.end()))
    return spans


def highlight_text(text, annotations):
    """
    Build an HTML string with highlighted spans.
    annotations: list of (start, end, css_class, label)
    Overlapping spans: longest wins.
    """
    # Sort by start, then by length descending (longest first)
    annotations = sorted(annotations, key=lambda a: (a[0], -(a[1] - a[0])))

    # Build non-overlapping set (greedy, longest wins per position)
    used = [False] * len(text)
    active = []
    for start, end, css_class, label in annotations:
        if any(used[i] for i in range(start, end)):
            continue
        active.append((start, end, css_class, label))
        for i in range(start, end):
            used[i] = True

    active.sort(key=lambda a: a[0])

    html_parts = []
    cursor = 0
    for start, end, css_class, label in active:
        if cursor < start:
            html_parts.append(_escape(text[cursor:start]))
        span_text = _escape(text[start:end])
        html_parts.append(
            f'<mark class="ent {css_class}" title="{_escape(label)}">{span_text}</mark>'
        )
        cursor = end
    if cursor < len(text):
        html_parts.append(_escape(text[cursor:]))

    return "".join(html_parts)


def _escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_doc_data(pred_doc, gt_doc):
    text = pred_doc["Texto"]
    pred_entities = pred_doc.get("Entidad", [])
    gt_entities = gt_doc.get("Entidad", []) if gt_doc else []

    gt_specific = [e for e in gt_entities if e.get("tipo") == "SpecificDisease"]
    gt_norms = {normalize(e["texto"]) for e in gt_specific}
    pred_norms = {normalize(e["texto"]) for e in pred_entities}

    # Classify predictions
    classified_preds = []
    for ent in pred_entities:
        norm = normalize(ent["texto"])
        status = "tp" if norm in gt_norms else "fp"
        classified_preds.append({**ent, "status": status})

    # False negatives
    fn_entities = [e for e in gt_specific if normalize(e["texto"]) not in pred_norms]

    # Build annotations for text highlighting
    annotations = []

    # Predicted entities (use stored spans if available, else search text)
    for ent in classified_preds:
        css = "tp" if ent["status"] == "tp" else "fp"
        strategies_str = ", ".join(ent.get("strategies", []))
        label = f"{ent['texto']} [{strategies_str}] conf={ent.get('confidence', '?'):.2f}"
        spans = ent.get("spans") or []
        if spans:
            # Use the first valid span
            for sp in spans:
                s, e = sp["start"], sp["end"]
                if 0 <= s < e <= len(text):
                    annotations.append((s, e, css, label))
                    break
        else:
            for s, e in find_spans(text, ent["texto"])[:1]:
                annotations.append((s, e, css, label))

    # FN entities (find in text)
    for ent in fn_entities:
        label = f"MISSED: {ent['texto']}"
        for s, e in find_spans(text, ent["texto"])[:1]:
            annotations.append((s, e, "fn", label))

    highlighted = highlight_text(text, annotations)

    return {
        "pmid": pred_doc["PMID"],
        "highlighted_text": highlighted,
        "predictions": classified_preds,
        "ground_truth": gt_specific,
        "false_negatives": fn_entities,
        "tp": sum(1 for e in classified_preds if e["status"] == "tp"),
        "fp": sum(1 for e in classified_preds if e["status"] == "fp"),
        "fn": len(fn_entities),
    }


def generate_html(docs_data):
    docs_json = json.dumps(docs_data, ensure_ascii=False)

    strategy_colors = {
        "regex": "#6c757d",
        "gemma3_max_sensitivity": "#0d6efd",
        "gemma3_balanced": "#6610f2",
        "gemma3_high_precision": "#0dcaf0",
        "qwen25_diversity": "#fd7e14",
    }
    strategy_colors_json = json.dumps(strategy_colors)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NER Results Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f8f9fa; color: #212529; }}
  header {{
    background: #212529; color: white; padding: 12px 24px;
    display: flex; align-items: center; gap: 16px;
  }}
  header h1 {{ font-size: 1.1rem; font-weight: 600; }}
  .nav {{ display: flex; align-items: center; gap: 8px; margin-left: auto; }}
  .nav button {{
    background: #495057; border: none; color: white; padding: 6px 14px;
    border-radius: 4px; cursor: pointer; font-size: 0.85rem;
  }}
  .nav button:hover {{ background: #6c757d; }}
  .nav button:disabled {{ opacity: 0.4; cursor: default; }}
  #doc-counter {{ font-size: 0.85rem; color: #adb5bd; }}
  .container {{ display: grid; grid-template-columns: 1fr 340px; gap: 16px; padding: 16px; max-width: 1600px; margin: 0 auto; }}
  .card {{
    background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
    padding: 16px;
  }}
  .card h2 {{ font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #6c757d; margin-bottom: 10px; letter-spacing: .05em; }}
  #text-panel {{ line-height: 1.85; font-size: 0.95rem; }}
  mark.ent {{
    border-radius: 3px; padding: 1px 3px; cursor: default;
    font-weight: 500; position: relative;
  }}
  mark.tp {{ background: #d1e7dd; color: #0a3622; border-bottom: 2px solid #198754; }}
  mark.fp {{ background: #f8d7da; color: #58151c; border-bottom: 2px solid #dc3545; }}
  mark.fn {{ background: #fff3cd; color: #664d03; border-bottom: 2px solid #ffc107; }}
  .sidebar {{ display: flex; flex-direction: column; gap: 16px; }}
  .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 4px; }}
  .stat-box {{
    text-align: center; padding: 10px 6px; border-radius: 6px;
  }}
  .stat-box.tp-box {{ background: #d1e7dd; color: #0a3622; }}
  .stat-box.fp-box {{ background: #f8d7da; color: #58151c; }}
  .stat-box.fn-box {{ background: #fff3cd; color: #664d03; }}
  .stat-box .num {{ font-size: 1.6rem; font-weight: 700; }}
  .stat-box .lbl {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; }}
  .pmid-badge {{
    display: inline-block; background: #e9ecef; border-radius: 4px;
    padding: 2px 8px; font-size: 0.78rem; font-weight: 600; color: #495057; margin-bottom: 10px;
  }}
  .entity-list {{ list-style: none; display: flex; flex-direction: column; gap: 6px; }}
  .entity-item {{
    padding: 7px 10px; border-radius: 5px; font-size: 0.85rem;
  }}
  .entity-item.tp {{ background: #d1e7dd; border-left: 3px solid #198754; }}
  .entity-item.fp {{ background: #f8d7da; border-left: 3px solid #dc3545; }}
  .entity-item.fn {{ background: #fff3cd; border-left: 3px solid #ffc107; }}
  .entity-item .name {{ font-weight: 600; }}
  .entity-item .meta {{ margin-top: 3px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .badge {{
    display: inline-block; padding: 1px 6px; border-radius: 10px;
    font-size: 0.7rem; font-weight: 600; color: white;
  }}
  .badge-conf {{ background: #6c757d; }}
  .legend {{ display: flex; gap: 12px; flex-wrap: wrap; font-size: 0.78rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 2px; }}
  .pmid-select {{ width: 100%; padding: 5px 8px; border-radius: 4px; border: 1px solid #ced4da; font-size: 0.85rem; }}
</style>
</head>
<body>
<header>
  <h1>NER Results Viewer</h1>
  <div style="display:flex;gap:8px;align-items:center;margin-left:auto;">
    <select class="pmid-select" id="pmid-select" style="width:160px;"></select>
    <div class="nav">
      <button id="btn-prev" onclick="navigate(-1)">&#8592; Prev</button>
      <span id="doc-counter"></span>
      <button id="btn-next" onclick="navigate(1)">Next &#8594;</button>
    </div>
  </div>
</header>

<div class="container">
  <div class="card">
    <h2>Document Text</h2>
    <div class="legend" style="margin-bottom:12px;">
      <div class="legend-item"><div class="legend-dot" style="background:#d1e7dd;border-bottom:2px solid #198754;"></div> True Positive</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f8d7da;border-bottom:2px solid #dc3545;"></div> False Positive</div>
      <div class="legend-item"><div class="legend-dot" style="background:#fff3cd;border-bottom:2px solid #ffc107;"></div> False Negative (missed)</div>
    </div>
    <div id="text-panel"></div>
  </div>

  <div class="sidebar">
    <div class="card">
      <h2>Document</h2>
      <span class="pmid-badge" id="pmid-display"></span>
      <div class="stats">
        <div class="stat-box tp-box"><div class="num" id="stat-tp">-</div><div class="lbl">TP</div></div>
        <div class="stat-box fp-box"><div class="num" id="stat-fp">-</div><div class="lbl">FP</div></div>
        <div class="stat-box fn-box"><div class="num" id="stat-fn">-</div><div class="lbl">FN</div></div>
      </div>
    </div>

    <div class="card">
      <h2>Predicted Entities</h2>
      <ul class="entity-list" id="pred-list"></ul>
    </div>

    <div class="card">
      <h2>Ground Truth (missed)</h2>
      <ul class="entity-list" id="fn-list"></ul>
    </div>
  </div>
</div>

<script>
const DOCS = {docs_json};
const STRATEGY_COLORS = {strategy_colors_json};
let current = 0;

const pmidSelect = document.getElementById('pmid-select');
DOCS.forEach((d, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = 'PMID ' + d.pmid;
  pmidSelect.appendChild(opt);
}});
pmidSelect.addEventListener('change', () => {{ current = parseInt(pmidSelect.value); render(); }});

function navigate(dir) {{
  current = Math.max(0, Math.min(DOCS.length - 1, current + dir));
  pmidSelect.value = current;
  render();
}}

function strategyBadge(name) {{
  const color = STRATEGY_COLORS[name] || '#6c757d';
  return `<span class="badge" style="background:${{color}}">${{name}}</span>`;
}}

function render() {{
  const doc = DOCS[current];
  document.getElementById('doc-counter').textContent = `${{current + 1}} / ${{DOCS.length}}`;
  document.getElementById('btn-prev').disabled = current === 0;
  document.getElementById('btn-next').disabled = current === DOCS.length - 1;
  document.getElementById('pmid-display').textContent = 'PMID: ' + doc.pmid;
  document.getElementById('stat-tp').textContent = doc.tp;
  document.getElementById('stat-fp').textContent = doc.fp;
  document.getElementById('stat-fn').textContent = doc.fn;

  document.getElementById('text-panel').innerHTML = doc.highlighted_text;

  // Predicted entities
  const predList = document.getElementById('pred-list');
  predList.innerHTML = '';
  if (doc.predictions.length === 0) {{
    predList.innerHTML = '<li style="color:#6c757d;font-size:.85rem;">None</li>';
  }}
  doc.predictions.forEach(ent => {{
    const li = document.createElement('li');
    li.className = 'entity-item ' + ent.status;
    const conf = ent.confidence !== undefined ? ent.confidence.toFixed(2) : '?';
    const badges = (ent.strategies || []).map(strategyBadge).join(' ');
    li.innerHTML = `<div class="name">${{ent.texto}}</div>
      <div class="meta">
        <span class="badge badge-conf">conf ${{conf}}</span>
        ${{badges}}
      </div>`;
    predList.appendChild(li);
  }});

  // False negatives
  const fnList = document.getElementById('fn-list');
  fnList.innerHTML = '';
  if (doc.false_negatives.length === 0) {{
    fnList.innerHTML = '<li style="color:#6c757d;font-size:.85rem;">None — all GT entities found!</li>';
  }}
  doc.false_negatives.forEach(ent => {{
    const li = document.createElement('li');
    li.className = 'entity-item fn';
    li.innerHTML = `<div class="name">${{ent.texto}}</div>`;
    fnList.appendChild(li);
  }});
}}

render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate NER Results Viewer HTML")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL file")
    parser.add_argument("--benchmark", required=True, help="Ground truth JSONL file")
    parser.add_argument("--output", default="ner_viewer.html", help="Output HTML file")
    args = parser.parse_args()

    print(f"Loading predictions from {args.predictions}")
    preds = load_jsonl(args.predictions)
    print(f"Loading ground truth from {args.benchmark}")
    truth = load_jsonl(args.benchmark)

    docs_data = []
    for pmid in sorted(preds.keys()):
        pred_doc = preds[pmid]
        gt_doc = truth.get(pmid)
        docs_data.append(build_doc_data(pred_doc, gt_doc))

    print(f"Processing {len(docs_data)} documents...")
    html = generate_html(docs_data)

    output = Path(args.output)
    output.write_text(html, encoding="utf-8")
    print(f"Viewer saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
