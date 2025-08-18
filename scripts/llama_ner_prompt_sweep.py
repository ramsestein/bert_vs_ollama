# infer_llama_eval_progress_optimized.py
# Few-shot NER con Llama 3.2:3B en Ollama
# - Más marcadores de progreso
# - Warm-up y keep_alive
# - Contexto y num_predict ajustados
# - ThreadPoolExecutor (5 workers)
# - Métricas strict por mención (case-insensitive)

import argparse
import json
import re
import http.client
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, deque
from typing import List, Dict, Any, Tuple

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_WORKERS = 5
TIMEOUT_S = 120

# Ajustes de decodificación para extracción factual y salida corta
GEN_OPTIONS = {
    "temperature": DEFAULT_TEMPERATURE,
    "top_p": 0.9,
    "num_predict": 384,   # salida JSON breve; reducir acelera
    "num_ctx": 1024, 
    "cache_prompt": True,
    "num_gpu": 1,            # << fuerza uso de GPU si hay
    "num_thread": 4 
}

# -------------------
# Limpieza / Normalización
# -------------------
_punct_edge = re.compile(r"(^[\W_]+|[\W_]+$)", flags=re.UNICODE)
_multi_space = re.compile(r"\s+", flags=re.UNICODE)

def clean_mention(s: str) -> str:
    if not s:
        return ""
    x = s.replace("##", "")
    x = x.replace(" – ", "-").replace(" — ", "-").replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    x = _multi_space.sub(" ", x)
    x = _punct_edge.sub("", x)
    return x.strip()

def norm_for_match(s: str) -> str:
    return clean_mention(s).lower()

def dedup_list(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def suppress_substrings(mentions: List[str]) -> List[str]:
    ms = sorted(mentions, key=len, reverse=True)
    keep = []
    for m in ms:
        if any(m != k and m in k for k in keep):
            continue
        keep.append(m)
    return keep

def postprocess_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    mencs = [clean_mention(e.get("texto", "")) for e in entities if e.get("texto")]
    mencs = [m for m in mencs if m]
    mencs = suppress_substrings(dedup_list(mencs))
    return [{"texto": m, "tipo": "SpecificDisease"} for m in mencs]

# -------------------
# Prompts (FEW-SHOT)
# -------------------
JSON_SCHEMA_HINT = """
Devuelve únicamente un JSON válido con esta forma exacta:
{
  "PMID": "<string>",
  "Texto": "<string>",
  "Entidad": [
    { "texto": "<string>", "tipo": "SpecificDisease" }
  ]
}
- "Entidad" es un array (posiblemente vacío).
- No incluyas offsets ni otros campos.
- No repitas subfragmentos si existe uno más completo (usa "breast cancer" en vez de "cancer").
- Evita adjetivos sueltos o términos genéricos (e.g., "chronic", "enzyme", "deficiency").
- Si no hay enfermedades, "Entidad": [].
Responde SOLO con el JSON, sin comentarios ni texto adicional.
""".strip()

SYSTEM_FEW = (
    "Eres un extractor clínico de alta precisión. Extrae SOLO menciones de enfermedad humanas (NCBI-disease). "
    "Devuelve JSON estricto según el esquema indicado; nada más."
)

FEW_SHOT_EXAMPLES = [
    {
        "PMID": "E1",
        "Texto": "Patients with Alzheimer disease often present with memory loss.",
        "Entidad": [{"texto": "Alzheimer disease", "tipo": "SpecificDisease"}]
    },
    {
        "PMID": "E2",
        "Texto": "The mutation is common but does not cause any disease by itself.",
        "Entidad": []
    },
    {
        "PMID": "E3",
        "Texto": "Glucose-6-phosphate dehydrogenase (G6PD) deficiency leads to hemolytic anemia.",
        "Entidad": [
            {"texto": "G6PD deficiency", "tipo":"SpecificDisease"},
            {"texto": "hemolytic anemia", "tipo":"SpecificDisease"}
        ]
    },
]

def build_few_shot_prompt(pmid: str, text: str) -> str:
    demos = "\n\n".join(
        [
            f"PMID: {ex['PMID']}\nTEXTO:\n{ex['Texto']}\nSALIDA:\n{json.dumps(ex, ensure_ascii=False)}"
            for ex in FEW_SHOT_EXAMPLES
        ]
    )
    return f"""
{JSON_SCHEMA_HINT}

EJEMPLOS:
{demos}

PMID: {pmid}
TEXTO:
{text}

SALIDA:
""".strip()

# -------------------
# Cliente Ollama con conexión persistente por worker
# -------------------
class OllamaClient:
    def __init__(self, host="localhost", port=11434, timeout=TIMEOUT_S):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        self.lock = threading.Lock()  # por si hay reintentos secuenciales

    def _request(self, path: str, body: dict) -> str:
        payload = json.dumps(body)
        with self.lock:
            self.conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
            resp = self.conn.getresponse()
            raw = resp.read().decode("utf-8", errors="ignore")
            # reabrir si el server cerró
            if resp.closed:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            return raw

    def pull(self, model: str):
        body = {"name": model, "stream": False}
        return self._request("/api/pull", body)

    def generate(self, model: str, system_prompt: str, user_prompt: str, options: dict) -> str:
        body = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt, # << usa campo system separado
            "stream": False,
            "keep_alive": "15m",      # evita descargas entre peticiones
            "options": options,
            # corta en patrones comunes para terminar rápido
            "stop": ["\nUser:", "\nUSER:", "\nAssistant:", "\nASSISTANT:"]
        }
        raw = self._request("/api/generate", body)
        # ✅ DEVOLVER SOLO EL TEXTO GENERADO
        try:
            data = json.loads(raw)
            return data.get("response", raw)
        except Exception:
            return raw

def parse_json_with_retries(text: str, pmid: str, original_text: str, max_retries: int = 2) -> Dict[str, Any]:
    # Si por error llega el JSON envoltorio de Ollama, extrae su 'response'
    try:
        maybe = json.loads(text)
        if isinstance(maybe, dict) and "response" in maybe and "model" in maybe:
            text = maybe["response"]
    except Exception:
        pass

    obj = None
    t = text
    for _ in range(max_retries + 1):
        try:
            obj = json.loads(t)
            break
        except Exception:
            m = re.search(r"\{.*\}", t, flags=re.S)
            if m:
                t = m.group(0)  # reintenta con el primer bloque {...}
                continue
            # no hay bloque JSON reconocible -> rompe el bucle
            break

    if not isinstance(obj, dict):
        obj = {"PMID": pmid, "Texto": original_text, "Entidad": []}

    # Normaliza y garantiza campos
    obj.setdefault("PMID", pmid)
    obj.setdefault("Texto", original_text)
    obj.setdefault("Entidad", [])
    obj["Entidad"] = postprocess_entities(obj.get("Entidad", []))
    return obj


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------------
# Worker
# -------------------
def worker_init() -> OllamaClient:
    return OllamaClient()

def worker_predict(client: OllamaClient, job: Tuple[str, str, str, float]) -> Dict[str, Any]:
    pmid, text, model, temperature = job
    opts = GEN_OPTIONS.copy()
    opts["temperature"] = float(temperature)
    user_prompt = build_few_shot_prompt(pmid, text)

    t0 = time.time()
    try:
        resp = client.generate(model, SYSTEM_FEW, user_prompt, options=opts)
        obj = parse_json_with_retries(resp, pmid, text)
        dt = time.time() - t0
        obj = {"PMID": pmid, "Texto": text, "Entidad": postprocess_entities(obj.get("Entidad", []))}
        obj["_latency_sec"] = round(dt, 3)
        return obj
    except Exception as e:
        dt = time.time() - t0
        # Fallback seguro si algo falla
        return {"PMID": pmid, "Texto": text, "Entidad": [], "_latency_sec": round(dt, 3), "_error": str(e)}

# -------------------
# Evaluación strict por mención
# -------------------
def evaluate_strict(gold_records: List[Dict[str, Any]], pred_records: List[Dict[str, Any]]):
    gold_by_id = {str(r.get("PMID")): r for r in gold_records}
    pred_by_id = {str(r.get("PMID")): r for r in pred_records}
    total_tp = total_pred = total_gold = 0
    for pmid, grec in gold_by_id.items():
        prec = pred_by_id.get(pmid, {"Entidad": []})
        gold_mentions = [norm_for_match(e.get("texto","")) for e in (grec.get("Entidad") or []) if e.get("texto")]
        pred_mentions = [norm_for_match(e.get("texto","")) for e in (prec.get("Entidad") or []) if e.get("texto")]
        c_gold = Counter(gold_mentions)
        c_pred = Counter(pred_mentions)
        c_tp   = c_gold & c_pred
        total_tp   += sum(c_tp.values())
        total_gold += sum(c_gold.values())
        total_pred += sum(c_pred.values())
    precision = total_tp / total_pred if total_pred else 0.0
    recall    = total_tp / total_gold if total_gold else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, total_pred, total_gold, total_tp

# -------------------
# Heartbeat de progreso
# -------------------
def start_heartbeat(total_jobs: int, qps_window=50):
    state = {
        "done": 0,
        "errors": 0,
        "start": time.time(),
        "lat_hist": deque(maxlen=qps_window),
        "stop": False
    }
    def beat():
        while not state["stop"]:
            time.sleep(60)
            done = state["done"]; errs = state["errors"]
            elapsed = time.time() - state["start"]
            avg_lat = sum(state["lat_hist"])/len(state["lat_hist"]) if state["lat_hist"] else 0.0
            qps = done/elapsed if elapsed > 0 else 0.0
            eta = (elapsed/done*(total_jobs-done)) if done>0 else 0.0
            print(f"[HEARTBEAT] {done}/{total_jobs} ({100*done/total_jobs:.1f}%) | "
                  f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s | QPS={qps:.2f} | avg_latency={avg_lat:.2f}s | errors={errs}")
    th = threading.Thread(target=beat, daemon=True)
    th.start()
    return state

# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser(description="Few-shot NER con Llama (Ollama) + progreso y optimización básica")
    ap.add_argument("--develop_jsonl", required=True)
    ap.add_argument("--out_pred", default="results_llama_few.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--n_workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    print(f"[INFO] Arrancando. modelo={args.model} | temp={args.temperature} | workers={args.n_workers}")
    gold = list(load_jsonl(args.develop_jsonl))
    jobs = [(str(r.get("PMID")), r.get("Texto",""), args.model, args.temperature) for r in gold]
    total = len(jobs)
    print(f"[INFO] Cargados {total} textos de develop")

    # Warm-up: pull + inferencia dummy para compilar y cargar a VRAM
    warm_client = OllamaClient()
    print("[WARMUP] Pull del modelo (si ya está, es instantáneo)...")
    try:
        warm_client.pull(args.model)
    except Exception as e:
        print(f"[WARN] Pull falló (posible si ya está): {e}")
    print("[WARMUP] Inferencia de calentamiento...")
    try:
        dummy = build_few_shot_prompt("WARMUP", "This is a warm-up prompt. No diseases here.")
        warm_client.generate(args.model, SYSTEM_FEW, dummy, options=GEN_OPTIONS)
        print("[WARMUP] OK")
    except Exception as e:
        print(f"[WARN] Warm-up falló: {e}")

    preds = []
    heartbeat = start_heartbeat(total_jobs=total)

    t0 = time.time()
    # Un cliente por worker (conexión persistente)
    def _task(job):
        client = OllamaClient()
        res = worker_predict(client, job)
        client.conn.close()
        return res

    print("[INFO] Comenzando inferencia paralela...")
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(_task, job): job[0] for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            pmid = futures[fut]
            try:
                obj = fut.result()
                preds.append(obj)
                heartbeat["lat_hist"].append(obj.get("_latency_sec", 0.0))
            except Exception as e:
                print(f"[WARN] Error en PMID={pmid}: {e}")
                preds.append({"PMID": pmid, "Texto": "", "Entidad": []})
                heartbeat["errors"] += 1
            heartbeat["done"] = i
            if i % 10 == 0:
                print(f"[PROGRESS] Completados {i}/{total} ({100*i/total:.1f}%)")

    heartbeat["stop"] = True
    elapsed = time.time() - t0
    print(f"[INFO] Inferencia terminada en {elapsed:.1f}s. Guardando predicciones...")

    preds_sorted = sorted(preds, key=lambda x: str(x.get("PMID")))
    write_jsonl(args.out_pred, preds_sorted)
    print(f"[INFO] Predicciones guardadas en {args.out_pred}")

    print("[INFO] Calculando métricas (STRICT)...")
    p, r, f1, npred, ngold, ntp = evaluate_strict(gold, preds_sorted)
    print("\n== Métricas (match por mención; case-insensitive) ==")
    print(f"Predicciones: {npred}")
    print(f"Gold:        {ngold}")
    print(f"TP:          {ntp}")
    print(f"Precision:   {p:.4f}")
    print(f"Recall:      {r:.4f}")
    print(f"F1:          {f1:.4f}")
    print("[INFO] Hecho.")

if __name__ == "__main__":
    main()
