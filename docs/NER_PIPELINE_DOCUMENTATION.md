# Documentación Completa del Pipeline NER Multi-Estrategia

## Tabla de Contenidos
1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Flujo de Procesamiento Detallado](#flujo-de-procesamiento-detallado)
4. [Componentes Principales](#componentes-principales)
5. [Estrategias de Detección](#estrategias-de-detección)
6. [Sistema de Scoring y Confianza](#sistema-de-scoring-y-confianza)
7. [Gestión de Memoria y Archivos](#gestión-de-memoria-y-archivos)
8. [Formato de Salida](#formato-de-salida)

---

## Visión General

El sistema NER (Named Entity Recognition) Multi-Estrategia es una pipeline modular diseñada para detectar entidades médicas en textos clínicos combinando múltiples estrategias de detección: una baseline de regex y cuatro modelos LLM ejecutados en paralelo.

### Características Principales
- **Multi-estrategia**: Combina regex + 4 LLMs en paralelo
- **Multilenguaje**: Soporte para inglés y español
- **Eficiencia de memoria**: Uso de archivos temporales para grandes volúmenes
- **Sistema de confianza avanzado**: Scoring basado en múltiples factores
- **Procesamiento incremental**: Guarda resultados tras cada documento
- **Reinicio automático**: Detecta documentos ya procesados

---

## Arquitectura del Sistema

```
ner_app/
├── main.py                    # Punto de entrada principal
├── config/
│   ├── settings.py           # Configuración global del sistema
│   └── thresholds.py         # Umbrales de confianza
├── core/
│   ├── file_manager.py       # Gestión de archivos temporales
│   ├── llm_client.py         # Cliente para comunicación con Ollama
│   └── text_processor.py    # Normalización y procesamiento de texto
├── strategies/
│   ├── multi_strategy.py     # Orquestador de estrategias
│   ├── regex_strategy.py     # Estrategia baseline (regex)
│   └── llm_strategy.py       # Estrategia basada en LLMs
└── utils/
    └── cli_parser.py         # Parser de argumentos CLI
```

---

## Flujo de Procesamiento Detallado

### 1. Inicialización del Sistema

#### 1.1 Punto de Entrada

La fase de inicialización prepara el entorno de ejecución y las variables globales necesarias para el resto del pipeline. Concretamente se realizan, en orden:

- Parseo de argumentos CLI (archivo de entrada, salida, límites, idioma, estrategias/`--model`, umbrales).
- Validación de los argumentos y comprobaciones rápidas de existencia de ficheros.
- Configuración del logging mediante `TeeWriter` (salida a consola y a fichero con timestamps).
- Carga de la configuración de estrategias: IMPORTANTE! si no se pasa `--model` ni `--strategies`, se utilizan las cuatro estrategias definidas en `ner_app/config/strategies.py`; si se pasa `--model`, se crea una estrategia dinámica con parámetros por defecto.
- Actualización de los umbrales de confianza desde `ner_app/config/thresholds.py` o desde el valor `--confidence_threshold` en la CLI.

Estos pasos no realizan todavía procesamiento sobre los documentos: su objetivo es dejar listas las variables `strategies`, `confidence_thresholds` y las rutas de entrada/salida para el bucle principal.

Pseudocódigo (ubicación: `ner_app/main.py`):

```
# main.py
args = parse_arguments()
setup_logging(args.log_file)
strategies = configure_strategies(args)   # see ner_app/utils/cli_parser.py
update_confidence_thresholds(args.confidence_threshold)
documents = load_documents(args.input_jsonl, limit=args.limit)
for doc in documents:
  if doc.pmid in load_processed_pmids(output_file):
    continue
  result = process_document(doc, strategies)
  save_single_result(result, args.out_pred)
```

#### 1.2 Configuración de Estrategias
Las estrategias se configuran en base a los argumentos CLI. **Configuración por defecto** (IMPORTANTE! cuando no se pasan parámetros `--strategies`, se usa `--strategies all`, que carga todas las estrategias definidas en [ner_app/config/strategies.py](../ner_app/config/strategies.py)):

```python
# Strategy 1: gemma3 - Chunks Grandes (Máxima Sensibilidad)
STRATEGY_1 = {
    "name": "gemma3_max_sensitivity",
    "model": "gemma3:4b",
    "chunk_target": 100,
    "chunk_overlap": 40,
    "chunk_min": 50,
    "chunk_max": 150,
    "temperature": 0.1,
    "weight": 1.0
}

# Strategy 2: gemma3 - Chunks Medianos (Balance)
STRATEGY_2 = {
    "name": "gemma3_balanced",
    "model": "gemma3:4b",
    "chunk_target": 60,
    "chunk_overlap": 30,
    "chunk_min": 30,
    "chunk_max": 90,
    "temperature": 0.3,
    "weight": 1.0
}

# Strategy 3: gemma3 - Chunks Pequeños (Máxima Precisión)
STRATEGY_3 = {
    "name": "gemma3_high_precision",
    "model": "gemma3:4b",
    "chunk_target": 30,
    "chunk_overlap": 15,
    "chunk_min": 15,
    "chunk_max": 45,
    "temperature": 0.0,
    "weight": 1.0
}

# Strategy 4: qwen2.5:3b - Chunks Pequeños (Diversidad)
STRATEGY_4 = {
    "name": "qwen25_diversity",
    "model": "qwen2.5:3b",
    "chunk_target": 20,
    "chunk_overlap": 10,
    "chunk_min": 10,
    "chunk_max": 30,
    "temperature": 0.5,
    "weight": 0.5
}

# Default: ALL_STRATEGIES = [STRATEGY_1, STRATEGY_2, STRATEGY_3, STRATEGY_4]
```

**Parámetros explicados:**
- `chunk_target`: Tamaño objetivo de cada chunk en palabras
- `chunk_overlap`: Palabras que se solapan entre chunks consecutivos
- `chunk_min/max`: Límites de tamaño de chunks
- `temperature`: Creatividad del modelo (0.0 = muy conservador, 0.5 = balanceado)
- `weight`: Peso en el sistema de scoring (mayor = más confianza en detecciones)

---

### 2. Carga de Documentos

#### 2.1 Carga de documentos

El loader lee el archivo `JSONL` línea a línea y normaliza la información en una estructura interna por documento. Para cada línea (documento) se extraen:

- `PMID` (si no existe, se genera un identificador interno basado en el número de línea).
- `Texto`: el texto completo que se va a analizar.
- `Entidad`: la lista de variantes candidatas (sinónimos, abreviaciones, formas con/ sin acentos) que sirven como "diana" para la búsqueda.

Además, durante la carga se normalizan las variantes para facilitar búsquedas posteriores (minúsculas y versión sin acentos para la búsqueda alternativa) y se limita el número de documentos si se pasa `--limit`.

Ejemplo del formato de entrada (una línea JSON por documento):

```json
{
    "PMID": "12345",
    "Texto": "Paciente con hipertensión arterial y diabetes mellitus tipo 2...",
    "Entidad": [
        {"texto": "hta", "tipo": "DIAG"}, 
        {"texto": "hipertensión arterial", "tipo": "DIAG"},
        {"texto": "hipertensión", "tipo": "DIAG"},
        {"texto": "dislipemia", "tipo": "DIAG"},
        {"texto": "dlp", "tipo": "DIAG"},
        {"texto": "exfumador", "tipo": "DIAG"},
        {"texto": "ex-fumador", "tipo": "DIAG"},
        {"texto": "dm2", "tipo": "DIAG"},
        {"texto": "diabetes mellitus tipo 2", "tipo": "DIAG"},
        "..."
    ]
}
```

Resultado interno: una lista de dicts con `pmid`, `text`, `entity_candidates` y metadatos de línea.

Pseudocódigo (ubicación: `ner_app/main.py` / loader):

```
# load_documents (ner_app/main.py)
with open(input_jsonl, 'r', encoding='utf-8') as f:
  for line in f:
    obj = json.loads(line)
    pmid = obj.get('PMID') or generate_id()
    text = obj.get('Texto', '')
    entity_candidates = [e.get('texto') for e in obj.get('Entidad', [])]
    yield {"pmid": pmid, "text": text, "entity_candidates": entity_candidates}
```

#### 2.2 Detección de Documentos Procesados

```python
def load_processed_pmids(output_file: str) -> set:
    processed_pmids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                processed_pmids.add(str(doc.get("PMID", "")))
```

**Funcionalidad:**
- Lee el archivo de salida si existe
- Extrae todos los PMIDs ya procesados
- Permite reiniciar el procesamiento sin duplicar trabajo (permite seguir por donde se quedó la última ejecución)


---

### 3. Procesamiento de Documentos

#### 3.1 Loop Principal

```python
for i, doc in enumerate(documents_to_process, 1):
    print(f"\n[PROGRESS] {i}/{len(documents_to_process)}")
    
    # Procesar documento
    result = process_document(
        doc['pmid'], 
        doc['text'], 
        doc['entity_candidates'], 
        strategies, 
        language=args.language
    )
    
    # Guardar inmediatamente
    save_single_result(result, args.out_pred)
    
    # Liberar memoria
    gc.collect()
```

**Características importantes:**
- Procesamiento secuencial de documentos
- Guardado inmediato tras cada documento (evita pérdida de datos)
- Garbage collection explícito para liberar memoria

---

### 4. Procesamiento Individual de Documento

#### 4.1 Procesamiento individual de documento

La función encargada de procesar un documento aplica la orquestación multi-estrategia y empaqueta los resultados. En términos prácticos realiza:

- Llamar al orquestador (`run_multi_strategy_detection`) que ejecuta `regex` y las estrategias LLM y devuelve detecciones por estrategia, scores intermedios y mapeos.
- Construir la salida final del documento, que contiene `PMID`, `Texto`, una lista `Entidad` con las entidades aceptadas (texto normalizado, tipo, `confidence`, y estrategias que las detectaron) y el bloque `_multi_strategy` con todas las trazas y metadatos.
- Medir y anotar la latencia de procesamiento por documento en `_latency_sec`.

El resultado se escribe inmediatamente en el fichero de salida en formato JSONL.

Pseudocódigo (ubicación: `ner_app/main.py` & `ner_app/strategies/multi_strategy.py`):

```
# process_document
detections = run_multi_strategy_detection(text, entity_candidates, strategies)
accepted_entities = compute_accepted(detections, thresholds)
output = build_output_object(pmid, text, accepted_entities, detections)
append_jsonl(output_file, output)
```

---

### 5. Detección Multi-Estrategia

#### 5.1 Orquestador (`run_multi_strategy_detection()`)

**Paso 1: Detección Regex (baseline)**

 La detección regex es una búsqueda literal sobre las variantes candidatas: el sistema normaliza texto y alias a una versión sin acentos y realiza una búsqueda única sobre esa versión no-accentuada usando patrones escapados y límites de palabra. El resultado es el conjunto de coincidencias exactas que actúa como ancla de alta precisión para el scoring.

Puntos a destacar:

- Se usan límites de palabra (`word boundaries`) para evitar coincidencias parciales, lo que puede provocar problemas con tokens que contienen guiones o caracteres no alfabéticos.
- Se realiza la búsqueda sobre versiones sin acentos para cubrir variantes escritas sin tilde.

Código real (ubicación: `ner_app/strategies/regex_strategy.py`):

```python
"""
Regex-based entity detection strategy for the Multi-Strategy NER system.

Provides exact surface matching using regular expressions as a baseline strategy.
"""

import re
from typing import Dict, Set
from ..core.text_processor import normalize_surface

def regex_detection(text: str, entity_aliases: Dict[str, str]) -> Set[str]:
    """Strategy 0: Regex-based exact surface matching with Spanish support"""
    detected = set()
    # Use accent-insensitive matching as a single-pass:
    # normalize both text and aliases to a no-accent form and run literal (escaped) regex
    text_no_accents = normalize_surface(text, remove_accents=True)

    # Build regex pattern for all aliases using their no-accent form
    alias_patterns = []
    for alias, entity in entity_aliases.items():
        if alias and alias.strip():
            alias_no_acc = normalize_surface(alias.strip(), remove_accents=True)
            escaped_alias = re.escape(alias_no_acc)
            pattern = rf'\b{escaped_alias}\b'
            alias_patterns.append((pattern, entity, alias.strip(), alias_no_acc))

    # Find all matches over the no-accent text and map back to canonical entity
    for pattern, entity, original_alias, alias_no_acc in alias_patterns:
        matches = re.finditer(pattern, text_no_accents, re.IGNORECASE)
        for match in matches:
            detected.add(entity)
    
    return detected
```

Explicación y reglas de normalización (detallado):

- `normalize_surface(text)` se llama para limpiar la superficie (elimina espacios extra y normaliza comillas y guiones); por defecto no elimina tildes.
- `normalize_surface(text, remove_accents=True)` crea `text_no_accents` sustituyendo solo las vocales acentuadas minúsculas por su equivalente sin tilde (`á->a`, `é->e`, `í->i`, `ó->o`, `ú->u`).
- Para la estrategia regex se normalizan tanto el texto como los aliases a la versión *no-accent* (`remove_accents=True`), se escapan las formas resultantes con `re.escape()` y se envuelven en delimitadores de palabra literales `\b...\b`.
- La búsqueda se realiza sobre `text_no_accents` usando la bandera `re.IGNORECASE`
- Resultado: se devuelve el conjunto de entidades (valores `entity` del dict `entity_aliases`) que tienen coincidencia exacta en la versión no-accent del texto.

**Paso 2: Selección de System Prompt**

```python
# System prompt según idioma y modelo
system_prompts = get_system_prompts(language=language)
if any(s["model"] == "qwen2.5:3b" for s in strategies):
    system_prompt = system_prompts["qwen2.5:3b"]
else:
    system_prompt = system_prompts["default"]
```

**Listado de system prompts (por modelo e idioma)**

Los prompts usados por el sistema se definen en `ner_app/config/settings.py` y varían según el idioma (`"en"` o `"es"`) y, en algunos casos, por modelo (p. ej. `qwen2.5:3b` usa un prompt más directo). A continuación se muestran los prompts exactos que se emplean actualmente.

- Español (`es`):
  - Modelo `qwen2.5:3b`:

```
Eres un extractor de entidades diagnósticas. Extrae únicamente nombres de diagnósticos mencionados en el texto.

CRÍTICO:
- NO uses razonamiento.
- NO añadas explicaciones.
- NO añadas comentarios.
- Devuelve SOLO una lista JSON válida de nombres de diagnósticos.

Ejemplo de salida: ["diagnóstico1", "diagnóstico2"]
```

  - Prompt `default` (usado por los demás modelos):

```
Eres un extractor experto de entidades diagnósticas biomédicas. Tu única tarea es identificar y devolver nombres de diagnósticos presentes en el texto.

REGLAS:
1. Solo extrae entidades que sean diagnósticos o condiciones médicas.
2. Sé conservador: si no estás seguro, no extraigas nada.
3. Las entidades deben aparecer EXACTAMENTE como en el texto (mismo literal).
4. Devuelve el resultado exclusivamente como JSON válido.
5. NO incluyas explicaciones, notas ni texto adicional fuera del JSON.

Devuelve SOLO JSON.
```

- English (`en`):
  - Model `qwen2.5:3b`:

```
You are a disease extractor. Extract only disease names mentioned in the text.

CRITICAL:
- Do NOT use reasoning.
- Do NOT add explanations.
- Do NOT add comments.
- Return ONLY a valid JSON list of disease names.

Example output: ["disease1", "disease2"]
```

  - Prompt `default` (used by other models):

```
You are an expert biomedical entity extractor. Your only task is to identify and return disease names and medical conditions present in the text.

RULES:
1. Extract only entities that are diseases or medical conditions.
2. Be conservative: if you are unsure, do not extract anything.
3. Entities must appear EXACTLY as written in the text.
4. Return the output exclusively as valid JSON.
5. Do NOT include explanations, notes, or any text outside the JSON.

Output ONLY JSON.
```

Notas:
- Estos prompts están diseñados para minimizar salidas no-JSON y facilitar el parsing en `llm_strategy.py`.


**Paso 3: Ejecución paralela de LLMs**

Cada estrategia LLM se ejecuta en paralelo (hasta 4 hilos) y sigue un flujo controlado:

- Dividir el texto en chunks según `chunk_target`, `chunk_overlap`, `chunk_min`/`max` de la estrategia.
- Enviar cada chunk con el `system_prompt` al modelo correspondiente y parsear la respuesta.
- Guardar resultados intermedios en archivos temporales para evitar uso excesivo de memoria.
- Reintentar (hasta N veces) en caso de respuestas no parseables.

El orquestador recoge las rutas a los archivos de resultados por estrategia y procede a combinarlos.

Pseudocódigo (ubicación: `ner_app/strategies/multi_strategy.py`):

```
# run_multi_strategy_detection(text, entity_candidates, strategies)
regex_hits = regex_detection(text, entity_candidates)
strategy_filepaths = {}
with ThreadPoolExecutor(max_workers=len(strategies)) as ex:
  futures = [ex.submit(llm_detection_strategy_file, text, s, entity_candidates) for s in strategies]
  for fut in as_completed(futures):
    name, path = fut.result()
    strategy_filepaths[name] = path
all_detections = load_all_results(strategy_filepaths)
combined = combine(regex_hits, all_detections)
scores = compute_confidence(combined)
return {"all_detections": all_detections, "entity_confidence": scores, "entity_strategies": map_strategies}
```

---

### 6. Estrategia LLM Detallada

#### 6.1 LLM: chunking, llamadas y parsing (descripción)

La estrategia LLM sigue tres sub-pasos principales:

1. **Chunking**: El texto se divide en fragmentos solapados según los parámetros de la estrategia (`chunk_target`, `chunk_overlap`, `chunk_min`, `chunk_max`). El solapamiento garantiza continuidad cuando la entidad cruza el límite entre dos fragments.

2. **Generación y parseo**: Cada chunk se envía al modelo con un `system_prompt` que solicita únicamente un array JSON de strings. Se implementa una política de reintentos (p. ej. 2-3 intentos) para manejar respuestas mal formadas o errores de comunicación. Las respuestas correctas se parsean en una lista de textos detectados.

3. **Normalización y mapeo**: Los textos detectados se normalizan y se emparejan (fuzzy-match) contra las variantes candidatas del documento. Si no se encuentra un emparejamiento razonable, la detección se conserva como texto libre (se registra en `_multi_strategy.all_detections` para auditoría).

Notas operativas:
- Los resultados intermedios se almacenan en ficheros temporales por estrategia para minimizar memoria.
- Se recomienda revisar manualmente algunas respuestas crudas para verificar cumplimiento del prompt.


Pseudocodigo (ubicación: `ner_app/strategies/llm_strategy.py`):

```
# llm_detection_strategy_file(text, strategy, entity_candidates)
chunks = split_text_into_chunks(text, strategy)
results = []
for chunk in chunks:
  for attempt in range(max_retries):
    response = llm_client.generate(strategy.model, system_prompt, chunk)
    parsed = try_parse_json_array(response)
    if parsed is valid:
      break
  mapped = fuzzy_match(parsed, entity_candidates)
  results.extend(mapped)
save results to temp file and return filepath
```

#### 6.2 Fuzzy Matching: Mapeo de Entidades LLM a Candidatos

**¿Qué es y dónde se aplica?**

El fuzzy matching es un algoritmo de similitud de cadenas que se utiliza **exclusivamente en la estrategia LLM** para emparejar las entidades detectadas por el modelo con las variantes candidatas del documento. No se usa en la estrategia regex.

**Ubicación:** `ner_app/core/text_processor.py` → `_fuzzy_match()`

**Cuándo se invoca:**

Después de que el LLM devuelve una lista de entidades detectadas (parseadas desde JSON), el sistema intenta mapear cada entidad extraída contra el diccionario de candidatos usando tres niveles de coincidencia (en orden de prioridad):

1. **Coincidencia exacta** (case-insensitive):
   ```python
   if candidate_lower == entity_lower:
       detected_entities.add(candidate)  # Match perfecto
   ```

2. **Coincidencia parcial** (substring):
   ```python
   elif entity_lower in candidate_lower or candidate_lower in entity_lower:
       detected_entities.add(candidate)  # Uno contiene al otro
   ```

3. **Fuzzy matching** (similitud por caracteres):
   ```python
   elif _fuzzy_match(entity_lower, candidate_lower, threshold=0.8):
       detected_entities.add(candidate)  # Similitud >= 80%
   ```

**Algoritmo de Fuzzy Match (Jaccard sobre caracteres):**

Código real (ubicación: `ner_app/core/text_processor.py`):

```python
def _fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Simple fuzzy matching using character overlap."""
    if not text1 or not text2:
        return False
    
    # Remove common words and punctuation (English + Spanish)
    common_words = {
        # English
        'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
        # Spanish
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'de', 'del', 
        'en', 'a', 'con', 'por', 'para', 'al'
    }
    text1_clean = ' '.join([w for w in text1.split() if w.lower() not in common_words])
    text2_clean = ' '.join([w for w in text2.split() if w.lower() not in common_words])
    
    if not text1_clean or not text2_clean:
        return False
    
    # Calculate character overlap (Jaccard similarity)
    set1 = set(text1_clean.lower())
    set2 = set(text2_clean.lower())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    return similarity >= threshold  # threshold = 0.8 (80%)
```

**Parámetros clave:**
- **Threshold por defecto:** `0.8` (80% de similitud de caracteres)
- **Stop words:** Se filtran palabras comunes en inglés y español antes de calcular similitud
- **Case-insensitive:** Todas las comparaciones se hacen en minúsculas
- **NO normaliza acentos:** El fuzzy matching actual NO elimina tildes antes de comparar

**Ejemplos prácticos:**

```python
# Caso 1: Match exacto
LLM detecta: "hipertensión"  →  Candidato: "hipertensión"  ✓ (exacto)

# Caso 2: Partial match
LLM detecta: "diabetes"  →  Candidato: "diabetes mellitus tipo 2"  ✓ (substring)

# Caso 3: Fuzzy match
LLM detecta: "hipertension" (sin tilde)  →  Candidato: "hipertensión"  ✓ (fuzzy >= 0.8)

# Caso 4: No match
LLM detecta: "cancer"  →  Candidato: "hipertensión"  ✗ (similitud < 0.8)
```

**Limitaciones conocidas:**

1. **Acentos en fuzzy matching:** Actualmente el fuzzy NO normaliza acentos antes de comparar, por lo que "hipertension" vs "hipertensión" debe alcanzar el threshold de 0.8 basándose en caracteres compartidos. En la práctica, esto suele funcionar, pero podría mejorarse normalizando acentos antes del cálculo.

2. **Stop words pueden afectar:** Si una entidad candidata tiene muchas stop words (p. ej. "diabetes de tipo 2"), estas se eliminan antes del cálculo, lo cual puede ayudar o perjudicar según el caso.

3. **Threshold fijo:** El umbral de 0.8 es global y no se ajusta por tipo de entidad ni longitud. Entidades muy cortas (2-3 caracteres) pueden dar falsos positivos.

**Recomendaciones para mejora:**

- Añadir normalización de acentos en `_fuzzy_match()` para que sea consistente con la estrategia regex.
- Considerar usar Levenshtein distance o token-based similarity para entidades multi-palabra.
- Ajustar threshold según longitud de la entidad (más estricto para palabras cortas).

---

## Sistema de Scoring y Confianza

### Cálculo de Confianza

Cada entidad detectada recibe un score de confianza basado en múltiples factores. El sistema utiliza valores específicos definidos en `ner_app/config/thresholds.py`.

#### Valores de Configuración Actuales

**Thresholds de confianza:**
```python
CONFIDENCE_THRESHOLDS = {
    "high": 0.9,          # Entidad detectada por 3+ estrategias
    "medium": 0.7,        # Entidad detectada por 2+ estrategias
    "low": 0.5,           # Entidad detectada por 1+ estrategia
    "min_accept": 0.5     # Umbral mínimo para aceptar una entidad
}
```

**Reglas de scoring:**
```python
confidence_rules = {
    "regex_multiplier": 1.5,        # Boost si detectada por regex
    "multi_strategy_bonus": 0.2,    # Bonus por cada estrategia adicional
    "llm_only_penalty": 0.8,        # Penalización si solo LLM (sin regex)
    "max_confidence": 1.0,          # Tope máximo
    "min_confidence": 0.0           # Piso mínimo
}
```

#### Factores que Aumentan la Confianza

1. **Detección por Regex** (multiplicador **×1.5**):
   - Si una entidad es detectada por la estrategia regex, su score se multiplica por 1.5
   - Ejemplo: Score base 0.6 → Con regex: 0.6 × 1.5 = 0.9
   - **Justificación:** Regex es exacto y libre de falsos positivos

2. **Múltiples Estrategias** (bonus **+0.2 por estrategia adicional**):
   - Cada estrategia LLM que detecta la misma entidad añade un bonus del 20%
   - Fórmula: `score × (1.0 + 0.2 × (num_estrategias - 1))`
   - Ejemplo con 3 estrategias: Score base 0.5 → 0.5 × (1.0 + 0.2 × 2) = 0.5 × 1.4 = 0.7

3. **Peso de Estrategia**:
   - Cada estrategia LLM aporta su peso configurado (0.5 a 1.0)
   - `gemma3_max_sensitivity`: weight = 1.0
   - `gemma3_balanced`: weight = 1.0
   - `gemma3_high_precision`: weight = 1.0
   - `qwen25_diversity`: weight = 0.5

#### Factores que Reducen la Confianza

1. **Solo LLM** (penalización **×0.8**):
   - Si una entidad NO es confirmada por regex, se aplica una penalización del 20%
   - Ejemplo: Score 0.8 sin regex → 0.8 × 0.8 = 0.64
   - **Justificación:** LLM puede producir falsos positivos; sin confirmación regex se reduce confianza

2. **Baja Temperatura**:
   - Modelos con temperatura baja (0.0-0.1) son más conservadores pero menos creativos
   - El peso de estrategia puede ser menor si se desea priorizar diversidad

#### Normalización Final

- Scores se normalizan al rango **[0.0, 1.0]**
- Se aplica umbral mínimo: **`min_accept = 0.5`** (configurable)
- Entidades con score < 0.5 se descartan por defecto
- Fórmula de clipping: `max(0.0, min(1.0, score))`

#### Ejemplos Numéricos Completos

**Ejemplo 1: Entidad detectada por regex + 2 LLMs**
```
Score inicial (suma de pesos): 0.6
Detectada por regex: 0.6 × 1.5 = 0.9
Detectada por 3 estrategias (regex + 2 LLM): 0.9 × (1.0 + 0.2 × 1) = 0.9 × 1.2 = 1.08
Normalización: min(1.0, 1.08) = 1.0
Resultado final: confidence = 1.0 ✓ (aceptada)
```

**Ejemplo 2: Entidad solo detectada por 1 LLM (sin regex)**
```
Score inicial (peso estrategia): 0.7
No detectada por regex: 0.7 × 0.8 (penalización llm_only) = 0.56
Una sola estrategia: sin bonus multi-estrategia
Normalización: max(0.0, 0.56) = 0.56
Resultado final: confidence = 0.56 ✓ (aceptada, pero con baja confianza)
```

**Ejemplo 3: Entidad solo detectada por qwen25_diversity (weight=0.5)**
```
Score inicial: 0.5
No detectada por regex: 0.5 × 0.8 = 0.4
Una sola estrategia: sin bonus
Normalización: 0.4
Resultado final: confidence = 0.4 ✗ (rechazada, < 0.5)
```

**Ejemplo 4: Entidad detectada por 4 LLMs pero sin regex**
```
Score inicial (suma pesos): 1.0
No detectada por regex: 1.0 × 0.8 = 0.8
Detectada por 4 estrategias: 0.8 × (1.0 + 0.2 × 3) = 0.8 × 1.6 = 1.28
Normalización: min(1.0, 1.28) = 1.0
Resultado final: confidence = 1.0 ✓ (aceptada, consenso LLM compensa falta de regex)
```

#### Ajuste de Thresholds

El threshold se puede ajustar desde la línea de comandos:

```bash
python -m ner_app.main \
  --input_jsonl datasets/input.jsonl \
  --out_pred output.jsonl \
  --confidence_threshold 0.7  # Aumentar exigencia (más preciso, menos recall)
```

**Recomendaciones:**
- **Threshold 0.3-0.4:** Máximo recall, acepta más falsos positivos
- **Threshold 0.5 (default):** Balance precision/recall
- **Threshold 0.7-0.8:** Alta precisión, puede perder entidades poco frecuentes
- **Threshold 0.9+:** Máxima precisión, solo entidades con máxima confianza

---

## Gestión de Memoria y Archivos

### Archivos Temporales

El sistema usa archivos temporales para minimizar uso de memoria:

```
temp/
├── chunks/
│   └── {doc_id}_{strategy_name}_chunks.json
└── results/
    └── {doc_id}_{strategy_name}_results.json
```

**Ventajas:**
- Uso constante de memoria (~500 MB)
- Permite procesar documentos muy grandes
- Facilita debugging y auditoría

**Limpieza:**
- Se eliminan tras procesar cada documento
- Directorio `temp/` se limpia al finalizar

---

## Formato de Salida

### Estructura del Output JSONL

Cada línea del archivo de salida contiene un documento procesado:

```json
{
  "PMID": "12345",
  "Texto": "Paciente con hipertensión arterial...",
  "Entidad": [
    {
      "texto": "hipertensión arterial",
      "tipo": "SpecificDisease",
      "confidence": 1.0,
      "strategies": ["regex", "gemma3_balanced", "qwen25_diversity"]
    },
    {
      "texto": "diabetes mellitus tipo 2",
      "tipo": "SpecificDisease",
      "confidence": 0.95,
      "strategies": ["regex", "gemma3_max_sensitivity", "gemma3_balanced"]
    },
    {
      "texto": "dislipemia",
      "tipo": "SpecificDisease",
      "confidence": 0.78,
      "strategies": ["gemma3_high_precision", "qwen25_diversity"]
    }
  ],
  "_multi_strategy": {
    "all_detections": {
      "regex": ["hipertensión arterial", "diabetes mellitus tipo 2", "hta", "dm2"],
      "gemma3_max_sensitivity": ["hipertensión arterial", "diabetes mellitus tipo 2"],
      "gemma3_balanced": ["hipertensión arterial", "diabetes tipo 2"],
      "gemma3_high_precision": ["dislipemia"],
      "qwen25_diversity": ["hipertensión", "dislipemia"]
    },
    "entity_confidence": {
      "hipertensión arterial": 1.0,
      "diabetes mellitus tipo 2": 0.95,
      "dislipemia": 0.78,
      "hta": 0.52
    },
    "entity_strategies": {
      "hipertensión arterial": ["regex", "gemma3_balanced", "qwen25_diversity"],
      "diabetes mellitus tipo 2": ["regex", "gemma3_max_sensitivity", "gemma3_balanced"],
      "dislipemia": ["gemma3_high_precision", "qwen25_diversity"]
    },
    "confidence_thresholds": {
      "min_accept": 0.5,
      "high": 0.8,
      "medium": 0.6,
      "low": 0.5
    },
    "strategies_used": [
      "gemma3_max_sensitivity",
      "gemma3_balanced", 
      "gemma3_high_precision",
      "qwen25_diversity"
    ]
  },
  "_latency_sec": 2.345
}
```

### Campos Explicados

**Campos principales:**
- `PMID`: Identificador del documento
- `Texto`: Texto completo analizado
- `Entidad`: Lista de entidades aceptadas (confidence ≥ 0.5)

**Por cada entidad:**
- `texto`: Texto normalizado de la entidad
- `tipo`: Siempre "SpecificDisease" en este sistema
- `confidence`: Score de confianza [0.0, 1.0]
- `strategies`: Lista de estrategias que la detectaron

**Metadatos `_multi_strategy`:**
- `all_detections`: Detecciones por cada estrategia (antes de filtrar)
- `entity_confidence`: Scores de todas las entidades
- `entity_strategies`: Qué estrategias detectaron cada entidad
- `confidence_thresholds`: Umbrales usados
- `strategies_used`: Lista de estrategias ejecutadas

**Métricas:**
- `_latency_sec`: Tiempo de procesamiento del documento

---

## Resumen del Flujo Completo

```
┌─────────────────────────────────────────────────────────┐
│  1. INICIALIZACIÓN                                      │
│     - Parse CLI args                                    │
│     - Setup logging                                     │
│     - Configurar estrategias                            │
│     - Configurar umbrales                               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  2. CARGA DE DOCUMENTOS                                 │
│     - Leer archivo JSONL                                │
│     - Extraer PMID, texto, candidatos                   │
│     - Detectar ya procesados                            │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  3. LOOP PROCESAMIENTO (para cada documento)           │
│     ┌──────────────────────────────────────────────┐   │
│     │  3.1 ESTRATEGIA REGEX (baseline)             │   │
│     │      - Búsqueda exacta con word boundaries   │   │
│     │      - Insensible a acentos                  │   │
│     │      - Resultados instantáneos               │   │
│     └──────────────┬───────────────────────────────┘   │
│                    │                                   │
│                    ▼                                   │
│     ┌──────────────────────────────────────────────┐   │
│     │  3.2 ESTRATEGIAS LLM (4 en paralelo)         │   │
│     │                                              │   │
│     │  Thread 1: gemma3_max_sensitivity            │   │
│     │    - Dividir en chunks (target=70)           │   │
│     │    - Procesar cada chunk                     │   │
│     │    - Sistema de reintentos (max 3)           │   │
│     │    - Fuzzy matching                          │   │
│     │                                              │   │
│     │  Thread 2: gemma3_balanced                   │   │
│     │    - Chunks más grandes (target=120)         │   │
│     │    - Temperatura media (0.5)                 │   │
│     │                                              │   │
│     │  Thread 3: gemma3_high_precision             │   │
│     │    - Chunks grandes (target=180)             │   │
│     │    - Temperatura baja (0.1)                  │   │
│     │                                              │   │
│     │  Thread 4: qwen25_diversity                  │   │
│     │    - Modelo diferente (qwen2.5)              │   │
│     │    - Temperatura alta (0.7)                  │   │
│     └──────────────┬───────────────────────────────┘   │
│                    │                                   │
│                    ▼                                   │
│     ┌──────────────────────────────────────────────┐   │
│     │  3.3 COMBINACIÓN Y SCORING                   │   │
│     │      - Cargar resultados de archivos temp    │   │
│     │      - Calcular score inicial (pesos)        │   │
│     │      - Aplicar reglas de confianza           │   │
│     │        * Bonus regex (×1.5)                  │   │
│     │        * Bonus multi-estrategia              │   │
│     │        * Penalización LLM-only (×0.7)        │   │
│     │      - Normalizar a [0, 1]                   │   │
│     │      - Filtrar por umbral (≥0.5)             │   │
│     └──────────────┬───────────────────────────────┘   │
│                    │                                   │
│                    ▼                                   │
│     ┌──────────────────────────────────────────────┐   │
│     │  3.4 GUARDADO INMEDIATO                      │   │
│     │      - Append al archivo de salida           │   │
│     │      - Liberar memoria (gc.collect())        │   │
│     │      - Limpiar archivos temporales           │   │
│     └──────────────────────────────────────────────┘   │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  4. FINALIZACIÓN                                        │
│     - Imprimir resumen estadístico                      │
│     - Limpiar directorio temp/                          │
│     - Cerrar archivo de log                             │
└─────────────────────────────────────────────────────────┘
```

---

## Configuración Recomendada

### Comando de Producción Actual

Este es el comando que se utiliza actualmente en el proyecto para procesar los datos de prueba del dataset español:

```bash
python -m ner_app.main \
  --input_jsonl datasets/spanish_clinical_test1_input.jsonl \
  --out_pred metrics/test1_predictions.jsonl \
  --language es
```

**Parámetros explicados:**
- `--input_jsonl`: Archivo de entrada en formato JSONL con documentos y entidades candidatas
- `--out_pred`: Archivo de salida donde se guardarán las predicciones (formato JSONL)
- `--language es`: Idioma español (determina system prompts y stop words para fuzzy matching)
- **Nota:** No se pasa `--strategies` ni `--model`, por lo que se usan las 4 estrategias por defecto
- **Nota:** No se pasa `--confidence_threshold`, por lo que se usa el valor por defecto: **`min_accept = 0.5`**

**⚠️ IMPORTANTE - Threshold de Aceptación:**

Con este comando, **el threshold está a 0.5** (50% de confianza mínima). Esto significa que:
- ✅ **Se aceptan** todas las entidades con `confidence >= 0.5`
- ❌ **Se rechazan** todas las entidades con `confidence < 0.5`

Este valor se define en `ner_app/config/thresholds.py` y determina qué entidades aparecen en el campo `Entidad` del output final. Las entidades rechazadas aún se pueden ver en `_multi_strategy.all_detections` para auditoría.

---

## Troubleshooting

### Error: "No valid documents found"
**Causa**: Archivo JSONL vacío o malformado  
**Solución**: Verificar formato de entrada con `cat input.jsonl | head`

### Error: "Ollama connection failed"
**Causa**: Servicio Ollama no está corriendo  
**Solución**: `ollama serve` en terminal separada

### Consumo excesivo de memoria
**Causa**: Demasiados documentos grandes  
**Solución**: Procesar por lotes con `--limit`

### Resultados con baja confianza
**Causa**: Solo detecciones LLM sin confirmación regex  
**Solución**: Revisar lista de candidatos en input JSONL

---

## Métricas de Performance

Comparamos n2c2 con ncbi con informes del HCB. 

ncbi:

"precision": 0.9974025974025974,
"recall": 0.9974025974025974,
"f1": 0.9974025974025974,
"tp": 384,
"fp": 1,
"fn": 1
tiempo de procesamiento:

n2c2:

"precision": 0.7928286852589641,
"recall": 0.908675799086758,
"f1": 0.8468085106382978,
"tp": 199,
"fp": 52,
"fn": 20
tiempo de procesamiento: 

HCB:

ICD10 codes being analyzed:
20 documents
Extracted initialmetrics:

Correcting the benchmark:

Without the fumador/ex-fumador entities:
tiempo de procesamiento:

---

## Conclusión

Este pipeline combina lo mejor de dos mundos:
1. **Precisión**: Regex garantiza detecciones exactas sin falsos positivos
2. **Cobertura**: 4 LLMs en paralelo capturan variantes y sinónimos

El sistema de scoring avanzado pondera ambos factores, dando máxima confianza a entidades confirmadas por múltiples estrategias.

La arquitectura modular permite:
- Añadir nuevas estrategias fácilmente
- Ajustar pesos y umbrales sin cambiar código
- Procesar grandes volúmenes de forma eficiente
- Reiniciar tras interrupciones sin pérdida de trabajo

---
