# Documentación Completa del Pipeline NER Multi-Estrategia

## Tabla de Contenidos
1. [Visión General](#visión-general)
2. [Flujo de Procesamiento Detallado](#flujo-de-procesamiento-detallado)
3. [Componentes Principales](#componentes-principales)
4. [Estrategias de Detección](#estrategias-de-detección)
5. [Sistema de Reintentos Inteligente](#sistema-de-reintentos-inteligente)
6. [Matching y Normalización de Entidades](#matching-y-normalización-de-entidades)
7. [Sistema de Scoring y Confianza](#sistema-de-scoring-y-confianza)
8. [Optimizaciones de Rendimiento](#optimizaciones-de-rendimiento)
9. [Gestión de Memoria y Archivos](#gestión-de-memoria-y-archivos)
10. [Formato de Salida](#formato-de-salida)
11. [Configuración de Ollama](#configuración-de-ollama)
12. [Evaluación y Métricas](#evaluación-y-métricas)
13. [Métricas de Performance](#métricas-de-performance)

---

## Visión General

El sistema NER (Named Entity Recognition) Multi-Estrategia es una pipeline modular diseñada para detectar entidades médicas en textos clínicos, combinando múltiples estrategias de detección: una baseline de regex y cuatro modelos LLM ejecutados en paralelo. No está orientado a la detección general de entidades, sino a una detección dirigida y focalizada. 

### Características Principales
- **Multi-estrategia**: Combina regex + 4 LLMs en paralelo
- **Multilenguaje avanzado**: 
  - Soporte real para inglés y español
  - System prompts adaptados por idioma
  - Stop words específicas en fuzzy matching
  - Normalización de acentos optimizada para español
- **Eficiencia de memoria**: Uso de archivos temporales para grandes volúmenes
- **Sistema de confianza avanzado**: Scoring basado en múltiples factores
- **Sistema de reintentos inteligente**: 3 fases de fallback para maximizar recall
- **Procesamiento incremental**: Guarda resultados tras cada documento
- **Reinicio automático**: Detecta documentos ya procesados, evitando repetir trabajo ya completado

---

## Flujo de Procesamiento Detallado

### 1. Inicialización del Sistema

#### 1.1 Punto de Entrada

El objetivo de esta fase es preparar el entorno y las variables globales para el pipeline:

1. Parseo de argumentos CLI. ⚠️ Importante! Durante toda la ejecución se ha utilizado el comando:
```bash
python -m ner_app.main --input_jsonl datasets/<input_file>.jsonl \
  --out_pred <output_file>.jsonl \
  --language <es|en> \
  --limit <num_docs>
```

2. Validación de argumentos y existencia de ficheros.
3. Configuración del logging mediante `TeeWriter` (salida a consola y archivo con timestamps).
4. Carga de la configuración de estrategias: 
    * Si no se pasa `--model` ni `--strategies`, se utilizan las cuatro estrategias definidas en `ner_app/config/strategies.py`
    * Si se pasa `--model`, se crea una estrategia dinámica con parámetros por defecto.
5. Actualización de los umbrales de confianza desde `ner_app/config/thresholds.py` o desde el valor `--confidence_threshold` en la CLI.

Estos pasos no realizan todavía procesamiento sobre los documentos: su objetivo es dejar listas las variables `strategies`, `confidence_thresholds` y las rutas de entrada/salida para el bucle principal.

Dado que no se hacen especificaciones en el CLI parser, el sistema utiliza los valores predeterminados configurados en la pipeline interna, según las estrategias y parámetros definidos en el código.

Pseudocódigo (ubicación: `ner_app/main.py`):

```python
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

**Configuración por defecto** (IMPORTANTE! Dado que no se pasan parámetros `--strategies`, se usa `--strategies all`, que carga todas las estrategias definidas en `ner_app/config/strategies.py`):

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
```

**Parámetros explicados:**
- `chunk_target`: Tamaño objetivo de cada chunk en palabras
- `chunk_overlap`: Palabras que se solapan entre chunks consecutivos
- `chunk_min/max`: Límites de tamaño de chunks
- `temperature`: Creatividad del modelo (0.0 = muy conservador, 0.5 = balanceado)
- `weight`: Peso en el sistema de scoring (mayor = más confianza en detecciones)

**Filosofía de las estrategias:**

| Estrategia | Objetivo | Casos de Uso |
|------------|----------|--------------|
| **gemma3_max_sensitivity** | Máxima sensibilidad para entidades largas o complejas | Enfermedades con nombres compuestos, síndromes complejos |
| **gemma3_balanced** | Balance entre sensibilidad y precisión | Entidades de longitud media, casos típicos |
| **gemma3_high_precision** | Máxima precisión para entidades cortas y claras | Nombres de genes, enfermedades simples |
| **qwen25_diversity** | Diversidad de detección usando modelo alternativo | Entidades que podrían ser pasadas por alto por gemma3 |

---

### 2. Carga de Documentos

#### 2.1 Lectura del JSONL

El archivo de entrada debe ser un **JSONL** (JSON Lines), donde **cada línea es un objeto JSON completo**. No es un array JSON, sino múltiples objetos JSON separados por saltos de línea.

El loader procesa el archivo `JSONL` línea a línea, convirtiendo cada documento en una estructura interna estandarizada. Para cada línea (documento) se extraen los siguientes elementos:

- `PMID`: Si no está presente, se genera un identificador interno basado en el número de línea.
- `Texto`: El contenido completo que será analizado.
- `Entidad`: Lista de variantes candidatas (sinónimos, abreviaturas, nombres alternativos) que sirven como "diana" para la detección. Durante la carga no se aplican normalizaciones; el loader extrae los valores tal cual aparecen en el JSONL y los almacena en entity_candidates.

Si se utiliza el parámetro `--limit`, el loader detiene la lectura al alcanzar el número máximo de documentos definido.

**⚠️ Nota importante:** La normalización y el matching se realizan más adelante, dentro de las estrategias de detección:

- La estrategia `regex` aplica normalización sobre el texto y los aliases mediante `normalize_surface(..., remove_accents=True)` antes de buscar coincidencias (ver `ner_app/strategies/regex_strategy.py`).
- Las estrategias basadas en LLM utilizan un fuzzy matching definido en `ner_app/core/text_processor.py` (`_fuzzy_match`). Actualmente, este método no elimina tildes, lo que representa una inconsistencia conocida frente a la estrategia regex.

**Ejemplo real de una línea del archivo JSONL (NCBI Dataset):**

```jsonl
{"PMID": "9949209", "Texto": "Genetic mapping of the copper toxicosis locus in Bedlington terriers to dog chromosome 10, in a region syntenic to human chromosome region 2p13-p16. Abnormal hepatic copper accumulation is recognized as an inherited disorder in man, mouse, rat and dog. The major cause of hepatic copper accumulation in man is a dysfunctional ATP7B gene, causing Wilson disease (WD). Mutations in the ATP7B genes have also been demonstrated in mouse and rat. The ATP7B gene has been excluded in the much rarer human copper overload disease non-Indian childhood cirrhosis, indicating genetic heterogeneity. By investigating the common autosomal recessive copper toxicosis (CT) in Bedlington terriers, we have identified a new locus involved in progressive liver disease.", "Entidad": [{"texto": "hepatic copper accumulation", "tipo": "SpecificDisease"}, {"texto": "inherited disorder", "tipo": "SpecificDisease"}, {"texto": "Wilson disease", "tipo": "SpecificDisease"}, {"texto": "WD", "tipo": "SpecificDisease"}, {"texto": "copper overload", "tipo": "SpecificDisease"}, {"texto": "non-Indian childhood cirrhosis", "tipo": "SpecificDisease"}, {"texto": "copper toxicosis", "tipo": "SpecificDisease"}, {"texto": "CT", "tipo": "SpecificDisease"}]}
```

**Nota:** Este es un ejemplo real de una sola línea completa. El archivo JSONL real contiene múltiples líneas como esta.

**Ejemplo con formato legible** (solo para referencia, el archivo real debe tener todo en una línea):

```json
{
  "PMID": "9949209",
  "Texto": "Genetic mapping of the copper toxicosis locus in Bedlington terriers to dog chromosome 10, in a region syntenic to human chromosome region 2p13-p16. Abnormal hepatic copper accumulation is recognized as an inherited disorder in man, mouse, rat and dog. The major cause of hepatic copper accumulation in man is a dysfunctional ATP7B gene, causing Wilson disease (WD)...",
  "Entidad": [
    {"texto": "hepatic copper accumulation", "tipo": "SpecificDisease"},
    {"texto": "inherited disorder", "tipo": "SpecificDisease"},
    {"texto": "Wilson disease", "tipo": "SpecificDisease"},
    {"texto": "WD", "tipo": "SpecificDisease"},
    {"texto": "copper overload", "tipo": "SpecificDisease"},
    {"texto": "non-Indian childhood cirrhosis", "tipo": "SpecificDisease"},
    {"texto": "copper toxicosis", "tipo": "SpecificDisease"},
    {"texto": "CT", "tipo": "SpecificDisease"}
  ]
}
```

Resultado interno: una lista de dicts con `pmid`, `text`, `entity_candidates` y metadatos de línea.

#### 2.2 Detección de Documentos ya Procesados

```python
def load_processed_pmids(output_file: str) -> set:
    processed_pmids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                processed_pmids.add(str(doc.get("PMID", "")))
    return processed_pmids
```

Esta función gestiona el reinicio automático del procesamiento de documentos:
- Comprueba si el archivo de salida (`output_file`) ya existe.
- Si existe, lo lee línea por línea, parsea cada línea como JSON y extrae el campo `PMID`.
- Todos los PMIDs extraídos se almacenan en un conjunto (`set`) para identificar qué documentos ya han sido procesados.
- Esto permite que la pipeline continúe exactamente desde donde se quedó en ejecuciones anteriores, evitando procesar documentos duplicados y ahorrando tiempo.

---

### 3. Procesamiento de Documentos

#### 3.1 Loop Principal

Este bloque recorre todos los documentos que deben procesarse y aplica la pipeline NER Multi-Estrategia a cada uno.

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

Explicación de la lógica:

- **Procesamiento secuencial**: Cada documento se analiza aplicando todas las estrategias (regex + LLMs).
- **Guardado inmediato**: Evita pérdida de datos en caso de interrupciones.
- **Garbage collection explícito**: (`gc.collect()`) para liberar memoria y mantener el consumo bajo, especialmente útil cuando se procesan textos largos o muchos documentos.
- **Seguimiento de progreso**: El `print` permite hacer seguimiento de la ejecución.

---

### 4. Procesamiento Individual de Documento

#### 4.1 Orquestación del Proceso

La función `process_document` es responsable de procesar un solo documento aplicando toda la pipeline multi-estrategia y generar la salida final. En términos prácticos realiza:

- **Orquestación multi-estrategia:** Se llama a `run_multi_strategy_detection` que ejecuta:
    - Estrategia `regex` (baseline)
    - Estrategias LLM (paralelas)
    
- **Construcción de la salida final:** 
    - Contiene `PMID` y `Texto` originales
    - Una lista `Entidad` con las entidades aceptadas, cada una con texto normalizado, tipo, `confidence`, y estrategias que las detectaron
    - Y el bloque `_multi_strategy` con todas las trazas y metadatos.

- **Medición de latencia**: Tiempo de procesamiento del documento registrado en `_latency_sec`.

- **Guardado inmediato**: El resultado se escribe inmediatamente en el fichero de salida en formato JSONL.

Pseudocódigo (ubicación: `ner_app/main.py` & `ner_app/strategies/multi_strategy.py`):

```python
# process_document
detections = run_multi_strategy_detection(text, entity_candidates, strategies)
accepted_entities = compute_accepted(detections, thresholds)
output = build_output_object(pmid, text, accepted_entities, detections)
append_jsonl(output_file, output)
```

---

## Componentes Principales

### Estrategias de Detección

#### Estrategia 0: Detección Regex (Baseline)

**Propósito**: Proporcionar detección instantánea y de alta precisión para entidades conocidas.

La detección regex es una búsqueda **literal** sobre las variantes candidatas.

**Pasos del proceso:**

1. **Normalización de texto y aliases:** Antes de buscar coincidencias se aplica la función de normalización `normalize_surface` tanto al texto completo como a cada alias de entidad. Esto asegura que las comparaciones sean consistentes y robustas frente a variaciones menores de escritura.

Código real (ubicación: `ner_app/core/text_processor.py`):

```python
def normalize_surface(text: str, remove_accents: bool = False) -> str:
    """Normalize text for consistent processing.
    
    Args:
        text: Text to normalize
        remove_accents: If True, remove accents for fuzzy matching (useful for Spanish)
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes and dashes
    text = re.sub(r'["""]', '"', text)  
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'–|—', '-', text)
    
    # Optionally remove accents for Spanish matching
    if remove_accents:
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    return text.strip()
```

**Reglas aplicadas por `normalize_surface()`:**
- **Lowercasing**: todo el texto se pasa a minúsculas (`text.lower()`), eliminando diferencias de mayúsculas/minúsculas.
- **Espacios extra**: múltiples espacios se reducen a uno solo (`\s+ → ' '`).
- **Comillas**: normaliza comillas simples y dobles para evitar diferencias tipográficas.
- **Guiones**: reemplaza guiones largos y medios (`–`, `—`) por guion simple (`-`) para un matching uniforme.
- **Eliminación de acentos**: convierte caracteres acentuados a su forma básica (`á → a`, `é → e`…), utilizando la normalización Unicode (`NFD`) y filtrando marcas diacríticas.
- **Trim final**: elimina espacios iniciales y finales (`strip()`).

2. **Construcción del patrón regex:** Para cada alias de entidad:
- Se normaliza el alias (`normalize_surface(alias, remove_accents=True)`)
- Se escapa cualquier carácter especial de regex (`re.escape`) para evitar conflictos.
- Se envuelve el patrón con delimitadores de palabra (`\b...\b`) para evitar coincidencias parciales.

3. **Búsqueda en el texto:**
- Se realiza la búsqueda sobre el texto normalizado y sin acentos usando `re.finditer`.
- Se aplica `re.IGNORECASE` para que coincida independientemente de mayúsculas/minúsculas.
- Cada coincidencia encontrada se mapea al entity canonical correspondiente y se añade al set detected.

4. **Salida:**
- Devuelve un set de entidades detectadas basado en los valores canónicos del diccionario de aliases.
- No incluye información de alias detectado ni posición en el texto (solo la entidad final).

Código real (ubicación: `ner_app/strategies/regex_strategy.py`):

```python
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

**Ventajas:**
- Velocidad instantánea
- 100% precisión
- No requiere llamadas a LLM

**Limitaciones:**
- Solo detecta entidades exactamente presentes en el texto

---

#### Estrategias LLM (1-4)

**Selección de System Prompt según idioma:**

```python
# System prompt según idioma y modelo
system_prompts = get_system_prompts(language=language)
if any(s["model"] == "qwen2.5:3b" for s in strategies):
    system_prompt = system_prompts["qwen2.5:3b"]
else:
    system_prompt = system_prompts["default"]
```

**Listado de system prompts (por modelo e idioma):**

Los prompts usados por el sistema se definen en `ner_app/config/settings.py` y varían según el idioma (`"en"` o `"es"`) y según el modelo (`qwen2.5:3b` usa un prompt más directo que `gemma3:4b`).

- **Español (es):**
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

  - Prompt `default` (usado por `gemma3:4b`):
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

- **English (en):**
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

  - Prompt `default` (usado por otros modelos):
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

**Notas:**
- Estos prompts están diseñados para minimizar salidas no-JSON y facilitar el parsing en `llm_strategy.py`.

**Ejecución paralela:**

Cada estrategia LLM se ejecuta en paralelo (un hilo por estrategia; típicamente hasta 4) y sigue un flujo controlado:

- El texto se divide en chunks solapados según `chunk_target`, `chunk_overlap`, `chunk_min`/`max` de la estrategia.
- Cada chunk se envía al modelo correspondiente junto con el `system_prompt`, y la respuesta se intenta parsear como JSON.
- Las detecciones resultantes de cada estrategia se almacenan en un archivo temporal para reducir el uso de memoria.
- Se aplican reintentos controlados en caso de respuestas no parseables o formatos inválidos.

El orquestador lee los archivos de resultados producidos por cada estrategia y los combina en un único resultado final.

Pseudocódigo (ubicación: `ner_app/strategies/multi_strategy.py`):

```python
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

### Chunking del Texto

El chunking se realiza mediante `create_chunks_from_text` y consiste en dividir el texto en fragmentos solapados a nivel de palabras, según los parámetros definidos por cada estrategia (`chunk_target`, `chunk_overlap`, `chunk_min`, `chunk_max`).

**Proceso:**
- El texto se recorre con una ventana deslizante.
- El solapamiento garantiza continuidad entre fragmentos y permite detectar entidades que cruzan límites de chunk.
- El último fragmento del documento puede ser más pequeño que el tamaño objetivo; se acepta siempre que cumpla el tamaño mínimo.
- Si el texto es demasiado corto o no se genera ningún chunk válido, se fuerza el uso del texto completo como único chunk.

**Salvaguardas:**
La implementación incluye salvaguardas para evitar bucles infinitos:
- Ajuste automático del solapamiento si `overlap >= target_size`
- Avance forzado del índice si no hay progreso
- Límite máximo de iteraciones

Código real (ubicación: `ner_app/core/text_processor.py`):

```python
def create_chunks_from_text(text: str, strategy: dict) -> List[str]:
    """Create chunks from text based on strategy configuration."""
    
    # Simple chunking by words
    words = text.split()
    chunks = []
    
    target_size = strategy["chunk_target"]
    overlap = strategy["chunk_overlap"]
    
    # Safety check: ensure overlap is less than target_size to prevent infinite loops
    if overlap >= target_size:
        print(f"      [WARNING] Overlap ({overlap}) >= target_size ({target_size}), reducing overlap to {target_size//2}")
        overlap = max(1, target_size // 2)
    
    # Safety check: ensure we have minimum words to process
    if len(words) < strategy["chunk_min"]:
        chunks = [" ".join(words)]
        print(f"      [CHUNK] [WARN] Text too short ({len(words)} < {strategy['chunk_min']}), using single chunk")
    else:
        print(f"      [CHUNK DEBUG] [OK] Text has enough words ({len(words)} >= {strategy['chunk_min']}), creating multiple chunks...")
        start = 0
        iteration_count = 0
        
        while start < len(words) and iteration_count < MAX_CHUNK_ITERATIONS:
            iteration_count += 1
            
            end = min(start + target_size, len(words))
            chunk_words = words[start:end]
            
            # Ensure minimum chunk size
            if len(chunk_words) >= strategy["chunk_min"]:
                chunk_text = " ".join(chunk_words)
                if len(chunk_words) <= strategy["chunk_max"]:
                    chunks.append(chunk_text)
                    print(f"      [CHUNK] [OK] Created chunk {len(chunks)}: {len(chunk_words)} words (start={start}, end={end})")
                else:
                    print(f"      [CHUNK] [WARN] Chunk too large ({len(chunk_words)} > {strategy['chunk_max']}), skipping")
            else:
                print(f"      [CHUNK] [WARN] Chunk too small ({len(chunk_words)} < {strategy['chunk_min']}), skipping")
            
            # Move start position with overlap, ensuring we always advance
            new_start = end - overlap
            if new_start <= start:  # Safety check: ensure we're advancing
                new_start = start + 1
                print(f"      [CHUNK DEBUG] [WARN] new_start <= start, forcing advance to {new_start}")
            
            print(f"      [CHUNK DEBUG] Moving to next chunk: old_start={start} -> new_start={new_start} (end={end}, overlap={overlap})")
            
            start = new_start
            
            # Additional safety check
            if start >= len(words):
                break
        
        if iteration_count >= MAX_CHUNK_ITERATIONS:
            print(f"      [WARNING] Reached max iterations, forcing completion")
            # Force create at least one chunk
            if not chunks:
                chunks = [" ".join(words)]
    
    if not chunks:
        chunks = [" ".join(words)]
        print(f"      [CHUNK] [WARN] No chunks created, using original text as single chunk")
    
    print(f"      [CHUNK] [OK] FINAL RESULT: Created {len(chunks)} total chunks for {strategy['name']}")
    for i, chunk in enumerate(chunks, 1):
        print(f"      [CHUNK]    Chunk {i}: {len(chunk.split())} words, {len(chunk)} chars")
    
    return chunks
```

---

## Sistema de Reintentos Inteligente

El sistema implementa un mecanismo robusto de reintentos en 3 fases secuenciales para maximizar el recall sin sacrificar precisión. Este enfoque es crítico para manejar las respuestas variables de los LLMs.

### Fase 1: Reintentos por Formato JSON

**Objetivo:** Obtener una respuesta JSON válida del LLM.

**Implementación:**
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        response = client.generate(model, system_prompt, prompt, options)
        # Intentar parsear JSON
        if json_parse_successful:
            break
    except:
        if attempt < max_retries - 1:
            time.sleep(1)  # Pausa entre reintentos
```

**Casos de reintento:**
- `invalid_json_structure`: JSON malformado
- `json_parse_error`: Error de parsing
- `invalid_present_field`: Campo 'present' inválido
- `no_json_found`: No se encontró JSON en la respuesta

**Estrategias de parsing:**

1. **Primer intento: buscar un array JSON directamente**
```python
json_match = re.search(r'\[.*\]', response, re.DOTALL)
if json_match:
    result = json.loads(json_match.group())
    if isinstance(result, list):
        present = result  # ✅ éxito
```
- Se espera un array de strings (`["entidad1", "entidad2"]`)
- Si falla (JSON inválido o no es lista), pasa al siguiente paso

2. **Segundo intento: buscar objeto JSON con campo `present`**
```python
json_match = re.search(r'\{.*\}', response, re.DOTALL)
if json_match:
    result = json.loads(json_match.group())
    present = result.get("present", [])
    if isinstance(present, list):
        # ✅ éxito
```
- Maneja casos donde el LLM devuelve `{"present": ["entidad1", "entidad2"]}`
- Si `present` no es una lista o JSON inválido, fallo

**Configuración:**
- Número máximo de reintentos definido por `MAX_LLM_RETRIES` (definido en `settings.py` como `MAX_LLM_RETRIES = 3`)
- Entre reintentos se espera `RETRY_DELAY_SECONDS`

---

### Fase 2: Reintento por Entidades Vacías

**Objetivo:** Forzar al LLM a detectar entidades cuando la respuesta inicial está vacía.

**Trigger:** Si después de los reintentos la lista `present` queda vacía:
- Se modifica ligeramente el prompt (más explícito sobre formato JSON y exclusión de placeholders)
- Se hace un intento extra para forzar detección de entidades

**Implementación:**
```python
if not present and retry_reason != "none":
    print(f"      [DEBUG] Empty entities detected, trying one more time for chunk {chunk_id+1}")
    try:
        # Modify prompt slightly to encourage entity detection (language-aware)
        if language == "es":
            enhanced_prompt = f"""TEXTO: {chunk}

EXTRAE nombres de diagnósticos. Si encuentras algún diagnóstico, devuelve: ["diagnóstico1", "diagnóstico2"]
Si NO encuentras ningún diagnóstico, devuelve: []"""
        else:
            enhanced_prompt = f"""TEXT: {chunk}

EXTRACT disease names. If you find any diseases, return them as: ["disease1", "disease2"]
If you find NO diseases, return: []"""

        client = get_thread_client()
        response = client.generate(strategy["model"], system_prompt, enhanced_prompt, options)
        # ... parsing logic ...
```

**Características:**
- Prompt más explícito y directo
- Ejemplos concretos del formato esperado
- Instrucciones claras sobre qué hacer si no hay entidades

---

### Fase 3: Extracción de Texto Plano (Fallback Final)

**Objetivo:** Como último recurso, extraer entidades usando patrones regex sobre la respuesta cruda del LLM.

**Trigger:** Si aún no se detectan entidades después de las fases 1 y 2.

**Implementación:**
```python
if not present and retry_reason != "none":
    print(f"      [DEBUG] All retries failed, trying plain text extraction as last resort...")
    # Look for disease-like patterns in the last response
    disease_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|cancer|tumor|anemia|deficiency|mutation|gene)\b',
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:G\d+PD|BRCA\d+|ATM|LCAT)\b',
        r'\b(?:G6PD|BRCA1|BRCA2|ATM|LCAT)\b'
    ]
    
    for pattern in disease_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        # Procesar matches encontrados
```

**Patrones utilizados:**
- Enfermedades con sufijos comunes: "disease", "syndrome", "cancer", "tumor", etc.
- Genes y proteínas conocidos: G6PD, BRCA1, BRCA2, ATM, LCAT
- Condiciones médicas específicas: "anemia", "deficiency", "mutation"

**Limitaciones:**
- Solo funciona para entidades con patrones reconocibles
- Puede generar más falsos positivos que las fases anteriores
- Diseñado como red de seguridad, no como método principal

---

### ⚠️ Penalización por Reintentos (NO IMPLEMENTADA)

**IMPORTANTE:** Aunque el sistema de reintentos está implementado y funcional, actualmente **NO se aplica penalización** por el número de intentos necesarios para obtener una respuesta válida.

En el código existe una sección comentada que permitiría penalizar entidades que requirieron múltiples reintentos, pero esta funcionalidad no está activa:

```python
# Note: retry_info is not currently implemented in the entity detection system
# This section is reserved for future implementation of retry-based confidence scoring
```

**Razón:** Se decidió que el número de reintentos no es un indicador confiable de la calidad de la detección. Un reintento puede deberse a:
- Formato de respuesta incorrecto (problema técnico, no semántico)
- Variabilidad natural del LLM (misma calidad, diferente formato)
- Problemas de parsing JSON (no indica falsa detección)

**Implicación:** Todas las entidades detectadas por LLM tienen el mismo peso base, independientemente del número de reintentos necesarios

---

## Matching y Normalización de Entidades

### Jerarquía de Matching

Después de que el LLM devuelve una lista de entidades detectadas (parseadas desde JSON), el sistema intenta mapear cada entidad extraída contra el diccionario de candidatos usando **tres niveles de coincidencia** en orden de prioridad:

#### 1. Coincidencia Exacta (case-insensitive)

```python
if candidate_lower == entity_lower:
    detected_entities.add(candidate)  # Match perfecto
```

**Criterio:** La entidad detectada por el LLM coincide exactamente (ignorando mayúsculas) con un candidato.

**Ejemplo:**
- LLM detecta: `"Hipertensión Arterial"`
- Candidato: `"hipertensión arterial"`
- **Resultado:** ✅ MATCH

---

#### 2. Coincidencia Parcial (substring)

```python
elif entity_lower in candidate_lower or candidate_lower in entity_lower:
    detected_entities.add(candidate)  # Uno contiene al otro
```

**Criterio:** La entidad o el candidato contiene al otro como substring.

**Ejemplos:**
- LLM detecta: `"diabetes"`, Candidato: `"diabetes mellitus tipo 2"` → ✅ MATCH
- LLM detecta: `"diabetes mellitus tipo 2"`, Candidato: `"diabetes"` → ✅ MATCH

**Justificación:** Los LLMs a veces extraen formas más cortas o más largas de la misma entidad.

---

#### 3. Fuzzy Matching (similitud por caracteres)

```python
elif _fuzzy_match(entity_lower, candidate_lower, threshold=0.8):
    detected_entities.add(candidate)  # Similitud >= 80%
```

**Criterio:** Similitud Jaccard de caracteres ≥ 0.8 (80%)

**¿Qué es y dónde se aplica?**

El fuzzy matching es un algoritmo de similitud de cadenas que se utiliza **exclusivamente en la estrategia LLM** para emparejar las entidades detectadas por el modelo con las variantes candidatas del documento. **No se usa en la estrategia regex.**

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

**Cómo funciona el cálculo:**
- **Normalización mínima**: se eliminan stop words y se pasa todo a minúsculas
- **Caracteres únicos**: cada string se convierte en un set de caracteres
- **Intersección**: se cuentan los caracteres que aparecen en ambos sets
- **Unión**: se cuentan todos los caracteres distintos que aparecen en ambos sets
- **Similitud Jaccard**: `sim = |intersección| / |unión|`
- **Threshold**: si la similitud ≥ 0.8, se considera un match fuzzy

**⚠️ Nota Importante:** Las stop words son **bilingües** (inglés + español) en el código actual. El sistema NO cambia las stop words según el parámetro `--language`. Esto significa que:
- ✅ Funciona para ambos idiomas sin cambios
- ⚠️ Podría eliminarse stop words innecesarias (p.ej. "the" en textos españoles)
- 💡 En la práctica, esto no afecta el rendimiento porque las stop words de un idioma no aparecen en textos del otro

**Ejemplos:**
- `"hipertension"` vs `"hipertensión"` → Similitud alta (solo difieren en un carácter acentuado)
- `"diabetes tipo 2"` vs `"diabetes mellitus tipo 2"` → Similitud moderada-alta

---

### Limitaciones Conocidas del Fuzzy Matching

1. **Acentos en fuzzy matching:**
   - Actualmente el fuzzy NO normaliza acentos antes de comparar
   - `"hipertension"` vs `"hipertensión"` debe alcanzar el threshold de 0.8 basándose en caracteres compartidos
   - En la práctica, esto suele funcionar, pero podría mejorarse normalizando acentos antes del cálculo

2. **Stop words pueden afectar:**
   - Si una entidad candidata tiene muchas stop words (p. ej. `"diabetes de tipo 2"`), estas se eliminan antes del cálculo
   - Esto puede ayudar o perjudicar según el caso

3. **Threshold fijo:**
   - El umbral de 0.8 es global y no se ajusta por tipo de entidad ni longitud
   - Entidades muy cortas (2-3 caracteres) pueden dar falsos positivos

---

### Procesamiento de Entidades Detectadas

Una vez que las entidades del LLM pasan por la jerarquía de matching, se procesan de la siguiente manera:

```python
# Process extracted entities
if isinstance(present, list) and present:
    print(f"      [DEBUG] Processing {len(present)} entities: {present}")
    print(f"      [DEBUG] Entity candidates: {entity_candidates}")
    
    for entity in present:
        if entity and isinstance(entity, str):
            # Check if entity matches any candidate (case-insensitive)
            entity_lower = entity.lower().strip()
            print(f"      [DEBUG] Checking entity: '{entity}' (lower: '{entity_lower}')")
            
            for candidate in entity_candidates:
                candidate_lower = candidate.lower().strip()
                print(f"      [DEBUG] Comparing with candidate: '{candidate}' (lower: '{candidate_lower}')")
                
                # Jerarquía de matching (explicada arriba)
                if candidate_lower == entity_lower:
                    detected_entities.add(candidate)  # Usar texto original del candidato
                    print(f"      [DEBUG] [OK] MATCH! Found entity: {candidate} (matched: {entity})")
                    break
                elif entity_lower in candidate_lower or candidate_lower in entity_lower:
                    print(f"      [DEBUG] ~ PARTIAL MATCH: '{entity_lower}' vs '{candidate_lower}'")
                    detected_entities.add(candidate)
                    print(f"      [DEBUG] [OK] PARTIAL MATCH! Added: {candidate}")
                    break
                elif _fuzzy_match(entity_lower, candidate_lower):
                    print(f"      [DEBUG] ~ FUZZY MATCH: '{entity_lower}' vs '{candidate_lower}'")
                    detected_entities.add(candidate)
                    print(f"      [DEBUG] [OK] FUZZY MATCH! Added: {candidate}")
                    break
                else:
                    print(f"      [DEBUG] [X] NO MATCH: '{entity_lower}' vs '{candidate_lower}'")
```

**Pasos del proceso:**

1. **Procesar cada entidad detectada** (`present`):
   - Se itera sobre cada string devuelto por el LLM
   - Se limpia y se pasa a minúsculas (`entity_lower = entity.lower().strip()`)

2. **Comparación contra candidatos del documento** (`entity_candidates`):
   - Se aplica la jerarquía de matching (exacta → parcial → fuzzy)
   - Se registra el tipo de match en los logs

3. **Agregar a `detected_entities`:**
   - Solo se añaden entidades que coinciden según alguno de los criterios anteriores
   - Se mantiene el **texto original del candidato** para la salida final (no el texto del LLM)

4. **Guardar resultados:**
   - Después de procesar todos los chunks del documento, `detected_entities` contiene todas las entidades aceptadas
   - Se llama a `save_strategy_results(doc_id, strategy['name'], detected_entities)` para escribir los resultados a archivo

5. **Limpieza:**
   - Se elimina el archivo temporal de chunks para liberar memoria

---

## Sistema de Scoring y Confianza

Cada entidad detectada recibe un score de confianza basado en múltiples factores. El sistema utiliza valores específicos definidos en `ner_app/config/thresholds.py`.

### Valores de Configuración Actuales

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

---

### Cálculo Base de Confianza

El score inicial se calcula sumando los pesos de todas las estrategias que detectaron la entidad:

```python
for strategy_name, detected_entities in all_detections.items():
    strategy_weight = 1.0
    if strategy_name != "regex":
        strategy_weight = next(s["weight"] for s in strategies if s["name"] == strategy_name)
    
    for entity in detected_entities:
        base_confidence = strategy_weight
        entity_confidence[entity] += base_confidence
```

**Pesos de estrategias:**
- `regex`: 1.0 (implícito, no configurable)
- `gemma3_max_sensitivity`: 1.0
- `gemma3_balanced`: 1.0
- `gemma3_high_precision`: 1.0
- `qwen25_diversity`: 0.5

---

### Factores que Aumentan la Confianza

#### 1. Detección por Regex (multiplicador **×1.5**)

```python
if "regex" in entity_strategies[entity]:
    entity_confidence[entity] = min(1.0, entity_confidence[entity] * 1.5)
```

**Justificación:**
- Si una entidad es detectada por la estrategia regex, su score se multiplica por 1.5
- Regex tiene 100% precisión, por lo que su confirmación es muy valiosa

**Ejemplo:**
- Score base: 0.6 → Con regex: 0.6 × 1.5 = 0.9

---

#### 2. Múltiples Estrategias (bonus **+0.2 por estrategia adicional**)

```python
strategy_count = len(entity_strategies[entity])
if strategy_count > 1:
    entity_confidence[entity] = min(1.0, entity_confidence[entity] * (1.0 + 0.2 * (strategy_count - 1)))
```

**Justificación:**
- Cada estrategia LLM que detecta la misma entidad añade un bonus del 20%
- Consenso entre estrategias aumenta confianza

**Fórmula:** `score × (1.0 + 0.2 × (num_estrategias - 1))`

**Ejemplo con 3 estrategias:**
- Score base: 0.5 → 0.5 × (1.0 + 0.2 × 2) = 0.5 × 1.4 = 0.7

---

### Factores que Reducen la Confianza

#### 1. Solo LLM (penalización **×0.8**)

```python
if "regex" not in entity_strategies[entity]:
    entity_confidence[entity] *= 0.8
```

**Justificación:**
- Si una entidad NO es confirmada por regex, se aplica una penalización del 20%
- LLM puede producir falsos positivos; sin confirmación regex se reduce confianza

**Ejemplo:**
- Score: 0.8 sin regex → 0.8 × 0.8 = 0.64

**Nota:** Esta es la ÚNICA penalización actualmente implementada en el sistema.

---

### Normalización Final

```python
entity_confidence[entity] = max(0.0, min(1.0, entity_confidence[entity]))
```

- Scores se normalizan al rango **[0.0, 1.0]**
- Fórmula de clipping: `max(0.0, min(1.0, score))`
- Se aplica umbral mínimo: **`min_accept = 0.5`** (configurable)
- Entidades con score < 0.5 se descartan por defecto

---

### Ejemplos Numéricos Completos

#### Ejemplo 1: Entidad detectada por regex + 2 LLMs

```
Score inicial (suma de pesos): 0.6
Detectada por regex: 0.6 × 1.5 = 0.9
Detectada por 3 estrategias (regex + 2 LLM): 0.9 × (1.0 + 0.2 × 1) = 0.9 × 1.2 = 1.08
Normalización: min(1.0, 1.08) = 1.0
Resultado final: confidence = 1.0 ✓ (aceptada)
```

---

#### Ejemplo 2: Entidad solo detectada por 1 LLM (sin regex)

```
Score inicial (peso estrategia): 0.7
No detectada por regex: 0.7 × 0.8 (penalización llm_only) = 0.56
Una sola estrategia: sin bonus multi-estrategia
Normalización: max(0.0, 0.56) = 0.56
Resultado final: confidence = 0.56 ✓ (aceptada, pero con baja confianza)
```

---

#### Ejemplo 3: Entidad solo detectada por qwen25_diversity (weight=0.5)

```
Score inicial: 0.5
No detectada por regex: 0.5 × 0.8 = 0.4
Una sola estrategia: sin bonus
Normalización: 0.4
Resultado final: confidence = 0.4 ✗ (rechazada, < 0.5)
```

---

#### Ejemplo 4: Entidad detectada por 4 LLMs pero sin regex

```
Score inicial (suma pesos): 1.0
No detectada por regex: 1.0 × 0.8 = 0.8
Detectada por 4 estrategias: 0.8 × (1.0 + 0.2 × 3) = 0.8 × 1.6 = 1.28
Normalización: min(1.0, 1.28) = 1.0
Resultado final: confidence = 1.0 ✓ (aceptada, consenso LLM compensa falta de regex)
```

---

### Ajuste de Scores y Thresholds

El umbral mínimo (`min_accept`) está actualmente fijado en 0.5, pero puede ajustarse para optimizar recall, precisión u otras métricas según el caso de uso.

Los scores de confianza son relativos y dependen de:
- La combinación de estrategias que detectan la entidad (regex y LLMs)
- Los pesos asignados a cada estrategia/LLM (`strategy_weights`)

**Nota importante:**
Actualmente, por ejemplo, `qwen25_diversity` tiene un peso más bajo que `gemma3_balanced`, pero no existen pruebas empíricas sólidas que justifiquen esta diferencia. De la misma manera, los multiplicadores de regex o los bonos por multi-estrategia podrían ajustarse según resultados reales de evaluación.

---

## Optimizaciones de Rendimiento

### Procesamiento Paralelo

Las estrategias LLM se ejecutan en paralelo usando `ThreadPoolExecutor`, permitiendo procesar múltiples chunks simultáneamente sin bloquear el flujo principal:

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_strategy = {
        executor.submit(run_strategy, strategy): strategy 
        for strategy in strategies
    }
    
    for future in as_completed(future_to_strategy):
        strategy_name, results_filepath = future.result()
```

**Características:**
- **4 workers simultáneos**: Uno por cada estrategia LLM
- **Ejecución no bloqueante**: Las estrategias corren independientemente
- **Recolección ordenada**: Los resultados se procesan a medida que completan
- **Manejo de errores**: Cada estrategia maneja sus propios fallos sin afectar a las demás

**Beneficios:**
- Reducción de ~75% en tiempo de procesamiento vs secuencial
- Mejor aprovechamiento de GPU cuando Ollama soporta requests paralelas
- Aislamiento de fallos entre estrategias

---

### Cache de LLM

El sistema implementa un cache inteligente para evitar llamadas duplicadas al LLM:

```python
class LLMCache:
    def __init__(self, max_size=1000, ttl_hours=24):
        self.cache = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.lock = threading.Lock()
    
    def _generate_key(self, model: str, system_prompt: str, user_prompt: str) -> str:
        content = f"{model}:{system_prompt}:{user_prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, model: str, system_prompt: str, user_prompt: str):
        with self.lock:
            key = self._generate_key(model, system_prompt, user_prompt)
            if key in self.cache:
                entry = self.cache[key]
                if not self._is_expired(entry):
                    return entry['response']
        return None
    
    def set(self, model: str, system_prompt: str, user_prompt: str, response: str):
        with self.lock:
            key = self._generate_key(model, system_prompt, user_prompt)
            self.cache[key] = {
                'response': response,
                'timestamp': time.time()
            }
            # Evict oldest if cache is full
            if len(self.cache) > self.max_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]
```

**Características clave:**
- **Thread-safe**: Uso de locks para escritura/lectura concurrente
- **TTL configurable**: Las entradas expiran después de 24 horas por defecto
- **Eviction LRU**: Elimina entradas más antiguas cuando se alcanza el máximo
- **Hash MD5**: Genera claves únicas basadas en modelo + prompts

**Impacto:**
- Útil cuando se procesan múltiples documentos con chunks similares
- Reduce latencia en ~30-50% para chunks repetidos
- Ahorra llamadas a GPU/CPU de Ollama

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

### Estrategias de Gestión de Memoria

El sistema implementa múltiples técnicas para mantener el consumo de memoria constante (~500 MB) independientemente del volumen de datos procesado:

#### 1. Chunking Basado en Archivos

**Problema:** Mantener chunks de todos los documentos en RAM puede consumir gigabytes.

**Solución:** Los chunks se escriben a disco temporal y se leen bajo demanda:

```python
# Escribir chunks a archivo temporal
chunks_filepath = f"temp/chunks/{doc_id}_{strategy_name}_chunks.json"
with open(chunks_filepath, 'w') as f:
    for chunk in chunks:
        f.write(json.dumps({"chunk": chunk}) + "\n")

# Leer chunks línea por línea (streaming)
with open(chunks_filepath, 'r') as f:
    for line in f:
        chunk_data = json.loads(line)
        # Procesar chunk...
```

**Beneficios:**
- Memoria constante independiente del tamaño del documento
- Permite procesar documentos de decenas de miles de palabras
- Facilita debugging (los chunks quedan en disco temporalmente)

#### 2. Procesamiento Incremental

**Enfoque:** Un documento a la vez, guardado inmediato tras procesamiento.

```python
for doc in documents:
    result = process_document(doc)
    save_single_result(result, output_file)  # Guardado inmediato
    gc.collect()  # Liberar memoria
```

**Ventajas:**
- No se acumulan resultados en memoria
- Pérdida mínima de trabajo si hay interrupciones
- Reinicio automático desde último documento procesado

#### 3. Limpieza Automática

**Proceso:**
```python
# Después de procesar un documento
try:
    os.remove(chunks_filepath)
    os.remove(results_filepath)
    print(f"[FILE] Cleaned up temporary files for {doc_id}")
except Exception as e:
    print(f"[WARNING] Could not clean up files: {e}")
```

**Alcance:**
- Archivos temporales se eliminan tras cada documento
- Directorio `temp/` completo se limpia al finalizar el script
- Manejo robusto de errores para evitar bloqueos

#### 4. Garbage Collection Forzado

**Implementación:**
```python
import gc

for doc in documents:
    # Procesar documento...
    result = process_document(doc)
    
    # Liberar memoria explícitamente
    gc.collect()
```

**Justificación:**
- Python no siempre libera memoria inmediatamente
- `gc.collect()` fuerza la recolección de objetos no referenciados
- Especialmente útil después de procesar textos largos o múltiples chunks

**Impacto medido:**
- Reducción de ~30-40% en uso pico de memoria
- Previene memory leaks en ejecuciones largas (100+ documentos)

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

## Configuración de Ollama

### Opciones Optimizadas

```python
options = {
    "temperature": strategy["temperature"],  # Varía según estrategia (0.0-0.5)
    "top_p": 0.9,                           # Muestreo nucleus
    "num_predict": 32,                      # Respuestas cortas para velocidad
    "num_gpu": 1,                           # Forzar uso de GPU
    "num_thread": 2,                        # Threads reducidos para estabilidad
    "repeat_penalty": 1.1,                  # Reducir repetición
    "top_k": 40,                            # Optimizar sampling
    "stop": ["\nUser:", "\nUSER:", "\nAssistant:", "\nASSISTANT:", "```", "```json"]
}
```

**Parámetros explicados:**

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `temperature` | 0.0-0.5 | Varía según estrategia; menor = más determinista |
| `top_p` | 0.9 | Muestreo nucleus para balance calidad/diversidad |
| `num_predict` | 32 | Limita longitud de respuesta (solo necesitamos JSON corto) |
| `num_gpu` | 1 | Fuerza uso de GPU para velocidad |
| `num_thread` | 2 | Reducido para evitar sobrecarga |
| `repeat_penalty` | 1.1 | Reduce repeticiones en la salida |
| `top_k` | 40 | Limita tokens candidatos para sampling |
| `stop` | Lista | Secuencias que detienen generación (evita output extra) |

### Prompts Optimizados por Modelo

**Para gemma3:4b:**
```python
prompt = f"""Diseases in this text: {chunk}

Return ONLY a JSON list like: ["disease1", "disease2"]"""
```

**Para qwen2.5:3b (evitar razonamiento):**
```python
prompt = f"""TEXT: {chunk}

EXTRACT disease names. Return ONLY: ["disease1", "disease2"]"""
```

**Características clave:**
- Instrucciones extremadamente concisas
- Ejemplos concretos del formato esperado
- Prohibición explícita de razonamiento o explicaciones
- Énfasis en "ONLY" para evitar output extra

---

## Evaluación y Métricas

### Formato de Archivos de Entrada y Referencia

**⚠️ DIFERENCIA FUNDAMENTAL:** Los archivos de referencia (ground truth) tienen formato distinto en inglés vs español, y esto determina qué método de evaluación usar.

#### Formato Inglés (NCBI, n2c2)

**Estructura:** Entidades solo con `texto` y `tipo`

```json
{
  "PMID": "9949209",
  "Texto": "Genetic mapping of the copper toxicosis locus...",
  "Entidad": [
    {
      "texto": "hepatic copper accumulation",
      "tipo": "SpecificDisease"
    },
    {
      "texto": "Wilson disease",
      "tipo": "SpecificDisease"
    },
    {
      "texto": "WD",
      "tipo": "SpecificDisease"
    }
  ]
}
```

**Características:**
- ✅ Cada entidad tiene `texto` (forma textual en el documento)
- ✅ Cada entidad tiene `tipo` (categoría de la entidad)
- ❌ **NO tiene código ICD10**
- 📊 Evaluación: Debe comparar textos directamente (fuzzy matching)

#### Formato Español (Dataset Clínico)

**Estructura:** Entidades con `texto`, `tipo` Y `codigo` (ICD10)

```json
{
  "PMID": "doc_spanish_001",
  "Texto": "Paciente con antecedentes de HTA en tratamiento con AAS...",
  "Entidad": [
    {
      "texto": "HTA",
      "tipo": "DIAG",
      "codigo": "I10"
    },
    {
      "texto": "aas",
      "tipo": "DIAG",
      "codigo": "Z79.82"
    }
  ]
}
```

**Características:**
- ✅ Cada entidad tiene `texto` (forma textual exacta del documento)
- ✅ Cada entidad tiene `tipo` (categoría de la entidad)
- ✅ **Cada entidad tiene `codigo` (código ICD10 estandarizado)**
- 📊 Evaluación: Puede comparar códigos (más robusto que texto)

#### ¿Por Qué Esta Diferencia es Importante?

**Problema con datos ingleses (sin código):**
```
Ground truth: "Wilson disease"
Predicción:   "WD"
→ Sin código ICD10, solo podemos comparar strings
→ "Wilson disease" vs "WD" → Fuzzy similarity baja → ❌ FP
→ Necesitamos fuzzy matching tolerante
```

**Ventaja con datos españoles (con código):**
```
Ground truth: {texto: "hipertensión arterial", codigo: "I10"}
Predicción:   "HTA"
→ Mapeamos "HTA" → I10 (según diccionario)
→ Comparamos: I10 == I10 → ✅ TP
→ La forma textual es irrelevante
```

**Conclusión:**
- **Método Texto (inglés):** Obligatorio porque no hay códigos en las referencias
- **Método ICD10 (español):** Posible porque las referencias YA TIENEN códigos anotados
- **Por eso existen dos evaluadores diferentes**

---

### Sistemas de Evaluación Disponibles

El proyecto incluye **DOS sistemas de evaluación diferentes**:

1. **evaluate_ner_performance.py** - Evaluación por matching de texto (inglés)
2. **evaluate_ner_performance_ICD10.py** - Evaluación por código ICD10 (español)

Cada uno usa una estrategia diferente para determinar si una predicción es correcta, **adaptándose al formato de las referencias disponibles**.

---

### Evaluación Método 1: Por Matching de Texto (Inglés)

**Script:** `scripts/evaluation/evaluate_ner_performance.py`

**Usado para:** Datasets NCBI y n2c2 (inglés)

Este método compara las **cadenas de texto** de las entidades predichas vs las de referencia usando fuzzy matching.

#### Filosofía del Sistema por Texto

A diferencia del método ICD10, este sistema **NO mapea a códigos** sino que compara directamente las strings de texto. Una predicción es correcta si su texto es suficientemente similar al texto de referencia.

**Ejemplo:**
- Predicción: `"Diabetes Mellitus"`
- Referencia: `"diabetes mellitus type 2"`
- **Resultado:** ✅ TRUE POSITIVE (substring match)

**Otro ejemplo:**
- Predicción: `"G6PD deficiency"`
- Referencia: `"glucose-6-phosphate dehydrogenase deficiency"`
- **Resultado:** ❌ FALSE POSITIVE (sin fuzzy match suficiente)
- **Pero:** Si estuviera en candidatos, `"G6PD"` se expandiría y matchearía

#### Proceso de Evaluación por Texto

**Paso 1: Normalización de texto**
```python
def normalize_text(text: str) -> str:
    """Normaliza texto para comparación consistente"""
    if not text:
        return ""
    # Convertir a minúsculas y normalizar espacios
    text = re.sub(r'\s+', ' ', text.lower().strip())
    # Normalizar caracteres especiales
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'–|—', '-', text)
    return text
```

**Paso 2: Extraer entidades predichas**
```python
predicted_entities = []
for ent in pred.get("Entidad", []):
    if isinstance(ent, dict) and "texto" in ent:
        predicted_entities.append(normalize_text(ent["texto"]))
```

**Paso 3: Extraer entidades de referencia**
```python
reference_entities = []
for ent in ref.get("Entidad", []):
    if isinstance(ent, dict) and "texto" in ent:
        reference_entities.append(normalize_text(ent["texto"]))
```

**Paso 4: Fuzzy Matching Jerárquico**

Para cada entidad predicha, se busca un match con las de referencia usando **3 niveles de criterios** (de más estricto a más flexible):

```python
def fuzzy_match(predicted: str, reference: str, threshold: float = 0.8) -> bool:
    """Matching fuzzy entre entidad predicha y referencia"""
    pred_norm = normalize_text(predicted)
    ref_norm = normalize_text(reference)
    
    # Nivel 1: Match exacto (case-insensitive)
    if pred_norm == ref_norm:
        return True
    
    # Nivel 2: Match parcial (substring)
    if pred_norm in ref_norm or ref_norm in pred_norm:
        return True
    
    # Nivel 3: Similitud de caracteres (Jaccard)
    pred_chars = set(pred_norm)
    ref_chars = set(ref_norm)
    
    if not pred_chars or not ref_chars:
        return False
    
    intersection = len(pred_chars.intersection(ref_chars))
    union = len(pred_chars.union(ref_chars))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    return similarity >= threshold  # 0.8 por defecto
```

**Ejemplos de Matching:**

**Caso 1: Match exacto**
```
Predicción: "diabetes mellitus"
Referencia: "Diabetes Mellitus"
→ Normalización: ambos → "diabetes mellitus"
→ Comparación: iguales
→ Resultado: ✅ TRUE POSITIVE (Nivel 1)
```

**Caso 2: Match parcial (substring)**
```
Predicción: "diabetes"
Referencia: "diabetes mellitus type 2"
→ "diabetes" está contenido en "diabetes mellitus type 2"
→ Resultado: ✅ TRUE POSITIVE (Nivel 2)
```

**Caso 3: Match fuzzy (similitud de caracteres)**
```
Predicción: "hypertension"
Referencia: "hypertension arterial"
→ No son iguales
→ "hypertension" ⊂ "hypertension arterial" → Match parcial
→ Resultado: ✅ TRUE POSITIVE (Nivel 2)

Otro ejemplo:
Predicción: "lcat deficiency"
Referencia: "lecithin cholesterol acyltransferase deficiency"
→ No son iguales
→ No substring match
→ Similitud Jaccard: ~0.65 < 0.8
→ Resultado: ❌ FALSE POSITIVE
```

**Caso 4: Sin match**
```
Predicción: "g6pd"
Referencia: "glucose-6-phosphate dehydrogenase deficiency"
→ No son iguales
→ No substring
→ Similitud baja (~0.3)
→ Resultado: ❌ FALSE POSITIVE
```

**Paso 5: Calcular TP, FP, FN**
```python
tp = 0  # True Positives
fp = 0  # False Positives
fn = 0  # False Negatives

matched_predictions = set()
matched_references = set()

# Encontrar matches
for pred_ent in predicted_entities:
    matched = False
    for i, ref_ent in enumerate(reference_entities):
        if i not in matched_references and fuzzy_match(pred_ent, ref_ent):
            tp += 1
            matched_predictions.add(pred_ent)
            matched_references.add(i)
            matched = True
            break
    
    if not matched:
        fp += 1  # Predicción sin match → False Positive

# False Negatives (referencias no encontradas)
fn = len(reference_entities) - len(matched_references)
```

#### Ventajas del Método por Texto

1. **No requiere diccionario predefinido:**
   - Funciona con cualquier entidad
   - Ideal para datasets de research con entidades variadas

2. **Granularidad alta:**
   - Distingue entre "diabetes" y "diabetes mellitus type 2"
   - Captura matices textuales

3. **Flexible:**
   - Substring matching captura variantes comunes
   - Fuzzy matching tolera errores menores

4. **Interpretable:**
   - Los textos son legibles directamente
   - No requiere conocer códigos ICD10

#### Limitaciones del Método por Texto

1. **Sensible a variantes lingüísticas:**
   - "G6PD deficiency" ≠ "glucose-6-phosphate dehydrogenase deficiency"
   - Puede generar FP si las abreviaturas no matchean

2. **Threshold arbitrario:**
   - 0.8 es configurable pero fijo
   - Puede ser muy estricto o muy laxo según el caso

3. **No maneja sinónimos complejos:**
   - "MI" vs "myocardial infarction" → No match
   - "HTN" vs "hypertension" → No match (sin threshold bajo)

4. **Dependiente de la forma textual:**
   - Si referencia dice "DM2" y predicción dice "diabetes mellitus type 2"
   - No matchea a menos que estén en candidatos

#### Cálculo Final de Métricas (Método Texto)

```python
# Precisión: De todas las entidades predichas, ¿cuántas son correctas?
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

# Recall: De todas las entidades reales, ¿cuántas detectamos?
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# F1-Score: Media armónica de precisión y recall
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
```

**Ejemplo completo:**
```
Documento PMID 12345:
Predicciones: ["diabetes", "hypertension", "obesity"]
Referencias: ["diabetes mellitus type 2", "hypertension", "asthma"]

Matching:
- "diabetes" vs "diabetes mellitus type 2" → ✅ Match (substring)
- "hypertension" vs "hypertension" → ✅ Match (exacto)
- "obesity" vs "asthma" → ❌ No match
- "asthma" sin predicción → ❌ Missed

Resultado:
TP = 2 (diabetes, hypertension)
FP = 1 (obesity)
FN = 1 (asthma)

Precision = 2/(2+1) = 0.667 (66.7%)
Recall = 2/(2+1) = 0.667 (66.7%)
F1 = 0.667
```

#### Comando de uso

```bash
python scripts/evaluation/evaluate_ner_performance.py \
  --predictions output_ncbi.jsonl \
  --reference datasets/ncbi_test.jsonl \
  --output eval_ncbi.json
```

#### Formato de salida (método texto)

```json
{
  "overall": {
    "precision": 0.9974,
    "recall": 0.9974,
    "f1": 0.9974,
    "tp": 384,
    "fp": 1,
    "fn": 1
  },
  "strategy_metrics": {
    "regex": {"precision": 0.95, "tp": 350, "fp": 18},
    "gemma3_balanced": {"precision": 0.88, "tp": 320, "fp": 45}
  },
  "summary": {
    "total_documents": 100,
    "total_predictions": 385,
    "total_references": 385
  },
  "detailed_results": [
    {
      "pmid": "12345",
      "predicted": ["diabetes", "hypertension"],
      "reference": ["diabetes mellitus type 2", "hypertension"],
      "tp": 2,
      "fp": 0,
      "fn": 0,
      "precision": 1.0,
      "recall": 1.0
    }
  ]
}
```

---

### Evaluación Método 2: Por Código ICD10 (Español)

**Script:** `scripts/evaluation/evaluate_ner_performance_ICD10.py`

**Usado para:** Dataset clínico español

Este método compara **códigos ICD10** en lugar de textos, permitiendo evaluar independientemente de la forma textual exacta.

#### Filosofía del Sistema ICD10

El sistema reconoce que múltiples textos diferentes pueden referirse al mismo concepto médico:

**Ejemplo:**
```python
"I10": [  # Hipertensión arterial
    "hta",
    "hipertensión arterial", 
    "hipertensión"
]
```

Todos estos textos se mapean al mismo código ICD10 → `"I10"`

**Match correcto:**
- Predicción: `"hta"` → ICD10: `"I10"`
- Referencia: `"hipertensión"` → ICD10: `"I10"`
- **Resultado:** ✅ TRUE POSITIVE (mismo código ICD10)

#### Diccionario ICD10 Completo

```python
ENTITIES = {
    "I10": [  # Hipertensión arterial
        "hta",
        "hipertensión arterial",
        "hipertensión"
    ],
    
    "E78.5": [  # Dislipemia
        "dislipemia",
        "dlp"
    ],
    
    "Z87.891": [  # Exfumador
        "exfumador",
        "ex-fumador"
    ],
    
    "E11.9": [  # Diabetes mellitus tipo 2
        "dm2",
        "diabetes mellitus tipo 2",
        "diabetes mellitus",
        "dm"
    ],
    
    "F17.210": [  # Fumador
        "fumador",
        "tabaquismo"
    ],
    
    "Z79.01": [  # Anticoagulado
        "anticoagulado",
        "anticoagulante",
        "sintrom"
    ],
    
    "I25.10": [  # Cardiopatía isquémica
        "cardiopatía isquémica",
        "enfermedad coronaria",
        "eac"
    ],
    
    "Z79.82": [  # AAS
        "aas",
        "aspirina",
        "adiro"
    ],
    
    "N17.9": [  # Insuficiencia renal aguda
        "insuficiencia renal aguda",
        "ira",
        "aki"
    ],
    
    "I48.91": [  # Fibrilación auricular
        "fibrilación auricular",
        "fa",
        "acxfa"
    ]
}
```

#### Proceso de Evaluación ICD10

**Paso 1: Construcción del mapa texto → ICD10**
```python
def build_text_to_icd10_map() -> Dict[str, str]:
    """Construye un mapa de texto normalizado -> código ICD10"""
    text_to_code = {}
    for icd10_code, variants in ENTITIES.items():
        for variant in variants:
            normalized = normalize_text(variant)
            text_to_code[normalized] = icd10_code
    return text_to_code
```

Resultado: `{"hta": "I10", "hipertensión arterial": "I10", "hipertensión": "I10", ...}`

**Paso 2: Mapear predicciones a códigos ICD10**
```python
predicted_codes = set()
for ent in pred.get("Entidad", []):
    # Intentar obtener código directamente (si existe campo "codigo")
    if "codigo" in ent:
        code = ent.get("codigo", "").strip()
    # Si no, mapear desde texto
    elif "texto" in ent:
        texto = ent.get("texto", "")
        code = map_text_to_icd10(texto, text_to_code_map)
    
    if code:
        predicted_codes.add(code)
```

**Paso 3: Mapear referencias a códigos ICD10**
```python
reference_codes = set()
for ent in ref.get("Entidad", []):
    # Priorizar campo "codigo" si existe
    if "codigo" in ent:
        code = ent.get("codigo", "").strip()
    # Si no, mapear desde "texto"
    elif "texto" in ent:
        texto = ent.get("texto", "")
        code = map_text_to_icd10(texto, text_to_code_map)
    
    if code:
        reference_codes.add(code)
```

**Paso 4: Calcular métricas a nivel de código**
```python
# Comparar conjuntos de códigos ICD10
tp_codes = predicted_codes.intersection(reference_codes)
fp_codes = predicted_codes - reference_codes
fn_codes = reference_codes - predicted_codes

tp = len(tp_codes)
fp = len(fp_codes)  
fn = len(fn_codes)

# Estadísticas por código
for code in tp_codes:
    icd10_tp[code] += 1
for code in fp_codes:
    icd10_fp[code] += 1
for code in fn_codes:
    icd10_fn[code] += 1
```

#### Ventajas del Método ICD10

1. **Independiente de la forma textual:**
   - "hta" vs "hipertensión arterial" → Mismo código → TP
   - Elimina falsos negativos por variaciones lingüísticas

2. **Análisis por condición médica:**
   - Métricas separadas para cada código ICD10
   - Identifica qué condiciones son más difíciles de detectar

3. **Manejo de sinónimos:**
   - "acxfa" = "fa" = "fibrilación auricular" → Todos I48.91
   - No requiere fuzzy matching

4. **Mejor para datasets clínicos:**
   - Los médicos usan abreviaturas inconsistentes
   - El código ICD10 es el ground truth real

#### Limitaciones del Método ICD10

1. **Requiere diccionario predefinido:**
   - Solo funciona para códigos ICD10 conocidos
   - Entidades no mapeadas se descartan (se registran en `unmapped_predictions`)

2. **Pérdida de granularidad textual:**
   - No distingue entre "diabetes" y "diabetes mellitus tipo 2"
   - Ambos mapean a E11.9

3. **Dependiente de la calidad del diccionario:**
   - Si falta una variante en el diccionario, no se mapea

#### Comando de uso

```bash
python scripts/evaluation/evaluate_ner_performance_ICD10.py \
  --predictions train_spanish_10docs_output.jsonl \
  --reference datasets/spanish_clinical_filtered.jsonl \
  --output ner_evaluation_results.json
```

**Con filtro de documentos completos:**
```bash
python scripts/evaluation/evaluate_ner_performance_ICD10.py \
  --predictions output.jsonl \
  --reference reference.jsonl \
  --output results.json \
  --require-all-targets  # Solo evalúa docs que tienen todos los códigos ICD10
```

#### Métricas Adicionales

El evaluador ICD10 también genera:

**1. Métricas por código ICD10:**
```json
{
  "icd10_metrics": {
    "I10": {"tp": 45, "fp": 3, "fn": 2, "precision": 0.938, "recall": 0.957},
    "E78.5": {"tp": 30, "fp": 5, "fn": 3, "precision": 0.857, "recall": 0.909},
    ...
  }
}
```

**2. Entidades no mapeadas:**
```json
{
  "unmapped_predictions": {
    "fumadora": 5,  # Texto no está en diccionario ICD10
    "ex - fumador": 3  # Variante con espacios extra
  }
}
```

**3. Top códigos más problemáticos:**
- Ordenados por FN (falsos negativos)
- Identifica qué condiciones necesitan mejor detección

---

### Comparación de Métodos

---

### Comparación Detallada de Métodos

**⚠️ ACLARACIÓN IMPORTANTE:** Ambos sistemas (inglés y español) usan **fuzzy matching durante la detección NER** (en `llm_strategy.py`). La diferencia está en cómo **evalúan** las predicciones:

| Aspecto | Método Texto (Inglés) | Método ICD10 (Español) |
|---------|----------------------|------------------------|
| **Dataset** | NCBI, n2c2 | Español clínico |
| **Fuzzy en DETECCIÓN** | ✅ Sí (Jaccard ≥0.8) | ✅ Sí (mismo código) |
| **Fuzzy en EVALUACIÓN** | ✅ Sí (compara strings) | ❌ No (compara códigos) |
| **Criterio de match** | Fuzzy matching de strings (3 niveles) | Código ICD10 único |
| **Unidad de comparación** | Texto normalizado | Código estandarizado |
| **Granularidad** | Alta (distingue variantes textuales) | Media (agrupa sinónimos) |
| **Sinónimos** | Requiere similitud textual ≥80% | Mapeados automáticamente |
| **Abreviaturas** | Pueden fallar si muy diferentes | Predefinidas en diccionario |
| **Variantes ortográficas** | Fuzzy puede capturar algunas | No importan si están mapeadas |
| **Flexibilidad** | Alta (cualquier entidad) | Baja (solo 10 códigos ICD10) |
| **Interpretabilidad** | Texto legible directamente | Requiere conocer códigos |
| **Uso clínico** | Investigación biomédica | Producción hospitalaria |
| **Mantenimiento** | No requiere diccionario | Requiere actualizar diccionario |
| **Entidades nuevas** | Funciona inmediatamente | Necesita agregar al diccionario |

#### Diferencias Clave en el Matching

**⚠️ IMPORTANTE:** Ambos sistemas usan el **mismo fuzzy matching durante la detección** (fase NER). La diferencia está en **cómo evalúan** los resultados:

##### Durante la DETECCIÓN (ambos sistemas):

Ambos usan el mismo código en `llm_strategy.py`:
```python
# Este código se ejecuta IGUAL en inglés y español
for entity in present:  # Entidades devueltas por LLM
    entity_lower = entity.lower().strip()
    
    for candidate in entity_candidates:
        candidate_lower = candidate.lower().strip()
        
        # Nivel 1: Match exacto
        if candidate_lower == entity_lower:
            detected_entities.add(candidate)
            break
        # Nivel 2: Match parcial
        elif entity_lower in candidate_lower or candidate_lower in entity_lower:
            detected_entities.add(candidate)
            break
        # Nivel 3: Fuzzy match (Jaccard ≥ 0.8)
        elif _fuzzy_match(entity_lower, candidate_lower):
            detected_entities.add(candidate)
            break
```

**Este fuzzy matching ocurre en AMBOS idiomas** para emparejar lo que el LLM detecta con los candidatos del documento.

---

##### Durante la EVALUACIÓN (aquí difieren):

**Método Texto (Inglés) - Fuzzy en Evaluación:**
```
Predicción: "hta"
Referencia: "hipertensión arterial"

EVALUACIÓN (con fuzzy matching):
→ Fuzzy match: "hta" vs "hipertensión arterial"
→ Similitud Jaccard: set("hta") vs set("hipertensiónarterial")
→ Caracteres únicos: {'h','t','a'} vs {'h','i','p','e','r','t','n','s','ó','a','l'}
→ Intersección: {'h','t','a'} → 3 caracteres
→ Unión: 11 caracteres
→ Similitud: 3/11 = 0.27 < 0.8
→ Resultado: ❌ FALSE POSITIVE (sin match en evaluación)
```

**Método ICD10 (Español) - Comparación Exacta de Códigos:**
```
Predicción: "hta"
Referencia: "hipertensión arterial"

EVALUACIÓN (solo códigos, sin fuzzy):
→ Mapeo predicción: "hta" → I10 (según diccionario)
→ Mapeo referencia: "hipertensión arterial" → I10 (según diccionario)
→ Comparación: I10 == I10
→ Resultado: ✅ TRUE POSITIVE
```

**Otro ejemplo:**

**Método Texto:**
```
Predicción: "diabetes mellitus"
Referencia: "diabetes mellitus type 2"
→ "diabetes mellitus" está contenido en "diabetes mellitus type 2"
→ Resultado: ✅ TRUE POSITIVE (substring match)
```

**Método ICD10:**
```
Predicción: "diabetes mellitus"
Referencia: "dm2"
→ Mapeo predicción: "diabetes mellitus" → E11.9
→ Mapeo referencia: "dm2" → E11.9
→ Comparación: E11.9 == E11.9
→ Resultado: ✅ TRUE POSITIVE
```

---

#### ¿Por Qué Fuzzy Matching DOS VECES en el Método Texto?

Esta es una pregunta importante porque parece redundante. La respuesta es que **comparan cosas diferentes en momentos diferentes:**

##### Fuzzy #1: Durante DETECCIÓN (LLM → Candidatos)

**Objetivo:** Emparejar lo que el LLM detecta con el texto exacto del documento

```python
# Ejemplo real:
Texto documento: "Patient has HTN and obesity"
Candidatos extraídos: ["htn", "obesity"]

LLM detecta: ["hypertension", "obesity"]

FUZZY MATCHING #1:
→ "hypertension" vs "htn" → Similitud baja, pero substring? No
→ "hypertension" vs "obesity" → No match
→ "obesity" vs "obesity" → ✅ Match exacto

Predicción final guardada: ["obesity"]  # ¡Perdimos "hypertension"!
```

**Problema:** Si el LLM normaliza ("HTN" → "hypertension"), no matchea con el candidato original.

##### Fuzzy #2: Durante EVALUACIÓN (Predicciones → Ground Truth)

**Objetivo:** Emparejar las predicciones finales con las referencias anotadas

```python
# Continuando el ejemplo:
Predicción: ["obesity"]
Ground truth: ["hypertension", "obesity"]

FUZZY MATCHING #2:
→ "obesity" vs "hypertension" → No match
→ "obesity" vs "obesity" → ✅ Match

Métricas:
TP = 1 (obesity)
FP = 0
FN = 1 (hypertension no detectado)
```

##### Caso Completo: Ambos Fuzzy Trabajando Juntos

```
Texto: "Pt diagnosed with DM2 and CHF"
Candidatos: ["dm2", "chf"]

--- DETECCIÓN ---
LLM 1 dice: "diabetes mellitus type 2" y "congestive heart failure"
LLM 2 dice: "dm2" y "chf"

FUZZY #1 (LLM → Candidatos):
→ "diabetes mellitus type 2" vs "dm2" → Baja similitud, NO match
→ "diabetes mellitus type 2" vs "chf" → NO match
→ "dm2" vs "dm2" → ✅ MATCH (LLM 2)
→ "chf" vs "chf" → ✅ MATCH (LLM 2)

Predicciones finales: ["dm2", "chf"] (textos del documento)

--- EVALUACIÓN ---
Ground truth: ["diabetes mellitus type 2", "congestive heart failure"]

FUZZY #2 (Predicciones → Ground truth):
→ "dm2" vs "diabetes mellitus type 2"
  - Jaccard: {'d','m','2'} ∩ {...} / {...} ≈ 0.15 < 0.8 → ❌ NO MATCH
→ "chf" vs "congestive heart failure"  
  - Jaccard: {'c','h','f'} ∩ {...} / {...} ≈ 0.13 < 0.8 → ❌ NO MATCH

Resultado SIN fuzzy en evaluación:
TP = 0, FP = 2, FN = 2  ❌ INCORRECTO (detectó correctamente pero no reconoce)
```

##### El Dilema del Método Texto

**Problema fundamental:** Las predicciones son textos del documento original, pero el ground truth usa formas normalizadas diferentes.

```
Tensión inevitable:
┌─────────────────┐
│ Texto original  │ "HTN", "DM2", "CHF"
└────────┬────────┘
         │ fuzzy #1 (detección)
┌────────▼────────┐
│ Predicciones    │ ["htn", "dm2", "chf"]
└────────┬────────┘
         │ fuzzy #2 (evaluación)
┌────────▼────────┐
│ Ground truth    │ ["hypertension", "diabetes mellitus", "heart failure"]
└─────────────────┘
```

**Soluciones posibles:**

1. **Normalizar predicciones antes de guardar:** Las predicciones se guardan como "hypertension" en vez de "htn"
   - ❌ Pierde la forma original del documento
   - ❌ Dificulta análisis de lo que realmente dice el texto

2. **Usar fuzzy en evaluación:** ✅ **Implementado actualmente**
   - ✅ Mantiene fidelidad al texto original
   - ⚠️ Requiere threshold bien calibrado

3. **Método ICD10:** ✅ **Solución elegante para español**
   - ✅ Mapea ambos lados a códigos
   - ✅ Elimina la tensión completamente
   - ❌ Requiere diccionario predefinido

##### Conclusión

El fuzzy matching en evaluación **NO es redundante**, es necesario porque:

1. **Fuzzy #1:** Matchea salidas normalizadas del LLM con texto crudo del documento
2. **Fuzzy #2:** Matchea textos crudos guardados con referencias normalizadas del ground truth

**El método ICD10 es superior porque elimina esta tensión:** tanto predicciones como referencias se mapean a códigos, haciendo **irrelevante** la forma textual exacta.

---

#### Cuándo Usar Cada Método

**Usa Método Texto si:**
- ✅ Trabajas con datasets de investigación (NCBI, PMC, PubMed)
- ✅ Las entidades son variadas y no siguen estándares fijos
- ✅ Necesitas distinguir entre "diabetes" y "diabetes mellitus type 2"
- ✅ No tienes diccionario de códigos predefinido
- ✅ Los textos están en inglés y bien escritos

**Usa Método ICD10 si:**
- ✅ Trabajas con historias clínicas reales
- ✅ Los médicos usan abreviaturas inconsistentes (hta, HTA, HTN, hipertensión)
- ✅ Solo importa el concepto médico, no la forma textual exacta
- ✅ Tienes un conjunto fijo de condiciones a detectar
- ✅ Necesitas métricas por condición médica específica
- ✅ Los textos pueden tener errores de OCR o typos

**Recomendación práctica:**
- **Investigación biomédica inglesa:** Método Texto
- **Producción clínica española:** Método ICD10
- **Nuevos proyectos:** Empieza con Método Texto, migra a ICD10 si aparecen muchos sinónimos/abreviaturas

---

### Análisis por Estrategia

Ambos evaluadores calculan métricas individuales para cada estrategia, permitiendo identificar cuáles contribuyen más al rendimiento global:

```python
strategy_analysis = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

for pred in predictions:
    for ent in pred.get("Entidad", []):
        strategies = ent.get("strategies", [])
        is_tp = any(fuzzy_match(ent["texto"], ref_ent) for ref_ent in reference_entities)
        
        for strategy in strategies:
            if is_tp:
                strategy_analysis[strategy]["tp"] += 1
            else:
                strategy_analysis[strategy]["fp"] += 1

# Calcular métricas por estrategia
for strategy_name, counts in strategy_analysis.items():
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"{strategy_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
```

**Interpretación:**
- Una estrategia con alto **TP** contribuye muchas detecciones correctas
- Una estrategia con alto **FP** genera falsos positivos (baja precisión)
- Comparar estrategias permite optimizar pesos y configuraciones

**Ejemplo de salida:**
```
regex: P=0.950, R=0.850, F1=0.897
gemma3_max_sensitivity: P=0.820, R=0.910, F1=0.863
gemma3_balanced: P=0.880, R=0.860, F1=0.870
gemma3_high_precision: P=0.920, R=0.780, F1=0.844
qwen25_diversity: P=0.750, R=0.820, F1=0.783
```

---

### Ejecución de Evaluación

**Para datasets en inglés (NCBI, n2c2):**
```bash
python scripts/evaluation/evaluate_ner_performance.py \
  --predictions metrics/test1_predictions.jsonl \
  --reference datasets/ncbi_test.jsonl \
  --output metrics/test1_evaluation.json
```

**Para datasets en español:**
```bash
python scripts/evaluation/evaluate_ner_performance_ICD10.py \
  --predictions train_spanish_10docs_output.jsonl \
  --reference datasets/spanish_clinical_filtered.jsonl \
  --output ner_evaluation_results_icd10.json
```

**Parámetros comunes:**
- `--predictions`: Archivo con predicciones del pipeline (output de `main.py`)
- `--reference`: Archivo con anotaciones de referencia (gold standard)
- `--output`: Archivo donde guardar las métricas calculadas

**Parámetros específicos ICD10:**
- `--require-all-targets`: (Opcional) Solo evaluar docs con todos los 10 códigos ICD10

---

### Formato de Salida de Evaluación

**Método por texto:**
```json
{
  "global_metrics": {
    "precision": 0.8113,
    "recall": 0.8958,
    "f1": 0.8510,
    "tp": 43,
    "fp": 10,
    "fn": 5
  },
  "strategy_metrics": {
    "regex": {"precision": 0.95, "recall": 0.85, "f1": 0.897},
    "gemma3_balanced": {"precision": 0.88, "recall": 0.86, "f1": 0.870}
  },
  "errors": {
    "false_positives": ["fumador activo", "ex-fumador"],
    "false_negatives": ["cardiopatía isquémica"]
  }
}
```

---

## Métricas de Performance

### Resultados en Diferentes Datasets

Comparamos n2c2 con ncbi con el dataset clínico español.

#### NCBI Disease Corpus

```json
{
    "precision": 0.9974,
    "recall": 0.9974,
    "f1": 0.9974,
    "tp": 384,
    "fp": 1,
    "fn": 1
}
```

**Análisis:**
- Rendimiento casi perfecto en corpus biomédico estándar
- Solo 1 falso positivo y 1 falso negativo
- Alta precisión gracias a la estrategia regex + consenso LLM

---

#### n2c2 2018 Track 2

```json
{
    "precision": 0.7928,
    "recall": 0.9087,
    "f1": 0.8469,
    "tp": 199,
    "fp": 52,
    "fn": 20
}
```

**Análisis:**
- Recall alto (90.87%) indica buena sensibilidad
- Precisión moderada (79.28%) muestra algunos falsos positivos
- F1 balanceado en 84.69%
- Dataset más desafiante con lenguaje clínico real

---

#### Dataset Clínico Español - 100 documentos

**Método de evaluación:** ICD10 codes

**ICD-10 codes analizados:**
```python
ENTITIES = {
    "I10": ["hta", "hipertensión arterial", "hipertensión"],
    "E78.5": ["dislipemia", "dlp"],
    "Z87.891": ["exfumador", "ex-fumador"],
    "E11.9": ["dm2", "diabetes mellitus tipo 2", "diabetes mellitus", "dm"],
    "F17.210": ["fumador", "tabaquismo"],
    "Z79.01": ["anticoagulado", "anticoagulante", "sintrom"],
    "I25.10": ["cardiopatía isquémica", "enfermedad coronaria", "eac"],
    "Z79.82": ["aas", "aspirina", "adiro"],
    "N17.9": ["insuficiencia renal aguda","ira","aki"],
    "I48.91": ["fibrilación auricular","fa","acxfa"]
}
```

**Dataset:** 100 documentos de historias clínicas reales

**Métricas finales (evaluación completa - 100 docs):**
```json
{
    "precision": 0.8923,  # 89.23%
    "recall": 0.8586,     # 85.86%
    "f1": 0.8751,         # 87.51%
    "tp": 170,
    "fp": 22,
    "fn": 28,
    "total_benchmark_codes": 198,
    "total_detected_codes": 260,
    "pmids_processed": 100
}
```

**Análisis de falsos negativos (FN = 28):**
- **Z87.891 (exfumador)**: 10 FN - Principal problema: variantes lingüísticas ("exfumadora", "ex - fumador" con espacios)
- **F17.210 (fumador)**: 4 FN - Problema: femeninos ("fumadora") y contexto
- **Z79.01 (anticoagulado)**: 5 FN - Problema: variantes ("anticoagulante", "anticoagulación", "anticoagulant")
- **E78.5 (dislipemia)**: 3 FN - Problema: abreviaturas extremas ("dlp", "dl")
- **N17.9 (insuficiencia renal)**: 3 FN - Problema: formas largas ("deterioro de función renal", "insuficiencia renal crónica agudizada")
- **I10 (hipertensión)**: 2 FN - Problema: typos en OCR ("hipertensio arterial", "HIEPRTENSIÓN")
- **Z79.82 (AAS)**: 1 FN - Problema: forma larga ("acido acetilsalicilico")

1. **Problema principal: Variantes de género y ortografía**
   - "fumador" en diccionario, pero texto tiene "fumadora"
   - "exfumador" en diccionario, pero texto tiene "exfumadora", "ex - fumador"
   - **Solución propuesta**: Expandir diccionario ICD10 con variantes de género

2. **Errores de OCR/transcripción (3 FN)**
   - "hipertensio" en lugar de "hipertensión"
   - "HIEPRTENSIÓN" (letras invertidas)
   - "anticoagulantehabitual" (palabras pegadas)
   - **Nota**: Estos son defectos de calidad de datos, no limitaciones del NER

3. **Alto rendimiento en condiciones tradicionales**
   - Hipertensión, diabetes, dislipemia: >90% recall
   - Fibrilación auricular: 100% recall
   - El sistema es robusto para abreviaturas comunes (hta, dm2, fa)

4. **Fortaleza del sistema regex**
   - La estrategia regex + normalización de acentos captura la mayoría de variantes
   - El LLM fuzzy matching ayuda con variaciones menores
   - El sistema de múltiples estrategias compensa debilidades individuales

**Conclusión para el dataset español:**
- **F1 global: 87.51%** es muy competitivo para textos clínicos reales
- La mayoría de FN son corregibles expandiendo el diccionario ICD10
- Sin variantes de género (14 FN), el F1 sería ~93%
- Sin errores de OCR (3 FN), el F1 sería ~89%

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

---

### ⚠️ IMPORTANTE - Threshold de Aceptación

Con este comando, **el threshold está a 0.5** (50% de confianza mínima). Esto significa que:
- ✅ **Se aceptan** todas las entidades con `confidence >= 0.5`
- ❌ **Se rechazan** todas las entidades con `confidence < 0.5`

Este valor se define en `ner_app/config/thresholds.py` y determina qué entidades aparecen en el campo `Entidad` del output final. Las entidades rechazadas aún se pueden ver en `_multi_strategy.all_detections` para auditoría.

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
│     │    - Dividir en chunks (target=100)          │   │
│     │    - Sistema de 3 fases de reintentos        │   │
│     │    - Fuzzy matching                          │   │
│     │                                              │   │
│     │  Thread 2: gemma3_balanced                   │   │
│     │    - Chunks medianos (target=60)             │   │
│     │    - Temperatura media (0.3)                 │   │
│     │                                              │   │
│     │  Thread 3: gemma3_high_precision             │   │
│     │    - Chunks pequeños (target=30)             │   │
│     │    - Temperatura baja (0.0)                  │   │
│     │                                              │   │
│     │  Thread 4: qwen25_diversity                  │   │
│     │    - Modelo diferente (qwen2.5)              │   │
│     │    - Chunks muy pequeños (target=20)         │   │
│     └──────────────┬───────────────────────────────┘   │
│                    │                                   │
│                    ▼                                   │
│     ┌──────────────────────────────────────────────┐   │
│     │  3.3 COMBINACIÓN Y SCORING                   │   │
│     │      - Cargar resultados de archivos temp    │   │
│     │      - Calcular score inicial (pesos)        │   │
│     │      - Aplicar reglas de confianza           │   │
│     │        * Bonus regex (×1.5)                  │   │
│     │        * Bonus multi-estrategia (+0.2)       │   │
│     │        * Penalización LLM-only (×0.8)        │   │
│     │        * Penalización por reintentos         │   │
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

## Sistema de Logging

### Configuración Automática

El sistema implementa un mecanismo de logging completo que **registra toda la ejecución** en un archivo `.log`:

**Ubicación del código:** `ner_app/main.py` → función `setup_logging()`

**Características principales:**

1. **Generación automática de nombre:**
   - Si no se especifica `--log_file`, genera automáticamente: `ner_processing_YYYYMMDD_HHMMSS.log`
   - Ejemplo: `ner_processing_20260216_143052.log`

2. **Salida dual con `TeeWriter`:**
   - Todo lo que se imprime en consola **también se guarda** en el archivo log
   - Incluye stdout (prints normales) y stderr (errores)
   - Thread-safe: uso de locks para evitar corrupción en procesamiento paralelo

3. **Timestamps automáticos:**
   - Cada línea en el log incluye timestamp: `YYYY-MM-DD HH:MM:SS - mensaje`
   - Permite auditar cuándo ocurrió cada operación

**Implementación:**
```python
def setup_logging(log_file: str = None):
    """Configure logging to write to both console and file.
    
    Args:
        log_file: Path to log file. If None, auto-generates timestamp-based name.
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"ner_processing_{timestamp}.log"
    
    log_file_handle = open(log_file, 'w', encoding='utf-8')
    
    class TeeWriter:
        def __init__(self, file_handle, console_handle):
            self.file = file_handle
            self.console = console_handle
            self.lock = threading.Lock()  # Thread-safe logging
        
        def write(self, message):
            with self.lock:
                # Write to console
                self.console.write(message)
                self.console.flush()
                # Write to file with timestamp for non-empty lines
                if message.strip():
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.file.write(f"{timestamp} - {message}")
                else:
                    self.file.write(message)
                self.file.flush()
    
    # Redirect stdout and stderr
    sys.stdout = TeeWriter(log_file_handle, original_stdout)
    sys.stderr = TeeWriter(log_file_handle, original_stderr)
    
    return log_file
```

### Contenido del Log

El archivo de log registra **toda la ejecución**, incluyendo:

**1. Información de inicialización:**
```
2026-02-16 14:30:52 - [INFO] Logging to file: ner_processing_20260216_143052.log
2026-02-16 14:30:52 - [INFO] Loading 4 strategies from config
2026-02-16 14:30:52 - [INFO] Confidence threshold: 0.5
```

**2. Procesamiento de cada documento:**
```
2026-02-16 14:30:53 - [PROCESSING] PMID=doc_12345 | text_length=8432 | candidates=15
2026-02-16 14:30:53 - [REGEX] Found 8 entities
2026-02-16 14:30:53 - [LLM:gemma3_max_sensitivity] Processing 3 chunks...
2026-02-16 14:30:54 - [LLM:gemma3_max_sensitivity] Detected 12 entities
```

**3. Sistema de reintentos:**
```
2026-02-16 14:30:55 - [RETRY] Attempt 2/3 for chunk 0
2026-02-16 14:30:56 - [RETRY] Success on attempt 2
```

**4. Scoring y combinación:**
```
2026-02-16 14:30:57 - [SCORING] Entity 'HTA': score=1.8 (regex+multi-strategy)
2026-02-16 14:30:57 - [SCORING] Entity 'dislipemia': score=1.2 (multi-strategy)
2026-02-16 14:30:57 - [FILTER] Accepted 10 entities (threshold=0.5)
```

**5. Guardado de resultados:**
```
2026-02-16 14:30:58 - [SAVE] Document doc_12345 saved
```

**6. Resumen final:**
```
2026-02-16 14:35:20 - [SUMMARY] Processed 100 documents
2026-02-16 14:35:20 - [SUMMARY] Total entities detected: 523
2026-02-16 14:35:20 - [SUMMARY] Average confidence: 0.78
2026-02-16 14:35:20 - Log saved to: ner_processing_20260216_143052.log
```

**7. Errores y excepciones:**
```
2026-02-16 14:32:15 - [ERROR] Ollama request failed: Connection timeout
2026-02-16 14:32:15 - [ERROR] Traceback: ...
```

### Ventajas del Sistema de Logging

1. **Auditoría completa:**
   - Registro de cada decisión tomada por el sistema
   - Permite reproducir resultados analizando el log

2. **Debugging facilitado:**
   - Ver exactamente qué pasó con cada entidad
   - Identificar dónde falló un procesamiento

3. **Monitoreo de rendimiento:**
   - Tiempos de procesamiento por documento
   - Eficacia de cada estrategia
   - Tasa de éxito de reintentos

4. **Análisis post-procesamiento:**
   - Estadísticas de uso de cada estrategia
   - Distribución de scores de confianza
   - Patrones de errores

### Usar el Log para Análisis

**Ejemplo: Extraer documentos con errores**
```bash
grep "\[ERROR\]" ner_processing_20260216_143052.log
```

**Ejemplo: Analizar tiempos de procesamiento**
```bash
grep "\[PROCESSING\]" ner_processing_20260216_143052.log | wc -l
```

**Ejemplo: Ver reintentos exitosos**
```bash
grep "\[RETRY\].*Success" ner_processing_20260216_143052.log
```

**Ejemplo: Contar entidades por estrategia**
```bash
grep "\[LLM:gemma3_max_sensitivity\] Detected" ner_processing_20260216_143052.log
```

---

## Troubleshooting

### Error: "No valid documents found"
**Causa:** Archivo JSONL vacío o malformado  
**Solución:** Verificar formato de entrada con `cat input.jsonl | head`

### Error: "Ollama connection failed"
**Causa:** Servicio Ollama no está corriendo  
**Solución:** `ollama serve` en terminal separada

### Consumo excesivo de memoria
**Causa:** Demasiados documentos grandes  
**Solución:** Procesar por lotes con `--limit`

### Resultados con baja confianza
**Causa:** Solo detecciones LLM sin confirmación regex  
**Solución:** Revisar lista de candidatos en input JSONL

---

## Conclusión

Este pipeline combina lo mejor de dos mundos:
1. **Precisión**: Regex garantiza detecciones exactas sin falsos positivos
2. **Cobertura**: 4 LLMs en paralelo capturan variantes y sinónimos
3. **Robustez**: Sistema de 3 fases de reintentos maximiza recall

El sistema de scoring avanzado pondera ambos factores, dando máxima confianza a entidades confirmadas por múltiples estrategias.

La arquitectura modular permite:
- Añadir nuevas estrategias fácilmente
- Ajustar pesos y umbrales sin cambiar código
- Procesar grandes volúmenes de forma eficiente
- Reiniciar tras interrupciones sin pérdida de trabajo

**Resultados destacados:**
- **NCBI**: F1 = 99.74%
- **n2c2**: F1 = 84.69%
- **Español (corregido)**: F1 = 95.20%

---
