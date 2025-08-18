# 📚 Metodología Detallada del Sistema NER Multi-Estrategia

## 🎯 Visión General

El sistema NER Multi-Estrategia implementa un enfoque híbrido que combina detección basada en reglas (regex) con inferencia de modelos de lenguaje local (LLMs) para maximizar la recuperación de entidades biomédicas mientras mantiene alta precisión.

## 🏗️ Arquitectura del Sistema

### 1. **Estrategia 0: Detección Regex (Baseline)**
- **Propósito**: Proporcionar detección instantánea y de alta precisión para entidades conocidas
- **Implementación**: Matching exacto con word boundaries usando las entidades del dataset como patrones
- **Ventajas**: 
  - Velocidad instantánea
  - 100% precisión
  - No requiere llamadas a LLM
- **Limitaciones**: Solo detecta entidades exactamente presentes en el texto

### 2. **Estrategias 1-4: Detección LLM Multi-Modelo**
- **Propósito**: Capturar entidades que no son detectadas por regex usando diferentes enfoques de chunking
- **Implementación**: 4 estrategias complementarias ejecutándose en paralelo

#### **Estrategia 1: llama32_max_sensitivity**
```python
STRATEGY_1 = {
    "name": "llama32_max_sensitivity",
    "model": "llama3.2:3b",
    "chunk_target": 100,      # Chunks grandes para contexto rico
    "chunk_overlap": 40,      # Overlap alto para no perder entidades
    "chunk_min": 50,          # Mínimo razonable
    "chunk_max": 150,         # Máximo para capturar entidades largas
    "temperature": 0.1,       # Baja temperatura para precisión
    "weight": 1.0
}
```
- **Objetivo**: Máxima sensibilidad para entidades largas o complejas
- **Casos de uso**: Enfermedades con nombres compuestos, síndromes complejos

#### **Estrategia 2: llama32_balanced**
```python
STRATEGY_2 = {
    "name": "llama32_balanced",
    "model": "llama3.2:3b",
    "chunk_target": 60,       # Tamaño medio balanceado
    "chunk_overlap": 30,      # Overlap moderado
    "chunk_min": 30,          # Mínimo funcional
    "chunk_max": 90,          # Máximo controlado
    "temperature": 0.3,       # Temperatura moderada para balance
    "weight": 1.0
}
```
- **Objetivo**: Balance entre sensibilidad y precisión
- **Casos de uso**: Entidades de longitud media, casos típicos

#### **Estrategia 3: llama32_high_precision**
```python
STRATEGY_3 = {
    "name": "llama32_high_precision",
    "model": "llama3.2:3b",
    "chunk_target": 30,       # Chunks pequeños para precisión
    "chunk_overlap": 15,      # Overlap mínimo
    "chunk_min": 15,          # Mínimo funcional
    "chunk_max": 45,          # Máximo controlado
    "temperature": 0.0,       # Temperatura 0 para máxima determinismo
    "weight": 1.0
}
```
- **Objetivo**: Máxima precisión para entidades cortas y claras
- **Casos de uso**: Nombres de genes, enfermedades simples

#### **Estrategia 4: qwen25_diversity**
```python
STRATEGY_4 = {
    "name": "qwen25_diversity",
    "model": "qwen2.5:3b",
    "chunk_target": 20,       # Chunks muy pequeños
    "chunk_overlap": 10,      # Overlap mínimo
    "chunk_min": 10,          # Mínimo funcional
    "chunk_max": 30,          # Máximo controlado
    "temperature": 0.5,       # Temperatura moderada para diversidad
    "weight": 0.5             # Peso reducido por ser modelo alternativo
}
```
- **Objetivo**: Diversidad de detección usando modelo alternativo
- **Casos de uso**: Entidades que podrían ser pasadas por alto por llama3.2:3b

## 🔄 Sistema de Reintentos Inteligente

### **Fase 1: Reintentos por Formato JSON**
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

### **Fase 2: Reintento por Entidades Vacías**
```python
if not present and retry_reason != "none":
    enhanced_prompt = f"""TEXT: {chunk}
    
EXTRACT disease names. If you find any diseases, return them as: ["disease1", "disease2"]
If you find NO diseases, return: []"""
    
    response = client.generate(model, system_prompt, enhanced_prompt, options)
```

### **Fase 3: Extracción de Texto Plano (Fallback)**
```python
if not present and retry_reason != "none":
    # Patrones regex para enfermedades
    disease_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:disease|syndrome|cancer|tumor)\b',
        r'\b(?:G6PD|BRCA1|BRCA2|ATM|LCAT)\b'
    ]
    
    for pattern in disease_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        # Procesar matches encontrados
```

## 🎯 Sistema de Puntuación de Confianza

### **Cálculo Base de Confianza**
```python
for strategy_name, detected_entities in all_detections.items():
    strategy_weight = 1.0
    if strategy_name != "regex":
        strategy_weight = next(s["weight"] for s in strategies if s["name"] == strategy_name)
    
    for entity in detected_entities:
        base_confidence = strategy_weight
        entity_confidence[entity] += base_confidence
```

### **Reglas de Confianza Avanzadas**

#### **Regla 1: Boost por Detección Regex**
```python
if "regex" in entity_strategies[entity]:
    entity_confidence[entity] = min(1.0, entity_confidence[entity] * 1.5)
```
- **Justificación**: Las entidades detectadas por regex tienen máxima confianza
- **Multiplicador**: 1.5x (capped en 1.0)

#### **Regla 2: Boost por Múltiples Estrategias**
```python
strategy_count = len(entity_strategies[entity])
if strategy_count > 1:
    entity_confidence[entity] = min(1.0, entity_confidence[entity] * (1.0 + 0.2 * (strategy_count - 1)))
```
- **Justificación**: Consistencia entre estrategias aumenta confianza
- **Incremento**: +20% por cada estrategia adicional

#### **Regla 3: Penalización por Solo LLM**
```python
if "regex" not in entity_strategies[entity]:
    entity_confidence[entity] *= 0.8
```
- **Justificación**: Entidades no validadas por regex son menos confiables
- **Penalización**: 20% de reducción

#### **Regla 4: Penalización por Reintentos**
```python
if retry_info.get("final_attempt", 0) > 1:
    retry_penalty = max(0.4, 1.0 - (retry_info["final_attempt"] - 1) * 0.2)
    base_confidence *= retry_penalty
```
- **Escala de penalización**:
  - 1er intento: 100% (sin penalización)
  - 2do intento: 80%
  - 3er intento: 60%
  - 4to intento: 40%

### **Normalización Final**
```python
entity_confidence[entity] = max(0.0, min(1.0, entity_confidence[entity]))
```

## 🔧 Procesamiento de Texto

### **Chunking Inteligente**
```python
def text_to_chunks_file(text: str, strategy: Dict, doc_id: str) -> str:
    words = text.split()
    chunks = []
    
    target_size = strategy["chunk_target"]
    overlap = strategy["chunk_overlap"]
    
    # Validación de seguridad
    if overlap >= target_size:
        overlap = max(1, target_size // 2)
    
    start = 0
    while start < len(words):
        end = min(start + target_size, len(words))
        chunk_words = words[start:end]
        
        if len(chunk_words) >= strategy["chunk_min"]:
            chunk_text = " ".join(chunk_words)
            if len(chunk_text) <= strategy["chunk_max"]:
                chunks.append(chunk_text)
        
        # Avance con overlap
        new_start = end - overlap
        if new_start <= start:  # Prevención de bucles infinitos
            new_start = start + 1
        start = new_start
```

### **Normalización de Texto**
```python
def normalize_surface(text: str) -> str:
    if not text:
        return ""
    
    # Normalización de espacios
    text = re.sub(r'\s+', ' ', text)
    
    # Normalización de comillas
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Normalización de guiones
    text = re.sub(r'–|—', '-', text)
    
    return text.strip()
```

## 🔍 Matching de Entidades

### **Algoritmo de Matching Fuzzy**
```python
def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    # Limpieza de palabras comunes
    common_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    text1_clean = ' '.join([w for w in text1.split() if w.lower() not in common_words])
    text2_clean = ' '.join([w for w in text2.split() if w.lower() not in common_words])
    
    # Matching exacto
    if text1_clean == text2_clean:
        return True
    
    # Matching de contención
    if text1_clean in text2_clean or text2_clean in text1_clean:
        return True
    
    # Similitud de caracteres (Jaccard)
    set1 = set(text1_clean.lower())
    set2 = set(text2_clean.lower())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    return similarity >= threshold
```

### **Jerarquía de Matching**
1. **Matching exacto**: `entity == reference`
2. **Matching case-insensitive**: `entity.lower() == reference.lower()`
3. **Matching de contención**: `entity in reference` o `reference in entity`
4. **Matching fuzzy**: Similitud Jaccard ≥ 0.8

## ⚡ Optimizaciones de Rendimiento

### **Procesamiento Paralelo**
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_strategy = {
        executor.submit(run_strategy, strategy): strategy 
        for strategy in strategies
    }
    
    for future in as_completed(future_to_strategy):
        strategy_name, results_filepath = future.result()
```

### **Gestión de Memoria**
- **Chunking basado en archivos**: Los chunks se guardan en disco temporal
- **Procesamiento incremental**: Un documento a la vez
- **Limpieza automática**: Archivos temporales se eliminan después del uso
- **Garbage collection**: Forzado después de cada documento

### **Cache de LLM**
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
```

## 📊 Evaluación y Métricas

### **Cálculo de Métricas**
```python
# True Positives: Entidades correctamente detectadas
tp = len(matched_entities)

# False Positives: Entidades incorrectamente detectadas
fp = len(predicted_entities) - len(matched_entities)

# False Negatives: Entidades de referencia no detectadas
fn = len(reference_entities) - len(matched_entities)

# Precisión
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

# Recall
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# F1-Score
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
```

### **Análisis por Estrategia**
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
```

## 🎯 Configuración de Ollama

### **Opciones Optimizadas**
```python
options = {
    "temperature": strategy["temperature"],
    "top_p": 0.9,
    "num_predict": 32,        # Respuestas cortas para velocidad
    "num_gpu": 1,             # Forzar uso de GPU
    "num_thread": 2,          # Threads reducidos para estabilidad
    "repeat_penalty": 1.1,    # Reducir repetición
    "top_k": 40,              # Optimizar sampling
    "stop": ["\nUser:", "\nUSER:", "\nAssistant:", "\nASSISTANT:", "```", "```json"]
}
```

### **Prompts Optimizados**
```python
# Para llama3.2:3b
prompt = f"""Diseases in this text: {chunk}

Return ONLY a JSON list like: ["disease1", "disease2"]"""

# Para qwen2.5:3b (evitar razonamiento)
prompt = f"""TEXT: {chunk}

EXTRACT disease names. Return ONLY: ["disease1", "disease2"]"""
```

## 🔬 Validación y Testing

### **Validación de Estrategias**
- **Chunking**: Verificación de que no se creen bucles infinitos
- **Overlap**: Validación de que overlap < target_size
- **Tamaños**: Verificación de límites min/max de chunks

### **Testing de Robustez**
- **Respuestas malformadas**: Manejo de JSON inválido
- **Timeouts**: Manejo de respuestas lentas de Ollama
- **Errores de red**: Reconexión automática en fallos

## 📈 Análisis de Resultados

### **Distribución de Confianza**
```python
confidence_ranges = {
    "high": [0.9, 1.0],      # 3+ estrategias + regex
    "medium": [0.7, 0.9),    # 2+ estrategias
    "low": [0.5, 0.7),       # 1+ estrategia
    "minimal": [0.3, 0.5)    # Solo LLM
}
```

### **Análisis de Errores**
- **False Positives**: Entidades incorrectas detectadas
- **False Negatives**: Entidades reales no detectadas
- **Análisis por tipo**: Patrones en errores para mejora futura

## 🚀 Mejoras Futuras

### **Optimizaciones Planificadas**
1. **Adaptive Chunking**: Tamaño de chunks basado en complejidad del texto
2. **Model Ensemble**: Combinación de más modelos para mayor diversidad
3. **Active Learning**: Feedback del usuario para mejorar detección
4. **Domain Adaptation**: Fine-tuning específico para biomedicina

### **Escalabilidad**
- **Distributed Processing**: Procesamiento en múltiples máquinas
- **Streaming**: Procesamiento en tiempo real
- **API REST**: Interfaz web para uso en producción

---

**Esta metodología representa un enfoque robusto y científicamente fundamentado para NER biomédico, combinando la precisión de reglas con la flexibilidad de LLMs modernos.**
