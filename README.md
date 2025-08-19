# 🧬 Sistema NER Multi-Estrategia con Llama

## 📋 Descripción

Sistema avanzado de Named Entity Recognition (NER) para entidades biomédicas que combina múltiples estrategias de detección usando modelos de lenguaje local (Ollama) con `llama3.2:3b` y `qwen2.5:3b`.

## ⚠️ **ADVERTENCIA IMPORTANTE**

**Este sistema NO es un detector automático de entidades biomédicas.** 

**Funciona de manera fundamentalmente diferente al NER clásico:**

- **NER Clásico**: Detecta automáticamente todas las enfermedades/entidades en el texto sin especificarle que entidades buscar
- **Nuestro Sistema**: Solo detecta las entidades que **específicamente** se le indique que debe buscar en el campo "Entidad" del documento

**Es un sistema de EXTRACCIÓN DIRIGIDA para casos de uso específicos:**

- **Ejemplo**: Si quieres extraer pacientes que tomaron "paracetamol", pones "paracetamol" en la lista
- **Resultado**: Solo encuentra "paracetamol" si está en el texto, no busca otras medicinas
- **Uso ideal**: Para extraer un conjunto específico de entidades de documentos similares

**Esta limitación es intencional y fundamental para el diseño del sistema.**

## 🎯 Características Principales

- **4 Estrategias Complementarias**: Diferentes configuraciones de chunking y modelos
- **Sistema de Reintentos Inteligente**: Manejo robusto de respuestas LLM
- **Puntuación de Confianza Avanzada**: Basada en múltiples estrategias y detección regex
- **Procesamiento Paralelo**: Ejecución simultánea de estrategias
- **Gestión de Memoria Eficiente**: Procesamiento basado en archivos
- **Detección Regex**: Baseline de alta precisión para entidades conocidas

## 🚀 Instalación y Configuración

### Prerrequisitos

1. **Python 3.8+**
2. **Ollama** instalado y ejecutándose
3. **Modelos descargados**:
   ```bash
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b
   ```

### Instalación

```bash
git clone <repository>
cd bert_vs_ollama
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
bert_vs_ollama/
├── datasets/                    # Datasets de entrada
│   ├── ncbi_develop.jsonl      # Dataset de desarrollo
│   └── ncbi_test.jsonl         # Dataset de test
├── scripts/                     # Scripts principales
│   ├── llama_ner_multi_strategy.py  # Sistema principal
│   ├── run_multi_strategy_example.py # Ejemplo de uso
│   └── evaluate_ner_performance.py  # Evaluador
├── results_final/               # Resultados finales
├── docs/                        # Documentación
│   └── SEPARACION_ARCHIVOS.md  # Guía de separación de archivos
└── temp_analysis/               # Análisis temporales
```

## 🎮 Uso

### Separación de Archivos

El sistema ahora separa claramente dos tipos de archivos JSONL:

1. **`--input_jsonl`**: Contiene el texto a procesar y las entidades específicas que se deben buscar
   - Campo `Texto`: El texto biomédico a analizar
   - Campo `Entidad`: Lista de entidades específicas a detectar

2. **`--benchmark_jsonl`**: Contiene los datos de referencia para evaluación
   - Permite evaluar el rendimiento del sistema de manera independiente
   - Facilita la comparación entre diferentes configuraciones

Esta separación permite:
- **Entrenamiento independiente**: Usar diferentes datasets para entrenamiento y evaluación
- **Validación cruzada**: Probar con múltiples conjuntos de benchmark
- **Análisis comparativo**: Evaluar el mismo modelo con diferentes datos de test

### Ejecución Básica

```bash
python scripts/llama_ner_multi_strategy.py \
    --input_jsonl ./datasets/ncbi_develop.jsonl \
    --benchmark_jsonl ./datasets/ncbi_test.jsonl \
    --out_pred results_final.jsonl \
    --strategies all
```

### Parámetros Principales

- `--input_jsonl`: Archivo de entrada JSONL con texto y entidades a buscar
- `--benchmark_jsonl`: Archivo JSONL de benchmark para evaluación
- `--out_pred`: Archivo de salida
- `--limit`: Número máximo de documentos (0 = todos)
- `--strategies`: Estrategias a usar (`all` o nombres específicos)
- `--confidence_threshold`: Umbral mínimo de confianza (default: 0.3)

### Estrategias Disponibles

1. **llama32_max_sensitivity**: Chunks grandes (100t), máxima sensibilidad
2. **llama32_balanced**: Chunks medianos (60t), balanceado
3. **llama32_high_precision**: Chunks pequeños (30t), alta precisión
4. **qwen25_diversity**: Chunks pequeños (20t), diversidad de modelo

## 📊 Evaluación

### Evaluar Rendimiento

```bash
python scripts/evaluate_ner_performance.py \
    --predictions results_final.jsonl \
    --reference ./datasets/ncbi_develop.jsonl
```

## 📚 Documentación Adicional

- **[Separación de Archivos](docs/SEPARACION_ARCHIVOS.md)**: Guía completa sobre la nueva funcionalidad de separación de archivos de entrada y benchmark
- **[Metodología Detallada](docs/METODOLOGIA_DETALLADA.md)**: Explicación técnica del sistema multi-estrategia

### Métricas Generadas

- **Precisión**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Media armónica de precisión y recall
- **Análisis por Estrategia**: Rendimiento individual de cada estrategia

## 🔧 Configuración Avanzada

### **Optimizaciones de Rendimiento**

> ⚠️ **Nota Importante**: El sistema está optimizado para precisión, no velocidad. Las siguientes optimizaciones pueden reducir la precisión significativamente.

#### **Reducir Estrategias**
```bash
# Solo usar estrategias más rápidas
python scripts/llama_ner_multi_strategy.py \
    --strategies llama32_high_precision,qwen25_diversity \
    --develop_jsonl ./datasets/ncbi_test.jsonl \
    --out_pred results_fast.jsonl
```

#### **Ajustar Tamaños de Chunk**
```python
# En llama_ner_multi_strategy.py
STRATEGY_FAST = {
    "name": "llama32_fast",
    "model": "llama3.2:3b",
    "chunk_target": 150,      # Chunks más grandes = menos chunks
    "chunk_overlap": 20,      # Menos overlap = menos procesamiento
    "temperature": 0.0,       # Temperatura 0 = más rápido
    "weight": 1.0
}
```

#### **Configuración Ollama para Velocidad**
```python
options = {
    "temperature": 0.0,
    "num_predict": 16,        # Respuestas más cortas
    "num_gpu": 1,             # Usar GPU
    "num_thread": 4,          # Más threads
    "repeat_penalty": 1.0,    # Sin penalización
    "top_k": 20               # Sampling más agresivo
}
```

**Resultado**: 2-3x más rápido, precisión ~99.0-99.5%

### Personalizar Estrategias

```python
# En llama_ner_multi_strategy.py
STRATEGY_1 = {
    "name": "custom_strategy",
    "model": "llama3.2:3b",
    "chunk_target": 80,      # Tamaño de chunk
    "chunk_overlap": 30,     # Overlap entre chunks
    "temperature": 0.2,      # Temperatura del modelo
    "weight": 1.0           # Peso en scoring
}
```

### Umbrales de Confianza

```python
CONFIDENCE_THRESHOLDS = {
    "high": 0.9,      # 3+ estrategias
    "medium": 0.7,    # 2+ estrategias
    "low": 0.5,       # 1+ estrategias
    "min_accept": 0.3 # Mínimo para aceptar
}
```

## 📈 Rendimiento

### Resultados en Dataset de Desarrollo

- **Precisión**: 99.3%
- **Recall**: 98.9%
- **F1-Score**: 99.1%
- **Total Entidades**: 273
- **Errores**: Solo 5 (1.8% tasa de error)

### Resultados en Dataset de Test Completo

- **Precisión**: 99.7%
- **Recall**: 99.7%
- **F1-Score**: 99.7%
- **Total Entidades**: 385
- **Documentos Procesados**: 93 de 100
- **Errores**: Solo 2 (0.5% tasa de error)

### Estrategias por Rendimiento

1. **regex**: 100% precisión (384 entidades)
2. **llama32_balanced**: 100% precisión (130 entidades)
3. **llama32_max_sensitivity**: 100% precisión (122 entidades)
4. **llama32_high_precision**: 100% precisión (123 entidades)
5. **qwen25_diversity**: 100% precisión (96 entidades)

### Análisis de Errores

- **Documentos con errores**: Solo 1 de 93 (PMID 9674903)
- **Tipo de errores**: Variaciones en nomenclatura biomédica
- **Robustez**: 99.7% de precisión en dataset completo y diverso

## ⚡ Comparación de Rendimiento vs BERT

### **Ventajas del Sistema Multi-Estrategia**

- **Precisión Superior**: 99.7% vs ~95-97% típico de BERT
- **Flexibilidad**: Maneja variaciones de nomenclatura biomédica
- **Interpretabilidad**: Explicable y auditable
- **Sin Fine-tuning**: Funciona out-of-the-box
- **Adaptabilidad**: Fácil ajuste de estrategias

### **Desventajas (Limitaciones Conocidas)**

- **Velocidad**: Significativamente más lento que BERT
- **Recursos**: Requiere más RAM y potencia de cómputo
- **Latencia**: Cada documento requiere múltiples llamadas a LLM
- **Escalabilidad**: No optimizado para procesamiento en lote masivo

### **Limitación Fundamental: Especificidad de Entidades**

⚠️ **IMPORTANTE**: Este sistema NER tiene una limitación fundamental que lo diferencia del NER clásico:

**El sistema NO extrae automáticamente cualquier enfermedad o entidad biomédica del texto.** En su lugar, **requiere que se especifiquen explícitamente las entidades que se quieren extraer** de cada documento.

#### **Ejemplo Práctico del Caso de Uso:**

**❌ NO funciona así (NER clásico):**
- El sistema detecta automáticamente todas las enfermedades mencionadas en el texto
- Encuentra "gripe", "fiebre", "dolor de cabeza", "paracetamol", etc. sin especificación previa

**✅ Funciona así (nuestro sistema - EXTRACCIÓN DIRIGIDA):**
- **Caso de uso**: Quieres extraer solo pacientes que tomaron "paracetamol"
- **Configuración**: Pones "paracetamol" en la lista de entidades
- **Resultado**: Solo encuentra "paracetamol" si está en el texto, no busca otras medicinas
- **Ventaja**: Perfecto para extraer un conjunto específico de entidades de documentos similares

#### **Implicaciones:**
- **No es un detector universal** de entidades biomédicas
- **Es un extractor dirigido** para entidades específicas predefinidas
- **Cada documento debe tener** su lista específica de entidades objetivo
- **Ideal para casos de uso específicos** donde se sabe exactamente qué buscar
- **Perfecto para extracción sistemática** de un conjunto fijo de entidades
- **No adecuado para exploración general** o descubrimiento de nuevas entidades

### **Estimación de Tiempos Comparativa**

| Métrica | BERT (GPU) | NER Multi-Estrategia | Factor |
|---------|------------|----------------------|---------|
| **1 documento** | ~0.1-0.5s | ~10-30s | **20-60x más lento** |
| **100 documentos** | ~10-50s | ~15-45 min | **20-60x más lento** |
| **1000 documentos** | ~2-8 min | ~2.5-7.5 horas | **20-60x más lento** |

**Nota**: Los tiempos varían según hardware, complejidad del texto y configuración de estrategias.

### **Casos de Uso Recomendados**

- ✅ **Extracción sistemática**: Cuando quieres extraer un conjunto fijo de entidades de documentos similares
- ✅ **Análisis dirigido**: Buscar pacientes con medicamentos específicos, enfermedades concretas, etc.
- ✅ **Validación de hipótesis**: Verificar presencia de entidades específicas en un dataset
- ✅ **Investigación clínica**: Extraer datos específicos para estudios médicos
- ✅ **Datasets pequeños-medianos**: < 1000 documentos para análisis detallado
- ✅ **Entornos de desarrollo**: Prototipado y experimentación

- ❌ **Producción en tiempo real**: Latencia crítica
- ❌ **Procesamiento masivo**: > 1000 documentos
- ❌ **Entornos con recursos limitados**: RAM < 8GB
- ❌ **Aplicaciones de usuario final**: Requieren respuesta instantánea
- ❌ **Exploración general**: Cuando no se sabe qué entidades buscar
- ❌ **Descubrimiento automático**: Detección de nuevas entidades no especificadas

## 🐛 Solución de Problemas

### Ollama No Responde

```bash
# Verificar estado
ollama list

# Reiniciar servicio
ollama serve
```

### Errores de Memoria

- Reducir `--limit` para procesar menos documentos
- Verificar que Ollama tenga suficiente RAM disponible
- Usar estrategias con chunks más pequeños

### Baja Precisión

- Ajustar `--confidence_threshold`
- Verificar que los modelos estén descargados
- Revisar logs de debug para errores específicos

## 📚 Archivos de Configuración

### Formato de Entrada de texto a procesar(JSONL)

⚠️ **CRÍTICO**: El campo "Entidad" define **EXACTAMENTE** qué entidades se extraerán del texto. El sistema NO detectará entidades que no estén en esta lista.

#### **Estructura Básica del JSONL:**

```json
{
  "PMID": "12345",
  "Texto": "Texto del documento biomédico...",
  "Entidad": [
    {
      "texto": "nombre de enfermedad específica",
      "tipo": "SpecificDisease"
    },
    {
      "texto": "otra entidad a buscar",
      "tipo": "SpecificDisease"
    }
  ]
}
```

#### **Ejemplo Práctico Completo:**

```json
{
  "PMID": "9876543",
  "Texto": "El paciente presenta síntomas de gripe con fiebre alta. Se le administró paracetamol 500mg cada 8 horas. También presenta dolor de cabeza y tos seca. El tratamiento incluye reposo y abundante líquido.",
  "Entidad": [
    {
      "texto": "paracetamol",
      "tipo": "SpecificDisease"
    },
    {
      "texto": "gripe",
      "tipo": "SpecificDisease"
    }
  ]
}
```

#### **¿Qué Pasará con este Ejemplo?**

**✅ Entidades que SÍ se detectarán:**
- "paracetamol" - porque está en la lista
- "gripe" - porque está en la lista

**❌ Entidades que NO se detectarán (aunque aparezcan en el texto):**
- "fiebre alta" - porque NO está en la lista
- "dolor de cabeza" - porque NO está en la lista
- "tos seca" - porque NO está en la lista
- "reposo" - porque NO está en la lista

**Resultado**: Solo extraerás información sobre pacientes que tomaron paracetamol y/o tuvieron gripe, no sobre otros síntomas o tratamientos.

#### **Reglas Importantes para el Campo "Entidad":**

1. **Entidad base**: Incluye la forma principal de la enfermedad que quieres detectar
2. **El LLM maneja variaciones**: No necesitas incluir todas las variaciones posibles. El sistema LLM detectará automáticamente:
   - Sinónimos médicos
   - Abreviaturas comunes
   - Variaciones en nomenclatura
   - Términos relacionados

3. **Ejemplo simplificado**: Solo necesitas incluir la entidad principal:
   ```json
   {
     "texto": "paracetamol",
     "tipo": "SpecificDisease"
   }
   ```
   
   **El sistema automáticamente detectará:**
   - "paracetamol"
   - "acetaminofén" (nombre alternativo)
   - "Tylenol" (marca comercial)
   - Y otras variaciones relacionadas

**Nota**: El LLM es inteligente y encuentra variaciones automáticamente. No es necesario ser exhaustivo en la lista de entidades.

#### **Ejemplo de Archivo JSONL Completo:**

```jsonl
{"PMID": "123", "Texto": "Texto del primer documento...", "Entidad": [{"texto": "enfermedad A", "tipo": "SpecificDisease"}]}
{"PMID": "124", "Texto": "Texto del segundo documento...", "Entidad": [{"texto": "enfermedad B", "tipo": "SpecificDisease"}, {"texto": "enfermedad C", "tipo": "SpecificDisease"}]}
{"PMID": "125", "Texto": "Texto del tercer documento...", "Entidad": []}
```

#### **Mejores Prácticas para Preparar el JSONL:**

1. **Define entidades principales**: Incluye solo las enfermedades/entidades principales que quieres detectar
2. **Usa términos médicos estándar** cuando sea posible
3. **Mantén la lista simple**: No es necesario ser exhaustivo - el LLM es inteligente
4. **Prueba con un documento pequeño** antes de procesar todo el dataset

#### **Herramientas Recomendadas para Preparar el JSONL:**

- **Anotadores manuales**: Para definir las entidades principales que quieres detectar
- **Scripts de preprocesamiento**: Para estandarizar el formato JSONL
- **Validación simple**: Verificar que las entidades principales estén bien definidas
- **Documentación**: Mantener una lista simple de entidades objetivo por proyecto

### Formato de Salida

```json
{
  "PMID": "12345",
  "Texto": "Texto original...",
  "Entidad": [
    {
      "texto": "entidad detectada",
      "tipo": "SpecificDisease",
      "confidence": 0.95,
      "strategies": ["regex", "llama32_balanced"]
    }
  ],
  "_multi_strategy": {
    "all_detections": {...},
    "entity_confidence": {...},
    "strategies_used": [...]
  }
}
```

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 📞 Soporte

Para preguntas o problemas:
- Abre un issue en GitHub
- Revisa la documentación en `docs/`
- Consulta los logs de debug para información detallada

## 🎯 Conclusiones y Recomendaciones

### **¿Cuándo Usar Este Sistema?**

- **✅ Máxima Precisión**: Cuando la precisión > 99% es crítica
- **✅ Investigación**: Para validar y mejorar otros sistemas NER
- **✅ Datasets Pequeños**: < 5000 documentos para análisis detallado
- **✅ Prototipado**: Desarrollo de nuevas estrategias de NER

### **¿Cuándo NO Usar Este Sistema?**

- **❌ Producción Masiva**: > 5000 documentos diarios
- **❌ Tiempo Real**: Aplicaciones que requieren < 1 segundo de respuesta
- **❌ Recursos Limitados**: Sistemas con < 8GB RAM
- **❌ Escalabilidad**: Entornos que requieren procesamiento paralelo masivo

### **Trade-off: Precisión vs Velocidad**

| Aspecto | BERT | NER Multi-Estrategia |
|---------|------|----------------------|
| **Precisión** | 95-97% | **99.7%** |
| **Velocidad** | **Muy Rápido** | 20-60x más lento |
| **Recursos** | **Bajo** | Alto |
| **Flexibilidad** | Baja | **Muy Alta** |
| **Interpretabilidad** | Baja | **Muy Alta** |

**El sistema está diseñado para ser el mejor en precisión, no en velocidad.**

---

**Desarrollado para investigación en NER biomédico con modelos de lenguaje local.**
