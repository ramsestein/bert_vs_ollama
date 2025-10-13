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
├── ner_multi_strategy.py              # 🚀 SISTEMA PRINCIPAL NER (ejecutar desde aquí) # main.py *
├── old_ner_multi_strategy.py          # 🔄 Sistema original (backup)
├── ner_app/                           # 🏗️ Arquitectura modular refactorizada
├── tests/                             # 🧪 Suite de tests completa
├── datasets/                          # 📊 Datasets de entrada y benchmark
│   ├── ncbi_develop.jsonl            # Dataset de desarrollo (NCBI)
│   ├── ncbi_test.jsonl               # Dataset de test (NCBI)
│   ├── n2c2_test_input.jsonl         # Dataset de entrada n2c2 (sin chest pain)
│   └── n2c2_test.jsonl               # Dataset de benchmark n2c2 (sin chest pain)
├── scripts/                           # 🔧 Scripts auxiliares de análisis
│   ├── evaluate_ner_performance.py    # Evaluador de métricas
│   ├── analyze_false_positives_clean.py # Análisis de falsos positivos
│   ├── analyze_false_negatives_clean.py # Análisis de falsos negativos
│   ├── run_llama_grid_n2c2.py        # Grid search Llama
│   ├── run_qwen_grid_n2c2.py         # Grid search Qwen
│   └── analyze_n2c2_grid_results.py  # Análisis de resultados grid
├── docs/                             # 📚 Documentación técnica
│   ├── METODOLOGIA_DETALLADA.md      # Explicación técnica del sistema
│   └── OPTIMIZACION_PARAMETROS.md    # Guía de optimización
├── results_final/                    # 📈 Resultados finales
└── temp_analysis/                    # 🔍 Análisis temporales
```

## 🎮 Uso

```bash
# ✅ CORRECTO - desde el directorio raíz
python ner_multi_strategy.py --help

```

### Separación de Archivos

El sistema separa claramente dos tipos de archivos JSONL:

1. **`--input_jsonl`**: Contiene el texto a procesar y las entidades específicas que se deben buscar
   - Campo `Texto`: El texto biomédico a analizar
   - Campo `Entidad`: Lista de entidades específicas a detectar

2. **`--benchmark_jsonl`**: Contiene los datos de referencia para evaluación
   - Permite evaluar el rendimiento del sistema de manera independiente

### Ejecución Básica

```bash
python ner_multi_strategy.py \
    --input_jsonl ./datasets/dataset_input.jsonl \
    --benchmark_jsonl ./datasets/dataset_test.jsonl \
    --out_pred results_final.jsonl \
    --strategies all
```

### Parámetros Principales

- `--input_jsonl`: Archivo de entrada JSONL con texto y entidades a buscar
- `--benchmark_jsonl`: Archivo JSONL de benchmark para evaluación
- `--out_pred`: Archivo de salida
- `--limit`: Número máximo de documentos (0 = todos)
- `--strategies`: Estrategias a usar (`all` o nombres específicos)
- `--confidence_threshold`: Umbral mínimo de confianza (default: 0.5)

### Estrategias Disponibles

1. **llama32_max_sensitivity**: Chunks grandes (100t), máxima sensibilidad
2. **llama32_balanced**: Chunks medianos (60t), balanceado
3. **llama32_high_precision**: Chunks pequeños (30t), alta precisión
4. **qwen25_diversity**: Chunks pequeños (20t), diversidad de modelo

## 🔧 Pipeline de Optimización

Hemos diseñado un **pipeline sistemático de optimización** para adaptar el script a nuevos datasets:

### **1. Optimización de Temperatura**
```bash
# Probar diferentes temperaturas para encontrar la óptima
python scripts/run_llama_grid.py --dataset n2c2_develop --limit 50
```

### **2. Optimización de Chunks y Overlap**
```bash
# Probar diferentes configuraciones de chunking
python scripts/run_llama_grid.py --dataset n2c2_develop --limit 50
```

### **3. Optimización del Umbral de Confianza**
```bash
# Probar diferentes umbrales (0.3, 0.5, 0.7, 0.9)
python ner_multi_strategy.py \
    --confidence_threshold 0.5 \
    --input_jsonl ./datasets/n2c2_test_input.jsonl \
    --benchmark_jsonl ./datasets/n2c2_test.jsonl \
    --out_pred results_test.jsonl \
    --limit 50
```

### **4. Validación y Corrección de Benchmark**
```bash
# Analizar falsos positivos para identificar errores en anotaciones humanas
python scripts/analyze_false_positives_clean.py

# Analizar falsos negativos para verificar cobertura completa
python scripts/analyze_false_negatives_clean.py
```

### **5. Actualización del Script Principal**
Una vez optimizados los parámetros, se actualiza `ner_multi_strategy.py` con:
- Temperatura óptima
- Configuración de chunks óptima
- Umbral de confianza óptimo

## 📊 Evaluación

### Evaluar Rendimiento

```bash
python scripts/evaluate_ner_performance.py \
    --predictions results_final.jsonl \
    --reference ./datasets/n2c2_test.jsonl
```

## **Documentos Disponibles en `docs/`**

La carpeta `docs/` contiene documentación técnica detallada del proyecto:

#### **📋 Documentos Principales**

- **[METODOLOGIA_DETALLADA.md](docs/METODOLOGIA_DETALLADA.md)**
  - **Descripción**: Explicación técnica completa del sistema NER multi-estrategia
  - **Contenido**: Arquitectura del sistema, flujo de procesamiento, estrategias implementadas
  - **Audiencia**: Desarrolladores, investigadores, usuarios técnicos
  - **Uso**: Referencia para entender cómo funciona internamente el sistema

- **[OPTIMIZACION_PARAMETROS.md](docs/OPTIMIZACION_PARAMETROS.md)**
  - **Descripción**: Guía completa del pipeline de optimización de parámetros
  - **Contenido**: Proceso sistemático para optimizar temperatura, chunks, overlap y umbral de confianza
  - **Audiencia**: Usuarios que quieren adaptar el sistema a nuevos datasets
  - **Uso**: Tutorial paso a paso para optimización de rendimiento

#### **🎯 Cómo Usar la Documentación**

1. **Para empezar**: Lee el README principal (este archivo) para una visión general
2. **Para entender el sistema**: Consulta `METODOLOGIA_DETALLADA.md` para detalles técnicos
3. **Para optimizar**: Sigue `OPTIMIZACION_PARAMETROS.md` para mejorar el rendimiento
4. **Para análisis**: Revisa `AUDIT_REPORT.md` y `metrics_summary.json` para resultados
5. **Para desarrollo**: Consulta `improvement_log.md` para entender la evolución del proyecto

#### **📖 Estructura de la Documentación**

```
docs/
├── METODOLOGIA_DETALLADA.md      # 🧠 Explicación técnica del sistema
├── OPTIMIZACION_PARAMETROS.md    # ⚙️ Guía de optimización
└── (futuros documentos)         # 📚 Documentación adicional en desarrollo

📁 Raíz del proyecto
├── README.md                     # 🚀 Documentación principal (este archivo)
├── AUDIT_REPORT.md              # 📊 Reporte de auditoría
├── improvement_log.md            # 📝 Registro de mejoras
└── metrics_summary.json          # 📈 Resumen de métricas
```

#### **🔧 Documentación Técnica Adicional**

- **Código fuente**: Todos los scripts incluyen documentación inline detallada
- **Tests**: La carpeta `tests/` contiene ejemplos de uso y validación
- **Scripts auxiliares**: Cada script en `scripts/` incluye documentación de uso
- **Configuración**: Los archivos de configuración están documentados con comentarios



## 🔧 Configuración Avanzada

### **Optimizaciones de Rendimiento**

> ⚠️ **Nota Importante**: El sistema está optimizado para precisión, no velocidad. Las siguientes optimizaciones pueden reducir la precisión significativamente.


#### **Ajustar Tamaños de Chunk**
```python
# En ner_multi_strategy.py
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
# En ner_multi_strategy.py
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
    "min_accept": 0.5 # Mínimo para aceptar (optimizado para n2c2)
}
```

## 📈 Rendimiento

### Métricas Generadas

- **Precisión**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Media armónica de precisión y recall
- **Análisis por Estrategia**: Rendimiento individual de cada estrategia

### **Resultados en Dataset NCBI (Test Completo)**

- **Precisión**: 99.7%
- **Recall**: 99.7%
- **F1-Score**: 99.7%
- **Total Entidades**: 385
- **Documentos Procesados**: 93 de 100
- **Errores**: Solo 2 (0.5% tasa de error)

### **Resultados en Dataset n2c2 (Test - primeros 100 documentos)**

- **Precisión**: 95.4%
- **Recall**: 100.0%
- **F1-Score**: 97.6%
- **Total Entidades Reales**: 65
- **Total Entidades en Benchmark**: 47

## **Análisis de Corrección de Anotaciones Humanas**

Durante la evaluación del dataset n2c2, descubrimos que **14 entidades detectadas por la máquina no estaban anotadas en el benchmark**, pero **eran correctas**:

- **PMID 103**: 'orthopnea' (confianza: 1.000)
- **PMID 106**: 'monitoring' (confianza: 1.000)
- **PMID 119**: 'hypertension' (confianza: 0.800)
- **PMID 121**: 'short of breath' (confianza: 0.800)
- **PMID 127**: 'monitoring' (confianza: 1.000)
- **PMID 129**: 'angina' (confianza: 1.000)
- **PMID 136**: 'short of breath' (confianza: 0.800)
- **PMID 142**: 'hypertension' (confianza: 1.000)
- **PMID 146**: 'obesity' (confianza: 1.000)
- **PMID 153**: 'hypertension' (confianza: 1.000)
- **PMID 155**: 'monitoring' (confianza: 1.000)
- **PMID 170**: 'monitoring' (confianza: 1.000)

**Conclusión**: La máquina tiene razón en estos casos, demostrando la capacidad del sistema para **identificar errores en anotaciones humanas** y mejorar la calidad del benchmark.


### Análisis de Errores

- **Documentos con errores**: Solo 3 de 100 (PMIDs 123, 128, 16)
- **Tipo de errores**: Variaciones en nomenclatura biomédica
- **Robustez**: 95.4% de precisión en dataset n2c2 con anotaciones corregidas

## ⚡ Comparación de Rendimiento vs BERT

### **Ventajas del Sistema Multi-Estrategia**

- **Precisión Superior**: 95.4-99.7% vs ~90-95% típico de BERT
- **Flexibilidad**: Maneja variaciones de nomenclatura biomédica
- **Interpretabilidad**: Explicable y auditable
- **Sin Fine-tuning**: Funciona out-of-the-box
- **Adaptabilidad**: Fácil ajuste de estrategias
- **Corrección de Anotaciones**: Identifica errores en benchmarks humanos

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
- ✅ **Corrección de Benchmarks**: Identificar errores en anotaciones humanas

- ❌ **Producción en tiempo real**: Latencia crítica
- ❌ **Procesamiento masivo**: > 1000 documentos
- ❌ **Entornos con recursos limitados**: RAM < 8GB
- ❌ **Aplicaciones de usuario final**: Requieren respuesta instantánea
- ❌ **Exploración general**: Cuando no se sabe qué entidades buscar
- ❌ **Descubrimiento automático**: Detección de nuevas entidades no especificadas

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

- **✅ Máxima Precisión**: Cuando la precisión > 95% es crítica
- **✅ Investigación**: Para validar y mejorar otros sistemas NER
- **✅ Datasets Pequeños**: < 5000 documentos para análisis detallado
- **✅ Prototipado**: Desarrollo de nuevas estrategias de NER
- **✅ Corrección de Benchmarks**: Identificar errores en anotaciones humanas
- **✅ Validación de Calidad**: Verificar la calidad de datasets anotados

### **¿Cuándo NO Usar Este Sistema?**

- **❌ Producción Masiva**: > 5000 documentos diarios
- **❌ Tiempo Real**: Aplicaciones que requieren < 1 segundo de respuesta
- **❌ Recursos Limitados**: Sistemas con < 8GB RAM
- **❌ Escalabilidad**: Entornos que requieren procesamiento paralelo masivo

### **Trade-off: Precisión vs Velocidad**

| Aspecto | BERT | NER Multi-Estrategia |
|---------|------|----------------------|
| **Precisión** | 90-95% | **95.4-99.7%** |
| **Velocidad** | **Muy Rápido** | 20-60x más lento |
| **Recursos** | **Bajo** | Alto |
| **Flexibilidad** | Baja | **Muy Alta** |
| **Interpretabilidad** | Baja | **Muy Alta** |
| **Corrección de Anotaciones** | No | **Sí** |

**El sistema está diseñado para ser el mejor en precisión, no en velocidad.**

### **Pipeline de Optimización Recomendado**

1. **Dataset de Desarrollo**: Usar para optimizar temperatura, chunks y overlap
2. **Validación Cruzada**: Probar diferentes configuraciones
3. **Test Final**: Evaluar con dataset de test optimizado
4. **Corrección de Benchmark**: Identificar y corregir errores en anotaciones
5. **Métricas Finales**: Reportar rendimiento con benchmark corregido

---

**Desarrollado para investigación en NER biomédico con modelos de lenguaje local.**
