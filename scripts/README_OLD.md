# Scripts NER - Documentación

Sistema de scripts para grid search, evaluación y análisis de modelos NER con LLMs.

---

## 📁 Estructura

```
scripts/
├── ner_analysis/                    # Módulo compartido de análisis
│   ├── __init__.py                  # Exporta funciones públicas
│   ├── file_parser.py               # Parseo y lectura de archivos
│   ├── evaluator.py                 # Wrapper de evaluate_ner_performance.py
│   ├── aggregator.py                # Funciones de agregación
│   └── README.md                    # Documentación del módulo
│
├── run_grid_search.py               # Grid search general (chunks + overlaps + temp)
├── run_temperature_grid.py          # Grid search temperaturas + confidence
│
├── evaluate_ner_performance.py      # Calcula métricas P/R/F1 (1 archivo)
│
├── analyze_grid_results.py          # Análisis universal de grid search
├── analyze_temperature_grid.py      # Análisis enfocado en temperaturas
│
├── analyze_false_negatives_clean.py # Análisis detallado de FN
└── analyze_false_positives_clean.py # Análisis detallado de FP
```

---

## 🚀 EJECUCIÓN (Grid Search)

### `run_grid_search.py`
**Propósito**: Grid search completo para optimizar chunks, overlaps y temperaturas.

**Uso**:
```bash
# Grid search completo con defaults
python scripts/run_grid_search.py --model qwen --dataset n2c2 --limit 50

# Personalizar parámetros
python scripts/run_grid_search.py \
  --model qwen \
  --dataset n2c2 \
  --limit 100 \
  --custom-chunks 40,60,80 \
  --custom-overlaps 10,20,30 \
  --custom-temps 0.0,0.3,0.5,0.7

# Con log personalizado
python scripts/run_grid_search.py --model gemma --dataset ncbi --log-file gemma_ncbi.log
```

**Salida**:
- Directorio: `{model}_grid_{dataset}_{timestamp}/`
- Archivos: `results_{model}_{dataset}_chunk{n}_ov{n}_temp{n}.jsonl`
- Log: `grid_search_{model}_{dataset}_{timestamp}.log`

---

### `run_temperature_grid.py`
**Propósito**: Grid search especializado para temperaturas y confidence threshold (chunks fijos).

**Uso**:
```bash
# Temperaturas y confidence (chunks fijos)
python scripts/run_temperature_grid.py --model qwen --dataset n2c2 --limit 50

# Personalizar chunks y parámetros
python scripts/run_temperature_grid.py \
  --model qwen \
  --dataset n2c2 \
  --chunk 60 \
  --overlap 20 \
  --custom-temps 0.0,0.1,0.2,0.3,0.4,0.5 \
  --custom-confidence 0.1,0.2,0.3,0.4,0.5
```

**Salida**:
- Directorio: `{model}_temp_grid_{dataset}_{timestamp}/`
- Archivos: `results_{model}_{dataset}_chunk{n}_ov{n}_temp{n}_conf{n}.jsonl`

---

## 📊 EVALUACIÓN

### `evaluate_ner_performance.py`
**Propósito**: Calcula métricas básicas (Precision, Recall, F1) para un solo archivo.

**Uso**:
```bash
# Evaluar predicciones vs referencia
python scripts/evaluate_ner_performance.py \
  --predictions results_qwen_n2c2_chunk60_ov20_temp0.3.jsonl \
  --reference datasets/n2c2_test.jsonl
```

**Salida** (consola):
```
Precisión: 0.920
Recall: 0.880
F1-Score: 0.900
True Positives (TP): 42
False Positives (FP): 3
False Negatives (FN): 5
```

---

## 🔍 ANÁLISIS COMPARATIVO

### `analyze_grid_results.py`
**Propósito**: Análisis universal de múltiples resultados de grid search.

**Características**:
- ✅ Funciona con cualquier modelo, dataset y combinación de parámetros
- ✅ Detecta automáticamente temperatura y confidence
- ✅ Genera rankings por F1-Score y entidades detectadas
- ✅ Exporta CSV con todos los resultados

**Uso**:
```bash
# Sin métricas (solo análisis de entidades)
python scripts/analyze_grid_results.py qwen_grid_n2c2_20251203_102158

# Con métricas P/R/F1
python scripts/analyze_grid_results.py \
  qwen_grid_n2c2_20251203_102158 \
  --reference datasets/n2c2_test.jsonl

# Top 20 configuraciones
python scripts/analyze_grid_results.py \
  qwen_grid_n2c2_20251203_102158 \
  --reference datasets/n2c2_test.jsonl \
  --top 20
```

**Salida**:
- Análisis por chunk, overlap, temperature, confidence
- Top N configuraciones por F1-Score
- Top N configuraciones por entidades detectadas
- CSV: `grid_search_analysis.csv`

---

### `analyze_temperature_grid.py`
**Propósito**: Análisis enfocado en temperaturas (compatible con run_grid_search.py y run_temperature_grid.py).

**Uso**:
```bash
python scripts/analyze_temperature_grid.py \
  qwen_temp_grid_n2c2_20251203_120000 \
  --reference datasets/n2c2_test.jsonl \
  --top 10
```

**Salida**:
- Análisis detallado por temperatura
- Análisis por chunk y overlap
- CSV: `temperature_grid_analysis.csv`

---

## 🐛 ANÁLISIS DETALLADO DE ERRORES

### `analyze_false_negatives_clean.py`
**Propósito**: Identifica qué entidades se perdieron (FN).

**Uso**:
```bash
python scripts/analyze_false_negatives_clean.py \
  --predictions results_qwen_n2c2_chunk60_ov20_temp0.3.jsonl \
  --benchmark datasets/n2c2_test.jsonl \
  --output fn_analysis.json
```

**Salida**:
- Lista de entidades no detectadas
- Análisis por PMID
- JSON con casos detallados

---

### `analyze_false_positives_clean.py`
**Propósito**: Identifica qué entidades se detectaron incorrectamente (FP).

**Uso**:
```bash
python scripts/analyze_false_positives_clean.py \
  --predictions results_qwen_n2c2_chunk60_ov20_temp0.3.jsonl \
  --benchmark datasets/n2c2_test.jsonl \
  --output fp_analysis.json
```

**Salida**:
- Lista de entidades incorrectas
- Análisis por estrategia
- JSON con casos detallados

---

## 🔄 Workflow típico

### 1. Ejecutar grid search
```bash
# Optimizar chunks y overlaps
python scripts/run_grid_search.py --model qwen --dataset n2c2 --limit 50

# Output: qwen_grid_n2c2_20251203_150000/
```

### 2. Analizar resultados
```bash
# Análisis comparativo con métricas
python scripts/analyze_grid_results.py \
  qwen_grid_n2c2_20251203_150000 \
  --reference datasets/n2c2_test.jsonl

# Identifica mejor configuración: chunk=60, overlap=20, temp=0.3, F1=0.92
```

### 3. Optimizar temperatura y confidence
```bash
# Grid search fino con parámetros óptimos
python scripts/run_temperature_grid.py \
  --model qwen \
  --dataset n2c2 \
  --chunk 60 \
  --overlap 20 \
  --limit 100

# Output: qwen_temp_grid_n2c2_20251203_160000/
```

### 4. Analizar errores
```bash
# Analizar falsos negativos
python scripts/analyze_false_negatives_clean.py \
  --predictions qwen_temp_grid_n2c2_20251203_160000/results_qwen_n2c2_chunk60_ov20_temp0.3_conf0.3.jsonl \
  --benchmark datasets/n2c2_test.jsonl

# Analizar falsos positivos
python scripts/analyze_false_positives_clean.py \
  --predictions qwen_temp_grid_n2c2_20251203_160000/results_qwen_n2c2_chunk60_ov20_temp0.3_conf0.3.jsonl \
  --benchmark datasets/n2c2_test.jsonl
```

---

## 📦 Módulo ner_analysis

Módulo compartido para evitar duplicación de código. Ver `ner_analysis/README.md` para detalles.

**Funciones principales**:
- `parse_filename()`: Extrae parámetros de nombres de archivo
- `read_results()`: Lee y analiza archivos JSONL
- `evaluate_performance()`: Calcula métricas P/R/F1
- `aggregate_by_parameter()`: Agrupa resultados por parámetro
- `get_top_configurations()`: Obtiene top N configuraciones

**Uso**:
```python
from ner_analysis import (
    parse_filename,
    read_results,
    evaluate_performance
)
```

---

## 🎯 Resumen rápido

| Script | Propósito | Input | Output |
|--------|-----------|-------|--------|
| `run_grid_search.py` | Grid search completo | Modelo, dataset | Directorio con N configuraciones |
| `run_temperature_grid.py` | Grid search temp/conf | Modelo, dataset, chunks | Directorio con N configuraciones |
| `evaluate_ner_performance.py` | Métricas 1 archivo | Predicciones, referencia | P/R/F1 en consola |
| `analyze_grid_results.py` | Análisis comparativo | Directorio resultados | Rankings + CSV |
| `analyze_temperature_grid.py` | Análisis temperaturas | Directorio resultados | Rankings + CSV |
| `analyze_false_negatives_clean.py` | Análisis FN | Predicciones, benchmark | JSON con FN |
| `analyze_false_positives_clean.py` | Análisis FP | Predicciones, benchmark | JSON con FP |

---

## ⚠️ Notas

- Todos los scripts usan `[INFO]`, `[ERROR]`, `[WARN]` en lugar de emojis (compatible con Windows cp1252)
- Los scripts de análisis requieren que `evaluate_ner_performance.py` esté en `scripts/`
- CSV requiere pandas instalado (opcional, se muestra warning si no está)
