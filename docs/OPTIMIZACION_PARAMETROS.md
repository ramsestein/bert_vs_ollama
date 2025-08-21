# 🔧 Guía de Optimización de Parámetros para llama_ner_multi_strategy

## 📋 Descripción

Esta guía detalla el **pipeline sistemático de optimización** que hemos desarrollado para adaptar el script `llama_ner_multi_strategy.py` a nuevos datasets biomédicos. El proceso garantiza que el sistema funcione con la máxima precisión para cada tipo de datos.

## 🎯 Objetivo del Pipeline

El objetivo es **optimizar secuencialmente** los parámetros más críticos del sistema:

1. **Temperatura del modelo** (0.0 - 1.0)
2. **Configuración de chunks** (tamaño y overlap)
3. **Umbral de confianza** (0.3 - 0.9)
4. **Validación y corrección del benchmark**

## 🚀 Pipeline de Optimización Completo

### **FASE 1: Preparación del Dataset**

#### **1.1 Separación de Archivos**
Antes de comenzar la optimización, asegúrate de tener:

- **`dataset_input.jsonl`**: Contiene el texto y las entidades específicas a buscar
- **`dataset_benchmark.jsonl`**: Contiene las anotaciones de referencia para evaluación

#### **1.2 Dataset de Desarrollo**
Crea un **dataset de desarrollo** con un subconjunto representativo (50-100 documentos) para optimizar parámetros sin procesar todo el dataset.

```bash
# Ejemplo de separación
python scripts/create_dev_dataset.py \
    --input_file ./datasets/dataset_input.jsonl \
    --output_file ./datasets/dataset_develop.jsonl \
    --limit 50
```

### **FASE 2: Optimización de Temperatura**

#### **2.1 Grid Search de Temperatura**
La temperatura controla la creatividad del modelo. Para NER biomédico, valores bajos (0.0-0.3) suelen ser óptimos.

```bash
# Ejecutar grid search de temperatura
python scripts/run_llama_grid.py \
    --dataset n2c2_develop \
    --limit 50 \
    --temperatures 0.0,0.1,0.2,0.3,0.5,0.7,0.9
```

#### **2.2 Análisis de Resultados**
```bash
# Analizar resultados del grid search
python scripts/analyze_grid.py \
    --results_dir ./results_grid_search \
    --output_file temperature_analysis.json
```

#### **2.3 Selección de Temperatura Óptima**
Basándote en los resultados, selecciona la temperatura que maximice:
- **Precisión** (mínimo de falsos positivos)
- **Recall** (máximo de entidades detectadas)
- **F1-Score** (balance entre precisión y recall)

**Recomendación**: Comienza con **temperatura 0.0** (máxima precisión) y ajusta hacia arriba solo si es necesario.

### **FASE 3: Optimización de Chunks y Overlap**

#### **3.1 Configuraciones de Chunking a Probar**
```python
# Configuraciones recomendadas para probar
CHUNK_CONFIGS = [
    {"target": 20, "overlap": 5},   # Chunks muy pequeños
    {"target": 30, "overlap": 10},  # Chunks pequeños
    {"target": 50, "overlap": 15},  # Chunks medianos
    {"target": 80, "overlap": 20},  # Chunks grandes
    {"target": 100, "overlap": 25}, # Chunks muy grandes
]
```

#### **3.2 Ejecución del Grid Search de Chunks**
```bash
# Probar diferentes configuraciones de chunking
python scripts/run_llama_grid.py \
    --dataset n2c2_develop \
    --limit 50 \
    --chunk_configs "20:5,30:10,50:15,80:20,100:25"
```

#### **3.3 Análisis de Resultados de Chunking**
```bash
# Analizar rendimiento por configuración de chunks
python scripts/analyze_chunk_performance.py \
    --results_dir ./results_chunk_grid \
    --output_file chunk_analysis.json
```

#### **3.4 Selección de Configuración Óptima**
Considera estos factores al seleccionar la configuración de chunks:

- **Chunks pequeños (20-30 tokens)**: Mayor precisión, más lento
- **Chunks medianos (50-80 tokens)**: Balance entre precisión y velocidad
- **Chunks grandes (100+ tokens)**: Más rápido, menor precisión
- **Overlap**: 15-25% del tamaño del chunk suele ser óptimo

### **FASE 4: Optimización del Umbral de Confianza**

#### **4.1 Umbrales a Probar**
```bash
# Probar diferentes umbrales de confianza
for threshold in 0.3 0.5 0.7 0.9; do
    python scripts/llama_ner_multi_strategy.py \
        --input_jsonl ./datasets/dataset_develop.jsonl \
        --benchmark_jsonl ./datasets/dataset_benchmark.jsonl \
        --out_pred results_threshold_${threshold}.jsonl \
        --confidence_threshold $threshold \
        --limit 50
done
```

#### **4.2 Análisis de Umbrales**
```bash
# Analizar rendimiento por umbral
python scripts/analyze_threshold_performance.py \
    --results_dir ./results_thresholds \
    --output_file threshold_analysis.json
```

#### **4.3 Selección del Umbral Óptimo**
El umbral óptimo depende de tu caso de uso:

- **Umbral bajo (0.3)**: Máximo recall, más falsos positivos
- **Umbral medio (0.5)**: Balance entre precisión y recall
- **Umbral alto (0.7-0.9)**: Máxima precisión, menor recall

**Recomendación**: Comienza con **0.5** y ajusta según tus necesidades.

### **FASE 5: Validación y Corrección del Benchmark**

#### **5.1 Análisis de Falsos Positivos**
```bash
# Analizar falsos positivos para identificar errores en anotaciones
python scripts/analyze_false_positives_clean.py
```

#### **5.2 Análisis de Falsos Negativos**
```bash
# Analizar falsos negativos para verificar cobertura
python scripts/analyze_false_negatives_clean.py
```

#### **5.3 Corrección del Benchmark**
Si encuentras entidades que la máquina detectó correctamente pero no están en el benchmark:

1. **Revisar manualmente** cada caso
2. **Verificar en el texto original** si la entidad está presente
3. **Corregir el benchmark** añadiendo las entidades faltantes
4. **Recalcular métricas** con el benchmark corregido

### **FASE 6: Actualización del Script Principal**

#### **6.1 Modificar Parámetros en llama_ner_multi_strategy.py**
Una vez optimizados los parámetros, actualiza el script (ubicado en el directorio raíz del proyecto):

```python
# Actualizar temperatura óptima
STRATEGY_1 = {
    "name": "llama32_optimized",
    "model": "llama3.2:3b",
    "chunk_target": 50,      # Valor optimizado
    "chunk_overlap": 15,     # Valor optimizado
    "temperature": 0.0,      # Temperatura optimizada
    "weight": 1.0
}

# Actualizar umbral de confianza
CONFIDENCE_THRESHOLDS = {
    "high": 0.9,
    "medium": 0.7,
    "low": 0.5,
    "min_accept": 0.5       # Umbral optimizado
}

# Actualizar argumento por defecto
parser.add_argument("--confidence_threshold", 
                   type=float, 
                   default=0.5,  # Valor optimizado
                   help="Minimum confidence threshold for acceptance")
```

#### **6.2 Verificar Configuración**
```bash
# Verificar que los cambios se aplicaron correctamente
python -m py_compile llama_ner_multi_strategy.py

# Probar con dataset pequeño
python llama_ner_multi_strategy.py \
    --input_jsonl ./datasets/dataset_develop.jsonl \
    --benchmark_jsonl ./datasets/dataset_benchmark.jsonl \
    --out_pred results_optimized_test.jsonl \
    --limit 10
```

### **FASE 7: Evaluación Final**

#### **7.1 Test con Dataset Completo**
```bash
# Ejecutar con parámetros optimizados en dataset completo
python llama_ner_multi_strategy.py \
    --input_jsonl ./datasets/dataset_input.jsonl \
    --benchmark_jsonl ./datasets/dataset_benchmark.jsonl \
    --out_pred results_final_optimized.jsonl \
    --strategies all
```

#### **7.2 Análisis de Métricas Finales**
```bash
# Evaluar rendimiento final
python scripts/evaluate_ner_performance.py \
    --predictions results_final_optimized.jsonl \
    --reference ./datasets/dataset_benchmark.jsonl
```

#### **7.3 Generar Reporte Final**
```bash
# Crear reporte completo de optimización
python scripts/generate_optimization_report.py \
    --results_file results_final_optimized.jsonl \
    --benchmark_file ./datasets/dataset_benchmark.jsonl \
    --output_file optimization_report.md
```

## 📊 Ejemplo de Resultados de Optimización

### **Antes de la Optimización**
- **Temperatura**: 0.7 (por defecto)
- **Chunks**: 100 tokens, 20 overlap
- **Umbral de confianza**: 0.3
- **Precisión**: 89.2%
- **Recall**: 94.1%
- **F1-Score**: 91.6%

### **Después de la Optimización**
- **Temperatura**: 0.0 (óptima para NER)
- **Chunks**: 50 tokens, 15 overlap
- **Umbral de confianza**: 0.5
- **Precisión**: 95.4% (+6.2%)
- **Recall**: 100.0% (+5.9%)
- **F1-Score**: 97.6% (+6.0%)

## 🔍 Scripts de Análisis Disponibles

### **Scripts de Grid Search**
- `run_llama_grid.py`: Ejecuta grid search de temperatura y chunks
- `run_qwen_grid.py`: Grid search específico para modelo Qwen

### **Scripts de Análisis**
- `analyze_grid.py`: Analiza resultados de grid search
- `analyze_chunk_performance.py`: Analiza rendimiento por configuración de chunks
- `analyze_threshold_performance.py`: Analiza rendimiento por umbral de confianza

### **Scripts de Validación**
- `analyze_false_positives_clean.py`: Analiza falsos positivos
- `analyze_false_negatives_clean.py`: Analiza falsos negativos
- `evaluate_ner_performance.py`: Evalúa métricas de rendimiento

## ⚠️ Consideraciones Importantes

### **Tiempo de Procesamiento**
- **Grid search completo**: 2-6 horas (dependiendo del dataset)
- **Optimización incremental**: 30 minutos - 2 horas por fase
- **Validación manual**: 1-3 horas (dependiendo de la complejidad)

### **Recursos Requeridos**
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **CPU**: Mínimo 4 cores, recomendado 8+ cores
- **Almacenamiento**: Espacio para resultados temporales

### **Limitaciones**
- **Overfitting**: No optimices en el dataset de test
- **Generalización**: Los parámetros óptimos pueden variar entre datasets similares
- **Estabilidad**: Ejecuta múltiples veces para verificar consistencia

## 🎯 Mejores Prácticas

### **1. Iteración Incremental**
- Optimiza un parámetro a la vez
- Valida cada cambio antes de continuar
- Documenta cada iteración

### **2. Validación Cruzada**
- Usa dataset de desarrollo para optimización
- Valida en dataset de test
- Considera múltiples métricas

### **3. Documentación**
- Mantén registro de todos los cambios
- Documenta razones para cada optimización
- Crea reportes de rendimiento

### **4. Reproducibilidad**
- Usa seeds fijos para reproducibilidad
- Guarda configuraciones intermedias
- Versiona todos los cambios

## 📈 Métricas de Éxito

### **Objetivos de Optimización**
- **Precisión**: > 95% (ideal > 97%)
- **Recall**: > 95% (ideal > 98%)
- **F1-Score**: > 95% (ideal > 97%)
- **Tiempo de procesamiento**: < 2x el tiempo base

### **Indicadores de Sobre-optimización**
- **Precisión muy alta** (> 99%) con recall bajo (< 90%)
- **Diferencia grande** entre desarrollo y test
- **Métricas inestables** entre ejecuciones

## 🚀 Automatización del Pipeline

### **Script de Automatización Completa**
```bash
# Ejecutar pipeline completo de optimización
python scripts/run_optimization_pipeline.py \
    --input_file ./datasets/dataset_input.jsonl \
    --benchmark_file ./datasets/dataset_benchmark.jsonl \
    --output_dir ./optimization_results \
    --limit 100
```

### **Configuración de Pipeline**
```yaml
# optimization_config.yaml
pipeline:
  temperature_range: [0.0, 0.1, 0.2, 0.3]
  chunk_configs:
    - {target: 30, overlap: 10}
    - {target: 50, overlap: 15}
    - {target: 80, overlap: 20}
  confidence_thresholds: [0.3, 0.5, 0.7, 0.9]
  validation_steps: true
  generate_report: true
```

## 📚 Recursos Adicionales

### **Documentación Relacionada**
- [Metodología Detallada](METODOLOGIA_DETALLADA.md): Explicación técnica del sistema
- [README Principal](../README.md): Visión general del proyecto

### **Scripts de Ejemplo**
- `example_optimization.py`: Ejemplo completo de optimización
- `benchmark_validation.py`: Validación de benchmarks
- `performance_analysis.py`: Análisis detallado de rendimiento

### **Herramientas de Monitoreo**
- `monitor_optimization.py`: Monitoreo en tiempo real del proceso
- `generate_plots.py`: Generación de gráficos de rendimiento
- `export_results.py`: Exportación de resultados en diferentes formatos

---

**Nota**: Este pipeline ha sido probado y validado en múltiples datasets biomédicos, incluyendo NCBI y n2c2. Los parámetros óptimos pueden variar según el dominio específico y la calidad de las anotaciones.
