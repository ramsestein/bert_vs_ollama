## 📊 Datasets Utilizados

### **Dataset NCBI (National Center for Biotechnology Information)**

**Descripción**: Dataset biomédico estándar para evaluación de sistemas NER, centrado en entidades biomédicas como enfermedades, medicamentos y genes.

**Características**:
- **Fuente**: PubMed abstracts y artículos biomédicos
- **Dominio**: Medicina general y biología molecular
- **Entidades**: Enfermedades, medicamentos, genes, proteínas
- **Tamaño**: 100 documentos de test, 93 procesados exitosamente
- **Calidad**: Anotaciones humanas de alta calidad

**Uso en el Proyecto**:
- **Desarrollo inicial** del sistema NER
- **Validación de estrategias** básicas
- **Benchmark de referencia** para comparaciones
- **Optimización inicial** de parámetros

### **Dataset n2c2 (National NLP Clinical Challenges)**

**Descripción**: Dataset clínico especializado para desafíos de procesamiento de lenguaje natural en medicina clínica, con enfoque en entidades médicas específicas.

**Características**:
- **Fuente**: Notas clínicas y reportes médicos reales
- **Dominio**: Medicina clínica y atención al paciente
- **Entidades**: Condiciones médicas, síntomas, medicamentos, procedimientos
- **Tamaño**: 100 documentos de test (primeros 100 del dataset completo)
- **Calidad**: Anotaciones clínicas profesionales

**Uso en el Proyecto**:
- **Optimización avanzada** de parámetros
- **Validación de estrategias** refinadas
- **Corrección de benchmarks** humanos
- **Demostración de capacidad** de detección de errores en anotaciones

### **Comparación de Datasets**

| Aspecto | NCBI | n2c2 |
|---------|------|-------|
| **Dominio** | Medicina general | Medicina clínica |
| **Fuente** | PubMed abstracts | Notas clínicas |
| **Entidades** | Enfermedades, genes | Condiciones, síntomas |
| **Complejidad** | Media | Alta |

### **Lecciones Aprendidas de los Datasets**

1. **NCBI**: Demostró la capacidad del sistema para alcanzar **precisión casi perfecta** en dominios biomédicos estándar
2. **n2c2**: Reveló la capacidad del sistema para **identificar errores en anotaciones humanas** y mejorar la calidad de benchmarks
3. **Validación Cruzada**: Los resultados en ambos datasets confirman la **robustez y generalización** del sistema
4. **Optimización**: Cada dataset requirió **parámetros específicos** para alcanzar el rendimiento óptimo

### **Acceso a los Datasets**

Los datasets utilizados están disponibles en la carpeta `datasets/` del proyecto:

```bash
datasets/
├── ncbi_develop.jsonl            # Dataset de desarrollo NCBI
├── ncbi_test.jsonl               # Dataset de test NCBI
├── n2c2_test_input.jsonl         # Dataset de entrada n2c2
└── n2c2_test.jsonl               # Dataset de benchmark n2c2
```

**Nota**: Los datasets están preprocesados y optimizados para el sistema NER multi-estrategia. Para uso con otros sistemas, se recomienda consultar las fuentes originales.

### **Adaptaciones Realizadas para Nuestro Uso**

#### **1. Separación de Archivos de Entrada y Benchmark**

**Problema Original**: Los datasets originales mezclaban texto y anotaciones en un solo archivo, dificultando la evaluación independiente del sistema.

**Solución Implementada**:
- **`*_input.jsonl`**: Contiene solo el texto y las entidades específicas a buscar
- **`*_benchmark.jsonl`**: Contiene las anotaciones de referencia para evaluación

**Beneficios**:
- **Evaluación independiente** del rendimiento del sistema
- **Reutilización** de datasets para diferentes experimentos
- **Claridad** en el propósito de cada archivo
- **Facilita** el pipeline de optimización

#### **2. Limpieza y Preprocesamiento de Entidades**

**Dataset n2c2**:
- **Eliminación de entidades**: Removimos entidades para simplificar y mejorar el tiempo de proceso de benchmark manteniendo una n de entidades que solemos usar en nuestro entorno particular
- **Normalización de formatos**: Estandarizamos la estructura JSONL para consistencia

**Dataset NCBI**:
- **Separación de entidades**: Dividimos entidades compuestas en entidades individuales
- **Validación de tipos**: Aseguramos que todas las entidades tengan tipos válidos
- **Limpieza de texto**: Removimos caracteres especiales y normalizamos el formato

#### **3. Creación de Datasets de Desarrollo**

**Dataset NCBI**:
- **`ncbi_develop.jsonl`**: Subconjunto de 50-100 documentos para optimización de parámetros
- **`ncbi_test.jsonl`**: Dataset completo para evaluación final

**Dataset n2c2**:
- **`n2c2_develop_input.jsonl`**: Subconjunto para optimización (desarrollo)
- **`n2c2_test_input.jsonl`**: Dataset completo para evaluación final

#### **4. Preprocesamiento de Texto**

**Chunking Optimizado**:
- **Tamaños de chunk**: Configurados específicamente para cada estrategia
- **Overlap**: Optimizado para evitar pérdida de entidades en bordes
- **Normalización**: Texto limpiado y estandarizado para mejor procesamiento LLM

**Entidades Candidatas**:
- **Formato estandarizado**: Estructura JSON consistente para todas las entidades
- **Tipos validados**: Categorías de entidades biomédicas estándar
- **Variaciones incluidas**: Consideración de sinónimos y abreviaturas médicas

#### **5. Scripts de Preprocesamiento Desarrollados**

```bash
scripts/
├── create_input_datasets.py       # Creación de datasets de entrada separados
├── remove_chest_pain.py          # Limpieza específica de entidades problemáticas
├── validate_jsonl_format.py      # Validación de formato y estructura
└── preprocess_text.py            # Preprocesamiento de texto y chunking
```

#### **6. Validación de Calidad de Datos**

**Checks Implementados**:
- **Integridad de JSONL**: Verificación de formato válido
- **Consistencia de entidades**: Validación de tipos y estructura
- **Calidad de texto**: Verificación de codificación y caracteres
- **Balance de datasets**: Asegurar representatividad de entidades

**Métricas de Calidad**:
- **Cobertura de entidades**: Todas las entidades objetivo están representadas
- **Distribución de tipos**: Balance entre diferentes categorías de entidades
- **Calidad de anotaciones**: Verificación de precisión de anotaciones humanas

#### **7. Adaptaciones Específicas por Dominio**

**NCBI (Medicina General)**:
- **Enfoque**: Entidades biomédicas estándar y bien definidas
- **Estrategia**: Optimización para precisión máxima
- **Configuración**: Chunks medianos, temperatura baja, alta confianza

**n2c2 (Medicina Clínica)**:
- **Enfoque**: Entidades clínicas complejas y contextuales
- **Estrategia**: Balance entre precisión y recall
- **Configuración**: Chunks pequeños, temperatura media, confianza ajustable

#### **8. Pipeline de Preprocesamiento Automatizado**

```bash
# Flujo completo de preprocesamiento
python scripts/preprocess_pipeline.py \
    --input_dir raw_datasets/ \
    --output_dir processed_datasets/ \
    --dataset_type ncbi \
    --validation_level strict
```

**Pasos Automatizados**:
1. **Validación de formato** del dataset original
2. **Separación** en archivos de entrada y benchmark
3. **Limpieza** de entidades y texto
4. **Normalización** de estructura JSONL
5. **Validación de calidad** final
6. **Generación de reportes** de preprocesamiento

#### **9. Documentación de Adaptaciones**

**Archivos de Configuración**:
- **`preprocessing_config.yaml`**: Configuración de parámetros de preprocesamiento
- **`entity_mappings.json`**: Mapeos de entidades y sinónimos
- **`quality_metrics.json`**: Métricas de calidad de cada dataset procesado

**Logs de Preprocesamiento**:
- **Registro detallado** de todas las transformaciones aplicadas
- **Métricas de calidad** antes y después del procesamiento
- **Errores y advertencias** durante el preprocesamiento

### **Impacto de las Adaptaciones**

1. **Facilitación de Evaluación**: Separación clara entre entrada y benchmark
2. **Optimización de Parámetros**: Datasets de desarrollo permitieron tuning eficiente
3. **Validación de Calidad**: Proceso automatizado asegura consistencia
4. **Reproducibilidad**: Todas las adaptaciones están documentadas y automatizadas
