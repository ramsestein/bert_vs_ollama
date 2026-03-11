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

### **Dataset Hospital Clínic Barcelona (Spanish Clinical)**

**Descripción**: Dataset clínico en español procedente de notas clínicas reales del Hospital Clínic de Barcelona, enfocado en las 10 entidades diagnósticas más frecuentes.

**Características**:
- **Fuente**: Notas clínicas y episodios médicos reales
- **Dominio**: Medicina clínica - atención primaria y hospitalaria
- **Idioma**: Castellano y catalán
- **Entidades**: 10 códigos ICD10 más frecuentes 
  - I10: Hipertensión arterial
  - E78.5: Dislipemia
  - Z87.891: Exfumador
  - E11.9: Diabetes mellitus tipo 2
  - F17.210: Fumador
  - Z79.01: Anticoagulado
  - I25.10: Cardiopatía isquémica
  - Z79.82: AAS (Ácido acetilsalicílico)
  - N17.9: Insuficiencia renal aguda
  - I48.91: Fibrilación auricular
- **Tamaño**: 
  - Corpus original: 158,717 documentos clínicos
  - Dataset final: 93,678 documentos con entidades válidas
  - Total entidades: 202,779 menciones diagnósticas
- **Calidad**: Anotaciones médicas profesionales con validación automática de presencia textual y posterior revisión humana. 

**Uso en el Proyecto**:
- **Validación multilingüe** del sistema NER en español
- **Benchmark de entidades frecuentes** en medicina clínica española
- **Evaluación de robustez** con variantes textuales y abreviaturas médicas
- **Demostración de capacidad** de procesamiento a gran escala

### **Comparación Extendida de Datasets**

| Aspecto | NCBI | n2c2 | Hospital Clínic BCN |
|---------|------|------|---------------------|
| **Dominio** | Medicina general | Medicina clínica (EN) | Medicina clínica (ES) |
| **Fuente** | PubMed abstracts | Notas clínicas USA | Notas clínicas España |
| **Idioma** | Inglés | Inglés | Español/Catalán |
| **Entidades** | Enfermedades, genes | Condiciones, síntomas | Top 10 ICD10 diagnósticos |
| **Complejidad** | Media | Alta | Alta + Multilingüe |
| **Tamaño** | ~100 docs | ~100 docs | ~93,678 docs |

### **Adaptaciones Específicas del Dataset Español**

#### **1. Proceso de Filtrado y Validación en 3 Fases**

El dataset original contenía anotaciones diagnósticas que no siempre aparecían literalmente en el texto. Se implementó un sistema de validación en 3 fases para asegurar que cada entidad anotada tiene evidencia textual:

**Fase 1: Búsqueda Exacta del REV (Término Revisado)**
- Normalización de texto (sin acentos, case-insensitive)
- Búsqueda exacta del término anotado
- **Resultado**: 190,180 coincidencias (93.8% del total)

**Fase 2: Búsqueda de Términos Canónicos por ICD10**
- Para cada código ICD10, se definieron términos canónicos de búsqueda
- Ejemplo: E11.9 (Diabetes) → ["diabetes mellitus", "dm", "dm2", "diabetes tipo 2"]
- Búsqueda exacta: 19,356 coincidencias (9.5%)
- Búsqueda fuzzy (similitud ≥ 0.85): 3,772 coincidencias (1.9%)

**Fase 3: Búsqueda Fuzzy del REV**
- Fuzzy matching del término original con umbral de similitud 0.85
- **Resultado**: 182 coincidencias (0.1%)

**Estadísticas de Filtrado**:
```
Total entidades anotadas:        221,001
Entidades validadas (en texto):  202,779 (91.8%)
Entidades eliminadas:             18,222 (8.2%)

Por tipo de coincidencia:
  - REV exacto:                   190,180 (93.8%)
  - Término canónico exacto:       19,356 (9.5%)
  - Término canónico fuzzy:         3,772 (1.9%)
  - REV fuzzy:                        182 (0.1%)
  - Fuzzy total:                    1.9%
```

#### **2. Eliminación de Documentos con Detección Parcial**

**Estrategia**: Solo se incluyeron documentos donde **TODAS** las entidades anotadas fueron encontradas en el texto.

**Resultados**:
- Documentos con todas las entidades encontradas: **93,678** (93.5%)
- Documentos con errores de detección (parcial): **6,544** (6.5%)
- Documentos con excepciones de procesamiento: **0**

Esta decisión garantiza un benchmark de alta calidad donde cada documento es completamente verificable.

#### **3. Normalización Multilingüe (Español/Catalán)**

**Desafíos**:
- Textos mezclados en castellano y catalán
- Diferentes formas de acentuación
- Variantes regionales de términos médicos

**Soluciones Implementadas**:
- Normalización Unicode (NFD) para remover diacríticos
- Búsqueda case-insensitive
- Mapeo de variantes detectadas:
  - "hipertensió arterial" → "hipertensión arterial" (catalán)
  - "dislipèmia" → "dislipemia" (catalán)
  - "cardiopatia" → "cardiopatía" (sin acento)

#### **4. Manejo de Abreviaturas Médicas Españolas**

**Términos Canónicos Definidos** (extracto):

| ICD10 | Término Principal | Abreviaturas/Variantes |
|-------|-------------------|------------------------|
| I10 | Hipertensión arterial | HTA |
| E78.5 | Dislipemia | DLP, DL |
| E11.9 | Diabetes mellitus tipo 2 | DM, DM2, DMII, DM II |
| Z87.891 | Exfumador | Ex-fumador, Ex fumador |
| F17.210 | Fumador | Tabaco, Tabaquismo |
| Z79.01 | Anticoagulado | Anticoagulante, Sintrom |
| I25.10 | Cardiopatía isquémica | EAC |
| Z79.82 | AAS | Adiro, Aspirina |
| N17.9 | Insuficiencia renal aguda | IRA, AKI |
| I48.91 | Fibrilación auricular | FA, ACXFA, AC x FA |

#### **5. Distribución de Entidades en el Corpus**

**Top 10 ICD10 por Frecuencia**:
```
 1. I10      | Hipertensión arterial        | 45,254 menciones | 45,066 docs
 2. E78.5    | Dislipemia                   | 34,979 menciones | 34,800 docs
 3. Z87.891  | Exfumador                    | 25,578 menciones | 25,550 docs
 4. E11.9    | Diabetes mellitus tipo 2     | 19,735 menciones | 19,659 docs
 5. F17.210  | Fumador                      | 17,295 menciones | 17,249 docs
 6. Z79.01   | Anticoagulado                | 14,330 menciones | 14,251 docs
 7. I25.10   | Cardiopatía isquémica        | 12,514 menciones | 12,450 docs
 8. Z79.82   | AAS                          | 11,699 menciones | 11,670 docs
 9. N17.9    | Insuficiencia renal aguda    | 11,311 menciones | 11,286 docs
10. I48.91   | Fibrilación auricular        | 10,084 menciones | 10,017 docs
```

**Observaciones**:
- Alta prevalencia de factores de riesgo cardiovascular (HTA, Dislipemia, Diabetes)
- Significativa documentación de hábitos tabáquicos (Fumador/Exfumador)
- Representación de tratamientos comunes (Anticoagulado, AAS)

#### **6. Scripts de Preprocesamiento Desarrollados**

**`filter_spanish_entities.py`**:
- Filtrado de entidades diagnósticas (sem='DIAG')
- Validación de presencia textual en 3 fases
- Normalización multilingüe (ES/CA)
- Fuzzy matching con umbral configurable
- Generación de estadísticas detalladas
- Logging exhaustivo del proceso

**`create_input_with_entity_variants.py`**:
- Generación de archivo de entrada con todas las variantes textuales
- Formato JSONL estandarizado para evaluación
- Inclusión de abreviaturas y sinónimos médicos
- Compatible con el pipeline de evaluación existente

#### **7. Estructura de Archivos Generados**
```bash
datasets/
├── salida_episodios/              # Corpus original (158,717 JSONs)
│   └── salida_episodios/
│       └── *.json                 # Episodios clínicos individuales
├── spanish_clinical_filtered.jsonl           # Dataset filtrado (93,678 docs)
└── spanish_clinical_filtered_input_final.jsonl  # Input con variantes para evaluación
```

**Formato del Dataset Filtrado** (`spanish_clinical_filtered.jsonl`):
```json
{
  "PMID": "nombre_archivo",
  "Texto": "texto completo del episodio clínico...",
  "Entidad": [
    {"texto": "HTA", "tipo": "DIAG", "codigo": "I10"},
    {"texto": "DM2", "tipo": "DIAG", "codigo": "E11.9"}
  ]
}
```

**Formato del Input de Evaluación** (`spanish_clinical_filtered_input_final.jsonl`):
```json
{
  "PMID": "nombre_archivo",
  "Texto": "texto completo del episodio clínico...",
  "Entidad": [
    {"texto": "hta", "tipo": "DIAG"},
    {"texto": "hipertensión arterial", "tipo": "DIAG"},
    {"texto": "hipertensión", "tipo": "DIAG"},
    {"texto": "dislipemia", "tipo": "DIAG"},
    {"texto": "dlp", "tipo": "DIAG"},
    ... // Todas las variantes de los 10 códigos ICD10
  ]
}
```

#### **8. Validación de Calidad del Dataset**

**Métricas de Calidad Implementadas**:

1. **Integridad de Documentos**: 100% de documentos procesados sin excepciones
2. **Validación Textual**: 91.8% de entidades anotadas tienen evidencia en texto
3. **Completitud de Documentos**: Solo documentos con 100% de entidades detectadas
4. **Cobertura de Variantes**: Múltiples formas textuales por cada código ICD10

**Casos de Uso de Fuzzy Matching**:

El fuzzy matching (1.9% del total) capturó variaciones importantes:
- Errores tipográficos: "diabetis" → "diabetes"
- Variantes morfológicas: "anticoagulat" → "anticoagulado"
- Abreviaturas no estándar: "d.m." → "dm"

#### **9. Lecciones Aprendidas del Dataset Español**

1. **Validación Textual Esencial**: El 8.2% de anotaciones originales no tenían evidencia textual, destacando la importancia de la validación automática

2. **Variabilidad Terminológica**: Una misma entidad puede aparecer con 5-10 variantes diferentes en español clínico (ej: "DM2", "Diabetes mellitus tipo 2", "DM", "DMII")

3. **Multilingüismo Implícito**: En contextos bilingües (ES/CA), la normalización de acentos es crítica para alcanzar alta cobertura

4. **Balance Precision-Recall**: El fuzzy matching con umbral 0.85 logró capturar variaciones legítimas (1.9%) sin introducir ruido significativo

5. **Escala del Corpus**: Con casi 100K documentos y 200K+ entidades, este dataset permite validación estadísticamente significativa


### **Impacto de las Adaptaciones**

1. **Alta Calidad de Benchmark**: Solo documentos con 100% de entidades detectadas garantiza evaluación fiable

2. **Cobertura de Variantes**: El sistema debe reconocer tanto "Diabetes mellitus tipo 2" como "DM2" para ser útil en clínica real

3. **Escalabilidad Demostrada**: Procesamiento exitoso de ~160K documentos sin errores críticos

4. **Reproducibilidad**: Pipeline completamente automatizado y documentado con logging detallado

5. **Validación Multilingüe**: Primera evaluación del sistema NER en español clínico a gran escala