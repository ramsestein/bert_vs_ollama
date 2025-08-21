# 🧪 **TESTS DEL SISTEMA NER MULTI-STRATEGY**

## 📋 **RESUMEN**

Este directorio contiene una suite completa de tests para validar tanto el **sistema refactorizado** como el **sistema original** del NER Multi-Strategy.

## 🏗️ **ESTRUCTURA DE TESTS**

```
tests/
├── README.md                      # Esta documentación
├── conftest.py                    # Configuración global de pytest
├── run_tests.py                   # Script ejecutor de tests
├── unit/                          # 🔧 TESTS UNITARIOS
│   ├── test_config.py             # Tests de configuración
│   ├── test_core.py               # Tests de módulos core
│   ├── test_strategies.py         # Tests de estrategias
│   └── test_utils.py              # Tests de utilidades
├── integration/                   # 🔗 TESTS DE INTEGRACIÓN
│   ├── test_refactored_system.py  # Tests sistema refactorizado
│   └── test_original_system.py    # Tests sistema original
├── data/                          # 📊 DATOS DE PRUEBA
│   ├── test_input.jsonl           # Entrada de prueba
│   └── test_benchmark.jsonl       # Benchmark de prueba
└── fixtures/                      # 🛠️ FIXTURES (futuro)
```

## 🚀 **EJECUCIÓN RÁPIDA**

### **Ejecutar todos los tests**
```bash
# Método simple
python tests/run_tests.py

# Con pytest directo
python -m pytest tests/ -v
```

### **Tests específicos**
```bash
# Solo tests unitarios
python tests/run_tests.py --type unit

# Solo tests de integración
python tests/run_tests.py --type integration

# Con cobertura de código
python tests/run_tests.py --type coverage
```

## 🔧 **TESTS UNITARIOS**

### **test_config.py**
- ✅ Validación de estrategias
- ✅ Configuración de umbrales
- ✅ Settings del sistema
- ✅ Validación de parámetros

### **test_core.py**
- ✅ Procesamiento de texto
- ✅ Gestión de archivos
- ✅ Cliente LLM y caché
- ✅ Chunking y normalización

### **test_strategies.py**
- ✅ Estrategia regex
- ✅ Orquestador multi-estrategia
- ✅ Sistema de confianza
- ✅ Manejo de errores

### **test_utils.py**
- ✅ Parser CLI
- ✅ Puntuación de confianza
- ✅ Matching de entidades
- ✅ Validación de argumentos

## 🔗 **TESTS DE INTEGRACIÓN**

### **test_refactored_system.py**
- ✅ Flujo completo del sistema refactorizado
- ✅ Carga y procesamiento de documentos
- ✅ Guardado de resultados
- ✅ Manejo de errores
- ✅ Modo sin benchmark

### **test_original_system.py**
- ✅ Compatibilidad del sistema original
- ✅ Validación de argumentos CLI
- ✅ Comparación de salidas
- ✅ Manejo graceful de errores
- ✅ Compatibilidad hacia atrás

## 📊 **DATOS DE PRUEBA**

### **test_input.jsonl**
```json
{"PMID": "test_001", "Texto": "Patient diagnosed with diabetes mellitus...", "Entidad": [...]}
{"PMID": "test_002", "Texto": "Patient presents with chest pain...", "Entidad": [...]}
```

### **test_benchmark.jsonl**
Misma estructura que el input para validación de métricas.

## 🛠️ **FIXTURES GLOBALES**

Definidas en `conftest.py`:

- **`sample_documents`**: Documentos de prueba sintéticos
- **`sample_entity_candidates`**: Lista de entidades candidatas
- **`sample_strategies`**: Estrategias de prueba simplificadas
- **`temp_jsonl_file`**: Archivo temporal para tests
- **`temp_dir`**: Directorio temporal
- **`disable_llm_calls`**: Mock para llamadas LLM
- **`sample_detection_results`**: Resultados de detección mock

## 🎯 **TIPOS DE TESTS**

### **🔧 Tests Unitarios**
- **Propósito**: Validar módulos individuales
- **Alcance**: Funciones y clases específicas
- **Velocidad**: Rápidos (< 1 segundo por test)
- **Dependencias**: Mínimas (mocks para LLM)

### **🔗 Tests de Integración**
- **Propósito**: Validar interacción entre módulos
- **Alcance**: Flujos completos del sistema
- **Velocidad**: Moderados (1-30 segundos por test)
- **Dependencias**: Archivos, pero sin Ollama real

### **⚡ Tests de Performance** (futuro)
- **Propósito**: Validar rendimiento
- **Alcance**: Tiempo de procesamiento
- **Velocidad**: Lentos (minutos)
- **Dependencias**: Datos reales grandes

## 📈 **MÉTRICAS Y COBERTURA**

### **Ejecutar con cobertura**
```bash
python tests/run_tests.py --type coverage
```

### **Ver reporte HTML**
```bash
# Después de ejecutar tests con cobertura
open htmlcov/index.html  # Linux/Mac
start htmlcov/index.html # Windows
```

### **Objetivos de cobertura**
- **Módulos core**: > 90%
- **Módulos config**: > 95%
- **Módulos strategies**: > 85%
- **Módulos utils**: > 90%

## 🚨 **TESTING SIN OLLAMA**

Los tests están diseñados para **NO requerir Ollama** en funcionamiento:

- **Mocks**: Llamadas LLM son interceptadas
- **Timeouts**: Tests con timeout para evitar cuelgues
- **Graceful failures**: Manejo de errores de conexión
- **Fallbacks**: Respuestas mock predefinidas

## 🔄 **CONTINUOUS INTEGRATION**

### **Pre-commit checks**
```bash
# Validar antes de commit
python tests/run_tests.py --test-scripts
python tests/run_tests.py --validate-data
python tests/run_tests.py --type unit
```

### **Full validation**
```bash
# Validación completa
python tests/run_tests.py --type all --verbose
```

## 🐛 **DEBUGGING TESTS**

### **Ejecutar test específico**
```bash
# Test específico
python -m pytest tests/unit/test_config.py::TestStrategies::test_all_strategies_loaded -v

# Con debugging
python -m pytest tests/unit/test_config.py -v -s --tb=long
```

### **Ejecutar con pdb**
```bash
python -m pytest tests/unit/test_config.py --pdb
```

### **Ver salida completa**
```bash
python -m pytest tests/ -v -s --tb=short
```

## ⚠️ **LIMITACIONES ACTUALES**

### **No se testa con Ollama real**
- **Razón**: Evitar dependencia externa
- **Solución**: Mocks y respuestas predefinidas
- **Futuro**: Tests opcionales con Ollama real

### **Datos de prueba sintéticos**
- **Razón**: Datos pequeños y controlados
- **Solución**: Casos de prueba específicos
- **Futuro**: Tests con datasets reales

### **Sin tests de carga**
- **Razón**: Enfoque en funcionalidad
- **Solución**: Tests de integración básicos
- **Futuro**: Tests de performance y carga

## 🎯 **MEJORES PRÁCTICAS**

### **Al escribir nuevos tests**
1. **Usar fixtures** existentes cuando sea posible
2. **Aislar dependencias** con mocks
3. **Nombres descriptivos** para tests
4. **Documentar casos edge** y por qué se testean
5. **Tests independientes** (sin orden específico)

### **Al modificar el código**
1. **Ejecutar tests** antes de commit
2. **Agregar tests** para nuevas funcionalidades
3. **Actualizar tests** para cambios de API
4. **Verificar cobertura** no disminuya

## 🚀 **FUTURAS MEJORAS**

### **Tests adicionales**
- Tests de performance con datasets grandes
- Tests de stress con muchos documentos
- Tests de compatibilidad entre versiones
- Tests de regresión automáticos

### **Infraestructura**
- GitHub Actions para CI/CD
- Tests automáticos en PRs
- Reportes de cobertura automáticos
- Tests nocturnos con datos reales

### **Métricas avanzadas**
- Tiempo de ejecución por módulo
- Uso de memoria durante tests
- Cobertura de branches avanzada
- Tests de mutación

## 📞 **SOPORTE**

Para problemas con tests:

1. **Verificar datos**: `python tests/run_tests.py --validate-data`
2. **Verificar scripts**: `python tests/run_tests.py --test-scripts`
3. **Ejecutar unitarios**: `python tests/run_tests.py --type unit -v`
4. **Ver logs completos**: `python -m pytest tests/ -v -s --tb=long`

## 🎉 **CONCLUSIÓN**

Esta suite de tests proporciona:

- ✅ **Validación completa** de ambos sistemas
- ✅ **Ejecución rápida** sin dependencias externas
- ✅ **Cobertura amplia** de casos de uso
- ✅ **Manejo robusto** de errores
- ✅ **Compatibilidad** hacia atrás
- ✅ **Documentación clara** para mantenimiento

Los tests aseguran que tanto el **sistema refactorizado** como el **original** funcionen correctamente y mantengan compatibilidad.
