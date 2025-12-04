# NER Analysis Module

Módulo compartido para análisis de resultados de grid search NER.

## Estructura

```
ner_analysis/
├── __init__.py           # Exporta todas las funciones públicas
├── file_parser.py        # Parseo de nombres de archivo y lectura de resultados
├── evaluator.py          # Wrapper para evaluate_ner_performance.py
└── aggregator.py         # Funciones de agregación y análisis
```

## Componentes

### file_parser.py

**`parse_filename(filename: str) -> Optional[Dict]`**
- Extrae parámetros del nombre de archivo de resultados
- Soporta formatos:
  - `results_{modelo}_{dataset}_chunk{n}_ov{n}.jsonl`
  - `results_{modelo}_{dataset}_chunk{n}_ov{n}_temp{n}.jsonl`
  - `results_{modelo}_{dataset}_chunk{n}_ov{n}_temp{n}_conf{n}.jsonl`
- Retorna: Dict con `model`, `dataset`, `chunk`, `overlap`, `temperature` (opcional), `confidence` (opcional)

**`read_results(filepath: str) -> Optional[Dict]`**
- Lee archivo JSONL y calcula estadísticas
- Retorna: Dict con `docs`, `entities`, `avg_conf`, `min_conf`, `max_conf`, `entities_set`

### evaluator.py

**`evaluate_performance(result_file: str, reference_file: str) -> Dict`**
- Ejecuta `evaluate_ner_performance.py` vía subprocess
- Extrae métricas: `precision`, `recall`, `f1_score`, `tp`, `fp`, `fn`
- Retorna: Dict con métricas o dict vacío si hay error

### aggregator.py

**`aggregate_by_parameter(results: List[Dict], param_name: str) -> Dict`**
- Agrupa resultados por parámetro (chunk, temperature, etc.)
- Retorna: Dict con listas de valores agregados

**`print_parameter_analysis(param_name: str, display_name: str, aggregated_data: Dict) -> None`**
- Imprime análisis agregado de un parámetro
- Muestra promedios de entidades, confianza, P/R/F1

**`get_top_configurations(results: List[Dict], metric: str, top_n: int) -> List[Dict]`**
- Obtiene top N configuraciones según una métrica
- Útil para rankings por F1, entidades, etc.

**`print_section(title: str, char: str, width: int) -> None`**
- Imprime sección formateada para organizar output

## Uso

```python
from ner_analysis import (
    parse_filename,
    read_results,
    evaluate_performance,
    aggregate_by_parameter,
    print_parameter_analysis,
    get_top_configurations,
    print_section
)

# Parsear archivo
params = parse_filename("results_qwen_n2c2_chunk60_ov20_temp0.3.jsonl")
# {'model': 'qwen', 'dataset': 'n2c2', 'chunk': 60, 'overlap': 20, 'temperature': 0.3}

# Leer resultados
results = read_results("path/to/file.jsonl")
# {'docs': 10, 'entities': 45, 'avg_conf': 0.85, ...}

# Evaluar
metrics = evaluate_performance("predictions.jsonl", "reference.jsonl")
# {'precision': 0.92, 'recall': 0.88, 'f1_score': 0.90, 'tp': 42, 'fp': 3, 'fn': 5}

# Agregar por parámetro
aggregated = aggregate_by_parameter(all_results, 'temperature')

# Mostrar análisis
print_parameter_analysis('temperature', 'Temperature', aggregated)

# Top configuraciones
top_f1 = get_top_configurations(all_results, 'f1_score', top_n=10)
```

## Scripts que usan este módulo

- **`analyze_grid_results.py`**: Análisis universal de grid search
- **`analyze_temperature_grid.py`**: Análisis enfocado en temperaturas

## Ventajas

✅ **DRY**: Evita duplicación de código  
✅ **Mantenibilidad**: Cambios en un solo lugar  
✅ **Testeable**: Funciones puras fáciles de probar  
✅ **Reutilizable**: Cualquier script puede importar y usar  
✅ **Extensible**: Fácil agregar nuevas funciones de análisis
