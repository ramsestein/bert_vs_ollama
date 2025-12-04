#!/usr/bin/env python3
"""
Aggregator Module
Functions for aggregating and analyzing results by parameters
"""

from collections import defaultdict
from typing import List, Dict


def print_section(title: str, char: str = "=", width: int = 80) -> None:
    """Imprime una sección formateada"""
    print(f"\n{title}")
    print(char * width)


def aggregate_by_parameter(results: List[Dict], param_name: str) -> Dict:
    """
    Agrega resultados agrupados por un parámetro específico
    
    Args:
        results: Lista de resultados a agregar
        param_name: Nombre del parámetro por el que agrupar (ej: 'chunk', 'temperature')
    
    Returns:
        Dict con agregaciones por valor del parámetro
    """
    param_analysis = defaultdict(lambda: {
        'entities': [],
        'avg_conf': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    })
    
    for result in results:
        if param_name not in result:
            continue
        
        param_value = result[param_name]
        param_analysis[param_value]['entities'].append(result['entities'])
        param_analysis[param_value]['avg_conf'].append(result['avg_conf'])
        
        if 'precision' in result:
            param_analysis[param_value]['precision'].append(result['precision'])
        if 'recall' in result:
            param_analysis[param_value]['recall'].append(result['recall'])
        if 'f1_score' in result:
            param_analysis[param_value]['f1_score'].append(result['f1_score'])
    
    return dict(param_analysis)


def print_parameter_analysis(param_name: str, display_name: str, aggregated_data: Dict) -> None:
    """
    Imprime análisis agregado de un parámetro
    
    Args:
        param_name: Nombre técnico del parámetro
        display_name: Nombre para mostrar
        aggregated_data: Datos agregados por aggregate_by_parameter
    """
    if not aggregated_data:
        return
    
    print(f"\n[ANALISIS POR {display_name.upper()}]")
    
    for value in sorted(aggregated_data.keys()):
        data = aggregated_data[value]
        
        avg_entities = sum(data['entities']) / len(data['entities'])
        avg_conf = sum(data['avg_conf']) / len(data['avg_conf'])
        avg_precision = sum(data['precision']) / len(data['precision']) if data['precision'] else 0
        avg_recall = sum(data['recall']) / len(data['recall']) if data['recall'] else 0
        avg_f1 = sum(data['f1_score']) / len(data['f1_score']) if data['f1_score'] else 0
        
        # Formatear el valor según el tipo
        if isinstance(value, float):
            value_str = f"{value:.1f}"
        else:
            value_str = str(value)
        
        print(f"  {param_name}={value_str:>6}: "
              f"entidades={avg_entities:6.2f}, "
              f"conf={avg_conf:5.3f}, "
              f"P={avg_precision:5.3f}, "
              f"R={avg_recall:5.3f}, "
              f"F1={avg_f1:5.3f}")


def get_top_configurations(results: List[Dict], metric: str, top_n: int = 10) -> List[Dict]:
    """
    Obtiene las mejores configuraciones según una métrica
    
    Args:
        results: Lista de resultados
        metric: Métrica para ordenar ('f1_score', 'entities', etc.)
        top_n: Número de resultados a retornar
    
    Returns:
        Lista con los top N resultados
    """
    valid_results = [r for r in results if metric in r and r[metric] > 0]
    return sorted(valid_results, key=lambda x: x[metric], reverse=True)[:top_n]
