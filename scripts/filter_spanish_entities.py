#!/usr/bin/env python3
"""
Script para filtrar entidades diagnósticas que aparecen literalmente en el texto.

Formato de entrada (salida_episodios):
{
  "texto": "...",
  "menciones": [
    {"sem": "DIAG", "ICD10": "...", "ORIG": "...", "REV": "..."},
    ...
  ]
}

Formato de salida:
{
  "PMID": "nombre_archivo",
  "Texto": "...",
  "Entidad": [
    {"texto": "REV", "tipo": "sem", "codigo": "ICD10"},
    ...
  ]
}

Solo se incluyen entidades donde:
1. sem == "DIAG"
2. REV aparece literalmente en el texto (búsqueda case-insensitive)
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

# Códigos ICD10 de interés (top 10 más frecuentes)
TARGET_ICD10_CODES = {
    'I10',      # Hipertensión arterial
    'E78.5',    # Dislipemia
    'Z87.891',  # Exfumador
    'E11.9',    # Diabetes mellitus tipo 2
    'F17.210',  # Fumador
    'Z79.01',   # Anticoagulado
    'I25.10',   # Cardiopatía isquémica
    'Z79.82',   # AAS
    'N17.9',    # Insuficiencia renal aguda
    'I48.91'    # Fibrilación auricular
}


def normalize_text(text: str) -> str:
    """
    Normaliza el texto removiendo acentos y convirtiendo a minúsculas.
    Útil para comparar textos ignorando diferencias de acentuación.
    """
    # Normalizar unicode (NFD separa caracteres base de diacríticos)
    nfd = unicodedata.normalize('NFD', text)
    # Filtrar solo caracteres que no sean diacríticos (Mn = Nonspacing Mark)
    without_accents = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    return without_accents.lower()

def text_contains_entity(texto: str, entity_text: str) -> bool:
    """
    Verifica si el texto contiene la entidad de forma literal.
    Normaliza ambos textos (quita acentos y minúsculas) para evitar problemas con acentuación.
    
    Args:
        texto: Texto completo del documento
        entity_text: Texto de la entidad a buscar
        
    Returns:
        True si la entidad aparece en el texto, False en caso contrario
    """
    # Normalizar ambos textos para ignorar acentos
    texto_norm = normalize_text(texto)
    entity_norm = normalize_text(entity_text)
    
    # Escapar caracteres especiales de regex
    pattern = re.escape(entity_norm)
    return bool(re.search(pattern, texto_norm))


def process_document(input_path: Path) -> tuple:
    """
    Procesa un documento, filtrando solo entidades DIAG con códigos ICD10 específicos.
    
    Args:
        input_path: Ruta al archivo JSON de entrada
        
    Returns:
        Tupla (documento_salida, estadisticas_doc) donde:
        - documento_salida: Dict con formato de salida o None si no hay entidades válidas
        - estadisticas_doc: Dict con estadísticas del documento (coincidencias, no coincidencias)
    """
    # Leer el archivo de entrada
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer el nombre del archivo sin extensión para usar como PMID
    pmid = input_path.stem
    
    # Obtener el texto
    texto = data.get('texto', '')
    
    # Filtrar entidades
    entidades_filtradas = []
    stats = {
        'icd10_found': set(),
        'matches': [],
        'no_matches': []
    }
    
    for mencion in data.get('menciones', []):
        # Solo procesar si es DIAG
        sem = mencion.get('sem', '').strip()
        if sem != 'DIAG':
            continue
        
        # Obtener campos necesarios
        rev = mencion.get('REV', '').strip()
        icd10 = mencion.get('ICD10', '').strip()
        
        # Verificar que tengamos los datos necesarios
        if not rev or not icd10:
            continue
        
        # Filtrar solo los códigos ICD10 de interés
        if icd10 not in TARGET_ICD10_CODES:
            continue
        
        # Verificar si la entidad aparece en el texto
        if text_contains_entity(texto, rev):
            entidades_filtradas.append({
                "texto": rev,
                "tipo": sem,
                "codigo": icd10
            })
            stats['icd10_found'].add(icd10)
            stats['matches'].append({'icd10': icd10, 'rev': rev})
        else:
            stats['no_matches'].append({'icd10': icd10, 'rev': rev})
    
    # Si no hay entidades válidas, retornar None
    if not entidades_filtradas:
        return None, stats
    
    # Construir el documento de salida
    output_doc = {
        "PMID": pmid,
        "Texto": texto,
        "Entidad": entidades_filtradas
    }
    
    return output_doc, stats


def process_folder(input_folder: str, output_file: str):
    """
    Procesa todos los archivos JSON de una carpeta y genera un JSONL con entidades filtradas.
    
    Args:
        input_folder: Carpeta con los archivos JSON de entrada
        output_file: Archivo JSONL de salida
    """
    input_path = Path(input_folder)
    
    # Verificar que la carpeta existe
    if not input_path.exists():
        raise FileNotFoundError(f"La carpeta {input_folder} no existe")
    
    # Obtener todos los archivos JSON
    json_files = sorted(input_path.glob('*.json'))
    
    if not json_files:
        print(f"No se encontraron archivos JSON en {input_folder}")
        return
    
    print(f"Encontrados {len(json_files)} archivos JSON")
    print()
    
    # Procesar todos los documentos
    documents_with_entities = []
    documents_without_entities = []
    total_entities_before = 0
    total_entities_after = 0
    icd10_freq = {}  # Diccionario para contar frecuencias: {codigo: {texto: count}}
    
    # Estadísticas de coincidencias
    docs_per_icd10 = defaultdict(set)  # {icd10: set(PMIDs)}
    total_matches = 0
    total_no_matches = 0
    no_match_examples = defaultdict(list)  # {icd10: [(pmid, rev), ...]}
    
    print("Procesando documentos y filtrando entidades...")
    for json_file in tqdm(json_files, desc="Procesando", unit="archivo"):
        try:
            # Leer para estadísticas
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                menciones_diag = [m for m in data.get('menciones', []) if m.get('sem') == 'DIAG']
                menciones_target = [m for m in menciones_diag if m.get('ICD10', '').strip() in TARGET_ICD10_CODES]
                total_entities_before += len(menciones_target)
            
            # Procesar documento
            doc, stats = process_document(json_file)
            
            # Actualizar estadísticas de coincidencias
            total_matches += len(stats['matches'])
            total_no_matches += len(stats['no_matches'])
            
            # Guardar TODOS los casos de no coincidencias
            for no_match in stats['no_matches']:
                icd10 = no_match['icd10']
                no_match_examples[icd10].append((json_file.stem, no_match['rev']))
            
            if doc:
                documents_with_entities.append(doc)
                total_entities_after += len(doc['Entidad'])
                
                # Registrar qué documentos contienen cada código ICD10
                for icd10 in stats['icd10_found']:
                    docs_per_icd10[icd10].add(json_file.stem)
                
                # Contar frecuencias de ICD10
                for entidad in doc['Entidad']:
                    codigo = entidad['codigo']
                    texto = entidad['texto'].lower()  # Normalizar para agrupar variantes
                    
                    if codigo not in icd10_freq:
                        icd10_freq[codigo] = {}
                    if texto not in icd10_freq[codigo]:
                        icd10_freq[codigo][texto] = 0
                    icd10_freq[codigo][texto] += 1
            else:
                documents_without_entities.append(json_file.stem)
                
        except Exception as e:
            tqdm.write(f"✗ Error procesando {json_file.name}: {e}")
    
    # Mostrar estadísticas
    print()
    print("=" * 70)
    print("ESTADÍSTICAS DE FILTRADO (Solo top 10 ICD10)")
    print("=" * 70)
    print(f"Total archivos procesados: {len(json_files)}")
    print(f"Documentos con entidades válidas: {len(documents_with_entities)}")
    print(f"Documentos sin entidades válidas: {len(documents_without_entities)}")
    print()
    print(f"Total entidades target antes del filtrado: {total_entities_before}")
    print(f"Total entidades que aparecen en texto: {total_entities_after}")
    print(f"Entidades eliminadas (no aparecen): {total_entities_before - total_entities_after}")
    if total_entities_before > 0:
        print(f"Porcentaje retenido: {100 * total_entities_after / total_entities_before:.1f}%")
    print()
    print(f"Total coincidencias REV-texto: {total_matches}")
    print(f"Total NO coincidencias REV-texto: {total_no_matches}")
    print("=" * 70)
    print()
    
    # Guardar el archivo de salida
    print(f"💾 Guardando resultado: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents_with_entities:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✓ Archivo guardado: {len(documents_with_entities)} documentos")
    print(f"✓ Total entidades en archivo final: {total_entities_after}")
    
    # Mostrar algunos ejemplos de documentos sin entidades
    if documents_without_entities:
        print()
        print(f"⚠️  Documentos sin entidades válidas (primeros 10):")
        for pmid in documents_without_entities[:10]:
            print(f"  - {pmid}")
        if len(documents_without_entities) > 10:
            print(f"  ... y {len(documents_without_entities) - 10} más")
    
    # Mostrar TODOS los casos de NO coincidencia
    if no_match_examples:
        print()
        print("=" * 70)
        print("⚠️  TODOS LOS CASOS DONDE REV NO COINCIDE CON EL TEXTO")
        print("=" * 70)
        total_no_match_shown = sum(len(examples) for examples in no_match_examples.values())
        print(f"\nTotal de no coincidencias: {total_no_match_shown}\n")
        
        for icd10 in sorted(no_match_examples.keys()):
            examples = no_match_examples[icd10]
            print(f"\nCódigo {icd10}: {len(examples)} casos de no coincidencia")
            for pmid, rev in examples:
                print(f"  - PMID: {pmid} | REV: \"{rev}\"")
        print("=" * 70)
    
    # Estadísticas por código ICD10
    print()
    print("=" * 70)
    print("DOCUMENTOS QUE CONTIENEN CADA CÓDIGO ICD10")
    print("=" * 70)
    
    # Nombres descriptivos para cada código
    icd10_names = {
        'I10': 'Hipertensión arterial',
        'E78.5': 'Dislipemia',
        'Z87.891': 'Exfumador',
        'E11.9': 'Diabetes mellitus tipo 2',
        'F17.210': 'Fumador',
        'Z79.01': 'Anticoagulado',
        'I25.10': 'Cardiopatía isquémica',
        'Z79.82': 'AAS',
        'N17.9': 'Insuficiencia renal aguda',
        'I48.91': 'Fibrilación auricular'
    }
    
    # Ordenar por número de documentos (descendente)
    sorted_icd10 = sorted(docs_per_icd10.items(), key=lambda x: len(x[1]), reverse=True)
    
    print()
    for icd10, pmids in sorted_icd10:
        name = icd10_names.get(icd10, 'Desconocido')
        print(f"{icd10:8} | {name:35} | {len(pmids):5} documentos")
    
    # Mostrar códigos que NO aparecieron
    missing_codes = TARGET_ICD10_CODES - set(docs_per_icd10.keys())
    if missing_codes:
        print()
        print("Códigos ICD10 que NO aparecieron en ningún documento:")
        for icd10 in sorted(missing_codes):
            name = icd10_names.get(icd10, 'Desconocido')
            print(f"  - {icd10}: {name}")
    
    print("=" * 70)
    
    # Mostrar ranking de códigos ICD10 más frecuentes
    print()
    print("=" * 70)
    print("RANKING DE CÓDIGOS ICD10 MÁS FRECUENTES")
    print("=" * 70)
    
    # Calcular total por código (sumando todas las variantes de texto)
    codigo_totales = {}
    for codigo, textos_dict in icd10_freq.items():
        codigo_totales[codigo] = sum(textos_dict.values())
    
    # Ordenar por frecuencia
    codigos_ordenados = sorted(codigo_totales.items(), key=lambda x: x[1], reverse=True)
    
    # Mostrar top 30
    print(f"\nTop 30 códigos ICD10 más frecuentes:\n")
    for i, (codigo, total) in enumerate(codigos_ordenados[:30], 1):
        # Obtener el texto más frecuente para este código
        textos_dict = icd10_freq[codigo]
        texto_mas_frecuente = max(textos_dict.items(), key=lambda x: x[1])
        
        print(f"{i:2}. {codigo:8} | Frecuencia: {total:4} | Texto: \"{texto_mas_frecuente[0]}\"")
        
        # Si hay variantes, mostrarlas
        if len(textos_dict) > 1:
            otras_variantes = sorted(
                [(t, c) for t, c in textos_dict.items() if t != texto_mas_frecuente[0]], 
                key=lambda x: x[1], 
                reverse=True
            )
            for texto, count in otras_variantes[:3]:  # Máximo 3 variantes adicionales
                print(f"     {'':8}   Variante ({count:3}): \"{texto}\"")
    
    print(f"\nTotal de códigos ICD10 únicos: {len(codigo_totales)}")
    print("=" * 70)


def main():
    """Función principal"""
    # Configuración de rutas
    input_folder = "datasets/salida_episodios/salida_episodios"
    output_file = "datasets/spanish_clinical_filtered.jsonl"
    
    print("=" * 70)
    print("Filtrado de entidades diagnósticas en texto clínico español")
    print("=" * 70)
    print(f"Carpeta de entrada: {input_folder}")
    print(f"Archivo de salida: {output_file}")
    print()
    print("Criterios de filtrado:")
    print("  1. Solo entidades con sem='DIAG'")
    print("  2. Solo códigos ICD10 del top 10 más frecuentes")
    print("  3. La entidad (REV) debe aparecer literalmente en el texto")
    print("  4. Búsqueda normalizada (sin acentos, case-insensitive)")
    print()
    print("Códigos ICD10 objetivo:")
    for code in sorted(TARGET_ICD10_CODES):
        print(f"  - {code}")
    print("=" * 70)
    print()
    
    # Realizar el procesamiento
    process_folder(input_folder, output_file)


if __name__ == "__main__":
    main()
