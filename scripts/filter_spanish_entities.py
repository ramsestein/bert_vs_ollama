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
import logging
from difflib import SequenceMatcher
from datetime import datetime

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

# Umbral de similitud para fuzzy matching (0.0 a 1.0)
FUZZY_THRESHOLD = 0.85

# Logger global
logger = None


def setup_logging(log_file: str):
    """Configura el sistema de logging."""
    global logger
    
    # Crear logger
    logger = logging.getLogger('filter_spanish_entities')
    logger.setLevel(logging.INFO)
    
    # Crear handler para archivo
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # Crear formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Añadir handler al logger
    logger.addHandler(fh)
    
    return logger


def log_print(message: str):
    """Imprime en consola y guarda en log."""
    print(message)
    if logger:
        logger.info(message)


def similarity_ratio(text1: str, text2: str) -> float:
    """Calcula la similitud entre dos textos (0.0 a 1.0)."""
    return SequenceMatcher(None, text1, text2).ratio()


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

def get_canonical_terms_by_icd10(icd10_code: str) -> List[str]:
    """
    Devuelve los términos canónicos de búsqueda para cada código ICD10.
    Estos términos se buscarán en el texto si REV no coincide exactamente.
    
    Args:
        icd10_code: Código ICD10 (ej: 'I10', 'E78.5')
        
    Returns:
        Lista de términos canónicos a buscar
    """
    # Mapeo de códigos ICD10 a términos de búsqueda
    canonical_map = {
        'I10': ['hipertensión arterial', 'hta'],
        'E78.5': ['dislipemia', 'dlp', 'dl'],
        'Z87.891': ['exfumador', 'ex fumador', 'ex-fumador', 'exfumadora', 'ex fumadora', 'ex-fumadora'],
        # Añadimos más variantes para diabetes (E11.9)
        'E11.9': [
            'diabetes mellitus', 'diabetes', 'diabetes ii', 'diabetes II', 'diabetes tipo 2',
            'dm', 'dm2', 'dm 2', 'dm-2', 'dmii', 'dm ii', 'dmii', 'd.m. ii', 'd.m.ii',
            'diabético', 'diabetico'
        ],
        # Añadimos 'tabaco' y variantes a fumador (F17.210)
        'F17.210': [
            'fumador', 'fumadora', 'tabaco', 'tabaquismo', 'tabaquismo activo',
            'habito tabaquico', 'hábito tabáquico', 'consumo de tabaco', 'habito tabáquico'
        ],
        'Z79.01': ['anticoagulado', 'anticoagulada', 'anticoagulante', 'sintrom', 'heparina'],
        'I25.10': ['cardiopatía isquémica', 'cardiopatia isquemica', 'eac'],
        'Z79.82': ['aas', 'ácido acetilsalicílico', 'acido acetilsalicilico'],
        'N17.9': ['insuficiencia renal aguda', 'ira', 'fracaso renal agudo'],
        'I48.91': ['fibrilación auricular', 'fibrilacion auricular', 'acxfa', 'ac x fa', 'acfa', 'fa']
    }
    
    return canonical_map.get(icd10_code, [])

def try_exact_match(texto_norm: str, search_term: str) -> tuple:
    """
    Intenta encontrar una coincidencia exacta (normalizada) en el texto.
    
    Returns:
        Tupla (encontrado: bool, matched_text: str or None)
    """
    term_norm = normalize_text(search_term)
    pattern = re.escape(term_norm)
    match = re.search(pattern, texto_norm)
    if match:
        return True, search_term
    return False, None

def try_fuzzy_match(texto: str, search_term: str) -> tuple:
    """
    Intenta encontrar una coincidencia fuzzy en el texto.
    
    Returns:
        Tupla (encontrado: bool, matched_text: str or None)
    """
    texto_norm = normalize_text(texto)
    term_norm = normalize_text(search_term)
    term_len = len(term_norm)
    words = texto_norm.split()
    
    # Ventanas de palabras (1 a 5)
    for window_size in range(1, min(6, len(words) + 1)):
        for i in range(len(words) - window_size + 1):
            fragment = ' '.join(words[i:i + window_size])
            
            # Solo considerar fragmentos de tamaño similar
            if abs(len(fragment) - term_len) > max(1, term_len * 0.5):
                continue
            
            # Calcular similitud
            ratio = similarity_ratio(term_norm, fragment)
            
            if ratio >= FUZZY_THRESHOLD:
                # Encontrar el texto original correspondiente
                original_words = texto.split()
                matched_original = ' '.join(original_words[i:i + window_size])
                return True, matched_original
    
    return False, None

def text_contains_entity(texto: str, entity_text: str, icd10_code: str) -> tuple:
    """
    Verifica si el texto contiene la entidad usando una estrategia de 3 fases:
    
    1. Búsqueda EXACTA de REV en el texto (normalizado, sin fuzzy)
    2. Si falla: Búsqueda de términos canónicos según ICD10 (EXACTA + FUZZY)
    3. Si falla: Búsqueda FUZZY de REV en el texto
    
    Args:
        texto: Texto completo del documento
        entity_text: Texto de la entidad (REV) a buscar
        icd10_code: Código ICD10 de la entidad
        
    Returns:
        Tupla (coincide, match_type, matched_text) donde:
        - coincide: True si hay match, False en caso contrario
        - match_type: 'exact_rev', 'exact_canonical', 'fuzzy_canonical', 'fuzzy_rev', o None
        - matched_text: El texto que coincidió o None
    """
    texto_norm = normalize_text(texto)
    
    # FASE 1: Búsqueda EXACTA de REV
    found, matched = try_exact_match(texto_norm, entity_text)
    if found:
        return True, 'exact_rev', matched
    
    # FASE 2: Búsqueda de términos canónicos según ICD10 (EXACTA + FUZZY)
    canonical_terms = get_canonical_terms_by_icd10(icd10_code)
    
    # 2.1: Intentar match EXACTO con términos canónicos
    for term in canonical_terms:
        found, matched = try_exact_match(texto_norm, term)
        if found:
            return True, 'exact_canonical', matched
    
    # 2.2: Intentar match FUZZY con términos canónicos
    for term in canonical_terms:
        found, matched = try_fuzzy_match(texto, term)
        if found:
            return True, 'fuzzy_canonical', matched
    
    # FASE 3: Búsqueda FUZZY de REV
    found, matched = try_fuzzy_match(texto, entity_text)
    if found:
        return True, 'fuzzy_rev', matched
    
    return False, None, None


def process_document(input_path: Path) -> tuple:
    """
    Procesa un documento, filtrando solo entidades DIAG con códigos ICD10 específicos.
    
    Args:
        input_path: Ruta al archivo JSON de entrada
        
    Returns:
        Tupla (documento_salida, estadisticas_doc, all_found) donde:
        - documento_salida: Dict con formato de salida o None si no hay entidades válidas
        - estadisticas_doc: Dict con estadísticas del documento
        - all_found: Bool indicando si todas las entidades target fueron encontradas
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
        'exact_rev_matches': [],
        'exact_canonical_matches': [],
        'fuzzy_canonical_matches': [],
        'fuzzy_rev_matches': [],
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
        coincide, match_type, matched_text = text_contains_entity(texto, rev, icd10)
        
        if coincide:
            entidades_filtradas.append({
                "texto": rev,
                "tipo": sem,
                "codigo": icd10
            })
            stats['icd10_found'].add(icd10)
            
            # Clasificar por tipo de match
            if match_type == 'exact_rev':
                stats['exact_rev_matches'].append({'icd10': icd10, 'rev': rev})
            elif match_type == 'exact_canonical':
                stats['exact_canonical_matches'].append({'icd10': icd10, 'rev': rev, 'matched': matched_text})
            elif match_type == 'fuzzy_canonical':
                stats['fuzzy_canonical_matches'].append({'icd10': icd10, 'rev': rev, 'matched': matched_text})
            elif match_type == 'fuzzy_rev':
                stats['fuzzy_rev_matches'].append({'icd10': icd10, 'rev': rev, 'matched': matched_text})
        else:
            stats['no_matches'].append({'icd10': icd10, 'rev': rev})
    
    # Determinar si se encontraron todas las entidades target
    all_entities_found = len(stats['no_matches']) == 0
    
    # Si no hay entidades válidas, retornar None
    if not entidades_filtradas:
        return None, stats, False
    
    # Construir el documento de salida
    output_doc = {
        "PMID": pmid,
        "Texto": texto,
        "Entidad": entidades_filtradas
    }
    
    return output_doc, stats, all_entities_found


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
    
    log_print(f"Encontrados {len(json_files)} archivos JSON")
    log_print("")
    
    # Procesar todos los documentos
    documents_with_entities = []
    documents_without_entities = []
    total_entities_before = 0
    total_entities_after = 0
    icd10_freq = {}  # Diccionario para contar frecuencias: {codigo: {texto: count}}
    
    # Estadísticas de coincidencias
    docs_per_icd10 = defaultdict(set)  # {icd10: set(PMIDs)}
    total_exact_rev = 0
    total_exact_canonical = 0
    total_fuzzy_canonical = 0
    total_fuzzy_rev = 0
    total_no_matches = 0
    
    no_match_examples = defaultdict(list)  # {icd10: [(pmid, rev), ...]}
    exact_canonical_examples = []
    fuzzy_canonical_examples = []
    fuzzy_rev_examples = []
    
    # Nuevos contadores por documento solicitados
    total_docs_all_entities_found = 0          # documentos donde TODOS los mentions_target fueron encontrados
    total_docs_detection_errors = 0           # documentos con al menos una detección fallida (parcial)
    total_docs_processing_exceptions = 0      # documentos que lanzaron excepción en el procesamiento
    
    log_print("Procesando documentos y filtrando entidades...")
    for json_file in tqdm(json_files, desc="Procesando", unit="archivo"):
        try:
            # Leer para estadísticas
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                menciones_diag = [m for m in data.get('menciones', []) if m.get('sem') == 'DIAG']
                menciones_target = [m for m in menciones_diag if m.get('ICD10', '').strip() in TARGET_ICD10_CODES]
                total_entities_before += len(menciones_target)
                per_doc_targets_count = len(menciones_target)
            
            # Procesar documento
            doc, stats, all_found = process_document(json_file)
            
            # Actualizar estadísticas
            total_exact_rev += len(stats['exact_rev_matches'])
            total_exact_canonical += len(stats['exact_canonical_matches'])
            total_fuzzy_canonical += len(stats['fuzzy_canonical_matches'])
            total_fuzzy_rev += len(stats['fuzzy_rev_matches'])
            total_no_matches += len(stats['no_matches'])
            
            # Evaluar si en este documento se encontraron todas las entidades target
            if per_doc_targets_count > 0:
                if all_found:
                    total_docs_all_entities_found += 1
                else:
                    total_docs_detection_errors += 1
            
            # SOLO agregar documentos donde se encontraron TODAS las entidades
            if doc and all_found:
                documents_with_entities.append(doc)
                total_entities_after += len(doc['Entidad'])
                
                for icd10 in stats['icd10_found']:
                    docs_per_icd10[icd10].add(json_file.stem)
                
                for entidad in doc['Entidad']:
                    codigo = entidad['codigo']
                    texto = entidad['texto'].lower()
                    
                    if codigo not in icd10_freq:
                        icd10_freq[codigo] = {}
                    if texto not in icd10_freq[codigo]:
                        icd10_freq[codigo][texto] = 0
                    icd10_freq[codigo][texto] += 1
            else:
                documents_without_entities.append(json_file.stem)
                
        except Exception as e:
            # Contabilizar excepciones de procesamiento
            total_docs_processing_exceptions += 1
            tqdm.write(f"✗ Error procesando {json_file.name}: {e}")
    
    # Mostrar estadísticas
    log_print("")
    log_print("=" * 70)
    log_print("ESTADÍSTICAS DE FILTRADO (Solo top 10 ICD10)")
    log_print("=" * 70)
    log_print(f"Total archivos procesados: {len(json_files)}")
    log_print(f"Documentos con entidades válidas: {len(documents_with_entities)}")
    log_print(f"Documentos sin entidades válidas: {len(documents_without_entities)}")
    log_print("")
    log_print(f"Total entidades target antes del filtrado: {total_entities_before}")
    log_print(f"Total entidades que aparecen en texto: {total_entities_after}")
    log_print(f"Entidades eliminadas (no aparecen): {total_entities_before - total_entities_after}")
    if total_entities_before > 0:
        log_print(f"Porcentaje retenido: {100 * total_entities_after / total_entities_before:.1f}%")
    log_print("")
    log_print("Coincidencias por tipo:")
    log_print(f"  1. REV exacto: {total_exact_rev}")
    log_print(f"  2. Término canónico exacto: {total_exact_canonical}")
    log_print(f"  3. Término canónico fuzzy: {total_fuzzy_canonical}")
    log_print(f"  4. REV fuzzy: {total_fuzzy_rev}")
    log_print(f"  Total NO coincidencias: {total_no_matches}")
    total_all_matches = total_exact_rev + total_exact_canonical + total_fuzzy_canonical + total_fuzzy_rev
    if total_all_matches > 0:
        log_print(f"  % fuzzy (total): {100 * (total_fuzzy_canonical + total_fuzzy_rev) / total_all_matches:.1f}%")
    log_print("=" * 70)
    log_print("")
    
    # Mostrar estadística nueva solicitada
    log_print("")
    log_print("=" * 70)
    log_print("ESTADÍSTICAS ADICIONALES DE DETECCIÓN POR DOCUMENTO")
    log_print("=" * 70)
    log_print(f"Documentos (con al menos 1 mention_target) donde SE ENCONTRARON TODAS las entidades: {total_docs_all_entities_found}")
    log_print(f"Documentos (con al menos 1 mention_target) donde HUBO ERRORES en la detección (alguna entidad no encontrada): {total_docs_detection_errors}")
    log_print(f"Documentos que lanzaron EXCEPCIONES durante el procesamiento: {total_docs_processing_exceptions}")
    log_print("=" * 70)
    
    # Guardar el archivo de salida
    log_print(f"💾 Guardando resultado: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents_with_entities:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    log_print(f"✓ Archivo guardado: {len(documents_with_entities)} documentos")
    log_print(f"✓ Total entidades en archivo final: {total_entities_after}")
    
    # Mostrar ejemplos de términos canónicos exactos
    if exact_canonical_examples:
        log_print("")
        log_print("=" * 70)
        log_print("🎯 CASOS: TÉRMINO CANÓNICO EXACTO (primeros 20)")
        log_print("=" * 70)
        for pmid, icd10, rev, matched in exact_canonical_examples[:20]:
            log_print(f"PMID: {pmid} | Código: {icd10}")
            log_print(f"  REV: \"{rev}\" → Encontrado: \"{matched}\"")
            log_print("")
        log_print(f"Total: {len(exact_canonical_examples)} casos")
        log_print("=" * 70)
    
    # Mostrar ejemplos de términos canónicos fuzzy
    if fuzzy_canonical_examples:
        log_print("")
        log_print("=" * 70)
        log_print("🔍 CASOS: TÉRMINO CANÓNICO FUZZY (primeros 20)")
        log_print("=" * 70)
        for pmid, icd10, rev, matched in fuzzy_canonical_examples[:20]:
            log_print(f"PMID: {pmid} | Código: {icd10}")
            log_print(f"  REV: \"{rev}\" → Encontrado: \"{matched}\"")
            log_print("")
        log_print(f"Total: {len(fuzzy_canonical_examples)} casos")
        log_print("=" * 70)
    
    # Mostrar ejemplos de REV fuzzy
    if fuzzy_rev_examples:
        log_print("")
        log_print("=" * 70)
        log_print("🔍 CASOS: REV FUZZY (primeros 20)")
        log_print("=" * 70)
        for pmid, icd10, rev, matched in fuzzy_rev_examples[:20]:
            log_print(f"PMID: {pmid} | Código: {icd10}")
            log_print(f"  REV: \"{rev}\" → Encontrado: \"{matched}\"")
            log_print("")
        log_print(f"Total: {len(fuzzy_rev_examples)} casos")
        log_print("=" * 70)
    
    # Mostrar casos de NO coincidencia (primeros 50 por código)
    if no_match_examples:
        log_print("")
        log_print("=" * 70)
        log_print("⚠️  CASOS SIN COINCIDENCIA (primeros 50 por código)")
        log_print("=" * 70)
        total_no_match_shown = sum(len(examples) for examples in no_match_examples.values())
        log_print(f"\nTotal de no coincidencias: {total_no_match_shown}\n")
        
        for icd10 in sorted(no_match_examples.keys()):
            examples = no_match_examples[icd10]
            log_print(f"\nCódigo {icd10}: {len(examples)} casos")
            for pmid, rev in examples[:50]:
                log_print(f"  - PMID: {pmid} | REV: \"{rev}\"")
            if len(examples) > 50:
                log_print(f"  ... y {len(examples) - 50} más")
        log_print("=" * 70)
    
    # Estadísticas por código ICD10
    log_print("")
    log_print("=" * 70)
    log_print("DOCUMENTOS QUE CONTIENEN CADA CÓDIGO ICD10")
    log_print("=" * 70)
    
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
    
    log_print("")
    for icd10, pmids in sorted_icd10:
        name = icd10_names.get(icd10, 'Desconocido')
        log_print(f"{icd10:8} | {name:35} | {len(pmids):5} documentos")
    
    # Mostrar códigos que NO aparecieron
    missing_codes = TARGET_ICD10_CODES - set(docs_per_icd10.keys())
    if missing_codes:
        log_print("")
        log_print("Códigos ICD10 que NO aparecieron en ningún documento:")
        for icd10 in sorted(missing_codes):
            name = icd10_names.get(icd10, 'Desconocido')
            log_print(f"  - {icd10}: {name}")
    
    log_print("=" * 70)
    
    # Mostrar ranking de códigos ICD10 más frecuentes
    log_print("")
    log_print("=" * 70)
    log_print("RANKING DE CÓDIGOS ICD10 MÁS FRECUENTES")
    log_print("=" * 70)
    
    # Calcular total por código (sumando todas las variantes de texto)
    codigo_totales = {}
    for codigo, textos_dict in icd10_freq.items():
        codigo_totales[codigo] = sum(textos_dict.values())
    
    # Ordenar por frecuencia
    codigos_ordenados = sorted(codigo_totales.items(), key=lambda x: x[1], reverse=True)
    
    # Mostrar top 30
    log_print(f"\nTop 30 códigos ICD10 más frecuentes:\n")
    for i, (codigo, total) in enumerate(codigos_ordenados[:30], 1):
        # Obtener el texto más frecuente para este código
        textos_dict = icd10_freq[codigo]
        texto_mas_frecuente = max(textos_dict.items(), key=lambda x: x[1])
        
        log_print(f"{i:2}. {codigo:8} | Frecuencia: {total:4} | Texto: \"{texto_mas_frecuente[0]}\"")
        
        # Si hay variantes, mostrarlas
        if len(textos_dict) > 1:
            otras_variantes = sorted(
                [(t, c) for t, c in textos_dict.items() if t != texto_mas_frecuente[0]], 
                key=lambda x: x[1], 
                reverse=True
            )
            for texto, count in otras_variantes[:3]:  # Máximo 3 variantes adicionales
                log_print(f"     {'':8}   Variante ({count:3}): \"{texto}\"")
    
    log_print(f"\nTotal de códigos ICD10 únicos: {len(codigo_totales)}")
    log_print("=" * 70)


def main():
    """Función principal"""
    # Configuración de rutas
    input_folder = "datasets/salida_episodios/salida_episodios"
    output_file = "datasets/spanish_clinical_filtered.jsonl"
    
    # Configurar logging con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"filter_spanish_entities_{timestamp}.log"
    setup_logging(log_file)
    
    log_print("=" * 70)
    log_print("Filtrado de entidades diagnósticas en texto clínico español")
    log_print("=" * 70)
    log_print(f"Carpeta de entrada: {input_folder}")
    log_print(f"Archivo de salida: {output_file}")
    log_print(f"Archivo de log: {log_file}")
    log_print("")
    log_print("Criterios de filtrado:")
    log_print("  1. Solo entidades con sem='DIAG'")
    log_print("  2. Solo códigos ICD10 del top 10 más frecuentes")
    log_print("  3. La entidad (REV) debe aparecer literalmente en el texto")
    log_print("  4. Búsqueda normalizada (sin acentos, case-insensitive)")
    log_print(f"  5. Fuzzy matching con umbral de similitud: {FUZZY_THRESHOLD}")
    log_print("")
    log_print("Códigos ICD10 objetivo:")
    for code in sorted(TARGET_ICD10_CODES):
        log_print(f"  - {code}")
    log_print("=" * 70)
    log_print("")
    
    # Realizar el procesamiento
    process_folder(input_folder, output_file)
    
    log_print("")
    log_print(f"✓ Proceso completado. Log guardado en: {log_file}")


if __name__ == "__main__":
    main()
