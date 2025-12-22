#!/usr/bin/env python3
"""
Script para convertir documentos clínicos en español al formato JSONL esperado.

Formato de entrada (salida_episodios):
{
  "texto": "...",
  "menciones": [
    {"sem": "DIAG", "ICD10": "...", "ORIG": "...", "REV": "..."},
    ...
  ]
}

Formato de salida:
- Para cada conjunto (train, validation, test) se generan dos archivos:
  * _input.jsonl: Sin entidades (para procesamiento)
  * .jsonl: Con entidades (benchmark/ground truth)

{
  "PMID": "nombre_archivo",
  "Texto": "...",
  "Entidad": [
    {"texto": "REV", "tipo": "sem"},
    ...
  ]
}
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def convert_document(input_path: Path) -> Dict:
    """
    Convierte un documento del formato de entrada al formato de salida.
    
    Args:
        input_path: Ruta al archivo JSON de entrada
        
    Returns:
        Diccionario con el formato de salida
    """
    # Leer el archivo de entrada
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer el nombre del archivo sin extensión para usar como PMID
    pmid = input_path.stem
    
    # Convertir las menciones al formato de entidades
    # Solo incluir entidades cuyo ICD10 empieza por letra (diagnósticos)
    entidades = []
    for mencion in data.get('menciones', []):
        # Solo agregar si tiene el campo REV, sem e ICD10
        rev = mencion.get('REV', '').strip()
        sem = mencion.get('sem', '').strip()
        icd10 = mencion.get('ICD10', '').strip()
        
        # Filtrar: solo incluir entidades cuyo ICD10 empieza por letra
        if rev and sem and icd10 and icd10[0].isalpha():
            entidades.append({
                "texto": rev,
                "tipo": sem
            })
    
    # Construir el documento de salida
    output_doc = {
        "PMID": pmid,
        "Texto": data.get('texto', ''),
        "Entidad": entidades
    }
    
    return output_doc


def split_dataset(documents: List[Dict], train_size: int = 100, 
                  validation_size: int = 100, test_size: int = 100) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Divide los documentos en conjuntos de train, validation y test.
    
    Args:
        documents: Lista de documentos
        train_size: Número de documentos para entrenamiento (por defecto 100)
        validation_size: Número de documentos para validación (por defecto 100)
        test_size: Número de documentos para prueba (por defecto 100)
        
    Returns:
        Tupla con (train_docs, validation_docs, test_docs)
    """
    # Mezclar aleatoriamente
    random.shuffle(documents)
    
    # Verificar que hay suficientes documentos
    total_needed = train_size + validation_size + test_size
    if len(documents) < total_needed:
        raise ValueError(f"No hay suficientes documentos. Se necesitan {total_needed} pero solo hay {len(documents)}")
    
    # Dividir en conjuntos de tamaño fijo
    train_docs = documents[:train_size]
    validation_docs = documents[train_size:train_size + validation_size]
    test_docs = documents[train_size + validation_size:train_size + validation_size + test_size]
    
    return train_docs, validation_docs, test_docs


def write_dataset_files(documents: List[Dict], output_prefix: str, split_name: str):
    """
    Escribe dos archivos JSONL para un conjunto de datos:
    - _input.jsonl: Con entidades (para procesamiento y evaluación)
    - .jsonl: Con entidades (benchmark/ground truth)
    
    Args:
        documents: Lista de documentos a escribir
        output_prefix: Prefijo para los archivos de salida
        split_name: Nombre del conjunto (train, validation, test)
    """
    # Archivo con entidades (benchmark)
    benchmark_file = f"{output_prefix}_{split_name}.jsonl"
    # Archivo con entidades (input)
    input_file = f"{output_prefix}_{split_name}_input.jsonl"
    
    # Escribir archivo benchmark (con entidades)
    with open(benchmark_file, 'w', encoding='utf-8') as f_benchmark:
        for doc in documents:
            f_benchmark.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Escribir archivo input (con entidades también)
    with open(input_file, 'w', encoding='utf-8') as f_input:
        for doc in documents:
            # Ahora mantener las entidades
            doc_input = {
                "PMID": doc["PMID"],
                "Texto": doc["Texto"],
                "Entidad": doc["Entidad"]
            }
            f_input.write(json.dumps(doc_input, ensure_ascii=False) + '\n')
    
    print(f"✓ {split_name.capitalize()}: {len(documents)} documentos")
    print(f"  - Benchmark: {benchmark_file}")
    print(f"  - Input: {input_file}")


def convert_folder(input_folder: str, output_prefix: str, seed: int = 42):
    """
    Convierte todos los archivos JSON de una carpeta y los divide en train/validation/test.
    
    Args:
        input_folder: Carpeta con los archivos JSON de entrada
        output_prefix: Prefijo para los archivos de salida
        seed: Semilla para la división aleatoria
    """
    # Establecer semilla para reproducibilidad
    random.seed(seed)
    
    input_path = Path(input_folder)
    
    # Verificar que la carpeta existe
    if not input_path.exists():
        raise FileNotFoundError(f"La carpeta {input_folder} no existe")
    
    # Archivo completo con todos los documentos
    all_docs_file = Path(input_folder).parent / "spanish_clinical_all.jsonl"
    
    # Si el archivo completo ya existe, leerlo en lugar de procesar de nuevo
    if all_docs_file.exists():
        print(f"📄 Archivo completo encontrado: {all_docs_file}")
        print("Leyendo documentos...")
        documents = []
        with open(all_docs_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                # Las entidades ya están filtradas en convert_document
                documents.append(doc)
        print(f"✓ {len(documents)} documentos cargados (entidades con ICD10 que empieza por letra)")
        print()
    else:
        # Obtener todos los archivos JSON
        json_files = sorted(input_path.glob('*.json'))
        
        if not json_files:
            print(f"No se encontraron archivos JSON en {input_folder}")
            return
        
        print(f"Encontrados {len(json_files)} archivos JSON")
        print()
        
        # Convertir todos los documentos
        documents = []
        converted_count = 0
        
        print("Convirtiendo documentos...")
        for json_file in tqdm(json_files, desc="Procesando", unit="archivo"):
            try:
                doc = convert_document(json_file)
                documents.append(doc)
                converted_count += 1
            except Exception as e:
                tqdm.write(f"✗ Error procesando {json_file.name}: {e}")
        
        print(f"\n✓ {converted_count}/{len(json_files)} archivos convertidos exitosamente")
        print()
        
        # Guardar el archivo completo
        print(f"💾 Guardando archivo completo: {all_docs_file}")
        with open(all_docs_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        print(f"✓ Archivo completo guardado ({len(documents)} documentos)")
        print()
    
    # Dividir en train, validation y test
    train_docs, validation_docs, test_docs = split_dataset(documents)
    
    # Escribir los archivos
    write_dataset_files(train_docs, output_prefix, "train")
    write_dataset_files(validation_docs, output_prefix, "validation")
    write_dataset_files(test_docs, output_prefix, "test")


def main():
    """Función principal"""
    # Configuración de rutas
    input_folder = "datasets/salida_episodios/salida_episodios"
    output_prefix = "datasets/spanish_clinical"
    
    print("=" * 70)
    print("Conversión de documentos clínicos en español")
    print("=" * 70)
    print(f"Carpeta de entrada: {input_folder}")
    print(f"Prefijo de salida: {output_prefix}")
    print()
    print("Se generarán 6 archivos con 100 documentos cada uno:")
    print("  - spanish_clinical_train.jsonl (100 docs con entidades)")
    print("  - spanish_clinical_train_input.jsonl (100 docs con entidades)")
    print("  - spanish_clinical_validation.jsonl (100 docs con entidades)")
    print("  - spanish_clinical_validation_input.jsonl (100 docs con entidades)")
    print("  - spanish_clinical_test.jsonl (100 docs con entidades)")
    print("  - spanish_clinical_test_input.jsonl (100 docs con entidades)")
    print("=" * 70)
    print()
    
    # Realizar la conversión
    convert_folder(input_folder, output_prefix)


if __name__ == "__main__":
    main()
