import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

def read_bio_file(filepath: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Lee un archivo BIO y extrae el texto completo y las entidades.
    
    Args:
        filepath: Ruta al archivo BIO
        
    Returns:
        Tupla con (texto_completo, lista_de_entidades)
    """
    tokens = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Si la línea no está vacía
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    tokens.append(token)
                    labels.append(label)
    
    # Reconstruir el texto completo
    texto_completo = ' '.join(tokens)
    
    # Extraer entidades
    entidades = extract_entities(tokens, labels)
    
    return texto_completo, entidades

def extract_entities(tokens: List[str], labels: List[str]) -> List[Dict[str, str]]:
    """
    Extrae las entidades del formato BIO.
    
    Args:
        tokens: Lista de tokens
        labels: Lista de etiquetas BIO
        
    Returns:
        Lista de diccionarios con las entidades
    """
    entidades = []
    current_entity = []
    current_type = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            # Si hay una entidad en progreso, guardarla
            if current_entity:
                entidades.append({
                    "texto": ' '.join(current_entity),
                    "tipo": current_type
                })
            # Comenzar nueva entidad
            current_type = label[2:]  # Quitar "B-"
            current_entity = [token]
            
        elif label.startswith('I-'):
            # Continuar la entidad actual
            if current_entity and label[2:] == current_type:
                current_entity.append(token)
            else:
                # Caso raro: I- sin B- previo o tipo diferente
                if current_entity:
                    entidades.append({
                        "texto": ' '.join(current_entity),
                        "tipo": current_type
                    })
                current_type = label[2:]
                current_entity = [token]
                
        else:  # label == 'O'
            # Si hay una entidad en progreso, guardarla
            if current_entity:
                entidades.append({
                    "texto": ' '.join(current_entity),
                    "tipo": current_type
                })
                current_entity = []
                current_type = None
    
    # Guardar la última entidad si existe
    if current_entity:
        entidades.append({
            "texto": ' '.join(current_entity),
            "tipo": current_type
        })
    
    return entidades

def normalize_entity_type(tipo: str) -> str:
    """
    Normaliza los tipos de entidad según el formato esperado.
    Puedes modificar esta función según tus necesidades.
    
    Args:
        tipo: Tipo de entidad del formato BIO
        
    Returns:
        Tipo normalizado
    """
    # Mapeo de tipos si es necesario
    # Por ejemplo, si quieres convertir "problem" a "SpecificDisease"
    type_mapping = {
        "problem": "SpecificDisease",
        "treatment": "Treatment",
        "test": "Test"
    }
    
    return type_mapping.get(tipo, tipo)

def process_bio_files(input_folder: str, output_file: str = "output.jsonl"):
    """
    Procesa todos los archivos .bio en una carpeta y genera un archivo JSONL.
    
    Args:
        input_folder: Carpeta con los archivos .bio
        output_file: Nombre del archivo JSONL de salida
    """
    bio_files = list(Path(input_folder).glob("*.bio"))
    
    if not bio_files:
        print(f"No se encontraron archivos .bio en {input_folder}")
        return
    
    print(f"Procesando {len(bio_files)} archivos .bio...")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, bio_file in enumerate(bio_files, 1):
            print(f"Procesando {bio_file.name}...")
            
            try:
                # Leer archivo BIO
                texto, entidades = read_bio_file(bio_file)
                
                # Normalizar tipos de entidad si es necesario
                for entidad in entidades:
                    entidad["tipo"] = normalize_entity_type(entidad["tipo"])
                
                # Crear objeto JSON
                # Usar el nombre del archivo sin extensión como PMID
                # o puedes modificar esto según tus necesidades
                pmid = bio_file.stem
                
                json_obj = {
                    "PMID": pmid,
                    "Texto": texto,
                    "Entidad": entidades
                }
                
                # Escribir al archivo JSONL
                json.dump(json_obj, outfile, ensure_ascii=False)
                outfile.write('\n')
                
                print(f"  ✓ {bio_file.name} procesado ({len(entidades)} entidades encontradas)")
                
            except Exception as e:
                print(f"  ✗ Error procesando {bio_file.name}: {e}")
    
    print(f"\n✅ Proceso completado. Archivo generado: {output_file}")

def main():
    """
    Función principal del script.
    """
    # Configuración
    INPUT_FOLDER = "./data"  # Carpeta actual, cambiar según necesidad
    OUTPUT_FILE = "n2c2.jsonl"
    
    # Procesar archivos
    process_bio_files(INPUT_FOLDER, OUTPUT_FILE)
    
    # Mostrar ejemplo del resultado
    print("\nEjemplo del primer registro convertido:")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            data = json.loads(first_line)
            print(json.dumps(data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()