import json
from pathlib import Path
from typing import List, Dict

def create_reference_jsonl(input_file: str, output_file: str, reference_entities: List[Dict[str, str]]):
    """
    Crea un nuevo archivo JSONL donde cada registro tiene el mismo bloque de entidades de referencia.
    
    Args:
        input_file: Ruta al archivo JSONL de entrada
        output_file: Ruta al archivo JSONL de salida
        reference_entities: Lista de entidades de referencia a incluir en cada registro
    """
    
    # Estadísticas
    total_records = 0
    records_with_original_entities = 0
    
    print("Procesando archivo...")
    print("-"*60)
    
    # Procesar archivo
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Cargar registro JSON
                record = json.loads(line)
                total_records += 1
                
                # Contar registros que tenían entidades originales
                if record.get("Entidad") and len(record.get("Entidad", [])) > 0:
                    records_with_original_entities += 1
                
                # Reemplazar las entidades con el bloque de referencia
                record["Entidad"] = reference_entities
                
                # Escribir registro actualizado
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write('\n')
                
                # Mostrar progreso cada 10 registros
                if line_num % 10 == 0:
                    print(f"  Procesados {line_num} registros...")
                
            except json.JSONDecodeError as e:
                print(f"Error decodificando JSON en línea {line_num}: {e}")
            except Exception as e:
                print(f"Error procesando línea {line_num}: {e}")
    
    # Mostrar estadísticas finales
    print("\n" + "="*60)
    print("RESUMEN DEL PROCESO")
    print("="*60)
    print(f"Total de registros procesados: {total_records}")
    print(f"Registros que tenían entidades originales: {records_with_original_entities}")
    print(f"Todos los registros ahora tienen: {len(reference_entities)} entidades de referencia")
    print(f"\nArchivo de salida creado: {output_file}")

def show_sample_output(output_file: str, num_samples: int = 2):
    """
    Muestra ejemplos del archivo de salida.
    
    Args:
        output_file: Ruta al archivo JSONL de salida
        num_samples: Número de ejemplos a mostrar
    """
    print("\n" + "="*60)
    print(f"EJEMPLOS DEL ARCHIVO CON ENTIDADES DE REFERENCIA")
    print("="*60)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
                
            record = json.loads(line)
            print(f"\nRegistro {i+1} - PMID: {record.get('PMID', 'Unknown')}")
            print(f"Texto (primeros 150 caracteres): {record.get('Texto', '')[:150]}...")
            print(f"Entidades de referencia ({len(record.get('Entidad', []))}):")
            for entity in record.get("Entidad", []):
                print(f"  • {entity['texto']} ({entity['tipo']})")

def main():
    """
    Función principal del script.
    """
    # Configuración
    INPUT_FILE = "n2c2_developt.jsonl"  # Cambiar según tu archivo de entrada
    OUTPUT_FILE = "n2c2_developt_input.jsonl"  # Archivo de salida
    
    # Bloque fijo de entidades de referencia
    # Estas entidades se incluirán en TODOS los registros
    REFERENCE_ENTITIES = [
        {"texto": "Short of breath", "tipo": "SpecificDisease"},
        {"texto": "Chest pain", "tipo": "SpecificDisease"},
        {"texto": "Orthopnea", "tipo": "SpecificDisease"},
        {"texto": "Hypertension", "tipo": "SpecificDisease"},
        {"texto": "Hyperlipidemia", "tipo": "SpecificDisease"},
        {"texto": "Angina", "tipo": "SpecificDisease"},
        {"texto": "Obesity", "tipo": "SpecificDisease"},
        {"texto": "Aspirin", "tipo": "Treatment"},
        {"texto": "Nitroglycerin", "tipo": "Treatment"},
        {"texto": "Monitoring", "tipo": "Treatment"}
    ]
    
    print("="*60)
    print("CREADOR DE JSONL CON ENTIDADES DE REFERENCIA")
    print("="*60)
    print(f"Archivo de entrada: {INPUT_FILE}")
    print(f"Archivo de salida: {OUTPUT_FILE}")
    print(f"\nBloque de entidades de referencia ({len(REFERENCE_ENTITIES)} entidades):")
    print("Este bloque se incluirá en TODOS los registros:")
    for entity in REFERENCE_ENTITIES:
        print(f"  • {entity['texto']} ({entity['tipo']})")
    print()
    
    # Verificar que existe el archivo de entrada
    if not Path(INPUT_FILE).exists():
        print(f"❌ Error: No se encontró el archivo {INPUT_FILE}")
        return
    
    # Procesar archivo
    create_reference_jsonl(INPUT_FILE, OUTPUT_FILE, REFERENCE_ENTITIES)
    
    # Mostrar ejemplos del resultado
    show_sample_output(OUTPUT_FILE)
    
    print("\n✅ Proceso completado exitosamente!")
    print(f"   Todos los registros en '{OUTPUT_FILE}' ahora tienen")
    print(f"   exactamente las mismas {len(REFERENCE_ENTITIES)} entidades de referencia.")

if __name__ == "__main__":
    main()