import json
from pathlib import Path
from typing import List, Dict

def filter_entities(input_file: str, output_file: str, allowed_entities: List[str]):
    """
    Filtra las entidades en un archivo JSONL manteniendo solo las especificadas.
    
    Args:
        input_file: Ruta al archivo JSONL de entrada
        output_file: Ruta al archivo JSONL de salida
        allowed_entities: Lista de entidades permitidas
    """
    
    # Convertir a minúsculas para comparación insensible a mayúsculas
    allowed_entities_lower = [entity.lower() for entity in allowed_entities]
    
    # Estadísticas
    total_records = 0
    total_entities_before = 0
    total_entities_after = 0
    
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
                
                # Contar entidades originales
                original_entities = record.get("Entidad", [])
                total_entities_before += len(original_entities)
                
                # Filtrar entidades
                filtered_entities = []
                for entity in original_entities:
                    entity_text = entity.get("texto", "")
                    entity_type = entity.get("tipo", "")
                    
                    # Comparar insensible a mayúsculas
                    if entity_text.lower() in allowed_entities_lower:
                        filtered_entities.append({
                            "texto": entity_text,
                            "tipo": entity_type
                        })
                
                # Actualizar registro con entidades filtradas
                record["Entidad"] = filtered_entities
                total_entities_after += len(filtered_entities)
                
                # Escribir registro actualizado
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write('\n')
                
                # Mostrar progreso para registros con entidades filtradas
                if filtered_entities:
                    print(f"Registro {line_num} (PMID: {record.get('PMID', 'Unknown')}): "
                          f"{len(original_entities)} → {len(filtered_entities)} entidades")
                
            except json.JSONDecodeError as e:
                print(f"Error decodificando JSON en línea {line_num}: {e}")
            except Exception as e:
                print(f"Error procesando línea {line_num}: {e}")
    
    # Mostrar estadísticas finales
    print("\n" + "="*60)
    print("RESUMEN DE FILTRADO")
    print("="*60)
    print(f"Registros procesados: {total_records}")
    print(f"Entidades originales: {total_entities_before}")
    print(f"Entidades después del filtrado: {total_entities_after}")
    print(f"Entidades eliminadas: {total_entities_before - total_entities_after}")
    print(f"Porcentaje retenido: {(total_entities_after/total_entities_before)*100:.1f}%")
    print(f"\nArchivo de salida: {output_file}")

def show_sample_output(output_file: str, num_samples: int = 3):
    """
    Muestra ejemplos del archivo de salida.
    
    Args:
        output_file: Ruta al archivo JSONL de salida
        num_samples: Número de ejemplos a mostrar
    """
    print("\n" + "="*60)
    print(f"EJEMPLOS DEL ARCHIVO FILTRADO (primeros {num_samples} con entidades)")
    print("="*60)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        samples_shown = 0
        for line in f:
            if samples_shown >= num_samples:
                break
                
            record = json.loads(line)
            if record.get("Entidad"):  # Solo mostrar si tiene entidades
                samples_shown += 1
                print(f"\nPMID: {record.get('PMID', 'Unknown')}")
                print(f"Texto (primeros 200 caracteres): {record.get('Texto', '')[:200]}...")
                print(f"Entidades encontradas ({len(record.get('Entidad', []))}):")
                for entity in record.get("Entidad", []):
                    print(f"  - {entity['texto']} ({entity['tipo']})")

def main():
    """
    Función principal del script.
    """
    # Configuración
    INPUT_FILE = "n2c2.jsonl"  # Cambiar según tu archivo de entrada
    OUTPUT_FILE = "n2c2_developt.jsonl"  # Archivo de salida
    
    # Lista de entidades permitidas (exactamente como las especificaste)
    ALLOWED_ENTITIES = [
        "Short of breath",
        "Chest pain", 
        "Orthopnea",
        "Hypertension",
        "Hyperlipidemia",
        "Angina",
        "Obesity",
        "Aspirin",
        "Nitroglycerin",
        "Monitoring"
    ]
    
    print("="*60)
    print("FILTRADOR DE ENTIDADES MÉDICAS")
    print("="*60)
    print(f"Archivo de entrada: {INPUT_FILE}")
    print(f"Archivo de salida: {OUTPUT_FILE}")
    print(f"\nEntidades a mantener ({len(ALLOWED_ENTITIES)}):")
    for entity in ALLOWED_ENTITIES:
        print(f"  • {entity}")
    print()
    
    # Verificar que existe el archivo de entrada
    if not Path(INPUT_FILE).exists():
        print(f"❌ Error: No se encontró el archivo {INPUT_FILE}")
        return
    
    # Procesar archivo
    print("Procesando archivo...")
    print("-"*60)
    filter_entities(INPUT_FILE, OUTPUT_FILE, ALLOWED_ENTITIES)
    
    # Mostrar ejemplos del resultado
    show_sample_output(OUTPUT_FILE)
    
    print("\n✅ Proceso completado exitosamente!")

if __name__ == "__main__":
    main()