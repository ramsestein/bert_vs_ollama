import glob
import json
import os
import re
from collections import defaultdict, Counter


def read_entities_from_file(filepath):
    """Read all entities from a result file with detailed information"""
    entities_info = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                pmid = data.get("PMID", "unknown")
                ents = data.get("Entidad", [])
                
                for ent in ents:
                    entity_text = ent.get("texto", "").strip()
                    confidence = ent.get("confidence", 0.0)
                    strategies = ent.get("strategies", [])
                    
                    if entity_text:
                        entities_info.append({
                            "pmid": pmid,
                            "entity": entity_text,
                            "confidence": confidence,
                            "strategies": strategies,
                            "source_file": os.path.basename(filepath)
                        })
    except Exception as e:
        print(f"[ERROR] Reading {filepath}: {e}")
    
    return entities_info


def parse_cfg_from_name(name):
    """Extract chunk_target and overlap from filename"""
    m = re.search(r"chunk(\d+)_ov(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def analyze_entity_differences():
    """Analyze differences between entities found by different models and configurations"""
    
    # Find all result files
    llama_files = sorted(glob.glob("results_llama_chunk*_ov*.jsonl"))
    qwen_files = sorted(glob.glob("results_qwen_chunk*_ov*.jsonl"))
    
    if not llama_files and not qwen_files:
        print("❌ No se encontraron archivos de resultados")
        return
    
    print("🔬 ANÁLISIS DE DIFERENCIAS ENTRE ENTIDADES ENCONTRADAS")
    print("=" * 70)
    
    # Collect all entities from both models
    all_llama_entities = []
    all_qwen_entities = []
    
    # Process llama files
    if llama_files:
        print(f"\n📁 Procesando {len(llama_files)} archivos de llama3.2:3b...")
        for fp in llama_files:
            entities = read_entities_from_file(fp)
            all_llama_entities.extend(entities)
            print(f"   {os.path.basename(fp)}: {len(entities)} entidades")
    
    # Process qwen files
    if qwen_files:
        print(f"\n📁 Procesando {len(qwen_files)} archivos de qwen2.5:3b...")
        for fp in qwen_files:
            entities = read_entities_from_file(fp)
            all_qwen_entities.extend(entities)
            print(f"   {os.path.basename(fp)}: {len(entities)} entidades")
    
    # Analyze entity differences
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   Total entidades llama3.2:3b: {len(all_llama_entities)}")
    print(f"   Total entidades qwen2.5:3b: {len(all_qwen_entities)}")
    
    # Extract unique entities by model
    llama_unique = set(ent["entity"].lower() for ent in all_llama_entities)
    qwen_unique = set(ent["entity"].lower() for ent in all_qwen_entities)
    
    print(f"   Entidades únicas llama3.2:3b: {len(llama_unique)}")
    print(f"   Entidades únicas qwen2.5:3b: {len(qwen_unique)}")
    
    # Find common and unique entities
    common_entities = llama_unique.intersection(qwen_unique)
    llama_only = llama_unique - qwen_unique
    qwen_only = qwen_unique - llama_unique
    
    print(f"   Entidades comunes: {len(common_entities)}")
    print(f"   Solo llama3.2:3b: {len(llama_only)}")
    print(f"   Solo qwen2.5:3b: {len(qwen_only)}")
    
    # Detailed analysis of common entities
    print(f"\n🔍 ANÁLISIS DE ENTIDADES COMUNES:")
    if common_entities:
        print(f"   Entidades detectadas por ambos modelos:")
        for entity in sorted(common_entities):
            llama_count = sum(1 for e in all_llama_entities if e["entity"].lower() == entity)
            qwen_count = sum(1 for e in all_qwen_entities if e["entity"].lower() == entity)
            print(f"     - {entity}: llama({llama_count}x) vs qwen({qwen_count}x)")
    
    # Analysis of llama-only entities
    print(f"\n🥇 ENTIDADES SOLO DETECTADAS POR llama3.2:3b:")
    if llama_only:
        for entity in sorted(llama_only):
            # Find all instances of this entity
            instances = [e for e in all_llama_entities if e["entity"].lower() == entity]
            confidences = [e["confidence"] for e in instances]
            strategies = [s for e in instances for s in e["strategies"]]
            
            print(f"   - {entity}:")
            print(f"     Instancias: {len(instances)}")
            print(f"     Confianza promedio: {sum(confidences)/len(confidences):.3f}")
            print(f"     Estrategias: {Counter(strategies)}")
    else:
        print("   Ninguna entidad única")
    
    # Analysis of qwen-only entities
    print(f"\n🥈 ENTIDADES SOLO DETECTADAS POR qwen2.5:3b:")
    if qwen_only:
        for entity in sorted(qwen_only):
            instances = [e for e in all_qwen_entities if e["entity"].lower() == entity]
            confidences = [e["confidence"] for e in instances]
            strategies = [s for e in instances for s in e["strategies"]]
            
            print(f"   - {entity}:")
            print(f"     Instancias: {len(instances)}")
            print(f"     Confianza promedio: {sum(confidences)/len(confidences):.3f}")
            print(f"     Estrategias: {Counter(strategies)}")
    else:
        print("   Ninguna entidad única")
    
    # Analyze by chunk configuration
    print(f"\n🔧 ANÁLISIS POR CONFIGURACIÓN DE CHUNKS:")
    
    # Group llama entities by configuration
    llama_by_config = defaultdict(list)
    for ent in all_llama_entities:
        chunk, ov = parse_cfg_from_name(ent["source_file"])
        if chunk is not None:
            config_key = f"chunk{chunk}_ov{ov}"
            llama_by_config[config_key].append(ent)
    
    # Group qwen entities by configuration
    qwen_by_config = defaultdict(list)
    for ent in all_qwen_entities:
        chunk, ov = parse_cfg_from_name(ent["source_file"])
        if chunk is not None:
            config_key = f"chunk{chunk}_ov{ov}"
            qwen_by_config[config_key].append(ent)
    
    # Compare configurations
    all_configs = set(llama_by_config.keys()) | set(qwen_by_config.keys())
    
    print(f"   Comparación por configuración:")
    for config in sorted(all_configs):
        llama_count = len(llama_by_config.get(config, []))
        qwen_count = len(qwen_by_config.get(config, []))
        llama_unique = len(set(e["entity"].lower() for e in llama_by_config.get(config, [])))
        qwen_unique = len(set(e["entity"].lower() for e in qwen_by_config.get(config, [])))
        
        print(f"     {config}:")
        print(f"       llama3.2:3b: {llama_count} entidades ({llama_unique} únicas)")
        print(f"       qwen2.5:3b: {qwen_count} entidades ({qwen_unique} únicas)")
        print(f"       Diferencia: {llama_count - qwen_count:+d} entidades")
    
    # Quality analysis
    print(f"\n📈 ANÁLISIS DE CALIDAD:")
    
    # Average confidence by model
    llama_confidences = [e["confidence"] for e in all_llama_entities]
    qwen_confidences = [e["confidence"] for e in all_qwen_entities]
    
    if llama_confidences:
        llama_avg_conf = sum(llama_confidences) / len(llama_confidences)
        print(f"   Confianza promedio llama3.2:3b: {llama_avg_conf:.3f}")
    
    if qwen_confidences:
        qwen_avg_conf = sum(qwen_confidences) / len(qwen_confidences)
        print(f"   Confianza promedio qwen2.5:3b: {qwen_avg_conf:.3f}")
    
    # Strategy distribution
    print(f"\n🎯 DISTRIBUCIÓN DE ESTRATEGIAS:")
    
    llama_strategies = Counter()
    for ent in all_llama_entities:
        for strategy in ent["strategies"]:
            llama_strategies[strategy] += 1
    
    qwen_strategies = Counter()
    for ent in all_qwen_entities:
        for strategy in ent["strategies"]:
            qwen_strategies[strategy] += 1
    
    print(f"   llama3.2:3b:")
    for strategy, count in llama_strategies.most_common():
        print(f"     {strategy}: {count} entidades")
    
    print(f"   qwen2.5:3b:")
    for strategy, count in qwen_strategies.most_common():
        print(f"     {strategy}: {count} entidades")
    
    print(f"\n✅ Análisis de diferencias completado!")


if __name__ == "__main__":
    analyze_entity_differences()
