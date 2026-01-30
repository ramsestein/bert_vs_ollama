#!/usr/bin/env python3
"""
Genera un archivo JSONL de entrada donde cada documento contiene todas las variantes
textuales de las entidades objetivo (para usar como input limpio de evaluación).

Salida por defecto: datasets/spanish_clinical_filtered_clean_input.jsonl
Entrada por defecto: datasets/spanish_clinical_filtered_clean.jsonl
"""

import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse

# Mapa de entidades -> variantes (textos)
ENTITIES = {
    "I10": ["hta", "hipertensión arterial", "hipertensión"],
    "E78.5": ["dislipemia", "dlp"],
    "Z87.891": ["exfumador"],
    "E11.9": ["dm2", "diabetes mellitus tipo 2", "diabetes mellitus", "dm"],
    "F17.210": ["fumador", "tabaquismo"],
    "Z79.01": ["anticoagulado", "anticoagulante", "sintrom"],
    "I25.10": ["cardiopatía isquémica", "enfermedad coronaria", "eac"],
    "Z79.82": ["aas", "aspirina", "adiro"],
    "N17.9": ["insuficiencia renal aguda", "ira", "aki"],
    "I48.91": ["fibrilación auricular", "fa", "acxfa"]
}

# Generar la lista plana de variantes (texto, tipo)
def generate_all_variants():
    variants = []
    for code, texts in ENTITIES.items():
        for t in texts:
            variants.append({"texto": t, "tipo": "DIAG"})
    return variants


def process(input_file: str, output_file: str):
    in_path = Path(input_file)
    out_path = Path(output_file)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    variants = generate_all_variants()

    with in_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Generando input variants"):
            if not line.strip():
                continue
            doc = json.loads(line)
            # Keep PM ID and Texto fields if present, and replace/insert Entidad
            pmid = doc.get('PMID') or doc.get('pmid') or doc.get('Id') or doc.get('id')
            texto = doc.get('Texto', doc.get('texto', ''))
            out_doc = {
                "PMID": pmid if pmid else "",
                "Texto": texto,
                "Entidad": variants
            }
            fout.write(json.dumps(out_doc, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} from {in_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create input JSONL with all entity variants')
    parser.add_argument('--input', default='datasets/spanish_clinical_filtered_clean.jsonl', help='Input cleaned JSONL')
    parser.add_argument('--output', default='datasets/spanish_clinical_filtered_clean_input.jsonl', help='Output JSONL to generate')
    args = parser.parse_args()

    process(args.input, args.output)
