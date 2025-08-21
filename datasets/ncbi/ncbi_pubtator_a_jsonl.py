# ncbi_pubtator_a_jsonl_v2.py
# Convierte PubTator -> JSONL (PMID, Texto, Entidad) con filtrado y mapeo de DiseaseClass
# - Elimina: CompositeMention, Modifier
# - Mantiene: SpecificDisease (sin cambios)
# - Agrupa DiseaseClass en:
#       'tumor'    -> {cancer, tumor, tumour, adenoma, adenomas, malignancy}
#       'genetica' -> {autosomal dominant disease, genetic defect, autosomal recessive disorder,
#                       autosomal recessive lysosomal storage disorder, complement deficiency,
#                       inherited disorder, allelic disorders, maternal disomy, uniparental disomy,
#                       congenital eye malformations, genetic diseases}
# - Otras DiseaseClass se descartan

import argparse
import json
import re

# Conjuntos de mapeo (minúsculas para matching case-insensitive)
DISEASECLASS_TUMOR = {
    "cancer", "tumor", "tumour", "adenoma", "adenomas", "malignancy"
}

DISEASECLASS_GENETICA = {
    "autosomal dominant disease",
    "genetic defect",
    "autosomal recessive disorder",
    "autosomal recessive lysosomal storage disorder",
    "complement deficiency",
    "inherited disorder",
    "allelic disorders",
    "maternal disomy",
    "uniparental disomy",
    "congenital eye malformations",
    "genetic diseases",
}

def normalize_text(s: str) -> str:
    # Normaliza espacios y minúsculas para matching robusto
    return re.sub(r"\s+", " ", s.strip()).lower()

def parse_pubtator_to_jsonl(in_path: str, out_path: str, dedup: bool = True):
    """
    Convierte un fichero PubTator (líneas tipo):
      PMID|t|Título
      PMID|a|Abstract
      pmid <TAB> start <TAB> end <TAB> mention <TAB> type <TAB> id
    a JSONL con: PMID, Texto, Entidad=[{texto, tipo}] aplicando las reglas indicadas.
    """
    pmid_re_t = re.compile(r"^(\d+)\|t\|(.*)$")
    pmid_re_a = re.compile(r"^(\d+)\|a\|(.*)$")

    current_pmid = None
    current_title = []
    current_abstract = []
    current_entities = []

    def transform_entity(mention: str, etype: str):
        """Aplica reglas de filtrado/mapeo y devuelve dict {texto, tipo} o None si se descarta."""
        etype_norm = etype.strip()
        if etype_norm in {"CompositeMention", "Modifier"}:
            return None  # descartar

        if etype_norm == "DiseaseClass":
            m_norm = normalize_text(mention)
            if m_norm in DISEASECLASS_TUMOR:
                return {"texto": mention, "tipo": "tumor"}
            if m_norm in DISEASECLASS_GENETICA:
                return {"texto": mention, "tipo": "genetica"}
            return None  # otras DiseaseClass se eliminan

        if etype_norm == "SpecificDisease":
            return {"texto": mention, "tipo": etype_norm}

        # Cualquier otro tipo se descarta por seguridad
        return None

    def flush(outf):
        nonlocal current_pmid, current_title, current_abstract, current_entities
        if current_pmid is None:
            return

        texto = " ".join([t.strip() for t in current_title + current_abstract if t is not None]).strip()
        if not texto:
            texto = ""

        # Deduplicar (texto, tipo)
        if dedup:
            seen = set()
            uniq = []
            for e in current_entities:
                key = (e["texto"], e["tipo"])
                if key not in seen:
                    seen.add(key)
                    uniq.append(e)
            ents = uniq
        else:
            ents = current_entities

        record = {"PMID": current_pmid, "Texto": texto, "Entidad": ents}
        outf.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Reset
        current_pmid = None
        current_title = []
        current_abstract = []
        current_entities = []

    with open(in_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as outf:
        for raw in f:
            line = raw.rstrip("\n")

            # Separador de documentos
            if not line.strip():
                flush(outf)
                continue

            # Título
            m_t = pmid_re_t.match(line)
            if m_t:
                pmid, title = m_t.groups()
                if current_pmid is not None and pmid != current_pmid:
                    flush(outf)
                current_pmid = pmid
                current_title.append(title)
                continue

            # Abstract
            m_a = pmid_re_a.match(line)
            if m_a:
                pmid, abstract = m_a.groups()
                if current_pmid is not None and pmid != current_pmid:
                    flush(outf)
                    current_pmid = pmid
                elif current_pmid is None:
                    current_pmid = pmid
                current_abstract.append(abstract)
                continue

            # Anotación (tab-separated)
            parts = line.split("\t")
            if len(parts) >= 5 and parts[0].isdigit():
                pmid = parts[0]
                if current_pmid is not None and pmid != current_pmid:
                    flush(outf)
                    current_pmid = pmid
                elif current_pmid is None:
                    current_pmid = pmid

                mention = parts[3].strip()
                etype = parts[4].strip()
                if mention:
                    transformed = transform_entity(mention, etype)
                    if transformed is not None:
                        current_entities.append(transformed)
                continue

            # Líneas inesperadas -> ignorar

        # Fin de archivo
        flush(outf)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PubTator NCBI-disease -> JSONL filtrado (sin CompositeMention/Modifier; DiseaseClass mapeado).")
    ap.add_argument("--input", required=True, help="Ruta al .txt (p. ej., NCBItestset_corpus.txt)")
    ap.add_argument("--output", required=True, help="Salida .jsonl")
    ap.add_argument("--no-dedup", action="store_true", help="No eliminar duplicados exactos de (texto, tipo)")
    args = ap.parse_args()

    parse_pubtator_to_jsonl(args.input, args.output, dedup=not args.no-dedup if hasattr(args, 'no-dedup') else not args.no_dedup)
    print(f"Listo. JSONL creado en: {args.output}")
