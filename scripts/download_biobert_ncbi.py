# download_biobert_ncbi.py
# Descarga y guarda en local un BioBERT fine-tuneado para NER (NCBI-disease)
# No evalúa, solo deja el modelo y tokenizer listos para uso offline.

from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

# 1) Checkpoint BioBERT fine-tuneado para enfermedades (NCBI + BC5CDR)
MODEL_NAME = "kamalkraj/bluebert-ncbi-disease"
MODEL_NAME2 = "Ishan0612/biobert-ner-disease-ncbi"


# 2) Carpeta de salida local (cámbiala si quieres)
SAVE_DIR = "./bluebert_ncbi_ner"
SAVE_DIR2 = "./biobert_ner_ishan"


def main():
#    os.makedirs(SAVE_DIR, exist_ok=True)
#
#    print(f"Descargando tokenizer y modelo desde: {MODEL_NAME}")
#    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
#
#    print(f"Guardando en: {SAVE_DIR}")
#    tokenizer.save_pretrained(SAVE_DIR)
#    model.save_pretrained(SAVE_DIR)
#
#    print("\nListo ✅")
#    print(f"Puedes cargarlo offline así:")
#    print(f"from transformers import AutoTokenizer, AutoModelForTokenClassification")
#    print(f"tok = AutoTokenizer.from_pretrained('{SAVE_DIR}')")
#    print(f"mdl = AutoModelForTokenClassification.from_pretrained('{SAVE_DIR}')")


    os.makedirs(SAVE_DIR2, exist_ok=True)

    print(f"Descargando tokenizer y modelo desde: {MODEL_NAME2}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME2)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME2)

    print(f"Guardando en: {SAVE_DIR2}")
    tokenizer.save_pretrained(SAVE_DIR2)
    model.save_pretrained(SAVE_DIR2)

    print("\nListo ✅")
    print(f"Puedes cargarlo offline así:")
    print(f"from transformers import AutoTokenizer, AutoModelForTokenClassification")
    print(f"tok = AutoTokenizer.from_pretrained('{SAVE_DIR2}')")
    print(f"mdl = AutoModelForTokenClassification.from_pretrained('{SAVE_DIR2}')")

if __name__ == '__main__':
    main()
