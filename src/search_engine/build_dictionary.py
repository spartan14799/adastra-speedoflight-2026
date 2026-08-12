import json
import os
import re

from symspellpy import SymSpell


def build_dictionary_from_metadata(metadata_path: str, output_dict_path: str):
    """Lee el archivo metadata.jsonl y genera un archivo dictionary.txt optimizado para SymSpell."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de metadata en: {metadata_path}"
        )

    print(f"Leyendo metadata desde {metadata_path}...")
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    count_chunks = 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            text = item.get("texto", item.get("text", ""))
            if text:
                # Extraer palabras individuales (incluye caracteres en español como ñ, á, é, etc.)
                words = re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+", text.lower())
                for word in words:
                    if len(word) > 1:
                        sym_spell.create_dictionary_entry(word, count=1)
                count_chunks += 1

    # Asegurar que el directorio de salida exista
    os.makedirs(os.path.dirname(output_dict_path), exist_ok=True)

    print(
        f"Guardando diccionario precalculado ({count_chunks} fragmentos procesados)..."
    )

    # Escribir las palabras y sus frecuencias en formato 'palabra frecuencia'
    with open(output_dict_path, "w", encoding="utf-8") as f:
        for word, count in sym_spell.words.items():
            f.write(f"{word} {count}\n")

    print(f"Diccionario guardado exitosamente en: {output_dict_path}")


if __name__ == "__main__":
    DEFAULT_METADATA = "entrega/base_vectorial/encoder_modelo/metadata.jsonl"
    DEFAULT_OUTPUT = "entrega/base_vectorial/encoder_modelo/dictionary.txt"

    build_dictionary_from_metadata(DEFAULT_METADATA, DEFAULT_OUTPUT)
