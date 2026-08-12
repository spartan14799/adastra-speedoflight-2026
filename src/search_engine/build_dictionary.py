import json
import os
import re
from pathlib import Path
from symspellpy import SymSpell


def build_global_dictionary(base_vectorial_dir: str):
    """Genera un único dictionary.txt global a partir del primer metadata.jsonl encontrado."""
    base_path = Path(base_vectorial_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"No existe el directorio: {base_vectorial_dir}")

    # Buscar el primer metadata.jsonl
    metadata_files = list(base_path.glob("**/metadata.jsonl"))
    if not metadata_files:
        raise FileNotFoundError(f"No se encontró ningún 'metadata.jsonl' en {base_vectorial_dir}")

    source_metadata = metadata_files[0]
    output_dict_path = base_path / "dictionary.txt"

    print(f"Leyendo metadata única desde: {source_metadata}")
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    count_chunks = 0
    with open(source_metadata, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            text = item.get("texto", item.get("text", ""))
            if text:
                # Extraer palabras individuales (incluye acentos y caracteres especiales)
                words = re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜâêôãçÂÊÔÃÇàÀ]+", text.lower())
                for word in words:
                    if len(word) > 1:
                        sym_spell.create_dictionary_entry(word, count=1)
                count_chunks += 1

    # Guardar el diccionario global único
    with open(output_dict_path, "w", encoding="utf-8") as f:
        for word, count in sym_spell.words.items():
            f.write(f"{word} {count}\n")

    print(f"Diccionario global creado ({count_chunks} fragmentos) en: {output_dict_path}")


if __name__ == "__main__":
    DEFAULT_BASE_VECTORIAL = "entrega/base_vectorial"
    build_global_dictionary(DEFAULT_BASE_VECTORIAL)
