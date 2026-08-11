import json
from pathlib import Path
import pytest
from transformers import AutoTokenizer

# Importación directa desde la carpeta src/
from src.chunker import ChunkingConfig, chunk_document

# Definición de rutas relativas basadas en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = (
    ROOT_DIR / "data" / "processed"
)  # Cambia 'processed' por la subcarpeta que corresponda
OUTPUT_DIR = ROOT_DIR / "data" / "metadata_example"


def test_chunking_sample_10_files():
    """
    Test para procesar una muestra de máximo 10 archivos JSON
    y generar el archivo metadata.jsonl de prueba.
    """
    # 1. Buscar recursivamente todos los archivos .json en data/ y sus subcarpetas
    todos_los_jsons = list(INPUT_DIR.rglob("*.json"))

    if not todos_los_jsons:
        pytest.skip(f"No se encontraron archivos .json en la ruta {INPUT_DIR}")

    # Seleccionar solo los primeros 10 archivos
    muestra_jsons = todos_los_jsons[:10]

    # 2. Crear el directorio de salida si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archivo_salida = OUTPUT_DIR / "metadata.jsonl"

    # 3. Configurar el Tokenizer
    encoder_name = "BAAI/bge-m3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    except Exception as e:
        print(
            f"No se pudo cargar el tokenizer {encoder_name}: {e}. Se usarán estimaciones."
        )
        tokenizer = None

    config = ChunkingConfig(max_words=250, overlap_sentences=1, tokenizer=tokenizer)
    total_chunks = 0

    # 4. Procesar la muestra
    with open(archivo_salida, "w", encoding="utf-8") as f_out:
        # Cabecera con metadatos del Encoder
        header = {
            "type": "encoder_header",
            "encoder_name": encoder_name,
            "max_words": config.max_words,
            "overlap_sentences": config.overlap_sentences,
        }
        f_out.write(json.dumps(header, ensure_ascii=False) + "\n")

        for json_path in muestra_jsons:
            with open(json_path, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)
                documentos = data if isinstance(data, list) else [data]

                for doc in documentos:
                    texto_limpio = doc.get("texto", "")
                    meta = doc.get("metadata", {})

                    doc_id = meta.get("doc_id", json_path.stem)
                    fuente = meta.get("fuente", json_path.name)
                    formato = meta.get(
                        "tipo_fuente", meta.get("formato", "pdf")
                    ).lower()
                    fenomeno = int(meta.get("Fenomeno", meta.get("fenomeno", 1)))
                    idioma = meta.get("idioma", "es")

                    chunks = chunk_document(
                        texto_limpio=texto_limpio,
                        doc_id=doc_id,
                        fuente=fuente,
                        formato=formato,
                        fenomeno=fenomeno,
                        idioma=idioma,
                        config=config,
                    )

                    for chunk in chunks:
                        f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        total_chunks += 1

    # 5. Validaciones para Pytest
    assert archivo_salida.exists(), f"El archivo {archivo_salida} no fue creado."
    assert total_chunks > 0, "No se generó ningún chunk durante el proceso."


if __name__ == "__main__":
    # Permite ejecutarlo como un script de python normal además de test
    test_chunking_sample_10_files()
    print(f"Proceso finalizado. Revisa la carpeta: {OUTPUT_DIR}")
