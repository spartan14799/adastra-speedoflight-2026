import json
import time
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer

from src.chunker.core import ChunkingConfig, chunk_document

# ---------------------------------------------------------------------
# Cálculo Dinámico de Rutas
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
SRC_DIR = CURRENT_DIR.parent

CONFIG_PATH = SRC_DIR / "config.json"
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_OUTPUT_BASE = ROOT_DIR / "entrega" / "base_vectorial"


def cargar_configuracion(config_path: Path = CONFIG_PATH) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generar_metadata_por_encoder(
    encoder_info: Dict[str, str],
    json_files: List[Path],
    input_base_dir: Path,
    output_base_dir: Path,
    max_words: int = 250,
    overlap_sentences: int = 1,
    log_interval: int = 25,
) -> None:
    model_name = encoder_info["model_name"]
    folder_name = encoder_info["folder_name"]

    target_dir = output_base_dir / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = target_dir / "metadata.jsonl"

    print("=" * 70)
    print(f"[INICIO] Procesando Encoder: {model_name}")
    print(f"[RUTAS] Archivo destino: {output_filepath.relative_to(ROOT_DIR)}")
    print("=" * 70)

    try:
        print(f"[INFO] Cargando Tokenizer ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[INFO] Tokenizer cargado exitosamente.")
    except Exception as e:
        print(
            f"[ADVERTENCIA] No se pudo cargar el tokenizer {model_name}: {e}. Se usará estimación."
        )
        tokenizer = None

    config = ChunkingConfig(
        max_words=max_words,
        overlap_sentences=overlap_sentences,
        tokenizer=tokenizer,
    )

    total_files = len(json_files)
    total_chunks = 0
    archivos_procesados = 0
    archivos_omitidos = 0

    start_time = time.time()
    last_heartbeat_time = start_time

    with open(output_filepath, "w", encoding="utf-8") as f_out:
        header = {
            "type": "encoder_header",
            "encoder_name": model_name,
            "max_words": max_words,
            "overlap_sentences": overlap_sentences,
        }
        f_out.write(json.dumps(header, ensure_ascii=False) + "\n")

        for idx, json_path in enumerate(json_files, 1):
            current_time = time.time()

            if (
                idx == 1
                or idx % log_interval == 0
                or (current_time - last_heartbeat_time) > 10
            ):
                elapsed = current_time - start_time
                pct = (idx / total_files) * 100
                rel_path = json_path.relative_to(input_base_dir)
                print(
                    f"[PROGRESO] [{idx}/{total_files}] ({pct:.1f}%) | "
                    f"Tiempo transcurrido: {elapsed:.1f}s | "
                    f"Procesando: {rel_path}"
                )
                last_heartbeat_time = current_time

            try:
                with open(json_path, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
            except (json.JSONDecodeError, UnicodeDecodeError):
                archivos_omitidos += 1
                continue

            documentos = data if isinstance(data, list) else [data]

            for doc in documentos:
                texto_limpio = doc.get("texto", "")
                meta = doc.get("metadata", {})

                if not texto_limpio.strip():
                    continue

                doc_id = meta.get("doc_id", json_path.stem)
                fuente = meta.get("fuente", json_path.name)
                formato = meta.get("tipo_fuente", meta.get("formato", "pdf")).lower()

                try:
                    fenomeno = int(meta.get("Fenomeno", meta.get("fenomeno", 1)))
                except (ValueError, TypeError):
                    fenomeno = 1

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

            archivos_procesados += 1

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"[COMPLETADO] Finalizado {folder_name} en {total_time:.2f}s:")
    print(f"   - Archivos JSON procesados: {archivos_procesados}/{total_files}")
    if archivos_omitidos > 0:
        print(f"   - Archivos omitidos (LFS/corruptos): {archivos_omitidos}")
    print(f"   - Chunks totales generados: {total_chunks}")
    print("=" * 70 + "\n")


def run_pipeline_build_metadata(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_base_dir: Path = DEFAULT_OUTPUT_BASE,
    config_path: Path = CONFIG_PATH,
) -> None:
    cfg = cargar_configuracion(config_path)
    max_words = cfg.get("max_words", 250)
    overlap_sentences = cfg.get("overlap_sentences", 1)
    log_interval = cfg.get("log_interval", 25)
    encoders = cfg.get("encoders", [])

    if not encoders:
        print("[ADVERTENCIA] No hay encoders definidos en config.json.")
        return

    if not input_dir.exists():
        raise FileNotFoundError(f"El directorio de entrada no existe: {input_dir}")

    print(f"[BUSQUEDA] Escaneando archivos JSON en: {input_dir}")
    json_files = sorted(list(input_dir.rglob("*.json")))
    print(f"[INFO] Archivos .json encontrados: {len(json_files)}")

    if not json_files:
        print("[ADVERTENCIA] No se encontraron archivos para procesar.")
        return

    for encoder_info in encoders:
        generar_metadata_por_encoder(
            encoder_info=encoder_info,
            json_files=json_files,
            input_base_dir=input_dir,
            output_base_dir=output_base_dir,
            max_words=max_words,
            overlap_sentences=overlap_sentences,
            log_interval=log_interval,
        )


if __name__ == "__main__":
    run_pipeline_build_metadata()
