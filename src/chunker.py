import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

# ---------------------------------------------------------------------
# 1. Configuración y Recursos Lingüísticos (NLTK)
# ---------------------------------------------------------------------

NLTK_LANGUAGE_MAP = {"es": "spanish", "en": "english", "pt": "portuguese"}

ABREVIATURAS_DOMINIO = {
    "art",
    "num",
    "pag",
    "cap",
    "vol",
    "ee.uu",
    "p.ej",
    "cf",
    "res",
    "dec",
    "ing",
    "lic",
}


def _ensure_nltk_resources() -> None:
    """Garantiza la presencia del tokenizador Punkt."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


_ensure_nltk_resources()


def _merge_spurious_splits(sentences: List[str]) -> List[str]:
    """Fusiona oraciones cortadas por abreviaturas o iniciales."""
    if not sentences:
        return sentences

    merged: List[str] = []
    buffer = ""
    for sent in sentences:
        buffer = f"{buffer} {sent}".strip() if buffer else sent
        last_token = re.split(r"\s+", buffer)[-1].rstrip(".").lower()

        es_abreviatura = last_token in ABREVIATURAS_DOMINIO
        es_inicial = re.fullmatch(r"[a-záéíóúñ]", last_token) is not None

        if es_abreviatura or es_inicial:
            continue

        merged.append(buffer)
        buffer = ""

    if buffer:
        merged.append(buffer)

    return merged


def split_into_sentences(paragraph: str, idioma: str = "es") -> List[str]:
    """Divide un párrafo en oraciones respetando la completitud lingüística."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    nltk_lang = NLTK_LANGUAGE_MAP.get(idioma, "spanish")

    try:
        raw_sentences = sent_tokenize(paragraph, language=nltk_lang)
    except LookupError:
        _ensure_nltk_resources()
        raw_sentences = sent_tokenize(paragraph, language=nltk_lang)

    return _merge_spurious_splits(raw_sentences)


def split_into_paragraphs(text: str) -> List[str]:
    """Separa el texto por saltos de párrafo."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def count_words(text: str) -> int:
    """Cuenta el número de palabras en un texto."""
    return len(text.split())


# ---------------------------------------------------------------------
# 2. Configuración y Lógica Principales del Chunking
# ---------------------------------------------------------------------


@dataclass
class ChunkingConfig:
    max_words: int = 250  # Límite duro del reto (250 palabras)
    overlap_sentences: int = 1  # Solapamiento de oraciones
    tokenizer: Optional[object] = None


def count_tokens_real(text: str, tokenizer=None) -> int:
    """Calcula la cantidad real de tokens usando el tokenizer del Encoder."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, round(count_words(text) * 1.3))


def chunk_document(
    texto_limpio: str,
    doc_id: str,
    fuente: str,
    formato: str,
    fenomeno: int,
    idioma: str,
    config: ChunkingConfig = ChunkingConfig(),
) -> List[Dict]:
    """Fragmenta un texto garantizando completitud lingüística y límite de palabras."""
    if formato not in {"pdf", "html", "md"}:
        formato = "pdf"  # Fallback predeterminado si viene otro tipo de formato

    if idioma not in NLTK_LANGUAGE_MAP:
        idioma = "es"

    all_sentences: List[str] = []
    for paragraph in split_into_paragraphs(texto_limpio):
        all_sentences.extend(split_into_sentences(paragraph, idioma=idioma))

    n = len(all_sentences)
    if n == 0:
        return []

    chunks: List[Dict] = []
    posicion = 0
    chunk_start = 0

    while chunk_start < n:
        current_words = 0
        chunk_end = chunk_start

        while chunk_end < n:
            sentence_words = count_words(all_sentences[chunk_end])

            if sentence_words > config.max_words and chunk_end == chunk_start:
                chunk_end += 1
                break

            if current_words + sentence_words > config.max_words:
                break

            current_words += sentence_words
            chunk_end += 1

        if chunk_end == chunk_start:
            chunk_end = chunk_start + 1

        chunk_sentences = all_sentences[chunk_start:chunk_end]
        chunks.append(
            _build_chunk(
                chunk_sentences, doc_id, fuente, formato, fenomeno, posicion, config
            )
        )
        posicion += 1

        if chunk_end >= n:
            break

        next_start = chunk_end - config.overlap_sentences
        chunk_start = max(next_start, chunk_start + 1)

    return chunks


def _build_chunk(
    sentences, doc_id, fuente, formato, fenomeno, posicion, config
) -> Dict:
    texto = " ".join(sentences).strip()
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}-chunk-{posicion:03d}",
        "fuente": fuente,
        "formato": formato,
        "fenomeno": fenomeno,
        "posicion": posicion,
        "num_tokens": count_tokens_real(texto, config.tokenizer),
        "texto": texto,
    }


# ---------------------------------------------------------------------
# 3. Función Principal: Procesamiento de JSON y exportación a metadata.jsonl
# ---------------------------------------------------------------------


def procesar_json_a_metadata(
    json_entrada_path: str,
    output_metadata_path: str,
    encoder_name: str = "BAAI/bge-m3",
    max_words: int = 250,
    overlap_sentences: int = 1,
) -> None:
    """
    Lee los documentos en formato JSON, aplica el chunkeo con el tokenizer del encoder
    y genera el archivo metadata.jsonl con la cabecera en la primera línea.
    """
    print(f"Cargando Tokenizer de: {encoder_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    except Exception as e:
        print(
            f"Error al cargar el tokenizer '{encoder_name}': {e}. Se usará la estimación por palabras."
        )
        tokenizer = None

    config = ChunkingConfig(
        max_words=max_words, overlap_sentences=overlap_sentences, tokenizer=tokenizer
    )

    print(f"Leyendo archivo de entrada: {json_entrada_path}...")
    with open(json_entrada_path, "r", encoding="utf-8") as f:
        documentos = json.load(f)

    print(f"Generando {output_metadata_path}...")
    with open(output_metadata_path, "w", encoding="utf-8") as f_out:
        # Información del Encoder para el generador de FAISS
        header_info = {
            "type": "encoder_header",
            "encoder_name": encoder_name,
            "max_words": max_words,
            "overlap_sentences": overlap_sentences,
        }
        f_out.write(json.dumps(header_info, ensure_ascii=False) + "\n")

        # Chunks procesados
        total_chunks = 0
        for doc in documentos:
            texto_limpio = doc.get("texto", "")
            meta = doc.get("metadata", {})

            # Mapeo de campos requeridos
            doc_id = meta.get("doc_id", "DOC-000")
            fuente = meta.get("fuente", "desconocido.pdf")
            formato = meta.get("tipo_fuente", meta.get("formato", "pdf")).lower()

            # Obtención del entero del fenómeno
            fenomeno_raw = meta.get("Fenomeno", meta.get("fenomeno", 1))
            fenomeno = int(fenomeno_raw)

            idioma = meta.get("idioma", "es")

            # Generar chunks del documento actual
            chunks = chunk_document(
                texto_limpio=texto_limpio,
                doc_id=doc_id,
                fuente=fuente,
                formato=formato,
                fenomeno=fenomeno,
                idioma=idioma,
                config=config,
            )

            # Escribir registros en formato JSON Lines
            for chunk in chunks:
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1
