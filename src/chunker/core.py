import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import nltk
from nltk.tokenize import sent_tokenize

# ---------------------------------------------------------------------
# Configuración y Recursos Lingüísticos (NLTK)
# ---------------------------------------------------------------------

NLTK_LANGUAGE_MAP = {"es": "spanish", "en": "english", "pt": "portuguese"}

ABREVIATURAS_DOMINIO = {
    "art", "num", "pag", "cap", "vol", "ee.uu", "p.ej", "cf",
    "res", "dec", "ing", "lic",
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
# Configuración y Lógica Principales del Chunking
# ---------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    max_words: int = 250
    overlap_sentences: int = 1
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
        formato = "pdf"

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
