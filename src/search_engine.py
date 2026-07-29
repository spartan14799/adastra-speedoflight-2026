import os
import json
import re
import faiss
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell


class SearchEngine:
    """Motor de búsqueda semántica basado en FAISS y Encoders densos."""

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: str = "BAAI/bge-m3",
        top_k_docs: int = 3,
        top_k_chunks: int = 10,
        max_words_per_chunk: int = 250,
    ):
        self.top_k_docs = top_k_docs
        self.top_k_chunks = top_k_chunks
        self.max_words_per_chunk = max_words_per_chunk

        # Cargar Encoder
        self.encoder = SentenceTransformer(model_name)

        # Cargar índice FAISS
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Índice FAISS no encontrado en: {index_path}")
        self.index = faiss.read_index(index_path)

        # Cargar almacenamiento de Metadatos
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Archivo metadata no encontrado en: {metadata_path}"
            )

        self.metadata: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self._build_corpus_dictionary()

    def _build_corpus_dictionary(self):
        """
        Crea el diccionario de SymSpell usando el texto real de los documentos.
        """
        for item in self.metadata:
            text = item.get("texto", "")
            if text:
                self.sym_spell.create_dictionary_entry(text, count=1)

    def preprocess_query(self, query_text: str) -> str:
        """
        Limpia, normaliza y corrige ortográficamente la consulta sin usar LLMs.
        1. Normalización Unicode (NFC) y limpieza de caracteres de control.
        2. Preservación de puntuación básica en español.
        3. Colapso de espacios múltiples.
        4. Corrección ortográfica basada en el vocabulario del corpus con SymSpell.
        """
        if not query_text or not isinstance(query_text, str):
            return ""

        # Normalización Unicode (NFC)
        text = unicodedata.normalize("NFC", query_text)

        # Limpieza de saltos de línea, pestañas y caracteres de control
        text = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", " ", text)

        # Retener caracteres alfanuméricos y puntuación estándar
        text = re.sub(r"[^\w\s\dÁÉÍÓÚáéíóúÑñÜü.,?!¿¡\-]", " ", text)

        # Colapsar espacios múltiples
        cleaned = re.sub(r"\s+", " ", text).strip()

        # Corrección ortográfica con SymSpell
        if cleaned and hasattr(self, "sym_spell"):
            try:
                # lookup_compound procesa frases completas con múltiples typos
                suggestions = self.sym_spell.lookup_compound(
                    cleaned, max_edit_distance=2, ignore_non_words=True
                )
                if suggestions:
                    cleaned = suggestions[0].term
            except Exception:
                # Fallback de seguridad
                pass

        return cleaned

    def _vector_search(self, query_text: str, k: int = 50) -> List[Dict[str, Any]]:
        """Codifica la consulta y ejecuta la búsqueda por producto interno en FAISS."""
        # Generar embedding y normalizar para Similitud Coseno
        vector = self.encoder.encode([query_text])
        faiss.normalize_L2(vector)

        # Limitar K si hay menos elementos en el índice
        fetch_k = min(k, self.index.ntotal)
        distances, index = self.index.search(vector, fetch_k)

        candidates = []
        for idx, score in zip(index[0], distances[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(score)
                candidates.append(item)

        return candidates

    def _aggregate_documents(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Agrupa los scores de los chunks por doc_id usando Max Pooling
        para seleccionar los Top 3 documentos más relevantes.
        """
        doc_scores: Dict[str, float] = {}

        for cand in candidates:
            doc_id = cand["doc_id"]
            score = cand["score"]
            # Max Pooling: Asignar la mayor puntuación de chunk al documento
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        # Ordenar documentos por score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[: self.top_k_docs]

        return [
            {"rank": rank + 1, "doc_id": doc_id}
            for rank, (doc_id, _) in enumerate(top_docs)
        ]

    # TODO: arreglar y mejorar este metodo, revisar como se puede mejorar desde el chunk
    def _clip_text_smartly(self, text: str, max_words: int = 250) -> str:
        """
        Evita recortar ideas cuando el chunk supera las 250 palabras, asi no corta la idea, sino se devuelve a el ultimo punto
        """

        words = text.split()
        if len(words) <= max_words:
            return text

        # Toma un bloque de palabras un poco menor al límite
        truncated_words = words[:max_words]
        raw_truncated = " ".join(truncated_words)

        # Busca el último punto final '.'
        last_punct = max(
            raw_truncated.rfind("."), raw_truncated.rfind("?"), raw_truncated.rfind("!")
        )

        if last_punct != -1:
            # Corta donde termina la última oración completa
            return raw_truncated[: last_punct + 1]

        # Devolvemos recorte crudo
        return raw_truncated

    def _format_fragments(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Selecciona los Top 10 fragmentos y garantiza que ninguno supere el límite máximo de palabras permitido.
        """
        top_chunks = candidates[: self.top_k_chunks]
        fragments = []

        for rank, cand in enumerate(top_chunks):
            # Usamos el recorte
            clipped_text = self._clip_text_smartly(
                cand["texto"], self.max_words_per_chunk
            )

            fragments.append(
                {
                    "rank": rank + 1,
                    "chunk_id": cand["chunk_id"],
                    "doc_id": cand["doc_id"],
                    "text": clipped_text,
                }
            )

        return fragments

    def search(self, query_id: str, query_text: str) -> Dict[str, Any]:
        """Procesa una consulta completa y devuelve el formato listo para JSONL."""
        # Limpiar y procesar consulta
        processed_query = self.preprocess_query(query_text)

        # Recuperar candidatos crudos de FAISS (Top 50)
        candidates = self._vector_search(processed_query, k=50)

        # Extraer Top 3 Documentos
        documents = self._aggregate_documents(candidates)

        # Extraer Top 10 Fragmentos
        fragments = self._format_fragments(candidates)

        # Formato de salida estandarizado
        return {"query_id": query_id, "documents": documents, "fragments": fragments}
