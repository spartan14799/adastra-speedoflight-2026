import os
import json
import re
import unicodedata
import faiss
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
        dict_path: Optional[str] = None,
        model_name: str = "BAAI/bge-m3",
        top_k_docs: int = 3,
        top_k_chunks: int = 10,
        max_words_per_chunk: int = 250,
        device: Optional[str] = None,
    ):
        self.top_k_docs = top_k_docs
        self.top_k_chunks = top_k_chunks
        self.max_words_per_chunk = max_words_per_chunk

        # 1. Determinar la ruta del diccionario (si no se pasa, busca en la misma carpeta que metadata_path)
        if dict_path is None:
            base_dir = os.path.dirname(metadata_path)
            self.dict_path = os.path.join(base_dir, "dictionary.txt")
        else:
            self.dict_path = dict_path

        # Variable privada para Carga Lazy
        self._sym_spell: Optional[SymSpell] = None

        # 2. Cargar Encoder semántico
        self.encoder = SentenceTransformer(model_name, device=device)

        # 3. Cargar índice FAISS
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Índice FAISS no encontrado en: {index_path}")
        self.index = faiss.read_index(index_path)

        # 4. Cargar almacén de metadatos (JSONL)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Archivo metadata no encontrado en: {metadata_path}"
            )

        self.metadata: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line.strip()))

    @property
    def sym_spell(self) -> SymSpell:
        """
        Propiedad Lazy: Carga SymSpell en memoria solo la primera vez que se consulta.
        Si existe el archivo 'dictionary.txt' precalculado, lo carga en milisegundos.
        """
        if self._sym_spell is None:
            self._sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

            if os.path.exists(self.dict_path):
                # Carga instantánea desde archivo en disco
                self._sym_spell.load_dictionary(
                    self.dict_path, term_index=0, count_index=1
                )
            else:
                # Fallback: construir en memoria desde la metadata si no existe el archivo
                for item in self.metadata:
                    text = item.get("texto", item.get("text", ""))
                    if text:
                        self._sym_spell.create_dictionary_entry(text, count=1)

        return self._sym_spell

    def preprocess_query(self, query_text: str) -> str:
        """
        Limpia, normaliza y corrige ortográficamente la consulta sin modelos generativos.
        """
        if not query_text or not isinstance(query_text, str):
            return ""

        # Normalización Unicode NFC
        text = unicodedata.normalize("NFC", query_text)

        # Limpieza de saltos de línea y caracteres de control
        text = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", " ", text)

        # Retener caracteres alfanuméricos y puntuación
        text = re.sub(r"[^\w\s\dÁÉÍÓÚáéíóúÑñÜüÃãÇçÂâÊêÔôÀà.,?!¿¡\-]", " ", text)

        cleaned = re.sub(r"\s+", " ", text).strip()

        # Aplicar SymSpell mediante la propiedad lazy
        if cleaned:
            try:
                suggestions = self.sym_spell.lookup_compound(
                    cleaned, max_edit_distance=2, ignore_non_words=True
                )
                if suggestions:
                    cleaned = suggestions[0].term
            except Exception:
                pass

        return cleaned

    def _vector_search(self, query_text: str, k: int = 50) -> List[Dict[str, Any]]:
        """Codifica la consulta y ejecuta búsqueda vectorial con similitud coseno."""
        vector = self.encoder.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(vector)

        fetch_k = min(k, self.index.ntotal)
        if fetch_k == 0:
            return []

        distances, indices = self.index.search(vector, fetch_k)

        candidates = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(score)
                candidates.append(item)

        return candidates

    def _aggregate_documents(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Agrupa scores por doc_id usando Max Pooling para obtener el Top 3.
        """
        doc_scores: Dict[str, float] = {}

        for cand in candidates:
            doc_id = cand["doc_id"]
            score = cand["score"]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[: self.top_k_docs]

        return [
            {"rank": rank + 1, "doc_id": doc_id}
            for rank, (doc_id, _) in enumerate(top_docs)
        ]

    def _clip_text_smartly(self, text: str, max_words: int = 250) -> str:
        """
        Garantiza completitud lingüística y límite <= 250 palabras.
        """
        words = text.split()
        if len(words) <= max_words:
            return text

        truncated_words = words[:max_words]
        raw_truncated = " ".join(truncated_words)
        # Cortar oraciones hasta puntos
        match = list(re.finditer(r"[.!?](?:\s+|$)", raw_truncated))
        if match:
            last_end_idx = match[-1].end()
            return raw_truncated[:last_end_idx].strip()

        # Fallback
        clipped = raw_truncated.strip()
        if not clipped.endswith((".", "!", "?")):
            clipped += "."
        return clipped

    def _format_fragments(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Formatea los Top 10 fragmentos asegurando <= 250 palabras por chunk."""
        top_chunks = candidates[: self.top_k_chunks]
        fragments = []

        for rank, cand in enumerate(top_chunks):
            clipped_text = self._clip_text_smartly(
                cand.get("texto", cand.get("text", "")), self.max_words_per_chunk
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
        """Procesa una consulta y devuelve la respuesta en la estructura JSON oficial."""
        processed_query = self.preprocess_query(query_text)
        candidates = self._vector_search(processed_query, k=50)

        documents = self._aggregate_documents(candidates)
        fragments = self._format_fragments(candidates)

        return {
            "query_id": query_id,
            "documents": documents,
            "fragments": fragments,
        }
