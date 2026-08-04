import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell


class SearchEngine:
    """Motor de búsqueda semántica basado en FAISS y Encoders densos con soporte Multi-Encoder y RRF."""

    def __init__(
        self,
        metadata_path: str,
        index_path: Optional[str] = None,
        dict_path: Optional[str] = None,
        model_name: str = "BAAI/bge-m3",
        encoders_config: Optional[List[Dict[str, Any]]] = None,
        top_k_docs: int = 3,
        top_k_chunks: int = 10,
        max_words_per_chunk: int = 250,
        device: Optional[str] = None,
    ):
        self.top_k_docs = top_k_docs
        self.top_k_chunks = top_k_chunks
        self.max_words_per_chunk = max_words_per_chunk

        # 1. Determinar la ruta del diccionario para SymSpell
        if dict_path is None:
            base_dir = os.path.dirname(metadata_path)
            self.dict_path = os.path.join(base_dir, "dictionary.txt")
        else:
            self.dict_path = dict_path

        self._sym_spell: Optional[SymSpell] = None

        # 2. Cargar almacén único de metadatos (JSONL)
        # Se asume que todos los índices FAISS coinciden exactamente fila por fila con este archivo
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Archivo metadata no encontrado en: {metadata_path}"
            )

        self.metadata: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line.strip()))

        # 3. Configurar Encoders e Índices FAISS (Un solo encoder vs Multi-Encoder)
        if encoders_config is None:
            if index_path is None:
                raise ValueError("Debe proporcionar 'encoders_config' o 'index_path'.")
            encoders_config = [
                {
                    "model_name": model_name,
                    "index_path": index_path,
                    "prefix": "",
                }
            ]

        self.encoders: List[Dict[str, Any]] = []
        for cfg in encoders_config:
            m_name = cfg["model_name"]
            i_path = cfg["index_path"]
            prefix = cfg.get("prefix", "")

            if not os.path.exists(i_path):
                raise FileNotFoundError(f"Índice FAISS no encontrado en: {i_path}")

            encoder_model = SentenceTransformer(m_name, device=device)
            faiss_index = faiss.read_index(i_path)

            self.encoders.append(
                {
                    "model": encoder_model,
                    "index": faiss_index,
                    "prefix": prefix,
                }
            )

    @property
    def sym_spell(self) -> SymSpell:
        """Propiedad Lazy: Carga SymSpell en memoria solo la primera vez que se consulta."""
        if self._sym_spell is None:
            self._sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

            if os.path.exists(self.dict_path):
                self._sym_spell.load_dictionary(
                    self.dict_path, term_index=0, count_index=1
                )
            else:
                for item in self.metadata:
                    text = item.get("texto", item.get("text", ""))
                    if text:
                        self._sym_spell.create_dictionary_entry(text, count=1)

        return self._sym_spell

    def preprocess_query(self, query_text: str) -> str:
        """Limpia, normaliza y corrige ortográfica la consulta."""
        if not query_text or not isinstance(query_text, str):
            return ""

        text = unicodedata.normalize("NFC", query_text)
        text = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", " ", text)
        text = re.sub(r"[^\w\s\dÁÉÍÓÚáéíóúÑñÜüÃãÇçÂâÊêÔôÀà.,?!¿¡\-]", " ", text)
        cleaned = re.sub(r"\s+", " ", text).strip()

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

    def _search_single_encoder(
        self, encoder_item: Dict[str, Any], query_text: str, k: int = 50
    ) -> List[Dict[str, Any]]:
        """Ejecuta la búsqueda semántica en un único par (encoder, faiss_index)."""
        text_to_encode = encoder_item["prefix"] + query_text
        vector = encoder_item["model"].encode([text_to_encode], convert_to_numpy=True)
        faiss.normalize_L2(vector)

        index = encoder_item["index"]
        fetch_k = min(k, index.ntotal)
        if fetch_k == 0:
            return []

        distances, indices = index.search(vector, fetch_k)

        candidates = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(score)
                item["_faiss_idx"] = int(idx)  # ID interno único para fusionar
                candidates.append(item)

        return candidates

    def _rrf_fusion(
        self, rank_lists: List[List[Dict[str, Any]]], k_rrf: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combina rankings de múltiples encoders usando Reciprocal Rank Fusion (RRF).
        Fórmula: RRF(d) = sum_i 1 / (k_rrf + rank_i(d))
        """
        rrf_scores: Dict[int, float] = {}
        item_map: Dict[int, Dict[str, Any]] = {}

        for rank_list in rank_lists:
            for rank_idx, item in enumerate(rank_list):
                rank = rank_idx + 1  # Rangos basados en 1 (1-based index)
                faiss_idx = item["_faiss_idx"]

                if faiss_idx not in rrf_scores:
                    rrf_scores[faiss_idx] = 0.0
                    item_map[faiss_idx] = item

                rrf_scores[faiss_idx] += 1.0 / (k_rrf + rank)

        # Ordenar los fragmentos candidatos por su puntaje RRF descendente
        sorted_indices = sorted(
            rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True
        )

        fused_candidates = []
        for idx in sorted_indices:
            cand = item_map[idx].copy()
            cand["score"] = float(rrf_scores[idx])
            cand.pop("_faiss_idx", None)  # Limpiamos la clave interna de control
            fused_candidates.append(cand)

        return fused_candidates

    def _vector_search(self, query_text: str, k: int = 50) -> List[Dict[str, Any]]:
        """Ejecuta búsqueda vectorial directa o fusión RRF según la cantidad de encoders."""
        if len(self.encoders) == 1:
            candidates = self._search_single_encoder(self.encoders[0], query_text, k)
            for cand in candidates:
                cand.pop("_faiss_idx", None)
            return candidates

        # Múltiples encoders: Recuperación independiente y Fusión RRF
        rank_lists = [
            self._search_single_encoder(enc, query_text, k) for enc in self.encoders
        ]
        return self._rrf_fusion(rank_lists, k_rrf=60)

    def _aggregate_documents(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Agrupa scores por doc_id usando Max Pooling para obtener el Top 3."""
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
        """Garantiza completitud lingüística y límite <= 250 palabras."""
        words = text.split()
        if len(words) <= max_words:
            return text

        truncated_words = words[:max_words]
        raw_truncated = " ".join(truncated_words)

        match = list(re.finditer(r"[.!?](?:\s+|$)", raw_truncated))
        if match:
            last_end_idx = match[-1].end()
            return raw_truncated[:last_end_idx].strip()

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
        """Procesa una consulta y devuelve la respuesta en el formato JSON oficial."""
        processed_query = self.preprocess_query(query_text)
        candidates = self._vector_search(processed_query, k=50)

        documents = self._aggregate_documents(candidates)
        fragments = self._format_fragments(candidates)

        return {
            "query_id": query_id,
            "documents": documents,
            "fragments": fragments,
        }
