import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell


class SearchEngine:
    """Motor de búsqueda semántica Multi-Encoder con RRF, deduplicación por contenido,

    filtrado de idioma/ruido y alineación estricta de resultados.
    """

    DEFAULT_ENCODERS = [
        {
            "model_name": "BAAI/bge-m3",
            "folder_name": "encoder_bge-m3",
            "prefix": "",
        },
        {
            "model_name": "intfloat/multilingual-e5-large",
            "folder_name": "encoder_e5",
            "prefix": "query: ",
        },
    ]

    def __init__(
        self,
        base_vectorial_dir: str = "entrega/base_vectorial",
        index_type: str = "hnsw",
        encoders_config: Optional[List[Dict[str, Any]]] = None,
        top_k_docs: int = 3,
        top_k_chunks: int = 10,
        max_words_per_chunk: int = 250,
        device: Optional[str] = None,
    ):
        self.base_vectorial_dir = base_vectorial_dir
        self.index_type = index_type.lower()
        self.top_k_docs = top_k_docs
        self.top_k_chunks = top_k_chunks
        self.max_words_per_chunk = max_words_per_chunk
        self.device = device

        if encoders_config is None:
            encoders_config = self.DEFAULT_ENCODERS

        self.encoders: List[Dict[str, Any]] = []
        self._sym_spell: Optional[SymSpell] = None

        for cfg in encoders_config:
            m_name = cfg["model_name"]
            folder = cfg["folder_name"]
            prefix = cfg.get("prefix", "query: " if "e5" in m_name.lower() else "")

            encoder_folder = Path(base_vectorial_dir) / folder
            index_filename = f"index.faiss.{self.index_type}"
            index_path = encoder_folder / index_filename
            metadata_path = encoder_folder / "metadata.jsonl"
            dict_path = encoder_folder / "dictionary.txt"

            if not index_path.exists():
                fallback_path = encoder_folder / "index.faiss"
                if fallback_path.exists():
                    index_path = fallback_path
                else:
                    raise FileNotFoundError(
                        f"No se encontró el índice FAISS en: {index_path}"
                    )

            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"No se encontró la metadata en: {metadata_path}"
                )

            print(f"Cargando Encoder [{m_name}] | Índice: {index_path.name}")
            encoder_model = SentenceTransformer(m_name, device=device)
            faiss_index = faiss.read_index(str(index_path))

            metadata = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        metadata.append(json.loads(line.strip()))

            self.encoders.append(
                {
                    "model": encoder_model,
                    "index": faiss_index,
                    "metadata": metadata,
                    "prefix": prefix,
                    "dict_path": str(dict_path) if dict_path.exists() else None,
                }
            )

    @property
    def sym_spell(self) -> SymSpell:
        if self._sym_spell is None:
            self._sym_spell = SymSpell(
                max_dictionary_edit_distance=2, prefix_length=7
            )
            global_dict = Path(self.base_vectorial_dir) / "dictionary.txt"
            if global_dict.exists():
                self._sym_spell.load_dictionary(
                    str(global_dict), term_index=0, count_index=1
                )
            else:
                if self.encoders and "metadata" in self.encoders[0]:
                    for item in self.encoders[0]["metadata"]:
                        text = item.get("texto", item.get("text", ""))
                        if text:
                            self._sym_spell.create_dictionary_entry(text, count=1)
        return self._sym_spell

    def preprocess_query(self, query_text: str) -> str:
        """Limpia la consulta protegiendo siglas y acrónimos para evitar deformaciones de SymSpell."""
        if not query_text or not isinstance(query_text, str):
            return ""

        text = unicodedata.normalize("NFC", query_text)
        text = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", " ", text)
        text = re.sub(r"[^\w\s\dÁÉÍÓÚáéíóúÑñÜüÃãÇçÂâÊêÔôÀà.,?!¿¡\-]", " ", text)
        cleaned = re.sub(r"\s+", " ", text).strip()

        # Si la consulta contiene acrónimos en mayúsculas (ej: CEPAL, AWS, SIPRI), evitamos SymSpell
        words = cleaned.split()
        has_acronyms = any(w.isupper() and len(w) > 1 for w in words)

        if cleaned and not has_acronyms:
            try:
                suggestions = self.sym_spell.lookup_compound(
                    cleaned, max_edit_distance=2, ignore_non_words=True
                )
                if suggestions:
                    cleaned = suggestions[0].term
            except Exception:
                pass

        return cleaned

    def _is_valid_candidate(self, text: str, item: Dict[str, Any]) -> bool:
        """Filtra fragmentos con ruido de idioma (ej: coreano) o faltos de texto."""
        if not text or not text.strip():
            return False

        # Descartar caracteres asiáticos (Hangul / Coreano) si el corpus es en ES/EN
        if re.search(r"[\uac00-\ud7a3\u3131-\u318e]", text):
            return False

        # Filtrar si el metadato de idioma está presente y es distinto de es/en
        lang = item.get("language", item.get("lang", "")).lower()
        if lang and lang not in ["es", "en", "spanish", "english"]:
            return False

        return True

    def _search_single_encoder(
        self, encoder_item: Dict[str, Any], query_text: str, k: int = 50
    ) -> List[Dict[str, Any]]:
        """Ejecuta búsqueda vectorial y estandariza los scores para que siempre 'mayor = mejor'."""
        text_to_encode = encoder_item["prefix"] + query_text
        vector = encoder_item["model"].encode([text_to_encode], convert_to_numpy=True)
        faiss.normalize_L2(vector)

        index = encoder_item["index"]
        metadata = encoder_item["metadata"]
        fetch_k = min(k, index.ntotal)

        if fetch_k == 0:
            return []

        distances, indices = index.search(vector, fetch_k)
        is_l2 = index.metric_type == faiss.METRIC_L2

        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(metadata):
                item = metadata[idx].copy()
                raw_text = item.get("texto", item.get("text", ""))

                if not self._is_valid_candidate(raw_text, item):
                    continue

                # Estandarizar score: Mayor siempre es mejor
                raw_score = float(dist)
                item["score"] = (
                    1.0 / (1.0 + raw_score) if is_l2 else raw_score
                )

                # Clave única de chunk y hash de contenido para deduplicación
                item["_chunk_key"] = f"{item['doc_id']}::{item['chunk_id']}"
                norm_text = " ".join(raw_text.lower().split())
                item["_text_hash"] = hashlib.md5(
                    norm_text.encode("utf-8")
                ).hexdigest()

                candidates.append(item)

        return candidates

    def _rrf_fusion(
        self, rank_lists: List[List[Dict[str, Any]]], k_rrf: int = 60
    ) -> List[Dict[str, Any]]:
        """Fusiona resultados mediante RRF deduplicando fragmentos con texto idéntico."""
        rrf_scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}
        text_hash_to_key: Dict[str, str] = {}

        for rank_list in rank_lists:
            for rank_idx, item in enumerate(rank_list):
                rank = rank_idx + 1
                key = item["_chunk_key"]
                text_hash = item["_text_hash"]

                # Si el mismo texto exacto ya existe bajo otro chunk_id, fusionar en la misma clave
                if text_hash in text_hash_to_key:
                    key = text_hash_to_key[text_hash]
                else:
                    text_hash_to_key[text_hash] = key

                if key not in rrf_scores:
                    rrf_scores[key] = 0.0
                    item_map[key] = item

                rrf_scores[key] += 1.0 / (k_rrf + rank)

        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
        )

        fused_candidates = []
        for key in sorted_keys:
            cand = item_map[key].copy()
            cand["score"] = float(rrf_scores[key])
            cand.pop("_chunk_key", None)
            cand.pop("_text_hash", None)
            fused_candidates.append(cand)

        return fused_candidates

    def _vector_search(self, query_text: str, k: int = 50) -> List[Dict[str, Any]]:
        """Aplica búsqueda multi-encoder y deduplica los candidatos finales."""
        if len(self.encoders) == 1:
            candidates = self._search_single_encoder(
                self.encoders[0], query_text, k
            )
            # Deduplicación por hash de texto para encoder único
            seen_hashes = set()
            dedup_candidates = []
            for c in candidates:
                th = c.pop("_text_hash", None)
                c.pop("_chunk_key", None)
                if th not in seen_hashes:
                    seen_hashes.add(th)
                    dedup_candidates.append(c)
            return dedup_candidates

        rank_lists = [
            self._search_single_encoder(enc, query_text, k)
            for enc in self.encoders
        ]
        return self._rrf_fusion(rank_lists, k_rrf=60)

    def _clip_text_smartly(self, text: str, max_words: int = 250) -> str:
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

    def _build_aligned_response(
        self, query_id: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construye 'fragments' y 'documents' garantizando alineación estricta 

        y asegurando siempre la cantidad exacta de documentos requerida (top_k_docs).
        """
        top_chunks = candidates[: self.top_k_chunks]
        fragments = []
        doc_order = []

        # 1. Procesar fragmentos y capturar sus doc_ids en orden de aparición
        for rank, cand in enumerate(top_chunks):
            doc_id = cand["doc_id"]
            if doc_id not in doc_order:
                doc_order.append(doc_id)

            raw_text = cand.get("texto", cand.get("text", ""))
            clipped_text = self._clip_text_smartly(
                raw_text, self.max_words_per_chunk
            )

            fragments.append(
                {
                    "rank": rank + 1,
                    "chunk_id": cand["chunk_id"],
                    "doc_id": doc_id,
                    "text": clipped_text,
                }
            )

        # 2. Si los Top 10 fragmentos pertenecen a menos de top_k_docs (3) documentos distintos,
        # rellenar con los doc_ids de los siguientes candidatos disponibles.
        if len(doc_order) < self.top_k_docs:
            for cand in candidates:
                doc_id = cand["doc_id"]
                if doc_id not in doc_order:
                    doc_order.append(doc_id)
                if len(doc_order) == self.top_k_docs:
                    break

        # 3. Construir la estructura final de 'documents' con exactamente top_k_docs elementos
        documents = [
            {"rank": rank + 1, "doc_id": doc_id}
            for rank, doc_id in enumerate(doc_order[: self.top_k_docs])
        ]

        return {
            "query_id": query_id,
            "documents": documents,
            "fragments": fragments,
        }

    def search(self, query_id: str, query_text: str) -> Dict[str, Any]:
        processed_query = self.preprocess_query(query_text)
        candidates = self._vector_search(processed_query, k=50)
        return self._build_aligned_response(query_id, candidates)
