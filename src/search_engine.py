import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


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

    def preprocess_query(self, query_text: str) -> str:
        # TODO: Crear la funcion que preprocese la query
        """
        Limpia y optimiza la consulta antes de vectorizarla.
        """
        cleaned = query_text.strip()
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
