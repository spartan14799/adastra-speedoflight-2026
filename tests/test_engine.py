import os
import json
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.search_engine import SearchEngine
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_search_engine_integration():
    print("🧪 Iniciando test de integración para SearchEngine...")

    # 1. Crear datos sintéticos temporales
    model_name = "BAAI/bge-m3"
    encoder = SentenceTransformer(model_name)

    chunks_mock = [
        {
            "doc_id": "DOC-001",
            "chunk_id": "DOC-001-chunk-000",
            "texto": "La inteligencia artificial en sistemas de defensa permite optimizar procesos de toma de decisiones estratégicas en tiempo real.",
        },
        {
            "doc_id": "DOC-001",
            "chunk_id": "DOC-001-chunk-001",
            "texto": "El uso de redes neuronales profundas mejora el procesamiento de imágenes satelitales.",
        },
        {
            "doc_id": "DOC-002",
            "chunk_id": "DOC-002-chunk-000",
            "texto": "La basura espacial en la órbita baja terrestre representa un peligro constante para las misiones comerciales y satélites operativos.",
        },
    ]

    # Generar embeddings sintéticos y crear índice IndexFlatIP
    texts = [c["texto"] for c in chunks_mock]
    embeddings = encoder.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 2. Guardar archivos temporales (.faiss y .jsonl)
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = os.path.join(tmp_dir, "index.faiss")
        metadata_path = os.path.join(tmp_dir, "metadata.jsonl")

        faiss.write_index(index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            for item in chunks_mock:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 3. Inicializar SearchEngine
        engine = SearchEngine(
            index_path=index_path, metadata_path=metadata_path, model_name=model_name
        )

        # 4. Probar preprocess_query con typo (SymSpell)
        query_raw = "  ¿¿Cómo   afecta la IA a la defensaaa?? \n\t "
        cleaned = engine.preprocess_query(query_raw)
        print(f"  [Preprocess] Original: '{query_raw}' -> Limpio: '{cleaned}'")
        assert "\n" not in cleaned and "  " not in cleaned

        # 5. Ejecutar búsqueda completa (End-to-End)
        query_id = "q001"
        response = engine.search(query_id, query_raw)

        print("\n📊 Respuesta del SearchEngine:")
        print(json.dumps(response, indent=2, ensure_ascii=False))

        # 6. Validaciones del esquema (Acceptance Criteria)
        assert response["query_id"] == query_id
        assert "documents" in response
        assert "fragments" in response
        assert len(response["documents"]) <= 3
        assert len(response["fragments"]) <= 10

        for frag in response["fragments"]:
            word_count = len(frag["text"].split())
            assert word_count <= 250, (
                f"El fragmento supera las 250 palabras ({word_count})"
            )

    print("\n✅ ¡Test de integración completado exitosamente!")


if __name__ == "__main__":
    test_search_engine_integration()
