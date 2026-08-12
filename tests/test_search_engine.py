import pytest
from unittest.mock import MagicMock
from search_engine.search_engine import SearchEngine


@pytest.fixture
def dummy_engine(tmp_path):
    """Crea una instancia ligera de SearchEngine sin cargar modelos pesados."""
    # Instanciación omitiendo __init__ pesado para probar métodos aislados
    engine = SearchEngine.__new__(SearchEngine)
    engine.max_words_per_chunk = 250 #
    engine.top_k_docs = 3 
    return engine


def test_clip_text_smartly_under_limit(dummy_engine):
    """Verifica que un texto corto no se altere."""
    text = "Este es un texto corto de prueba."
    clipped = dummy_engine._clip_text_smartly(text, max_words=250)
    assert clipped == text


def test_clip_text_smartly_over_limit_sentence_boundary(dummy_engine):
    """Verifica que si se pasa de 250 palabras, corte en el último punto dentro del límite."""
    # Crear un texto de 300 palabras con varias oraciones
    sentence_1 = " ".join(["palabra"] * 100) + "."
    sentence_2 = " ".join(["test"] * 100) + "."
    sentence_3 = " ".join(["exceso"] * 100) + "."
    full_text = f"{sentence_1} {sentence_2} {sentence_3}"

    clipped = dummy_engine._clip_text_smartly(full_text, max_words=250) 
    word_count = len(clipped.split())

    assert word_count <= 250
    assert clipped.endswith(".")
    assert "exceso" not in clipped  # La tercera oración debía quedar fuera


def test_aggregate_documents(dummy_engine):
    """Verifica que la agregación por max pooling extraiga los top 3 documentos únicos ordenados."""
    candidates = [
        {"doc_id": "DOC-A", "score": 0.85},
        {"doc_id": "DOC-B", "score": 0.95},
        {"doc_id": "DOC-A", "score": 0.90},  # Máximo de DOC-A es 0.90
        {"doc_id": "DOC-C", "score": 0.70},
        {"doc_id": "DOC-D", "score": 0.60},
    ]

    docs = dummy_engine._aggregate_documents(candidates) 

    assert len(docs) == 3
    assert docs[0] == {"rank": 1, "doc_id": "DOC-B"}  # Score 0.95
    assert docs[1] == {"rank": 2, "doc_id": "DOC-A"}  # Score 0.90
    assert docs[2] == {"rank": 3, "doc_id": "DOC-C"}  # Score 0.70


def test_rrf_fusion(dummy_engine):
    """Verifica que la fusión RRF combine correctamente dos listas de rankings."""
    rank_list_1 = [
        {"_chunk_key": "DOC-1::C1", "chunk_id": "C1", "doc_id": "DOC-1"},
        {"_chunk_key": "DOC-2::C2", "chunk_id": "C2", "doc_id": "DOC-2"},
    ]
    rank_list_2 = [
        {"_chunk_key": "DOC-2::C2", "chunk_id": "C2", "doc_id": "DOC-2"},
        {"_chunk_key": "DOC-3::C3", "chunk_id": "C3", "doc_id": "DOC-3"},
    ]

    fused = dummy_engine._rrf_fusion([rank_list_1, rank_list_2], k_rrf=60) 

    # C2 aparece en ambas listas por lo que debería tener el score RRF más alto
    assert fused[0]["doc_id"] == "DOC-2"
    assert "score" in fused[0]
