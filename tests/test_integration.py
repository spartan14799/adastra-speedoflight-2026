import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from search_engine.build_dictionary import build_global_dictionary
from search_engine.search_engine import SearchEngine


def test_full_pipeline():
    base_dir = "entrega/base_vectorial"

    print("=== 1. Probando Generación de Diccionario Global ===")
    build_global_dictionary(base_dir) #
    dict_path = Path(base_dir) / "dictionary.txt"
    assert dict_path.exists(), "❌ El archivo dictionary.txt no fue creado."
    print(f"✅ Diccionario generado correctamente en {dict_path}")

    print("\n=== 2. Cargando SearchEngine con Índice HNSW ===")
    # Puedes cambiar index_type a 'flat' o 'ivfflat' para probar las variantes
    engine = SearchEngine(
        base_vectorial_dir=base_dir,
        index_type="hnsw", #
        device="cpu"  # Usa 'cuda' si tienes GPU disponible
    )

    print("\n=== 3. Ejecutando Búsqueda de Prueba ===")
    test_query_id = "q001"
    test_query_text = "Inteligencia artificial aplicada a la defensa y entorno militar"

    response = engine.search(test_query_id, test_query_text) #

    print("\n=== 4. Validando Esquema Oficial del Resultado ===")
    # Validar claves principales
    assert response["query_id"] == test_query_id
    assert "documents" in response
    assert "fragments" in response

    # Validar requerimiento de Documentos (Exactamente 3)
    docs = response["documents"]
    print(f"Documentos devueltos ({len(docs)}): {docs}")
    assert len(docs) == 3, f"❌ Se esperaban 3 documentos, se recibieron {len(docs)}"

    # Validar requerimiento de Fragmentos (Exactamente 10)
    fragments = response["fragments"]
    print(f"Fragmentos devueltos ({len(fragments)})")
    assert len(fragments) == 10, f"❌ Se esperaban 10 fragmentos, se recibieron {len(fragments)}"

    # Validar límite de palabras por fragmento (<= 250 palabras)
    for frag in fragments:
        word_count = len(frag["text"].split())
        assert word_count <= 250, f"❌ Fragmento {frag['chunk_id']} excede 250 palabras ({word_count})"
        assert len(frag["text"]) > 0, f"❌ Fragmento {frag['chunk_id']} está vacío"

    print("\n🎉 ¡TODAS LAS PRUEBAS DE INTEGRACIÓN PASARON EXITOSAMENTE!")


if __name__ == "__main__":
    test_full_pipeline()
