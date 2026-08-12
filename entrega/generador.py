import json
import sys
from pathlib import Path

# Configuración de rutas del proyecto
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    CURRENT_DIR.parent
    if (CURRENT_DIR / "src").exists() is False
    else CURRENT_DIR
)
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from search_engine.search_engine import SearchEngine


def generate_results():
    print("Cargando SearchEngine...")

    base_vectorial_path = PROJECT_ROOT / "entrega" / "base_vectorial"
    queries_file = PROJECT_ROOT / "preguntas.json"
    if not queries_file.exists():
        queries_file = CURRENT_DIR / "preguntas.json"

    output_path = PROJECT_ROOT / "entrega" / "resultados.jsonl"

    # Inicializar el motor de búsqueda
    engine = SearchEngine(
        base_vectorial_dir=str(base_vectorial_path),
        index_type="hnsw",
        device="cpu",
    )

    # Cargar preguntas
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Procesando {len(queries)} consultas...")

    results_lines = []

    # Procesar ordenadamente de q001 a q050
    for query_id, query_text in sorted(queries.items()):
        result = engine.search(query_id, query_text)

        # Convertir a una sola línea JSON en formato UTF-8
        line = json.dumps(result, ensure_ascii=False)
        results_lines.append(line)

    # Guardar archivo .jsonl (exactamente 1 objeto JSON por línea)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines) + "\n")

    print(f"¡Listo! Archivo .jsonl generado en: {output_path}")
    print(f"Total de consultas procesadas: {len(results_lines)}")


if __name__ == "__main__":
    generate_results()
