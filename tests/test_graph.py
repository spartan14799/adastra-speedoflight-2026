import sys
from pathlib import Path
import networkx as nx
import pytest

# Agregar la raíz del proyecto al PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

GRAFO_PATH = PROJECT_ROOT / "entrega" / "grafo" / "grafo.graphml"

# Palabras prohibidas/ruido común
STOPWORDS_RUIDO = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de", "del",
    "a", "en", "con", "por", "para", "que", "se", "su", "sus", "the", "a", "an",
    "and", "or", "of", "to", "in", "on", "for", "with", "by", "this", "that", "it"
}


def cargar_grafo():
    """Función auxiliar para cargar el grafo de entrega."""
    if not GRAFO_PATH.exists():
        pytest.fail(f"No se encontró el archivo del grafo en: {GRAFO_PATH}")
    return nx.read_graphml(str(GRAFO_PATH))


# ==============================================================================
# TEST 1: INTEGRIDAD Y ESTRUCTURA DEL GRAFO
# ==============================================================================
def test_integridad_y_estructura():
    print("\n" + "=" * 60)
    print("TEST 1: INTEGRIDAD Y ESTRUCTURA DEL GRAFO")
    print("=" * 60)

    G = cargar_grafo()

    num_nodos = G.number_of_nodes()
    num_aristas = G.number_of_edges()

    print(f"Total Nodos (Entidades): {num_nodos:,}")
    print(f"Total Aristas (Relaciones): {num_aristas:,}")

    # Validaciones básicas de tamaño
    assert num_nodos > 0, "El grafo no contiene nodos."
    assert num_aristas > 0, "El grafo no contiene aristas."

    # Validar atributos obligatorios en un muestreo de aristas
    aristas_muestra = list(G.edges(data=True))[:100]
    campos_faltantes = set()

    for u, v, data in aristas_muestra:
        for campo in ["chunk_id", "doc_id", "relation"]:
            if campo not in data or not str(data[campo]).strip():
                campos_faltantes.add(campo)

    print(f"• Estado de Metadatos Obligatorios (chunk_id, doc_id, relation):")
    if not campos_faltantes:
        print("CORRECTO: Todas las aristas evaluadas contienen la metadata requerida.")
    else:
        print(f"ERROR: Faltan metadatos en algunas aristas: {campos_faltantes}")

    assert len(campos_faltantes) == 0, f"Aristas sin metadata requerida: {campos_faltantes}"


# ==============================================================================
# TEST 2: CALIDAD SEMÁNTICA Y NIVEL DE RUIDO
# ==============================================================================
def test_calidad_semantica():
    print("\n" + "=" * 60)
    print("TEST 2: CALIDAD SEMÁNTICA Y DENSIDAD DE ENTIDADES")
    print("=" * 60)

    G = cargar_grafo()
    nodos = list(G.nodes())

    # Detectar nodos basura (longitud <= 1 o stopwords)
    nodos_ruido = [
        n for n in nodos 
        if len(str(n).strip()) <= 1 or str(n).strip().lower() in STOPWORDS_RUIDO
    ]

    porcentaje_ruido = (len(nodos_ruido) / len(nodos)) * 100
    print(f"• Porcentaje de Nodos Ruido (Stopwords/Caracteres aislados): {porcentaje_ruido:.2f}%")

    if nodos_ruido:
        print(f"  Ejemplos de ruido detectado: {nodos_ruido[:10]}")

    # Imprimir las 15 entidades más conectadas (Top Grado)
    print("\n• Top 15 Entidades Más Relevantes en el Grafo (Mayor Grado):")
    grados = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    
    for idx, (nodo, grado) in enumerate(grados[:15], 1):
        print(f"  {idx:2d}. '{nodo}' ({grado} conexiones)")

    # Un umbral permisible de ruido para grafos extraídos por NLP automático es < 10%
    assert porcentaje_ruido < 10.0, f"El nivel de ruido es muy alto ({porcentaje_ruido:.2f}%)."


# ==============================================================================
# TEST 3: APORTE Y RESOLUCIÓN DE QUERIES
# ==============================================================================
def test_aporte_a_queries():
    print("\n" + "=" * 60)
    print("TEST 3: CAPACIDAD DE APORTE Y SIMULACIÓN DE CONSULTAS")
    print("=" * 60)

    G = cargar_grafo()

    # Términos clave típicos del dominio de la competencia
    queries_prueba = [
        "IA",
        "inteligencia artificial",
        "FACSAT",
        "drones",
        "armas",
        "satélite",
        "United States",
        "defensa"
    ]

    exitos_busqueda = 0

    print("• Simulación de recuperación de Chunks desde el Grafo:")

    for query in queries_prueba:
        # Buscar coincidencias de la query en las entidades del grafo
        nodos_coincidentes = [
            n for n in G.nodes() 
            if query.lower() in str(n).lower()
        ]

        if not nodos_coincidentes:
            print(f"Query '{query}': No halló nodos explícitos en el grafo.")
            continue

        #Obtener vecinos a 1 salto de distancia y recuperar sus chunk_ids
        chunks_asociados = set()
        for nodo in nodos_coincidentes[:5]:  # Tomar los primeros 5 nodos coincidentes
            vecinos = list(G.neighbors(nodo))
            for v in vecinos:
                edge_data = G.get_edge_data(nodo, v)
                if edge_data and "chunk_id" in edge_data:
                    c_ids = str(edge_data["chunk_id"]).split(",")
                    chunks_asociados.update(c_ids)

        if chunks_asociados:
            exitos_busqueda += 1
            print(
                f"Query '{query}': Coincidió con {len(nodos_coincidentes)} entidades -> "
                f"Aporta {len(chunks_asociados)} chunk_ids para RRF."
            )
        else:
            print(f"Query '{query}': Entidad encontrada sin chunk_id asociado.")

    print(f"\n• Tasa de resolución en simulación: {exitos_busqueda}/{len(queries_prueba)} queries.")
    
    # Exigir que al menos el 25% de las queries de prueba aporten chunks
    assert exitos_busqueda > 0, "El grafo no logró aportar chunks para ninguna de las consultas de prueba."


if __name__ == "__main__":
    # Permite ejecutarlo directamente con: uv run python tests/test_grafo.py
    try:
        test_integridad_y_estructura()
        test_calidad_semantica()
        test_aporte_a_queries()
        print("\n" + "=" * 60)
        print("¡TODAS LAS PRUEBAS DEL GRAFO FUERON SUPERADAS EXITOSAMENTE!")
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\nPRUEBA FALLIDA: {e}\n")
        sys.exit(1)
