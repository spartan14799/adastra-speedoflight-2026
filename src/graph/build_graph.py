import json
import logging
from pathlib import Path
import networkx as nx
import spacy
from langdetect import detect, DetectorFactory

# Semilla fija para detección determinista de idioma
DetectorFactory.seed = 0

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BuildKnowledgeGraph")


def cargar_modelos_spacy() -> dict:
    """
    Carga los modelos SpaCy instalados en el entorno.
    """
    logger.info("Cargando modelos de SpaCy para ES, EN y PT...")
    modelos = {}

    try:
        modelos["es"] = spacy.load("es_core_news_sm")
    except OSError:
        logger.warning("Modelo 'es_core_news_sm' no encontrado.")

    try:
        modelos["en"] = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("Modelo 'en_core_web_sm' no encontrado.")

    try:
        modelos["pt"] = spacy.load("pt_core_news_sm")
    except OSError:
        logger.warning("Modelo 'pt_core_news_sm' no encontrado.")

    if not modelos:
        raise RuntimeError("No se pudo cargar ningún modelo de SpaCy.")

    return modelos


def detectar_idioma(texto: str) -> str:
    """
    Detecta el idioma del fragmento de texto (es, en, pt).
    """
    try:
        lang = detect(texto)
        if lang in ["es", "en", "pt"]:
            return lang
    except Exception:
        pass
    return "es"


def cargar_y_desduplicar_metadata(base_vectorial_dir: Path) -> dict:
    """
    Lee todos los metadata.jsonl dentro de subcarpetas (encoder_bge-m3, encoder_e5, etc.).
    Filtra encabezados y desduplica/rectifica registros por chunk_id.
    """
    chunks_unicos = {}
    total_leidos = 0
    encabezados_omitidos = 0

    if not base_vectorial_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de origen: {base_vectorial_dir}")

    # Recorre automáticamente cualquier carpeta dentro de entrega/base_vectorial/
    for encoder_dir in sorted(base_vectorial_dir.iterdir()):
        if not encoder_dir.is_dir():
            continue

        jsonl_path = encoder_dir / "metadata.jsonl"
        if not jsonl_path.exists():
            logger.warning(f"Omitiendo subcarpeta sin metadata.jsonl: {encoder_dir.name}")
            continue

        logger.info(f"Leyendo metadatos desde: {encoder_dir.name}/metadata.jsonl")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except Exception as e:
                    logger.error(f"Error parseando JSON en línea {line_idx} ({jsonl_path.name}): {e}")
                    continue

                # Ignorar encabezado del encoder
                if item.get("type") == "encoder_header":
                    encabezados_omitidos += 1
                    continue

                chunk_id = item.get("chunk_id")
                if not chunk_id:
                    continue

                total_leidos += 1

                # Desduplicación entre encoders (bge-m3 y e5)
                if chunk_id not in chunks_unicos:
                    chunks_unicos[chunk_id] = {
                        "doc_id": item.get("doc_id", ""),
                        "chunk_id": chunk_id,
                        "fuente": item.get("fuente", ""),
                        "fenomeno": item.get("fenomeno", 0),
                        "texto": item.get("texto", "")
                    }

    logger.info(f"Procesadas {total_leidos} entradas válidas (Encabezados omitidos: {encabezados_omitidos}).")
    logger.info(f"Total de fragmentos (chunks) únicos consolidados: {len(chunks_unicos)}")
    return chunks_unicos


def extraer_tripletas(chunks_unicos: dict, nlp_models: dict) -> list:
    """
    Extrae entidades (NER) y relaciones sintácticas de cada fragmento único.
    """
    logger.info("Iniciando extracción de Entidades y Relaciones (Multilingüe)...")
    tripletas = []

    for idx, (chunk_id, data) in enumerate(chunks_unicos.items()):
        texto = data.get("texto", "")
        if not texto or len(texto.strip()) < 10:
            continue

        # Seleccionar modelo según idioma detectado
        lang = detectar_idioma(texto)
        nlp = nlp_models.get(lang, nlp_models.get("es"))

        doc = nlp(texto)

        for sent in doc.sents:
            ents = [e for e in sent.ents if len(e.text.strip()) > 1]

            # Requerimos al menos 2 entidades en la misma oración para conectar una relación
            if len(ents) >= 2:
                verbos = [tok.lemma_.lower() for tok in sent if tok.pos_ in ("VERB", "AUX")]
                relacion = "_".join(verbos[:2]) if verbos else "relacionado_con"

                for i in range(len(ents) - 1):
                    e1 = ents[i].text.strip().replace("\n", " ")
                    e2 = ents[i + 1].text.strip().replace("\n", " ")

                    if e1.lower() != e2.lower():
                        tripletas.append({
                            "sujeto": e1,
                            "tipo_sujeto": ents[i].label_,
                            "relacion": relacion[:40],
                            "objeto": e2,
                            "tipo_objeto": ents[i + 1].label_,
                            "chunk_id": str(chunk_id),
                            "doc_id": str(data["doc_id"]),
                            "fenomeno": int(data["fenomeno"])
                        })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(chunks_unicos):
            logger.info(f"Procesados {idx + 1}/{len(chunks_unicos)} fragmentos...")

    logger.info(f"Total tripletas extraídas: {len(tripletas)}")
    return tripletas


def guardar_grafo_graphml(tripletas: list, output_path: Path):
    """
    Exporta el grafo resultante a entrega/grafo/grafo.graphml.
    """
    logger.info("Construyendo grafo en NetworkX...")
    G = nx.DiGraph()

    for t in tripletas:
        subj = str(t["sujeto"])
        obj = str(t["objeto"])

        if not G.has_node(subj):
            G.add_node(subj, label="Entidad", entity_type=str(t["tipo_sujeto"]))

        if not G.has_node(obj):
            G.add_node(obj, label="Entidad", entity_type=str(t["tipo_objeto"]))

        if G.has_edge(subj, obj):
            chunks = G[subj][obj].get("chunk_id", "")
            if t["chunk_id"] not in chunks.split(","):
                G[subj][obj]["chunk_id"] = f"{chunks},{t['chunk_id']}"
        else:
            G.add_edge(
                subj,
                obj,
                relation=str(t["relacion"]),
                chunk_id=str(t["chunk_id"]),
                doc_id=str(t["doc_id"]),
                fenomeno=int(t["fenomeno"])
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Guardando {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas...")
    nx.write_graphml(G, str(output_path))
    logger.info(f"¡Grafo generado exitosamente en: {output_path}!")


def main():
    # Obtener la raíz del proyecto (3 niveles arriba desde src/graph/build_graph.py)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    base_vectorial_dir = project_root / "entrega" / "base_vectorial"
    grafo_output_path = project_root / "entrega" / "grafo" / "grafo.graphml"

    logger.info(f"Raíz del proyecto identificada: {project_root}")

    # 1. Cargar SpaCy
    nlp_models = cargar_modelos_spacy()

    # 2. Cargar y consolidar metadatos de encoder_bge-m3 y encoder_e5
    chunks_unicos = cargar_y_desduplicar_metadata(base_vectorial_dir)

    if not chunks_unicos:
        logger.error("No se encontraron metadatos para procesar. Abortando.")
        return

    # 3. Extraer tripletas con NER multilingüe
    tripletas = extraer_tripletas(chunks_unicos, nlp_models)

    # 4. Guardar archivo GraphML final
    guardar_grafo_graphml(tripletas, grafo_output_path)


if __name__ == "__main__":
    main()
