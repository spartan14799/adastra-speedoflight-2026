# Proyecto Ad Astra - Buscador Semántico

Sistema de recuperación semántica y agregación de documentos heterogéneos desarrollado para el reto **CODEFEST AD ASTRA**.

---

## Descripción

Este proyecto implementa un sistema de **búsqueda semántica** capaz de indexar documentos heterogéneos y recuperar la información más relevante mediante embeddings y una base vectorial utilizando **FAISS**.

El flujo de trabajo se divide en las siguientes etapas:

1. **Procesamiento de documentos:** los documentos procesados se almacenan en formato JSON dentro de `data/processed/`.
2. **Fragmentación (chunking) y generación de metadatos:** antes de construir los índices vectoriales, el módulo de chunking divide los documentos en fragmentos siguiendo las reglas de segmentación configuradas y genera los metadatos asociados a cada fragmento.
3. **Construcción de la base vectorial:** los fragmentos y sus metadatos se utilizan como entrada para la generación de los índices vectoriales de los encoders configurados.
4. **Procesamiento de consultas:** el motor de búsqueda utiliza los índices disponibles para recuperar la información más relevante y generar los resultados requeridos por la competencia.

La fase de **chunking y generación de metadatos** es un paso previo a la indexación y búsqueda. Su objetivo es producir una representación estructurada de los documentos, preservando la información necesaria para localizar y recuperar posteriormente cada fragmento.

---

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.10** o superior.
- **uv** (gestor de paquetes recomendado para un entorno rápido y reproducible).

---

## Instalación de `uv`

Si aún no tienes `uv` instalado, utiliza uno de los siguientes métodos.

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Instalación mediante pip

```bash
pip install uv
```

---

## Configuración del Proyecto

Clona el repositorio:

```bash
git clone https://github.com/spartan14799/adastra-speedoflight-2026
cd codefest-ad-astra
```

Instala todas las dependencias utilizando **uv**:

```bash
uv sync
```

> **Nota:** `uv` crea y administra automáticamente el entorno virtual, por lo que **no es necesario activarlo manualmente**.

---

## Ejecución del Proyecto

### 1. Preparar los documentos

Asegúrate de que los archivos JSON procesados estén disponibles en:

```text
data/processed/
```

Estos archivos constituyen la entrada para la fase de fragmentación y generación de metadatos.

### 2. Ejecutar el chunking y la generación de metadatos

Desde la raíz del proyecto, ejecuta:

```bash
uv run python -m src.chunker.build_metadata
```

Este comando ejecuta el módulo `src.chunker.build_metadata`, que:

1. Lee los archivos JSON disponibles en `data/processed/`.
2. Procesa los documentos utilizando las reglas de segmentación definidas por el módulo de chunking.
3. Aplica las reglas de oraciones y palabras configuradas para controlar el tamaño de los fragmentos y su solapamiento.
4. Genera los metadatos correspondientes a cada fragmento.
5. Escribe los archivos `metadata.jsonl` dentro de `entrega/base_vectorial/`, creando la salida correspondiente para cada encoder configurado en `src/chunker/config.json`.

La ejecución de este paso debe realizarse antes de la indexación vectorial, ya que los archivos `metadata.jsonl` contienen la información estructurada asociada a los fragmentos que posteriormente serán utilizados por el sistema de recuperación.

### 3. Continuar con la construcción y búsqueda

Una vez generados los metadatos, los archivos resultantes pueden utilizarse como entrada para las etapas posteriores de construcción de la base vectorial y procesamiento de consultas.

La configuración de los encoders y de los parámetros utilizados por el chunker se encuentra en:

```text
src/chunker/config.json
```

---

## Estructura del Proyecto

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── queries.json
├── docs/
│   └── chunker.md
├── src/
│   ├── chunker/
│   │   ├── __init__.py
│   │   ├── config.json
│   │   ├── core.py
│   │   └── build_metadata.py
│   ├── build_dictionary.py
│   ├── search_engine.py
│   └── universal_extractor.py
├── entrega/
│   ├── base_vectorial/
│   │   ├── encoder_bge-m3/
│   │   │   ├── dictionary.txt
│   │   │   ├── index.faiss
│   │   │   └── metadata.jsonl
│   │   └── encoder_e5/
│   │       ├── index.faiss
│   │       └── metadata.jsonl
│   ├── generador.py
│   ├── informe_tecnico.pdf
│   └── resultados.jsonl
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Arquitectura y Funcionamiento

El motor de búsqueda (`SearchEngine`) implementa una arquitectura **Multi-Encoder** sin el uso de modelos generativos, cumpliendo de forma estricta con las restricciones del reto.

Antes de que los fragmentos lleguen a los índices FAISS, el módulo de chunking procesa los documentos y genera los metadatos que permiten mantener la relación entre cada fragmento y su documento de origen.

```text
Documentos JSON
       │
       ▼
data/processed/
       │
       ▼
Chunking y generación de metadatos
(src/chunker/build_metadata.py)
       │
       ▼
metadata.jsonl por encoder
       │
       ▼
Construcción de índices FAISS
       │
       ▼
Entrada (Consulta)
       │
       ▼
Preprocesamiento & Corrección Ortográfica (SymSpell)
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
Encoder 1 (BGE-M3)         Encoder 2 (E5)            Encoder N...
       │                         │                         │
  Índice FAISS 1            Índice FAISS 2            Índice FAISS N
       │                         │                         │
       └─────────────────────────┼─────────────────────────┘
                                 ▼
                     Fusión RRF (Reciprocal Rank Fusion)
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
      Top 10 Fragmentos              Top 3 Documentos
   (<= 250 palabras / completitud)     (Max Pooling de Scores)
```

## Documentación Adicional

La carpeta `docs/` contiene documentación específica de los componentes del proyecto. En particular, `docs/chunker.md` describe en profundidad el funcionamiento del algoritmo de segmentación, el flujo de tokenización, la configuración de los tokenizers y los parámetros disponibles en `src/chunker/config.json`.

La documentación del chunker complementa este README y debe consultarse para conocer con mayor detalle las reglas utilizadas durante la generación de fragmentos y metadatos.

---

## Tecnologías Utilizadas

- Python 3.10+
- uv
- FAISS
- Embeddings semánticos
- Procesamiento de lenguaje natural (NLP)

---

## Autores

Proyecto desarrollado para el **CODEFEST AD ASTRA**.
