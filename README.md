# Ad Astra

Documentación principal del repositorio.

Este archivo contiene la información necesaria para instalar, ejecutar y navegar por el código. La documentación técnica detallada de cada módulo se encuentra en [`docs/`](docs/README.md).

## Tabla de contenidos

- [Estructura del repositorio](#estructura-del-repositorio)
- [Documentación](#documentación)
- [Archivos Pesados y Git LFs](#Archivos pesados y Git LFS)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Workflow](#workflow)
- [Ejecución](#ejecución)
- [Pruebas](#pruebas)
- [Entrega](#entrega)

## Estructura del repositorio

```text
.
├── data/
├── docs/
│   ├── README.md
│   ├── extractor.md
│   ├── chunker.md
│   ├── graph.md
│   ├── indexer.md
│   └── search_engine.md
├── entrega/
│   ├── base_vectorial/
│   │   ├── encoder_bge-m3/
│   │   │   ├── index.faiss
│   │   │   └── metadata.jsonl
│   │   └── encoder_e5/
│   │       ├── index.faiss
│   │       └── metadata.jsonl
│   ├── grafo/
│   │   └── grafo.graphml
│   ├── generador.py
│   ├── informe_tecnico.pdf
│   └── resultados.jsonl
├── src/
│   ├── chunker/
│   ├── extractor/
│   ├── graph/
│   ├── indexer/
│   └── search_engine/
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
````

## Directorios principales

| Ruta             | Descripción                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `data/`          | Archivos de entrada utilizados durante la ejecución.                |
| `src/`           | Código fuente organizado por módulos.                               |
| `tests/`         | Pruebas del proyecto.                                               |
| `docs/`          | Documentación técnica detallada.                                    |
| `entrega/`       | Archivos y artefactos requeridos para la ejecución final y entrega. |
| `pyproject.toml` | Configuración del proyecto y dependencias.                          |
| `uv.lock`        | Versiones bloqueadas de las dependencias.                           |

## Documentación

La documentación detallada está separada por responsabilidad para evitar concentrar toda la información técnica en este archivo.

| Módulo            | Documentación                                    |
| ----------------- | ------------------------------------------------ |
| Extracción        | [`docs/extractor.md`](docs/extractor.md)         |
| Fragmentación     | [`docs/chunker.md`](docs/chunker.md)             |
| Grafo             | [`docs/graph.md`](docs/graph.md)                 |
| Indexación        | [`docs/indexer.md`](docs/indexer.md)             |
| Motor de búsqueda | [`docs/search_engine.md`](docs/search_engine.md) |

## Archivos pesados y Git LFS

Algunos archivos grandes del repositorio están gestionados mediante Git LFS, incluyendo archivos procesados, índices FAISS, archivos JSONL y grafos GraphML.

Si deseas utilizar los archivos ya incluidos en el repositorio, asegúrate de tener Git LFS instalado y ejecuta:

```bash
git lfs install
git lfs pull
```

Puedes verificar los archivos gestionados por Git LFS con:

```bash
git lfs ls-files
```

La configuración de los archivos gestionados se encuentra en `.gitattributes`.

## Requisitos

El repositorio utiliza `uv` como gestor de paquetes y entorno de ejecución.

Todos los comandos documentados deben ejecutarse desde la raíz del repositorio.

## Instalación

Sincroniza las dependencias:

```bash
uv sync
```

`uv` utiliza `pyproject.toml` y `uv.lock` para preparar el entorno con las versiones definidas por el proyecto.

## Workflow

El flujo general de trabajo del repositorio es:

```text
data/
  │
  ▼
src/extractor/
  │
  ▼
src/chunker/
  │
  ├──────────────► src/graph/
  │
  └──────────────► src/indexer/
                       │
                       ▼
                 src/search_engine/
                       │
                       ▼
                    entrega/
```

El workflow recomendado para trabajar con el código es:

1. Instalar las dependencias con `uv sync`.
2. Revisar la documentación del módulo que se desea modificar.
3. Ejecutar únicamente el módulo necesario durante el desarrollo.
4. Ejecutar el workflow completo cuando se requiera regenerar los artefactos.
5. Ejecutar las pruebas correspondientes desde `tests/`.
6. Ejecutar el generador principal para producir los archivos de salida de `entrega/`.

## Ejecución

### Extracción

```bash
uv run python -m src.extractor.main
```

Documentación:

[`docs/extractor.md`](docs/extractor.md)

### Generación de chunks y metadata

```bash
uv run python -m src.chunker.build_metadata
```

Documentación:

[`docs/chunker.md`](docs/chunker.md)

### Construcción del grafo

```bash
uv run python -m src.graph.build_graph
```

Documentación:

[`docs/graph.md`](docs/graph.md)

### Construcción del diccionario

```bash
uv run python -m src.search_engine.build_dictionary
```

Documentación:

[`docs/search_engine.md`](docs/search_engine.md)

### Generador principal

```bash
uv run python entrega/generador.py
```

Este comando ejecuta el punto de entrada principal ubicado en `entrega/generador.py`.

## Ejecución completa

Para ejecutar las etapas principales en orden:

```bash
uv sync

uv run python -m src.extractor.main

uv run python -m src.chunker.build_metadata

uv run python -m src.graph.build_graph

uv run python -m src.search_engine.build_dictionary

uv run python entrega/generador.py
```

Durante el desarrollo no es necesario ejecutar todas las etapas cuando únicamente se está modificando un módulo aislado.

La documentación de cada módulo especifica sus responsabilidades, archivos relacionados y consideraciones para modificar su implementación.

## Pruebas

Las pruebas se encuentran en:

```text
tests/
```

Las pruebas y scripts del repositorio deben ejecutarse utilizando el entorno administrado por `uv`.

La estructura concreta de pruebas debe mantenerse alineada con los módulos ubicados en `src/`.

## Entrega

El directorio `entrega/` contiene los archivos requeridos por el proceso de entrega.

```text
entrega/
├── base_vectorial/
│   ├── encoder_bge-m3/
│   │   ├── index.faiss
│   │   └── metadata.jsonl
│   └── encoder_e5/
│       ├── index.faiss
│       └── metadata.jsonl
├── grafo/
│   └── grafo.graphml
├── generador.py
├── informe_tecnico.pdf
└── resultados.jsonl
```

### Archivos principales

#### `entrega/generador.py`

Punto de entrada principal de la entrega.

Se ejecuta mediante:

```bash
uv run python entrega/generador.py
```

#### `entrega/base_vectorial/`

Contiene los directorios de las bases vectoriales utilizadas por el generador.

Cada encoder dispone de:

```text
index.faiss
metadata.jsonl
```

#### `entrega/grafo/grafo.graphml`

Archivo GraphML incluido dentro de la estructura de entrega.

#### `entrega/informe_tecnico.pdf`

Informe técnico incluido como parte de los archivos de entrega.

#### `entrega/resultados.jsonl`

Archivo generado por el punto de entrada principal.

## Navegación rápida

* Código fuente: [`src/`](src/)
* Documentación: [`docs/`](docs/)
* Pruebas: [`tests/`](tests/)
* Entrega: [`entrega/`](entrega/)
* Configuración: [`pyproject.toml`](pyproject.toml)





