# Documentación técnica

Este directorio contiene la documentación detallada del código fuente.

La documentación está organizada por módulos para mantener el `README.md` principal centrado en instalación, estructura, workflow y ejecución.

## Módulos

| Módulo | Documentación |
|---|---|
| `src/extractor/` | [`extractor.md`](extractor.md) |
| `src/chunker/` | [`chunker.md`](chunker.md) |
| `src/graph/` | [`graph.md`](graph.md) |
| `src/indexer/` | [`indexer.md`](indexer.md) |
| `src/search_engine/` | [`search_engine.md`](search_engine.md) |

## Uso de esta documentación

Antes de modificar un módulo:

1. Identifica el directorio correspondiente en `src/`.
2. Revisa su documentación.
3. Localiza los puntos de entrada y archivos relacionados.
4. Realiza los cambios necesarios.
5. Ejecuta el módulo mediante `uv run`.
6. Ejecuta las pruebas correspondientes.

## Convenciones

La documentación de cada módulo debe describir:

- Responsabilidad del módulo.
- Estructura interna.
- Responsabilidad de cada archivo.
- Puntos de entrada.
- Entradas y salidas relevantes.
- Dependencias con otros módulos.
- Consideraciones para modificar el código.

# Módulo `src/extractor/`

## Propósito

Este módulo concentra la lógica de lectura, extracción, limpieza y normalización de contenido procedente de diferentes formatos.

La implementación está separada por formato para evitar concentrar lógica específica en un único componente y facilitar la incorporación o modificación de extractores.

## Estructura

```text
src/extractor/
├── __init__.py
├── base.py
├── cvs_extractor.py
├── html_extractor.py
├── json_extractor.py
├── pbf_extractor.py
├── pdf_extractor.py
└── main.py
```

## Archivos

### `base.py`

Define la abstracción común utilizada por los extractores.

Las nuevas implementaciones de extracción deben mantener compatibilidad con la interfaz y comportamiento esperado por el orquestador.

### `cvs_extractor.py`

Implementa la lógica específica para el formato CSV.

### `html_extractor.py`

Implementa la lógica específica para el formato HTML.

### `json_extractor.py`

Implementa la lógica específica para estructuras JSON.

### `pbf_extractor.py`

Implementa la lógica específica para el formato PBF.

### `pdf_extractor.py`

Implementa la lógica específica para documentos PDF.

### `main.py`

Punto de entrada del módulo.

Coordina la ejecución de los extractores y el procesamiento masivo de las fuentes disponibles.

## Ejecución

Desde la raíz del repositorio:

```bash
uv run python -m src.extractor.main
```

## Consideraciones para modificaciones

Al agregar soporte para un nuevo formato:

1. Crear un extractor especializado.
2. Mantener la interfaz definida en `base.py`.
3. Registrar o integrar el extractor en el flujo coordinado por `main.py`.
4. Agregar pruebas específicas.
5. Documentar cualquier nueva dependencia.

## Módulos relacionados

- [`chunker.md`](chunker.md)