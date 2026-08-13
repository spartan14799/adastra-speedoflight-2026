# Módulo `src/chunker/`

## Propósito

Este módulo implementa la lógica de segmentación y la construcción de metadatos asociados a los fragmentos generados.

La configuración de la segmentación se mantiene separada de la implementación para permitir modificar parámetros sin alterar la lógica principal.

## Estructura

```text
src/chunker/
├── __init__.py
├── config.json
├── core.py
└── build_metadata.py
```

## Archivos

### `config.json`

Contiene los parámetros utilizados por la lógica de fragmentación.

Incluye la configuración relacionada con:

- Ventanas.
- Solapamiento.
- Tokenizadores.

Los cambios de comportamiento parametrizable deben realizarse preferiblemente en este archivo cuando no requieran modificar la implementación.

### `core.py`

Contiene la lógica principal de segmentación.

Es responsable de aplicar la estrategia de fragmentación y respetar las fronteras lingüísticas definidas por la implementación.

### `build_metadata.py`

Punto de entrada para la generación de metadatos asociados a los fragmentos procesados.

Su ejecución genera la información utilizada posteriormente por las etapas dependientes de los datos fragmentados.

## Ejecución

```bash
uv run python -m src.chunker.build_metadata
```

## Consideraciones para modificaciones

Antes de modificar este módulo, revisar:

1. Los parámetros definidos en `config.json`.
2. La lógica central de `core.py`.
3. El formato de metadatos generado por `build_metadata.py`.
4. Los consumidores posteriores de los archivos producidos.

Los cambios en el formato de metadata deben evaluarse considerando los módulos que dependan de esos archivos.

## Módulos relacionados

- [`extractor.md`](extractor.md)
- [`indexer.md`](indexer.md)
- [`graph.md`](graph.md)
- [`search_engine.md`](search_engine.md)