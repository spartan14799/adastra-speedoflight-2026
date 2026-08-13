# Módulo `src/search_engine/`

## Propósito

Este módulo implementa la lógica principal de procesamiento de consultas y recuperación.

Contiene la construcción de recursos auxiliares para consultas y el motor que coordina las diferentes etapas de búsqueda.

## Estructura

```text
src/search_engine/
├── build_dictionary.py
└── search_engine.py
```

## Archivos

### `build_dictionary.py`

Construye el diccionario utilizado por el componente de corrección ortográfica de consultas.

## Ejecución

```bash
uv run python -m src.search_engine.build_dictionary
```

### `search_engine.py`

Contiene la implementación principal del motor de búsqueda.

Coordina:

- Procesamiento de consultas.
- Recuperación sobre las estructuras disponibles.
- Combinación de rankings.
- Aplicación de filtros.
- Agregación de resultados cuando corresponde.

La implementación utiliza una estrategia Multi-Encoder con BGE-M3 y E5 y fusiona resultados mediante Reciprocal Rank Fusion.

## Consideraciones para modificaciones

Antes de cambiar la lógica del motor, revisar:

1. El formato esperado de las consultas.
2. La ubicación y formato de los índices.
3. El formato de los metadatos.
4. La lógica de combinación de rankings.
5. Los post-filtros aplicados.
6. La agregación final a nivel de documento.
7. El formato consumido por `entrega/generador.py`.

Los cambios en estos contratos pueden requerir modificaciones coordinadas en otros módulos.

## Módulos relacionados

- [`indexer.md`](indexer.md)
- [`chunker.md`](chunker.md)
- [`../README.md`](../README.md)