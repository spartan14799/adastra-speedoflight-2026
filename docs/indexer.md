# Módulo `src/indexer/`

## Propósito

Este directorio agrupa los componentes relacionados con la preparación y gestión de estructuras de indexación utilizadas por otras partes del repositorio.

## Estructura

```text
src/indexer/
```

## Consideraciones para modificaciones

Antes de realizar cambios en esta capa se deben identificar:

1. Las fuentes de datos utilizadas para construir los índices.
2. El formato de los metadatos asociados.
3. La ubicación de persistencia de los artefactos.
4. Los módulos consumidores de los índices.

La estructura de los índices utilizados por el repositorio se refleja en:

```text
entrega/base_vectorial/
├── encoder_bge-m3/
│   ├── index.faiss
│   └── metadata.jsonl
└── encoder_e5/
    ├── index.faiss
    └── metadata.jsonl
```

Cualquier modificación en los formatos persistidos debe evaluarse junto con los componentes que los cargan posteriormente.

## Módulos relacionados

- [`chunker.md`](chunker.md)
- [`search_engine.md`](search_engine.md)