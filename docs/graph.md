# Módulo `src/graph/`

## Propósito

Este módulo contiene la lógica encargada de construir y exportar una representación estructurada a partir de la información procesada.

## Estructura

```text
src/graph/
└── build_graph.py
```

## `build_graph.py`

Contiene el punto de entrada para la construcción del grafo.

La implementación procesa la información de entrada, realiza las operaciones NLP necesarias para identificar entidades y relaciones y exporta la estructura resultante en formato GraphML.

## Ejecución

```bash
uv run python -m src.graph.build_graph
```

## Salida

La ejecución produce el archivo:

```text
entrega/grafo/grafo.graphml
```

## Consideraciones para modificaciones

Los cambios en este módulo deben considerar:

- El formato esperado de las entradas.
- La consistencia de nodos y relaciones.
- Los atributos exportados.
- La compatibilidad del archivo GraphML generado.
- Los consumidores que puedan depender del formato de salida.

## Módulos relacionados

- [`chunker.md`](chunker.md)