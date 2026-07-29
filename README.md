# Proyecto Ad Astra - Buscador Semántico

Sistema de recuperación semántica y agregación de documentos heterogéneos desarrollado para el reto **CODEFEST AD ASTRA**.

---

## Descripción
### TODO: Crear el flujo de ejecución

Este proyecto implementa un sistema de **búsqueda semántica** capaz de indexar documentos heterogéneos y recuperar la información más relevante mediante embeddings y una base vectorial utilizando **FAISS**.

El flujo de trabajo se divide en dos etapas:

1. **Construcción de la base vectorial** a partir de los documentos fuente.
2. **Procesamiento de consultas** para generar los resultados requeridos por la competencia.

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
git clone https://github.com/tu-usuario/codefest-ad-astra.git
cd codefest-ad-astra
```

Instala todas las dependencias utilizando **uv**:

```bash
uv sync
```

> **Nota:** `uv` crea y administra automáticamente el entorno virtual, por lo que **no es necesario activarlo manualmente**.

---

## Ejecución del Proyecto
### TODO: Crear el flujo de ejecución

## Estructura del Proyecto

```text
.
├── data/
│   ├── raw/
│   └── queries.json
├── src/
│   └── search_engine.py
├── entrega
│    └── generador.py
├── uv.lock
├── pyproject.toml
└── README.md
```

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
