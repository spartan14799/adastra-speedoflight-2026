import abc
import json
import re
import os
import uuid
from typing import Any, Dict, List
from langdetect import detect, LangDetectException
from bs4 import BeautifulSoup

class BaseExtractor(abc.ABC):
    """
    Clase base abstracta para la extracción y sanitización de texto.
    Ajustada estrictamente para retornar el Contrato de Datos exigido.
    """

    @abc.abstractmethod
    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        pass

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Remoción de caracteres de control
        cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Remoción de boilerplate basado en patrones textuales
        boilerplate_patterns = r'(?i)^\s*(page \d+ of \d+|página \d+ de \d+|copyright \d+|derechos reservados|all rights reserved)\b.*$'
        cleaned_text = re.sub(boilerplate_patterns, '', cleaned_text, flags=re.MULTILINE)
        # Normalización de espacios preservando saltos de línea y etiquetas
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        # Evitar truncar espacios dentro de las etiquetas limpias
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        return cleaned_text.strip()

    def process(self, file_path: str, phenomenon: int = 1) -> List[Dict[str, Any]]:
        """
        Orquestador principal.
        Genera el contrato de datos exacto exigido por las restricciones de negocio.
        """
        extracted_documents = self.extract_documents(file_path)
        final_results = []
        source_name = os.path.basename(file_path)

        for doc in extracted_documents:
            raw_text = doc.get("raw_text", "")
            cleaned_text = self.clean_text(raw_text)
            
            if not cleaned_text:
                continue
            
            # Sanitización temporal para métricas e idioma (ignorar etiquetas HTML)
            texto_sin_etiquetas = re.sub(r'<[^>]+>', ' ', cleaned_text)
            
            # Detección higiénica de idioma basada en el texto puro
            try:
                language = detect(texto_sin_etiquetas)
            except LangDetectException:
                language = "es"
                
            base_metadata = doc.get("metadata", {})
            doc_id = doc.get("doc_id", str(uuid.uuid4()))

            # Adaptación estricta al esquema de salida (Contrato de Datos Inviolable)
            output_schema = {
                "texto": cleaned_text,
                "metadata": {
                    # Conteo real de palabras omitiendo la sintaxis HTML
                    "total_palabras": len(texto_sin_etiquetas.split()),
                    "atributo_adicional": base_metadata.get("title", "sin_titulo"),
                    "fuente": source_name,
                    "tipo_fuente": "html",
                    "idioma": language,
                    "doc_id": doc_id,
                    "Fenomeno": str(phenomenon)
                }
            }
            
            final_results.append(output_schema)

        return final_results


class HTMLExtractor(BaseExtractor):
    """
    Clase concreta para la ingesta de archivos HTML.
    Conserva etiquetas semánticas clave (h1-h6, p, li) para facilitar 
    estrategias de chunking jerárquico posteriores.
    """
    
    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        raw_documents = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
                
            # Parseo del DOM utilizando BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 1. Extracción de metadatos básicos
            title_tag = soup.find('title')
            doc_title = title_tag.get_text(strip=True) if title_tag else "html_sin_titulo"
            
            # 2. Destrucción de elementos nocivos o ruidosos
            tags_to_decompose = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'noscript', 'iframe']
            for tag in soup.find_all(tags_to_decompose):
                tag.decompose()
                
            # 3. Extracción de bloques preservando etiquetas estructurales
            # Se apuntan etiquetas que delimitan ideas u oraciones completas
            etiquetas_permitidas = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']
            text_blocks = []
            
            for element in soup.find_all(etiquetas_permitidas):
                # Se limpia el contenido de la etiqueta para evitar espacios rotos
                # El separator=' ' previene la concatenación de palabras hijas sin espacio
                clean_content = element.get_text(separator=' ', strip=True)
                clean_content = re.sub(r'[ \t]+', ' ', clean_content)
                
                # Si la etiqueta contiene texto, la reconstruimos
                if clean_content:
                    tag_name = element.name
                    text_blocks.append(f"<{tag_name}>{clean_content}</{tag_name}>")
            
            # Unimos usando saltos de línea para preservar limpieza visual y estructural
            visible_text = "\n".join(text_blocks)
            
            raw_documents.append({
                "doc_id": str(uuid.uuid4()),
                "raw_text": visible_text,
                "metadata": {
                    "title": doc_title,
                    "format": "html"
                }
            })
            
        except Exception as e:
            raise RuntimeError(f"Falla de procesamiento en {file_path}: {str(e)}")
            
        return raw_documents


if __name__ == "__main__":
    # Solicitud estricta para la ruta del archivo
    input_path = input("Ingrese la ruta del archivo HTML: ").strip()
    file_path = input_path.strip("\"'")
    
    # Asignación del fenómeno (1, 2 o 3)
    try:
        fenomeno_input = int(input("Ingrese el número del fenómeno (1, 2 o 3): ").strip())
        if fenomeno_input not in [1, 2, 3]:
            fenomeno_input = 1
    except ValueError:
        fenomeno_input = 1

    if not file_path or not os.path.exists(file_path):
        print(json.dumps([]))
    else:
        try:
            extractor = HTMLExtractor()
            obtained_docs = extractor.process(file_path, phenomenon=fenomeno_input)
            print(json.dumps(obtained_docs, indent=2, ensure_ascii=False))
        except Exception as e:
            print(json.dumps([]))