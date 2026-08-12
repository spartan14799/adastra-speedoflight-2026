import abc
import json
import re
import os
import uuid
from typing import Any, Dict, List
from langdetect import detect, LangDetectException
from bs4 import BeautifulSoup

from .base import BaseExtractor 



class HTMLExtractor(BaseExtractor):
    """
    Clase concreta para la ingesta de archivos HTML.
    Conserva etiquetas semánticas clave (h1-h6, p, li) para facilitar 
    estrategias de chunking jerárquico posteriores.
    """
    @property
    def tipo_fuente(self) -> str:
        return "html"
    
    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        raw_documents = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
                
            # Parseo del DOM utilizando BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extracción de metadatos básicos
            title_tag = soup.find('title')
            doc_title = title_tag.get_text(strip=True) if title_tag else "html_sin_titulo"
            
            # Destrucción de elementos nocivos o ruidosos
            tags_to_decompose = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'noscript', 'iframe']
            for tag in soup.find_all(tags_to_decompose):
                tag.decompose()
                
            # Extracción de bloques preservando etiquetas estructurales
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



