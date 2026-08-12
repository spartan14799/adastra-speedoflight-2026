import abc
import re
import os
import uuid
from typing import Any, Dict, List
from langdetect import detect, LangDetectException

class BaseExtractor(abc.ABC):
    """
    Clase base abstracta. Contiene la lógica estándar compartida.
    Fuerza a todas las clases hijas a cumplir estrictamente el Contrato de Datos en RAG.
    """

    @property
    @abc.abstractmethod
    def tipo_fuente(self) -> str:
        pass

    @abc.abstractmethod
    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        pass

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Remoción de caracteres de control
        cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Remoción de boilerplate
        boilerplate_patterns = r'(?i)^\s*(page \d+ of \d+|página \d+ de \d+|copyright \d+|derechos reservados|all rights reserved)\b.*$'
        cleaned_text = re.sub(boilerplate_patterns, '', cleaned_text, flags=re.MULTILINE)
        # Normalización de espacios
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text).strip()
        
        # REGLA DE NEGOCIO: Integridad Oracional (Forzar cierre natural)
        if cleaned_text and cleaned_text[-1] not in ('.', '?', '!'):
            cleaned_text += "."
            
        return cleaned_text

    def process(self, file_path: str, phenomenon: str = "1") -> List[Dict[str, Any]]:
        extracted_documents = self.extract_documents(file_path)
        final_results = []
        source_name = os.path.basename(file_path)

        for doc in extracted_documents:
            raw_text = doc.get("raw_text", "")
            cleaned_text = self.clean_text(raw_text)
            
            if not cleaned_text:
                continue
            
            # Optimización: Se detecta el idioma solo con los primeros 2000 caracteres (Alta velocidad)
            try:
                language = detect(cleaned_text[:2000])
                if language not in ['es', 'en', 'pt']:
                    language = 'es'
            except LangDetectException:
                language = "es"
                
            base_metadata = doc.get("metadata", {})
            doc_id = doc.get("doc_id", str(uuid.uuid4()))

            # Homogeneización de atributo adicional (mapea title u otros valores de los extractores)
            atributo_adicional = str(base_metadata.get("atributo_adicional") or base_metadata.get("title") or "sin_atributo_especifico")

            # ESQUEMA ESTRICTO DE SALIDA (Contrato de Datos)
            output_schema = {
                "texto": cleaned_text,
                "metadata": {
                    "total_palabras": len(cleaned_text.split()),
                    "atributo_adicional": atributo_adicional,
                    "fuente": source_name,
                    "tipo_fuente": self.tipo_fuente,
                    "idioma": language,
                    "doc_id": doc_id,
                    "Fenomeno": phenomenon
                }
            }
            
            final_results.append(output_schema)

        return final_results
