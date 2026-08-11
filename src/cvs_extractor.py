import abc
import os
import re
import uuid
import json
import pandas as pd
from typing import Any, Dict, List
from langdetect import detect, LangDetectException

class BaseExtractor(abc.ABC):
    """
    Clase base abstracta para la extracción y sanitización de texto.
    Ajustada estrictamente para retornar el Contrato de Datos requerido para sistemas RAG.
    """

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
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        return cleaned_text.strip()

    def process(self, file_path: str, phenomenon: str = "1") -> List[Dict[str, Any]]:
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
            
            # Detección de idioma higiénica
            try:
                language = detect(cleaned_text[:2000]) # Detectar con una muestra para optimizar rendimiento
            except LangDetectException:
                language = "es"
                
            base_metadata = doc.get("metadata", {})
            doc_id = doc.get("doc_id", str(uuid.uuid4()))

            # Adaptación estricta al esquema de salida (Contrato de Datos Inviolable)
            doc_metadata = {
                "total_palabras": len(cleaned_text.split()),
                "atributo_adicional": base_metadata.get("atributo_adicional", ""),
                "fuente": source_name,
                "tipo_fuente": "csv",
                "idioma": language,
                "doc_id": doc_id,
                "Fenomeno": phenomenon
            }
            
            output_schema = {
                "texto": cleaned_text,
                "metadata": doc_metadata
            }
            
            final_results.append(output_schema)

        return final_results


class CSVExtractor(BaseExtractor):
    """
    Clase concreta para la ingesta de archivos CSV.
    Transforma registros tabulares en texto explícito mapeado.
    Recompila toda la información de las filas en UN ÚNICO documento consolidado.
    """
    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")
            
        try:
            # Procesamiento en chunks para optimización de RAM (esencial para CSV con muchas URLs/peso)
            chunk_size = 5000
            
            # Almacenará todas las oraciones (filas) generadas
            all_rows_text = []
            filas_procesadas = 0
            
            # dtype=str preserva las URLs y textos sin conversiones erróneas
            # keep_default_na=False evita convertir celdas vacías en texto "NaN"
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=str, keep_default_na=False):
                for _, row in chunk.iterrows():
                    row_elements = []
                    for col_name, value in row.items():
                        val_str = str(value).strip()
                        if val_str:  # Ignorar celdas sin datos
                            row_elements.append(f"{str(col_name).strip()}: {val_str}")
                            
                    # Integridad oracional: Se separan por | y se fuerza el cierre con "."
                    if row_elements:
                        all_rows_text.append(" | ".join(row_elements) + ".")
                        filas_procesadas += 1
                        
            # Generamos UN ÚNICO bloque de texto consolidado uniendo todas las líneas
            full_raw_text = " ".join(all_rows_text)
            
            # Retornamos una lista de 1 solo documento consolidado
            return [{
                "doc_id": str(uuid.uuid4()),
                "raw_text": full_raw_text,
                "metadata": {
                    "atributo_adicional": f"total_filas_consolidadas_{filas_procesadas}"
                }
            }]
                        
        except Exception as e:
            raise RuntimeError(f"Fallo de procesamiento estructurado en {file_path}: {str(e)}")


if __name__ == "__main__":
    # 1. Única línea de ingreso: solicitud estricta para la ruta del archivo
    input_path = input("Please enter the CSV file path on your device: ").strip()
    file_path = input_path.strip("\"'")
    
    # 2. Asignación automática del fenómeno extrayendo el número de la carpeta contenedora
    directorio_padre = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
    fenomeno_asociado = "1"  # Valor por defecto seguro
    
    if "3" in directorio_padre:
        fenomeno_asociado = "3"
    elif "2" in directorio_padre:
        fenomeno_asociado = "2"
    elif "1" in directorio_padre:
        fenomeno_asociado = "1"
    
    if not file_path or not os.path.exists(file_path):
        # 3. Retorna lista vacía si no existe el archivo o hay un error en la ruta
        print(json.dumps([]))
    else:
        try:
            extractor = CSVExtractor()
            obtained_docs = extractor.process(file_path, phenomenon=fenomeno_asociado)
            
            # 1 y 3. Imprime estrictamente la LISTA completa, o una lista vacía si no hay resultados
            if obtained_docs:
                print(json.dumps(obtained_docs, indent=2, ensure_ascii=False))
            else:
                print(json.dumps([]))
        except Exception:
            # 3. Retorna lista vacía en caso de que la ejecución falle por cualquier motivo
            print(json.dumps([]))