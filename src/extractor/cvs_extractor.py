import abc
import os
import re
import uuid
import json
import pandas as pd
from typing import Any, Dict, List
from langdetect import detect, LangDetectException


from .base import BaseExtractor


class CSVExtractor(BaseExtractor):
    """
    Clase concreta para la ingesta de archivos CSV.
    Transforma registros tabulares en texto explícito mapeado.
    Recompila toda la información de las filas en UN ÚNICO documento consolidado.
    """
    @property
    def tipo_fuente(self) -> str:
        return "csv"
    
    @.setter
    def (self, value):
        self._ = value
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



