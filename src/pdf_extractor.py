import abc
import json
import re
import os
import uuid
from collections import Counter
from typing import Any, Dict, List, Set, Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError("La biblioteca 'pdfplumber' es requerida. Ejecuta: pip install pdfplumber")

try:
    from langdetect import detect, LangDetectException
except ImportError:
    raise ImportError("La biblioteca 'langdetect' es requerida. Ejecuta: pip install langdetect")


class BaseExtractor(abc.ABC):
    """
    Clase base abstracta para la extracción y sanitización de texto.
    Garantiza el Contrato de Datos estricto requerido para sistemas RAG sin LLMs.
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
            
        # 1. Remoción de caracteres de control (excepto puntuación vital y saltos de línea estructurados)
        cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 2. Limpieza de artefactos remanentes (espacios múltiples o tabulaciones)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        
        return cleaned_text.strip()

    def process(self, file_path: str, phenomenon: str = "1") -> List[Dict[str, Any]]:
        """
        Orquestador principal que genera el contrato de datos exacto exigido.
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
                language = detect(cleaned_text)
            except LangDetectException:
                language = "es"  # Fallback por defecto
                
            base_metadata = doc.get("metadata", {})
            doc_id = doc.get("doc_id", str(uuid.uuid4()))

            # Empaquetado estricto cumpliendo el contrato de datos
            output_schema = {
                "texto": cleaned_text,
                "metadata": {
                    "total_palabras": len(cleaned_text.split()),
                    "atributo_adicional": base_metadata.get("atributo_adicional", "vacio_o_especifico_del_formato"),
                    "fuente": source_name,
                    "tipo_fuente": self.tipo_fuente,
                    "idioma": language,
                    "doc_id": doc_id,
                    "Fenomeno": phenomenon
                }
            }
            
            final_results.append(output_schema)

        return final_results


class PDFExtractor(BaseExtractor):
    """
    Extracción modular de PDFs integrando tablas dentro del flujo 
    espacial del texto, transformando cada fila tabular en una oración 
    estructurada para mantener la integridad relacional.
    """
    
    @property
    def tipo_fuente(self) -> str:
        return "pdf"

    def _identify_repetitive_artifacts(self, pdf: pdfplumber.PDF) -> Set[str]:
        """
        Realiza un escaneo de frecuencia para detectar encabezados y pies de página.
        """
        line_counter = Counter()
        num_pages = len(pdf.pages)
        threshold = max(3, int(num_pages * 0.3))
        
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                for line in text.split('\n'):
                    norm_line = re.sub(r'\d+', '', line).strip().lower()
                    if len(norm_line) > 4: 
                        line_counter[norm_line] += 1
                        
        return {line for line, count in line_counter.items() if count >= threshold}

    def _process_text_block(self, text: Optional[str], repetitive_artifacts: Set[str], output_list: List[str]) -> None:
        """
        Subrutina para limpiar y añadir líneas de texto provenientes de una sección de la página.
        """
        if not text:
            return
            
        for linea in text.split('\n'):
            linea_limpia = linea.strip()
            if not linea_limpia:
                continue
                
            norm_line = re.sub(r'\d+', '', linea_limpia).lower()
            
            # Filtro heurístico y Regex de Paginación
            if norm_line in repetitive_artifacts:
                continue
            if re.search(r'(?i)^(p[áa]gina\s+\d+|page\s+\d+)', linea_limpia):
                continue
                
            # Asegurar un punto final si parece ser el fin de una oración / párrafo independiente
            if not linea_limpia[-1] in '.?!;:,':
                linea_limpia += "."
                
            output_list.append(linea_limpia)

    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        raw_documents = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")
            
        try:
            texto_completo = []
            
            with pdfplumber.open(file_path) as pdf:
                repetitive_artifacts = self._identify_repetitive_artifacts(pdf)
                
                for page in pdf.pages:
                    tables = page.find_tables()
                    page_height = page.height
                    page_width = page.width
                    
                    # Ordenamiento espacial de tablas de arriba hacia abajo
                    if tables:
                        tables.sort(key=lambda t: t.bbox[1])
                    
                    current_y = 0
                    
                    # Procesamiento intercalado (Bandas horizontales)
                    for table_obj in (tables or []):
                        bbox = table_obj.bbox  # (x0, top, x1, bottom)
                        
                        # 1. Extraer texto previo a la tabla
                        if current_y < bbox[1]:
                            crop_box = (0, current_y, page_width, bbox[1])
                            try:
                                top_crop = page.crop(crop_box)
                                top_text = top_crop.extract_text(x_tolerance=2, y_tolerance=3)
                                self._process_text_block(top_text, repetitive_artifacts, texto_completo)
                            except ValueError:
                                pass 
                                
                        # 2. Lógica de extracción tabular orientada a FILAS (Evita rotura relacional)
                        extracted_table = table_obj.extract()
                        if extracted_table and len(extracted_table) > 1:
                            # Normalizar encabezados
                            headers = [
                                str(h).replace('\n', ' ').strip() if h else f"Columna_{i+1}"
                                for i, h in enumerate(extracted_table[0])
                            ]
                            
                            # Iterar fila por fila preservando la relación Columna: Valor
                            for row in extracted_table[1:]:
                                if not row or all(cell is None or str(cell).strip() == "" for cell in row):
                                    continue 
                                    
                                row_sentences = []
                                for i, raw_cell in enumerate(row):
                                    # Omitir celdas vacías para no agregar ruido
                                    if not raw_cell or str(raw_cell).strip() == "":
                                        continue
                                        
                                    cell_val = str(raw_cell).replace('\n', ' ').strip()
                                    header = headers[i] if i < len(headers) else f"Columna_{i+1}"
                                    row_sentences.append(f"{header}: {cell_val}")
                                
                                # Si hubo datos en la fila, se convierte en una oración terminada en punto
                                if row_sentences:
                                    oracion_fila = " | ".join(row_sentences) + "."
                                    texto_completo.append(oracion_fila)
                        
                        # Mover el puntero espacial al final de la tabla
                        current_y = max(current_y, bbox[3])
                    
                    # 3. Extraer texto posterior a la última tabla
                    if current_y < page_height:
                        crop_box = (0, current_y, page_width, page_height)
                        try:
                            bottom_crop = page.crop(crop_box)
                            bottom_text = bottom_crop.extract_text(x_tolerance=2, y_tolerance=3)
                            self._process_text_block(bottom_text, repetitive_artifacts, texto_completo)
                        except ValueError:
                            pass

            # 4. Reconstrucción con Integridad Oracional
            texto_unificado = ""
            for linea in texto_completo:
                if not linea:
                    continue
                # Asegurar un espacio entre líneas al juntarlas
                texto_unificado += (" " + linea) if texto_unificado else linea
            
            # Sanitización final de espacios
            texto_unificado = re.sub(r'\s+', ' ', texto_unificado).strip()
            
            raw_documents.append({
                "doc_id": str(uuid.uuid4()),
                "raw_text": texto_unificado,
                "metadata": {
                    "atributo_adicional": "procesamiento_espacial_tabular_fila_por_fila"
                }
            })
            
        except Exception as e:
            raise RuntimeError(f"Fallo de procesamiento en {file_path}: {str(e)}")
            
        return raw_documents


if __name__ == "__main__":
    # 1. Única línea de ingreso: solicitud estricta para la ruta del archivo
    input_path = input("Ingresa la ruta del archivo PDF: ").strip()
    file_path = input_path.strip("\"'")
    
    # 2. Validaciones estrictas de extensión y existencia (Fallo Seguro -> Lista Vacía)
    if not file_path.lower().endswith('.pdf') or not os.path.exists(file_path):
        print(json.dumps([]))
    else:
        # 3. Asignación automática del fenómeno extrayendo el número de la carpeta contenedora
        directorio_padre = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
        fenomeno_asociado = "1"  # Valor por defecto seguro
        
        if "3" in directorio_padre:
            fenomeno_asociado = "3"
        elif "2" in directorio_padre:
            fenomeno_asociado = "2"
        elif "1" in directorio_padre:
            fenomeno_asociado = "1"
            
        try:
            extractor = PDFExtractor()
            obtained_docs = extractor.process(file_path, phenomenon=fenomeno_asociado)
            
            # Imprime estrictamente la LISTA completa, o una lista vacía si no hay resultados
            if obtained_docs:
                print(json.dumps(obtained_docs, indent=2, ensure_ascii=False))
            else:
                print(json.dumps([]))
        except Exception:
            # Retorna lista vacía en caso de que la ejecución falle por cualquier motivo interno
            print(json.dumps([]))