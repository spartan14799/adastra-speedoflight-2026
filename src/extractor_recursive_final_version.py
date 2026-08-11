import abc
import os
import re
import uuid
import json
import tempfile
import warnings
from collections import Counter
from typing import Any, Dict, List, Set, Optional

# Manejo de dependencias (High-Performance RAG)
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
import pdfplumber
import pytesseract
from PIL import Image

# Supresión de warnings espaciales comunes para mantener logs limpios
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")


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


class CSVExtractor(BaseExtractor):
    @property
    def tipo_fuente(self) -> str:
        return "csv"

    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")
            
        try:
            chunk_size = 5000
            all_rows_text = []
            filas_procesadas = 0
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=str, keep_default_na=False):
                for _, row in chunk.iterrows():
                    row_elements = []
                    for col_name, value in row.items():
                        val_str = str(value).strip()
                        if val_str:
                            row_elements.append(f"{str(col_name).strip()}: {val_str}")
                            
                    if row_elements:
                        all_rows_text.append(" | ".join(row_elements) + ".")
                        filas_procesadas += 1
                        
            full_raw_text = " ".join(all_rows_text)
            
            return [{
                "doc_id": str(uuid.uuid4()),
                "raw_text": full_raw_text,
                "metadata": {
                    "atributo_adicional": f"total_filas_consolidadas_{filas_procesadas}"
                }
            }]
                        
        except Exception as e:
            raise RuntimeError(f"Fallo de procesamiento estructurado en {file_path}: {str(e)}")


class HTMLExtractor(BaseExtractor):
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
                
            soup = BeautifulSoup(html_content, 'html.parser')
            
            title_tag = soup.find('title')
            doc_title = title_tag.get_text(strip=True) if title_tag else "html_sin_titulo"
            
            tags_to_decompose = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'noscript', 'iframe']
            for tag in soup.find_all(tags_to_decompose):
                tag.decompose()
                
            etiquetas_permitidas = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']
            text_blocks = []
            
            for element in soup.find_all(etiquetas_permitidas):
                clean_content = element.get_text(separator=' ', strip=True)
                clean_content = re.sub(r'[ \t]+', ' ', clean_content)
                if clean_content:
                    if not clean_content.endswith(('.', '?', '!')):
                        clean_content += "."
                    text_blocks.append(clean_content)
            
            visible_text = " ".join(text_blocks)
            
            raw_documents.append({
                "doc_id": str(uuid.uuid4()),
                "raw_text": visible_text,
                "metadata": {
                    "atributo_adicional": doc_title
                }
            })
        except Exception as e:
            raise RuntimeError(f"Falla de procesamiento en {file_path}: {str(e)}")
            
        return raw_documents


class JSONExtractor(BaseExtractor):
    @property
    def tipo_fuente(self) -> str:
        return "json"

    def _recursive_extract(self, obj: Any, prefix: str = "") -> List[str]:
        blocks = []
        ignore_keys = {"id", "doc_id", "uuid", "_id", "created_at", "updated_at", 
                       "deleted_at", "timestamp", "version", "status", "type", 
                       "format", "width", "height", "size", "index"}

        if isinstance(obj, dict):
            for k, v in obj.items():
                k_lower = str(k).lower()
                
                if k_lower in ignore_keys or k_lower.endswith("_id") or k_lower.endswith("id"):
                    continue
                
                new_prefix = str(k).replace('_', ' ').strip().capitalize()
                blocks.extend(self._recursive_extract(v, new_prefix))
                
        elif isinstance(obj, list):
            for item in obj:
                blocks.extend(self._recursive_extract(item, prefix))
                
        elif isinstance(obj, str):
            val = obj.strip()
            if not val:
                return blocks
                
            val = re.sub(r'\s+', ' ', val)
            
            if val.startswith("http://") or val.startswith("https://"):
                block = f"Enlace de {prefix}: {val}" if prefix else f"Enlace: {val}"
                if not block.endswith(('.', '?', '!')):
                    block += "."
                blocks.append(block)
                return blocks

            palabras = val.split()
            prefijo_lower = prefix.lower()
            
            claves_titulo = ["titul", "title", "name", "nombre", "tema", "subject", "encabezado", "header", "categoria", "category"]
            es_titulo = any(clave in prefijo_lower for clave in claves_titulo)
            es_parrafo = len(palabras) >= 5
            
            if es_titulo or es_parrafo:
                palabras_genericas = ["texto", "text", "body", "content", "contenido", "paragraphs", "parrafos", "value", "valor", "data"]
                
                if prefix and prefijo_lower not in palabras_genericas:
                    if len(palabras) > 20: 
                        block = val
                    else:
                        block = f"{prefix}: {val}"
                else:
                    block = val
                
                if not block.endswith(('.', '?', '!')):
                    block += "."
                blocks.append(block)
                
        return blocks

    def _process_article(self, item: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = str(item.get("doc_id") or item.get("id") or uuid.uuid4())
        title = str(item.get("title") or item.get("titulo") or item.get("name") or "Untitled").strip()
        
        text_blocks = []
        if title != "Untitled":
            title_block = f"Título: {title}."
            text_blocks.append(title_block)
            
        extracted_blocks = self._recursive_extract(item)
        
        for block in extracted_blocks:
            if block not in text_blocks:
                text_blocks.append(block)
                
        raw_text = " ".join(text_blocks)
        
        return {
            "doc_id": doc_id,
            "raw_text": raw_text,
            "metadata": {
                "atributo_adicional": title
            }
        }
    
    def _unpack_objects(self, data: Any) -> List[Dict[str, Any]]:
        documents = []
        if isinstance(data, dict):
            documents.append(self._process_article(data))
        elif isinstance(data, list):
            for element in data:
                documents.extend(self._unpack_objects(element))
        else:
            if isinstance(data, str) and len(data.split()) >= 3:
                documents.append({
                    "doc_id": str(uuid.uuid4()),
                    "raw_text": str(data),
                    "metadata": {"atributo_adicional": "json_data"}
                })
        return documents

    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        raw_documents = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)
                try:
                    json.loads(first_line)
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            raw_documents.extend(self._unpack_objects(obj))
                except json.JSONDecodeError:
                    f.seek(0)
                    complete_data = json.load(f)
                    raw_documents.extend(self._unpack_objects(complete_data))
        except Exception as e:
            raise RuntimeError(f"Processing failure in {file_path}: {str(e)}")
        return raw_documents

    def process(self, file_path: str, phenomenon: str = "1") -> List[Dict[str, Any]]:
        try:
            if not file_path or not os.path.exists(file_path):
                return []

            extracted_documents = self.extract_documents(file_path)
            if not extracted_documents:
                return []

            source_name = os.path.basename(file_path)
            textos_combinados = []
            
            doc_id = str(uuid.uuid4())
            if extracted_documents and "doc_id" in extracted_documents[0]:
                doc_id = str(extracted_documents[0]["doc_id"])

            for doc in extracted_documents:
                raw_text = doc.get("raw_text", "")
                cleaned = self.clean_text(raw_text)
                if cleaned:
                    textos_combinados.append(cleaned)

            texto_final = " ".join(textos_combinados)
            texto_final = self.clean_text(texto_final)

            if not texto_final: return []

            try:
                language = detect(texto_final[:2000])
                if language not in ['es', 'en', 'pt']: language = 'es'
            except LangDetectException:
                language = "es"

            output_schema = {
                "texto": texto_final,
                "metadata": {
                    "total_palabras": len(texto_final.split()),
                    "atributo_adicional": "json_consolidado",
                    "fuente": source_name,
                    "tipo_fuente": self.tipo_fuente,
                    "idioma": language,
                    "doc_id": doc_id,
                    "Fenomeno": phenomenon
                }
            }
            return [output_schema]
        except Exception as e:
            raise RuntimeError(f"Fallo en procesamiento de {file_path}: {str(e)}")


class PBFExtractor(BaseExtractor):
    @property
    def tipo_fuente(self) -> str:
        return "pbf"

    def _convert_pbf_to_json(self, pbf_path: str) -> str:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("La biblioteca 'geopandas' es obligatoria para procesar archivos .pbf")

        try:
            gdf = gpd.read_file(pbf_path)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8')
            temp_file.close()
            gdf.to_file(temp_file.name, driver="GeoJSON")
            return temp_file.name
        except Exception as e:
            raise RuntimeError(f"Fallo en la conversión vectorial PBF -> JSON: {str(e)}")

    def _process_geojson_feature(self, item: Dict[str, Any], collection_name: str = "") -> Dict[str, Any]:
        properties = item.get("properties", {}) or {}
        doc_id = str(properties.get("b_ID_concatenated") or properties.get("au_ID_concatenated") or item.get("id") or properties.get("fid") or uuid.uuid4())
        
        country = properties.get("au_country") or properties.get("b_adm1_geral") or ""
        level1 = properties.get("au_level1") or properties.get("b_ADM1_PT") or ""
        level2 = properties.get("au_level2") or properties.get("b_ADM2_PT") or ""
        
        location_parts = [p for p in [level2, level1, country] if p]
        loc_str = ", ".join(location_parts) if location_parts else "Ubicación no especificada"
        title = f"Registro Geográfico: {loc_str}"
        if collection_name:
            title = f"{collection_name} - {title}"
            
        text_blocks = [f"{title}."]
        
        geom = item.get("geometry")
        geom_type = "N/A"
        if isinstance(geom, dict):
            geom_type = geom.get("type", "N/A")
            coords = geom.get("coordinates")
            if geom_type.lower() == "point" and isinstance(coords, list) and len(coords) >= 2:
                lat = round(float(coords[1]), 4) if isinstance(coords[1], (int, float)) else coords[1]
                lon = round(float(coords[0]), 4) if isinstance(coords[0], (int, float)) else coords[0]
                text_blocks.append(f"Punto de interés espacial ubicado en las coordenadas: latitud {lat}, longitud {lon}.")
            elif geom_type.lower() in ["polygon", "multipolygon"]:
                num_nodos = str(coords).count("]") // 2 if coords else 0
                text_blocks.append(f"Delimitación espacial estructurada como {geom_type}. Esta área geográfica está definida por un polígono cerrado compuesto por aproximadamente {num_nodos} nodos topológicos.")
            elif geom_type.lower() in ["linestring", "multilinestring"]:
                num_nodos = str(coords).count("]") // 2 if coords else 0
                text_blocks.append(f"Estructura lineal geográfica de tipo {geom_type}, trazada a través de {num_nodos} puntos de referencia.")

        if "geometry" in item: del item["geometry"]
        if "geometry" in properties: del properties["geometry"]

        kv_pairs = []
        narrative_blocks = []
        for k, v in properties.items():
            if v is None or v == "": continue
            str_v = str(v).strip()
            if not str_v: continue
            k_lower = str(k).lower()
            if any(term in k_lower for term in ["popup", "description", "resumen", "texto", "notes"]):
                cleaned_val = re.sub(r'[\r\n]+', ' ', str_v)
                cleaned_val = re.sub(r'\s+', ' ', cleaned_val).strip()
                if not cleaned_val.endswith(('.', '?', '!')): cleaned_val += "."
                narrative_blocks.append(f"Información Detallada ({k}): {cleaned_val}")
            else:
                kv_pairs.append(f"{k}: {str_v}")
                
        if kv_pairs:
            structured_text = " | ".join(kv_pairs)
            if not structured_text.endswith(('.', '?', '!')): structured_text += "."
            text_blocks.append(structured_text)
            
        if narrative_blocks: text_blocks.extend(narrative_blocks)
        raw_text = " ".join(text_blocks)
        
        return {"doc_id": doc_id, "raw_text": raw_text, "metadata": {"atributo_adicional": geom_type}}

    def _unpack_objects(self, data: Any) -> List[Dict[str, Any]]:
        documents = []
        if isinstance(data, dict) and data.get("type") == "FeatureCollection" and isinstance(data.get("features"), list):
            collection_name = str(data.get("name", "PBF_FeatureCollection"))
            for feature in data["features"]:
                if isinstance(feature, dict):
                    feat_doc = self._process_geojson_feature(feature, collection_name=collection_name)
                    if feat_doc.get("raw_text"):
                        documents.append(feat_doc)
        return documents

    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        if not file_path or not os.path.exists(file_path):
            return []
        if not file_path.lower().endswith(".pbf"):
            return []
        raw_documents = []
        target_path = None
        try:
            target_path = self._convert_pbf_to_json(file_path)
            with open(target_path, 'r', encoding='utf-8') as f:
                complete_data = json.load(f)
                raw_documents.extend(self._unpack_objects(complete_data))
        except Exception:
            return []
        finally:
            if target_path and os.path.exists(target_path):
                os.remove(target_path)
        return raw_documents

    def process(self, file_path: str, phenomenon: str = "1") -> List[Dict[str, Any]]:
        try:
            if not file_path or not os.path.exists(file_path):
                return []

            extracted_documents = self.extract_documents(file_path)
            if not extracted_documents:
                return []

            source_name = os.path.basename(file_path)
            textos_combinados = []
            doc_id = str(uuid.uuid4())

            for doc in extracted_documents:
                raw_text = doc.get("raw_text", "")
                cleaned = self.clean_text(raw_text)
                if cleaned:
                    textos_combinados.append(cleaned)
                doc_id = str(doc.get("doc_id") or doc_id)

            texto_final = " ".join(textos_combinados)
            texto_final = self.clean_text(texto_final)

            if not texto_final: return []

            try:
                language = detect(texto_final[:2000])
                if language not in ['es', 'en', 'pt']: language = 'es'
            except LangDetectException:
                language = "es"

            output_schema = {
                "texto": texto_final,
                "metadata": {
                    "total_palabras": len(texto_final.split()),
                    "atributo_adicional": "pbf_document",
                    "fuente": source_name,
                    "tipo_fuente": self.tipo_fuente,
                    "idioma": language,
                    "doc_id": doc_id,
                    "Fenomeno": phenomenon
                }
            }
            return [output_schema]
        except Exception:
            return []


class PDFExtractor(BaseExtractor):
    @property
    def tipo_fuente(self) -> str:
        return "pdf"

    def _identify_repetitive_artifacts(self, pdf: 'pdfplumber.PDF') -> Set[str]:
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
        if not text: return
        for linea in text.split('\n'):
            linea_limpia = linea.strip()
            if not linea_limpia: continue
            norm_line = re.sub(r'\d+', '', linea_limpia).lower()
            
            if norm_line in repetitive_artifacts: continue
            if re.search(r'(?i)^(p[áa]gina\s+\d+|page\s+\d+)', linea_limpia): continue
                
            if not linea_limpia[-1] in '.?!;:,':
                linea_limpia += "."
                
            output_list.append(linea_limpia)

    def extract_documents(self, file_path: str) -> List[Dict[str, Any]]:
        raw_documents = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")
            
        try:
            texto_completo = []
            flag_procesamiento = "procesamiento_espacial_tabular_fila_por_fila"
            
            with pdfplumber.open(file_path) as pdf:
                repetitive_artifacts = self._identify_repetitive_artifacts(pdf)
                for page in pdf.pages:
                    # Intento de extracción nativa directa
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=3)

                    # RUTINA FALLBACK OCR: Si el documento es un escaneo impreso, extract_text será None o muy corto.
                    if not page_text or len(page_text.strip()) < 50:
                        try:
                            # Renderizado de la página a imagen alta resolución
                            pil_img = page.to_image(resolution=300).original
                            
                            # Intentamos detectar español por defecto (asumido para el dominio AD ASTRA)
                            try:
                                ocr_text = pytesseract.image_to_string(pil_img, lang='spa')
                            except Exception:
                                ocr_text = pytesseract.image_to_string(pil_img) # Fallback sin flag de idioma estricto

                            if ocr_text and ocr_text.strip():
                                self._process_text_block(ocr_text, repetitive_artifacts, texto_completo)
                                flag_procesamiento = "procesamiento_ocr_documento_escaneado"
                                
                        except Exception as e:
                            # Falla silenciosa si Tesseract no está configurado en el sistema
                            pass 
                        
                        # Al ser una imagen escaneada, no habrá objetos de tabla nativa para iterar,
                        # saltamos al siguiente ciclo.
                        continue
                    
                    # --- RUTINA ORIGINAL (Pdfs Nativos) ---
                    tables = page.find_tables()
                    page_height = page.height
                    page_width = page.width
                    
                    if tables:
                        tables.sort(key=lambda t: t.bbox[1])
                    current_y = 0
                    
                    for table_obj in (tables or []):
                        bbox = table_obj.bbox
                        if current_y < bbox[1]:
                            crop_box = (0, current_y, page_width, bbox[1])
                            try:
                                top_crop = page.crop(crop_box)
                                top_text = top_crop.extract_text(x_tolerance=2, y_tolerance=3)
                                self._process_text_block(top_text, repetitive_artifacts, texto_completo)
                            except ValueError:
                                pass 
                                
                        extracted_table = table_obj.extract()
                        if extracted_table and len(extracted_table) > 1:
                            headers = [str(h).replace('\n', ' ').strip() if h else f"Columna_{i+1}" for i, h in enumerate(extracted_table[0])]
                            for row in extracted_table[1:]:
                                if not row or all(cell is None or str(cell).strip() == "" for cell in row):
                                    continue 
                                row_sentences = []
                                for i, raw_cell in enumerate(row):
                                    if not raw_cell or str(raw_cell).strip() == "": continue
                                    cell_val = str(raw_cell).replace('\n', ' ').strip()
                                    header = headers[i] if i < len(headers) else f"Columna_{i+1}"
                                    row_sentences.append(f"{header}: {cell_val}")
                                if row_sentences:
                                    oracion_fila = " | ".join(row_sentences) + "."
                                    texto_completo.append(oracion_fila)
                        
                        current_y = max(current_y, bbox[3])
                    
                    if current_y < page_height:
                        crop_box = (0, current_y, page_width, page_height)
                        try:
                            bottom_crop = page.crop(crop_box)
                            bottom_text = bottom_crop.extract_text(x_tolerance=2, y_tolerance=3)
                            self._process_text_block(bottom_text, repetitive_artifacts, texto_completo)
                        except ValueError:
                            pass

            texto_unificado = ""
            for linea in texto_completo:
                if not linea: continue
                texto_unificado += (" " + linea) if texto_unificado else linea
            
            texto_unificado = re.sub(r'\s+', ' ', texto_unificado).strip()
            
            raw_documents.append({
                "doc_id": str(uuid.uuid4()),
                "raw_text": texto_unificado,
                "metadata": {
                    "atributo_adicional": flag_procesamiento
                }
            })
            
        except Exception as e:
            raise RuntimeError(f"Fallo de procesamiento en {file_path}: {str(e)}")
            
        return raw_documents


# ==========================================
# 6. ENRUTADOR PRINCIPAL (Router)
# ==========================================
def obtener_ruta_relativa_desde_raw(ruta_actual: str, input_dir_base: str) -> str:
    """
    Busca la carpeta 'raw' en la ruta absoluta y devuelve todo lo que esté después de ella.
    Si por algún motivo no la encuentra, usa la ruta relativa clásica.
    """
    ruta_norm = os.path.normpath(ruta_actual)
    partes = ruta_norm.split(os.sep)
    partes_lower = [p.lower() for p in partes]
    
    if 'raw' in partes_lower:
        raw_idx = len(partes_lower) - 1 - partes_lower[::-1].index('raw')
        partes_relativas = partes[raw_idx + 1:]
        
        if partes_relativas:
            return os.path.join(*partes_relativas)
        else:
            return "" 
    else:
        return os.path.relpath(ruta_norm, input_dir_base)


if __name__ == "__main__":
    input_dir = input("Ingrese la ruta de la carpeta con los archivos a procesar (ej. ...\\data\\raw): ").strip().strip("\"'")
    output_dir = input("Ingrese la ruta de la carpeta para guardar los resultados (.json) (ej. ...\\data\\processed): ").strip().strip("\"'")
    
    input_dir = os.path.abspath(os.path.normpath(input_dir))
    output_dir = os.path.abspath(os.path.normpath(output_dir))
    
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"Error: La ruta de entrada no existe o no es una carpeta ({input_dir})")
        exit(1)
        
    print("\n--- Replicando estructura de carpetas desde '\\raw' en adelante ---")
    
    for root, dirs, files in os.walk(input_dir):
        rel_dir = obtener_ruta_relativa_desde_raw(root, input_dir)
        target_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)
        
    supported_exts = {'csv', 'html', 'htm', 'json', 'pbf', 'pdf'}
    archivos_procesados = 0
    archivos_fallidos = 0
    
    print("\n--- Iniciando Procesamiento de Archivos ---")
    
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            ext = file_path.lower().split('.')[-1]
            
            if ext not in supported_exts:
                continue
                
            # Asignación de fenómeno basada estrictamente en el nombre de la carpeta base
            fenomeno_asociado = "1"
            file_path_lower = file_path.lower()
            
            if "f1_ia_y_capacidades_estrategicas" in file_path_lower:
                fenomeno_asociado = "1"
            elif "f2_seguridad_entorno_espacial" in file_path_lower:
                fenomeno_asociado = "2"
            elif "f3_dinamicas_territoriales" in file_path_lower:
                fenomeno_asociado = "3"
            
            extractor = None
            if ext == 'csv': extractor = CSVExtractor()
            elif ext in ['html', 'htm']: extractor = HTMLExtractor()
            elif ext == 'json': extractor = JSONExtractor()
            elif ext == 'pbf': extractor = PBFExtractor()
            elif ext == 'pdf': extractor = PDFExtractor()
            
            try:
                # Procesamiento polimórfico garantizando el Contrato de Datos
                obtained_docs = extractor.process(file_path, phenomenon=fenomeno_asociado)
                
                if obtained_docs:
                    rel_dir = obtener_ruta_relativa_desde_raw(root, input_dir)
                    target_dir = os.path.join(output_dir, rel_dir)
                    
                    base_name = os.path.splitext(file_name)[0]
                    out_file_path = os.path.join(target_dir, f"{base_name}.json")
                    
                    with open(out_file_path, 'w', encoding='utf-8') as f:
                        json.dump(obtained_docs, f, ensure_ascii=False, indent=4)
                        
                    archivos_procesados += 1
                    print(f"Procesado exitosamente: {os.path.basename(file_path)} -> Guardado en: {os.path.relpath(out_file_path, output_dir)}")
                else:
                    print(f"Sin contenido útil extraído: {file_path}")
                    
            except Exception as e:
                archivos_fallidos += 1
                print(f"Error procesando {file_path}: {e}")

    print("\n--- Resumen de Procesamiento ---")
    print(f"Archivos procesados correctamente: {archivos_procesados}")
    print(f"Archivos con fallos: {archivos_fallidos}")