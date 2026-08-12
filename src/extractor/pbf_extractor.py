import abc
import json
import re
import os
import uuid
import tempfile
import warnings
from typing import Any, Dict, List
from langdetect import detect, LangDetectException

from .base import BaseExtractor


# Supresión de warnings espaciales comunes para mantener logs limpios
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")


class PBFExtractor(BaseExtractor):
    """
    Extractor exclusivo para archivos .pbf. 
    Aplica consolidación forzada de colecciones geográficas y procesamiento 
    estructurado de coordenadas usando la conversión GeoJSON subyacente.
    """
    @property
    def tipo_fuente(self) -> str:
        return "pbf"

    def _convert_pbf_to_json(self, pbf_path: str) -> str:
        """
        Convierte de forma segura un archivo .pbf a .json temporal usando GeoPandas.
        Retorna la ruta del archivo temporal generado.
        """
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
        """Procesa un Feature GeoJSON transformando metadata y coordenadas a texto denso y oracional."""
        properties = item.get("properties", {}) or {}
        
        doc_id = str(
            properties.get("b_ID_concatenated") 
            or properties.get("au_ID_concatenated") 
            or item.get("id") 
            or properties.get("fid") 
            or uuid.uuid4()
        )
        
        country = properties.get("au_country") or properties.get("b_adm1_geral") or ""
        level1 = properties.get("au_level1") or properties.get("b_ADM1_PT") or ""
        level2 = properties.get("au_level2") or properties.get("b_ADM2_PT") or ""
        
        location_parts = [p for p in [level2, level1, country] if p]
        loc_str = ", ".join(location_parts) if location_parts else "Ubicación no especificada"
        
        title = f"Registro Geográfico: {loc_str}"
        if collection_name:
            title = f"{collection_name} - {title}"
            
        text_blocks = [f"{title}."]
        
        # --- INICIO MEJORA EXCLUSIVA: INTERPRETACIÓN ESPACIAL PROFUNDA ---
        geom = item.get("geometry")
        geom_type = "N/A"
        if isinstance(geom, dict):
            geom_type = geom.get("type", "N/A")
            coords = geom.get("coordinates")
            
            if geom_type.lower() == "point" and isinstance(coords, list) and len(coords) >= 2:
                # Redondeo a 4 decimales (~11m de precisión) para limpiar el ruido numérico
                lat = round(float(coords[1]), 4) if isinstance(coords[1], (int, float)) else coords[1]
                lon = round(float(coords[0]), 4) if isinstance(coords[0], (int, float)) else coords[0]
                text_blocks.append(f"Punto de interés espacial ubicado en las coordenadas: latitud {lat}, longitud {lon}.")
            
            elif geom_type.lower() in ["polygon", "multipolygon"]:
                num_nodos = str(coords).count("]") // 2 if coords else 0
                text_blocks.append(f"Delimitación espacial estructurada como {geom_type}. Esta área geográfica está definida por un polígono cerrado compuesto por aproximadamente {num_nodos} nodos topológicos.")
            
            elif geom_type.lower() in ["linestring", "multilinestring"]:
                num_nodos = str(coords).count("]") // 2 if coords else 0
                text_blocks.append(f"Estructura lineal geográfica de tipo {geom_type}, trazada a través de {num_nodos} puntos de referencia.")

        # CRÍTICO: Purgamos la geometría cruda
        if "geometry" in item:
            del item["geometry"]
        if "geometry" in properties:
            del properties["geometry"]
        # --- FIN MEJORA EXCLUSIVA ---

        kv_pairs = []
        narrative_blocks = []
        
        for k, v in properties.items():
            if v is None or v == "":
                continue
                
            str_v = str(v).strip()
            if not str_v:
                continue
                
            k_lower = str(k).lower()
            
            # Limpieza para descripciones extensas
            if any(term in k_lower for term in ["popup", "description", "resumen", "texto", "notes"]):
                cleaned_val = re.sub(r'[\r\n]+', ' ', str_v)
                cleaned_val = re.sub(r'\s+', ' ', cleaned_val).strip()
                if not cleaned_val.endswith(('.', '?', '!')):
                    cleaned_val += "."
                narrative_blocks.append(f"Información Detallada ({k}): {cleaned_val}")
            else:
                kv_pairs.append(f"{k}: {str_v}")
                
        if kv_pairs:
            structured_text = " | ".join(kv_pairs)
            if not structured_text.endswith(('.', '?', '!')):
                structured_text += "."
            text_blocks.append(structured_text)
            
        if narrative_blocks:
            text_blocks.extend(narrative_blocks)
            
        raw_text = " ".join(text_blocks)
        
        metadata = {
            "format": "pbf", 
            "geometry_type": geom_type,
            "atributo_adicional": "pbf_document"
        }
        
        return {
            "doc_id": doc_id,
            "title": title,
            "raw_text": raw_text,
            "metadata": metadata
        }

    def _unpack_objects(self, data: Any) -> List[Dict[str, Any]]:
        """Procesa exclusivamente la colección de features proveniente del PBF."""
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
        """Extrae el contenido de un archivo PBF convirtiéndolo y procesándolo higiénicamente."""
        if not file_path or not os.path.exists(file_path):
            return []
            
        if not file_path.lower().endswith(".pbf"):
            return []
            
        raw_documents = []
        target_path = None
        
        try:
            # Conversión temporal PBF -> JSON Vectorial
            target_path = self._convert_pbf_to_json(file_path)

            with open(target_path, 'r', encoding='utf-8') as f:
                # GeoPandas exporta la capa como un único objeto JSON estructurado (FeatureCollection)
                complete_data = json.load(f)
                raw_documents.extend(self._unpack_objects(complete_data))
                
        except Exception:
            # Captura higiénica de cualquier excepción para no romper la regla de retornar []
            return []
        finally:
            if target_path and os.path.exists(target_path):
                os.remove(target_path)
                
        return raw_documents


