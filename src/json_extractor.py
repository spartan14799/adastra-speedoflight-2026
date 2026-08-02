import abc
import json
import re
import os
import uuid
from typing import Any, Dict, List
from langdetect import detect, LangDetectException

class BaseExtractor(abc.ABC):
    """
    Clase abstracta base.
    """
    def __init__(self, limite_palabras: int = 250):
        self.limite_palabras = limite_palabras

    @abc.abstractmethod
    def extraer_documentos(self, ruta_archivo: str) -> List[Dict[str, Any]]:
        pass

    def limpiar_texto(self, texto: str) -> str:
        if not texto:
            return ""
        texto_limpio = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', texto)
        patrones_boilerplate = r'(?i)^\s*(página \d+ de \d+|copyright \d+|derechos reservados|all rights reserved)\b.*$'
        texto_limpio = re.sub(patrones_boilerplate, '', texto_limpio, flags=re.MULTILINE)
        texto_limpio = re.sub(r'\n{3,}', '\n\n', texto_limpio)
        texto_limpio = re.sub(r'[ \t]+', ' ', texto_limpio)
        return texto_limpio.strip()

    def fragmentar_texto(self, texto: str) -> List[str]:
        if not texto:
            return []
        oraciones = re.split(r'(?<=[.?!])\s+', texto)
        chunks, chunk_actual = [], []
        palabras_actuales = 0

        for oracion in oraciones:
            oracion = oracion.strip()
            if not oracion: continue
                
            palabras_oracion = len(oracion.split())
            
            if palabras_oracion > self.limite_palabras:
                if chunk_actual:
                    chunks.append(" ".join(chunk_actual))
                    chunk_actual, palabras_actuales = [], 0
                palabras_truncadas = oracion.split()[:self.limite_palabras]
                chunks.append(" ".join(palabras_truncadas) + ".")
                continue

            if palabras_actuales + palabras_oracion > self.limite_palabras:
                chunks.append(" ".join(chunk_actual))
                chunk_actual = [oracion]
                palabras_actuales = palabras_oracion
            else:
                chunk_actual.append(oracion)
                palabras_actuales += palabras_oracion

        if chunk_actual:
            chunks.append(" ".join(chunk_actual))
        return chunks

    def procesar(self, ruta_archivo: str, fenomeno: int = 1) -> List[Dict[str, Any]]:
        """
        Orquestador ajustado para inyectar los 8 campos obligatorios de la Tabla 1.
        Nota: Se agrega el parámetro 'fenomeno' (default 1).
        """
        documentos_extraidos = self.extraer_documentos(ruta_archivo)
        resultados_finales = []
        nombre_fuente = os.path.basename(ruta_archivo)

        for doc in documentos_extraidos:
            texto_crudo = doc.get("texto_bruto", "")
            texto_limpio = self.limpiar_texto(texto_crudo)
            
            try:
                idioma = detect(texto_limpio) if texto_limpio else "es"
            except LangDetectException:
                idioma = "es"
                
            fragmentos = self.fragmentar_texto(texto_limpio)
            metadata_base = doc.get("metadata", {})
            tipo_fuente = metadata_base.get("formato", "json")
            doc_id = doc.get("doc_id", str(uuid.uuid4()))

            for i, fragmento in enumerate(fragmentos):
                num_tokens = len(fragmento.split()) # Proxy aceptable para tokens sin tokenizer
                chunk_id = f"{doc_id}_chk_{str(i).zfill(3)}"
                
                # Inyección ESTRICTA de la Tabla 1 (Sección 3.4)
                metadata_chunk = metadata_base.copy()
                metadata_chunk["doc_id"] = doc_id
                metadata_chunk["chunk_id"] = chunk_id
                metadata_chunk["fuente"] = nombre_fuente
                metadata_chunk["formato"] = tipo_fuente
                metadata_chunk["fenomeno"] = fenomeno
                metadata_chunk["posicion"] = i
                metadata_chunk["num_tokens"] = num_tokens
                metadata_chunk["texto"] = fragmento
                
                # Contrato de Salida
                esquema_salida = {
                    "id_chunk": chunk_id,
                    "fuente": nombre_fuente,
                    "tipo_fuente": tipo_fuente,
                    "idioma": idioma,
                    "texto": fragmento,
                    "metadata": metadata_chunk
                }
                
                resultados_finales.append(esquema_salida)

        return resultados_finales


class JSONExtractor(BaseExtractor):
    """
    Clase concreta para ingesta de archivos JSON.
    Alineada con la Sección 2.1 y serialización estricta.
    """
    def _procesar_articulo(self, item: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = str(item.get("doc_id") or item.get("id") or uuid.uuid4())
        bloques_texto = []
        
        if item.get("title"):
            bloques_texto.append(f"{str(item['title']).strip()}.")
            
        if "sections" in item and isinstance(item["sections"], list):
            for sec in item["sections"]:
                if isinstance(sec, dict):
                    if "heading" in sec and sec["heading"]:
                        bloques_texto.append(f"{str(sec['heading']).strip()}.")
                    if "paragraphs" in sec and isinstance(sec["paragraphs"], list):
                        parrafos = [str(p).strip() for p in sec["paragraphs"] if str(p).strip()]
                        bloques_texto.append(" ".join(parrafos))
                        
        if "lists" in item and isinstance(item["lists"], list):
            elementos_lista = [f"{str(li).strip()}." for li in item["lists"] if str(li).strip()]
            bloques_texto.append(" ".join(elementos_lista))

        if item.get("body_text"):
            bloques_texto.append(str(item["body_text"]).strip())
            
        if item.get("body_paragraphs"):
            parrafos = item["body_paragraphs"]
            if isinstance(parrafos, list):
                bloques_texto.append(" ".join(str(p).strip() for p in parrafos if str(p).strip()))
            else:
                bloques_texto.append(str(parrafos).strip())

        texto_bruto = " ".join(bloques_texto)

        # Serialización limpia de metadata (evitando el error de str(list))
        llaves_texto = {"title", "sections", "lists", "body_text", "body_paragraphs", "id", "doc_id"}
        metadata = {}
        
        for k, v in item.items():
            if k not in llaves_texto:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    metadata[k] = v
                else:
                    # Convierte arrays anidados en strings JSON válidos
                    metadata[k] = json.dumps(v, ensure_ascii=False)
                
        metadata["formato"] = "json"
        
        return {
            "doc_id": doc_id,
            "texto_bruto": texto_bruto,
            "metadata": metadata
        }
    
    def _desempaquetar_objetos(self, data: Any) -> List[Dict[str, Any]]:
        documentos = []
        if isinstance(data, dict):
            documentos.append(self._procesar_articulo(data))
        elif isinstance(data, list):
            for elemento in data:
                documentos.extend(self._desempaquetar_objetos(elemento))
        else:
            documentos.append({
                "doc_id": str(uuid.uuid4()),
                "texto_bruto": str(data),
                "metadata": {"formato": "json"}
            })
        return documentos

    def extraer_documentos(self, ruta_archivo: str) -> List[Dict[str, Any]]:
        documentos_crudos = []
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"El archivo {ruta_archivo} no existe.")

        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                primera_linea = f.readline().strip()
                f.seek(0)
                try:
                    json.loads(primera_linea)
                    for linea in f:
                        if linea.strip():
                            obj = json.loads(linea)
                            documentos_crudos.extend(self._desempaquetar_objetos(obj))
                except json.JSONDecodeError:
                    f.seek(0)
                    data_completa = json.load(f)
                    documentos_crudos.extend(self._desempaquetar_objetos(data_completa))
        except Exception as e:
            raise RuntimeError(f"Falla de procesamiento en {ruta_archivo}: {str(e)}")

        return documentos_crudos

if __name__ == "__main__":
    print("==========================================================")
    print("   CODEFEST AD ASTRA 2026 - INGESTA DE FUENTES JSON      ")
    print("==========================================================")
    
    extractor = JSONExtractor()

    while True:
        ruta_input = input("\nIngresa la ruta de tu archivo JSON/JSONL local (o 'q' para salir):\n> ").strip()
        #C:\Users\Flia_Padilla_Camargo\Documents\adastra-speedoflight-2026\data\raw

        if ruta_input.lower() == 'q':
            print("Saliendo del pipeline...")
            break

        # Limpiar comillas si el usuario arrastró el archivo a la consola
        ruta_archivo = ruta_input.strip("\"'")

        if not os.path.exists(ruta_archivo):
            print(f"[!] Error: No se encontró el archivo en '{ruta_archivo}'. Intenta nuevamente.")
            continue

        try:
            print(f"\n[+] Procesando e higienizando datos de: {ruta_archivo}...")
            chunks_obtenidos = extractor.procesar(ruta_archivo)

            print(f"[+] Proceso completado exitosamente.")
            print(f"[+] Total de chunks generados (<= 250 palabras cada uno): {len(chunks_obtenidos)}")

            if chunks_obtenidos:
                print("\n=== MUESTRA DEL PRIMER CHUNK GENERADO (CONTRATO DE DATOS) ===")
                print(json.dumps(chunks_obtenidos[0], indent=2, ensure_ascii=False))

            print("\n" + "=" * 58)

        except Exception as e:
            print(f"\n[!] Error durante el procesamiento: {str(e)}")