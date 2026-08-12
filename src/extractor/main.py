import os
import json
import logging
from tqdm import tqdm

# Importamos nuestras clases limpias desde el módulo que creamos
from extractors import CSVExtractor, HTMLExtractor, JSONExtractor, PBFExtractor, PDFExtractor

# ==========================================
# CONFIGURACIÓN DEL LOG
# ==========================================
logging.basicConfig(
    filename='extraccion_rag.log', 
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def universal_extractor(file_path: str):
    """
    Recibe un archivo, determina su tipo y usa la clase correcta para extraerlo.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # Enrutar a la clase correcta
    if ext == '.pdf':
        extractor = PDFExtractor()
    elif ext == '.csv':
        extractor = CSVExtractor()
    elif ext == '.html':
        extractor = HTMLExtractor()
    elif ext == '.json':
        extractor = JSONExtractor()
    elif ext == '.pbf':
        extractor = PBFExtractor()
    else:
        return None  # Extensión no soportada, se ignora silenciosamente

    # Ejecutar la extracción
    try:
        resultado = extractor.extract(file_path)
        return resultado
    except Exception as e:
        logging.error(f"Error crítico en {file_path}: {e}")
        return None

def extractor_recursive_final_version(input_directory: str, output_directory: str):
    """
    Recorre los directorios, lista los archivos, omite los ya procesados 
    y muestra una barra de carga dinámica.
    """
    # Escaneo previo: Recolectar todas las rutas de archivos válidos
    archivos_a_procesar = []
    extensiones_validas = {'.pdf', '.csv', '.html', '.json', '.pbf'}
    
    print("Escaneando directorios en busca de archivos...")
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensiones_validas:
                archivos_a_procesar.append(os.path.join(root, file))

    total_archivos = len(archivos_a_procesar)
    if total_archivos == 0:
        print("No se encontraron archivos válidos para procesar.")
        return

    logging.info(f"=== INICIO DE SESIÓN: {total_archivos} archivos encontrados ===")

    # Configuración de la barra de progreso estilo Tmux
    # bar_format personaliza cómo se ve en la terminal
    formato_barra = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
    
    with tqdm(total=total_archivos, desc="Procesando", bar_format=formato_barra, dynamic_ncols=True) as pbar:
        for file_path in archivos_a_procesar:
            
            # Determinar la ruta relativa para replicar la estructura de carpetas
            ruta_relativa = os.path.relpath(file_path, input_directory)
            nombre_sin_ext = os.path.splitext(ruta_relativa)[0]
            ruta_salida = os.path.join(output_directory, f"{nombre_sin_ext}.json")
            
            nombre_archivo = os.path.basename(file_path)

            # Crear las subcarpetas necesarias en el output_directory
            os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

            # Lógica de Omitir si ya existe
            if os.path.exists(ruta_salida):
                logging.info(f"OMITIDO (Ya procesado): {ruta_relativa}")
                pbar.set_postfix_str(f"Omitido: {nombre_archivo[:20]}...")
                pbar.update(1)
                continue

            # Procesamiento Activo
            pbar.set_postfix_str(f"Extrayendo: {nombre_archivo[:20]}...")
            
            resultado = universal_extractor(file_path)

            if resultado is not None:
                # Guardado del archivo
                try:
                    with open(ruta_salida, 'w', encoding='utf-8') as f:
                        json.dump(resultado, f, ensure_ascii=False, indent=2)
                    logging.info(f"ÉXITO: {ruta_relativa}")
                except Exception as e:
                    logging.error(f"Error al guardar JSON para {ruta_relativa}: {e}")
            else:
                logging.warning(f"FALLO / IGNORADO: {ruta_relativa}")

            # Avanzar la barra
            pbar.update(1)

if __name__ == "__main__":
    DIRECTORIO_ENTRADA = "./data/raw"
    DIRECTORIO_SALIDA = "./data/processed"
    
    # Asegurarnos de que las carpetas existan
    os.makedirs(DIRECTORIO_ENTRADA, exist_ok=True)
    os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)
    
    # Limpiamos la terminal para que se vea más profesional (opcional)
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==================================================")
    print("MOTOR DE EXTRACCIÓN RAG INICIADO ")
    print("==================================================\n")
    print(f"Origen:  {DIRECTORIO_ENTRADA}")
    print(f"Destino: {DIRECTORIO_SALIDA}")
    print(f"Log:     extraccion_rag.log\n")
    
    extractor_recursive_final_version(DIRECTORIO_ENTRADA, DIRECTORIO_SALIDA)
    
    print("\n¡Proceso finalizado! Revisa 'extraccion_rag.log' para ver los detalles.")
