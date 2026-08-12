import sys
import time
import logging
from pathlib import Path

# --- Resolución de Rutas ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.indexer.indexer import (
    VectorDatabaseBuilder,
    FlatIndexStrategy,
    IVFFlatIndexStrategy,
    HNSWIndexStrategy,
    get_optimal_device
)

# ==========================================
# Configuración del Logger Principal
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MainIndexer")


def main():
    print("\n" + "="*70)
    logger.info("INICIANDO PIPELINE OPTIMIZADO FAISS (ACELERACIÓN GPU)")
    print("="*70)
    
    script_start_time = time.time()

    # Paso 0: Verificación del dispositivo de hardware disponible
    device = get_optimal_device()
    logger.info(f"Dispositivo de Ejecución Activo: {device}")

    # Ruta a los datos de entrada
    base_vectorial_dir = project_root / "entrega" / "base_vectorial"

    logger.info("Localizando directorio objetivo...")
    if not base_vectorial_dir.exists():
        logger.error(f"Directorio objetivo no encontrado en: {base_vectorial_dir}")
        return
    logger.info(f"Directorio encontrado: {base_vectorial_dir}")

    # Escaneo preliminar de carpetas
    logger.info("Escaneando carpetas de encoders...")
    encoder_dirs = [d for d in base_vectorial_dir.iterdir() if d.is_dir()]
    
    if not encoder_dirs:
        logger.warning("¡No se encontraron carpetas de encoders!")
        return
        
    logger.info(f"Se encontraron {len(encoder_dirs)} carpeta(s):")
    for d in encoder_dirs:
        json_file = d / "metadata.jsonl"
        status = "metadata.jsonl presente" if json_file.exists() else "FALTA metadata.jsonl"
        logger.info(f"   -> {d.name} ({status})")

    print("-" * 70)
    logger.info("Configurando Estrategias FAISS...")
    
    strategies = [
        FlatIndexStrategy(),
        IVFFlatIndexStrategy(nlist=100), 
        HNSWIndexStrategy(m=32)
    ]
    
    for s in strategies:
        logger.info(f"   - Estrategia Activada: {s.get_extension().upper()}")

    builder = VectorDatabaseBuilder(strategies=strategies)

    print("-" * 70)
    logger.info("Ejecutando Pipeline Optimizado (Batch Size: 128 | FP16 Activo)...")
    logger.info("(Revisa las métricas en tiempo real a continuación)\n")
    
    try:
        builder.process_directory(
            base_dir=str(base_vectorial_dir),
            batch_size=16,  # INFO: Cambiar dependiendo de la cantidad de vram de la grafica
            device=device
        )
        
        print("\n" + "="*70)
        total_time = time.time() - script_start_time
        mins, secs = divmod(total_time, 60)
        logger.info(f"PIPELINE FINALIZADO CON ÉXITO EN {int(mins)}m {int(secs)}s")
        logger.info("Todos los índices FAISS fueron creados y guardados correctamente.")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        logger.error(f"ERROR FATAL DURANTE LA EJECUCIÓN: {e}", exc_info=True)
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
