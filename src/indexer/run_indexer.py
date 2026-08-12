# src/indexer/run_indexer.py

import sys
import logging
from pathlib import Path

# --- Path Resolution ---
# This finds the absolute path of the directory containing this script (src/indexer)
current_dir = Path(__file__).resolve().parent
# Navigate up two levels to get the Project Root directory
project_root = current_dir.parent.parent

# Add the project root to sys.path so Python can find the 'src' module natively
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can safely import using absolute paths from the root
from src.indexer.indexer import (
    VectorDatabaseBuilder,
    FlatIndexStrategy,
    IVFFlatIndexStrategy,
    HNSWIndexStrategy
)

# ==========================================
# Main Script Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MainIndexer")


def main():
    logger.info("🚀 Starting FAISS Indexing Pipeline on real data...")

    # Point to the real data folder using the absolute project root path
    base_vectorial_dir = project_root / "entrega" / "base_vectorial"

    # Safety check
    if not base_vectorial_dir.exists():
        logger.error(f"Target directory not found at: {base_vectorial_dir}")
        return

    logger.info(f"Target directory located: {base_vectorial_dir}")

    # 1. Instantiate the FAISS strategies
    strategies = [
        FlatIndexStrategy(),
        IVFFlatIndexStrategy(nlist=100), 
        HNSWIndexStrategy(m=32)
    ]

    # 2. Initialize the Builder
    builder = VectorDatabaseBuilder(strategies=strategies)

    # 3. Execute Pipeline
    try:
        logger.info("Executing processing pipeline across all encoder folders...")
        builder.process_directory(str(base_vectorial_dir))
        
        logger.info("✅ Pipeline completed successfully! All FAISS indices have been generated and saved.")
        
    except Exception as e:
        logger.error(f"❌ An error occurred during processing: {e}", exc_info=True)


if __name__ == "__main__":
    main()