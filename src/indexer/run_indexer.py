# src/indexer/run_indexer.py

import sys
import time
import logging
from pathlib import Path

# --- Path Resolution ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.indexer.indexer import (
    VectorDatabaseBuilder,
    FlatIndexStrategy,
    IVFFlatIndexStrategy,
    HNSWIndexStrategy
)

# ==========================================
# Main Script Logging Configuration
# ==========================================
# Using a cleaner, more readable format for the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MainIndexer")


def main():
    print("\n" + "="*70)
    logger.info("🚀 STARTING FAISS INDEXING PIPELINE ON REAL DATA")
    print("="*70)
    
    script_start_time = time.time()

    # Point to the real data folder
    base_vectorial_dir = project_root / "entrega" / "base_vectorial"

    logger.info("🔍 STEP 1: Locating target directory...")
    if not base_vectorial_dir.exists():
        logger.error(f"❌ Target directory not found at: {base_vectorial_dir}")
        return
    logger.info(f"✅ Directory found: {base_vectorial_dir}")

    # --- Pre-scan to tell the user exactly what we are about to do ---
    logger.info("🔍 STEP 2: Scanning for encoder folders...")
    encoder_dirs = [d for d in base_vectorial_dir.iterdir() if d.is_dir()]
    
    if not encoder_dirs:
        logger.warning("⚠️ No encoder directories found! Please check your folder structure.")
        return
        
    logger.info(f"📊 Found {len(encoder_dirs)} encoder folder(s):")
    for d in encoder_dirs:
        json_file = d / "metadata.jsonl"
        status = "✅ Has metadata.jsonl" if json_file.exists() else "❌ MISSING metadata.jsonl"
        logger.info(f"   -> {d.name} ({status})")

    print("-" * 70)
    logger.info("⚙️  STEP 3: Configuring FAISS Strategies...")
    
    strategies = [
        FlatIndexStrategy(),
        IVFFlatIndexStrategy(nlist=100), 
        HNSWIndexStrategy(m=32)
    ]
    
    for s in strategies:
        logger.info(f"   - Enabled Strategy: {s.get_extension().upper()}")

    builder = VectorDatabaseBuilder(strategies=strategies)

    print("-" * 70)
    logger.info("🔥 STEP 4: Executing Processing Pipeline...")
    logger.info("(Note: Embedding massive datasets can take time. Watch the progress bar below.)\n")
    
    try:
        # This will trigger the logic inside indexer.py (which handles the batching progress bar)
        builder.process_directory(str(base_vectorial_dir))
        
        print("\n" + "="*70)
        total_time = time.time() - script_start_time
        mins, secs = divmod(total_time, 60)
        logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY IN {int(mins)}m {int(secs)}s")
        logger.info("All FAISS indices have been generated and saved to your encoder folders.")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        logger.error(f"❌ FATAL ERROR DURING PROCESSING: {e}", exc_info=True)
        print("="*70 + "\n")


if __name__ == "__main__":
    main()