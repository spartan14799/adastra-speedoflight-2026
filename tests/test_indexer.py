import os
import json
import shutil
import logging
from pathlib import Path

# Import from our source module
from src.indexer import (
    VectorDatabaseBuilder,
    FlatIndexStrategy,
    IVFFlatIndexStrategy,
    HNSWIndexStrategy
)

# ==========================================
# Test Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TestRunner")


def create_mock_data(test_dir: Path):
    """Creates a temporary mock structure matching your project tree."""
    encoder_dir = test_dir / "encoder_bge-m3"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = encoder_dir / "metadata.jsonl"
    
    logger.info(f"Creating mock data at {jsonl_path}...")
    
    # 1 header + 100 dummy lines mimicking your provided data
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        # Header
        f.write('{"type": "encoder_header", "encoder_name": "BAAI/bge-m3", "max_words": 250, "overlap_sentences": 1}\n')
        # Data chunks
        for i in range(100):
            chunk = {
                "doc_id": "04dccdd9-test",
                "chunk_id": f"chunk-{i:03d}",
                "texto": f"This is a dummy sentence number {i} to test the AI indexing report pipeline."
            }
            f.write(json.dumps(chunk) + '\n')
            
    return test_dir


def test_pipeline():
    logger.info("--- Starting FAISS Indexer Test ---")
    
    # Setup paths
    base_test_path = Path(__file__).parent / "mock_base_vectorial"
    
    # Clean up previous test runs if they exist
    if base_test_path.exists():
        logger.info("Cleaning up previous test data...")
        shutil.rmtree(base_test_path)
        
    create_mock_data(base_test_path)
    
    # Instantiate the strategies we want to test
    # Note: We use a small nlist for IVFFlat since we only have 100 mock samples
    strategies = [
        FlatIndexStrategy(),
        IVFFlatIndexStrategy(nlist=5), 
        HNSWIndexStrategy(m=16)
    ]
    
    # Initialize our main Builder with the strategies
    builder = VectorDatabaseBuilder(strategies=strategies)
    
    # Run the pipeline
    logger.info("Executing pipeline on mock directory...")
    builder.process_directory(str(base_test_path), limit=100)
    
    # Validate outputs exist
    logger.info("Validating output files...")
    encoder_dir = base_test_path / "encoder_bge-m3"
    
    try:
        assert (encoder_dir / "index.faiss.flat").exists(), "Flat index missing!"
        assert (encoder_dir / "index.faiss.ivfflat").exists(), "IVF-Flat index missing!"
        assert (encoder_dir / "index.faiss.hnsw").exists(), "HNSW index missing!"
        logger.info("✅ All tests passed! Indexes generated successfully in the test folder.")
    except AssertionError as e:
        logger.error(f"❌ Test Failed: {e}")


if __name__ == "__main__":
    test_pipeline()