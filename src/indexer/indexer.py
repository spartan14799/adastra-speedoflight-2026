import os
import json
import faiss
import numpy as np
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("FAISS_Indexer")

# ==========================================
# 1. FAISS Strategies (Strategy Pattern)
# ==========================================

class FaissIndexStrategy(ABC):
    """Abstract base class for FAISS indexing strategies."""
    
    @abstractmethod
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Builds and returns a FAISS index from the given embeddings."""
        pass
    
    @abstractmethod
    def get_extension(self) -> str:
        """Returns the file extension for this index type."""
        pass

class FlatIndexStrategy(FaissIndexStrategy):
    """Exact Search (Inner Product for Cosine Similarity)."""
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # Using IP for cosine similarity
        index.add(embeddings)
        return index
    
    def get_extension(self) -> str:
        return "flat"

class IVFFlatIndexStrategy(FaissIndexStrategy):
    """Inverted File with Exact Post-Verification."""
    
    def __init__(self, nlist: int = 100):
        self.nlist = nlist

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        num_samples = embeddings.shape[0]
        
        # Adjust nlist if dataset is too small for the requested centroids
        actual_nlist = min(self.nlist, max(1, int(np.sqrt(num_samples))))
        
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, actual_nlist, faiss.METRIC_INNER_PRODUCT)
        
        if not index.is_trained:
            logger.info(f"Training IVF index with {actual_nlist} centroids...")
            index.train(embeddings)
            
        index.add(embeddings)
        return index

    def get_extension(self) -> str:
        return "ivfflat"

class HNSWIndexStrategy(FaissIndexStrategy):
    """Hierarchical Navigable Small World graph search."""
    
    def __init__(self, m: int = 32):
        self.m = m

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.add(embeddings)
        return index

    def get_extension(self) -> str:
        return "hnsw"


# ==========================================
# 2. Embedder (Encapsulation)
# ==========================================

class TextEmbedder:
    """Handles loading the HuggingFace model and generating embeddings."""
    
    def __init__(self, encoder_name: str):
        logger.info(f"Loading transformer model: {encoder_name} (This may take a moment)...")
        start_time = time.time()
        self.model = SentenceTransformer(encoder_name)
        logger.info(f"Model {encoder_name} loaded in {time.time() - start_time:.2f} seconds.")
        
    def embed(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Starting to embed {len(texts)} chunks...")
        start_time = time.time()
        
        # ADD batch_size=16 (or even 8 if it still freezes)
        embeddings = self.model.encode(
            texts, 
            batch_size=16, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        logger.info("Normalizing embeddings for Cosine Similarity...")
        faiss.normalize_L2(embeddings)
        
        logger.info(f"Embedding completed in {time.time() - start_time:.2f} seconds.")
        return embeddings


# ==========================================
# 3. Main Processor (Orchestrator)
# ==========================================

class VectorDatabaseBuilder:
    """Reads JSONL, generates embeddings, and applies FAISS strategies."""
    
    def __init__(self, strategies: List[FaissIndexStrategy]):
        self.strategies = strategies

    def parse_jsonl(self, filepath: Path) -> Tuple[Dict[str, Any], List[str]]:
        """Parses the JSONL file to extract metadata header and chunk texts."""
        metadata = {}
        texts = []
        
        logger.info(f"Parsing JSONL file: {filepath}")
        start_time = time.time()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                if i == 0 and data.get("type") == "encoder_header":
                    metadata = data
                else:
                    texts.append(data.get("texto", ""))
                
                # Log progress every 5,000 lines so you know it's not frozen
                if i > 0 and i % 100 == 0:
                    logger.info(f"  ... Parsed {i} lines so far.")
                    
        elapsed = time.time() - start_time
        logger.info(f"Finished parsing {len(texts)} texts in {elapsed:.2f} seconds.")
        return metadata, texts

    def process_directory(self, base_dir: str, limit: int = None):
        """Iterates through encoder folders, embeds, and saves indexes."""
        base_path = Path(base_dir)
        
        if not base_path.exists():
            logger.error(f"Directory {base_path} does not exist!")
            return
            
        for encoder_dir in base_path.iterdir():
            if not encoder_dir.is_dir():
                continue
                
            jsonl_path = encoder_dir / "metadata.jsonl"
            if not jsonl_path.exists():
                logger.warning(f"No metadata.jsonl found in {encoder_dir}. Skipping.")
                continue
                
            logger.info(f"=== Processing Directory: {encoder_dir.name} ===")
            
            # 1. Parse JSONL
            header, texts = self.parse_jsonl(jsonl_path)
            if not header or "encoder_name" not in header:
                logger.error(f"Invalid or missing header in {jsonl_path}. Skipping.")
                continue
                
            encoder_name = header["encoder_name"]
            
            if limit:
                logger.info(f"Limit applied: slicing down to {limit} chunks.")
                texts = texts[:limit]
            
            # 2. Embed Data
            embedder = TextEmbedder(encoder_name)
            embeddings = embedder.embed(texts)
            
            # 3 & 4. Generate and save indices
            for strategy in self.strategies:
                index_ext = strategy.get_extension()
                logger.info(f"Building {index_ext.upper()} index...")
                build_start = time.time()
                
                index = strategy.build_index(embeddings)
                
                output_path = encoder_dir / f"index.faiss.{index_ext}"
                faiss.write_index(index, str(output_path))
                
                build_elapsed = time.time() - build_start
                logger.info(f"Saved {index_ext.upper()} index to {output_path} (Took {build_elapsed:.2f}s)")
            
            logger.info(f"=== Completed Directory: {encoder_dir.name} ===\n")