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
# 2. Embedder (Encapsulation with Periodic Logging)
# ==========================================

class TextEmbedder:
    """Handles loading the HuggingFace model and generating embeddings."""
    
    def __init__(self, encoder_name: str):
        logger.info(f"Loading transformer model: {encoder_name} (This may take a moment)...")
        start_time = time.time()
        self.model = SentenceTransformer(encoder_name)
        logger.info(f"Model {encoder_name} loaded in {time.time() - start_time:.2f} seconds.")
        
    def embed(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        log_interval_sec: float = 5.0
    ) -> np.ndarray:
        """
        Embeds texts in batches with periodic 5-second progress logging.
        """
        total_chunks = len(texts)
        logger.info(f"Starting embedding for {total_chunks} chunks (Batch size: {batch_size})...")
        
        start_time = time.time()
        last_log_time = start_time
        embeddings_list = []

        for i in range(0, total_chunks, batch_size):
            batch = texts[i : i + batch_size]
            
            # Encode mini-batch
            batch_emb = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_emb)

            current_chunk = min(i + batch_size, total_chunks)
            now = time.time()

            # Trigger a log message every 5 seconds (and at 100% completion)
            if (now - last_log_time >= log_interval_sec) or (current_chunk == total_chunks):
                elapsed = now - start_time
                speed = current_chunk / elapsed if elapsed > 0 else 0
                pct = (current_chunk / total_chunks) * 100
                remaining_chunks = total_chunks - current_chunk
                eta_sec = remaining_chunks / speed if speed > 0 else 0

                logger.info(
                    f"⏳ Progress: {current_chunk}/{total_chunks} chunks ({pct:.1f}%) | "
                    f"Speed: {speed:.1f} chunks/s | "
                    f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m"
                )
                last_log_time = now

        logger.info("Combining batch embeddings and normalizing for Cosine Similarity...")
        embeddings = np.vstack(embeddings_list).astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        total_elapsed = time.time() - start_time
        logger.info(f"✅ Embedding completed in {total_elapsed/60:.2f} minutes.")
        return embeddings


# ==========================================
# 3. Main Processor (Orchestrator)
# ==========================================

class VectorDatabaseBuilder:
    """Reads JSONL, generates embeddings, and applies FAISS strategies."""
    
    def __init__(self, strategies: List[FaissIndexStrategy]):
        self.strategies = strategies

    def parse_jsonl(
        self, 
        filepath: Path, 
        log_interval_sec: float = 5.0
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Parses JSONL file with periodic 5-second progress status."""
        metadata = {}
        texts = []
        
        logger.info(f"Parsing JSONL file: {filepath}")
        start_time = time.time()
        last_log_time = start_time
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                if i == 0 and data.get("type") == "encoder_header":
                    metadata = data
                else:
                    texts.append(data.get("texto", ""))
                
                # Check periodic log every 5 seconds
                now = time.time()
                if now - last_log_time >= log_interval_sec:
                    logger.info(f" ... Still parsing JSONL: {len(texts)} chunks read so far (line {i + 1}).")
                    last_log_time = now

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
            
            # 1. Parse JSONL with 5s updates
            header, texts = self.parse_jsonl(jsonl_path, log_interval_sec=5.0)
            if not header or "encoder_name" not in header:
                logger.error(f"Invalid or missing header in {jsonl_path}. Skipping.")
                continue
                
            encoder_name = header["encoder_name"]
            
            if limit:
                logger.info(f"Limit applied: slicing down to {limit} chunks.")
                texts = texts[:limit]
            
            # 2. Embed Data with 5s status updates
            embedder = TextEmbedder(encoder_name)
            embeddings = embedder.embed(texts, batch_size=32, log_interval_sec=5.0)
            
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