import os
import sys
import gc
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

import torch

# ==========================================
# Parche de Incompatibilidad PyTorch DirectML
# ==========================================
# DirectML falla con 'RuntimeError: Cannot set version_counter for inference tensor'
# al operar en torch.inference_mode(). Redirigimos a torch.no_grad().

class _SafeInferenceMode:
    def __init__(self, mode=True):
        if callable(mode):
            self._func = mode
            self._mode = True
        else:
            self._func = None
            self._mode = mode

    def __enter__(self):
        self._cm = torch.no_grad() if self._mode else torch.enable_grad()
        return self._cm.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._cm.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, *args, **kwargs):
        if self._func is not None:
            with torch.no_grad():
                return self._func(*args, **kwargs)
        if args and callable(args[0]):
            fn = args[0]
            def decorated(*a, **kw):
                with torch.no_grad():
                    return fn(*a, **kw)
            return decorated
        return self

torch.inference_mode = _SafeInferenceMode

# ==========================================
# Importaciones Secundarias (posteriores al parche)
# ==========================================
import faiss
import numpy as np
import orjson
from sentence_transformers import SentenceTransformer

# Intentar importar DirectML para soporte GPU AMD en Windows
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ==========================================
# Configuración de Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("FAISS_Indexer")


# ==========================================
# Detección Dinámica de Dispositivo
# ==========================================

def get_optimal_device() -> torch.device | str:
    """
    Detecta de forma dinámica el mejor hardware disponible:
    1. CUDA / ROCm (Linux AMD o GPUs NVIDIA)
    2. DirectML (Windows AMD)
    3. CPU (Fallback)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"⚡ GPU detectada: {device_name} (CUDA/ROCm)")
        return "cuda"
    elif HAS_DIRECTML and torch_directml.is_available():
        device = torch_directml.device()
        logger.info(f"⚡ GPU AMD detectada vía DirectML: {device}")
        return device
    else:
        logger.warning("⚠️ No se detectó GPU acelerada. Se utilizará CPU.")
        return "cpu"


# ==========================================
# Estrategias FAISS (Patrón Strategy)
# ==========================================

class FaissIndexStrategy(ABC):
    """Clase base abstracta para las estrategias de indexación en FAISS."""
    
    @abstractmethod
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Construye y retorna un índice FAISS a partir de los embeddings."""
        pass
    
    @abstractmethod
    def get_extension(self) -> str:
        """Retorna la extensión del archivo para este tipo de índice."""
        pass


class FlatIndexStrategy(FaissIndexStrategy):
    """Búsqueda exacta (Producto Punto para Similitud Coseno)."""
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index
    
    def get_extension(self) -> str:
        return "flat"


class IVFFlatIndexStrategy(FaissIndexStrategy):
    """Inverted File con post-verificación exacta."""
    
    def __init__(self, nlist: int = 100):
        self.nlist = nlist

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        num_samples = embeddings.shape[0]
        
        actual_nlist = min(self.nlist, max(1, int(np.sqrt(num_samples))))
        
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, actual_nlist, faiss.METRIC_INNER_PRODUCT)
        
        if not index.is_trained:
            logger.info(f"Entrenando índice IVF con {actual_nlist} centroides...")
            index.train(embeddings)
            
        index.add(embeddings)
        return index

    def get_extension(self) -> str:
        return "ivfflat"


class HNSWIndexStrategy(FaissIndexStrategy):
    """Hierarchical Navigable Small World (Búsqueda basada en grafos)."""
    
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
# Vectorizador (TextEmbedder)
# ==========================================

class TextEmbedder:
    """Carga modelos HuggingFace y genera embeddings usando aceleración GPU."""
    
    def __init__(self, encoder_name: str, device: Optional[Any] = None, use_fp16: bool = True):
        self.device = device if device is not None else get_optimal_device()
        self.use_fp16 = use_fp16
        
        logger.info(f"Cargando modelo Transformer: {encoder_name} en dispositivo [{self.device}]...")
        start_time = time.time()
        
        self.model = SentenceTransformer(encoder_name, device=str(self.device))

        self.model.max_seq_length = 256
        logger.info(f"Longitud máxima de secuencia configurada a: {self.model.max_seq_length} tokens.")
        
        if self.use_fp16 and str(self.device) != "cpu":
            try:
                self.model.half()
                logger.info("Precisión FP16 (Half) habilitada exitosamente para la GPU.")
            except Exception as e:
                logger.warning(f"No se pudo habilitar FP16 ({e}). Fallback a FP32.")
                self.use_fp16 = False
                
        logger.info(f"Modelo {encoder_name} cargado en {time.time() - start_time:.2f} segundos.")
        
    def _cleanup_vram(self):
        """Limpia la memoria caché de GPU y fuerza la recolección de basura."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def embed(
        self, 
        texts: List[str], 
        batch_size: int = 128, 
        log_interval_sec: float = 5.0
    ) -> np.ndarray:
        """
        Genera embeddings por lotes acelerados por GPU con reportes periódicos de avance.
        """
        total_chunks = len(texts)
        logger.info(
            f"Iniciando vectorización de {total_chunks} fragmentos "
            f"(Batch Size: {batch_size} | FP16: {self.use_fp16} | Device: {self.device})..."
        )
        
        start_time = time.time()
        last_log_time = start_time
        embeddings_list = []

        try:
            with torch.no_grad():
                for i in range(0, total_chunks, batch_size):
                    batch = texts[i : i + batch_size]
                    
                    batch_emb = self.model.encode(
                        batch,
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=False
                    )
                    
                    embeddings_list.append(batch_emb.astype(np.float32, copy=False))

                    current_chunk = min(i + batch_size, total_chunks)
                    now = time.time()

                    if (now - last_log_time >= log_interval_sec) or (current_chunk == total_chunks):
                        elapsed = now - start_time
                        speed = current_chunk / elapsed if elapsed > 0 else 0
                        pct = (current_chunk / total_chunks) * 100
                        remaining_chunks = total_chunks - current_chunk
                        eta_sec = remaining_chunks / speed if speed > 0 else 0

                        logger.info(
                            f"Progreso: {current_chunk}/{total_chunks} ({pct:.1f}%) | "
                            f"Velocidad: {speed:.1f} chunks/s | "
                            f"Transcurrido: {elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m"
                        )
                        last_log_time = now

        except Exception as e:
            logger.error(f"Error durante el proceso de generación de embeddings: {e}")
            self._cleanup_vram()
            raise e

        logger.info("Combinando lotes y aplicando normalización L2 para Similitud Coseno...")
        embeddings = np.vstack(embeddings_list).astype(np.float32, copy=False)
        faiss.normalize_L2(embeddings)
        
        self._cleanup_vram()
        
        total_elapsed = time.time() - start_time
        logger.info(f"Vectorización completada en {total_elapsed/60:.2f} minutos.")
        return embeddings


# ==========================================
# Procesador Principal (Orquestador)
# ==========================================

class VectorDatabaseBuilder:
    """Lee el JSONL mediante orjson, genera embeddings y construye los índices FAISS."""
    
    def __init__(self, strategies: List[FaissIndexStrategy]):
        self.strategies = strategies

    def parse_jsonl(
        self, 
        filepath: Path, 
        log_interval_sec: float = 5.0
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Parsea archivos JSONL a alta velocidad utilizando orjson."""
        metadata = {}
        texts = []
        
        logger.info(f"Lectura ultrarrápida de JSONL: {filepath}")
        start_time = time.time()
        last_log_time = start_time
        
        with open(filepath, 'rb') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                data = orjson.loads(line)
                if i == 0 and data.get("type") == "encoder_header":
                    metadata = data
                else:
                    texts.append(data.get("texto", ""))
                
                now = time.time()
                if now - last_log_time >= log_interval_sec:
                    logger.info(f" ... Leyendo JSONL: {len(texts)} fragmentos procesados (línea {i + 1}).")
                    last_log_time = now

        elapsed = time.time() - start_time
        logger.info(f"Parseo de {len(texts)} textos finalizado en {elapsed:.2f} segundos.")
        return metadata, texts

    def process_directory(
        self, 
        base_dir: str, 
        limit: Optional[int] = None, 
        batch_size: int = 128,
        device: Optional[Any] = None
    ):
        """Procesa los directorios de encoders, vectoriza en GPU y almacena los índices."""
        base_path = Path(base_dir)
        
        if not base_path.exists():
            logger.error(f"¡El directorio {base_path} no existe!")
            return
            
        for encoder_dir in base_path.iterdir():
            if not encoder_dir.is_dir():
                continue
                
            jsonl_path = encoder_dir / "metadata.jsonl"
            if not jsonl_path.exists():
                logger.warning(f"No se encontró metadata.jsonl en {encoder_dir}. Omitiendo.")
                continue
                
            logger.info(f"=== Procesando Directorio: {encoder_dir.name} ===")
            
            # Parsing optimizado del archivo JSONL
            header, texts = self.parse_jsonl(jsonl_path, log_interval_sec=5.0)
            if not header or "encoder_name" not in header:
                logger.error(f"Encabezado inválido o ausente en {jsonl_path}. Omitiendo.")
                continue
                
            encoder_name = header["encoder_name"]
            
            if limit:
                logger.info(f"Límite aplicado: recortando a {limit} fragmentos.")
                texts = texts[:limit]
            
            # Generación de Embeddings acelerada por GPU
            embedder = TextEmbedder(encoder_name=encoder_name, device=device)
            embeddings = embedder.embed(texts, batch_size=batch_size, log_interval_sec=5.0)
            
            # Construcción y almacenamiento de índices FAISS
            for strategy in self.strategies:
                index_ext = strategy.get_extension()
                logger.info(f"Construyendo índice {index_ext.upper()}...")
                build_start = time.time()
                
                index = strategy.build_index(embeddings)
                
                output_path = encoder_dir / f"index.faiss.{index_ext}"
                faiss.write_index(index, str(output_path))
                
                build_elapsed = time.time() - build_start
                logger.info(f"Guardado {index_ext.upper()} en {output_path} (Tomó {build_elapsed:.2f}s)")
            
            logger.info(f"=== Finalizado Directorio: {encoder_dir.name} ===\n")
