import time
from typing import Optional, Tuple, Dict, Any, Callable, List
import logging

from .milvus_store import MilvusStore
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        milvus_user: str = "",
        milvus_password: str = "",
        milvus_collection: str = "semantic_cache",
        embedding_dimension: int = 1536,
        threshold: float = 0.1,
        default_ttl: int = 86400
    ):
        self.threshold = threshold
        self.default_ttl = default_ttl
        
        self.redis_cache = RedisCache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password
        )
        
        self.milvus_store = MilvusStore(
            host=milvus_host,
            port=milvus_port,
            user=milvus_user,
            password=milvus_password,
            collection_name=milvus_collection,
            dimension=embedding_dimension
        )
        
        self._is_initialized = False
    
    def initialize(self, overwrite_collection: bool = False):
        logger.info("Initializing SemanticCache...")
        
        self.redis_cache.connect()
        self.milvus_store.connect()
        self.milvus_store.create_collection(overwrite=overwrite_collection)
        
        self._is_initialized = True
        logger.info("SemanticCache initialized successfully")
    
    def _ensure_initialized(self):
        if not self._is_initialized:
            raise RuntimeError("SemanticCache not initialized. Call initialize() first")
    
    def get(self, question: str, embedding: List[float]) -> Optional[str]:
        self._ensure_initialized()
        
        results = self.milvus_store.search(embedding, top_k=5)
        
        for cache_id, distance in results:
            if distance < self.threshold:
                cache_data = self.redis_cache.get_cache(cache_id)
                
                if cache_data:
                    if not self.redis_cache.is_expired(cache_id):
                        self.redis_cache.increment_hit_count(cache_id)
                        logger.debug(f"Cache hit for question: {question[:30]}...")
                        return cache_data["answer"]
                    else:
                        logger.debug(f"Cache expired: {cache_id}")
                        self.delete(cache_id)
        
        return None
    
    def set(
        self,
        question: str,
        answer: str,
        embedding: List[float],
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        self._ensure_initialized()
        
        cache_ttl = ttl if ttl is not None else self.default_ttl
        created_at = int(time.time())
        
        cache_id = self.milvus_store.insert(
            question=question,
            embedding=embedding,
            created_at=created_at
        )
        
        success = self.redis_cache.set_cache(
            cache_id=cache_id,
            question=question,
            answer=answer,
            ttl=cache_ttl,
            metadata=metadata
        )
        
        if not success:
            self.milvus_store.delete([cache_id])
            raise RuntimeError("Failed to set cache in Redis")
        
        logger.debug(f"Set cache for question: {question[:30]}... (id: {cache_id})")
        return cache_id
    
    def query(
        self,
        question: str,
        embedding: List[float],
        llm_func: Callable[[str], str],
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, bool]:
        self._ensure_initialized()
        
        self.redis_cache.increment_stat("total_queries")
        
        cached_answer = self.get(question, embedding)
        
        if cached_answer is not None:
            self.redis_cache.increment_stat("cache_hits")
            return cached_answer, True
        
        self.redis_cache.increment_stat("cache_misses")
        
        answer = llm_func(question)
        
        self.set(question, answer, embedding, ttl, metadata)
        
        return answer, False
    
    def delete(self, cache_id: str) -> bool:
        self._ensure_initialized()
        
        self.redis_cache.delete_cache(cache_id)
        self.milvus_store.delete([cache_id])
        
        logger.info(f"Deleted cache entry: {cache_id}")
        return True
    
    def delete_expired(self) -> int:
        self._ensure_initialized()
        
        current_time = int(time.time())
        deleted_count = self.milvus_store.delete_expired(current_time)
        
        logger.info(f"Deleted {deleted_count} expired entries from Milvus")
        return deleted_count
    
    def clear_all(self):
        self._ensure_initialized()
        
        self.redis_cache.clear_all()
        self.milvus_store.create_collection(overwrite=True)
        
        logger.info("Cleared all cache entries")
    
    def stats(self) -> Dict[str, Any]:
        self._ensure_initialized()
        
        redis_stats = self.redis_cache.get_stats()
        milvus_count = self.milvus_store.count()
        
        return {
            **redis_stats,
            "total_entries": milvus_count
        }
    
    def close(self):
        self.milvus_store.close()
        self.redis_cache.close()
        logger.info("SemanticCache closed")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
