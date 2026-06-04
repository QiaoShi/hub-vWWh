import json
import hashlib
import time
from typing import Optional, List, Tuple, Dict, Any, Callable
import redis
import logging

logger = logging.getLogger(__name__)


class EmbeddingsCache:
    """
    Embedding缓存类
    
    职责：
    - 缓存文本到向量的映射，避免重复计算
    - 支持多模型缓存隔离
    - 支持TTL过期
    """
    
    KEY_PREFIX = "embedding_cache"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_model: str = "text-embedding-3-small",
        default_dimension: int = 1536,
        default_ttl: int = 604800
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_model = default_model
        self.default_dimension = default_dimension
        self.default_ttl = default_ttl
        self.client = None
    
    def connect(self):
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            self.client.ping()
            logger.info("Successfully connected to Redis for EmbeddingsCache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _generate_key(self, text: str, model: Optional[str] = None) -> str:
        model_name = model or self.default_model
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
        return f"{self.KEY_PREFIX}:{model_name}:{text_hash}"
    
    def _serialize(self, data: Dict[str, Any]) -> str:
        return json.dumps(data)
    
    def _deserialize(self, data: Optional[str]) -> Optional[Dict[str, Any]]:
        if data is None:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.error("Failed to deserialize cache data")
            return None
    
    def get(
        self, 
        text: str, 
        model: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        获取缓存的embedding
        
        Args:
            text: 文本内容
            model: 模型名称（可选）
            
        Returns:
            缓存的向量，如果不存在返回None
        """
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = self._generate_key(text, model)
        data = self.client.get(key)
        
        if data:
            cached = self._deserialize(data)
            if cached:
                logger.debug(f"Embedding cache hit for text: {text[:30]}...")
                return cached.get("embedding")
        
        return None
    
    def set(
        self,
        text: str,
        embedding: List[float],
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        缓存embedding
        
        Args:
            text: 文本内容
            embedding: 向量
            model: 模型名称（可选）
            dimension: 向量维度（可选）
            ttl: 过期时间（秒）
            
        Returns:
            是否成功
        """
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = self._generate_key(text, model)
        model_name = model or self.default_model
        cache_ttl = ttl if ttl is not None else self.default_ttl
        cache_dimension = dimension if dimension is not None else self.default_dimension
        
        data = {
            "text": text,
            "embedding": embedding,
            "model": model_name,
            "dimension": cache_dimension,
            "created_at": int(time.time()),
            "ttl": cache_ttl
        }
        
        try:
            self.client.setex(key, cache_ttl, self._serialize(data))
            logger.debug(f"Cached embedding for text: {text[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
            return False
    
    def get_or_compute(
        self,
        text: str,
        embed_func: Callable[[str], List[float]],
        model: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> Tuple[List[float], bool]:
        """
        获取或计算embedding（自动处理缓存逻辑）
        
        Args:
            text: 文本内容
            embed_func: embedding计算函数
            model: 模型名称（可选）
            ttl: 过期时间（秒）
            
        Returns:
            (embedding, is_cache_hit): 向量和是否缓存命中
        """
        self._increment_stat("total_requests")
        
        cached_embedding = self.get(text, model)
        
        if cached_embedding is not None:
            self._increment_stat("cache_hits")
            return cached_embedding, True
        
        self._increment_stat("cache_misses")
        
        embedding = embed_func(text)
        
        self.set(text, embedding, model, ttl=ttl)
        
        return embedding, False
    
    def batch_get(
        self, 
        texts: List[str], 
        model: Optional[str] = None
    ) -> List[Optional[List[float]]]:
        """
        批量获取缓存的embedding
        
        Args:
            texts: 文本列表
            model: 模型名称（可选）
            
        Returns:
            向量列表（未命中的位置为None）
        """
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        if not texts:
            return []
        
        keys = [self._generate_key(text, model) for text in texts]
        
        pipe = self.client.pipeline()
        for key in keys:
            pipe.get(key)
        results = pipe.execute()
        
        embeddings = []
        for i, data in enumerate(results):
            if data:
                cached = self._deserialize(data)
                if cached:
                    embeddings.append(cached.get("embedding"))
                    continue
            embeddings.append(None)
        
        logger.debug(f"Batch get: {len([e for e in embeddings if e])}/{len(texts)} hits")
        return embeddings
    
    def batch_set(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        ttl: Optional[int] = None
    ) -> int:
        """
        批量缓存embedding
        
        Args:
            texts: 文本列表
            embeddings: 向量列表
            model: 模型名称（可选）
            dimension: 向量维度（可选）
            ttl: 过期时间（秒）
            
        Returns:
            成功缓存的数量
        """
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        
        if not texts:
            return 0
        
        model_name = model or self.default_model
        cache_ttl = ttl if ttl is not None else self.default_ttl
        cache_dimension = dimension if dimension is not None else self.default_dimension
        created_at = int(time.time())
        
        pipe = self.client.pipeline()
        
        for text, embedding in zip(texts, embeddings):
            key = self._generate_key(text, model)
            data = {
                "text": text,
                "embedding": embedding,
                "model": model_name,
                "dimension": cache_dimension,
                "created_at": created_at,
                "ttl": cache_ttl
            }
            pipe.setex(key, cache_ttl, self._serialize(data))
        
        pipe.execute()
        
        logger.info(f"Batch set {len(texts)} embeddings")
        return len(texts)
    
    def delete(self, text: str, model: Optional[str] = None) -> bool:
        """
        删除指定文本的缓存
        
        Args:
            text: 文本内容
            model: 模型名称（可选）
            
        Returns:
            是否成功删除
        """
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = self._generate_key(text, model)
        result = self.client.delete(key)
        
        return result > 0
    
    def delete_by_model(self, model: str) -> int:
        """
        删除指定模型的所有缓存
        
        Args:
            model: 模型名称
            
        Returns:
            删除的数量
        """
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        pattern = f"{self.KEY_PREFIX}:{model}:*"
        keys = []
        
        for key in self.client.scan_iter(match=pattern):
            keys.append(key)
        
        if keys:
            deleted = self.client.delete(*keys)
            logger.info(f"Deleted {deleted} caches for model {model}")
            return deleted
        
        return 0
    
    def clear_all(self) -> int:
        """
        清空所有embedding缓存
        
        Returns:
            删除的数量
        """
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        pattern = f"{self.KEY_PREFIX}:*"
        keys = []
        
        for key in self.client.scan_iter(match=pattern):
            keys.append(key)
        
        if keys:
            deleted = self.client.delete(*keys)
            logger.info(f"Cleared {deleted} embedding caches")
            return deleted
        
        return 0
    
    def _increment_stat(self, stat_name: str, increment: int = 1):
        """增加统计计数"""
        if not self.client:
            return
        
        key = f"{self.KEY_PREFIX}:stats:{stat_name}"
        self.client.incrby(key, increment)
    
    def stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        stats = {}
        for stat in ["total_requests", "cache_hits", "cache_misses"]:
            value = self.client.get(f"{self.KEY_PREFIX}:stats:{stat}")
            stats[stat] = int(value) if value else 0
        
        stats["hit_rate"] = (
            stats["cache_hits"] / stats["total_requests"]
            if stats["total_requests"] > 0 else 0.0
        )
        
        pattern = f"{self.KEY_PREFIX}:*:*"
        count = 0
        for _ in self.client.scan_iter(match=pattern):
            count += 1
        stats["total_cached"] = count
        
        return stats
    
    def close(self):
        """关闭Redis连接"""
        if self.client:
            self.client.close()
            logger.info("Closed EmbeddingsCache Redis connection")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
