import json
import time
from typing import Optional, Dict, Any, List
import redis
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
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
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get_cache(self, cache_id: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"semantic_cache:{cache_id}"
        data = self.client.get(key)
        
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode cache data for {cache_id}")
                return None
        
        return None
    
    def get_caches(self, cache_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        if not cache_ids:
            return []
        
        keys = [f"semantic_cache:{cache_id}" for cache_id in cache_ids]
        results = self.client.mget(keys)
        
        return [
            json.loads(data) if data else None
            for data in results
        ]
    
    def set_cache(
        self,
        cache_id: str,
        question: str,
        answer: str,
        ttl: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"semantic_cache:{cache_id}"
        data = {
            "question": question,
            "answer": answer,
            "created_at": int(time.time()),
            "ttl": ttl,
            "hit_count": 0,
            "metadata": metadata or {}
        }
        
        try:
            self.client.setex(key, ttl, json.dumps(data))
            logger.debug(f"Set cache for {cache_id} with TTL {ttl}")
            return True
        except Exception as e:
            logger.error(f"Failed to set cache for {cache_id}: {e}")
            return False
    
    def increment_hit_count(self, cache_id: str) -> int:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"semantic_cache:{cache_id}"
        data = self.client.get(key)
        
        if data:
            try:
                cache_data = json.loads(data)
                cache_data["hit_count"] = cache_data.get("hit_count", 0) + 1
                ttl = cache_data.get("ttl", 86400)
                self.client.setex(key, ttl, json.dumps(cache_data))
                return cache_data["hit_count"]
            except json.JSONDecodeError:
                logger.error(f"Failed to decode cache data for {cache_id}")
        
        return 0
    
    def delete_cache(self, cache_id: str) -> bool:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"semantic_cache:{cache_id}"
        result = self.client.delete(key)
        logger.debug(f"Deleted cache {cache_id}: {result > 0}")
        return result > 0
    
    def delete_caches(self, cache_ids: List[str]) -> int:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        if not cache_ids:
            return 0
        
        keys = [f"semantic_cache:{cache_id}" for cache_id in cache_ids]
        result = self.client.delete(*keys)
        logger.info(f"Deleted {result} caches")
        return result
    
    def increment_stat(self, stat_name: str, increment: int = 1):
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"semantic_cache:stats:{stat_name}"
        self.client.incrby(key, increment)
    
    def get_stats(self) -> Dict[str, int]:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        stats = {}
        for stat in ["total_queries", "cache_hits", "cache_misses"]:
            value = self.client.get(f"semantic_cache:stats:{stat}")
            stats[stat] = int(value) if value else 0
        
        stats["hit_rate"] = (
            stats["cache_hits"] / stats["total_queries"]
            if stats["total_queries"] > 0 else 0.0
        )
        
        return stats
    
    def clear_all(self) -> int:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        keys = self.client.keys("semantic_cache:*")
        if keys:
            result = self.client.delete(*keys)
            logger.info(f"Cleared {result} cache entries")
            return result
        
        return 0
    
    def is_expired(self, cache_id: str) -> bool:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = f"semantic_cache:{cache_id}"
        ttl = self.client.ttl(key)
        
        return ttl == -2 or ttl == 0
    
    def close(self):
        if self.client:
            self.client.close()
            logger.info("Closed Redis connection")
