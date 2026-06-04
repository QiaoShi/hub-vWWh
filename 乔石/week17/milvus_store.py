import uuid
from typing import List, Tuple, Optional, Any
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    IndexType,
    MetricType
)
import logging

logger = logging.getLogger(__name__)


class MilvusStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        collection_name: str = "semantic_cache",
        dimension: int = 1536
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection = None
        
    def connect(self):
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            logger.info("Successfully connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def create_collection(self, overwrite: bool = False):
        if self.collection_name in connections.list_collections():
            if overwrite:
                logger.info(f"Collection {self.collection_name} exists, dropping it")
                Collection(self.collection_name).drop()
            else:
                self.collection = Collection(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
                return
        
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
                auto_id=False
            ),
            FieldSchema(
                name="question",
                dtype=DataType.VARCHAR,
                max_length=2000
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64
            )
        ]
        
        schema = CollectionSchema(fields, description="Semantic Cache Collection")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        index_params = {
            "metric_type": MetricType.COSINE,
            "index_type": IndexType.HNSW,
            "params": {
                "M": 16,
                "efConstruction": 256
            }
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"Created collection: {self.collection_name}")
    
    def insert(
        self,
        question: str,
        embedding: List[float],
        created_at: int,
        cache_id: Optional[str] = None
    ) -> str:
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        if cache_id is None:
            cache_id = str(uuid.uuid4())
        
        entities = [
            [cache_id],
            [question],
            [embedding],
            [created_at]
        ]
        
        self.collection.insert(entities)
        self.collection.flush()
        
        logger.debug(f"Inserted cache entry: {cache_id}")
        return cache_id
    
    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        search_params: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        if search_params is None:
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
        
        self.collection.load()
        
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id"]
        )
        
        hits = []
        for hit in results[0]:
            hits.append((hit.id, hit.distance))
        
        logger.debug(f"Search completed, found {len(hits)} results")
        return hits
    
    def delete(self, cache_ids: List[str]) -> int:
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        expr = f"id in {cache_ids}"
        result = self.collection.delete(expr)
        
        logger.info(f"Deleted {len(cache_ids)} entries")
        return len(cache_ids)
    
    def delete_expired(self, before_timestamp: int) -> int:
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        expr = f"created_at < {before_timestamp}"
        result = self.collection.delete(expr)
        
        deleted_count = result.delete_count if hasattr(result, 'delete_count') else 0
        logger.info(f"Deleted {deleted_count} expired entries")
        return deleted_count
    
    def count(self) -> int:
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        return self.collection.num_entities
    
    def close(self):
        if connections.has_connection("default"):
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
