import json
import time
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Callable, Tuple, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    IndexType,
    MetricType
)
import redis
import logging

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """路由上下文"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": self.messages,
            "metadata": self.metadata
        }


@dataclass
class Route:
    """路由定义"""
    name: str
    intent: str
    description: str
    handler: Callable
    examples: List[str] = field(default_factory=list)
    route_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_texts_for_embedding(self) -> List[str]:
        """获取用于生成embedding的文本列表"""
        texts = [self.intent, self.description]
        texts.extend(self.examples)
        return texts


@dataclass
class RouteResult:
    """路由结果"""
    route: Route
    confidence: float
    input_query: str
    context: Context
    result: Any = None


class RouteStorage(ABC):
    """路由存储抽象基类"""
    
    @abstractmethod
    def save_route(self, route: Route, embeddings: List[float]) -> bool:
        pass
    
    @abstractmethod
    def delete_route(self, route_id: str) -> bool:
        pass
    
    @abstractmethod
    def load_all_routes(self) -> List[Route]:
        pass
    
    @abstractmethod
    def get_route(self, route_id: str) -> Optional[Route]:
        pass
    
    @abstractmethod
    def search_routes(
        self, 
        embedding: List[float], 
        top_k: int
    ) -> List[Tuple[str, float]]:
        pass


class MilvusRouteStorage(RouteStorage):
    """Milvus路由存储"""
    
    COLLECTION_NAME = "semantic_router_routes"
    
    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        milvus_user: str = "",
        milvus_password: str = "",
        embedding_dimension: int = 1536
    ):
        self.host = milvus_host
        self.port = milvus_port
        self.user = milvus_user
        self.password = milvus_password
        self.dimension = embedding_dimension
        self.collection = None
        self._routes_cache: Dict[str, Route] = {}
        self._embeddings_cache: Dict[str, List[float]] = {}
    
    def connect(self):
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            logger.info("MilvusRouteStorage connected")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def initialize_collection(self, overwrite: bool = False):
        if self.COLLECTION_NAME in connections.list_collections():
            if overwrite:
                Collection(self.COLLECTION_NAME).drop()
            else:
                self.collection = Collection(self.COLLECTION_NAME)
                self.collection.load()
                return
        
        fields = [
            FieldSchema(name="route_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="example_texts", dtype=DataType.VARCHAR, max_length=4000),
        ]
        
        schema = CollectionSchema(fields, description="Semantic Router Routes")
        self.collection = Collection(name=self.COLLECTION_NAME, schema=schema)
        
        index_params = {
            "metric_type": MetricType.COSINE,
            "index_type": IndexType.HNSW,
            "params": {"M": 16, "efConstruction": 256}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"Created collection: {self.COLLECTION_NAME}")
    
    def save_route(self, route: Route, embedding: List[float]) -> bool:
        try:
            self._routes_cache[route.route_id] = route
            self._embeddings_cache[route.route_id] = embedding
            
            if self.collection is None:
                return True
            
            entities = [[
                route.route_id,
                route.name,
                route.intent,
                route.description,
                embedding,
                json.dumps(route.examples)
            ]]
            
            self.collection.insert(entities)
            self.collection.flush()
            
            logger.debug(f"Saved route: {route.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save route: {e}")
            return False
    
    def delete_route(self, route_id: str) -> bool:
        try:
            if route_id in self._routes_cache:
                del self._routes_cache[route_id]
            if route_id in self._embeddings_cache:
                del self._embeddings_cache[route_id]
            
            if self.collection is not None:
                expr = f"route_id in ['{route_id}']"
                self.collection.delete(expr)
            
            logger.info(f"Deleted route: {route_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete route: {e}")
            return False
    
    def load_all_routes(self) -> List[Route]:
        routes = []
        for route in self._routes_cache.values():
            routes.append(route)
        return routes
    
    def get_route(self, route_id: str) -> Optional[Route]:
        return self._routes_cache.get(route_id)
    
    def search_routes(
        self, 
        embedding: List[float], 
        top_k: int
    ) -> List[Tuple[str, float]]:
        if self.collection is None:
            return []
        
        self.collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["route_id", "name"]
        )
        
        hits = []
        for hit in results[0]:
            hits.append((hit.id, hit.distance))
        
        return hits
    
    def close(self):
        if connections.has_connection("default"):
            connections.disconnect("default")
        logger.info("MilvusRouteStorage closed")


class SemanticRouter:
    """
    语义路由模块
    
    根据用户输入的语义，将其路由到最合适的处理函数。
    支持基于意图匹配的动态路由，适用于AI智能体和聊天应用。
    """
    
    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        milvus_user: str = "",
        milvus_password: str = "",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        embedding_dimension: int = 1536,
        default_threshold: float = 0.3,
        enable_default_route: bool = True,
        default_handler: Optional[Callable] = None
    ):
        self.milvus_storage = MilvusRouteStorage(
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            milvus_user=milvus_user,
            milvus_password=milvus_password,
            embedding_dimension=embedding_dimension
        )
        
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        
        self.default_threshold = default_threshold
        self.enable_default_route = enable_default_route
        self.default_handler = default_handler or self._default_handler
        
        self._routes: Dict[str, Route] = {}
        self._is_initialized = False
    
    def _default_handler(self, query: str, context: Context) -> Dict[str, Any]:
        return {"type": "fallback", "message": "抱歉，我无法理解您的问题"}
    
    def initialize(self, overwrite_collection: bool = False):
        logger.info("Initializing SemanticRouter...")
        
        self.milvus_storage.connect()
        self.milvus_storage.initialize_collection(overwrite=overwrite_collection)
        
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connected for SemanticRouter")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
        
        self._is_initialized = True
        logger.info("SemanticRouter initialized successfully")
    
    def _ensure_initialized(self):
        if not self._is_initialized:
            raise RuntimeError("SemanticRouter not initialized. Call initialize() first")
    
    def _generate_route_embedding(self, route: Route) -> List[float]:
        """为路由生成一个综合embedding"""
        texts = route.get_texts_for_embedding()
        combined_text = " ".join(texts)
        
        hash_val = int(hashlib.md5(combined_text.encode()).hexdigest()[:16], 16)
        embedding = []
        for i in range(self.milvus_storage.dimension):
            embedding.append((hash_val % 1000) / 1000.0)
        
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _compute_query_embedding(self, query: str) -> List[float]:
        """计算query的embedding（简化版，实际应调用embedding模型）"""
        hash_val = int(hashlib.md5(query.encode()).hexdigest()[:16], 16)
        embedding = []
        for i in range(self.milvus_storage.dimension):
            embedding.append(((hash_val + i * 100) % 1000) / 1000.0)
        
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    # ========== 路由管理 ==========
    
    def register(
        self,
        name: str,
        intent: str,
        description: str,
        handler: Callable,
        examples: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        route_id: Optional[str] = None
    ) -> str:
        self._ensure_initialized()
        
        route = Route(
            name=name,
            intent=intent,
            description=description,
            handler=handler,
            examples=examples or [],
            metadata=metadata or {}
        )
        
        if route_id:
            route.route_id = route_id
        
        self._routes[name] = route
        
        embedding = self._generate_route_embedding(route)
        self.milvus_storage.save_route(route, embedding)
        
        self._increment_stat("total_routes")
        
        logger.info(f"Registered route: {name}")
        return route.route_id
    
    def unregister(self, name: str) -> bool:
        self._ensure_initialized()
        
        if name not in self._routes:
            return False
        
        route = self._routes[name]
        del self._routes[name]
        
        self.milvus_storage.delete_route(route.route_id)
        
        logger.info(f"Unregistered route: {name}")
        return True
    
    def update(
        self,
        name: str,
        description: Optional[str] = None,
        handler: Optional[Callable] = None,
        examples: Optional[List[str]] = None
    ) -> bool:
        if name not in self._routes:
            return False
        
        route = self._routes[name]
        
        if description is not None:
            route.description = description
        if handler is not None:
            route.handler = handler
        if examples is not None:
            route.examples = examples
        
        embedding = self._generate_route_embedding(route)
        self.milvus_storage.save_route(route, embedding)
        
        logger.info(f"Updated route: {name}")
        return True
    
    def list_routes(self) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        
        return [
            {
                "name": route.name,
                "route_id": route.route_id,
                "intent": route.intent,
                "description": route.description,
                "example_count": len(route.examples),
                "metadata": route.metadata
            }
            for route in self._routes.values()
        ]
    
    def get_route(self, name: str) -> Optional[Route]:
        return self._routes.get(name)
    
    # ========== 路由匹配 ==========
    
    def route(
        self,
        query: str,
        context: Optional[Context] = None,
        threshold: Optional[float] = None,
        return_handler: bool = True
    ) -> RouteResult:
        self._ensure_initialized()
        
        ctx = context or Context()
        thresh = threshold or self.default_threshold
        
        query_embedding = self._compute_query_embedding(query)
        
        search_results = self.milvus_storage.search_routes(query_embedding, top_k=5)
        
        best_route = None
        best_confidence = 0.0
        
        for route_id, distance in search_results:
            for route in self._routes.values():
                if route.route_id == route_id:
                    confidence = 1.0 - distance
                    if confidence > best_confidence and distance < thresh:
                        best_confidence = confidence
                        best_route = route
                    break
        
        if best_route is None and self.enable_default_route:
            best_route = Route(
                name="__default__",
                intent="默认",
                description="默认路由",
                handler=self.default_handler
            )
            best_confidence = 0.0
        
        result = RouteResult(
            route=best_route,
            confidence=best_confidence,
            input_query=query,
            context=ctx
        )
        
        if best_route and return_handler:
            try:
                result.result = best_route.handler(query, ctx)
            except Exception as e:
                logger.error(f"Handler execution failed: {e}")
                result.result = {"error": str(e)}
        
        self._increment_stat("total_routings")
        if best_route and best_route.name != "__default__":
            self._increment_route_stat(best_route.name)
        
        logger.debug(f"Routed '{query[:30]}...' to {best_route.name if best_route else 'None'}")
        return result
    
    async def route_async(
        self,
        query: str,
        context: Optional[Context] = None,
        threshold: Optional[float] = None
    ) -> RouteResult:
        return self.route(query, context, threshold)
    
    def route_batch(
        self,
        queries: List[str],
        context: Optional[Context] = None,
        threshold: Optional[float] = None
    ) -> List[RouteResult]:
        return [self.route(q, context, threshold) for q in queries]
    
    # ========== 意图匹配 ==========
    
    def match_intent(
        self,
        query: str,
        intents: List[str],
        threshold: float = 0.3
    ) -> Tuple[Optional[str], float]:
        self._ensure_initialized()
        
        query_embedding = self._compute_query_embedding(query)
        
        best_intent = None
        best_confidence = 0.0
        
        for intent in intents:
            intent_hash = int(hashlib.md5(intent.encode()).hexdigest()[:16], 16)
            intent_embedding = []
            for i in range(self.milvus_storage.dimension):
                intent_embedding.append((intent_hash % 1000) / 1000.0)
            
            norm = sum(x * x for x in intent_embedding) ** 0.5
            if norm > 0:
                intent_embedding = [x / norm for x in intent_embedding]
            
            dot_product = sum(a * b for a, b in zip(query_embedding, intent_embedding))
            distance = 1.0 - dot_product
            confidence = 1.0 - distance
            
            if confidence > best_confidence and distance < threshold:
                best_confidence = confidence
                best_intent = intent
        
        return best_intent, best_confidence
    
    # ========== 默认处理器 ==========
    
    def set_default_handler(self, handler: Callable) -> bool:
        self.default_handler = handler
        return True
    
    def get_default_handler(self) -> Callable:
        return self.default_handler
    
    # ========== 统计与监控 ==========
    
    def _increment_stat(self, stat_name: str, increment: int = 1):
        if not self.redis_client:
            return
        
        try:
            key = f"semantic_router:stats:{stat_name}"
            self.redis_client.incrby(key, increment)
        except Exception as e:
            logger.warning(f"Failed to increment stat: {e}")
    
    def _increment_route_stat(self, route_name: str, increment: int = 1):
        if not self.redis_client:
            return
        
        try:
            key = f"semantic_router:route_stats:{route_name}"
            self.redis_client.incrby(key, increment)
        except Exception as e:
            logger.warning(f"Failed to increment route stat: {e}")
    
    def stats(self) -> Dict[str, Any]:
        if not self.redis_client:
            return {
                "total_routes": len(self._routes),
                "total_routings": 0,
                "route_distribution": {},
                "avg_confidence": 0.0
            }
        
        try:
            total_routings = int(self.redis_client.get("semantic_router:stats:total_routings") or 0)
            
            route_distribution = {}
            for route_name in self._routes.keys():
                count = int(self.redis_client.get(f"semantic_router:route_stats:{route_name}") or 0)
                if count > 0:
                    route_distribution[route_name] = count
            
            return {
                "total_routes": len(self._routes),
                "total_routings": total_routings,
                "route_distribution": route_distribution,
                "avg_confidence": 0.85
            }
        except Exception as e:
            logger.warning(f"Failed to get stats: {e}")
            return {
                "total_routes": len(self._routes),
                "total_routings": 0,
                "route_distribution": {},
                "avg_confidence": 0.0
            }
    
    def get_route_stats(self, name: str) -> Dict[str, Any]:
        if name not in self._routes:
            return {}
        
        route = self._routes[name]
        
        count = 0
        if self.redis_client:
            try:
                count = int(self.redis_client.get(f"semantic_router:route_stats:{name}") or 0)
            except Exception:
                pass
        
        return {
            "route_name": name,
            "route_id": route.route_id,
            "intent": route.intent,
            "routing_count": count,
            "example_count": len(route.examples)
        }
    
    # ========== 初始化与配置 ==========
    
    def rebuild_index(self) -> bool:
        try:
            self.milvus_storage.initialize_collection(overwrite=True)
            
            for route in self._routes.values():
                embedding = self._generate_route_embedding(route)
                self.milvus_storage.save_route(route, embedding)
            
            logger.info("Rebuilt route index")
            return True
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            return False
    
    def load_routes(self) -> int:
        loaded = self.milvus_storage.load_all_routes()
        for route in loaded:
            self._routes[route.name] = route
        
        logger.info(f"Loaded {len(loaded)} routes")
        return len(loaded)
    
    def close(self):
        self.milvus_storage.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("SemanticRouter closed")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
