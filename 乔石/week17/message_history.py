import json
import time
import uuid
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Literal
from dataclasses import dataclass, field, asdict
from pathlib import Path

import redis
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    created_at: int = field(default_factory=lambda: int(time.time()))
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "message_id": self.message_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            created_at=data.get("created_at", int(time.time())),
            message_id=data.get("message_id", str(uuid.uuid4())),
            metadata=data.get("metadata", {})
        )


@dataclass
class Conversation:
    id: str
    messages: List[Message] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        return cls(
            id=data["id"],
            messages=[Message.from_dict(msg) for msg in data.get("messages", [])],
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
            metadata=data.get("metadata", {})
        )


class StorageBackend(ABC):
    """存储后端抽象基类"""
    
    @abstractmethod
    def save_conversation(self, conversation: Conversation) -> bool:
        pass
    
    @abstractmethod
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        pass
    
    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        pass
    
    @abstractmethod
    def list_conversations(self, limit: int, offset: int) -> List[Conversation]:
        pass
    
    @abstractmethod
    def exists(self, conversation_id: str) -> bool:
        pass
    
    @abstractmethod
    def close(self):
        pass


class RedisStorage(StorageBackend):
    """Redis存储后端"""
    
    KEY_PREFIX = "message_history"
    CONVERSATIONS_KEY = f"{KEY_PREFIX}:conversations"
    CONVERSATION_KEY_TEMPLATE = f"{KEY_PREFIX}:conversation:{{conversation_id}}"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 604800
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl = ttl
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
            logger.info("RedisStorage connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _get_conversation_key(self, conversation_id: str) -> str:
        return self.CONVERSATION_KEY_TEMPLATE.format(conversation_id=conversation_id)
    
    def save_conversation(self, conversation: Conversation) -> bool:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        try:
            conversation.updated_at = int(time.time())
            key = self._get_conversation_key(conversation.id)
            
            data = {
                "id": conversation.id,
                "messages": json.dumps([msg.to_dict() for msg in conversation.messages]),
                "created_at": str(conversation.created_at),
                "updated_at": str(conversation.updated_at),
                "metadata": json.dumps(conversation.metadata)
            }
            
            pipe = self.client.pipeline()
            pipe.hset(key, mapping=data)
            pipe.expire(key, self.ttl)
            pipe.zadd(self.CONVERSATIONS_KEY, {conversation.id: conversation.updated_at})
            pipe.execute()
            
            logger.debug(f"Saved conversation: {conversation.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = self._get_conversation_key(conversation_id)
        data = self.client.hgetall(key)
        
        if not data:
            return None
        
        try:
            return Conversation(
                id=data["id"],
                messages=[Message.from_dict(msg) for msg in json.loads(data["messages"])],
                created_at=int(data["created_at"]),
                updated_at=int(data["updated_at"]),
                metadata=json.loads(data.get("metadata", "{}"))
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load conversation: {e}")
            return None
    
    def delete_conversation(self, conversation_id: str) -> bool:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = self._get_conversation_key(conversation_id)
        pipe = self.client.pipeline()
        pipe.delete(key)
        pipe.zrem(self.CONVERSATIONS_KEY, conversation_id)
        result = pipe.execute()
        
        logger.info(f"Deleted conversation: {conversation_id}")
        return result[0] > 0
    
    def list_conversations(self, limit: int = 10, offset: int = 0) -> List[Conversation]:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        conversation_ids = self.client.zrevrange(
            self.CONVERSATIONS_KEY, offset, offset + limit - 1
        )
        
        conversations = []
        for conv_id in conversation_ids:
            conv = self.load_conversation(conv_id)
            if conv:
                conversations.append(conv)
        
        return conversations
    
    def exists(self, conversation_id: str) -> bool:
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        
        key = self._get_conversation_key(conversation_id)
        return self.client.exists(key) > 0
    
    def close(self):
        if self.client:
            self.client.close()
            logger.info("RedisStorage closed")


class MemoryStorage(StorageBackend):
    """内存存储后端（单进程）"""
    
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}
    
    def save_conversation(self, conversation: Conversation) -> bool:
        conversation.updated_at = int(time.time())
        self._conversations[conversation.id] = conversation
        logger.debug(f"Saved conversation to memory: {conversation.id}")
        return True
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info(f"Deleted conversation from memory: {conversation_id}")
            return True
        return False
    
    def list_conversations(self, limit: int = 10, offset: int = 0) -> List[Conversation]:
        sorted_convs = sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True
        )
        return sorted_convs[offset:offset + limit]
    
    def exists(self, conversation_id: str) -> bool:
        return conversation_id in self._conversations
    
    def close(self):
        self._conversations.clear()
        logger.info("MemoryStorage closed")


class FileStorage(StorageBackend):
    """文件存储后端"""
    
    def __init__(self, storage_dir: str = "./conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "index.json"
        self._conversations: Dict[str, Conversation] = {}
        self._load_index()
    
    def _load_index(self):
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    for conv_id, metadata in index_data.items():
                        if os.path.exists(self.storage_dir / f"{conv_id}.json"):
                            conv = self._load_from_file(conv_id)
                            if conv:
                                self._conversations[conv_id] = conv
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load index: {e}")
    
    def _save_index(self):
        try:
            index_data = {
                conv_id: {"updated_at": conv.updated_at}
                for conv_id, conv in self._conversations.items()
            }
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f)
        except IOError as e:
            logger.error(f"Failed to save index: {e}")
    
    def _get_file_path(self, conversation_id: str) -> Path:
        return self.storage_dir / f"{conversation_id}.json"
    
    def _load_from_file(self, conversation_id: str) -> Optional[Conversation]:
        file_path = self._get_file_path(conversation_id)
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Conversation.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load conversation file: {e}")
            return None
    
    def _save_to_file(self, conversation: Conversation) -> bool:
        file_path = self._get_file_path(conversation.id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except IOError as e:
            logger.error(f"Failed to save conversation file: {e}")
            return False
    
    def save_conversation(self, conversation: Conversation) -> bool:
        conversation.updated_at = int(time.time())
        self._conversations[conversation.id] = conversation
        self._save_to_file(conversation)
        self._save_index()
        logger.debug(f"Saved conversation to file: {conversation.id}")
        return True
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        if conversation_id in self._conversations:
            return self._conversations[conversation_id]
        return self._load_from_file(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        file_path = self._get_file_path(conversation_id)
        if file_path.exists():
            file_path.unlink()
        
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
        
        self._save_index()
        logger.info(f"Deleted conversation file: {conversation_id}")
        return True
    
    def list_conversations(self, limit: int = 10, offset: int = 0) -> List[Conversation]:
        sorted_convs = sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True
        )
        return sorted_convs[offset:offset + limit]
    
    def exists(self, conversation_id: str) -> bool:
        return conversation_id in self._conversations or self._get_file_path(conversation_id).exists()
    
    def close(self):
        self._conversations.clear()
        logger.info("FileStorage closed")


class SemanticMessageHistory:
    """
    对话历史管理器
    
    提供对话历史的添加、获取、压缩和存储功能，
    支持多种存储后端（Redis、内存、文件）。
    """
    
    def __init__(
        self,
        storage_backend: StorageBackend,
        max_history: int = 20,
        max_tokens: int = 4000,
        summarize_trigger_tokens: int = 3500,
        conversation_id: Optional[str] = None,
        default_ttl: int = 604800
    ):
        self.storage = storage_backend
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.summarize_trigger_tokens = summarize_trigger_tokens
        self.default_ttl = default_ttl
        
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        self.conversation_id = conversation_id
        
        self._conversation: Optional[Conversation] = None
        self._load_or_create_conversation()
    
    def _load_or_create_conversation(self):
        existing = self.storage.load_conversation(self.conversation_id)
        if existing:
            self._conversation = existing
        else:
            self._conversation = Conversation(id=self.conversation_id)
            self.storage.save_conversation(self._conversation)
    
    def _count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def _count_messages_tokens(self, messages: List[Message]) -> int:
        return sum(self._count_tokens(msg.content) for msg in messages)
    
    def _ensure_initialized(self):
        if self._conversation is None:
            self._load_or_create_conversation()
    
    # ========== 消息管理 ==========
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> str:
        return self.add_message("user", content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> str:
        return self.add_message("assistant", content, metadata)
    
    def add_system_message(self, content: str, metadata: Optional[Dict] = None) -> str:
        return self.add_message("system", content, metadata)
    
    def add_message(
        self,
        role: Literal["system", "user", "assistant"],
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        self._ensure_initialized()
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self._conversation.messages.append(message)
        
        while (
            len(self._conversation.messages) > self.max_history
            or self._count_messages_tokens(self._conversation.messages) > self.max_tokens
        ):
            self._conversation.messages.pop(0)
        
        self.storage.save_conversation(self._conversation)
        
        logger.debug(f"Added {role} message: {message.message_id}")
        return message.message_id
    
    # ========== 历史获取 ==========
    
    def get_messages(
        self,
        last_n: Optional[int] = None,
        roles: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        
        messages = self._conversation.messages
        
        if roles:
            messages = [msg for msg in messages if msg.role in roles]
        
        if last_n is not None:
            messages = messages[-last_n:]
        
        return [msg.to_dict() for msg in messages]
    
    def get_context(
        self,
        max_tokens: Optional[int] = None,
        include_system: bool = True
    ) -> str:
        messages = self.get_messages()
        
        if not include_system:
            messages = [msg for msg in messages if msg["role"] != "system"]
        
        max_tok = max_tokens or self.max_tokens
        
        context_parts = []
        current_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = self._count_tokens(msg["content"])
            if current_tokens + msg_tokens <= max_tok:
                context_parts.insert(0, f"{msg['role']}: {msg['content']}")
                current_tokens += msg_tokens
            else:
                break
        
        return "\n".join(context_parts)
    
    def get_messages_for_llm(
        self,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        self._ensure_initialized()
        
        result = []
        max_tok = max_tokens or self.max_tokens
        
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
            max_tok -= self._count_tokens(system_prompt)
        else:
            system_messages = [msg for msg in self._conversation.messages if msg.role == "system"]
            for sys_msg in system_messages[:-1]:
                if self._count_tokens(sys_msg.content) <= max_tok:
                    result.append({"role": "system", "content": sys_msg.content})
                    max_tok -= self._count_tokens(sys_msg.content)
            
            if system_messages:
                sys_msg = system_messages[-1]
                result.append({"role": "system", "content": sys_msg.content})
        
        non_system_messages = [
            msg for msg in self._conversation.messages
            if msg.role != "system"
        ]
        
        current_tokens = 0
        included_messages = []
        
        for msg in reversed(non_system_messages):
            msg_tokens = self._count_tokens(msg.content)
            if current_tokens + msg_tokens <= max_tok:
                included_messages.insert(0, {"role": msg.role, "content": msg.content})
                current_tokens += msg_tokens
            else:
                break
        
        result.extend(included_messages)
        
        return result
    
    # ========== 对话管理 ==========
    
    def clear(self) -> bool:
        self._ensure_initialized()
        self._conversation.messages = []
        self.storage.save_conversation(self._conversation)
        logger.info(f"Cleared conversation: {self.conversation_id}")
        return True
    
    def delete_message(self, message_id: str) -> bool:
        self._ensure_initialized()
        
        for i, msg in enumerate(self._conversation.messages):
            if msg.message_id == message_id:
                self._conversation.messages.pop(i)
                self.storage.save_conversation(self._conversation)
                logger.info(f"Deleted message: {message_id}")
                return True
        
        return False
    
    def update_message(self, message_id: str, content: str) -> bool:
        self._ensure_initialized()
        
        for msg in self._conversation.messages:
            if msg.message_id == message_id:
                msg.content = content
                self.storage.save_conversation(self._conversation)
                logger.info(f"Updated message: {message_id}")
                return True
        
        return False
    
    # ========== 摘要压缩 ==========
    
    def summarize(
        self,
        summary_prompt: Optional[str] = None,
        llm_summarize_func: Optional[Callable] = None
    ) -> str:
        self._ensure_initialized()
        
        if not llm_summarize_func:
            logger.warning("No LLM summarize function provided, using basic summarization")
            summary = self._basic_summarize()
            return summary
        
        non_system_messages = [
            msg for msg in self._conversation.messages
            if msg.role != "system"
        ]
        
        if not non_system_messages:
            return ""
        
        default_prompt = summary_prompt or "请简要总结以下对话的主要内容，保留关键信息："
        summary_text = llm_summarize_func(non_system_messages, default_prompt)
        
        self._conversation.messages = [
            msg for msg in self._conversation.messages
            if msg.role == "system"
        ]
        
        summary_message = Message(
            role="system",
            content=f"[对话摘要] {summary_text}",
            metadata={"is_summary": True}
        )
        self._conversation.messages.append(summary_message)
        
        self.storage.save_conversation(self._conversation)
        
        logger.info(f"Created conversation summary: {len(summary_text)} chars")
        return summary_text
    
    def _basic_summarize(self) -> str:
        non_system_messages = [
            msg for msg in self._conversation.messages
            if msg.role != "system"
        ]
        
        if not non_system_messages:
            return ""
        
        first_msg = non_system_messages[0]
        last_msg = non_system_messages[-1]
        
        summary = f"对话从'{first_msg.content[:50]}...'开始，"
        summary += f"最后讨论到'{last_msg.content[:50] if len(last_msg.content) > 50 else last_msg.content}...'"
        summary += f"，共{len(non_system_messages)}轮对话。"
        
        return summary
    
    def should_summarize(self) -> bool:
        total_tokens = self._count_messages_tokens(self._conversation.messages)
        return total_tokens > self.summarize_trigger_tokens
    
    # ========== 对话ID管理 ==========
    
    def get_conversation_id(self) -> str:
        return self.conversation_id
    
    def set_conversation_id(self, conversation_id: str):
        if conversation_id != self.conversation_id:
            self.conversation_id = conversation_id
            self._load_or_create_conversation()
    
    def list_conversations(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        conversations = self.storage.list_conversations(limit, offset)
        return [
            {
                "id": conv.id,
                "message_count": len(conv.messages),
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "metadata": conv.metadata
            }
            for conv in conversations
        ]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        return self.storage.delete_conversation(conversation_id)
    
    # ========== 统计信息 ==========
    
    def stats(self) -> Dict[str, Any]:
        self._ensure_initialized()
        
        messages_by_role = {}
        for msg in self._conversation.messages:
            messages_by_role[msg.role] = messages_by_role.get(msg.role, 0) + 1
        
        return {
            "conversation_id": self.conversation_id,
            "message_count": len(self._conversation.messages),
            "total_tokens": self._count_messages_tokens(self._conversation.messages),
            "messages_by_role": messages_by_role,
            "max_history": self.max_history,
            "max_tokens": self.max_tokens,
            "should_summarize": self.should_summarize()
        }
    
    # ========== 工厂方法 ==========
    
    @staticmethod
    def create_redis(
        conversation_id: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_history: int = 20,
        max_tokens: int = 4000,
        summarize_trigger_tokens: int = 3500,
        ttl: int = 604800
    ) -> "SemanticMessageHistory":
        storage = RedisStorage(host=host, port=port, db=db, password=password, ttl=ttl)
        storage.connect()
        return SemanticMessageHistory(
            storage_backend=storage,
            max_history=max_history,
            max_tokens=max_tokens,
            summarize_trigger_tokens=summarize_trigger_tokens,
            conversation_id=conversation_id,
            default_ttl=ttl
        )
    
    @staticmethod
    def create_memory(
        conversation_id: Optional[str] = None,
        max_history: int = 20,
        max_tokens: int = 4000,
        summarize_trigger_tokens: int = 3500
    ) -> "SemanticMessageHistory":
        storage = MemoryStorage()
        return SemanticMessageHistory(
            storage_backend=storage,
            max_history=max_history,
            max_tokens=max_tokens,
            summarize_trigger_tokens=summarize_trigger_tokens,
            conversation_id=conversation_id
        )
    
    @staticmethod
    def create_file(
        conversation_id: Optional[str] = None,
        storage_dir: str = "./conversations",
        max_history: int = 20,
        max_tokens: int = 4000,
        summarize_trigger_tokens: int = 3500
    ) -> "SemanticMessageHistory":
        storage = FileStorage(storage_dir=storage_dir)
        return SemanticMessageHistory(
            storage_backend=storage,
            max_history=max_history,
            max_tokens=max_tokens,
            summarize_trigger_tokens=summarize_trigger_tokens,
            conversation_id=conversation_id
        )
    
    def close(self):
        if hasattr(self.storage, 'close'):
            self.storage.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
