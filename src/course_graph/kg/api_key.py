# -*- coding: utf-8 -*-
# Create Date: 2025/02/27
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: course_graph/kg/api_keys.py
# Description: API KEYS 管理

from datetime import datetime
from enum import Enum
import json
import uuid
import os

class KeyStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"

class APIKey:
    def __init__(self, 
                 key: str, 
                 created_at: datetime = None, 
                 expires_at: datetime = None,
                 metadata: dict = None):
        """ API Key 对象
        
        Args:
            key (str): API Key 值
            created_at (datetime, optional): 创建时间. Defaults to None.
            expires_at (datetime, optional): 过期时间. Defaults to None.
            metadata (dict, optional): 其他元数据. Defaults to None.
        """
        self.key = key
        self.created_at = created_at or datetime.now()
        self.expires_at = expires_at
        self.status = KeyStatus.ACTIVE
        self.last_used = None
        self.metadata = metadata

    def to_dict(self) -> dict:
        """将 API Key 对象转换为 JSON 格式
        
        Returns:
            dict: JSON 格式的 API Key 对象
        """
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "metadata": self.metadata
        }

class APIKeyManager:
    def __init__(self, storage_path: str = os.path.join(os.path.dirname(__file__), "api_keys.json")):
        """ API Key 管理器
        
        Args:
            storage_path (str, optional): 存储 API Key 的文件路径. Defaults to "./api_keys.json".
        """
        self.storage_path = storage_path
        self.keys: dict[str, APIKey] = {}
        self._load_keys()

    def generate_key(self, expires_at: datetime = None, metadata: dict = None) -> str:
        """生成新的 API Key
        
        Args:
            expires_at (datetime, optional): 过期时间. Defaults to None.
            metadata (dict, optional): 其他元数据. Defaults to None.
        
        Returns:
            str: 生成的 API Key
        """
        key = f"sk-{uuid.uuid4().hex}"
        self.keys[key] = APIKey(key, expires_at=expires_at, metadata=metadata)
        self._save_keys()
        return key
    
    def add_key(self, key: str, expires_at: datetime = None, metadata = None) -> bool:
        """手动添加一个新的 API Key
        
        Args:
            key (str): API Key 值
            expires_at (datetime, optional): 过期时间. Defaults to None.
            metadata (dict, optional): 其他元数据. Defaults to None.
            
        Returns:
            bool: 是否成功添加
        """
        if key in self.keys:
            return False
        self.keys[key] = APIKey(key, expires_at=expires_at, metadata=metadata)
        self._save_keys()
        return True
    
    def update_metadata(self, key: str, metadata: dict) -> bool:
        """更新 API Key 的元数据
        
        Args:
            key (str): API Key 值
            metadata (dict): 元数据
        Returns:
            bool: 是否成功更新
        """
        if key in self.keys:
            self.keys[key].metadata.update(metadata)
            self._save_keys()
            return True
        return False

    def validate_key(self, key: str) -> bool:
        """验证 API Key 是否有效
        
        Args:
            key (str): API Key 值
        Returns:
            bool: 是否有效
        """
        if key not in self.keys:
            return False
        
        api_key = self.keys[key]
        if api_key.status != KeyStatus.ACTIVE:
            return False

        if api_key.expires_at and datetime.now() > api_key.expires_at:
            api_key.status = KeyStatus.EXPIRED
            self._save_keys()
            return False

        api_key.last_used = datetime.now()
        self._save_keys()
        return True

    def revoke_key(self, key: str) -> bool:
        """废弃 API Key
        
        Args:
            key (str): API Key 值
        Returns:
            bool: 是否成功废弃
        """
        if key in self.keys:
            self.keys[key].status = KeyStatus.REVOKED
            self._save_keys()
            return True
        return False

    def _save_keys(self):
        """保存密钥到文件"""
        with open(self.storage_path, 'w', encoding='UTF-8') as f:
            json.dump({k: v.to_dict() for k, v in self.keys.items()}, f, ensure_ascii=False)

    def _load_keys(self):
        """从文件加载密钥"""
        try:
            with open(self.storage_path, 'r', encoding='UTF-8') as f:
                data = json.load(f)
                for k, v in data.items():
                    key = APIKey(
                        key=k,
                        created_at=datetime.fromisoformat(v['created_at']),
                        expires_at=datetime.fromisoformat(v['expires_at']) if v['expires_at'] else None
                    )
                    key.status = KeyStatus(v['status'])
                    key.last_used = datetime.fromisoformat(v['last_used']) if v['last_used'] else None
                    self.keys[k] = key
        except FileNotFoundError:
            pass