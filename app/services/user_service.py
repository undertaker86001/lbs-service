"""
用户管理服务
"""
import json
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
import logging

from app.core.database import get_mysql_connection, get_redis
from app.models.user import (
    User, UserCreate, UserUpdate, UserResponse,
    UserBehavior, UserBehaviorCreate
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class UserService:
    """用户管理服务"""
    
    def __init__(self):
        self.redis_client = get_redis()
    
    def create_user(self, user_data: UserCreate) -> UserResponse:
        """
        创建新用户
        
        Args:
            user_data: 用户创建数据
            
        Returns:
            创建的用户信息
        """
        try:
            with get_mysql_connection() as conn:
                cursor = conn.cursor()
                
                # 检查用户是否已存在
                check_sql = "SELECT user_id FROM users WHERE user_id = %s"
                cursor.execute(check_sql, (user_data.user_id,))
                existing_user = cursor.fetchone()
                
                if existing_user:
                    raise ValueError(f"用户 {user_data.user_id} 已存在")
                
                # 创建新用户
                insert_sql = """
                INSERT INTO users 
                (user_id, username, nickname, avatar_url, gender, age, bio, 
                 privacy_level, is_active, created_at, updated_at, last_active_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                now = datetime.now()
                cursor.execute(insert_sql, (
                    user_data.user_id,
                    user_data.username,
                    user_data.nickname,
                    user_data.avatar_url,
                    user_data.gender,
                    user_data.age,
                    user_data.bio,
                    user_data.privacy_level,
                    True,  # is_active
                    now,
                    now,
                    now
                ))
                
                user_id = cursor.lastrowid
                conn.commit()
                cursor.close()
                
                # 缓存用户信息到Redis
                self._cache_user_info(user_data.user_id, {
                    "id": user_id,
                    "user_id": user_data.user_id,
                    "username": user_data.username,
                    "nickname": user_data.nickname,
                    "avatar_url": user_data.avatar_url,
                    "gender": user_data.gender,
                    "age": user_data.age,
                    "bio": user_data.bio,
                    "privacy_level": user_data.privacy_level,
                    "is_active": True,
                    "created_at": now.isoformat(),
                    "last_active_at": now.isoformat()
                })
                
                logger.info(f"用户 {user_data.user_id} 创建成功")
                
                return UserResponse(
                    id=user_id,
                    user_id=user_data.user_id,
                    username=user_data.username,
                    nickname=user_data.nickname,
                    avatar_url=user_data.avatar_url,
                    gender=user_data.gender,
                    age=user_data.age,
                    bio=user_data.bio,
                    is_active=True,
                    privacy_level=user_data.privacy_level,
                    created_at=now,
                    last_active_at=now
                )
                
        except Exception as e:
            logger.error(f"创建用户失败: {e}")
            raise
    
    def get_user(self, user_id: str) -> Optional[UserResponse]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户信息
        """
        try:
            # 先从缓存获取
            cached_user = self.redis_client.hgetall(f"user:{user_id}")
            if cached_user:
                return UserResponse(
                    id=int(cached_user["id"]),
                    user_id=cached_user["user_id"],
                    username=cached_user.get("username"),
                    nickname=cached_user.get("nickname"),
                    avatar_url=cached_user.get("avatar_url"),
                    gender=cached_user.get("gender"),
                    age=int(cached_user["age"]) if cached_user.get("age") else None,
                    bio=cached_user.get("bio"),
                    is_active=cached_user["is_active"] == "True",
                    privacy_level=cached_user["privacy_level"],
                    created_at=datetime.fromisoformat(cached_user["created_at"]),
                    last_active_at=datetime.fromisoformat(cached_user["last_active_at"]) if cached_user.get("last_active_at") else None
                )
            
            # 缓存中没有，从数据库获取
            with get_mysql_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                sql = """
                SELECT id, user_id, username, nickname, avatar_url, gender, age, bio,
                       is_active, privacy_level, created_at, updated_at, last_active_at
                FROM users 
                WHERE user_id = %s AND is_active = TRUE
                """
                
                cursor.execute(sql, (user_id,))
                user_data = cursor.fetchone()
                cursor.close()
                
                if user_data:
                    # 缓存到Redis
                    cache_data = {k: str(v) if v is not None else "" for k, v in user_data.items()}
                    self._cache_user_info(user_id, cache_data)
                    
                    return UserResponse(**user_data)
                
                return None
                
        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            return None
    
    def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[UserResponse]:
        """
        更新用户信息
        
        Args:
            user_id: 用户ID
            user_data: 更新数据
            
        Returns:
            更新后的用户信息
        """
        try:
            # 构建更新字段
            update_fields = []
            update_values = []
            
            for field, value in user_data.dict(exclude_unset=True).items():
                if value is not None:
                    update_fields.append(f"{field} = %s")
                    update_values.append(value)
            
            if not update_fields:
                # 没有要更新的字段
                return self.get_user(user_id)
            
            update_fields.append("updated_at = %s")
            update_values.append(datetime.now())
            update_values.append(user_id)
            
            with get_mysql_connection() as conn:
                cursor = conn.cursor()
                
                sql = f"""
                UPDATE users 
                SET {', '.join(update_fields)}
                WHERE user_id = %s AND is_active = TRUE
                """
                
                cursor.execute(sql, update_values)
                affected_rows = cursor.rowcount
                conn.commit()
                cursor.close()
                
                if affected_rows > 0:
                    # 清除缓存
                    self.redis_client.delete(f"user:{user_id}")
                    
                    logger.info(f"用户 {user_id} 信息更新成功")
                    return self.get_user(user_id)
                
                return None
                
        except Exception as e:
            logger.error(f"更新用户信息失败: {e}")
            raise
    
    def delete_user(self, user_id: str) -> bool:
        """
        删除用户（软删除）
        
        Args:
            user_id: 用户ID
            
        Returns:
            删除是否成功
        """
        try:
            with get_mysql_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                UPDATE users 
                SET is_active = FALSE, updated_at = %s
                WHERE user_id = %s
                """
                
                cursor.execute(sql, (datetime.now(), user_id))
                affected_rows = cursor.rowcount
                conn.commit()
                cursor.close()
                
                if affected_rows > 0:
                    # 清除缓存
                    self.redis_client.delete(f"user:{user_id}")
                    
                    # 从位置信息中移除
                    self.redis_client.zrem("user_locations", user_id)
                    self.redis_client.delete(f"user_location:{user_id}")
                    
                    logger.info(f"用户 {user_id} 删除成功")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"删除用户失败: {e}")
            return False
    
    def search_users(self, keyword: str, limit: int = 20) -> List[UserResponse]:
        """
        搜索用户
        
        Args:
            keyword: 搜索关键词
            limit: 返回数量限制
            
        Returns:
            匹配的用户列表
        """
        try:
            with get_mysql_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                sql = """
                SELECT id, user_id, username, nickname, avatar_url, gender, age, bio,
                       is_active, privacy_level, created_at, updated_at, last_active_at
                FROM users 
                WHERE is_active = TRUE 
                AND (username LIKE %s OR nickname LIKE %s OR user_id LIKE %s)
                ORDER BY last_active_at DESC
                LIMIT %s
                """
                
                search_pattern = f"%{keyword}%"
                cursor.execute(sql, (search_pattern, search_pattern, search_pattern, limit))
                users_data = cursor.fetchall()
                cursor.close()
                
                return [UserResponse(**user_data) for user_data in users_data]
                
        except Exception as e:
            logger.error(f"搜索用户失败: {e}")
            return []
    
    def record_user_behavior(self, behavior_data: UserBehaviorCreate) -> bool:
        """
        记录用户行为
        
        Args:
            behavior_data: 行为数据
            
        Returns:
            记录是否成功
        """
        try:
            with get_mysql_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                INSERT INTO user_behaviors 
                (user_id, action_type, action_data, session_id, device_type, 
                 ip_address, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                now = datetime.now()
                action_data_str = json.dumps(behavior_data.action_data) if behavior_data.action_data else None
                
                cursor.execute(sql, (
                    behavior_data.user_id,
                    behavior_data.action_type,
                    action_data_str,
                    behavior_data.session_id,
                    behavior_data.device_type,
                    behavior_data.ip_address,
                    now,
                    now
                ))
                
                conn.commit()
                cursor.close()
                
                # 同时缓存到Redis用于实时分析
                behavior_key = f"user_behavior:{behavior_data.user_id}"
                self.redis_client.lpush(behavior_key, json.dumps({
                    "action_type": behavior_data.action_type,
                    "action_data": behavior_data.action_data,
                    "timestamp": now.isoformat()
                }))
                
                # 保留最近100条行为记录
                self.redis_client.ltrim(behavior_key, 0, 99)
                self.redis_client.expire(behavior_key, 7 * 24 * 3600)  # 7天过期
                
                return True
                
        except Exception as e:
            logger.error(f"记录用户行为失败: {e}")
            return False
    
    def get_user_behaviors(self, user_id: str, limit: int = 100) -> List[Dict]:
        """
        获取用户行为记录
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            
        Returns:
            行为记录列表
        """
        try:
            # 先从缓存获取
            cached_behaviors = self.redis_client.lrange(f"user_behavior:{user_id}", 0, limit - 1)
            if cached_behaviors:
                return [json.loads(behavior) for behavior in cached_behaviors]
            
            # 缓存中没有，从数据库获取
            with get_mysql_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                sql = """
                SELECT action_type, action_data, session_id, device_type, 
                       ip_address, created_at
                FROM user_behaviors 
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """
                
                cursor.execute(sql, (user_id, limit))
                behaviors = cursor.fetchall()
                cursor.close()
                
                # 处理action_data字段
                for behavior in behaviors:
                    if behavior['action_data']:
                        try:
                            behavior['action_data'] = json.loads(behavior['action_data'])
                        except:
                            behavior['action_data'] = {}
                    behavior['timestamp'] = behavior['created_at'].isoformat()
                
                return behaviors
                
        except Exception as e:
            logger.error(f"获取用户行为记录失败: {e}")
            return []
    
    def _cache_user_info(self, user_id: str, user_data: Dict):
        """缓存用户信息到Redis"""
        try:
            pipe = self.redis_client.pipeline()
            pipe.hset(f"user:{user_id}", mapping=user_data)
            pipe.expire(f"user:{user_id}", settings.cache_ttl)
            pipe.execute()
        except Exception as e:
            logger.error(f"缓存用户信息失败: {e}")
    
    def get_active_users_count(self, hours: int = 24) -> int:
        """
        获取活跃用户数量
        
        Args:
            hours: 最近多少小时内活跃
            
        Returns:
            活跃用户数量
        """
        try:
            with get_mysql_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                SELECT COUNT(DISTINCT user_id) as count
                FROM users
                WHERE is_active = TRUE 
                AND last_active_at >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                """
                
                cursor.execute(sql, (hours,))
                result = cursor.fetchone()
                cursor.close()
                
                return result[0] if result else 0
                
        except Exception as e:
            logger.error(f"获取活跃用户数量失败: {e}")
            return 0
    
    def update_last_active(self, user_id: str):
        """
        更新用户最后活跃时间
        
        Args:
            user_id: 用户ID
        """
        try:
            with get_mysql_connection() as conn:
                cursor = conn.cursor()
                
                sql = """
                UPDATE users 
                SET last_active_at = %s
                WHERE user_id = %s AND is_active = TRUE
                """
                
                cursor.execute(sql, (datetime.now(), user_id))
                conn.commit()
                cursor.close()
                
                # 更新缓存中的时间
                self.redis_client.hset(f"user:{user_id}", "last_active_at", datetime.now().isoformat())
                
        except Exception as e:
            logger.error(f"更新用户活跃时间失败: {e}")


# 创建全局用户服务实例
user_service = UserService()
