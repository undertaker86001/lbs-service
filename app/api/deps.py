"""
API依赖注入模块
"""
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import time
import logging

from app.core.database import get_db, get_redis, db_manager
from app.core.config import settings
from app.services.user_service import user_service

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


def get_database() -> Generator:
    """获取数据库会话依赖"""
    return get_db()


def get_redis_client():
    """获取Redis客户端依赖"""
    return get_redis()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    获取当前用户ID
    
    Args:
        credentials: JWT令牌
        
    Returns:
        用户ID
        
    Raises:
        HTTPException: 认证失败
    """
    try:
        # 解码JWT令牌
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 检查令牌是否过期
        exp = payload.get("exp")
        if exp and time.time() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="令牌已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_id
        
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user_optional(
    request: Request
) -> Optional[str]:
    """
    获取当前用户ID（可选）
    
    Args:
        request: HTTP请求
        
    Returns:
        用户ID或None
    """
    try:
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return None
        
        token = authorization.split(" ")[1]
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        user_id = payload.get("sub")
        exp = payload.get("exp")
        
        if user_id and (not exp or time.time() <= exp):
            return user_id
        
        return None
        
    except Exception:
        return None


def rate_limiter(
    request: Request,
    redis_client = Depends(get_redis_client),
    max_requests: int = 100,
    window: int = 60
):
    """
    API速率限制中间件
    
    Args:
        request: HTTP请求
        redis_client: Redis客户端
        max_requests: 最大请求数
        window: 时间窗口（秒）
        
    Raises:
        HTTPException: 超过速率限制
    """
    try:
        # 获取客户端IP
        client_ip = request.client.host
        
        # 构建Redis键
        key = f"rate_limit:{client_ip}"
        
        # 获取当前请求数
        current_requests = redis_client.get(key)
        
        if current_requests is None:
            # 第一次请求
            redis_client.setex(key, window, 1)
        else:
            current_requests = int(current_requests)
            if current_requests >= max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="请求过于频繁，请稍后再试"
                )
            else:
                redis_client.incr(key)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"速率限制检查失败: {e}")
        # 速率限制失败时不阻止请求


def check_database_health():
    """
    检查数据库健康状态
    
    Raises:
        HTTPException: 数据库连接失败
    """
    try:
        if not db_manager.test_redis_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis连接失败"
            )
        
        if not db_manager.test_mysql_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MySQL连接失败"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="数据库服务不可用"
        )


def log_user_activity(
    user_id: str = Depends(get_current_user_optional),
    request: Request = None
):
    """
    记录用户活动
    
    Args:
        user_id: 用户ID
        request: HTTP请求
    """
    if user_id:
        try:
            # 更新用户最后活跃时间
            user_service.update_last_active(user_id)
            
            # 记录API调用行为
            if request:
                user_service.record_user_behavior({
                    "user_id": user_id,
                    "action_type": "api_call",
                    "action_data": {
                        "method": request.method,
                        "url": str(request.url),
                        "user_agent": request.headers.get("User-Agent", ""),
                        "ip_address": request.client.host
                    },
                    "ip_address": request.client.host
                })
                
        except Exception as e:
            logger.warning(f"记录用户活动失败: {e}")


class PermissionChecker:
    """权限检查器"""
    
    def __init__(self, required_permissions: list = None):
        self.required_permissions = required_permissions or []
    
    def __call__(self, user_id: str = Depends(get_current_user_id)):
        """
        检查用户权限
        
        Args:
            user_id: 用户ID
            
        Raises:
            HTTPException: 权限不足
        """
        # 这里可以扩展复杂的权限检查逻辑
        # 目前简单检查用户是否存在且激活
        user = user_service.get_user(user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="用户不存在或已禁用"
            )
        
        return user_id


# 常用的权限检查器实例
check_user_permission = PermissionChecker()


def create_jwt_token(user_id: str, expires_delta: Optional[int] = None) -> str:
    """
    创建JWT令牌
    
    Args:
        user_id: 用户ID
        expires_delta: 过期时间增量（小时）
        
    Returns:
        JWT令牌字符串
    """
    if expires_delta is None:
        expires_delta = settings.jwt_expiration_hours
    
    expire = time.time() + (expires_delta * 3600)
    
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": time.time(),
        "type": "access"
    }
    
    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return token


def verify_jwt_token(token: str) -> Optional[str]:
    """
    验证JWT令牌
    
    Args:
        token: JWT令牌
        
    Returns:
        用户ID或None
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        user_id = payload.get("sub")
        exp = payload.get("exp")
        
        if user_id and (not exp or time.time() <= exp):
            return user_id
        
        return None
        
    except jwt.InvalidTokenError:
        return None
