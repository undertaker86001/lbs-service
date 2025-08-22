"""
用户管理API路由
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.security import HTTPBearer
import logging
import time

from app.models.user import (
    UserCreate, UserUpdate, UserResponse, UserBehaviorCreate,
    UserBehaviorResponse
)
from app.models.base import BaseResponse, PaginationParams
from app.services.user_service import user_service
from app.api.deps import (
    get_current_user_id, get_current_user_optional, 
    rate_limiter, log_user_activity, create_jwt_token
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["用户管理"])
security = HTTPBearer()


@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    request: Request,
    _: None = Depends(rate_limiter)
):
    """
    用户注册
    
    - **user_id**: 用户唯一标识
    - **username**: 用户名（可选）
    - **nickname**: 昵称（可选）
    - **privacy_level**: 隐私级别 (low/medium/high)
    """
    try:
        # 创建用户
        user = user_service.create_user(user_data)
        
        # 生成JWT令牌
        access_token = create_jwt_token(user.user_id)
        
        # 记录注册行为
        user_service.record_user_behavior(UserBehaviorCreate(
            user_id=user.user_id,
            action_type="user_register",
            action_data={
                "registration_time": user.created_at.isoformat(),
                "privacy_level": user.privacy_level
            },
            ip_address=request.client.host,
            device_type=request.headers.get("User-Agent", "")[:50]
        ))
        
        return {
            "status": "success",
            "message": "用户注册成功",
            "data": {
                "user": user,
                "access_token": access_token,
                "token_type": "bearer"
            }
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后重试"
        )


@router.post("/login", response_model=dict)
async def login_user(
    user_id: str,
    request: Request,
    _: None = Depends(rate_limiter)
):
    """
    用户登录
    
    - **user_id**: 用户ID
    """
    try:
        # 获取用户信息
        user = user_service.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="用户已被禁用"
            )
        
        # 生成JWT令牌
        access_token = create_jwt_token(user.user_id)
        
        # 更新最后活跃时间
        user_service.update_last_active(user.user_id)
        
        # 记录登录行为
        user_service.record_user_behavior(UserBehaviorCreate(
            user_id=user.user_id,
            action_type="user_login",
            action_data={
                "login_time": user.last_active_at.isoformat() if user.last_active_at else None
            },
            ip_address=request.client.host,
            device_type=request.headers.get("User-Agent", "")[:50]
        ))
        
        return {
            "status": "success",
            "message": "登录成功",
            "data": {
                "user": user,
                "access_token": access_token,
                "token_type": "bearer"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录失败，请稍后重试"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user_id: str = Depends(get_current_user_id),
    _: None = Depends(log_user_activity)
):
    """
    获取当前用户信息
    """
    try:
        user = user_service.get_user(current_user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户信息失败"
        )


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user_id: str = Depends(get_current_user_id),
    _: None = Depends(log_user_activity)
):
    """
    更新当前用户信息
    """
    try:
        user = user_service.update_user(current_user_id, user_data)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 记录更新行为
        user_service.record_user_behavior(UserBehaviorCreate(
            user_id=current_user_id,
            action_type="user_update",
            action_data={
                "updated_fields": list(user_data.dict(exclude_unset=True).keys())
            }
        ))
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新用户信息失败"
        )


@router.delete("/me", response_model=BaseResponse)
async def delete_current_user(
    current_user_id: str = Depends(get_current_user_id),
    _: None = Depends(log_user_activity)
):
    """
    删除当前用户账户
    """
    try:
        success = user_service.delete_user(current_user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        return BaseResponse(message="用户账户删除成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除用户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除用户失败"
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    current_user_id: Optional[str] = Depends(get_current_user_optional),
    _: None = Depends(log_user_activity)
):
    """
    根据用户ID获取用户信息
    """
    try:
        user = user_service.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 根据隐私设置过滤信息
        if current_user_id != user_id:
            if user.privacy_level == "high":
                # 高隐私级别只显示基本信息
                user.bio = None
                user.age = None
            elif user.privacy_level == "medium":
                # 中等隐私级别隐藏部分信息
                user.bio = None
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户信息失败"
        )


@router.get("/", response_model=List[UserResponse])
async def search_users(
    keyword: str = Query(..., min_length=1, description="搜索关键词"),
    limit: int = Query(default=20, ge=1, le=100, description="返回数量限制"),
    current_user_id: Optional[str] = Depends(get_current_user_optional),
    _: None = Depends(rate_limiter),
    __: None = Depends(log_user_activity)
):
    """
    搜索用户
    
    - **keyword**: 搜索关键词（用户名、昵称或用户ID）
    - **limit**: 返回数量限制 (1-100)
    """
    try:
        users = user_service.search_users(keyword, limit)
        
        # 根据隐私设置过滤信息
        for user in users:
            if current_user_id != user.user_id:
                if user.privacy_level == "high":
                    user.bio = None
                    user.age = None
                elif user.privacy_level == "medium":
                    user.bio = None
        
        return users
        
    except Exception as e:
        logger.error(f"搜索用户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="搜索用户失败"
        )


@router.post("/behavior", response_model=BaseResponse)
async def record_user_behavior(
    behavior_data: UserBehaviorCreate,
    current_user_id: str = Depends(get_current_user_id)
):
    """
    记录用户行为
    
    - **action_type**: 行为类型
    - **action_data**: 行为数据
    """
    try:
        # 确保只能记录自己的行为
        if behavior_data.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="只能记录自己的行为"
            )
        
        success = user_service.record_user_behavior(behavior_data)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="记录行为失败"
            )
        
        return BaseResponse(message="行为记录成功")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"记录用户行为失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="记录用户行为失败"
        )


@router.get("/me/behaviors", response_model=List[dict])
async def get_user_behaviors(
    limit: int = Query(default=50, ge=1, le=100, description="返回数量限制"),
    current_user_id: str = Depends(get_current_user_id),
    _: None = Depends(log_user_activity)
):
    """
    获取当前用户的行为记录
    
    - **limit**: 返回数量限制 (1-100)
    """
    try:
        behaviors = user_service.get_user_behaviors(current_user_id, limit)
        return behaviors
        
    except Exception as e:
        logger.error(f"获取用户行为记录失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户行为记录失败"
        )


@router.get("/stats/active", response_model=dict)
async def get_active_users_stats(
    hours: int = Query(default=24, ge=1, le=168, description="统计时间范围（小时）"),
    _: None = Depends(rate_limiter)
):
    """
    获取活跃用户统计
    
    - **hours**: 统计时间范围，最近多少小时内活跃 (1-168小时)
    """
    try:
        active_count = user_service.get_active_users_count(hours)
        
        return {
            "status": "success",
            "data": {
                "active_users_count": active_count,
                "time_range_hours": hours,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"获取活跃用户统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取统计数据失败"
        )
