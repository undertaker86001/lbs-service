"""
位置管理API路由
"""
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
import logging

from app.models.user import (
    UserLocationCreate, UserLocationResponse, NearbyUserRequest, 
    NearbyUserResponse, NearbyUser
)
from app.models.base import BaseResponse
from app.services.location_service import location_service
from app.services.user_service import user_service
from app.api.deps import (
    get_current_user_id, get_current_user_optional,
    rate_limiter, log_user_activity
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/locations", tags=["位置管理"])


@router.post("/update", response_model=dict, status_code=status.HTTP_201_CREATED)
async def update_user_location(
    location_data: UserLocationCreate,
    request: Request,
    current_user_id: str = Depends(get_current_user_id),
    _: None = Depends(rate_limiter),
    __: None = Depends(log_user_activity)
):
    """
    更新用户位置信息
    
    - **latitude**: 纬度 (-90 到 90)
    - **longitude**: 经度 (-180 到 180) 
    - **accuracy**: 定位精度（可选）
    - **address**: 地址信息（可选）
    """
    try:
        # 确保只能更新自己的位置
        if location_data.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="只能更新自己的位置信息"
            )
        
        # 验证用户是否存在
        user = user_service.get_user(current_user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 这里可以添加隐私保护逻辑（G-Casper算法）
        # 暂时使用原始位置
        anonymous_geohash = None
        
        # 更新位置信息
        location_record = location_service.add_user_location(
            location_data, 
            anonymous_geohash
        )
        
        # 记录位置更新行为
        user_service.record_user_behavior({
            "user_id": current_user_id,
            "action_type": "location_update",
            "action_data": {
                "latitude": location_data.latitude,
                "longitude": location_data.longitude,
                "accuracy": location_data.accuracy,
                "geohash_length": len(location_record.geohash) if location_record.geohash else 0
            },
            "ip_address": request.client.host
        })
        
        return {
            "status": "success",
            "message": "位置信息更新成功",
            "data": {
                "location_id": location_record.id,
                "geohash": location_record.geohash,
                "anonymous_geohash": location_record.anonymous_geohash,
                "timestamp": location_record.created_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新位置信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新位置信息失败"
        )


@router.post("/nearby", response_model=NearbyUserResponse)
async def find_nearby_users(
    request_data: NearbyUserRequest,
    current_user_id: Optional[str] = Depends(get_current_user_optional),
    _: None = Depends(rate_limiter),
    __: None = Depends(log_user_activity)
):
    """
    查找附近用户
    
    - **latitude**: 中心点纬度
    - **longitude**: 中心点经度
    - **radius**: 搜索半径(米) (100-10000)
    - **limit**: 返回结果数量限制 (1-100)
    - **exclude_user_id**: 排除的用户ID（可选）
    """
    try:
        # 设置排除用户ID
        exclude_user_id = request_data.exclude_user_id or current_user_id
        
        # 查找附近用户
        nearby_users = location_service.find_nearby_users(
            latitude=request_data.latitude,
            longitude=request_data.longitude,
            radius=request_data.radius,
            limit=request_data.limit,
            exclude_user_id=exclude_user_id
        )
        
        # 获取用户详细信息并应用隐私过滤
        enriched_users = []
        for nearby_user in nearby_users:
            # 获取用户详细信息
            user_info = user_service.get_user(nearby_user.user_id)
            if user_info:
                # 应用隐私设置
                if user_info.privacy_level == "high":
                    # 高隐私级别：只显示基本信息
                    nearby_user.username = None
                    nearby_user.nickname = user_info.nickname[:3] + "***" if user_info.nickname else None
                    nearby_user.avatar_url = None
                elif user_info.privacy_level == "medium":
                    # 中等隐私级别：显示昵称和头像
                    nearby_user.username = None
                    nearby_user.nickname = user_info.nickname
                    nearby_user.avatar_url = user_info.avatar_url
                else:
                    # 低隐私级别：显示所有信息
                    nearby_user.username = user_info.username
                    nearby_user.nickname = user_info.nickname
                    nearby_user.avatar_url = user_info.avatar_url
                
                nearby_user.last_active_at = user_info.last_active_at
                enriched_users.append(nearby_user)
        
        # 记录搜索行为
        if current_user_id:
            user_service.record_user_behavior({
                "user_id": current_user_id,
                "action_type": "nearby_search",
                "action_data": {
                    "search_latitude": request_data.latitude,
                    "search_longitude": request_data.longitude,
                    "search_radius": request_data.radius,
                    "results_count": len(enriched_users)
                }
            })
        
        return NearbyUserResponse(
            users=enriched_users,
            total_count=len(enriched_users),
            search_center={
                "latitude": request_data.latitude,
                "longitude": request_data.longitude
            },
            search_radius=request_data.radius
        )
        
    except Exception as e:
        logger.error(f"查找附近用户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="查找附近用户失败"
        )


@router.get("/me/current", response_model=dict)
async def get_current_location(
    current_user_id: str = Depends(get_current_user_id),
    _: None = Depends(log_user_activity)
):
    """
    获取当前用户的位置信息
    """
    try:
        location = location_service.get_user_current_location(current_user_id)
        
        if not location:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="未找到位置信息"
            )
        
        return {
            "status": "success",
            "data": location
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取当前位置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取当前位置失败"
        )


@router.get("/me/history", response_model=dict)
async def get_location_history(
    days: int = Query(default=7, ge=1, le=30, description="获取最近几天的数据"),
    limit: int = Query(default=100, ge=1, le=500, description="最大返回数量"),
    current_user_id: str = Depends(get_current_user_id),
    _: None = Depends(log_user_activity)
):
    """
    获取当前用户的位置历史记录
    
    - **days**: 获取最近几天的数据 (1-30天)
    - **limit**: 最大返回数量 (1-500)
    """
    try:
        history = location_service.get_user_location_history(
            current_user_id, 
            days=days, 
            limit=limit
        )
        
        return {
            "status": "success",
            "data": {
                "history": history,
                "total_count": len(history),
                "days_range": days
            }
        }
        
    except Exception as e:
        logger.error(f"获取位置历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取位置历史失败"
        )


@router.get("/stats/density", response_model=dict)
async def get_area_density(
    latitude: float = Query(..., ge=-90, le=90, description="纬度"),
    longitude: float = Query(..., ge=-180, le=180, description="经度"),
    precision: int = Query(default=8, ge=5, le=12, description="Geohash精度"),
    _: None = Depends(rate_limiter)
):
    """
    获取指定区域的用户密度
    
    - **latitude**: 纬度
    - **longitude**: 经度  
    - **precision**: Geohash精度 (5-12)
    """
    try:
        from app.services.geohash_service import geohash_service
        
        # 生成指定精度的Geohash
        geohash = geohash_service.encode(latitude, longitude, precision)
        
        # 获取用户密度
        density = location_service.get_area_user_density(geohash)
        
        # 获取精度信息
        lat_error, lon_error = geohash_service.get_precision_error(precision)
        
        return {
            "status": "success",
            "data": {
                "geohash": geohash,
                "precision": precision,
                "user_density": density,
                "area_info": {
                    "latitude_error_meters": lat_error,
                    "longitude_error_meters": lon_error,
                    "approximate_area_km2": (lat_error * lon_error) / 1000000
                },
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                }
            }
        }
        
    except Exception as e:
        logger.error(f"获取区域密度失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取区域密度失败"
        )


@router.post("/cleanup", response_model=BaseResponse)
async def cleanup_expired_locations(
    current_user_id: str = Depends(get_current_user_id)
):
    """
    清理过期的位置数据（管理员功能）
    """
    try:
        # 这里可以添加管理员权限检查
        # 暂时允许所有用户调用
        
        location_service.cleanup_expired_locations()
        
        return BaseResponse(message="过期位置数据清理完成")
        
    except Exception as e:
        logger.error(f"清理过期位置数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清理过期位置数据失败"
        )


@router.get("/geohash/info", response_model=dict)
async def get_geohash_info(
    latitude: float = Query(..., ge=-90, le=90, description="纬度"),
    longitude: float = Query(..., ge=-180, le=180, description="经度"),
    precision: int = Query(default=8, ge=1, le=12, description="Geohash精度")
):
    """
    获取Geohash编码信息
    
    - **latitude**: 纬度
    - **longitude**: 经度
    - **precision**: Geohash精度 (1-12)
    """
    try:
        from app.services.geohash_service import geohash_service
        
        # 编码
        geohash = geohash_service.encode(latitude, longitude, precision)
        
        # 解码验证
        decoded_lat, decoded_lon, lat_range, lon_range = geohash_service.decode(geohash)
        
        # 获取相邻区域
        neighbors = geohash_service.get_neighbors(geohash)
        
        # 获取精度信息
        lat_error, lon_error = geohash_service.get_precision_error(precision)
        
        return {
            "status": "success",
            "data": {
                "input": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "precision": precision
                },
                "geohash": geohash,
                "decoded": {
                    "latitude": decoded_lat,
                    "longitude": decoded_lon,
                    "latitude_range": lat_range,
                    "longitude_range": lon_range
                },
                "neighbors": neighbors,
                "precision_info": {
                    "latitude_error_meters": lat_error,
                    "longitude_error_meters": lon_error
                }
            }
        }
        
    except Exception as e:
        logger.error(f"获取Geohash信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取Geohash信息失败"
        )
