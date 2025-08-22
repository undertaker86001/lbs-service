"""
位置相关API接口
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db, get_redis
from app.core.security import get_current_user_id
from app.services.location_service import LocationService
from app.services.geohash_service import geohash_service
from app.models.user import NearbyUser, UserLocationCreate, UserLocationResponse
from app.models.base import ApiResponse, PaginatedResponse

router = APIRouter(prefix="/api/v1/location", tags=["位置服务"])


@router.post("/update", response_model=ApiResponse)
async def update_user_location(
    location_data: UserLocationCreate,
    current_user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """
    更新用户位置信息
    
    - **latitude**: 纬度 (-90 到 90)
    - **longitude**: 经度 (-180 到 180)
    - **accuracy**: 定位精度（米）
    - **address**: 地址描述（可选）
    """
    try:
        location_service = LocationService(redis_client, db)
        
        # 更新用户位置
        result = await location_service.update_user_location(
            user_id=current_user_id,
            latitude=location_data.latitude,
            longitude=location_data.longitude,
            geohash=location_data.geohash,
            anonymous_geohash=location_data.anonymous_geohash
        )
        
        return ApiResponse(
            success=True,
            message="位置信息更新成功",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新位置失败: {str(e)}")


@router.get("/nearby", response_model=ApiResponse[List[NearbyUser]])
async def find_nearby_users(
    latitude: float = Query(..., description="当前纬度"),
    longitude: float = Query(..., description="当前经度"),
    radius_meters: float = Query(1000, description="搜索半径（米）", ge=100, le=10000),
    limit: int = Query(50, description="返回结果数量", ge=1, le=100),
    current_user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """
    查找附近的用户
    
    - **latitude**: 当前用户纬度
    - **longitude**: 当前用户经度
    - **radius_meters**: 搜索半径（米），范围100-10000
    - **limit**: 返回结果数量，范围1-100
    """
    try:
        location_service = LocationService(redis_client, db)
        
        # 查找附近用户
        nearby_users = await location_service.find_nearby_users(
            user_id=current_user_id,
            latitude=latitude,
            longitude=longitude,
            radius_meters=radius_meters,
            limit=limit
        )
        
        return ApiResponse(
            success=True,
            message=f"找到 {len(nearby_users)} 个附近用户",
            data=nearby_users
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查找附近用户失败: {str(e)}")


@router.get("/users/{geohash}", response_model=ApiResponse[List[int]])
async def find_users_by_geohash(
    geohash: str,
    limit: int = Query(100, description="返回结果数量", ge=1, le=500),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """
    根据Geohash查找用户
    
    - **geohash**: Geohash编码
    - **limit**: 返回结果数量，范围1-500
    """
    try:
        location_service = LocationService(redis_client, db)
        
        # 验证Geohash格式
        if not geohash or len(geohash) > 12:
            raise HTTPException(status_code=400, detail="无效的Geohash编码")
        
        # 查找用户
        user_ids = await location_service.find_users_by_geohash(geohash, limit)
        
        return ApiResponse(
            success=True,
            message=f"Geohash {geohash} 区域内找到 {len(user_ids)} 个用户",
            data=user_ids
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查找用户失败: {str(e)}")


@router.get("/area/users", response_model=ApiResponse)
async def find_users_in_area(
    center_lat: float = Query(..., description="中心点纬度"),
    center_lon: float = Query(..., description="中心点经度"),
    radius_meters: float = Query(1000, description="搜索半径（米）", ge=100, le=10000),
    limit: int = Query(100, description="返回结果数量", ge=1, le=500),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """
    在指定区域内查找用户
    
    - **center_lat**: 中心点纬度
    - **center_lon**: 中心点经度
    - **radius_meters**: 搜索半径（米），范围100-10000
    - **limit**: 返回结果数量，范围1-500
    """
    try:
        location_service = LocationService(redis_client, db)
        
        # 查找用户
        users = await location_service.find_users_in_area(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_meters=radius_meters,
            limit=limit
        )
        
        return ApiResponse(
            success=True,
            message=f"在指定区域内找到 {len(users)} 个用户",
            data=users
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查找用户失败: {str(e)}")


@router.get("/user/{user_id}", response_model=ApiResponse[UserLocationResponse])
async def get_user_location(
    user_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """
    获取用户位置信息
    
    - **user_id**: 用户ID
    """
    try:
        location_service = LocationService(redis_client, db)
        
        # 获取用户位置
        location_info = await location_service.get_user_location(user_id)
        
        if not location_info:
            raise HTTPException(status_code=404, detail="用户位置信息不存在")
        
        # 转换为响应模型
        response_data = UserLocationResponse(
            user_id=location_info["user_id"],
            latitude=location_info["latitude"],
            longitude=location_info["longitude"],
            geohash=location_info["geohash"],
            anonymous_geohash=location_info["anonymous_geohash"],
            timestamp=location_info["timestamp"],
            updated_at=location_info["updated_at"]
        )
        
        return ApiResponse(
            success=True,
            message="获取用户位置成功",
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户位置失败: {str(e)}")


@router.get("/heatmap/{area_geohash}", response_model=ApiResponse)
async def get_user_density_heatmap(
    area_geohash: str,
    precision: int = Query(6, description="精度位数", ge=1, le=12),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """
    获取用户密度热力图数据
    
    - **area_geohash**: 区域Geohash编码
    - **precision**: 精度位数，范围1-12
    """
    try:
        location_service = LocationService(redis_client, db)
        
        # 验证Geohash格式
        if not area_geohash or len(area_geohash) > 12:
            raise HTTPException(status_code=400, detail="无效的Geohash编码")
        
        # 获取热力图数据
        heatmap_data = await location_service.get_user_density_heatmap(area_geohash, precision)
        
        return ApiResponse(
            success=True,
            message="获取热力图数据成功",
            data=heatmap_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取热力图数据失败: {str(e)}")


@router.get("/geohash/info/{geohash}", response_model=ApiResponse)
async def get_geohash_info(geohash: str):
    """
    获取Geohash编码的详细信息
    
    - **geohash**: Geohash编码
    """
    try:
        # 验证Geohash格式
        if not geohash or len(geohash) > 12:
            raise HTTPException(status_code=400, detail="无效的Geohash编码")
        
        # 获取Geohash信息
        area_info = geohash_service.get_geohash_area(geohash)
        
        return ApiResponse(
            success=True,
            message="获取Geohash信息成功",
            data=area_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取Geohash信息失败: {str(e)}")


@router.get("/geohash/neighbors/{geohash}", response_model=ApiResponse)
async def get_geohash_neighbors(geohash: str):
    """
    获取Geohash编码的邻居区域
    
    - **geohash**: Geohash编码
    """
    try:
        # 验证Geohash格式
        if not geohash or len(geohash) > 12:
            raise HTTPException(status_code=400, detail="无效的Geohash编码")
        
        # 获取邻居区域
        neighbors = geohash_service.get_neighbors(geohash)
        
        return ApiResponse(
            success=True,
            message=f"获取邻居区域成功，共 {len(neighbors)} 个",
            data=neighbors
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取邻居区域失败: {str(e)}")


@router.post("/geohash/optimize", response_model=ApiResponse)
async def optimize_geohash_precision(
    area_geohash: str,
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """
    优化指定区域的Geohash精度
    
    - **area_geohash**: 区域Geohash编码
    """
    try:
        location_service = LocationService(redis_client, db)
        
        # 验证Geohash格式
        if not area_geohash or len(area_geohash) > 12:
            raise HTTPException(status_code=400, detail="无效的Geohash编码")
        
        # 优化精度
        optimal_precision = await location_service.optimize_geohash_precision(area_geohash)
        
        # 获取精度信息
        precision_info = geohash_service.get_precision_info(optimal_precision)
        
        result = {
            "area_geohash": area_geohash,
            "current_precision": len(area_geohash),
            "optimal_precision": optimal_precision,
            "precision_info": precision_info
        }
        
        return ApiResponse(
            success=True,
            message=f"精度优化完成，建议使用 {optimal_precision} 位精度",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"精度优化失败: {str(e)}")


@router.post("/geohash/encode", response_model=ApiResponse)
async def encode_coordinates(
    latitude: float = Query(..., description="纬度", ge=-90, le=90),
    longitude: float = Query(..., description="经度", ge=-180, le=180),
    precision: int = Query(8, description="精度位数", ge=1, le=12),
    max_area_km2: Optional[float] = Query(None, description="最大允许面积（平方公里）")
):
    """
    将坐标编码为Geohash
    
    - **latitude**: 纬度 (-90 到 90)
    - **longitude**: 经度 (-180 到 180)
    - **precision**: 精度位数，范围1-12
    - **max_area_km2**: 最大允许面积（平方公里），如果提供则自动优化精度
    """
    try:
        if max_area_km2:
            # 在面积约束下编码
            geohash = geohash_service.encode_with_area_constraint(
                latitude, longitude, max_area_km2
            )
            actual_precision = len(geohash)
        else:
            # 使用指定精度编码
            geohash = geohash_service.encode(latitude, longitude, precision)
            actual_precision = precision
        
        # 获取Geohash信息
        area_info = geohash_service.get_geohash_area(geohash)
        
        result = {
            "geohash": geohash,
            "precision": actual_precision,
            "area_info": area_info
        }
        
        return ApiResponse(
            success=True,
            message="坐标编码成功",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"坐标编码失败: {str(e)}")


@router.get("/geohash/decode/{geohash}", response_model=ApiResponse)
async def decode_geohash(geohash: str):
    """
    解码Geohash为坐标
    
    - **geohash**: Geohash编码
    """
    try:
        # 验证Geohash格式
        if not geohash or len(geohash) > 12:
            raise HTTPException(status_code=400, detail="无效的Geohash编码")
        
        # 解码Geohash
        latitude, longitude, lat_range, lon_range = geohash_service.decode(geohash)
        
        # 获取精度信息
        precision_info = geohash_service.get_precision_info(len(geohash))
        
        result = {
            "geohash": geohash,
            "latitude": latitude,
            "longitude": longitude,
            "lat_range": lat_range,
            "lon_range": lon_range,
            "precision": len(geohash),
            "precision_info": precision_info
        }
        
        return ApiResponse(
            success=True,
            message="Geohash解码成功",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geohash解码失败: {str(e)}")


@router.get("/distance", response_model=ApiResponse)
async def calculate_distance(
    lat1: float = Query(..., description="第一个点纬度", ge=-90, le=90),
    lon1: float = Query(..., description="第一个点经度", ge=-180, le=180),
    lat2: float = Query(..., description="第二个点纬度", ge=-90, le=90),
    lon2: float = Query(..., description="第二个点经度", ge=-180, le=180)
):
    """
    计算两个坐标点之间的距离
    
    - **lat1, lon1**: 第一个点的经纬度
    - **lat2, lon2**: 第二个点的经纬度
    """
    try:
        # 计算距离
        distance_meters = geohash_service.calculate_distance(lat1, lon1, lat2, lon2)
        
        result = {
            "point1": {"latitude": lat1, "longitude": lon1},
            "point2": {"latitude": lat2, "longitude": lon2},
            "distance_meters": round(distance_meters, 2),
            "distance_km": round(distance_meters / 1000, 4)
        }
        
        return ApiResponse(
            success=True,
            message="距离计算成功",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"距离计算失败: {str(e)}")


@router.post("/cleanup", response_model=ApiResponse)
async def cleanup_expired_locations(
    current_user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
):
    """
    清理过期的位置数据（管理员功能）
    """
    try:
        # 这里应该检查用户权限，暂时跳过
        location_service = LocationService(redis_client, db)
        
        # 清理过期数据
        await location_service.cleanup_expired_locations()
        
        return ApiResponse(
            success=True,
            message="过期位置数据清理完成"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理过期数据失败: {str(e)}")
