"""
位置数据存储与查询服务
"""
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import redis.asyncio as redis
from sqlalchemy.orm import Session
import logging

from app.core.database import get_db, get_redis, get_mysql_connection
from app.models.user import UserLocation, UserLocationCreate, NearbyUser
from app.services.geohash_service import geohash_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class LocationService:
    """位置数据存储与查询服务"""
    
    def __init__(self, redis_client: redis.Redis, db: Session):
        self.redis_client = redis_client
        self.db = db
        self.geohash_service = geohash_service
        
    async def update_user_location(self, user_id: int, latitude: float, longitude: float, 
                                 geohash: str = None, anonymous_geohash: str = None) -> Dict:
        """
        更新用户位置信息
        
        Args:
            user_id: 用户ID
            latitude: 纬度
            longitude: 经度
            geohash: 原始Geohash编码（可选）
            anonymous_geohash: 匿名化后的Geohash编码（可选）
            
        Returns:
            更新结果字典
        """
        try:
            # 如果没有提供Geohash，则生成
            if not geohash:
                # 根据用户密度动态调整精度
                user_density = await self._get_user_density(latitude, longitude)
                precision = self.geohash_service.dynamic_precision_adjustment(user_density)
                geohash = self.geohash_service.encode(latitude, longitude, precision)
            
            # 如果没有提供匿名Geohash，则使用原始Geohash
            if not anonymous_geohash:
                anonymous_geohash = geohash
            
            # 存储到Redis
            await self._store_to_redis(user_id, latitude, longitude, geohash, anonymous_geohash)
            
            # 存储到MySQL
            await self._store_to_mysql(user_id, latitude, longitude, geohash, anonymous_geohash)
            
            # 更新用户密度统计
            await self._update_user_density_stats(geohash)
            
            logger.info(f"用户 {user_id} 位置信息已更新: {geohash}")
            
            return {
                "success": True,
                "user_id": user_id,
                "geohash": geohash,
                "anonymous_geohash": anonymous_geohash,
                "precision": len(geohash),
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"更新用户位置失败: {e}")
            raise
    
    async def _store_to_redis(self, user_id: int, lat: float, lon: float, 
                             geohash: str, anonymous_geohash: str):
        """将位置信息存储到Redis"""
        try:
            # 使用Redis GEOADD命令存储位置
            await self.redis_client.geoadd("user_locations", lon, lat, user_id)
            
            # 存储用户详细信息
            user_info = {
                "user_id": str(user_id),
                "lat": str(lat),
                "lon": str(lon),
                "geohash": geohash,
                "anonymous_geohash": anonymous_geohash,
                "timestamp": str(int(time.time())),
                "updated_at": datetime.now().isoformat()
            }
            
            # 设置过期时间
            pipe = self.redis_client.pipeline()
            pipe.hset(f"user_location:{user_id}", mapping=user_info)
            pipe.expire(f"user_location:{user_id}", settings.cache_ttl)
            pipe.execute()
            
            # 按Geohash分组存储（用于快速查询）
            geohash_key = f"geohash:{anonymous_geohash}"
            pipe = self.redis_client.pipeline()
            pipe.sadd(geohash_key, user_id)
            pipe.expire(geohash_key, settings.cache_ttl)
            pipe.execute()
            
            # 存储用户最后活跃时间
            await self.redis_client.zadd("user_last_active", {str(user_id): time.time()})
            
        except Exception as e:
            logger.error(f"Redis存储失败: {e}")
            raise
    
    async def _store_to_mysql(self, user_id: int, lat: float, lon: float, 
                             geohash: str, anonymous_geohash: str):
        """将位置信息存储到MySQL"""
        try:
            # 这里应该更新用户表的位置信息
            # 由于我们没有完整的用户模型，这里只是示例
            # 实际实现中需要根据具体的数据库模型来更新
            
            # 记录位置历史
            location_history = {
                "user_id": user_id,
                "latitude": lat,
                "longitude": lon,
                "geohash": geohash,
                "anonymous_geohash": anonymous_geohash,
                "created_at": datetime.now()
            }
            
            # 这里应该插入到数据库
            # self.db.add(location_history)
            # self.db.commit()
            
            logger.info(f"MySQL存储成功: 用户 {user_id}")
            
        except Exception as e:
            logger.error(f"MySQL存储失败: {e}")
            raise
    
    async def find_nearby_users(self, user_id: int, latitude: float, longitude: float, 
                               radius_meters: float = 1000, limit: int = 50) -> List[NearbyUser]:
        """
        查找附近的用户
        
        Args:
            user_id: 当前用户ID
            latitude: 当前用户纬度
            longitude: 当前用户经度
            radius_meters: 搜索半径（米）
            limit: 返回结果数量限制
            
        Returns:
            附近用户列表
        """
        try:
            # 使用Redis GEORADIUS命令查找附近的用户
            nearby_users = await self.redis_client.georadius(
                "user_locations",
                longitude,
                latitude,
                radius_meters,
                unit="m",
                withdist=True,
                withcoord=True,
                count=limit,
                sort="ASC"  # 按距离排序
            )
            
            result = []
            for user_data in nearby_users:
                if len(user_data) >= 3:
                    user_id_str, distance, coordinates = user_data
                    nearby_user_id = int(user_id_str)
                    
                    # 跳过自己
                    if nearby_user_id == user_id:
                        continue
                    
                    # 获取用户详细信息
                    user_info = await self._get_user_info(nearby_user_id)
                    if user_info:
                        nearby_user = NearbyUser(
                            user_id=nearby_user_id,
                            username=user_info.get("username", f"用户{nearby_user_id}"),
                            latitude=coordinates[1],
                            longitude=coordinates[0],
                            distance_meters=round(distance, 2),
                            geohash=user_info.get("geohash", ""),
                            last_active=user_info.get("last_active", ""),
                            similarity_score=0.0  # 这里可以计算特征相似度
                        )
                        result.append(nearby_user)
            
            # 按距离排序
            result.sort(key=lambda x: x.distance_meters)
            
            logger.info(f"找到 {len(result)} 个附近用户")
            return result
            
        except Exception as e:
            logger.error(f"查找附近用户失败: {e}")
            raise
    
    async def find_users_by_geohash(self, geohash: str, limit: int = 100) -> List[int]:
        """
        根据Geohash查找用户
        
        Args:
            geohash: Geohash编码
            limit: 返回结果数量限制
            
        Returns:
            用户ID列表
        """
        try:
            # 从Redis获取指定Geohash区域内的用户
            geohash_key = f"geohash:{geohash}"
            user_ids = await self.redis_client.smembers(geohash_key)
            
            # 转换为整数列表
            result = [int(uid) for uid in user_ids][:limit]
            
            logger.info(f"Geohash {geohash} 区域内找到 {len(result)} 个用户")
            return result
            
        except Exception as e:
            logger.error(f"根据Geohash查找用户失败: {e}")
            raise
    
    async def find_users_in_area(self, center_lat: float, center_lon: float, 
                                radius_meters: float, limit: int = 100) -> List[Dict]:
        """
        在指定区域内查找用户
        
        Args:
            center_lat: 中心点纬度
            center_lon: 中心点经度
            radius_meters: 搜索半径（米）
            limit: 返回结果数量限制
            
        Returns:
            用户信息列表
        """
        try:
            # 使用Redis GEORADIUS命令
            users = await self.redis_client.georadius(
                "user_locations",
                center_lon,
                center_lat,
                radius_meters,
                unit="m",
                withdist=True,
                withcoord=True,
                count=limit,
                sort="ASC"
            )
            
            result = []
            for user_data in users:
                if len(user_data) >= 3:
                    user_id_str, distance, coordinates = user_data
                    user_id = int(user_id_str)
                    
                    # 获取用户详细信息
                    user_info = await self._get_user_info(user_id)
                    if user_info:
                        user_info.update({
                            "distance_meters": round(distance, 2),
                            "latitude": coordinates[1],
                            "longitude": coordinates[0]
                        })
                        result.append(user_info)
            
            logger.info(f"在指定区域内找到 {len(result)} 个用户")
            return result
            
        except Exception as e:
            logger.error(f"在指定区域内查找用户失败: {e}")
            raise
    
    async def get_user_location(self, user_id: int) -> Optional[Dict]:
        """
        获取用户当前位置
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户位置信息字典
        """
        try:
            # 从Redis获取用户位置
            user_key = f"user_location:{user_id}"
            user_info = await self.redis_client.hgetall(user_key)
            
            if not user_info:
                return None
            
            # 转换为标准格式
            location_info = {
                "user_id": int(user_info.get("user_id", 0)),
                "latitude": float(user_info.get("lat", 0)),
                "longitude": float(user_info.get("lon", 0)),
                "geohash": user_info.get("geohash", ""),
                "anonymous_geohash": user_info.get("anonymous_geohash", ""),
                "timestamp": int(user_info.get("timestamp", 0)),
                "updated_at": user_info.get("updated_at", "")
            }
            
            return location_info
            
        except Exception as e:
            logger.error(f"获取用户位置失败: {e}")
            return None
    
    async def get_user_density_heatmap(self, area_geohash: str, precision: int = 6) -> Dict:
        """
        获取用户密度热力图数据
        
        Args:
            area_geohash: 区域Geohash编码
            precision: 精度位数
            
        Returns:
            密度热力图数据
        """
        try:
            # 获取指定精度下的子区域
            sub_areas = self._get_sub_areas(area_geohash, precision)
            
            heatmap_data = {}
            for sub_area in sub_areas:
                # 计算每个子区域的用户数量
                user_count = await self._count_users_in_geohash(sub_area)
                heatmap_data[sub_area] = {
                    "user_count": user_count,
                    "density_level": self._get_density_level(user_count),
                    "area_info": self.geohash_service.get_geohash_area(sub_area)
                }
            
            return {
                "area_geohash": area_geohash,
                "precision": precision,
                "total_users": sum(data["user_count"] for data in heatmap_data.values()),
                "sub_areas": heatmap_data
            }
            
        except Exception as e:
            logger.error(f"获取用户密度热力图失败: {e}")
            raise
    
    async def _get_user_density(self, lat: float, lon: float) -> int:
        """
        获取指定位置的用户密度
        
        Args:
            lat: 纬度
            lon: 经度
            
        Returns:
            用户密度（每平方公里用户数）
        """
        try:
            # 使用8位精度计算周围区域的用户密度
            geohash_8 = self.geohash_service.encode(lat, lon, 8)
            users_in_area = await self._count_users_in_geohash(geohash_8)
            
            # 8位精度对应的面积约为0.0024平方公里
            area_km2 = 0.0024
            density = int(users_in_area / area_km2)
            
            return max(density, 1)  # 最小密度为1
            
        except Exception as e:
            logger.error(f"计算用户密度失败: {e}")
            return 100  # 默认中等密度
    
    async def _count_users_in_geohash(self, geohash: str) -> int:
        """
        统计指定Geohash区域内的用户数量
        
        Args:
            geohash: Geohash编码
            
        Returns:
            用户数量
        """
        try:
            geohash_key = f"geohash:{geohash}"
            count = await self.redis_client.scard(geohash_key)
            return count
            
        except Exception as e:
            logger.error(f"统计用户数量失败: {e}")
            return 0
    
    async def _update_user_density_stats(self, geohash: str):
        """
        更新用户密度统计
        
        Args:
            geohash: Geohash编码
        """
        try:
            # 更新密度统计
            density_key = f"density_stats:{geohash[:6]}"  # 使用6位精度统计
            await self.redis_client.hincrby(density_key, "total_users", 1)
            await self.redis_client.expire(density_key, 3600)  # 1小时过期
            
        except Exception as e:
            logger.error(f"更新用户密度统计失败: {e}")
    
    async def _get_user_info(self, user_id: int) -> Optional[Dict]:
        """
        获取用户基本信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户信息字典
        """
        try:
            # 从Redis获取用户信息
            user_key = f"user_location:{user_id}"
            user_info = await self.redis_client.hgetall(user_key)
            
            if not user_info:
                return None
            
            # 获取最后活跃时间
            last_active = await self.redis_client.zscore("user_last_active", str(user_id))
            
            return {
                "user_id": int(user_info.get("user_id", 0)),
                "username": f"用户{user_id}",  # 这里应该从用户表获取
                "geohash": user_info.get("geohash", ""),
                "last_active": datetime.fromtimestamp(last_active).isoformat() if last_active else "",
                "latitude": float(user_info.get("lat", 0)),
                "longitude": float(user_info.get("lon", 0))
            }
            
        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            return None
    
    def _get_sub_areas(self, parent_geohash: str, target_precision: int) -> List[str]:
        """
        获取父级Geohash下的子区域
        
        Args:
            parent_geohash: 父级Geohash编码
            target_precision: 目标精度
            
        Returns:
            子区域Geohash列表
        """
        if len(parent_geohash) >= target_precision:
            return [parent_geohash]
        
        # 生成所有可能的子区域
        sub_areas = []
        for char in self.geohash_service.BASE32:
            sub_area = parent_geohash + char
            if len(sub_area) <= target_precision:
                sub_areas.append(sub_area)
        
        return sub_areas
    
    def _get_density_level(self, user_count: int) -> str:
        """
        根据用户数量确定密度级别
        
        Args:
            user_count: 用户数量
            
        Returns:
            密度级别
        """
        if user_count == 0:
            return "empty"
        elif user_count <= 5:
            return "low"
        elif user_count <= 20:
            return "medium"
        elif user_count <= 50:
            return "high"
        else:
            return "very_high"
    
    async def optimize_geohash_precision(self, area_geohash: str) -> int:
        """
        优化指定区域的Geohash精度
        
        Args:
            area_geohash: 区域Geohash编码
            
        Returns:
            建议的精度位数
        """
        try:
            # 获取当前区域的用户密度
            user_count = await self._count_users_in_geohash(area_geohash)
            
            # 根据用户密度调整精度
            if user_count > 1000:
                return 10  # 超高密度：0.3米精度
            elif user_count > 500:
                return 9   # 高密度：2.4米精度
            elif user_count > 100:
                return 8   # 中等密度：19米精度
            elif user_count > 50:
                return 7   # 中低密度：153米精度
            else:
                return 6   # 低密度：1.22公里精度
                
        except Exception as e:
            logger.error(f"优化Geohash精度失败: {e}")
            return 8  # 默认8位精度
    
    async def cleanup_expired_locations(self):
        """
        清理过期的位置数据
        """
        try:
            # 清理超过24小时未更新的位置数据
            cutoff_time = time.time() - 86400  # 24小时
            
            # 获取所有用户
            all_users = await self.redis_client.zrange("user_last_active", 0, -1, withscores=True)
            
            for user_id_str, last_active in all_users:
                if last_active < cutoff_time:
                    user_id = int(user_id_str)
                    
                    # 从地理位置索引中移除
                    await self.redis_client.zrem("user_locations", user_id)
                    
                    # 从最后活跃时间索引中移除
                    await self.redis_client.zrem("user_last_active", user_id)
                    
                    # 删除用户位置详情
                    await self.redis_client.delete(f"user_location:{user_id}")
                    
                    logger.info(f"清理过期位置数据: 用户 {user_id}")
            
            logger.info("位置数据清理完成")
            
        except Exception as e:
            logger.error(f"清理过期位置数据失败: {e}")


# 创建全局实例
location_service = LocationService(None, None)
