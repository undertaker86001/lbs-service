"""
Geohash地理位置编码服务
"""
import math
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class GeohashService:
    """Geohash地理位置编码服务"""
    
    # Base32字符集
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    
    def __init__(self):
        # 预计算的邻居查找表
        self.neighbors = {
            'right': {'even': "bc01fg45238967deuvhjyznpkmstqrwx", 'odd': "p0r21436x8zb9dcf5h7kjnmqesgutwvy"},
            'left': {'even': "238967debc01fg45kmstqrwxuvhjyznp", 'odd': "14365h7k9dcfesgujnmqp0r2twvyx8zb"},
            'top': {'even': "p0r21436x8zb9dcf5h7kjnmqesgutwvy", 'odd': "bc01fg45238967deuvhjyznpkmstqrwx"},
            'bottom': {'even': "14365h7k9dcfesgujnmqp0r2twvyx8zb", 'odd': "238967debc01fg45kmstqrwxuvhjyznp"}
        }
        
        self.borders = {
            'right': {'even': "bcfguvyz", 'odd': "prxz"},
            'left': {'even': "0145hjnp", 'odd': "028b"},
            'top': {'even': "prxz", 'odd': "bcfguvyz"},
            'bottom': {'even': "028b", 'odd': "0145hjnp"}
        }
    
    def encode(self, latitude: float, longitude: float, precision: int = 8) -> str:
        """
        将经纬度编码为Geohash字符串
        
        Args:
            latitude: 纬度 (-90 到 90)
            longitude: 经度 (-180 到 180)
            precision: 编码精度 (1-12位)
            
        Returns:
            Geohash编码字符串
        """
        if not (-90 <= latitude <= 90):
            raise ValueError("纬度必须在-90到90之间")
        if not (-180 <= longitude <= 180):
            raise ValueError("经度必须在-180到180之间")
        if not (1 <= precision <= 12):
            raise ValueError("精度必须在1到12之间")
            
        lat_range = [-90.0, 90.0]
        lon_range = [-180.0, 180.0]
        
        geohash = ""
        bits = 0
        bit = 0
        ch = 0
        even = True  # 开始处理经度
        
        while len(geohash) < precision:
            if even:  # 处理经度
                mid = (lon_range[0] + lon_range[1]) / 2
                if longitude >= mid:
                    ch |= (1 << (4 - bit))
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:  # 处理纬度
                mid = (lat_range[0] + lat_range[1]) / 2
                if latitude >= mid:
                    ch |= (1 << (4 - bit))
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            
            even = not even
            bit += 1
            
            if bit == 5:
                geohash += self.BASE32[ch]
                bit = 0
                ch = 0
                
        return geohash
    
    def decode(self, geohash: str) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        """
        解码Geohash字符串为经纬度
        
        Args:
            geohash: Geohash编码字符串
            
        Returns:
            (纬度, 经度, 纬度范围, 经度范围)
        """
        lat_range = [-90.0, 90.0]
        lon_range = [-180.0, 180.0]
        
        even = True
        
        for char in geohash:
            if char not in self.BASE32:
                raise ValueError(f"无效的Geohash字符: {char}")
            
            char_index = self.BASE32.index(char)
            
            for i in range(4, -1, -1):
                bit = (char_index >> i) & 1
                
                if even:  # 处理经度
                    mid = (lon_range[0] + lon_range[1]) / 2
                    if bit:
                        lon_range[0] = mid
                    else:
                        lon_range[1] = mid
                else:  # 处理纬度
                    mid = (lat_range[0] + lat_range[1]) / 2
                    if bit:
                        lat_range[0] = mid
                    else:
                        lat_range[1] = mid
                
                even = not even
        
        latitude = (lat_range[0] + lat_range[1]) / 2
        longitude = (lon_range[0] + lon_range[1]) / 2
        
        return latitude, longitude, tuple(lat_range), tuple(lon_range)
    
    def get_neighbors(self, geohash: str) -> List[str]:
        """
        获取指定Geohash的8个邻居
        
        Args:
            geohash: Geohash编码字符串
            
        Returns:
            邻居Geohash列表
        """
        neighbors = []
        
        # 获取8个方向的邻居
        directions = ['top', 'right', 'bottom', 'left']
        
        for direction in directions:
            neighbor = self._get_neighbor(geohash, direction)
            if neighbor:
                neighbors.append(neighbor)
                
                # 获取对角线方向的邻居
                if direction == 'top':
                    top_right = self._get_neighbor(neighbor, 'right')
                    if top_right:
                        neighbors.append(top_right)
                    top_left = self._get_neighbor(neighbor, 'left')
                    if top_left:
                        neighbors.append(top_left)
                elif direction == 'bottom':
                    bottom_right = self._get_neighbor(neighbor, 'right')
                    if bottom_right:
                        neighbors.append(bottom_right)
                    bottom_left = self._get_neighbor(neighbor, 'left')
                    if bottom_left:
                        neighbors.append(bottom_left)
        
        return neighbors
    
    def _get_neighbor(self, geohash: str, direction: str) -> Optional[str]:
        """
        获取指定方向的邻居Geohash
        
        Args:
            geohash: Geohash编码字符串
            direction: 方向 ('top', 'right', 'bottom', 'left')
            
        Returns:
            邻居Geohash字符串，如果不存在则返回None
        """
        if not geohash:
            return None
            
        # 获取最后一个字符
        last_char = geohash[-1]
        base = geohash[:-1]
        
        # 检查边界
        if last_char in self.borders[direction]['even'] if len(geohash) % 2 == 0 else self.borders[direction]['odd']:
            # 需要递归处理前一个字符
            base = self._get_neighbor(base, direction)
            if not base:
                return None
        
        # 获取邻居字符
        neighbor_char = self.neighbors[direction]['even'][self.BASE32.index(last_char)] if len(geohash) % 2 == 0 else self.neighbors[direction]['odd'][self.BASE32.index(last_char)]
        
        return base + neighbor_char
    
    def dynamic_precision_adjustment(self, user_density: int, area_type: str = "urban") -> int:
        """
        根据用户密度动态调整Geohash精度
        
        Args:
            user_density: 用户密度（每平方公里用户数）
            area_type: 区域类型 ("urban", "suburban", "rural")
            
        Returns:
            建议的Geohash精度位数
        """
        # 基础精度映射
        precision_map = {
            5: 4.9,    # 5位：4.9公里
            6: 1.22,   # 6位：1.22公里
            7: 153,    # 7位：153米
            8: 19,     # 8位：19米
            9: 2.4,    # 9位：2.4米
            10: 0.3,   # 10位：0.3米
            11: 0.037, # 11位：3.7厘米
            12: 0.0046 # 12位：4.6毫米
        }
        
        # 根据用户密度和区域类型调整精度
        if area_type == "urban":
            if user_density > 10000:  # 超高密度（如商业区）
                return 10  # 0.3米精度
            elif user_density > 5000:  # 高密度（如住宅区）
                return 9   # 2.4米精度
            elif user_density > 1000:  # 中等密度
                return 8   # 19米精度
            else:  # 低密度
                return 7   # 153米精度
        elif area_type == "suburban":
            if user_density > 500:
                return 7   # 153米精度
            else:
                return 6   # 1.22公里精度
        else:  # rural
            if user_density > 100:
                return 6   # 1.22公里精度
            else:
                return 5   # 4.9公里精度
    
    def get_precision_info(self, precision: int) -> Dict[str, float]:
        """
        获取指定精度的地理信息
        
        Args:
            precision: Geohash精度位数
            
        Returns:
            包含精度信息的字典
        """
        precision_info = {
            1: {"lat_error": 23.0, "lon_error": 23.0, "area_km2": 5000.0},
            2: {"lat_error": 2.8, "lon_error": 5.6, "area_km2": 625.0},
            3: {"lat_error": 0.35, "lon_error": 0.7, "area_km2": 78.0},
            4: {"lat_error": 0.044, "lon_error": 0.088, "area_km2": 9.8},
            5: {"lat_error": 0.0055, "lon_error": 0.011, "area_km2": 1.22},
            6: {"lat_error": 0.00068, "lon_error": 0.00136, "area_km2": 0.153},
            7: {"lat_error": 0.000085, "lon_error": 0.00017, "area_km2": 0.019},
            8: {"lat_error": 0.0000106, "lon_error": 0.0000212, "area_km2": 0.0024},
            9: {"lat_error": 0.00000133, "lon_error": 0.00000266, "area_km2": 0.0003},
            10: {"lat_error": 0.000000166, "lon_error": 0.000000332, "area_km2": 0.000037},
            11: {"lat_error": 0.0000000208, "lon_error": 0.0000000416, "area_km2": 0.0000046},
            12: {"lat_error": 0.0000000026, "lon_error": 0.0000000052, "area_km2": 0.00000058}
        }
        
        if precision not in precision_info:
            raise ValueError(f"不支持的精度位数: {precision}")
            
        return precision_info[precision]
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算两个坐标点之间的球面距离（Haversine公式）
        
        Args:
            lat1, lon1: 第一个点的经纬度
            lat2, lon2: 第二个点的经纬度
            
        Returns:
            距离（米）
        """
        # 地球半径（米）
        R = 6371000
        
        # 转换为弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # 计算差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine公式
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def get_geohash_area(self, geohash: str) -> Dict[str, float]:
        """
        获取Geohash编码对应的地理区域信息
        
        Args:
            geohash: Geohash编码字符串
            
        Returns:
            区域信息字典
        """
        precision = len(geohash)
        precision_info = self.get_precision_info(precision)
        
        # 解码获取中心点
        center_lat, center_lon, lat_range, lon_range = self.decode(geohash)
        
        return {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "lat_range": lat_range,
            "lon_range": lon_range,
            "precision": precision,
            "lat_error": precision_info["lat_error"],
            "lon_error": precision_info["lon_error"],
            "area_km2": precision_info["area_km2"]
        }
    
    def find_optimal_precision(self, target_area_km2: float) -> int:
        """
        根据目标区域大小找到最优的Geohash精度
        
        Args:
            target_area_km2: 目标区域大小（平方公里）
            
        Returns:
            最优的Geohash精度位数
        """
        for precision in range(1, 13):
            info = self.get_precision_info(precision)
            if info["area_km2"] <= target_area_km2:
                return precision
        
        return 12  # 返回最高精度
    
    def encode_with_area_constraint(self, lat: float, lon: float, max_area_km2: float) -> str:
        """
        在面积约束下编码经纬度
        
        Args:
            lat: 纬度
            lon: 经度
            max_area_km2: 最大允许面积（平方公里）
            
        Returns:
            Geohash编码字符串
        """
        optimal_precision = self.find_optimal_precision(max_area_km2)
        return self.encode(lat, lon, optimal_precision)


# 创建全局实例
geohash_service = GeohashService()
