"""
特征工程服务 - 实现用户多维度特征提取和工程化
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import hashlib

from app.services.geohash_service import geohash_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """特征工程服务 - 实现用户多维度特征提取和工程化"""
    
    def __init__(self):
        self.feature_dir = Path(settings.feature_data_path)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        
        # 特征配置
        self.feature_config = {
            "geographic": {
                "enabled": True,
                "precision_levels": [6, 7, 8, 9],  # Geohash精度级别
                "max_distance_km": 50.0,  # 最大距离范围
                "time_decay_factor": 0.95  # 时间衰减因子
            },
            "behavioral": {
                "enabled": True,
                "time_windows": [1, 7, 30],  # 时间窗口（天）
                "interaction_types": ["like", "message", "visit", "share", "follow"],
                "max_features": 100  # 最大特征数量
            },
            "social": {
                "enabled": True,
                "network_depth": 2,  # 社交网络深度
                "max_friends": 1000,  # 最大好友数量
                "influence_threshold": 0.1  # 影响力阈值
            },
            "demographic": {
                "enabled": True,
                "age_groups": [18, 25, 35, 45, 55, 65],
                "education_levels": 5,
                "income_levels": 5
            },
            "preference": {
                "enabled": True,
                "interest_categories": 20,
                "activity_preferences": 10,
                "location_preferences": 15
            }
        }
        
        # 特征缓存
        self.feature_cache = {}
        self.cache_ttl = 3600  # 缓存TTL（秒）
        
        # 特征统计
        self.feature_stats = {
            "total_features": 0,
            "feature_categories": {},
            "last_updated": None
        }
        
        logger.info("特征工程服务初始化完成")
    
    def extract_user_features(self, user_data: Dict, 
                            context_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        提取用户多维度特征
        
        Args:
            user_data: 用户数据
            context_data: 上下文数据（可选）
            
        Returns:
            特征字典，包含各个维度的特征向量
        """
        try:
            user_id = user_data.get("id")
            logger.info(f"开始提取用户 {user_id} 的特征")
            
            features = {}
            
            # 1. 地理位置特征
            if self.feature_config["geographic"]["enabled"]:
                features["geographic"] = self._extract_geographic_features(user_data)
            
            # 2. 行为特征
            if self.feature_config["behavioral"]["enabled"]:
                features["behavioral"] = self._extract_behavioral_features(user_data)
            
            # 3. 社交特征
            if self.feature_config["social"]["enabled"]:
                features["social"] = self._extract_social_features(user_data)
            
            # 4. 人口统计特征
            if self.feature_config["demographic"]["enabled"]:
                features["demographic"] = self._extract_demographic_features(user_data)
            
            # 5. 偏好特征
            if self.feature_config["preference"]["enabled"]:
                features["preference"] = self._extract_preference_features(user_data)
            
            # 6. 上下文特征（如果有）
            if context_data:
                features["contextual"] = self._extract_contextual_features(user_data, context_data)
            
            # 更新特征统计
            self._update_feature_stats(features)
            
            logger.info(f"用户 {user_id} 特征提取完成，共 {len(features)} 个维度")
            return features
            
        except Exception as e:
            logger.error(f"提取用户特征失败: {e}")
            raise
    
    def _extract_geographic_features(self, user_data: Dict) -> np.ndarray:
        """提取地理位置特征"""
        try:
            features = []
            
            # 基础位置信息
            current_location = user_data.get("current_location", {})
            if current_location:
                lat = current_location.get("latitude", 0)
                lon = current_location.get("longitude", 0)
                
                # 坐标归一化
                features.extend([
                    lat / 90.0,  # 纬度归一化
                    lon / 180.0,  # 经度归一化
                ])
                
                # Geohash编码特征
                geohash_code = geohash_service.encode(lat, lon)
                for precision in self.feature_config["geographic"]["precision_levels"]:
                    if len(geohash_code) >= precision:
                        # 将Geohash转换为数值特征
                        hash_value = int(geohash_code[:precision], 32)
                        features.append(hash_value / (32 ** precision))
                    else:
                        features.append(0.0)
                
                # 位置稳定性特征
                location_history = user_data.get("location_history", [])
                if location_history:
                    stability_score = self._calculate_location_stability(location_history)
                    features.append(stability_score)
                else:
                    features.append(0.0)
                
                # 活动范围特征
                activity_radius = self._calculate_activity_radius(location_history)
                features.append(min(activity_radius / 100.0, 1.0))  # 归一化到100km
                
            else:
                # 如果没有位置信息，填充默认值
                features.extend([0.0] * (2 + len(self.feature_config["geographic"]["precision_levels"]) + 2))
            
            # 位置偏好特征
            location_preferences = user_data.get("location_preferences", {})
            if location_preferences:
                # 常用地点类型偏好
                place_types = ["home", "work", "entertainment", "shopping", "transport"]
                for place_type in place_types:
                    preference_score = location_preferences.get(place_type, 0.0)
                    features.append(min(preference_score / 100.0, 1.0))
            else:
                features.extend([0.0] * 5)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"提取地理位置特征失败: {e}")
            return np.zeros(10)  # 返回默认特征向量
    
    def _extract_behavioral_features(self, user_data: Dict) -> np.ndarray:
        """提取行为特征"""
        try:
            features = []
            
            # 交互行为特征
            interaction_data = user_data.get("interaction_data", {})
            
            # 时间窗口特征
            for window_days in self.feature_config["behavioral"]["time_windows"]:
                window_interactions = self._get_interactions_in_window(
                    interaction_data, window_days
                )
                
                # 交互频率
                features.append(min(len(window_interactions) / 100.0, 1.0))
                
                # 交互多样性
                interaction_types = [i.get("type") for i in window_interactions]
                diversity_score = len(set(interaction_types)) / len(self.feature_config["behavioral"]["interaction_types"])
                features.append(diversity_score)
                
                # 时间分布特征
                time_distribution = self._calculate_time_distribution(window_interactions)
                features.extend(time_distribution)
            
            # 应用使用特征
            app_usage = user_data.get("app_usage", {})
            if app_usage:
                # 使用时长
                daily_minutes = app_usage.get("daily_minutes", 0)
                features.append(min(daily_minutes / 1440.0, 1.0))  # 归一化到24小时
                
                # 使用频率
                session_count = app_usage.get("session_count", 0)
                features.append(min(session_count / 50.0, 1.0))
                
                # 活跃时段
                active_hours = app_usage.get("active_hours", [])
                if active_hours:
                    morning_active = sum(1 for h in active_hours if 6 <= h <= 12) / len(active_hours)
                    afternoon_active = sum(1 for h in active_hours if 12 < h <= 18) / len(active_hours)
                    evening_active = sum(1 for h in active_hours if 18 < h <= 24) / len(active_hours)
                    night_active = sum(1 for h in active_hours if 0 <= h < 6) / len(active_hours)
                    
                    features.extend([morning_active, afternoon_active, evening_active, night_active])
                else:
                    features.extend([0.0] * 4)
            else:
                features.extend([0.0] * 6)
            
            # 内容偏好特征
            content_preferences = user_data.get("content_preferences", {})
            if content_preferences:
                # 内容类型偏好
                content_types = ["text", "image", "video", "audio", "link"]
                for content_type in content_types:
                    preference_score = content_preferences.get(content_type, 0.0)
                    features.append(min(preference_score / 100.0, 1.0))
            else:
                features.extend([0.0] * 5)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"提取行为特征失败: {e}")
            return np.zeros(20)  # 返回默认特征向量
    
    def _extract_social_features(self, user_data: Dict) -> np.ndarray:
        """提取社交特征"""
        try:
            features = []
            
            # 好友网络特征
            friendship_data = user_data.get("friendship_data", {})
            
            # 基础社交指标
            friend_count = friendship_data.get("friend_count", 0)
            features.append(min(friend_count / 1000.0, 1.0))
            
            mutual_friend_count = friendship_data.get("mutual_friend_count", 0)
            features.append(min(mutual_friend_count / 500.0, 1.0))
            
            # 社交网络密度
            network_density = self._calculate_network_density(friendship_data)
            features.append(network_density)
            
            # 影响力指标
            influence_score = self._calculate_influence_score(friendship_data)
            features.append(influence_score)
            
            # 社交活跃度
            social_activity = user_data.get("social_activity", {})
            if social_activity:
                # 发帖频率
                post_frequency = social_activity.get("post_frequency", 0)
                features.append(min(post_frequency / 100.0, 1.0))
                
                # 评论频率
                comment_frequency = social_activity.get("comment_frequency", 0)
                features.append(min(comment_frequency / 200.0, 1.0))
                
                # 分享频率
                share_frequency = social_activity.get("share_frequency", 0)
                features.append(min(share_frequency / 50.0, 1.0))
            else:
                features.extend([0.0] * 3)
            
            # 社交圈层特征
            social_circles = user_data.get("social_circles", [])
            if social_circles:
                # 圈层数量
                features.append(min(len(social_circles) / 10.0, 1.0))
                
                # 圈层活跃度
                circle_activity = np.mean([c.get("activity_level", 0) for c in social_circles])
                features.append(min(circle_activity / 100.0, 1.0))
            else:
                features.extend([0.0] * 2)
            
            # 社交偏好
            social_preferences = user_data.get("social_preferences", {})
            if social_preferences:
                # 社交方式偏好
                social_methods = ["online", "offline", "group", "one_on_one"]
                for method in social_methods:
                    preference_score = social_preferences.get(method, 0.0)
                    features.append(min(preference_score / 100.0, 1.0))
            else:
                features.extend([0.0] * 4)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"提取社交特征失败: {e}")
            return np.zeros(15)  # 返回默认特征向量
    
    def _extract_demographic_features(self, user_data: Dict) -> np.ndarray:
        """提取人口统计特征"""
        try:
            features = []
            
            profile = user_data.get("profile", {})
            
            # 年龄特征
            age = profile.get("age", 25)
            # 年龄组编码
            age_group = 0
            for i, threshold in enumerate(self.feature_config["demographic"]["age_groups"]):
                if age >= threshold:
                    age_group = i
            features.append(age_group / len(self.feature_config["demographic"]["age_groups"]))
            
            # 性别特征
            gender = profile.get("gender", 0)
            features.append(gender)
            
            # 教育程度
            education_level = profile.get("education_level", 0)
            features.append(education_level / self.feature_config["demographic"]["education_levels"])
            
            # 收入水平
            income_level = profile.get("income_level", 0)
            features.append(income_level / self.feature_config["demographic"]["income_levels"])
            
            # 职业特征
            occupation = profile.get("occupation", "")
            if occupation:
                # 职业类型编码
                occupation_types = ["student", "professional", "service", "business", "other"]
                occupation_code = 0
                for i, occ_type in enumerate(occupation_types):
                    if occ_type in occupation.lower():
                        occupation_code = i
                        break
                features.append(occupation_code / len(occupation_types))
            else:
                features.append(0.0)
            
            # 婚姻状况
            marital_status = profile.get("marital_status", 0)
            features.append(marital_status / 4.0)  # 假设有4种状态
            
            # 居住地特征
            residence = profile.get("residence", {})
            if residence:
                # 城市等级
                city_level = residence.get("city_level", 0)
                features.append(city_level / 5.0)
                
                # 居住时长
                residence_years = residence.get("years", 0)
                features.append(min(residence_years / 20.0, 1.0))
            else:
                features.extend([0.0, 0.0])
            
            # 语言能力
            languages = profile.get("languages", [])
            features.append(min(len(languages) / 5.0, 1.0))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"提取人口统计特征失败: {e}")
            return np.zeros(10)  # 返回默认特征向量
    
    def _extract_preference_features(self, user_data: Dict) -> np.ndarray:
        """提取偏好特征"""
        try:
            features = []
            
            # 兴趣特征
            interests = user_data.get("interests", {})
            if interests:
                # 兴趣类别
                categories = interests.get("categories", [])
                interest_vector = [0.0] * self.feature_config["preference"]["interest_categories"]
                for category in categories:
                    if category < len(interest_vector):
                        interest_vector[category] = 1.0
                features.extend(interest_vector)
                
                # 兴趣强度
                intensity = interests.get("intensity", 0.0)
                features.append(min(intensity / 100.0, 1.0))
            else:
                features.extend([0.0] * (self.feature_config["preference"]["interest_categories"] + 1))
            
            # 活动偏好
            activity_preferences = user_data.get("activity_preferences", {})
            if activity_preferences:
                activity_types = ["sports", "music", "travel", "food", "reading", 
                                "gaming", "art", "technology", "fashion", "nature"]
                for activity_type in activity_types:
                    preference_score = activity_preferences.get(activity_type, 0.0)
                    features.append(min(preference_score / 100.0, 1.0))
            else:
                features.extend([0.0] * self.feature_config["preference"]["activity_preferences"])
            
            # 位置偏好
            location_preferences = user_data.get("location_preferences", {})
            if location_preferences:
                location_types = ["urban", "suburban", "rural", "coastal", "mountain",
                                "park", "mall", "restaurant", "cafe", "gym",
                                "library", "museum", "theater", "stadium", "airport"]
                for location_type in location_types:
                    preference_score = location_preferences.get(location_type, 0.0)
                    features.append(min(preference_score / 100.0, 1.0))
            else:
                features.extend([0.0] * self.feature_config["preference"]["location_preferences"])
            
            # 时间偏好
            time_preferences = user_data.get("time_preferences", {})
            if time_preferences:
                # 工作日偏好
                weekday_preference = time_preferences.get("weekday", 0.0)
                features.append(min(weekday_preference / 100.0, 1.0))
                
                # 周末偏好
                weekend_preference = time_preferences.get("weekend", 0.0)
                features.append(min(weekend_preference / 100.0, 1.0))
                
                # 季节偏好
                season_preferences = time_preferences.get("seasons", {})
                seasons = ["spring", "summer", "autumn", "winter"]
                for season in seasons:
                    preference_score = season_preferences.get(season, 0.0)
                    features.append(min(preference_score / 100.0, 1.0))
            else:
                features.extend([0.0] * 6)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"提取偏好特征失败: {e}")
            return np.zeros(50)  # 返回默认特征向量
    
    def _extract_contextual_features(self, user_data: Dict, context_data: Dict) -> np.ndarray:
        """提取上下文特征"""
        try:
            features = []
            
            # 时间上下文
            current_time = datetime.now()
            features.extend([
                current_time.hour / 24.0,  # 小时
                current_time.weekday() / 7.0,  # 星期
                current_time.month / 12.0,  # 月份
            ])
            
            # 天气上下文
            weather = context_data.get("weather", {})
            if weather:
                temperature = weather.get("temperature", 20)
                features.append((temperature + 20) / 60.0)  # 归一化到-20到40度
                
                weather_condition = weather.get("condition", "sunny")
                weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "foggy"]
                weather_code = weather_conditions.index(weather_condition) if weather_condition in weather_conditions else 0
                features.append(weather_code / len(weather_conditions))
            else:
                features.extend([0.5, 0.0])  # 默认值
            
            # 事件上下文
            events = context_data.get("events", [])
            if events:
                # 附近事件数量
                features.append(min(len(events) / 10.0, 1.0))
                
                # 事件类型多样性
                event_types = [e.get("type") for e in events]
                event_diversity = len(set(event_types)) / max(len(event_types), 1)
                features.append(event_diversity)
            else:
                features.extend([0.0, 0.0])
            
            # 交通上下文
            traffic = context_data.get("traffic", {})
            if traffic:
                congestion_level = traffic.get("congestion_level", 0)
                features.append(congestion_level / 5.0)
            else:
                features.append(0.0)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"提取上下文特征失败: {e}")
            return np.zeros(8)  # 返回默认特征向量
    
    def _calculate_location_stability(self, location_history: List[Dict]) -> float:
        """计算位置稳定性分数"""
        try:
            if len(location_history) < 2:
                return 0.0
            
            # 计算位置变化的标准差
            locations = [(loc.get("latitude", 0), loc.get("longitude", 0)) 
                        for loc in location_history]
            
            if len(locations) < 2:
                return 0.0
            
            # 计算相邻位置的距离
            distances = []
            for i in range(1, len(locations)):
                lat1, lon1 = locations[i-1]
                lat2, lon2 = locations[i]
                distance = geohash_service.calculate_distance(lat1, lon1, lat2, lon2)
                distances.append(distance)
            
            if not distances:
                return 0.0
            
            # 稳定性分数：距离标准差越小，稳定性越高
            distance_std = np.std(distances)
            stability_score = max(0, 1 - (distance_std / 10.0))  # 10km作为基准
            
            return stability_score
            
        except Exception as e:
            logger.error(f"计算位置稳定性失败: {e}")
            return 0.0
    
    def _calculate_activity_radius(self, location_history: List[Dict]) -> float:
        """计算活动半径"""
        try:
            if len(location_history) < 2:
                return 0.0
            
            # 计算所有位置的中心点
            latitudes = [loc.get("latitude", 0) for loc in location_history]
            longitudes = [loc.get("longitude", 0) for loc in location_history]
            
            center_lat = np.mean(latitudes)
            center_lon = np.mean(longitudes)
            
            # 计算到中心点的最大距离
            max_distance = 0.0
            for lat, lon in zip(latitudes, longitudes):
                distance = geohash_service.calculate_distance(center_lat, center_lon, lat, lon)
                max_distance = max(max_distance, distance)
            
            return max_distance
            
        except Exception as e:
            logger.error(f"计算活动半径失败: {e}")
            return 0.0
    
    def _get_interactions_in_window(self, interaction_data: Dict, window_days: int) -> List[Dict]:
        """获取指定时间窗口内的交互数据"""
        try:
            interactions = interaction_data.get("interactions", [])
            cutoff_time = datetime.now() - timedelta(days=window_days)
            
            window_interactions = []
            for interaction in interactions:
                interaction_time = datetime.fromtimestamp(interaction.get("timestamp", 0))
                if interaction_time >= cutoff_time:
                    window_interactions.append(interaction)
            
            return window_interactions
            
        except Exception as e:
            logger.error(f"获取时间窗口交互数据失败: {e}")
            return []
    
    def _calculate_time_distribution(self, interactions: List[Dict]) -> List[float]:
        """计算时间分布特征"""
        try:
            if not interactions:
                return [0.0, 0.0, 0.0, 0.0]
            
            # 提取交互时间
            interaction_times = []
            for interaction in interactions:
                timestamp = interaction.get("timestamp", 0)
                if timestamp > 0:
                    interaction_time = datetime.fromtimestamp(timestamp)
                    interaction_times.append(interaction_time.hour)
            
            if not interaction_times:
                return [0.0, 0.0, 0.0, 0.0]
            
            # 计算时间分布
            morning_count = sum(1 for h in interaction_times if 6 <= h <= 12)
            afternoon_count = sum(1 for h in interaction_times if 12 < h <= 18)
            evening_count = sum(1 for h in interaction_times if 18 < h <= 24)
            night_count = sum(1 for h in interaction_times if 0 <= h < 6)
            
            total_count = len(interaction_times)
            
            return [
                morning_count / total_count,
                afternoon_count / total_count,
                evening_count / total_count,
                night_count / total_count
            ]
            
        except Exception as e:
            logger.error(f"计算时间分布失败: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _calculate_network_density(self, friendship_data: Dict) -> float:
        """计算社交网络密度"""
        try:
            friend_count = friendship_data.get("friend_count", 0)
            mutual_friend_count = friendship_data.get("mutual_friend_count", 0)
            
            if friend_count == 0:
                return 0.0
            
            # 网络密度 = 互相关注数 / 总关注数
            density = mutual_friend_count / friend_count
            return min(density, 1.0)
            
        except Exception as e:
            logger.error(f"计算网络密度失败: {e}")
            return 0.0
    
    def _calculate_influence_score(self, friendship_data: Dict) -> float:
        """计算影响力分数"""
        try:
            friend_count = friendship_data.get("friend_count", 0)
            follower_count = friendship_data.get("follower_count", 0)
            engagement_rate = friendship_data.get("engagement_rate", 0.0)
            
            # 影响力分数 = (关注者数 * 互动率) / 1000
            influence_score = (follower_count * engagement_rate) / 1000.0
            
            return min(influence_score, 1.0)
            
        except Exception as e:
            logger.error(f"计算影响力分数失败: {e}")
            return 0.0
    
    def _update_feature_stats(self, features: Dict[str, np.ndarray]):
        """更新特征统计信息"""
        try:
            total_features = sum(len(feature_vector) for feature_vector in features.values())
            
            self.feature_stats["total_features"] = total_features
            self.feature_stats["feature_categories"] = {
                category: len(feature_vector) 
                for category, feature_vector in features.items()
            }
            self.feature_stats["last_updated"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"更新特征统计失败: {e}")
    
    def get_feature_summary(self, user_id: str) -> Dict:
        """获取用户特征摘要"""
        try:
            # 检查缓存
            cache_key = f"feature_summary_{user_id}"
            if cache_key in self.feature_cache:
                cache_entry = self.feature_cache[cache_key]
                if (datetime.now().timestamp() - cache_entry["timestamp"]) < self.cache_ttl:
                    return cache_entry["data"]
            
            # 如果没有缓存，返回空摘要
            summary = {
                "user_id": user_id,
                "feature_count": 0,
                "feature_categories": [],
                "last_updated": None,
                "message": "特征尚未提取，请先调用extract_user_features"
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取特征摘要失败: {e}")
            return {"error": str(e)}
    
    def get_feature_stats(self) -> Dict:
        """获取特征统计信息"""
        return self.feature_stats.copy()
    
    def update_feature_config(self, new_config: Dict):
        """更新特征配置"""
        try:
            for category, config in new_config.items():
                if category in self.feature_config:
                    self.feature_config[category].update(config)
                    logger.info(f"特征配置已更新: {category}")
                else:
                    logger.warning(f"未知的特征类别: {category}")
            
        except Exception as e:
            logger.error(f"更新特征配置失败: {e}")
    
    def export_features(self, user_id: str, features: Dict[str, np.ndarray], 
                       format: str = "json") -> Dict:
        """导出特征数据"""
        try:
            export_data = {
                "user_id": user_id,
                "export_time": datetime.now().isoformat(),
                "feature_count": sum(len(f) for f in features.values()),
                "features": {}
            }
            
            for category, feature_vector in features.items():
                export_data["features"][category] = {
                    "dimension": len(feature_vector),
                    "vector": feature_vector.tolist(),
                    "statistics": {
                        "mean": float(np.mean(feature_vector)),
                        "std": float(np.std(feature_vector)),
                        "min": float(np.min(feature_vector)),
                        "max": float(np.max(feature_vector))
                    }
                }
            
            if format.lower() == "json":
                # 保存到文件
                export_file = self.feature_dir / f"features_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
                export_data["export_file"] = str(export_file)
            
            return export_data
            
        except Exception as e:
            logger.error(f"导出特征数据失败: {e}")
            return {"error": str(e)}


# 创建全局实例
feature_engineering_service = FeatureEngineeringService()
