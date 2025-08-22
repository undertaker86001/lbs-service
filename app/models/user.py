"""
用户相关数据模型
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.sql import func

from app.core.database import Base
from .base import TimeStampMixin


# SQLAlchemy模型
class User(Base, TimeStampMixin):
    """用户表模型"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(50), unique=True, index=True, nullable=False, comment="用户唯一标识")
    username = Column(String(50), unique=True, index=True, nullable=True, comment="用户名")
    nickname = Column(String(100), nullable=True, comment="昵称")
    avatar_url = Column(String(500), nullable=True, comment="头像URL")
    gender = Column(String(10), nullable=True, comment="性别")
    age = Column(Integer, nullable=True, comment="年龄")
    bio = Column(Text, nullable=True, comment="个人简介")
    is_active = Column(Boolean, default=True, nullable=False, comment="是否激活")
    privacy_level = Column(String(20), default="medium", nullable=False, comment="隐私级别")
    last_active_at = Column(DateTime, nullable=True, comment="最后活跃时间")


class UserLocation(Base, TimeStampMixin):
    """用户位置表模型"""
    __tablename__ = "user_locations"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(50), index=True, nullable=False, comment="用户ID")
    latitude = Column(Float, nullable=False, comment="纬度")
    longitude = Column(Float, nullable=False, comment="经度")
    geohash = Column(String(20), index=True, nullable=False, comment="Geohash编码")
    anonymous_geohash = Column(String(20), index=True, nullable=True, comment="匿名Geohash编码")
    accuracy = Column(Float, nullable=True, comment="定位精度")
    address = Column(String(500), nullable=True, comment="地址")
    is_current = Column(Boolean, default=True, nullable=False, comment="是否为当前位置")


class UserBehavior(Base, TimeStampMixin):
    """用户行为表模型"""
    __tablename__ = "user_behaviors"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(50), index=True, nullable=False, comment="用户ID")
    action_type = Column(String(50), nullable=False, comment="行为类型")
    action_data = Column(Text, nullable=True, comment="行为数据JSON")
    session_id = Column(String(100), nullable=True, comment="会话ID")
    device_type = Column(String(50), nullable=True, comment="设备类型")
    ip_address = Column(String(50), nullable=True, comment="IP地址")


# Pydantic模型
class UserCreate(BaseModel):
    """创建用户请求模型"""
    user_id: str = Field(..., min_length=1, max_length=50, description="用户唯一标识")
    username: Optional[str] = Field(None, max_length=50, description="用户名")
    nickname: Optional[str] = Field(None, max_length=100, description="昵称")
    avatar_url: Optional[str] = Field(None, max_length=500, description="头像URL")
    gender: Optional[str] = Field(None, description="性别")
    age: Optional[int] = Field(None, ge=1, le=120, description="年龄")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")
    privacy_level: str = Field(default="medium", description="隐私级别")
    
    @validator('privacy_level')
    def validate_privacy_level(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError('隐私级别必须是 low, medium, high 之一')
        return v


class UserUpdate(BaseModel):
    """更新用户请求模型"""
    username: Optional[str] = Field(None, max_length=50, description="用户名")
    nickname: Optional[str] = Field(None, max_length=100, description="昵称")
    avatar_url: Optional[str] = Field(None, max_length=500, description="头像URL")
    gender: Optional[str] = Field(None, description="性别")
    age: Optional[int] = Field(None, ge=1, le=120, description="年龄")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")
    privacy_level: Optional[str] = Field(None, description="隐私级别")
    
    @validator('privacy_level')
    def validate_privacy_level(cls, v):
        if v is not None and v not in ['low', 'medium', 'high']:
            raise ValueError('隐私级别必须是 low, medium, high 之一')
        return v


class UserResponse(BaseModel):
    """用户响应模型"""
    id: int
    user_id: str
    username: Optional[str]
    nickname: Optional[str]
    avatar_url: Optional[str]
    gender: Optional[str]
    age: Optional[int]
    bio: Optional[str]
    is_active: bool
    privacy_level: str
    created_at: datetime
    last_active_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class UserLocationCreate(BaseModel):
    """创建位置请求模型"""
    user_id: str = Field(..., description="用户ID")
    latitude: float = Field(..., ge=-90, le=90, description="纬度")
    longitude: float = Field(..., ge=-180, le=180, description="经度")
    accuracy: Optional[float] = Field(None, description="定位精度")
    address: Optional[str] = Field(None, max_length=500, description="地址")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="时间戳")


class UserLocationResponse(BaseModel):
    """位置响应模型"""
    id: int
    user_id: str
    latitude: float
    longitude: float
    geohash: str
    anonymous_geohash: Optional[str]
    accuracy: Optional[float]
    address: Optional[str]
    is_current: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class NearbyUserRequest(BaseModel):
    """查找附近用户请求模型"""
    latitude: float = Field(..., ge=-90, le=90, description="中心点纬度")
    longitude: float = Field(..., ge=-180, le=180, description="中心点经度")
    radius: float = Field(default=1000, ge=100, le=10000, description="搜索半径(米)")
    limit: int = Field(default=50, ge=1, le=100, description="返回结果数量限制")
    exclude_user_id: Optional[str] = Field(None, description="排除的用户ID")


class NearbyUser(BaseModel):
    """附近用户模型"""
    user_id: str = Field(..., description="用户ID")
    username: Optional[str] = Field(None, description="用户名")
    nickname: Optional[str] = Field(None, description="昵称")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    distance: float = Field(..., description="距离(米)")
    latitude: float = Field(..., description="纬度")
    longitude: float = Field(..., description="经度")
    geohash: str = Field(..., description="Geohash编码")
    similarity_score: Optional[float] = Field(None, description="特征相似度得分")
    comprehensive_score: Optional[float] = Field(None, description="综合评分")
    last_active_at: Optional[datetime] = Field(None, description="最后活跃时间")


class NearbyUserResponse(BaseModel):
    """查找附近用户响应模型"""
    users: List[NearbyUser] = Field(..., description="附近用户列表")
    total_count: int = Field(..., description="总用户数量")
    search_center: dict = Field(..., description="搜索中心点")
    search_radius: float = Field(..., description="搜索半径")
    
    
class UserBehaviorCreate(BaseModel):
    """创建用户行为请求模型"""
    user_id: str = Field(..., description="用户ID")
    action_type: str = Field(..., description="行为类型")
    action_data: Optional[dict] = Field(None, description="行为数据")
    session_id: Optional[str] = Field(None, description="会话ID")
    device_type: Optional[str] = Field(None, description="设备类型")
    ip_address: Optional[str] = Field(None, description="IP地址")


class UserBehaviorResponse(BaseModel):
    """用户行为响应模型"""
    id: int
    user_id: str
    action_type: str
    action_data: Optional[str]
    session_id: Optional[str]
    device_type: Optional[str]
    ip_address: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True
