"""
应用配置模块
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基本配置
    app_name: str = "精准LBS社交匹配系统"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Redis配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_max_connections: int = 20
    
    # MySQL配置
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "lbs_user"
    mysql_password: str = "lbs_password"
    mysql_database: str = "lbs_social"
    mysql_charset: str = "utf8mb4"
    
    # 数据库连接池配置
    mysql_pool_size: int = 5
    mysql_max_overflow: int = 10
    mysql_pool_timeout: int = 30
    mysql_pool_recycle: int = 3600
    
    # 隐私保护配置
    k_anonymity: int = 50
    geohash_max_length: int = 12
    geohash_min_length: int = 5
    
    # 性能配置
    cache_ttl: int = 3600  # 缓存过期时间(秒)
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    max_retries: int = 3
    
    # 监控配置
    prometheus_enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # 日志配置
    log_format: str = "json"
    log_file: str = "logs/app.log"
    log_max_size: str = "100MB"
    log_backup_count: int = 5
    
    # 安全配置
    secret_key: str = "your-secret-key-here"
    jwt_secret_key: str = "your-jwt-secret-key-here"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # CORS配置
    allowed_hosts: List[str] = ["localhost", "127.0.0.1"]
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # 机器学习模型配置
    model_path: str = "models/ralm_model.pth"
    feature_dim: int = 128
    max_sequence_length: int = 100
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # 训练参数
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    
    @property
    def mysql_url(self) -> str:
        """MySQL数据库连接URL"""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset={self.mysql_charset}"
        )
    
    @property
    def redis_url(self) -> str:
        """Redis连接URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()
