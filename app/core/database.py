"""
数据库连接管理模块
"""
import redis
import mysql.connector
from mysql.connector import pooling
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Generator
import logging

from .config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy配置
engine = create_engine(
    settings.mysql_url,
    pool_size=settings.mysql_pool_size,
    max_overflow=settings.mysql_max_overflow,
    pool_timeout=settings.mysql_pool_timeout,
    pool_recycle=settings.mysql_pool_recycle,
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DatabaseManager:
    """数据库连接管理器"""
    
    def __init__(self):
        self._redis_client = None
        self._mysql_pool = None
        
    @property
    def redis_client(self) -> redis.Redis:
        """获取Redis客户端"""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                ssl=settings.redis_ssl,
                max_connections=settings.redis_max_connections,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True
            )
            
        return self._redis_client
    
    @property
    def mysql_pool(self) -> pooling.MySQLConnectionPool:
        """获取MySQL连接池"""
        if self._mysql_pool is None:
            mysql_config = {
                'user': settings.mysql_user,
                'password': settings.mysql_password,
                'host': settings.mysql_host,
                'port': settings.mysql_port,
                'database': settings.mysql_database,
                'charset': settings.mysql_charset,
                'autocommit': False,
                'time_zone': '+00:00'
            }
            
            self._mysql_pool = pooling.MySQLConnectionPool(
                pool_name="lbs_pool",
                pool_size=settings.mysql_pool_size,
                pool_reset_session=True,
                **mysql_config
            )
            
        return self._mysql_pool
    
    @contextmanager
    def get_mysql_connection(self):
        """获取MySQL连接的上下文管理器"""
        connection = None
        try:
            connection = self.mysql_pool.get_connection()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"MySQL连接错误: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def test_redis_connection(self) -> bool:
        """测试Redis连接"""
        try:
            self.redis_client.ping()
            logger.info("Redis连接成功")
            return True
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            return False
    
    def test_mysql_connection(self) -> bool:
        """测试MySQL连接"""
        try:
            with self.get_mysql_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
            logger.info("MySQL连接成功")
            return True
        except Exception as e:
            logger.error(f"MySQL连接失败: {e}")
            return False
    
    def close_connections(self):
        """关闭所有数据库连接"""
        try:
            if self._redis_client:
                self._redis_client.close()
                self._redis_client = None
                
            if self._mysql_pool:
                # MySQL连接池会自动管理连接
                self._mysql_pool = None
                
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {e}")


# 创建全局数据库管理器实例
db_manager = DatabaseManager()


def get_db() -> Generator:
    """获取SQLAlchemy数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """获取Redis客户端"""
    return db_manager.redis_client


def get_mysql_connection():
    """获取MySQL连接"""
    return db_manager.get_mysql_connection()


# 数据库初始化函数
def init_database():
    """初始化数据库表"""
    try:
        # 创建所有表
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表创建成功")
        
        # 测试连接
        if db_manager.test_redis_connection() and db_manager.test_mysql_connection():
            logger.info("数据库初始化完成")
            return True
        else:
            logger.error("数据库连接测试失败")
            return False
            
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False


# 数据库关闭函数
def close_database():
    """关闭数据库连接"""
    db_manager.close_connections()
