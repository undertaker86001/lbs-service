"""
FastAPI主应用程序
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.core.database import init_database, close_database, db_manager
from app.api import users, location, privacy, features, training
from app.models.base import BaseResponse

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("正在启动LBS社交匹配系统...")
    
    # 初始化数据库
    if not init_database():
        logger.error("数据库初始化失败")
        raise RuntimeError("数据库初始化失败")
    
    logger.info("数据库连接成功")
    logger.info(f"应用启动完成，运行在 {settings.host}:{settings.port}")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭应用...")
    close_database()
    logger.info("应用已关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    description="基于Geohash技术与RALM模型的精准LBS社交匹配系统",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加可信主机中间件
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加请求处理时间头部"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    start_time = time.time()
    
    # 记录请求信息
    logger.info(f"请求开始: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 记录响应信息
        logger.info(
            f"请求完成: {request.method} {request.url} - "
            f"状态码: {response.status_code} - "
            f"处理时间: {process_time:.4f}s"
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"请求异常: {request.method} {request.url} - "
            f"错误: {str(e)} - "
            f"处理时间: {process_time:.4f}s"
        )
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {request.method} {request.url} - {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "服务器内部错误",
            "error": str(exc) if settings.debug else "请联系管理员",
            "timestamp": time.time()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP异常: {request.method} {request.url} - {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


# 注册路由
app.include_router(users.router, prefix="/api/v1")
app.include_router(location.router, prefix="/api/v1")
app.include_router(privacy.router, prefix="/api/v1")
app.include_router(features.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")


@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "status": "success",
        "message": "欢迎使用精准LBS社交匹配系统",
        "service": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs" if settings.debug else "文档已禁用",
        "timestamp": time.time()
    }


@app.get("/api/v1/health", response_model=dict)
async def health_check():
    """健康检查接口"""
    try:
        # 检查数据库连接
        redis_ok = db_manager.test_redis_connection()
        mysql_ok = db_manager.test_mysql_connection()
        
        status_code = status.HTTP_200_OK
        if not redis_ok or not mysql_ok:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if redis_ok and mysql_ok else "unhealthy",
                "service": settings.app_name,
                "version": settings.app_version,
                "timestamp": time.time(),
                "checks": {
                    "redis": "ok" if redis_ok else "failed",
                    "mysql": "ok" if mysql_ok else "failed"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": settings.app_name,
                "version": settings.app_version,
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@app.get("/api/v1/info", response_model=dict)
async def get_system_info():
    """获取系统信息"""
    return {
        "status": "success",
        "data": {
            "service_name": settings.app_name,
            "version": settings.app_version,
            "environment": "development" if settings.debug else "production",
            "features": [
                "地理位置精准匹配",
                "隐私保护 (k-匿名)",
                "RALM智能特征匹配",
                "种子用户筛选",
                "模型训练与优化",
                "在线学习",
                "性能监控",
                "A/B测试",
                "高性能架构"
            ],
            "api_endpoints": {
                "users": "/api/v1/users",
                "location": "/api/v1/location",
                "privacy": "/api/v1/privacy",
                "features": "/api/v1/features",
                "training": "/api/v1/training",
                "health": "/api/v1/health",
                "docs": "/docs" if settings.debug else None
            },
            "timestamp": time.time()
        }
    }


if __name__ == "__main__":
    # 直接运行应用
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower()
    )
