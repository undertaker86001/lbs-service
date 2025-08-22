"""
精准LBS社交匹配系统主启动文件
"""
import sys
import os

# 添加app目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.main import app

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings
    
    print(f"""
    ================================
    精准LBS社交匹配系统
    ================================
    服务名称: {settings.app_name}
    版本: {settings.app_version}
    运行地址: http://{settings.host}:{settings.port}
    API文档: http://{settings.host}:{settings.port}/docs
    健康检查: http://{settings.host}:{settings.port}/api/v1/health
    ================================
    """)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower()
    )
