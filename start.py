#!/usr/bin/env python
"""
快速启动脚本
"""
import os
import sys
import subprocess
import time

def check_dependencies():
    """检查依赖项"""
    print("正在检查依赖项...")
    
    try:
        import redis
        import mysql.connector
        import fastapi
        import uvicorn
        import pydantic
        print("✓ 所有Python依赖项已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖项: {e}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_services():
    """检查外部服务"""
    print("正在检查外部服务...")
    
    # 检查Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✓ Redis服务正常")
    except Exception as e:
        print(f"✗ Redis连接失败: {e}")
        print("请确保Redis服务正在运行")
        return False
    
    # 检查MySQL
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host='localhost',
            port=3306,
            user='root',
            password='root'
        )
        conn.close()
        print("✓ MySQL服务正常")
    except Exception as e:
        print(f"✗ MySQL连接失败: {e}")
        print("请确保MySQL服务正在运行，并检查连接配置")
        return False
    
    return True

def setup_environment():
    """设置环境"""
    print("正在设置环境...")
    
    # 检查环境配置文件
    if not os.path.exists('.env'):
        if os.path.exists('env.example'):
            print("创建 .env 文件...")
            with open('env.example', 'r', encoding='utf-8') as f:
                content = f.read()
            with open('.env', 'w', encoding='utf-8') as f:
                f.write(content)
            print("✓ .env 文件已创建，请根据需要修改配置")
        else:
            print("⚠ 未找到环境配置文件")
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    print("✓ 日志目录已创建")
    
    return True

def initialize_database():
    """初始化数据库"""
    print("正在初始化数据库...")
    
    try:
        import mysql.connector
        
        # 连接MySQL并创建数据库
        conn = mysql.connector.connect(
            host='localhost',
            port=3306,
            user='root',
            password='root'
        )
        
        cursor = conn.cursor()
        
        # 执行初始化脚本
        if os.path.exists('init.sql'):
            with open('init.sql', 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句并执行
            statements = sql_content.split(';')
            for statement in statements:
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
            
            conn.commit()
            print("✓ 数据库初始化完成")
        else:
            print("⚠ 未找到数据库初始化脚本")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"✗ 数据库初始化失败: {e}")
        return False

def start_application():
    """启动应用"""
    print("正在启动应用...")
    print("=" * 50)
    
    try:
        # 导入应用
        from app.main import app
        import uvicorn
        from app.core.config import settings
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     精准LBS社交匹配系统                          ║
║                        启动成功！                                ║
╠══════════════════════════════════════════════════════════════════╣
║ 服务名称: {settings.app_name:<50} ║
║ 版本号码: {settings.app_version:<50} ║
║ 运行地址: http://{settings.host}:{settings.port:<43} ║
║ API文档: http://{settings.host}:{settings.port}/docs{' ' * 36} ║
║ 健康检查: http://{settings.host}:{settings.port}/api/v1/health{' ' * 28} ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        
        # 启动服务器
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level=settings.log_level.lower()
        )
        
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    print("=" * 60)
    print("           精准LBS社交匹配系统启动器")
    print("=" * 60)
    
    # 步骤1: 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 步骤2: 检查外部服务
    if not check_services():
        print("\n提示: 请确保Redis和MySQL服务正在运行")
        response = input("是否继续启动? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # 步骤3: 设置环境
    if not setup_environment():
        sys.exit(1)
    
    # 步骤4: 初始化数据库
    if not initialize_database():
        print("\n数据库初始化失败，但可以继续启动应用")
        response = input("是否继续启动? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # 步骤5: 启动应用
    start_application()

if __name__ == "__main__":
    main()
