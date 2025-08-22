# 快速开始指南

## 📋 环境要求

- Python 3.9+
- Redis 6.0+
- MySQL 8.0+
- 操作系统：Windows/Linux/macOS

## 🚀 快速启动

### 方式一：使用启动脚本（推荐）

```bash
# 运行启动脚本，它会自动检查环境并启动服务
python start.py
```

启动脚本会自动：
- 检查Python依赖项
- 检查Redis和MySQL服务
- 创建环境配置文件
- 初始化数据库
- 启动应用服务

### 方式二：手动启动

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置环境变量**
```bash
# 复制环境配置文件
cp env.example .env

# 编辑配置文件，修改数据库连接等信息
# 根据你的实际环境调整配置
```

3. **启动外部服务**
```bash
# 启动Redis（Windows）
redis-server

# 启动MySQL（Windows）
net start mysql
```

4. **初始化数据库**
```bash
# 在MySQL中执行初始化脚本
mysql -u root -p < init.sql
```

5. **启动应用**
```bash
python main.py
```

## 🔧 配置说明

### 环境变量配置文件 (.env)

主要配置项说明：

```bash
# 应用配置
APP_NAME=精准LBS社交匹配系统
DEBUG=true                    # 开发模式
LOG_LEVEL=INFO               # 日志级别

# 服务器配置
HOST=0.0.0.0                 # 监听地址
PORT=8000                    # 监听端口

# Redis配置
REDIS_HOST=localhost         # Redis主机
REDIS_PORT=6379             # Redis端口
REDIS_PASSWORD=             # Redis密码（可选）

# MySQL配置
MYSQL_HOST=localhost         # MySQL主机
MYSQL_PORT=3306             # MySQL端口
MYSQL_USER=root             # MySQL用户名
MYSQL_PASSWORD=root         # MySQL密码
MYSQL_DATABASE=lbs_social   # 数据库名

# 隐私保护配置
K_ANONYMITY=50              # k-匿名参数
GEOHASH_MAX_LENGTH=12       # 最大Geohash长度
GEOHASH_MIN_LENGTH=5        # 最小Geohash长度
```

## 📡 API使用示例

### 1. 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

### 2. 用户注册

```bash
curl -X POST http://localhost:8000/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user001",
    "username": "testuser",
    "nickname": "测试用户",
    "gender": "male",
    "age": 25,
    "privacy_level": "medium"
  }'
```

### 3. 用户登录

```bash
curl -X POST "http://localhost:8000/api/v1/users/login?user_id=user001"
```

### 4. 更新位置

```bash
curl -X POST http://localhost:8000/api/v1/locations/update \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "user_id": "user001",
    "latitude": 39.9042,
    "longitude": 116.4074,
    "accuracy": 10.0,
    "address": "天安门广场"
  }'
```

### 5. 查找附近用户

```bash
curl -X POST http://localhost:8000/api/v1/locations/nearby \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 39.9042,
    "longitude": 116.4074,
    "radius": 1000,
    "limit": 10
  }'
```

## 🧪 运行测试

使用提供的测试脚本：

```bash
# 确保应用正在运行，然后执行测试
python test_api.py
```

测试脚本会自动：
- 注册测试用户
- 测试登录功能
- 更新位置信息
- 查找附近用户
- 验证各种API接口

## 🐳 Docker部署

### 使用docker-compose（推荐）

```bash
# 启动所有服务（应用、Redis、MySQL）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 单独构建镜像

```bash
# 构建应用镜像
docker build -t lbs-service .

# 运行容器
docker run -p 8000:8000 lbs-service
```

## 📊 监控和日志

### API文档
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 健康检查
- 健康状态: http://localhost:8000/api/v1/health
- 系统信息: http://localhost:8000/api/v1/info

### 日志文件
- 应用日志: `logs/app.log`
- 访问日志: 控制台输出

## 🔍 故障排除

### 常见问题

1. **Redis连接失败**
   - 确保Redis服务正在运行
   - 检查Redis配置和端口
   - 确认防火墙设置

2. **MySQL连接失败**
   - 确保MySQL服务正在运行
   - 检查用户名和密码
   - 确认数据库是否存在

3. **导入错误**
   - 确保所有依赖都已安装
   - 检查Python版本
   - 运行 `pip install -r requirements.txt`

4. **端口被占用**
   - 修改 `.env` 文件中的端口配置
   - 或者停止占用端口的其他服务

### 调试模式

启用调试模式获取更多信息：

```bash
# 修改 .env 文件
DEBUG=true
LOG_LEVEL=DEBUG
```

## 🔧 开发指南

### 项目结构

```
lbs-service/
├── app/                    # 应用主目录
│   ├── api/               # API路由
│   ├── core/              # 核心配置
│   ├── models/            # 数据模型
│   ├── services/          # 业务服务
│   └── utils/             # 工具函数
├── docs/                  # 文档目录
├── logs/                  # 日志目录
├── tests/                 # 测试目录
├── main.py               # 主启动文件
├── start.py              # 启动脚本
├── test_api.py           # API测试脚本
└── requirements.txt      # 依赖文件
```

### 添加新功能

1. 在 `app/models/` 中定义数据模型
2. 在 `app/services/` 中实现业务逻辑
3. 在 `app/api/` 中添加API接口
4. 在 `app/main.py` 中注册路由

### 代码规范

- 遵循PEP 8 Python代码规范
- 使用Type Hints进行类型注解
- 编写完整的文档字符串
- 为新功能添加测试用例

## 📞 获取帮助

如果遇到问题：

1. 查看应用日志 (`logs/app.log`)
2. 检查健康检查接口
3. 运行测试脚本验证功能
4. 查看详细的错误信息

## 🎯 下一步

第一阶段完成后，你可以：

1. 实现隐私保护模块（G-Casper算法）
2. 训练和集成RALM模型
3. 添加更多的特征工程
4. 优化性能和扩展性
5. 添加前端界面

恭喜！你已经成功完成了第一阶段的开发 🎉
