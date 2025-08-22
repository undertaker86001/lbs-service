# 精准LBS社交匹配系统

## 项目简介

这是一个基于Geohash技术与RALM模型的精准LBS社交匹配系统，旨在为用户提供高质量的位置社交匹配服务。系统通过结合地理位置信息和用户行为特征，实现精准的社交匹配，同时保护用户隐私。

## 核心特性

### 🌍 地理位置精准匹配
- 使用Geohash技术实现高效的位置编码和查询
- 动态精度调整，根据用户密度自动优化匹配精度
- 支持2-19米的高精度地理位置匹配

### 🔒 隐私保护
- 实现G-Casper算法的k-匿名位置保护
- 用户位置信息匿名化处理
- 支持用户自定义隐私级别设置

### 🤖 智能特征匹配
- 基于RALM模型的用户特征学习
- 多维度特征提取（地理位置、行为习惯、社交网络）
- 实时注意力机制，动态调整特征权重

### ⚡ 高性能架构
- Redis GeoHash实现毫秒级位置查询
- 分层架构设计，支持水平扩展
- 异步处理和缓存机制

## 系统架构

```
┌─────────────────────────────────────────┐
│           应用服务层 (API)               │
│         FastAPI + 用户界面              │
├─────────────────────────────────────────┤
│           模型计算层 (RALM)             │
│         PyTorch + 机器学习              │
├─────────────────────────────────────────┤
│           数据处理层 (ETL)              │
│         特征工程 + 数据清洗             │
├─────────────────────────────────────────┤
│           数据采集层 (SDK)              │
│        移动端SDK + 位置采集             │
└─────────────────────────────────────────┘
```

## 技术栈

### 后端技术
- **Python 3.9+**: 主要开发语言
- **FastAPI**: 高性能Web框架
- **Redis**: 位置数据存储和缓存
- **MySQL**: 用户数据和特征存储
- **PyTorch**: 深度学习框架

### 前端技术
- **React Native**: 跨平台移动应用
- **TypeScript**: 类型安全的JavaScript

### 部署和运维
- **Docker**: 容器化部署
- **Kubernetes**: 容器编排
- **Prometheus + Grafana**: 监控和可视化

## 功能模块

### 1. 用户位置管理
- 实时位置采集和更新
- 位置历史轨迹记录
- 位置隐私保护设置

### 2. 附近用户发现
- 基于地理位置的用户筛选
- 可配置的搜索半径
- 实时位置更新

### 3. 智能匹配推荐
- 多维度用户特征分析
- 基于RALM模型的相似度计算
- 个性化推荐算法

### 4. 隐私保护
- k-匿名位置保护
- 用户数据加密
- 隐私设置管理

## 快速开始

### 环境要求
- Python 3.9+
- Redis 6.0+
- MySQL 8.0+
- Docker (可选)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/lbs-service.git
cd lbs-service
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑.env文件，配置数据库连接等信息
```

4. **启动服务**
```bash
python main.py
```

### Docker部署
```bash
# 构建镜像
docker build -t lbs-service .

# 运行容器
docker run -p 8000:8000 lbs-service
```

## API接口

### 用户位置管理

#### 更新用户位置
```http
POST /api/v1/users/location
Content-Type: application/json

{
    "user_id": "user123",
    "lat": 39.9042,
    "lon": 116.4074,
    "timestamp": 1640995200
}
```

#### 查找附近用户
```http
POST /api/v1/users/nearby
Content-Type: application/json

{
    "lat": 39.9042,
    "lon": 116.4074,
    "radius": 1000,
    "limit": 50
}
```

### 响应格式
```json
{
    "status": "success",
    "data": {
        "users": [
            {
                "user_id": "user456",
                "distance": 150.5,
                "similarity_score": 0.85,
                "comprehensive_score": 0.78
            }
        ],
        "total_count": 25,
        "search_radius": 1000
    }
}
```

## 配置说明

### 环境变量配置
```bash
# 数据库配置
REDIS_HOST=localhost
REDIS_PORT=6379
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=lbs_social

# 隐私保护配置
K_ANONYMITY=50
GEOHASH_MAX_LENGTH=12
GEOHASH_MIN_LENGTH=5

# 性能配置
CACHE_TTL=3600
MAX_CONCURRENT_REQUESTS=100
```

### 模型参数配置
```python
# 特征维度配置
FEATURE_DIM = 128
MAX_SEQUENCE_LENGTH = 100

# 注意力机制配置
ATTENTION_HEADS = 8
ATTENTION_DROPOUT = 0.1

# 训练参数
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
```

## 性能指标

### 技术性能
- **地理位置精度**: 2-19米（可配置）
- **响应时间**: < 1秒
- **并发处理能力**: > 1000 QPS
- **位置查询效率**: O(logN)

### 业务指标
- **匹配准确率**: > 80%
- **用户满意度**: > 85%
- **系统可用性**: > 99.9%
- **隐私保护级别**: k-匿名（k=50）

## 开发指南

### 代码结构
```
lbs-service/
├── app/
│   ├── api/           # API接口
│   ├── core/          # 核心配置
│   ├── models/        # 数据模型
│   ├── services/      # 业务服务
│   └── utils/         # 工具函数
├── tests/             # 测试用例
├── docs/              # 文档
├── requirements.txt    # 依赖包
└── README.md          # 项目说明
```

### 开发规范
- 遵循PEP 8 Python代码规范
- 使用Type Hints进行类型注解
- 编写完整的文档字符串
- 单元测试覆盖率 > 80%

### 测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_geohash_service.py

# 生成测试覆盖率报告
pytest --cov=app tests/
```

## 部署说明

### 生产环境部署
1. **环境准备**
   - 配置生产环境数据库
   - 设置SSL证书
   - 配置反向代理

2. **容器化部署**
   ```bash
   # 使用docker-compose
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **监控配置**
   - 配置Prometheus监控
   - 设置Grafana仪表板
   - 配置告警规则

### 性能优化
- 启用Redis集群
- 配置MySQL读写分离
- 使用CDN加速静态资源
- 实现负载均衡

## 常见问题

### Q: 如何提高匹配精度？
A: 可以通过调整Geohash编码长度来提高精度，但会降低查询效率。建议根据实际需求平衡精度和性能。

### Q: 如何处理冷启动问题？
A: 系统为新用户提供基于规则的初始推荐，随着用户行为数据积累，逐步优化个性化推荐。

### Q: 如何保护用户隐私？
A: 系统使用G-Casper算法实现k-匿名保护，用户位置信息经过匿名化处理后存储，确保隐私安全。

## 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系我们

- 项目主页: [https://github.com/your-username/lbs-service](https://github.com/your-username/lbs-service)
- 问题反馈: [Issues](https://github.com/your-username/lbs-service/issues)
- 邮箱: your-email@example.com

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 实现基础LBS匹配功能
- 集成RALM模型
- 隐私保护机制

---

**注意**: 本项目仅供学习和研究使用，在生产环境中使用前请进行充分的安全测试和性能评估。
