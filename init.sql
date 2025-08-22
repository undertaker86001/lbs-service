-- 精准LBS社交匹配系统数据库初始化脚本

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS lbs_social CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE lbs_social;

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户主键ID',
    user_id VARCHAR(50) NOT NULL UNIQUE COMMENT '用户唯一标识',
    username VARCHAR(50) NULL COMMENT '用户名',
    nickname VARCHAR(100) NULL COMMENT '昵称',
    avatar_url VARCHAR(500) NULL COMMENT '头像URL',
    gender VARCHAR(10) NULL COMMENT '性别',
    age INT NULL COMMENT '年龄',
    bio TEXT NULL COMMENT '个人简介',
    is_active BOOLEAN NOT NULL DEFAULT TRUE COMMENT '是否激活',
    privacy_level VARCHAR(20) NOT NULL DEFAULT 'medium' COMMENT '隐私级别',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    last_active_at DATETIME NULL COMMENT '最后活跃时间',
    
    INDEX idx_user_id (user_id),
    INDEX idx_username (username),
    INDEX idx_is_active (is_active),
    INDEX idx_last_active (last_active_at),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB COMMENT='用户基础信息表';

-- 创建用户位置表
CREATE TABLE IF NOT EXISTS user_locations (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '位置记录主键ID',
    user_id VARCHAR(50) NOT NULL COMMENT '用户ID',
    latitude DECIMAL(10, 8) NOT NULL COMMENT '纬度',
    longitude DECIMAL(11, 8) NOT NULL COMMENT '经度',
    geohash VARCHAR(20) NOT NULL COMMENT 'Geohash编码',
    anonymous_geohash VARCHAR(20) NULL COMMENT '匿名Geohash编码',
    accuracy FLOAT NULL COMMENT '定位精度',
    address VARCHAR(500) NULL COMMENT '地址',
    is_current BOOLEAN NOT NULL DEFAULT TRUE COMMENT '是否为当前位置',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    INDEX idx_user_id (user_id),
    INDEX idx_geohash (geohash),
    INDEX idx_anonymous_geohash (anonymous_geohash),
    INDEX idx_is_current (is_current),
    INDEX idx_created_at (created_at),
    INDEX idx_location (latitude, longitude),
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='用户位置信息表';

-- 创建用户行为表
CREATE TABLE IF NOT EXISTS user_behaviors (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '行为记录主键ID',
    user_id VARCHAR(50) NOT NULL COMMENT '用户ID',
    action_type VARCHAR(50) NOT NULL COMMENT '行为类型',
    action_data TEXT NULL COMMENT '行为数据JSON',
    session_id VARCHAR(100) NULL COMMENT '会话ID',
    device_type VARCHAR(50) NULL COMMENT '设备类型',
    ip_address VARCHAR(50) NULL COMMENT 'IP地址',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    INDEX idx_user_id (user_id),
    INDEX idx_action_type (action_type),
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='用户行为记录表';

-- 创建用户特征表（为后续RALM模型使用）
CREATE TABLE IF NOT EXISTS user_features (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '特征记录主键ID',
    user_id VARCHAR(50) NOT NULL COMMENT '用户ID',
    feature_type VARCHAR(50) NOT NULL COMMENT '特征类型',
    feature_data JSON NOT NULL COMMENT '特征数据JSON',
    feature_vector TEXT NULL COMMENT '特征向量',
    model_version VARCHAR(20) NULL COMMENT '模型版本',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    INDEX idx_user_id (user_id),
    INDEX idx_feature_type (feature_type),
    INDEX idx_model_version (model_version),
    INDEX idx_created_at (created_at),
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='用户特征数据表';

-- 创建匹配记录表
CREATE TABLE IF NOT EXISTS match_records (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '匹配记录主键ID',
    requester_user_id VARCHAR(50) NOT NULL COMMENT '请求用户ID',
    target_user_id VARCHAR(50) NOT NULL COMMENT '目标用户ID',
    distance FLOAT NOT NULL COMMENT '距离(米)',
    geohash_similarity FLOAT NULL COMMENT '地理位置相似度',
    feature_similarity FLOAT NULL COMMENT '特征相似度',
    comprehensive_score FLOAT NULL COMMENT '综合评分',
    match_type VARCHAR(20) NOT NULL DEFAULT 'nearby' COMMENT '匹配类型',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    
    INDEX idx_requester (requester_user_id),
    INDEX idx_target (target_user_id),
    INDEX idx_match_type (match_type),
    INDEX idx_comprehensive_score (comprehensive_score),
    INDEX idx_created_at (created_at),
    
    FOREIGN KEY (requester_user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (target_user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='匹配记录表';

-- 创建系统配置表
CREATE TABLE IF NOT EXISTS system_configs (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '配置主键ID',
    config_key VARCHAR(100) NOT NULL UNIQUE COMMENT '配置键',
    config_value TEXT NOT NULL COMMENT '配置值',
    config_type VARCHAR(20) NOT NULL DEFAULT 'string' COMMENT '配置类型',
    description VARCHAR(200) NULL COMMENT '配置描述',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    INDEX idx_config_key (config_key),
    INDEX idx_config_type (config_type)
) ENGINE=InnoDB COMMENT='系统配置表';

-- 插入默认系统配置
INSERT INTO system_configs (config_key, config_value, config_type, description) VALUES
('geohash_default_precision', '8', 'integer', '默认Geohash编码精度'),
('k_anonymity_default', '50', 'integer', '默认k-匿名参数'),
('max_search_radius', '10000', 'integer', '最大搜索半径(米)'),
('default_search_limit', '50', 'integer', '默认搜索结果数量限制'),
('location_cache_ttl', '3600', 'integer', '位置信息缓存过期时间(秒)'),
('feature_model_version', '1.0.0', 'string', '当前特征模型版本')
ON DUPLICATE KEY UPDATE 
    config_value = VALUES(config_value),
    updated_at = CURRENT_TIMESTAMP;

-- 创建一些测试数据（可选）
INSERT INTO users (user_id, username, nickname, gender, age, bio, privacy_level) VALUES
('test_user_001', 'testuser1', '测试用户1', 'male', 25, '这是一个测试用户', 'medium'),
('test_user_002', 'testuser2', '测试用户2', 'female', 23, '另一个测试用户', 'low'),
('test_user_003', 'testuser3', '测试用户3', 'male', 28, '第三个测试用户', 'high')
ON DUPLICATE KEY UPDATE 
    updated_at = CURRENT_TIMESTAMP;

-- 为测试用户添加一些位置数据（北京地区）
INSERT INTO user_locations (user_id, latitude, longitude, geohash, address, is_current) VALUES
('test_user_001', 39.9042, 116.4074, 'wx4fbxxh', '天安门广场', TRUE),
('test_user_002', 39.9000, 116.4000, 'wx4fbx2w', '故宫博物院', TRUE),
('test_user_003', 39.9100, 116.4100, 'wx4fbxxt', '王府井大街', TRUE)
ON DUPLICATE KEY UPDATE 
    updated_at = CURRENT_TIMESTAMP;

-- 创建数据库视图
CREATE OR REPLACE VIEW active_users AS
SELECT 
    u.id,
    u.user_id,
    u.username,
    u.nickname,
    u.avatar_url,
    u.privacy_level,
    u.last_active_at,
    ul.latitude,
    ul.longitude,
    ul.geohash,
    ul.created_at as location_updated_at
FROM users u
LEFT JOIN user_locations ul ON u.user_id = ul.user_id AND ul.is_current = TRUE
WHERE u.is_active = TRUE
AND u.last_active_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR);

-- 创建索引优化查询性能
CREATE INDEX idx_users_active_privacy ON users(is_active, privacy_level, last_active_at);
CREATE INDEX idx_locations_current_geohash ON user_locations(is_current, geohash, created_at);
CREATE INDEX idx_behaviors_user_type_time ON user_behaviors(user_id, action_type, created_at);

-- 设置数据库时区
SET time_zone = '+08:00';

COMMIT;
