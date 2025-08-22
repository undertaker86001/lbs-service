"""
RALM模型服务 - 实时注意力机制下的Look-alike模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import json
import pickle
import os
from pathlib import Path

from app.services.feature_engineering import feature_engineering_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        output = self.output_proj(context)
        return output


class ProductiveAttention(nn.Module):
    """生产性注意力机制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 生产性注意力：使用外积计算注意力分数
        Q_expanded = Q.unsqueeze(-1)  # [batch, heads, seq_len, head_dim, 1]
        K_expanded = K.unsqueeze(-2)  # [batch, heads, seq_len, 1, head_dim]
        
        # 计算外积
        outer_product = torch.matmul(Q_expanded, K_expanded)  # [batch, heads, seq_len, seq_len, head_dim]
        
        # 计算注意力分数
        scores = torch.sum(outer_product, dim=-1)  # [batch, heads, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        output = self.output_proj(context)
        return output


class UserRepresentationTower(nn.Module):
    """用户表示学习塔"""
    
    def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [128, 256, 128], output_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 构建多层网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 注意力机制
        self.self_attention = SelfAttention(output_dim)
        self.productive_attention = ProductiveAttention(output_dim)
        
        # 空间转换
        self.spatial_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.PReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过基础网络
        features = self.network(x)
        
        # 添加序列维度用于注意力机制
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # [batch, 1, features]
        
        # 自注意力
        attended_features = self.self_attention(features)
        
        # 生产性注意力
        productive_features = self.productive_attention(attended_features)
        
        # 空间转换
        transformed_features = self.spatial_transform(productive_features)
        
        # 返回最终表示
        return transformed_features.squeeze(1)  # [batch, features]


class RALMModel(nn.Module):
    """RALM模型 - 实时注意力机制下的Look-alike模型"""
    
    def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [128, 256, 128], 
                 output_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # 种子用户塔
        self.seed_tower = UserRepresentationTower(input_dim, hidden_dims, output_dim)
        
        # 目标用户塔
        self.target_tower = UserRepresentationTower(input_dim, hidden_dims, output_dim)
        
        # 相似度计算层
        self.similarity_layer = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, seed_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            seed_features: 种子用户特征 [batch_size, input_dim]
            target_features: 目标用户特征 [batch_size, input_dim]
            
        Returns:
            相似度得分 [batch_size, num_classes]
        """
        # 通过各自的塔
        seed_representation = self.seed_tower(seed_features)
        target_representation = self.target_tower(target_features)
        
        # 拼接特征
        combined_features = torch.cat([seed_representation, target_representation], dim=1)
        
        # 计算相似度得分
        similarity_scores = self.similarity_layer(combined_features)
        
        return similarity_scores
    
    def get_user_representation(self, user_features: torch.Tensor, tower_type: str = "target") -> torch.Tensor:
        """
        获取用户表示向量
        
        Args:
            user_features: 用户特征 [batch_size, input_dim]
            tower_type: 塔类型 ("seed" 或 "target")
            
        Returns:
            用户表示向量 [batch_size, output_dim]
        """
        if tower_type == "seed":
            return self.seed_tower(user_features)
        else:
            return self.target_tower(user_features)


class RALMService:
    """RALM模型服务"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or "models/ralm_model.pth"
        
        # 模型参数
        self.input_dim = 64
        self.hidden_dims = [128, 256, 128]
        self.output_dim = 64
        self.num_classes = 2
        
        # 初始化模型
        self._init_model()
        
        # 特征工程服务
        self.feature_service = feature_engineering_service
        
        logger.info(f"RALM服务初始化完成，使用设备: {self.device}")
    
    def _init_model(self):
        """初始化模型"""
        try:
            self.model = RALMModel(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                num_classes=self.num_classes
            ).to(self.device)
            
            # 如果存在预训练模型，则加载
            if os.path.exists(self.model_path):
                self.load_model(self.model_path)
                logger.info(f"加载预训练模型: {self.model_path}")
            else:
                logger.info("使用随机初始化的模型")
                
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def train_model(self, training_data: List[Dict], validation_data: List[Dict] = None,
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """
        训练RALM模型
        
        Args:
            training_data: 训练数据
            validation_data: 验证数据
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            训练结果
        """
        try:
            logger.info(f"开始训练RALM模型，训练数据: {len(training_data)} 条")
            
            # 准备训练数据
            train_loader = self._prepare_data_loader(training_data, batch_size, shuffle=True)
            
            # 优化器和损失函数
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # 训练循环
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                epoch_loss = 0.0
                
                for batch in train_loader:
                    seed_features, target_features, labels = batch
                    
                    # 前向传播
                    outputs = self.model(seed_features, target_features)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # 验证阶段
                if validation_data and epoch % 10 == 0:
                    val_loss = self._validate_model(validation_data, criterion)
                    val_losses.append(val_loss)
                    
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
                
                # 早停检查
                if len(val_losses) > 5 and val_losses[-1] > val_losses[-5]:
                    logger.info("验证损失增加，提前停止训练")
                    break
            
            # 保存模型
            self.save_model()
            
            training_result = {
                "epochs_completed": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else 0,
                "final_val_loss": val_losses[-1] if val_losses else 0,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "model_saved": True
            }
            
            logger.info("RALM模型训练完成")
            return training_result
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def _prepare_data_loader(self, data: List[Dict], batch_size: int, shuffle: bool = False):
        """准备数据加载器"""
        try:
            # 提取特征和标签
            seed_features_list = []
            target_features_list = []
            labels_list = []
            
            for item in data:
                seed_features = item.get("seed_features", [0] * self.input_dim)
                target_features = item.get("target_features", [0] * self.input_dim)
                label = item.get("label", 0)
                
                # 确保特征维度正确
                if len(seed_features) != self.input_dim:
                    seed_features = seed_features[:self.input_dim] + [0] * (self.input_dim - len(seed_features))
                if len(target_features) != self.input_dim:
                    target_features = target_features[:self.input_dim] + [0] * (self.input_dim - len(target_features))
                
                seed_features_list.append(seed_features)
                target_features_list.append(target_features)
                labels_list.append(label)
            
            # 转换为张量
            seed_features_tensor = torch.FloatTensor(seed_features_list).to(self.device)
            target_features_tensor = torch.FloatTensor(target_features_list).to(self.device)
            labels_tensor = torch.LongTensor(labels_list).to(self.device)
            
            # 创建数据集
            dataset = torch.utils.data.TensorDataset(
                seed_features_tensor, target_features_tensor, labels_tensor
            )
            
            # 创建数据加载器
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )
            
            return data_loader
            
        except Exception as e:
            logger.error(f"准备数据加载器失败: {e}")
            raise
    
    def _validate_model(self, validation_data: List[Dict], criterion) -> float:
        """验证模型"""
        try:
            self.model.eval()
            val_loader = self._prepare_data_loader(validation_data, batch_size=32, shuffle=False)
            
            total_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    seed_features, target_features, labels = batch
                    outputs = self.model(seed_features, target_features)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
            
            return total_loss / len(val_loader)
            
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return float('inf')
    
    def predict_similarity(self, seed_user_id: int, target_user_id: int, 
                          seed_user_data: Dict, target_user_data: Dict) -> Dict:
        """
        预测用户相似度
        
        Args:
            seed_user_id: 种子用户ID
            target_user_id: 目标用户ID
            seed_user_data: 种子用户数据
            target_user_data: 目标用户数据
            
        Returns:
            相似度预测结果
        """
        try:
            # 构建用户特征
            seed_features = self.feature_service.build_user_features(seed_user_id, seed_user_data)
            target_features = self.feature_service.build_user_features(target_user_id, target_user_data)
            
            # 提取特征向量
            seed_feature_vector = seed_features["feature_vector"]
            target_feature_vector = target_features["feature_vector"]
            
            # 转换为张量
            seed_tensor = torch.FloatTensor([seed_feature_vector]).to(self.device)
            target_tensor = torch.FloatTensor([target_feature_vector]).to(self.device)
            
            # 模型推理
            self.model.eval()
            with torch.no_grad():
                similarity_scores = self.model(seed_tensor, target_tensor)
                probabilities = F.softmax(similarity_scores, dim=1)
                
                # 获取相似度得分
                similarity_score = probabilities[0][1].item()  # 正类概率
                
                # 获取用户表示向量
                seed_representation = self.model.get_user_representation(seed_tensor, "seed")
                target_representation = self.model.get_user_representation(target_tensor, "target")
                
                # 计算余弦相似度
                cosine_similarity = F.cosine_similarity(seed_representation, target_representation).item()
            
            result = {
                "seed_user_id": seed_user_id,
                "target_user_id": target_user_id,
                "similarity_score": similarity_score,
                "cosine_similarity": cosine_similarity,
                "prediction_confidence": max(probabilities[0]).item(),
                "timestamp": int(datetime.now().timestamp())
            }
            
            logger.info(f"用户相似度预测完成: {seed_user_id} vs {target_user_id}, 得分: {similarity_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"相似度预测失败: {e}")
            raise
    
    def batch_predict_similarity(self, seed_user_id: int, target_users_data: List[Dict]) -> List[Dict]:
        """
        批量预测用户相似度
        
        Args:
            seed_user_id: 种子用户ID
            target_users_data: 目标用户数据列表
            
        Returns:
            相似度预测结果列表
        """
        try:
            results = []
            
            for target_user_data in target_users_data:
                try:
                    target_user_id = target_user_data.get("user_id")
                    if target_user_id:
                        # 构建种子用户特征（这里需要从数据库获取）
                        seed_user_data = self._get_user_data(seed_user_id)
                        
                        if seed_user_data:
                            similarity_result = self.predict_similarity(
                                seed_user_id, target_user_id, seed_user_data, target_user_data
                            )
                            results.append(similarity_result)
                        else:
                            logger.warning(f"种子用户 {seed_user_id} 数据不存在")
                            
                except Exception as e:
                    logger.error(f"预测用户 {target_user_data.get('user_id', 'unknown')} 相似度失败: {e}")
                    continue
            
            # 按相似度排序
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"批量相似度预测完成，处理 {len(results)} 个用户")
            return results
            
        except Exception as e:
            logger.error(f"批量相似度预测失败: {e}")
            raise
    
    def _get_user_data(self, user_id: int) -> Optional[Dict]:
        """获取用户数据（这里应该从数据库获取）"""
        # 实际实现中应该从数据库获取用户数据
        # 这里返回模拟数据
        return {
            "current_location": {"latitude": 39.9042, "longitude": 116.4074},
            "interaction_data": {"daily_interactions": 50, "frequency": 0.8},
            "usage_patterns": {"daily_minutes": 120, "peak_hour": 20},
            "friendship_data": {"friend_count": 100, "mutual_friend_count": 50},
            "influence_metrics": {"score": 0.7, "engagement_rate": 0.6},
            "profile": {"age": 25, "gender": "male", "education": "bachelor"},
            "interests": {"categories": ["technology", "sports"], "intensity": 0.8},
            "active_hours": {"peak_hour": 20, "weekend_activity": 0.9}
        }
    
    def save_model(self, model_path: Optional[str] = None):
        """保存模型"""
        try:
            save_path = model_path or self.model_path
            
            # 创建目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存模型状态
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_dim': self.input_dim,
                    'hidden_dims': self.hidden_dims,
                    'output_dim': self.output_dim,
                    'num_classes': self.num_classes
                }
            })
            
            logger.info(f"模型已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            raise
    
    def load_model(self, model_path: Optional[str] = None):
        """加载模型"""
        try:
            load_path = model_path or self.model_path
            
            if not os.path.exists(load_path):
                logger.error(f"模型文件不存在: {load_path}")
                raise FileNotFoundError(f"模型文件不存在: {load_path}")
            
            # 加载模型状态
            state = torch.load(load_path, map_location=self.device)
            
            # 创建新的模型实例
            new_model = RALMModel(
                input_dim=state['model_config']['input_dim'],
                hidden_dims=state['model_config']['hidden_dims'],
                output_dim=state['model_config']['output_dim'],
                num_classes=state['model_config']['num_classes']
            ).to(self.device)
            
            # 加载模型状态
            new_model.load_state_dict(state['model_state_dict'])
            
            # 更新模型实例
            self.model = new_model
            
            logger.info(f"模型已加载: {load_path}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def batch_build_features(self, users_data: List[Dict]) -> List[Dict]:
        """批量构建用户特征"""
        return self.feature_service.batch_build_features(users_data)
    
    def __call__(self, seed_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """直接调用模型进行推理"""
        return self.model(seed_features, target_features)
    