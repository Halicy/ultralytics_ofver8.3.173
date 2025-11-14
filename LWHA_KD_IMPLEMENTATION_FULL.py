# 🚀 LWHA-KD 完整实现 (Innovation Point 4)
# LightWeight Hybrid Attention with Knowledge Distillation
#
# 本文件包含完整的实现代码，将被集成到 ultralytics/nn/modules/

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


# ========================================
# Part 1: 轻量化混合注意力 (LWHA)
# ========================================

class LinearAttention(nn.Module):
    """
    Linear Attention - 线性复杂度注意力机制

    设计思想:
        传统注意力: Attention(Q, K, V) = softmax(QK^T/√d) V  → O(N²)
        线性注意力: Attention(Q, K, V) = φ(Q) (φ(K)^T V)   → O(N)

        使用 kernel trick 将复杂度从 O(N²) 降低到 O(N)

    Args:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数
        qkv_bias (bool): Q, K, V 投影是否使用 bias
        feature_map_type (str): 特征映射函数类型 ('elu', 'relu', 'identity')

    References:
        - Transformers are RNNs (NeurIPS 2020)
        - Linear Transformers Are Secretly Fast Weight Programmers (ICML 2021)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        feature_map_type: str = 'elu',
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # 特征映射函数
        self.feature_map_type = feature_map_type
        if feature_map_type == 'elu':
            self.feature_map = lambda x: F.elu(x) + 1.0  # ELU + 1 (确保非负)
        elif feature_map_type == 'relu':
            self.feature_map = lambda x: F.relu(x)
        else:  # identity
            self.feature_map = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入特征 [B, N, C]

        Returns:
            torch.Tensor: 输出特征 [B, N, C]
        """
        B, N, C = x.shape

        # 1. 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]

        # 2. 应用特征映射
        q = self.feature_map(q)  # φ(Q)
        k = self.feature_map(k)  # φ(K)

        # 3. 线性注意力计算: φ(Q) (φ(K)^T V)
        # 先计算 KV = φ(K)^T V  [B, num_heads, head_dim, head_dim]
        kv = torch.einsum('bhnd,bhnc->bhdc', k, v)

        # 然后计算 Q @ KV  [B, num_heads, N, head_dim]
        out = torch.einsum('bhnd,bhdc->bhnc', q, kv)

        # 4. 归一化 (除以 sum(φ(K)))
        k_sum = k.sum(dim=2, keepdim=True)  # [B, num_heads, 1, head_dim]
        normalizer = torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(2)) + 1e-6  # [B, num_heads, N]
        out = out / normalizer.unsqueeze(-1)  # [B, num_heads, N, head_dim]

        # 5. 合并多头
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # 6. 输出投影
        out = self.proj(out)

        return out


class LocalEnhancement(nn.Module):
    """
    Local Enhancement - 局部特征增强模块

    设计思想:
        使用轻量级 Depthwise Convolution 捕获局部特征
        补充线性注意力的全局建模能力

    Args:
        dim (int): 输入特征维度
        kernel_size (int): 卷积核大小
        expand_ratio (float): 中间层扩展比例

    References:
        - MobileNetV2 (CVPR 2018): Depthwise separable convolution
        - ConvNeXt (CVPR 2022): Modern CNN design
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        expand_ratio: float = 2.0,
    ):
        super().__init__()

        hidden_dim = int(dim * expand_ratio)

        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,  # Depthwise
            bias=False
        )
        self.bn = nn.BatchNorm2d(dim)

        # Pointwise expansion
        self.pw_expand = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.act = nn.GELU()

        # Pointwise projection
        self.pw_project = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入特征 [B, N, C]
            h (int): 特征图高度
            w (int): 特征图宽度

        Returns:
            torch.Tensor: 输出特征 [B, N, C]
        """
        B, N, C = x.shape
        assert N == h * w, f"N={N} should equal h*w={h*w}"

        # Reshape to 2D: [B, N, C] -> [B, C, H, W]
        x_2d = x.permute(0, 2, 1).reshape(B, C, h, w)

        # Depthwise conv
        x_2d = self.bn(self.dwconv(x_2d))

        # Pointwise expand + activation
        x_2d = self.act(self.pw_expand(x_2d))

        # Pointwise project
        x_2d = self.pw_project(x_2d)

        # Reshape back: [B, C, H, W] -> [B, N, C]
        out = x_2d.reshape(B, C, N).permute(0, 2, 1)

        return out


class LWHybridAttention(nn.Module):
    """
    LightWeight Hybrid Attention - 轻量化混合注意力

    核心创新:
        1. 线性注意力 (O(N)) 替代传统注意力 (O(N²))
        2. 局部增强模块 (Depthwise Conv) 补充局部特征
        3. 全局-局部特征融合

    设计思想:
        - 高频分支: Depthwise Conv 捕获局部细节
        - 低频分支: Linear Attention 捕获全局上下文
        - 自适应融合: 学习全局-局部权重

    预期效果:
        - 参数: -25% (相比 AIFI)
        - 速度: +20% (O(N) vs O(N²))
        - mAP: -0.5% ~ +0.5% (轻微损失或持平)

    Args:
        c1 (int): 输入通道数
        cm (int): FFN 隐藏层维度
        num_heads (int): 注意力头数
        dropout (float): Dropout 比例
        act (nn.Module): 激活函数
        normalize_before (bool): 是否在注意力前归一化
        local_kernel_size (int): 局部增强卷积核大小
        global_local_ratio (float): 全局-局部特征融合比例

    Examples:
        >>> lwha = LWHybridAttention(c1=256, cm=2048, num_heads=8)
        >>> x = torch.randn(2, 256, 64, 64)  # [B, C, H, W]
        >>> out = lwha(x)
        >>> print(out.shape)  # torch.Size([2, 256, 64, 64])

    References:
        - Linear Attention (NeurIPS 2020)
        - HiLo Attention (CVPR 2023)
        - ConvNeXt (CVPR 2022)
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
        local_kernel_size: int = 3,
        global_local_ratio: float = 0.5,
    ):
        super().__init__()

        self.normalize_before = normalize_before
        self.global_local_ratio = global_local_ratio

        # 归一化层
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)

        # 全局分支: Linear Attention
        self.global_attn = LinearAttention(
            dim=c1,
            num_heads=num_heads,
            qkv_bias=True,
            feature_map_type='elu',
        )

        # 局部分支: Local Enhancement
        self.local_enhance = LocalEnhancement(
            dim=c1,
            kernel_size=local_kernel_size,
            expand_ratio=2.0,
        )

        # 全局-局部融合权重
        self.fusion_gate = nn.Sequential(
            nn.Linear(c1, c1 // 4),
            nn.GELU(),
            nn.Linear(c1 // 4, 2),
            nn.Softmax(dim=-1)
        )

        # FFN
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入特征 [B, C, H, W]

        Returns:
            torch.Tensor: 输出特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten: [B, C, H, W] -> [B, N, C]
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # ========== 混合注意力 ==========

        # 归一化
        if self.normalize_before:
            x_norm = self.norm1(x_flat)
        else:
            x_norm = x_flat

        # 1. 全局分支 (Linear Attention)
        global_out = self.global_attn(x_norm)  # [B, N, C]

        # 2. 局部分支 (Depthwise Conv)
        local_out = self.local_enhance(x_norm, H, W)  # [B, N, C]

        # 3. 自适应融合
        # 计算融合权重 (基于全局池化特征)
        pooled_feat = x_flat.mean(dim=1)  # [B, C]
        fusion_weights = self.fusion_gate(pooled_feat)  # [B, 2]

        # 加权融合
        attn_out = (
            fusion_weights[:, 0:1, None] * global_out +
            fusion_weights[:, 1:2, None] * local_out
        )  # [B, N, C]

        # Residual connection
        x_flat = x_flat + self.dropout1(attn_out)

        # 后归一化
        if not self.normalize_before:
            x_flat = self.norm1(x_flat)

        # ========== FFN ==========

        if self.normalize_before:
            x_norm = self.norm2(x_flat)
            ffn_out = self.fc2(self.dropout(self.act(self.fc1(x_norm))))
            x_flat = x_flat + self.dropout2(ffn_out)
        else:
            ffn_out = self.fc2(self.dropout(self.act(self.fc1(x_flat))))
            x_flat = self.norm2(x_flat + self.dropout2(ffn_out))

        # Reshape: [B, N, C] -> [B, C, H, W]
        out = x_flat.permute(0, 2, 1).reshape(B, C, H, W)

        return out

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """
        构建 2D sine-cosine 位置编码 (与 AIFI 兼容)

        Args:
            w (int): 特征图宽度
            h (int): 特征图高度
            embed_dim (int): 嵌入维度
            temperature (float): 温度系数

        Returns:
            torch.Tensor: 位置编码 [1, embed_dim, h*w]
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4"

        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


# ========================================
# Part 2: 知识蒸馏 (Knowledge Distillation)
# ========================================

class FeatureAdapter(nn.Module):
    """
    Feature Adapter - 特征对齐模块

    设计思想:
        教师模型和学生模型的特征维度可能不同，需要对齐
        使用 1x1 conv 进行维度转换

    Args:
        student_dim (int): 学生模型特征维度
        teacher_dim (int): 教师模型特征维度
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()

        if student_dim != teacher_dim:
            self.adapter = nn.Sequential(
                nn.Conv2d(student_dim, teacher_dim, 1, bias=False),
                nn.BatchNorm2d(teacher_dim)
            )
        else:
            self.adapter = nn.Identity()

    def forward(self, student_feat: torch.Tensor) -> torch.Tensor:
        """
        对齐学生特征到教师特征维度

        Args:
            student_feat (torch.Tensor): 学生特征 [B, C_s, H, W]

        Returns:
            torch.Tensor: 对齐后的特征 [B, C_t, H, W]
        """
        return self.adapter(student_feat)


class DistillationLoss(nn.Module):
    """
    Distillation Loss - 多层次知识蒸馏损失

    设计思想:
        1. Feature Distillation: L2 loss on neck features
        2. Query Distillation: L2 loss on decoder queries
        3. Response Distillation: KL divergence on logits

    Args:
        temperature (float): 蒸馏温度 (用于 response distillation)
        feature_loss_weight (float): 特征蒸馏损失权重
        query_loss_weight (float): 查询蒸馏损失权重
        response_loss_weight (float): 响应蒸馏损失权重

    References:
        - KD-DETR (CVPR 2024)
        - Focal-Global KD (CVPR 2021)
        - Knowledge Distillation (NeurIPS 2014)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        feature_loss_weight: float = 1.0,
        query_loss_weight: float = 0.5,
        response_loss_weight: float = 2.0,
    ):
        super().__init__()

        self.temperature = temperature
        self.feature_loss_weight = feature_loss_weight
        self.query_loss_weight = query_loss_weight
        self.response_loss_weight = response_loss_weight

    def feature_distillation_loss(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        特征蒸馏损失 (Neck features)

        Args:
            student_feats (list): 学生模型的 neck 特征
            teacher_feats (list): 教师模型的 neck 特征

        Returns:
            torch.Tensor: 特征蒸馏损失
        """
        loss = 0.0
        count = 0

        for s_feat, t_feat in zip(student_feats, teacher_feats):
            # 确保空间尺寸相同
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(
                    s_feat,
                    size=t_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # L2 loss
            loss += F.mse_loss(s_feat, t_feat, reduction='mean')
            count += 1

        return loss / count if count > 0 else loss

    def query_distillation_loss(
        self,
        student_queries: torch.Tensor,
        teacher_queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        查询蒸馏损失 (Decoder queries)

        Args:
            student_queries (torch.Tensor): 学生模型的查询特征 [B, nq, C]
            teacher_queries (torch.Tensor): 教师模型的查询特征 [B, nq, C]

        Returns:
            torch.Tensor: 查询蒸馏损失
        """
        # L2 loss on query embeddings
        return F.mse_loss(student_queries, teacher_queries, reduction='mean')

    def response_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        响应蒸馏损失 (Logits / Soft targets)

        Args:
            student_logits (torch.Tensor): 学生模型的 logits [B, nq, nc]
            teacher_logits (torch.Tensor): 教师模型的 logits [B, nq, nc]

        Returns:
            torch.Tensor: 响应蒸馏损失 (KL divergence)
        """
        # Soft targets with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return kl_loss

    def forward(
        self,
        student_feats: Optional[List[torch.Tensor]] = None,
        teacher_feats: Optional[List[torch.Tensor]] = None,
        student_queries: Optional[torch.Tensor] = None,
        teacher_queries: Optional[torch.Tensor] = None,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总蒸馏损失

        Args:
            student_feats: 学生模型 neck 特征
            teacher_feats: 教师模型 neck 特征
            student_queries: 学生模型查询
            teacher_queries: 教师模型查询
            student_logits: 学生模型 logits
            teacher_logits: 教师模型 logits

        Returns:
            dict: 各项蒸馏损失
        """
        losses = {}

        # 1. Feature distillation
        if student_feats is not None and teacher_feats is not None:
            feat_loss = self.feature_distillation_loss(student_feats, teacher_feats)
            losses['kd_feat_loss'] = feat_loss * self.feature_loss_weight

        # 2. Query distillation
        if student_queries is not None and teacher_queries is not None:
            query_loss = self.query_distillation_loss(student_queries, teacher_queries)
            losses['kd_query_loss'] = query_loss * self.query_loss_weight

        # 3. Response distillation
        if student_logits is not None and teacher_logits is not None:
            resp_loss = self.response_distillation_loss(student_logits, teacher_logits)
            losses['kd_resp_loss'] = resp_loss * self.response_loss_weight

        return losses


# ========================================
# 导出列表
# ========================================

__all__ = [
    'LinearAttention',
    'LocalEnhancement',
    'LWHybridAttention',
    'FeatureAdapter',
    'DistillationLoss',
]
