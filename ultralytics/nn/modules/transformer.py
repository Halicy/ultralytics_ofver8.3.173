# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer modules."""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "TransformerEncoderLayer",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ASDA",
    "LinearAttention",
    "LocalEnhancement",
    "LWHybridAttention",
    "FeatureAdapter",
    "DistillationLoss",
)


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    This class implements a standard transformer encoder layer with multi-head attention and feedforward network,
    supporting both pre-normalization and post-normalization configurations.

    Attributes:
        ma (nn.MultiheadAttention): Multi-head attention module.
        fc1 (nn.Linear): First linear layer in the feedforward network.
        fc2 (nn.Linear): Second linear layer in the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout (nn.Dropout): Dropout layer for the feedforward network.
        dropout1 (nn.Dropout): Dropout layer after attention.
        dropout2 (nn.Dropout): Dropout layer after feedforward network.
        act (nn.Module): Activation function.
        normalize_before (bool): Whether to apply normalization before attention and feedforward.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        Initialize the TransformerEncoderLayer with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass with post-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass with pre-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward propagate the input through the encoder module.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after transformer encoder layer.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """
    AIFI transformer layer for 2D data with positional embeddings.

    This class extends TransformerEncoderLayer to work with 2D feature maps by adding 2D sine-cosine positional
    embeddings and handling the spatial dimensions appropriately.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        Initialize the AIFI instance with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AIFI transformer layer.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W].

        Returns:
            (torch.Tensor): Output tensor with shape [B, C, H, W].
        """
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """
        Build 2D sine-cosine position embedding.

        Args:
            w (int): Width of the feature map.
            h (int): Height of the feature map.
            embed_dim (int): Embedding dimension.
            temperature (float): Temperature for the sine/cosine functions.

        Returns:
            (torch.Tensor): Position embedding with shape [1, embed_dim, h*w].
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c: int, num_heads: int):
        """
        Initialize a self-attention mechanism using linear transformations and multi-head attention.

        Args:
            c (int): Input and output channel dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a transformer block to the input x and return the output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after transformer layer.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """
    Vision Transformer block based on https://arxiv.org/abs/2010.11929.

    This class implements a complete transformer block with optional convolution layer for channel adjustment,
    learnable position embedding, and multiple transformer layers.

    Attributes:
        conv (Conv, optional): Convolution layer if input and output channels differ.
        linear (nn.Linear): Learnable position embedding.
        tr (nn.Sequential): Sequential container of transformer layers.
        c2 (int): Output channel dimension.
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """
        Initialize a Transformer module with position embedding and specified number of heads and layers.

        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagate the input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor with shape [b, c1, w, h].

        Returns:
            (torch.Tensor): Output tensor with shape [b, c2, w, h].
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """A single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """
        Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function.

        Args:
            embedding_dim (int): Input and output dimension.
            mlp_dim (int): Hidden dimension.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP block.
        """
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """
    A simple multi-layer perceptron (also called FFN).

    This class implements a configurable MLP with multiple linear layers, activation functions, and optional
    sigmoid output activation.

    Attributes:
        num_layers (int): Number of layers in the MLP.
        layers (nn.ModuleList): List of linear layers.
        sigmoid (bool): Whether to apply sigmoid to the output.
        act (nn.Module): Activation function.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act=nn.ReLU, sigmoid: bool = False
    ):
        """
        Initialize the MLP with specified input, hidden, output dimensions and number of layers.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of layers.
            act (nn.Module): Activation function.
            sigmoid (bool): Whether to apply sigmoid to the output.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid = sigmoid
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the entire MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP.
        """
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    This class implements layer normalization for 2D feature maps, normalizing across the channel dimension
    while preserving spatial dimensions.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter.
        bias (nn.Parameter): Learnable bias parameter.
        eps (float): Small constant for numerical stability.

    References:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        Initialize LayerNorm2d with the given parameters.

        Args:
            num_channels (int): Number of channels in the input.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass for 2D layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Normalized output tensor.
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    This module implements multiscale deformable attention that can attend to features at multiple scales
    with learnable sampling locations and attention weights.

    Attributes:
        im2col_step (int): Step size for im2col operations.
        d_model (int): Model dimension.
        n_levels (int): Number of feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points per attention head per feature level.
        sampling_offsets (nn.Linear): Linear layer for generating sampling offsets.
        attention_weights (nn.Linear): Linear layer for generating attention weights.
        value_proj (nn.Linear): Linear layer for projecting values.
        output_proj (nn.Linear): Linear layer for projecting output.

    References:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """
        Initialize MSDeformAttn with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_levels (int): Number of feature levels.
            n_heads (int): Number of attention heads.
            n_points (int): Number of sampling points per attention head per feature level.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: List,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass for multiscale deformable attention.

        Args:
            query (torch.Tensor): Query tensor with shape [bs, query_length, C].
            refer_bbox (torch.Tensor): Reference bounding boxes with shape [bs, query_length, n_levels, 2],
                range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area.
            value (torch.Tensor): Value tensor with shape [bs, value_length, C].
            value_shapes (list): List with shape [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})].
            value_mask (torch.Tensor, optional): Mask tensor with shape [bs, value_length], True for non-padding
                elements, False for padding elements.

        Returns:
            (torch.Tensor): Output tensor with shape [bs, Length_{query}, C].

        References:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    This class implements a single decoder layer with self-attention, cross-attention using multiscale deformable
    attention, and a feedforward network.

    Attributes:
        self_attn (nn.MultiheadAttention): Self-attention module.
        dropout1 (nn.Dropout): Dropout after self-attention.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn (MSDeformAttn): Cross-attention module.
        dropout2 (nn.Dropout): Dropout after cross-attention.
        norm2 (nn.LayerNorm): Layer normalization after cross-attention.
        linear1 (nn.Linear): First linear layer in the feedforward network.
        act (nn.Module): Activation function.
        dropout3 (nn.Dropout): Dropout in the feedforward network.
        linear2 (nn.Linear): Second linear layer in the feedforward network.
        dropout4 (nn.Dropout): Dropout after the feedforward network.
        norm3 (nn.LayerNorm): Layer normalization after the feedforward network.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """
        Initialize the DeformableTransformerDecoderLayer with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            d_ffn (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            n_levels (int): Number of feature levels.
            n_points (int): Number of sampling points.
        """
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the Feed-Forward Network part of the layer.

        Args:
            tgt (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after FFN.
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: List,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass through the entire decoder layer.

        Args:
            embed (torch.Tensor): Input embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Feature maps.
            shapes (list): Feature shapes.
            padding_mask (torch.Tensor, optional): Padding mask.
            attn_mask (torch.Tensor, optional): Attention mask.
            query_pos (torch.Tensor, optional): Query position embeddings.

        Returns:
            (torch.Tensor): Output tensor after decoder layer.
        """
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    Deformable Transformer Decoder based on PaddleDetection implementation.

    This class implements a complete deformable transformer decoder with multiple decoder layers and prediction
    heads for bounding box regression and classification.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        num_layers (int): Number of decoder layers.
        hidden_dim (int): Hidden dimension.
        eval_idx (int): Index of the layer to use during evaluation.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """
        Initialize the DeformableTransformerDecoder with the given parameters.

        Args:
            hidden_dim (int): Hidden dimension.
            decoder_layer (nn.Module): Decoder layer module.
            num_layers (int): Number of decoder layers.
            eval_idx (int): Index of the layer to use during evaluation.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # decoder embeddings
        refer_bbox: torch.Tensor,  # anchor
        feats: torch.Tensor,  # image features
        shapes: List,  # feature shapes
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_query_embed: bool = False,  # ✅ NEW: For HCP-DETR prototype learning
    ):
        """
        Perform the forward pass through the entire decoder.

        Args:
            embed (torch.Tensor): Decoder embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Image features.
            shapes (list): Feature shapes.
            bbox_head (nn.Module): Bounding box prediction head.
            score_head (nn.Module): Score prediction head.
            pos_mlp (nn.Module): Position MLP.
            attn_mask (torch.Tensor, optional): Attention mask.
            padding_mask (torch.Tensor, optional): Padding mask.
            return_query_embed (bool): If True, also return final query embeddings.
                Required for HCP-DETR prototype learning. Default: False (backward compatible).

        Returns:
            If return_query_embed is False (default):
                dec_bboxes (torch.Tensor): Decoded bounding boxes.
                dec_cls (torch.Tensor): Decoded classification scores.
            If return_query_embed is True:
                dec_bboxes (torch.Tensor): Decoded bounding boxes.
                dec_cls (torch.Tensor): Decoded classification scores.
                query_embed (torch.Tensor): Final query embeddings from last decoder layer.
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        # ✅ FIX Bug #1: Return query embeddings if requested (for HCP-DETR)
        # 'output' now contains the final query embeddings from the last decoder layer
        # Shape: [bs, num_queries, hidden_dim]
        # This is the CORRECT feature space for prototype learning (not encoder features!)
        if return_query_embed:
            return torch.stack(dec_bboxes), torch.stack(dec_cls), output
        else:
            # Backward compatible: only return boxes and scores
            return torch.stack(dec_bboxes), torch.stack(dec_cls)


class ASDA(MSDeformAttn):
    """
    Aspect-ratio Sensitive Deformable Attention for elongated object detection.

    This class extends MSDeformAttn to adaptively adjust sampling patterns based on object aspect ratios,
    making it particularly suitable for detecting elongated objects like cucumbers in agricultural scenes.

    Key innovations:
        1. Aspect-ratio prediction network: Predicts the aspect ratio of each query
        2. Elliptical sampling pattern: Adapts sampling points based on aspect ratio
        3. Adaptive offset scaling: Scales offsets differently along major and minor axes

    Attributes:
        aspect_ratio_predictor (nn.Sequential): MLP for predicting aspect ratios
        ellipse_bias (nn.Parameter): Learnable elliptical sampling bias
        aspect_ratio_range (tuple): Min and max aspect ratios (default: 1.0 to 10.0)

    Examples:
        >>> asda = ASDA(d_model=256, n_levels=4, n_heads=8, n_points=4)
        >>> query = torch.randn(2, 300, 256)
        >>> refer_bbox = torch.randn(2, 300, 4, 4)
        >>> value = torch.randn(2, 1000, 256)
        >>> value_shapes = [(20, 20), (10, 10), (5, 5), (3, 3)]
        >>> output = asda(query, refer_bbox, value, value_shapes)

    References:
        - Deformable DETR (ICLR 2021): https://arxiv.org/abs/2010.04159
        - D-LKA (Bearing-DETR, 2024): Large Kernel Deformable Attention
        - DAT (NeurIPS 2022): https://arxiv.org/abs/2201.00520
    """

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        aspect_ratio_range: tuple = (1.0, 10.0),
    ):
        """
        Initialize ASDA with aspect-ratio sensitive sampling.

        Args:
            d_model (int): Model dimension.
            n_levels (int): Number of feature levels.
            n_heads (int): Number of attention heads.
            n_points (int): Number of sampling points per head per level.
            aspect_ratio_range (tuple): (min_ratio, max_ratio) for aspect ratio prediction.
                Default (1.0, 10.0) means square to 10:1 elongated objects.
        """
        super().__init__(d_model, n_levels, n_heads, n_points)

        self.aspect_ratio_range = aspect_ratio_range

        # Aspect ratio prediction network (Innovation 1)
        # Input: query feature (d_model) -> Output: aspect ratio scalar
        self.aspect_ratio_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),  # Output [0, 1], will be mapped to aspect_ratio_range
        )

        # Elliptical sampling bias (Innovation 2)
        # Shape: [n_heads, n_levels, n_points, 2]
        # This adds a learnable elliptical pattern to sampling offsets
        self.ellipse_bias = nn.Parameter(torch.zeros(n_heads, n_levels, n_points, 2))

        # Re-initialize with elliptical pattern
        self._reset_aspect_parameters()

    def _reset_aspect_parameters(self):
        """Initialize elliptical sampling pattern for elongated objects."""
        # Standard MSDeformAttn initialization is already done in super().__init__()
        # Now we initialize the elliptical bias

        with torch.no_grad():
            for i in range(self.n_points):
                # Distribute points in elliptical pattern
                angle = 2 * math.pi * i / self.n_points

                # Major axis (horizontal) - 2x larger for elongated objects
                # Minor axis (vertical) - 0.5x smaller
                self.ellipse_bias[:, :, i, 0] = math.cos(angle) * 2.0  # x-direction
                self.ellipse_bias[:, :, i, 1] = math.sin(angle) * 0.5  # y-direction

        # Initialize aspect ratio predictor
        for m in self.aspect_ratio_predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: List,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with aspect-ratio sensitive deformable attention.

        Args:
            query (torch.Tensor): [bs, query_length, C] Query features
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2 or 4] Reference boxes
                If last dim = 2: center points (cx, cy)
                If last dim = 4: boxes (cx, cy, w, h)
            value (torch.Tensor): [bs, value_length, C] Value features
            value_shapes (List[tuple]): [(H_0, W_0), (H_1, W_1), ...] Feature map shapes
            value_mask (torch.Tensor, optional): [bs, value_length] Padding mask

        Returns:
            (torch.Tensor): [bs, query_length, C] Output features after attention

        Note:
            The aspect ratio is predicted per query and used to adaptively scale
            sampling offsets, creating an elliptical sampling pattern for elongated objects.
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        # === Innovation 3: Predict aspect ratio for each query ===
        aspect_ratios = self.aspect_ratio_predictor(query)  # [bs, len_q, 1]

        # Map from [0, 1] to [min_ratio, max_ratio]
        min_ratio, max_ratio = self.aspect_ratio_range
        aspect_ratios = aspect_ratios * (max_ratio - min_ratio) + min_ratio  # [bs, len_q, 1]

        # === Standard deformable attention computation ===
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)

        # Compute base sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # === Innovation 4: Apply aspect-ratio adaptive scaling ===
        # Create scaling factors: [aspect_ratio, 1/aspect_ratio] for [x, y]
        aspect_scale = torch.stack(
            [
                aspect_ratios.squeeze(-1),  # x-direction: scale by aspect_ratio
                1.0 / (aspect_ratios.squeeze(-1) + 1e-6),  # y-direction: scale by 1/aspect_ratio
            ],
            dim=-1,
        )  # [bs, len_q, 2]

        # Expand dimensions for broadcasting: [bs, len_q, 1, 1, 1, 2]
        aspect_scale = aspect_scale[:, :, None, None, None, :]

        # Apply adaptive scaling to offsets
        sampling_offsets = sampling_offsets * aspect_scale

        # === Innovation 2: Add elliptical bias ===
        # Expand ellipse_bias: [n_heads, n_levels, n_points, 2] -> [1, 1, n_heads, n_levels, n_points, 2]
        ellipse_bias_expanded = self.ellipse_bias.unsqueeze(0).unsqueeze(0)
        sampling_offsets = sampling_offsets + ellipse_bias_expanded

        # Compute attention weights
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points
        )

        # === Compute sampling locations ===
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            # Reference points are centers (cx, cy)
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            # Reference points are boxes (cx, cy, w, h)
            # Use box width and height to scale offsets
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of refer_bbox must be 2 or 4, but got {num_points}.")

        # === Apply multi-scale deformable attention ===
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)

        return self.output_proj(output)


# ========================================
# LWHA-KD: LightWeight Hybrid Attention with Knowledge Distillation
# Innovation Point 4
# ========================================


class LinearAttention(nn.Module):
    """
    Linear Attention - 线性复杂度注意力机制 O(N).

    设计思想:
        传统注意力: Attention(Q, K, V) = softmax(QK^T/√d) V  → O(N²)
        线性注意力: Attention(Q, K, V) = φ(Q) (φ(K)^T V)   → O(N)

        使用 kernel trick 将复杂度从 O(N²) 降低到 O(N)

    Args:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数
        qkv_bias (bool): Q, K, V 投影是否使用 bias
        feature_map_type (str): 特征映射函数类型 ('elu', 'relu', 'identity')

    Examples:
        >>> linear_attn = LinearAttention(dim=256, num_heads=8)
        >>> x = torch.randn(2, 100, 256)  # [B, N, C]
        >>> out = linear_attn(x)
        >>> print(out.shape)  # torch.Size([2, 100, 256])

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
        """Initialize Linear Attention with specified parameters."""
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
        Forward pass with linear complexity O(N).

        Args:
            x (torch.Tensor): Input features [B, N, C]

        Returns:
            (torch.Tensor): Output features [B, N, C]
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
    Local Enhancement - 局部特征增强模块.

    设计思想:
        使用轻量级 Depthwise Convolution 捕获局部特征
        补充线性注意力的全局建模能力

    Args:
        dim (int): 输入特征维度
        kernel_size (int): 卷积核大小
        expand_ratio (float): 中间层扩展比例

    Examples:
        >>> local_enh = LocalEnhancement(dim=256, kernel_size=3)
        >>> x = torch.randn(2, 100, 256)  # [B, N, C]
        >>> out = local_enh(x, h=10, w=10)
        >>> print(out.shape)  # torch.Size([2, 100, 256])

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
        """Initialize Local Enhancement module."""
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
        Forward pass with local enhancement.

        Args:
            x (torch.Tensor): Input features [B, N, C]
            h (int): Feature map height
            w (int): Feature map width

        Returns:
            (torch.Tensor): Enhanced features [B, N, C]
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
    LightWeight Hybrid Attention - 轻量化混合注意力.

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
        """Initialize LightWeight Hybrid Attention module."""
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
        Forward pass with hybrid attention.

        Args:
            x (torch.Tensor): Input features [B, C, H, W]

        Returns:
            (torch.Tensor): Output features [B, C, H, W]
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
        """Build 2D sine-cosine position embedding (compatible with AIFI)."""
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


class FeatureAdapter(nn.Module):
    """
    Feature Adapter - 特征对齐模块.

    设计思想:
        教师模型和学生模型的特征维度可能不同，需要对齐
        使用 1x1 conv 进行维度转换

    Args:
        student_dim (int): 学生模型特征维度
        teacher_dim (int): 教师模型特征维度

    Examples:
        >>> adapter = FeatureAdapter(student_dim=256, teacher_dim=512)
        >>> student_feat = torch.randn(2, 256, 64, 64)
        >>> aligned_feat = adapter(student_feat)
        >>> print(aligned_feat.shape)  # torch.Size([2, 512, 64, 64])
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        """Initialize Feature Adapter for dimension alignment."""
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
        Align student features to teacher feature dimension.

        Args:
            student_feat (torch.Tensor): Student features [B, C_s, H, W]

        Returns:
            (torch.Tensor): Aligned features [B, C_t, H, W]
        """
        return self.adapter(student_feat)


class DistillationLoss(nn.Module):
    """
    Distillation Loss - 多层次知识蒸馏损失.

    设计思想:
        1. Feature Distillation: L2 loss on neck features
        2. Query Distillation: L2 loss on decoder queries
        3. Response Distillation: KL divergence on logits

    Args:
        temperature (float): 蒸馏温度 (用于 response distillation)
        feature_loss_weight (float): 特征蒸馏损失权重
        query_loss_weight (float): 查询蒸馏损失权重
        response_loss_weight (float): 响应蒸馏损失权重

    Examples:
        >>> kd_loss = DistillationLoss(temperature=4.0)
        >>> student_feats = [torch.randn(2, 256, 64, 64)]
        >>> teacher_feats = [torch.randn(2, 256, 64, 64)]
        >>> losses = kd_loss(student_feats=student_feats, teacher_feats=teacher_feats)
        >>> print(losses.keys())  # dict_keys(['kd_feat_loss'])

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
        """Initialize Distillation Loss with specified parameters."""
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
        Compute feature distillation loss on neck features.

        Args:
            student_feats (list): Student neck features
            teacher_feats (list): Teacher neck features

        Returns:
            (torch.Tensor): Feature distillation loss
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
        Compute query distillation loss on decoder queries.

        Args:
            student_queries (torch.Tensor): Student query features [B, nq, C]
            teacher_queries (torch.Tensor): Teacher query features [B, nq, C]

        Returns:
            (torch.Tensor): Query distillation loss
        """
        # L2 loss on query embeddings
        return F.mse_loss(student_queries, teacher_queries, reduction='mean')

    def response_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute response distillation loss using KL divergence.

        Args:
            student_logits (torch.Tensor): Student logits [B, nq, nc]
            teacher_logits (torch.Tensor): Teacher logits [B, nq, nc]

        Returns:
            (torch.Tensor): Response distillation loss
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
    ) -> dict:
        """
        Compute total distillation loss.

        Args:
            student_feats: Student neck features
            teacher_feats: Teacher neck features
            student_queries: Student decoder queries
            teacher_queries: Teacher decoder queries
            student_logits: Student classification logits
            teacher_logits: Teacher classification logits

        Returns:
            (dict): Dictionary of distillation losses
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
