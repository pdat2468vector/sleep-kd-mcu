# -*- coding: utf-8 -*-
"""
model_v11/blocks_v11.py
Các khối xây dựng cho kiến trúc v11:

  1. EogBranch       : CNN nhánh EOG (kernel lớn ~50-100) — bắt sóng cuộn mắt chậm (SEM)
  2. EegBranch       : CNN nhánh EEG (kernel nhỏ ~15-30)  — bắt alpha dropout sắc nét
  3. MultiBranchCNN  : Ghép EOG+EEG → Linear → Embedding (Teacher-size)
  4. LightMultiBranchCNN : Phiên bản nhẹ cho Student
  5. LinearAttention : Khối O(N) Kernel Attention (xấp xỉ Softmax không có ma trận NxN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────
# 1. EogBranch — Kernel lớn, bắt sóng lượn chậm của mắt (EOG)
# ─────────────────────────────────────────────────────────────

class EogBranch(nn.Module):
    """
    Xử lý kênh EOG (1 kênh, index [2:3] trong tensor 3 kênh).
    Dùng Conv1d với kernel lớn để bắt Slow Eye Movements (SEM) —
    tín hiệu tần số thấp, biên độ rộng, đặc trưng của giai đoạn N1.

    Input:  (B, 1, T)
    Output: (B, out_ch, T')
    """
    def __init__(self, out_ch: int = 64, kernel_sizes=(51, 101),
                 stride: int = 5, dropout: float = 0.1):
        super().__init__()
        # Multi-scale Inception: 2 nhánh kernel lớn song song
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(1, out_ch // 2, kernel_size=k, stride=stride,
                          padding=k // 2, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_ch // 2),
                nn.GELU(),
            ))
        # Fusion: (out_ch//2 * 2) → out_ch
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) → (B, out_ch, T')"""
        parts = []
        for branch in self.branches:
            out = branch(x)
            parts.append(out)
        # Cắt về cùng chiều dài (do padding lẻ có thể lệch 1)
        min_len = min(p.shape[-1] for p in parts)
        parts = [p[..., :min_len] for p in parts]
        x_cat = torch.cat(parts, dim=1)   # (B, out_ch, T')
        return self.fusion(x_cat)


# ─────────────────────────────────────────────────────────────
# 2. EegBranch — Kernel nhỏ, bắt rớt Alpha sắc nét (EEG)
# ─────────────────────────────────────────────────────────────

class EegBranch(nn.Module):
    """
    Xử lý 2 kênh EEG (index [0:2] trong tensor 3 kênh).
    Dùng Conv1d với kernel nhỏ để bắt alpha dropout đột ngột
    và các gợn sóng nhanh đặc trưng của vùng chẩm khi chuyển sang N1.

    Input:  (B, 2, T)
    Output: (B, out_ch, T')
    """
    def __init__(self, out_ch: int = 64, kernel_sizes=(15, 31),
                 stride: int = 5, dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(2, out_ch // 2, kernel_size=k, stride=stride,
                          padding=k // 2, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_ch // 2),
                nn.GELU(),
            ))
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, T) → (B, out_ch, T')"""
        parts = []
        for branch in self.branches:
            out = branch(x)
            parts.append(out)
        min_len = min(p.shape[-1] for p in parts)
        parts = [p[..., :min_len] for p in parts]
        x_cat = torch.cat(parts, dim=1)
        return self.fusion(x_cat)


# ─────────────────────────────────────────────────────────────
# 3. MultiBranchCNN — Teacher-size, embed_dim 256
# ─────────────────────────────────────────────────────────────

class MultiBranchCNN(nn.Module):
    """
    Tách 3 kênh → 2 nhánh xử lý riêng biệt:
      - Nhánh EOG: kênh index 2 (1 channel)
      - Nhánh EEG: kênh index 0,1 (2 channels)
    Ghép (Concat) output, qua Linear → vector embedding (embed_dim).

    Input:  (B, 3, T=3000)
    Output: (B, embed_dim)
    """
    def __init__(self, eog_out=64, eeg_out=64, embed_dim=256,
                 stride=5, dropout=0.1):
        super().__init__()
        self.eog_branch = EogBranch(out_ch=eog_out, stride=stride, dropout=dropout)
        self.eeg_branch = EegBranch(out_ch=eeg_out, stride=stride, dropout=dropout)

        # Pooling + Projection: (eog_out + eeg_out) → embed_dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(eog_out + eeg_out, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T) → (B, embed_dim)"""
        eeg = self.eeg_branch(x[:, :2, :])   # (B, eeg_out, T')
        eog = self.eog_branch(x[:, 2:3, :])  # (B, eog_out, T')

        # Cắt về cùng T' nếu lệch
        min_t = min(eeg.shape[-1], eog.shape[-1])
        eeg, eog = eeg[..., :min_t], eog[..., :min_t]

        merged = torch.cat([eeg, eog], dim=1)  # (B, eog_out+eeg_out, T')
        merged = self.pool(merged).squeeze(-1)  # (B, eog_out+eeg_out)
        return self.proj(merged)                # (B, embed_dim)


# ─────────────────────────────────────────────────────────────
# 4. LightMultiBranchCNN — Student-size, embed_dim 128
# ─────────────────────────────────────────────────────────────

class LightMultiBranchCNN(nn.Module):
    """
    Phiên bản CNN nhẹ cho Student.
    Giữ triết lý 2-nhánh, giảm số filters và channels.

    Input:  (B, 3, T=3000)
    Output: (B, embed_dim)
    """
    def __init__(self, eog_out=32, eeg_out=32, embed_dim=128,
                 stride=5, dropout=0.1):
        super().__init__()
        # Chỉ dùng 1 kernel mỗi nhánh (đơn giản hơn Teacher)
        self.eog_conv = nn.Sequential(
            nn.Conv1d(1, eog_out, kernel_size=51, stride=stride,
                      padding=25, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=eog_out),
            nn.GELU(),
        )
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(2, eeg_out, kernel_size=15, stride=stride,
                      padding=7, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=eeg_out),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(eog_out + eeg_out, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eeg = self.eeg_conv(x[:, :2, :])
        eog = self.eog_conv(x[:, 2:3, :])
        min_t = min(eeg.shape[-1], eog.shape[-1])
        eeg, eog = eeg[..., :min_t], eog[..., :min_t]
        merged = torch.cat([eeg, eog], dim=1)
        merged = self.pool(merged).squeeze(-1)
        return self.proj(merged)


# ─────────────────────────────────────────────────────────────
# 5. LinearAttention — O(N) Kernel Attention (không có NxN matrix)
# ─────────────────────────────────────────────────────────────

class LinearAttentionLayer(nn.Module):
    """
    Kernel-based Linear Attention theo Katharopoulos et al. (2020):
    "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"

    Xấp xỉ Attention(Q, K, V) = phi(Q) @ (phi(K)^T @ V)
    với phi(x) = elu(x) + 1 (đảm bảo dương).

    Độ phức tạp: O(N * d^2) thay vì O(N^2 * d).

    Args:
        embed_dim: Chiều vector embedding.
        num_heads: Số đầu Attention.
        dropout:   Tỷ lệ dropout.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) phải chia hết cho num_heads ({num_heads})"
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** 0.5

        self.qkv_proj   = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(embed_dim)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        """Feature map phi(x) = elu(x) + 1. Đảm bảo kết quả dương."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) — B=batch, N=seq_len, D=embed_dim
        returns: (B, N, D)
        """
        B, N, D = x.shape
        H, Hd   = self.num_heads, self.head_dim

        # Project Q, K, V
        qkv = self.qkv_proj(x)                  # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)           # 3x (B, N, D)

        # Reshape thành multi-head: (B, H, N, Hd)
        q = q.view(B, N, H, Hd).permute(0, 2, 1, 3)
        k = k.view(B, N, H, Hd).permute(0, 2, 1, 3)
        v = v.view(B, N, H, Hd).permute(0, 2, 1, 3)

        # Áp dụng feature map phi
        q = self._phi(q)                         # (B, H, N, Hd)
        k = self._phi(k)                         # (B, H, N, Hd)

        # Linear Attention:
        # S = phi(K)^T @ V → (B, H, Hd, Hd)   [Không phụ thuộc N!]
        # Z = phi(Q) @ sum(phi(K), dim=N)^T     [Normalizer]
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)  # (B, H, Hd, Hd)
        # Phân tử: phi(Q) @ kv → (B, H, N, Hd)
        qkv_out = torch.einsum('bhnd,bhdm->bhnm', q, kv)

        # Chuẩn hóa (chia cho tổng phi(K))
        k_sum = k.sum(dim=2, keepdim=True)           # (B, H, 1, Hd)
        denom = torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(2)) + 1e-6
        denom = denom.unsqueeze(-1)                  # (B, H, N, 1)
        attn_out = qkv_out / denom                   # (B, H, N, Hd)

        # Merge heads
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(self.dropout(attn_out))
        return self.norm(x + out)                    # Residual + Norm


class LinearAttentionBlock(nn.Module):
    """
    1 block hoàn chỉnh: LinearAttention + Feed-Forward Network.
    Giống TransformerEncoderLayer nhưng thay Self-Attention bằng Linear Attention.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4,
                 ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = LinearAttentionLayer(embed_dim, num_heads, dropout)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        x = self.attn(x)
        return self.norm2(x + self.ffn(x))
