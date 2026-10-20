# -*- coding: utf-8 -*-
"""
model_v11/student_v11.py
Student Model siêu nhẹ cho Edge AI — Knowledge Distillation v11.

Kiến trúc:
  1. LightMultiBranchCNN: Tách EOG/EEG → embed_dim=128 (nhẹ hơn Teacher)
  2. LinearAttentionBlock: O(N) thay thế Transformer — không có ma trận NxN
  3. AdaptorLayer: Projection để khớp feature space với Teacher (cho MSE Loss)
  4. Classifier: LayerNorm → Linear(5)

Input:  (B, L, 3, 3000)
Output: dict {'logits': (B, 5), 'cnn_feat_proj': (B, teacher_embed_dim),
               'ctx_feat_proj': (B, teacher_embed_dim)}
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_v11.blocks_v11 import LightMultiBranchCNN, LinearAttentionBlock


class StudentSleepNetV11(nn.Module):
    """
    Student Model v11 — Tối ưu cho Edge AI.
    ~50K–100K parameters.

    Args:
        embed_dim:          Chiều embedding Student. Mặc định 128.
        teacher_embed_dim:  Chiều embedding Teacher (dùng cho Adaptor). Mặc định 256.
        num_heads:          Số đầu Linear Attention. Mặc định 4.
        num_layers:         Số lớp Linear Attention. Mặc định 2.
        num_classes:        Số nhãn. Mặc định 5.
        seq_len:            Độ dài chuỗi. Mặc định 5.
        dropout:            Dropout ratio. Mặc định 0.1.
    """

    def __init__(self,
                 embed_dim: int = 128,
                 teacher_embed_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 num_classes: int = 5,
                 seq_len: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim         = embed_dim
        self.teacher_embed_dim = teacher_embed_dim
        self.seq_len           = seq_len

        # --- Khối 1: Light Multi-branch CNN ---
        self.feature_extractor = LightMultiBranchCNN(
            eog_out=embed_dim // 2,
            eeg_out=embed_dim // 2,
            embed_dim=embed_dim,
            stride=5,
            dropout=dropout,
        )

        # --- Positional Embedding ---
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        # --- Khối 2: Linear Attention O(N) (thay Transformer) ---
        self.linear_attn_layers = nn.ModuleList([
            LinearAttentionBlock(embed_dim, num_heads, ff_mult=2, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ctx_norm = nn.LayerNorm(embed_dim)

        # --- Khối 3: Adaptor Layer (Student → Teacher space) ---
        # Dùng cho Feature Matching Loss (MSE giữa Student và Teacher)
        # Tránh ép khớp trực tiếp 2 không gian khác nhau
        self.cnn_adaptor = nn.Linear(embed_dim, teacher_embed_dim)
        self.ctx_adaptor = nn.Linear(embed_dim, teacher_embed_dim)

        # --- Khối 4: Classifier ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, L, 3, T)
        returns: dict {
            'logits'         : (B, num_classes),
            'cnn_feat_proj'  : (B, teacher_embed_dim) — để MSE với teacher cnn_feat,
            'ctx_feat_proj'  : (B, teacher_embed_dim) — để MSE với teacher ctx_feat
        }
        """
        B, L, C, T = x.shape

        # ① CNN Feature extraction
        x_flat = x.reshape(B * L, C, T)           # (B*L, 3, T)
        feats  = self.feature_extractor(x_flat)    # (B*L, embed_dim)
        feats  = feats.view(B, L, self.embed_dim)  # (B, L, embed_dim)

        # ② CNN feat (epoch trung tâm) để feature matching
        center   = L // 2
        cnn_feat = feats[:, center, :]             # (B, embed_dim)

        # ③ Positional Embedding
        pos_ids = torch.arange(L, device=x.device)
        feats   = feats + self.pos_emb(pos_ids)    # (B, L, embed_dim)

        # ④ Linear Attention O(N) — xử lý toàn bộ sequence
        ctx = feats
        for layer in self.linear_attn_layers:
            ctx = layer(ctx)                       # (B, L, embed_dim)
        ctx_seq = self.ctx_norm(ctx)
        ctx_feat = ctx_seq[:, center, :]           # (B, embed_dim)

        # ⑤ Adaptor projection (để dùng cho Feature Matching Loss)
        cnn_feat_proj = self.cnn_adaptor(cnn_feat) # (B, teacher_embed_dim)
        ctx_feat_proj = self.ctx_adaptor(ctx_feat) # (B, teacher_embed_dim)

        # ⑥ Classifier
        logits = self.classifier(ctx_feat)         # (B, num_classes)

        return {
            'logits'        : logits,
            'cnn_feat_proj' : cnn_feat_proj,
            'ctx_feat_proj' : ctx_feat_proj,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = StudentSleepNetV11()
    x = torch.randn(4, 5, 3, 3000)
    out = model(x)
    print(f'StudentSleepNetV11 | Params: {model.count_parameters():,}')
    print(f'Logits:       {out["logits"].shape}')
    print(f'CNN feat proj:{out["cnn_feat_proj"].shape}')
    print(f'CTX feat proj:{out["ctx_feat_proj"].shape}')
