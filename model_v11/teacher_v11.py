# -*- coding: utf-8 -*-
"""
model_v11/teacher_v11.py
Teacher Model cho Knowledge Distillation v11.

Kiến trúc:
  1. MultiBranchCNN: Tách EOG (kernel lớn) / EEG (kernel nhỏ) → embed_dim=256
  2. TransformerEncoder: Standard O(N²) Self-Attention để học context seq dài
  3. Classifier: LayerNorm → Linear → GELU → Dropout → Linear(5)

Input:  (B, L, 3, 3000) — L = seq_len (e.g. 5), 3 kênh, 3000 samples/epoch
Output: dict {'logits': (B, 5), 'cnn_feat': (B, embed_dim),
               'ctx_feat': (B, embed_dim)}
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_v11.blocks_v11 import MultiBranchCNN


class TeacherSleepNetV11(nn.Module):
    """
    Teacher Model v11.
    ~400K–600K parameters (tùy embed_dim và Transformer config).

    Args:
        embed_dim:    Chiều embedding (output của MultiBranchCNN). Mặc định 256.
        num_heads:    Số đầu Attention của Transformer. Mặc định 8.
        num_layers:   Số lớp TransformerEncoder. Mặc định 3.
        ff_mult:      Hệ số FFN (dim * ff_mult). Mặc định 4.
        num_classes:  Số nhãn phân loại (5 giai đoạn ngủ). Mặc định 5.
        seq_len:      Độ dài chuỗi epoch. Mặc định 5.
        dropout:      Tỷ lệ dropout. Mặc định 0.1.
    """

    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 ff_mult: int = 4,
                 num_classes: int = 5,
                 seq_len: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len   = seq_len

        # --- Khối 1: Multi-branch CNN (EOG/EEG) ---
        self.feature_extractor = MultiBranchCNN(
            eog_out=embed_dim // 2,
            eeg_out=embed_dim // 2,
            embed_dim=embed_dim,
            stride=5,
            dropout=dropout,
        )

        # --- Positional Embedding (Learnable) ---
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

        # --- Khối 2: Standard Transformer Encoder (O(N²)) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,   # (B, N, D) convention
            norm_first=True,    # Pre-LayerNorm (Pre-LN) — ổn định hơn
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # --- Khối 3: Classifier ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
            'logits'   : (B, num_classes),
            'cnn_feat' : (B, embed_dim)  — output của CNN, dùng cho Feature Matching,
            'ctx_feat' : (B, embed_dim)  — output của Transformer, dùng cho Context Matching
        }
        """
        B, L, C, T = x.shape

        # ① Trích đặc trưng từng epoch song song
        x_flat  = x.reshape(B * L, C, T)          # (B*L, 3, T)
        feats   = self.feature_extractor(x_flat)   # (B*L, embed_dim)
        feats   = feats.view(B, L, self.embed_dim) # (B, L, embed_dim)

        # ② Lưu CNN feature của epoch trung tâm (center epoch) để distill
        center  = L // 2
        cnn_feat = feats[:, center, :]             # (B, embed_dim)

        # ③ Positional Embedding
        pos_ids = torch.arange(L, device=x.device)
        feats   = feats + self.pos_emb(pos_ids)    # (B, L, embed_dim)

        # ④ Transformer context
        ctx_seq = self.transformer(feats)           # (B, L, embed_dim)
        ctx_feat = ctx_seq[:, center, :]            # (B, embed_dim) — lấy epoch trung tâm

        # ⑤ Classifier
        logits = self.classifier(ctx_feat)          # (B, num_classes)

        return {
            'logits'  : logits,
            'cnn_feat': cnn_feat,
            'ctx_feat': ctx_feat,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = TeacherSleepNetV11()
    x = torch.randn(4, 5, 3, 3000)
    out = model(x)
    print(f'TeacherSleepNetV11 | Params: {model.count_parameters():,}')
    print(f'Logits: {out["logits"].shape} | CNN feat: {out["cnn_feat"].shape} | CTX feat: {out["ctx_feat"].shape}')
