# -*- coding: utf-8 -*-
"""
model_v11/student_v2.py
Student V2 — Siêu nhỏ cho Edge AI / MCU / Smartwatch

Kiến trúc:
  1. DWSep Multi-branch CNN (Depthwise Separable Conv):
     - EOG nhánh: DWSep kernel=51 → pool → embed
     - EEG nhánh: DWSep kernel=15 → pool → embed
     - Concat + Linear → embed_dim=64

  2. Mini GRU (hidden_size=32, num_layers=1):
     - Quét tuần tự qua 21 epochs
     - O(1) RAM khi chạy inference real-time
     - Lấy hidden state cuối → Classifier

  3. Adaptor Layer (CNN embed→256, GRU hidden→256):
     - Project features lên không gian 256-dim của Student V1 projected features
     - Dùng cho Feature Matching MSE Loss khi distill từ Student V1

Ước tính params: ~40K
Sau INT8 Quantize: ~10KB file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Depthwise Separable Conv 1D block
# ─────────────────────────────────────────────────────────────

class DWSepConv1d(nn.Module):
    """
    Depthwise Separable Convolution 1D.
    Thay thế Conv1d thường để giảm params 5–10x.
    Bước 1 (Depthwise): mỗi channel xử lý riêng biệt.
    Bước 2 (Pointwise): trộn thông tin các channel lại.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 stride: int = 1, groups_dw: int = None):
        super().__init__()
        groups_dw = groups_dw or in_ch
        pad = kernel // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=kernel, stride=stride,
                            padding=pad, groups=groups_dw, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        num_groups = min(4, out_ch)
        while out_ch % num_groups != 0:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


# ─────────────────────────────────────────────────────────────
# DWSep Multi-branch CNN (siêu nhẹ)
# ─────────────────────────────────────────────────────────────

class DWSepMultiBranchCNN(nn.Module):
    """
    2 nhánh Depthwise Separable Conv:
      - Nhánh EOG (1ch, kernel=51): bắt Slow Eye Movements
      - Nhánh EEG (2ch, kernel=15): bắt alpha dropout

    Input:  (B, 3, T=3000)
    Output: (B, embed_dim)
    """
    def __init__(self, eog_ch: int = 16, eeg_ch: int = 16,
                 embed_dim: int = 64, stride: int = 5, dropout: float = 0.1):
        super().__init__()
        # EOG branch: 1 → eog_ch
        self.eog_branch = nn.Sequential(
            DWSepConv1d(1, eog_ch, kernel=51, stride=stride),
            DWSepConv1d(eog_ch, eog_ch, kernel=5, stride=2),
        )
        # EEG branch: 2 → eeg_ch
        self.eeg_branch = nn.Sequential(
            DWSepConv1d(2, eeg_ch, kernel=15, stride=stride),
            DWSepConv1d(eeg_ch, eeg_ch, kernel=5, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(eog_ch + eeg_ch, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T) → (B, embed_dim)"""
        eeg = self.eeg_branch(x[:, :2, :])   # (B, eeg_ch, T')
        eog = self.eog_branch(x[:, 2:3, :])  # (B, eog_ch, T')
        min_t = min(eeg.shape[-1], eog.shape[-1])
        merged = torch.cat([eeg[..., :min_t], eog[..., :min_t]], dim=1)
        merged = self.pool(merged).squeeze(-1)  # (B, eog_ch+eeg_ch)
        return self.proj(merged)                # (B, embed_dim)


# ─────────────────────────────────────────────────────────────
# Student V2: DWSep CNN + Mini GRU
# ─────────────────────────────────────────────────────────────

class StudentSleepNetV2(nn.Module):
    """
    Student V2 — Edge AI tối thượng.

    Args:
        embed_dim:      Chiều CNN embedding. Mặc định 64.
        gru_hidden:     Số neurons GRU. Mặc định 32.
        gru_layers:     Số layer GRU. Mặc định 1.
        v1_embed_dim:   Chiều Student V1 (Teacher của V2), để Adaptor. Mặc định 128.
        num_classes:    Số nhãn. Mặc định 5.
        seq_len:        Chiều dài chuỗi. Mặc định 21.
        dropout:        Dropout. Mặc định 0.1.
    """
    def __init__(self,
                 embed_dim:   int = 64,
                 gru_hidden:  int = 32,
                 gru_layers:  int = 1,
                 v1_embed_dim: int = 128,
                 num_classes: int = 5,
                 seq_len:     int = 21,
                 dropout:     float = 0.1):
        super().__init__()
        self.embed_dim   = embed_dim
        self.gru_hidden  = gru_hidden
        self.seq_len     = seq_len
        self.v1_embed_dim = v1_embed_dim

        # Khối 1: DWSep CNN
        self.cnn = DWSepMultiBranchCNN(
            eog_ch=embed_dim // 4,
            eeg_ch=embed_dim // 4 * 3,  # EEG nhận nhiều filter hơn (2 kênh)
            embed_dim=embed_dim,
            stride=5,
            dropout=dropout,
        )

        # Khối 2: Mini GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_norm = nn.LayerNorm(gru_hidden)
        self.gru_dropout = nn.Dropout(dropout)

        # Adaptor: GRU hidden → Student V1 space (cho Feature Matching Loss)
        # cnn_feat_proj: embed_dim → v1_embed_dim
        self.cnn_adaptor = nn.Linear(embed_dim, v1_embed_dim)
        # ctx_feat_proj: gru_hidden → v1_embed_dim
        self.ctx_adaptor = nn.Linear(gru_hidden, v1_embed_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Linear(gru_hidden, num_classes),
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
        returns: {
            'logits'        : (B, num_classes),
            'cnn_feat_proj' : (B, v1_embed_dim)  → khớp với student_v1['cnn_feat'],
            'ctx_feat_proj' : (B, v1_embed_dim)  → khớp với student_v1['ctx_feat']
        }
        """
        B, L, C, T = x.shape

        # ① CNN Feature extraction — xử lý song song toàn bộ sequence
        x_flat = x.reshape(B * L, C, T)        # (B*L, 3, T)
        feats  = self.cnn(x_flat)               # (B*L, embed_dim)
        feats  = feats.view(B, L, self.embed_dim)  # (B, L, embed_dim)

        # ② CNN feat của epoch trung tâm (để Feature Matching)
        center   = L // 2
        cnn_feat = feats[:, center, :]           # (B, embed_dim)

        # ③ GRU quét qua chuỗi sequence
        gru_out, hn = self.gru(feats)            # gru_out: (B, L, gru_hidden)
        # Lấy hidden state của epoch trung tâm
        ctx_feat = self.gru_norm(
            self.gru_dropout(gru_out[:, center, :])
        )                                        # (B, gru_hidden)

        # ④ Adaptor projection → Student V1 space
        cnn_feat_proj = self.cnn_adaptor(cnn_feat)   # (B, v1_embed_dim)
        ctx_feat_proj = self.ctx_adaptor(ctx_feat)   # (B, v1_embed_dim)

        # ⑤ Classifier
        logits = self.classifier(ctx_feat)           # (B, num_classes)

        return {
            'logits'        : logits,
            'cnn_feat_proj' : cnn_feat_proj,
            'ctx_feat_proj' : ctx_feat_proj,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = StudentSleepNetV2()
    x = torch.randn(4, 21, 3, 3000)
    out = model(x)
    print(f'StudentSleepNetV2 | Params: {model.count_parameters():,}')
    print(f'  logits:          {out["logits"].shape}')
    print(f'  cnn_feat_proj:   {out["cnn_feat_proj"].shape}')
    print(f'  ctx_feat_proj:   {out["ctx_feat_proj"].shape}')
