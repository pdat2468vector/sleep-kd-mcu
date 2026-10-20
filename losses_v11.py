# -*- coding: utf-8 -*-
"""
losses_v11.py
Loss functions cho V11 Teacher-Student Knowledge Distillation.

Gồm:
  1. FocalLoss            : Cho Teacher — phạt nặng khi sai N1/Wake
  2. KDLossV11            : Knowledge Distillation đầy đủ 3 thành phần:
                            - Hard Label Loss (CE)
                            - Soft Label Loss (KL Divergence, Temperature T)
                            - Feature Matching Loss (MSE qua Adaptor Layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. Focal Loss — Teacher Training
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss cho phân loại đa lớp mất cân bằng.
    Từ "Focal Loss for Dense Object Detection" (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma:         Hệ số tập trung (focusing param). Cao hơn → phạt mạnh hơn mẫu dễ.
                       Mặc định 2.0 (giá trị gốc của paper).
        class_weights: Trọng số từng lớp (Tensor shape [num_classes]).
                       Dùng để phạt nặng hơn lớp N1 (index 1).
                       Nếu None, sẽ tự tính class weight từ tần suất.
        label_smoothing: Làm mịn nhãn để tránh overfit. Mặc định 0.05.
    """
    def __init__(self, gamma: float = 2.0,
                 class_weights: torch.Tensor = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.class_weights   = class_weights  # Sẽ được đưa lên device khi dùng
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes) — raw logits
            targets: (B,)             — ground truth labels (integer)
        Returns:
            scalar loss
        """
        # Tính log_softmax + softmax
        log_prob = F.log_softmax(logits, dim=-1)    # (B, num_classes)
        prob     = log_prob.exp()                    # (B, num_classes)

        # Lấy xác suất đúng nhãn cho mỗi mẫu
        # p_t = prob của nhãn đúng
        p_t      = prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Focal weight = (1 - p_t)^gamma
        focal_w  = (1.0 - p_t) ** self.gamma                        # (B,)

        # Tính cross-entropy per-sample (có label smoothing)
        num_cls  = logits.size(1)
        smooth   = self.label_smoothing
        # CE với label smoothing: CE = (1-smooth)*CE_hard + smooth*CE_uniform
        ce_hard  = -log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        ce_unif  = -log_prob.mean(dim=-1)                                 # (B,)
        ce       = (1 - smooth) * ce_hard + smooth * ce_unif             # (B,)

        # Áp class weight nếu có
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)[targets]             # (B,)
            ce = ce * w

        loss = focal_w * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────────
# 2. KDLossV11 — Knowledge Distillation đầy đủ
# ─────────────────────────────────────────────────────────────

class KDLossV11(nn.Module):
    """
    Knowledge Distillation Loss V11 — kết hợp 3 thành phần:

      L_total = w_ce  * L_CE(student_logits, hard_labels)       [Hard Label]
              + w_kl  * T² * KL(student_soft || teacher_soft)   [Soft Label]
              + w_mse * (MSE(cnn_proj, teacher_cnn)             [Feature Match]
                        + MSE(ctx_proj, teacher_ctx))

    Với Adaptor Layer đã được tích hợp sẵn trong Student Model,
    MSE được tính giữa:
      - student.cnn_feat_proj ↔ teacher.cnn_feat
      - student.ctx_feat_proj ↔ teacher.ctx_feat

    Args:
        temperature: Nhiệt độ làm mịn Soft Labels. Nên cao (4-8) khi dùng Focal Loss.
        w_ce:        Trọng số Hard Label Loss. Mặc định 0.3.
        w_kl:        Trọng số Soft KL Loss.   Mặc định 0.5.
        w_mse:       Trọng số Feature Match.  Mặc định 0.2 (tùy chọn, có thể tắt).
        label_smoothing: Label smoothing cho CE. Mặc định 0.05.
    """
    def __init__(self,
                 temperature: float = 5.0,
                 w_ce:        float = 0.3,
                 w_kl:        float = 0.5,
                 w_mse:       float = 0.2,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.T      = temperature
        self.w_ce   = w_ce
        self.w_kl   = w_kl
        self.w_mse  = w_mse
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self,
                student_out:  dict,
                teacher_out:  dict,
                targets:      torch.Tensor) -> dict:
        """
        Args:
            student_out : dict từ StudentSleepNetV11.forward()
                          {logits, cnn_feat_proj, ctx_feat_proj}
            teacher_out : dict từ TeacherSleepNetV11.forward() — DETACHED
                          {logits, cnn_feat, ctx_feat}
            targets     : (B,) ground truth labels
        Returns:
            dict: {loss, ce_loss, kl_loss, mse_loss}
        """
        s_logits = student_out['logits']           # (B, C)
        t_logits = teacher_out['logits'].detach()  # Detach — không train Teacher
        t_cnn    = teacher_out['cnn_feat'].detach()
        t_ctx    = teacher_out['ctx_feat'].detach()

        # ── 1. Hard Label Loss ─────────────────────────────────
        ce_loss = self.ce_loss(s_logits, targets)

        # ── 2. Soft Label KL Loss với Temperature ─────────────
        # T cao → làm phẳng phân phối sharp của Teacher (do Focal Loss gây ra)
        s_soft = F.log_softmax(s_logits / self.T, dim=-1)
        t_soft = F.softmax(t_logits / self.T, dim=-1)
        kl_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.T ** 2)

        # ── 3. Feature Matching MSE Loss (qua Adaptor Layer) ──
        # student cnn_feat_proj đã được project về teacher_embed_dim
        mse_cnn = F.mse_loss(student_out['cnn_feat_proj'], t_cnn)
        mse_ctx = F.mse_loss(student_out['ctx_feat_proj'], t_ctx)
        mse_loss = (mse_cnn + mse_ctx) / 2.0

        # ── Tổng hợp ──────────────────────────────────────────
        total = (self.w_ce  * ce_loss +
                 self.w_kl  * kl_loss +
                 self.w_mse * mse_loss)

        return {
            'loss'    : total,
            'ce_loss' : ce_loss.item(),
            'kl_loss' : kl_loss.item(),
            'mse_loss': mse_loss.item(),
        }
