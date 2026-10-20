# -*- coding: utf-8 -*-
"""
model_v11: Teacher-Student Architecture for Sleep Staging
  - Teacher: Multi-branch CNN (EOG/EEG) + Standard Transformer
  - Student:  Light Multi-branch CNN + Linear Attention (O(N))
"""
from .teacher_v11 import TeacherSleepNetV11
from .student_v11 import StudentSleepNetV11
