"""Manuscript names of the training variants (shared by make_results_tables.py and make_supplementary.py)."""
NAME = {
    'T_focal': 'Teacher-Focal', 'T_ce': 'Teacher-CE',
    'A_scratch': 'Assistant-Scratch', 'A_kd_Tfocal': 'Assistant-KD (Teacher-Focal)',
    'A_kd_Tce': 'Assistant-KD (Teacher-CE)',
    'E_scratch': 'Edge-Scratch', 'E_direct_Tfocal': 'Edge-Direct (Teacher-Focal)',
    'E_staged_Tfocal': 'Edge-Staged (Assistant-KD, Teacher-Focal)',
    'E_from_Ascratch': 'Edge-KD (Assistant-Scratch)', 'E_direct_Tce': 'Edge-Direct (Teacher-CE)',
    'E_staged_Tce': 'Edge-Staged (Assistant-KD, Teacher-CE)'}
GROUPS = [('Teacher', ['T_focal', 'T_ce']), ('Linear Assistant', ['A_scratch', 'A_kd_Tfocal', 'A_kd_Tce']),
          ('Edge Student', ['E_scratch', 'E_direct_Tfocal', 'E_staged_Tfocal', 'E_from_Ascratch',
                            'E_direct_Tce', 'E_staged_Tce'])]
ARMS = [a for _, arms in GROUPS for a in arms]


def contrast_label(c):
    a, b = [x.strip() for x in c.split(' - ')]
    return f'{NAME[a]} $-$ {NAME[b]}'
