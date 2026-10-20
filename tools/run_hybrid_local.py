"""Run notebook cells 3b, 3e and 3f locally on the exported Colab runs (drive_export/) and check 3b against Colab.
usage: python run_hybrid_local.py <export_dir> <out_dir>   (PYTHONPATH must contain the code dir written by build_notebook.py)
The arm list and PRIMARY_CONTRASTS are read from the built notebook, so they are exactly those used on Colab."""
import os, sys, json
import numpy as np, pandas as pd
import skd

EXP, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
_here = os.path.dirname(os.path.abspath(__file__))
_nb = next(p for p in (os.path.join(_here, '..', 'R2_subject_level_rerun.ipynb'),                 # project tree
                       os.path.join(_here, '..', 'notebooks', 'R2_subject_level_rerun.ipynb'))    # code release
           if os.path.exists(p))
nb = json.load(open(_nb, encoding='utf-8'))
cfg = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and 'PRIMARY_CONTRASTS = [' in ''.join(c['source']))
ns = {}
exec(cfg[cfg.index('ARMS = ['):cfg.index('C = dict(')], ns)
ARMS, PRIMARY_CONTRASTS = ns['ARMS'], ns['PRIMARY_CONTRASTS']
C = dict(K_FOLDS=5, RUNS_DIR=os.path.join(EXP, 'runs'), ARMS=ARMS, SEEDS=[42])

# 3b
ct = skd.cluster_contrasts(C, PRIMARY_CONTRASTS, n_boot=10000)
colab = pd.read_csv(os.path.join(EXP, 'results', 'primary_cluster_bootstrap.csv'))
num = ['value_a', 'value_b', 'diff_pp', 'ci95_low_pp', 'ci95_high_pp', 'p_boot', 'q_bh']
same_3b = (len(ct) == len(colab) and (ct.contrast.values == colab.contrast.values).all()
           and np.allclose(ct[num].values, colab[num].values, rtol=0, atol=1e-9))
print('3b local == Colab primary_cluster_bootstrap.csv:', same_3b)
assert same_3b

# 3e
folds = {int(k): v for k, v in json.load(open(os.path.join(EXP, 'folds_subject_level.json'))).items()}
f1_subjects = sorted({r[:5] for r in folds[1]})
ct_x = skd.cluster_contrasts(C, PRIMARY_CONTRASTS, n_boot=10000, exclude_subjects=f1_subjects)
ct_x.to_csv(os.path.join(OUT, 'sensitivity_exclF1_cluster_bootstrap.csv'), index=False)

# 3f
dep = skd.tuned_dependent_arms(ARMS)
print('tuned-dependent arms:', sorted(dep), '| excluded subjects:', len(f1_subjects))
ct_h = skd.hybrid_contrasts(ct, ct_x, dep)
ct_h.to_csv(os.path.join(OUT, 'primary_hybrid_cluster_bootstrap.csv'), index=False)
pd.set_option('display.width', 250)
cols = ['contrast', 'metric', 'n_subjects', 'value_a', 'value_b', 'diff_pp', 'ci95_low_pp', 'ci95_high_pp', 'p_boot', 'q_bh']
print(ct_h[ct_h.family == 'primary'][cols].round(4).to_string(index=False))
print(ct_h[ct_h.family == 'secondary'][cols].round(4).to_string(index=False))
