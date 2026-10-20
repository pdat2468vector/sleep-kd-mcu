"""Smoke test of the full pipeline on tiny synthetic data (CPU, 1 epoch per run).
usage: python smoke_train.py <workdir>   (PYTHONPATH must contain the code dir written by build_notebook.py)"""
import os, sys, shutil
import numpy as np, pandas as pd, torch
import skd, prep

W = sys.argv[1]
shutil.rmtree(W, ignore_errors=True)
rng = np.random.default_rng(0)


def make_data(d, prefix, n_subj=5, nights=2):
    os.makedirs(d)
    rows = []
    for s in range(n_subj):
        for night in range(1, nights + 1):
            rec = f'{prefix}{s}{night}E0' if prefix.startswith('SC') else f'{prefix}{s}{night}'
            n = 40
            y = rng.integers(0, 5, n).astype(np.int8); y[:3] = -1
            np.savez(os.path.join(d, f'{rec}.npz'), x=rng.standard_normal((n, 3, 3000)).astype(np.float16), y=y,
                     ptp=np.zeros(n, np.float32), centers_trim=np.where(y >= 0)[0].astype(np.int64),
                     centers_full=np.zeros(0, np.int64), lo=0, n_total=n)
            rows.append(dict(rec=rec, subject=rec[:5], cohort=rec[:2]))
    return pd.DataFrame(rows)


meta = make_data(os.path.join(W, 'edf'), 'SC40')
meta, folds = prep.make_subject_folds(meta, k=5, seed=42)
roles = prep.fold_role_lists(meta, 5)
assert set(roles['fold0']['test']['subjects']).isdisjoint(roles['fold0']['train']['subjects'])
print('fold roles OK; fold0 test subjects', roles['fold0']['test']['subjects'])

ARMS = [
    dict(name='T_focal', kind='teacher', loss='focal', bs=8),
    dict(name='A_scratch', kind='assistant', loss='ce', bs=8),
    dict(name='A_kd_Tfocal', kind='assistant', loss='kd', src='T_focal', T=5.0, bs=8),
    dict(name='E_scratch', kind='edge', loss='ce', bs=8),
    dict(name='E_direct_Tfocal', kind='edge', loss='kd', src='T_focal', T=4.0, bs=8),
    dict(name='E_staged_Tfocal', kind='edge', loss='kd', src='A_kd_Tfocal', T=4.0, bs=8),
    dict(name='E_from_Ascratch', kind='edge', loss='kd', src='A_scratch', T=4.0, bs=8),
] + [dict(name=f'Ttune_ce_lr{lr:g}_ls{ls:g}', kind='teacher', loss='ce', lr=lr, ls=ls, bs=8, folds=[0], seeds='first')
     for lr in (1e-3, 3e-4) for ls in (0.05, 0.0)] + [
    dict(name='T_ce', kind='teacher', loss='ce', bs=8, tune=dict(prefix='Ttune_ce_', fold=0)),
    dict(name='A_kd_Tce', kind='assistant', loss='kd', src='T_ce', T=5.0, bs=8),
    dict(name='E_direct_Tce', kind='edge', loss='kd', src='T_ce', T=4.0, bs=8),
    dict(name='E_staged_Tce', kind='edge', loss='kd', src='A_kd_Tce', T=4.0, bs=8),
]
CFG = dict(EPOCHS=1, PATIENCE=20, LR=1e-3, WD=1e-4, LABEL_SMOOTHING=0.05, FOCAL_GAMMA=2.0, W_CE=0.3, W_KL=0.5,
           W_MSE=0.2, EVAL_BS=16, NUM_WORKERS=0, USE_BF16=True)
C = dict(CFG, K_FOLDS=5, RUNS_DIR=os.path.join(W, 'runs'), ARMS=ARMS, SEEDS=[42])
dev = torch.device('cpu')
store = skd.Store(os.path.join(W, 'edf'), sorted(r for v in folds.values() for r in v))
skd.run_all(C, store, folds, dev)

# tuning grid ran only on fold 0; T_ce fold 0 reused the selected grid run; folds 1-4 trained with the selected config
tune = skd.tuning_table(C)
assert set(tune.fold) == {0} and len(tune) == 4, tune
r0 = skd.load_result(C, 'T_ce', 0, 42)
r1 = skd.load_result(C, 'T_ce', 1, 42)
best = tune.iloc[tune.best_val_mf1.values.argmax()]          # ties -> earliest grid entry, as in skd.resolve_tuned
assert r0['reused_tuning_run'] and r0['selected_from'] == best.arm, r0
assert r1['selected_from'] == best.arm and r1['lr'] == best.lr and r1['ls'] == best.ls, r1
print('tuning OK: selected', best.arm, '| T_ce fold1 lr', r1['lr'], 'ls', r1['ls'])

df = skd.collect(C)
main = [a['name'] for a in ARMS if 'folds' not in a]
print(skd.summary(df[df.arm.isin(main)], 'test')[['mf1', 'n_runs']])
pooled = skd.pooled_table(C, main)
print(pooled[['mf1', 'acc', 'n_subjects']].round(2))
# pooled metrics must equal sklearn on the concatenated out-of-fold predictions
ys, ps = [], []
for f in range(5):
    d = np.load(os.path.join(skd.run_dir(C, 'E_staged_Tfocal', f, 42), 'preds_test.npz'), allow_pickle=True)
    ys.append(d['y']); ps.append(d['logits'].astype(np.float32).argmax(1))
m = skd.metrics(np.concatenate(ys), np.concatenate(ps))
assert abs(m['mf1'] * 100 - pooled.loc['E_staged_Tfocal', 'mf1']) < 1e-9 and abs(m['kappa'] - pooled.loc['E_staged_Tfocal', 'kappa']) < 1e-9
print('pooled metrics == sklearn OK')
ct = skd.cluster_contrasts(C, [('E_staged_Tfocal', 'E_direct_Tfocal'), ('T_ce', 'T_focal')], n_boot=500)
print(ct[['contrast', 'metric', 'family', 'diff_pp', 'ci95_low_pp', 'ci95_high_pp', 'p_boot', 'q_bh']].round(3))
print(skd.subject_wilcoxon(C, [('E_staged_Tfocal', 'E_direct_Tfocal')], n_boot=200)[['contrast', 'metric', 'p_wilcoxon']])

# HMC-style second dataset + cross evaluation both ways
hmeta = make_data(os.path.join(W, 'hmc'), 'SN0', n_subj=5, nights=1)
hmeta['cohort'] = 'HMC'; hmeta['subject'] = hmeta['rec']
hmeta, hfolds = prep.make_subject_folds(hmeta, k=5, seed=42)
hstore = skd.Store(os.path.join(W, 'hmc'), sorted(hmeta.rec))
CH = dict(CFG, K_FOLDS=5, RUNS_DIR=os.path.join(W, 'runs_hmc'), ARMS=ARMS[:1] + ARMS[2:3] + ARMS[5:6], SEEDS=[42])
skd.run_all(CH, hstore, hfolds, dev)
x1 = skd.eval_cross(C, 'E_staged_Tfocal', 42, hstore, sorted(hmeta.rec), 'HMC', dev)
x2 = skd.eval_cross(CH, 'E_staged_Tfocal', 42, store, sorted(meta.rec), 'SleepEDF', dev)
x1b = skd.eval_cross(C, 'E_staged_Tfocal', 42, hstore, sorted(hmeta.rec), 'HMC', dev)   # cached second call
assert len(x1) == len(x2) == 5 and x1.equals(x1b)
print('cross-dataset OK', x1.mf1.round(3).tolist(), x2.mf1.round(3).tolist())
print('SMOKE OK')
