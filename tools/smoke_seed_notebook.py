"""Execute the code cells of R2_seed_robustness.ipynb locally on synthetic recordings (CPU, 1 epoch per run).
Seed-42 runs (T_ce, A_kd_Tce and the three Edge arms) are trained first, as the Sleep-EDF notebook would have done.
usage: python smoke_seed_notebook.py <notebook> <workdir>   (PYTHONPATH must contain the code dir from build_notebook.py)"""
import os, sys, json, shutil
import numpy as np, pandas as pd, torch
import skd

NB, W = sys.argv[1:3]
shutil.rmtree(W, ignore_errors=True)
cells = [''.join(c['source']) for c in json.load(open(NB, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
cfg = next(c for c in cells if 'SEED_ARMS = [' in c)
body = [c for c in cells if not c.startswith(('%%writefile', '!', '# =====')) and 'drive.mount' not in c]

D = os.path.join(W, 'drive')
TRAIN_CFG = dict(EPOCHS=1, PATIENCE=20, LR=1e-3, WD=1e-4, LABEL_SMOOTHING=0.05, FOCAL_GAMMA=2.0,
                 W_CE=0.3, W_KL=0.5, W_MSE=0.2, EVAL_BS=16, NUM_WORKERS=0, USE_BF16=True)
ns = dict(os=os, np=np, pd=pd, torch=torch, display=print, K_FOLDS=5, SEEDS=[42], RUN_FOLDS=[0, 1, 2, 3, 4],
          TRAIN_CFG=TRAIN_CFG, RESULTS=f'{D}/results', RUNS_DIR=f'{D}/runs', DATA_DIR=f'{D}/data',
          LOCAL_DATA=f'{W}/local', FOLDS_JSON=f'{D}/folds.json', CODE_DIR=os.path.dirname(skd.__file__))
os.makedirs(ns['DATA_DIR']); os.makedirs(ns['RESULTS'])
exec(cfg[cfg.index('SEED_ARMS = ['):cfg.index("print('config OK')")], ns)

rng = np.random.default_rng(0)
recs = []
for s in range(10):                                   # 10 subjects x 2 nights, 40 epochs each
    for night in (1, 2):
        rid = f'SC4{s:02d}{night}E'
        y = rng.integers(0, 5, 40)
        x = (rng.standard_normal((40, 3, 3000)) + y[:, None, None] * 0.3).astype(np.float16)
        np.savez(f"{ns['DATA_DIR']}/{rid}.npz", x=x, y=y, centers_trim=np.arange(40), centers_full=np.zeros(0, int))
        recs.append(rid)
folds = {k: [r for r in recs if int(r[3:5]) % 5 == k] for k in range(5)}
json.dump({str(k): v for k, v in folds.items()}, open(ns['FOLDS_JSON'], 'w'))

# ---- seed 42 as produced by the Sleep-EDF notebook (same Edge arm definitions, no src_seed)
ARMS42 = [dict(name='T_ce', kind='teacher', loss='ce', bs=8),
          dict(name='A_kd_Tce', kind='assistant', loss='kd', src='T_ce', T=5.0, bs=8),
          dict(name='E_scratch', kind='edge', loss='ce', bs=64),
          dict(name='E_direct_Tce', kind='edge', loss='kd', src='T_ce', T=4.0, bs=64),
          dict(name='E_staged_Tce', kind='edge', loss='kd', src='A_kd_Tce', T=4.0, bs=64)]
store = skd.Store(ns['DATA_DIR'], recs)
skd.run_all(dict(TRAIN_CFG, K_FOLDS=5, RUNS_DIR=ns['RUNS_DIR'], ARMS=ARMS42, SEEDS=[42]), store, folds,
            torch.device('cpu'))
src_mtime = {f: os.path.getmtime(f'{skd.run_dir(ns, "A_kd_Tce", f, 42)}/best.pt') for f in range(5)}

# missing source -> clear error before any training
bad = dict(ns['C_NEW'], RUNS_DIR=f'{W}/empty_runs', SEEDS=[43])
try:
    skd.run_arm(ns['SEED_ARMS'][1], 0, 43, bad, store, folds, torch.device('cpu'))
    raise SystemExit('expected a missing-source error')
except RuntimeError as e:
    assert 'seed42 not trained yet' in str(e), e
    print('missing seed-42 source -> error OK:', e)

for i, c in enumerate(body):
    print(f'\n######## cell {i}: {c.strip().splitlines()[0][:80]}')
    exec(c, ns)

# checks: new seeds are trained from the seed-42 sources, which stay untouched; the seed changes the Edge model
for f in range(5):
    assert os.path.getmtime(f'{skd.run_dir(ns, "A_kd_Tce", f, 42)}/best.pt') == src_mtime[f]
    assert not os.path.exists(skd.run_dir(ns, 'A_kd_Tce', f, 43)) and not os.path.exists(skd.run_dir(ns, 'T_ce', f, 44))
    for s in (43, 44):
        r = skd.load_result(ns, 'E_staged_Tce', f, s)
        assert r['seed'] == s and r['src'] == 'A_kd_Tce' and r['src_seed'] == 42
w = {s: torch.load(f'{skd.run_dir(ns, "E_staged_Tce", 0, s)}/best.pt', map_location='cpu', weights_only=True)
     for s in (42, 43, 44)}
k0 = next(iter(w[42]))
assert not torch.equal(w[42][k0], w[43][k0]) and not torch.equal(w[43][k0], w[44][k0]), 'seeds gave identical weights'
ps = pd.read_csv(f"{ns['RESULTS']}/seed_robustness_pooled_per_seed.csv")
assert sorted(ps.seed.unique()) == [42, 43, 44] and (ps.n_subjects == 8).all(), ps
ct = pd.read_csv(f"{ns['RESULTS']}/seed_robustness_contrasts.csv")
print(ct.groupby(['seeds', 'subjects']).size().to_string())
assert len(ct) == 5 * 3 * 4                          # (pooled 79, pooled 100, 3 single seeds) x 3 contrasts x 4 metrics
pooled = ct[(ct.seeds == '42+43+44') & (ct.subjects == 79)]
assert (pooled.n_epochs == 3 * 8 * 2 * 40).all(), pooled.n_epochs   # 3 seeds x 8 subjects x 2 nights x 40 epochs
# single-seed-42 rows must equal the seed-42-only analysis of the main notebook
ref = skd.cluster_contrasts(dict(TRAIN_CFG, K_FOLDS=5, RUNS_DIR=ns['RUNS_DIR'], SEEDS=[42]), ns['SEED_CONTRASTS'],
                            n_boot=10000, exclude_subjects=ns['f1_subjects'])
got = ct[ct.seeds == '42'].reset_index(drop=True)
assert np.allclose(got.diff_pp, ref.diff_pp) and np.allclose(got.ci95_low_pp, ref.ci95_low_pp)
print('\nseed 42 rows == main-notebook analysis; seeds 43/44 distilled from seed-42 sources; sources untouched')
print(sorted(os.listdir(ns['RESULTS'])))
print('SMOKE SEED NOTEBOOK OK')
