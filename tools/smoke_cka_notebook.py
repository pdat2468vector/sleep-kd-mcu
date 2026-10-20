"""Execute the code cells of R2_cka_v2.ipynb locally on synthetic recordings with fake (random-weight) checkpoints.
usage: python smoke_cka_notebook.py <notebook> <workdir>   (PYTHONPATH must contain the code dir from build_notebook.py)"""
import os, sys, json, shutil
import numpy as np, pandas as pd, torch
import skd

NB, W = sys.argv[1:3]
shutil.rmtree(W, ignore_errors=True)
cells = [''.join(c['source']) for c in json.load(open(NB, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
cfg = next(c for c in cells if 'CKA_MODELS = {' in c)
body = [c for c in cells if not c.startswith(('%%writefile', '!', '# =====')) and 'drive.mount' not in c]

D = os.path.join(W, 'drive')
ns = dict(os=os, np=np, pd=pd, torch=torch, display=print, K_FOLDS=5, SEEDS=[42],
          RESULTS=f'{D}/results', RUNS_DIR=f'{D}/runs', DATA_DIR=f'{D}/data', LOCAL_DATA=f'{W}/local',
          FOLDS_JSON=f'{D}/folds.json', CODE_DIR=os.path.dirname(skd.__file__))
os.makedirs(ns['DATA_DIR']); os.makedirs(ns['RESULTS'])
exec(cfg[cfg.index('# CKA v2'):cfg.index("print('config OK')")], ns)
ns['UMAP_N'] = 300

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
for arm in ns['CKA_MODELS'].values():
    kind = {'T': 'teacher', 'A': 'assistant', 'E': 'edge'}[arm[0]]
    tdim = None if arm == 'E_scratch' else (256 if arm in ('A_kd_Tce', 'E_direct_Tce') else 128)
    for f in range(5):
        d = skd.run_dir(dict(RUNS_DIR=ns['RUNS_DIR']), arm, f, 42)
        os.makedirs(d)
        torch.manual_seed(f)
        torch.save(skd.build_model(kind, tdim).state_dict(), f'{d}/best.pt')
        json.dump(dict(kind=kind, target_dim=tdim), open(f'{d}/result.json', 'w'))

for i, c in enumerate(body):
    print(f'\n######## cell {i}: {c.splitlines()[0][:80]}')
    exec(c, ns)
res = pd.read_csv(f"{ns['RESULTS']}/cka_tests.csv")
print(res.groupby(['subjects', 'family']).size().to_string())
import cka
X = rng.standard_normal((200, 16))
assert abs(cka.linear_cka(X, X) - 1) < 1e-12 and abs(cka.linear_cka(X, 3 * X @ np.linalg.qr(rng.standard_normal((16, 16)))[0]) - 1) < 1e-9
print('linear CKA: self = 1 and invariant to orthogonal transform + isotropic scaling (checked)')
print(sorted(os.listdir(ns['RESULTS'])))
print('SMOKE CKA NOTEBOOK OK')
