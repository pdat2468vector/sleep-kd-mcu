"""Execute the code cells of R2_quant_eval.ipynb locally on synthetic recordings (real ESP-PPQ quantization).
usage: python smoke_quant_notebook.py <notebook> <workdir>   (PYTHONPATH must contain the code dir from build_notebook.py)"""
import os, sys, json, shutil
import numpy as np, pandas as pd, torch
import skd, qsim

NB, W = sys.argv[1:3]
shutil.rmtree(W, ignore_errors=True)
cells = [''.join(c['source']) for c in json.load(open(NB, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
cfg = next(c for c in cells if 'QE_ARM, QE_SEED' in c)
body = [c for c in cells if not c.startswith(('%%writefile', '!', '# =====')) and 'drive.mount' not in c]

D = os.path.join(W, 'drive')
TRAIN_CFG = dict(EPOCHS=1, PATIENCE=20, LR=1e-3, WD=1e-4, LABEL_SMOOTHING=0.05, FOCAL_GAMMA=2.0,
                 W_CE=0.3, W_KL=0.5, W_MSE=0.2, EVAL_BS=16, NUM_WORKERS=0, USE_BF16=True)
ns = dict(os=os, np=np, pd=pd, torch=torch, display=print, K_FOLDS=5, SEEDS=[42], RUN_FOLDS=[0, 1, 2, 3, 4],
          TRAIN_CFG=TRAIN_CFG, DRIVE_ROOT=D, RESULTS=f'{D}/results', RUNS_DIR=f'{D}/runs', DATA_DIR=f'{D}/data',
          LOCAL_DATA=f'{W}/local', FOLDS_JSON=f'{D}/folds.json', CODE_DIR=os.path.dirname(skd.__file__))
os.makedirs(ns['DATA_DIR']); os.makedirs(ns['RESULTS'])
exec(cfg[cfg.index('QE_ARM, QE_SEED'):cfg.index("print('config OK')")], ns)
ns.update(N_CAL_PER_CLASS=12, N_CAL=40, QE_WORK=f'{W}/qe_work', EXPECTED_SUBJECTS=10, EXPECTED_EPOCHS=10 * 2 * 34,
          N_PER_STAGE_500=5, PPQ_PY=sys.executable)        # local venv has esp-ppq (stands in for the Python 3.12 env)

rng = np.random.default_rng(0)
recs = []
for s in range(10):                                   # 10 subjects x 2 nights, 40 epochs, 34 target epochs (3..36)
    for night in (1, 2):
        rid = f'SC4{s:02d}{night}E'
        y = rng.integers(0, 5, 40)
        x = (rng.standard_normal((40, 3, 3000)) * (1 + 0.3 * y[:, None, None])).astype(np.float16)
        np.savez(f"{ns['DATA_DIR']}/{rid}.npz", x=x, y=y, centers_trim=np.arange(3, 37), centers_full=np.zeros(0, int))
        recs.append(rid)
folds = {k: [r for r in recs if int(r[3:5]) % 5 == k] for k in range(5)}
json.dump({str(k): v for k, v in folds.items()}, open(ns['FOLDS_JSON'], 'w'))

ARMS = [dict(name='E_staged_Tce', kind='edge', loss='ce', bs=8)]   # stand-in with the deployed architecture
store = skd.Store(ns['DATA_DIR'], recs)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skd.run_all(dict(TRAIN_CFG, K_FOLDS=5, RUNS_DIR=ns['RUNS_DIR'], ARMS=ARMS, SEEDS=[42]), store, folds, dev)

print('\n######## cell 0 (imports)'); exec(body[0], ns)
print('\n######## cell 1 (data)'); exec(body[1], ns)
# fake R2_hw_export output for fold 0 (calibration exactly as that notebook; 6 test windows of 11 causal epochs)
tr, va, te = skd.split_recs(folds, 0)
cal = ns['stratified'](ns['targets'](tr), ns['N_CAL_PER_CLASS'], 42)
os.makedirs(ns['HW_DIR'])
np.save(f"{ns['HW_DIR']}/calib_epochs.npy", np.stack([ns['store'].recs[ri]['x'][c].astype(np.float32) for ri, c, _ in cal]))
R0 = ns['store'].recs[ns['store'].pos[te[0]]]
win11 = np.stack([R0['x'][np.clip(np.arange(c - 10, c + 1), 0, len(R0['y']) - 1)] for c in (3, 10, 20, 30, 36, 5)])
np.savez(f"{ns['HW_DIR']}/test_windows.npz", x=win11)
for i, c in enumerate(body[2:], 2):
    print(f'\n######## cell {i}: {c.strip().splitlines()[0][:80]}')
    exec(c, ns)

# ---- checks
R = ns['RESULTS']
ct = pd.read_csv(f'{R}/quant_eval_contrasts.csv')
assert len(ct) == 2 * 2 * 8 and set(ct.metric) == {'mf1', 'acc', 'kappa', 'f1_W', 'f1_N1', 'f1_N2', 'f1_N3', 'f1_REM'}
assert set(ct[ct.subjects.str.startswith('79')].n_subjects) == {8} and set(ct[ct.subjects.str.startswith('100')].n_subjects) == {10}
p = pd.read_csv(f'{R}/quant_eval_pooled.csv')
assert set(p.model) == {'fp32', 'esp32', 'esp32s3', 'main_eval'}
# pooled FP32 Macro-F1 == recomputed from the saved logits; contrast point estimate == difference of pooled values
d = {k: np.concatenate([np.load(f"{ns['QE_DIR']}/fold{f}.npz", allow_pickle=True)[k] for f in range(5)]) for k in ('y', 'logits_fp32', 'logits_esp32', 'rec')}
m = skd.metrics(d['y'], d['logits_fp32'].argmax(1))
assert abs(100 * m['mf1'] - p[(p.model == 'fp32') & p.subjects.str.startswith('100')].mf1.iloc[0]) < 1e-9
r = ct[(ct.subjects.str.startswith('100')) & (ct.contrast == 'esp32 - fp32') & (ct.metric == 'mf1')].iloc[0]
pe = p[p.subjects.str.startswith('100')].set_index('model')
assert abs(r.diff_pp - (pe.loc['esp32', 'mf1'] - pe.loc['fp32', 'mf1'])) < 1e-9
# the INT8 simulation really changes the outputs, and only slightly
dl = np.abs(d['logits_esp32'] - d['logits_fp32'])
assert dl.max() > 0, 'quantization had no effect?'
# fold-0 simulation of the chip windows == quantize_export-style local simulation with the same calibration
info0 = json.load(open(f"{ns['QE_DIR']}/fold0_info.json"))
model = skd.build_model('edge', None)
model.load_state_dict(torch.load(f"{skd.run_dir(ns, 'E_staged_Tce', 0, 42)}/best.pt", map_location='cpu', weights_only=True))
model.eval()
g, _ = qsim.quantize_blocks(model, qsim.calibration_subset(np.load(f"{ns['HW_DIR']}/calib_epochs.npy"), ns['N_CAL'], 42),
                           'esp32', f'{W}/local_q', f'{W}/local_q_out')
sim = qsim.QuantEdge(model, g, torch.device('cpu'))
X = torch.from_numpy(win11.astype(np.float32))
with torch.no_grad():
    lq = qsim.causal_head(model, sim.features(X.reshape(-1, 3, 3000)).view(len(X), 11, -1)).numpy()
nb_ref = np.array(json.load(open(f"{ns['QE_DIR']}/sim_reference_fold0_esp32.json"))['logits_sim'])
print('fold-0 chip windows: notebook (GPU) vs local (CPU) simulation max |d| =', np.abs(lq - nb_ref).max())
assert np.abs(lq - nb_ref).max() < 1e-3
# 2b: the chip-comparison set (5 per stage here), FP32 reference == model forward on the window, all from fold-0 test subjects
w5 = np.load(f"{ns['HW_DIR']}/test_windows_500.npz")
r5 = json.load(open(f"{ns['HW_DIR']}/pc_reference_500.json"))
assert w5['x'].shape == (25, 11, 3, 3000) and list(np.bincount(w5['y'])) == [5] * 5 and r5['n'] == 25
assert {r[:5] for r in w5['rec']} <= {r[:5] for r in te}
with torch.no_grad():
    l11 = qsim.causal_head(model, model.cnn(torch.from_numpy(w5['x'].reshape(-1, 3, 3000).astype(np.float32))).view(25, 11, -1)).numpy()
assert np.abs(l11 - np.array(r5['logits_11'])).max() < 1e-5
assert os.path.exists(f"{ns['DRIVE_ROOT']}/hw_export_500.zip")
print(sorted(os.listdir(R)))
print('SMOKE QUANT NOTEBOOK OK')
