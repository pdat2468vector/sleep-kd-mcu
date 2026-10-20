"""Execute the code cells of R2_xai_dreams.ipynb locally (CPU) with fake checkpoints and a reduced configuration.
usage: python smoke_xai_notebook.py <notebook> <dreams_events_dir> <workdir>
(PYTHONPATH must contain the code dir written by build_notebook.py)"""
import os, sys, json, shutil
import numpy as np, pandas as pd, torch
import skd, xai

NB, SRC, W = sys.argv[1:4]
shutil.rmtree(W, ignore_errors=True)
cells = [''.join(c['source']) for c in json.load(open(NB, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
cfg = next(c for c in cells if 'SPECS = {' in c)
body = [c.replace('FINE_STEPS, FINE_LO, FLAG = 256, 1e-9, 0.10', 'FINE_STEPS, FINE_LO, FLAG = 8, 1e-9, 0.10')
        for c in cells if not c.startswith(('%%writefile', '!', '# =====')) and 'drive.mount' not in c]

# local stand-in for the CONFIG cell: same SPECS block, reduced sizes, local paths
DRIVE_ROOT = os.path.join(W, 'drive')
ns = dict(os=os, np=np, pd=pd, torch=torch, display=print, DRIVE_ROOT=DRIVE_ROOT,
          RUNS_DIR=f'{DRIVE_ROOT}/runs', RESULTS=f'{DRIVE_ROOT}/results', K_FOLDS=1, SEEDS=[42])
spec_block = cfg[cfg.index('# XAI v2'):cfg.index("print('config OK')")]
exec(spec_block, ns)
ns.update(IG_STEPS=4, N_RANDOM=1, XAI_BS=8, CODE_DIR=os.path.dirname(skd.__file__))

# fake trained checkpoints (random weights) in the Drive layout
for runs, arm in ns['SPECS'].values():
    kind = 'assistant' if arm.startswith('A_') else 'edge'
    d = skd.run_dir(dict(RUNS_DIR=runs), arm, 0, 42)
    os.makedirs(d, exist_ok=True)
    torch.save(skd.build_model(kind, 256 if arm == 'E_direct_Tce' else 128).state_dict(), f'{d}/best.pt')
    json.dump(dict(kind=kind, target_dim=256 if arm == 'E_direct_Tce' else 128), open(f'{d}/result.json', 'w'))

# cell 1 would download from Zenodo; build the same table from the local copy instead
dirs = {db: xai.find_excerpt_dir(os.path.join(SRC, db)) for db in xai.DBS}
xai.process_all(dirs, ns['DREAMS_DATA']).to_csv(f"{ns['RESULTS']}/xai_dreams_excerpts.csv", index=False)

def run_cells():
    for i, c in enumerate(body):
        print(f'\n######## cell {i}: {c.splitlines()[0][:80]}')
        exec(c, ns)


# pass 1: HMC Assistant not trained yet -> 3b/3c must skip it, the statistics cell must stop with a clear message
hold = skd.run_dir(dict(RUNS_DIR=ns['SPECS']['Assistant (HMC)'][0]), 'A_kd_Tce', 0, 42)
shutil.move(hold, hold + '_hold')
try:
    run_cells()
    raise SystemExit('expected the statistics cell to stop')
except AssertionError as e:
    assert 'Assistant (HMC)' in str(e), e
    print('\npass 1 stopped as expected:', e)
# pass 2: training finished -> rerun everything; finished models are reused
shutil.move(hold + '_hold', hold)
run_cells()
print('\nfiles:', sorted(os.listdir(ns['RESULTS'])))
res = pd.read_csv(f"{ns['RESULTS']}/xai_tests.csv")
print(res.groupby('family').size().to_string())
print('SMOKE XAI NOTEBOOK OK')
