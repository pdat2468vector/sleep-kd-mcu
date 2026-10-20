"""Local check of the XAI v2 pipeline on the DREAMS files already in the project (CPU, random-weight models only).
usage: python smoke_xai.py <dreams_events_dir> <workdir>   (PYTHONPATH must contain the code dir from build_notebook.py)"""
import os, sys, shutil, time
import numpy as np, pandas as pd, torch
import skd, xai

SRC, W = sys.argv[1:3]
shutil.rmtree(W, ignore_errors=True)
dirs = {db: xai.find_excerpt_dir(os.path.join(SRC, db)) for db in xai.DBS}
meta = xai.process_all(dirs, W)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(meta.drop(columns=['n_samples_native']).to_string(index=False))
assert (meta.hyp_values == 360).all() and (meta.n_epochs == 60).all()
print('target epochs:', meta.groupby('db').n_event_epochs.sum().to_dict())

store = skd.Store(W, list(meta.rec))
mask_of = {r: np.load(os.path.join(W, f'{r}.npz'))['mask'] for r in meta.rec}
x0 = store.recs[0]['x'].astype(np.float32)
print('z-score check: max |epoch mean|', float(np.abs(x0.mean(2)).max()), '| epoch std range',
      float(x0.std(2).min()), float(x0.std(2).max()))

b = xai.run_baselines(store, list(meta.rec), mask_of)
print(b.groupby([b.rec.str.split('_').str[0], 'method']).auroc.mean().round(3))

dev = torch.device('cpu')
t0 = time.time()
rows = []
for kind in ('assistant', 'edge'):
    m = xai.random_model(kind, 0, dev)
    r = xai.run_model(m, store, ['kcomplexes_excerpt9', 'spindles_excerpt3'], mask_of, dev, batch=8, ig_steps=8)
    rows.append(r.assign(model_id=f'{kind}_rand0', spec=f'{kind}-random'))
r = pd.concat(rows)
print(r.groupby(['spec', 'rec', 'method']).agg(auroc=('auroc', 'mean'), n=('auroc', 'size'),
                                               ig_rel_err=('ig_rel_err', 'median')).round(3))
print(f'attribution time {time.time() - t0:.0f}s')

# statistics plumbing on fake excerpt values
e = xai.excerpt_level(b.assign(model_id='none', spec=b.method, db=b.rec.str.split('_').str[0]), ['spec', 'db'])
tab = pd.DataFrame([xai.wilcoxon_row(g.auroc.values - 0.5, family=f'Q1|{db}', spec=s)
                    for (s, db), g in e.groupby(['spec', 'db'])])
print(xai.bh_by_family(tab).round(4).to_string(index=False))
print('SMOKE XAI OK')
