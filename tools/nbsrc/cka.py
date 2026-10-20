"""CKA v2 (R1: CKA baselines; plan C1, C3). Linear CKA between the centre-epoch representations of two models,
computed per test subject (unit = subject; both models are the fold model that holds that subject out).
Levels: 'spatial' = CNN embedding of the centre epoch (cnn_feat), 'temporal' = context representation used by the
classifier (ctx_feat: Transformer / linear-attention output after LayerNorm, GRU output after LayerNorm).
Baseline: the same comparison with the second model replaced by a randomly initialised model of its architecture."""
import itertools, os
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import skd

LEVELS = {'spatial': 'cnn_feat', 'temporal': 'ctx_feat'}


def linear_cka(X, Y):
    X = X.astype(np.float64) - X.mean(0, keepdims=True)
    Y = Y.astype(np.float64) - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else np.nan


@torch.no_grad()
def features(model, ds, device, bs=256):
    model.eval()
    out = {lv: [] for lv in LEVELS}
    for x, _, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0 if os.name == 'nt' else 2):
        o = model(x.to(device))
        for lv, key in LEVELS.items():
            out[lv].append(o[key].float().cpu().numpy())
    return {lv: np.concatenate(v) for lv, v in out.items()}


def trained_model(runs_dir, arm, fold, seed, device):
    C = dict(RUNS_DIR=runs_dir)
    res = skd.load_result(C, arm, fold, seed)
    assert res is not None, f'{arm} fold{fold} not trained'
    m = skd.build_model(res['kind'], res['target_dim']).to(device)
    m.load_state_dict(torch.load(f"{skd.run_dir(C, arm, fold, seed)}/best.pt", map_location=device, weights_only=True))
    return m, res['kind']


def random_model(kind, seed, device):
    torch.manual_seed(seed)
    return skd.build_model(kind).to(device)


def planned_tests(models, kind_of):
    """Fixed before computing any CKA. 'vs_random': each trained pair (X, Y) vs (X, random-init of Y's architecture).
    'pathway': the comparisons the manuscript interprets. One BH family per (test type, level)."""
    rows = []
    for lv in LEVELS:
        for x, y in itertools.combinations(models, 2):
            rows.append(dict(family=f'vs_random|{lv}', level=lv, a=f'{x} ~ {y}', b=f'{x} ~ random {kind_of[y]}'))
        for a, b in [('Assistant ~ Edge-Staged', 'Assistant ~ Edge-Direct'),
                     ('Teacher ~ Edge-Direct', 'Teacher ~ Edge-Staged'),
                     ('Assistant ~ Edge-Staged', 'Assistant ~ Edge-Scratch'),
                     ('Assistant ~ Edge-Direct', 'Assistant ~ Edge-Scratch')]:
            rows.append(dict(family=f'pathway|{lv}', level=lv, a=a, b=b))
    return pd.DataFrame(rows)


def fold_cka(feats, subj, models, kind_of):
    """feats: {name: {level: (N, d)}} incl. 'random <kind>' entries; subj: (N,) subject ids.
    Returns per-subject rows and the fold-level (all test epochs of the fold) rows."""
    pairs = [(x, y) for x, y in itertools.combinations(models, 2)]
    pairs += [(x, f'random {kind_of[y]}') for x, y in itertools.combinations(models, 2)]
    pairs = list(dict.fromkeys(pairs))
    per_subj, per_fold = [], []
    for lv in LEVELS:
        for x, y in pairs:
            X, Y = feats[x][lv], feats[y][lv]
            per_fold.append(dict(level=lv, pair=f'{x} ~ {y}', cka=linear_cka(X, Y), n_epochs=len(X)))
            for s in np.unique(subj):
                m = subj == s
                per_subj.append(dict(level=lv, pair=f'{x} ~ {y}', subject=s, cka=linear_cka(X[m], Y[m]),
                                     n_epochs=int(m.sum())))
    return pd.DataFrame(per_subj), pd.DataFrame(per_fold)


def run_tests(plan, per_subj, exclude_subjects=()):
    from xai import wilcoxon_row, bh_by_family
    t = per_subj[~per_subj.subject.isin(list(exclude_subjects))]
    P = t.set_index(['level', 'pair', 'subject']).cka
    out = []
    for r in plan.itertuples():
        a, b = P.loc[(r.level, r.a)], P.loc[(r.level, r.b)]
        d = (a - b.reindex(a.index)).dropna()
        out.append(wilcoxon_row(d.values, family=r.family, level=r.level, a=r.a, b=r.b,
                                median_a=float(a.median()), median_b=float(b.median())))
    res = bh_by_family(pd.DataFrame(out))
    return res.rename(columns={'n_excerpts': 'n_subjects'})
