# skd.py — training / evaluation / statistics for the subject-level re-run.
import os, json, time, random, shutil, contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

from model_v11.teacher_v11 import TeacherSleepNetV11
from model_v11.student_v11 import StudentSleepNetV11
from model_v11.student_v2 import StudentSleepNetV2
from losses_v11 import FocalLoss, KDLossV11

SEQ_LEN = 21
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
NATIVE_DIM = {'teacher': 256, 'assistant': 128}   # feature size a source model exposes for MSE matching


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def amp_ctx(device, use_bf16):
    if device.type == 'cuda' and use_bf16 and torch.cuda.is_bf16_supported():
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ================================================================ data
class Store:
    """All recordings in RAM: x (n,3,3000) fp16, y (n,), centers = epoch indices used as targets."""
    def __init__(self, data_dir, recs):
        self.recs, self.pos = [], {}
        for rid in recs:
            d = np.load(os.path.join(data_dir, f'{rid}.npz'))
            self.pos[rid] = len(self.recs)
            self.recs.append(dict(id=rid, x=d['x'], y=d['y'].astype(np.int64),
                                  trim=d['centers_trim'], full=d['centers_full']))

    def has_full(self):
        return all(len(r['full']) for r in self.recs)


class SeqDS(Dataset):
    """21-epoch windows centred on each target epoch; context never crosses a recording boundary
    (indices are clamped to the recording, i.e. edge-replicate padding)."""
    def __init__(self, store, recs, which='trim', seq_len=SEQ_LEN):
        self.store, self.half = store, seq_len // 2
        rows = [np.stack([np.full(len(store.recs[store.pos[r]][which]), store.pos[r]),
                          store.recs[store.pos[r]][which]], 1) for r in recs]
        self.index = np.concatenate(rows).astype(np.int64)

    def __len__(self):
        return len(self.index)

    def labels(self):
        return np.array([self.store.recs[r]['y'][c] for r, c in self.index])

    def rec_ids(self):
        return np.array([self.store.recs[r]['id'] for r, _ in self.index])

    def __getitem__(self, i):
        ri, c = self.index[i]
        r = self.store.recs[ri]
        idx = np.clip(np.arange(c - self.half, c + self.half + 1), 0, len(r['y']) - 1)
        return torch.from_numpy(r['x'][idx].astype(np.float32)), int(r['y'][c]), i


def split_recs(folds, i, k=5):
    test, val = folds[i], folds[(i + 1) % k]
    train = [r for j in range(k) if j not in (i, (i + 1) % k) for r in folds[j]]
    return train, val, test


# ================================================================ models
def build_model(kind, target_dim=None):
    if kind == 'teacher':
        return TeacherSleepNetV11(embed_dim=256, num_heads=8, num_layers=3, num_classes=5,
                                  seq_len=SEQ_LEN, dropout=0.1)
    if kind == 'assistant':
        return StudentSleepNetV11(embed_dim=128, teacher_embed_dim=target_dim or 256, num_heads=4,
                                  num_layers=2, num_classes=5, seq_len=SEQ_LEN, dropout=0.1)
    if kind == 'edge':
        return StudentSleepNetV2(embed_dim=64, gru_hidden=32, gru_layers=1, v1_embed_dim=target_dim or 128,
                                 num_classes=5, seq_len=SEQ_LEN, dropout=0.1)
    raise ValueError(kind)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    adaptor = sum(p.numel() for n, p in model.named_parameters() if 'adaptor' in n)
    return total, total - adaptor


# ================================================================ metrics
def metrics(y, p):
    f1 = f1_score(y, p, labels=list(range(5)), average=None, zero_division=0)
    return dict(acc=float((y == p).mean()), mf1=float(f1.mean()), kappa=float(cohen_kappa_score(y, p)),
                f1=[float(v) for v in f1], n=int(len(y)),
                cm=confusion_matrix(y, p, labels=list(range(5))).tolist())


@torch.no_grad()
def predict(model, ds, C, device):
    model.eval()
    out = []
    for x, _, _ in DataLoader(ds, batch_size=C['EVAL_BS'], shuffle=False,
                              num_workers=C['NUM_WORKERS'], pin_memory=True):
        with amp_ctx(device, C['USE_BF16']):
            out.append(model(x.to(device, non_blocking=True))['logits'].float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def cache_source(src, ds, C, device):
    """Frozen source outputs for every training window (eval mode, no augmentation -> identical to on-the-fly)."""
    src.eval()
    cache = None
    for x, _, i in DataLoader(ds, batch_size=C['EVAL_BS'], shuffle=False,
                              num_workers=C['NUM_WORKERS'], pin_memory=True):
        with amp_ctx(device, C['USE_BF16']):
            o = src(x.to(device, non_blocking=True))
        if cache is None:
            cache = {'logits': torch.empty(len(ds), 5, device=device),
                     'cnn_feat': torch.empty(len(ds), o['cnn_feat'].shape[1], dtype=torch.float16, device=device),
                     'ctx_feat': torch.empty(len(ds), o['ctx_feat'].shape[1], dtype=torch.float16, device=device)}
        i = i.to(device)
        cache['logits'][i] = o['logits'].float()
        cache['cnn_feat'][i] = o['cnn_feat'].half()
        cache['ctx_feat'][i] = o['ctx_feat'].half()
    return cache


# ================================================================ runs
def run_dir(C, arm_name, fold, seed):
    return os.path.join(C['RUNS_DIR'], f'seed{seed}', arm_name, f'fold{fold}')


def load_result(C, arm_name, fold, seed):
    p = os.path.join(run_dir(C, arm_name, fold, seed), 'result.json')
    return json.load(open(p)) if os.path.exists(p) else None


def resolve_source(C, src, fold, seed):
    return src


def resolve_tuned(C, arm):
    """Arm with arm['tune'] = dict(prefix=..., fold=f): take lr / label smoothing of the grid arm (names starting
    with prefix, trained on fold f only, first seed) that has the best VALIDATION Macro-F1. Test data never used."""
    t = arm['tune']
    seed0 = C['SEEDS'][0]
    grid = [a for a in C['ARMS'] if a['name'].startswith(t['prefix'])]
    res = {a['name']: load_result(C, a['name'], t['fold'], seed0) for a in grid}
    missing = [n for n, r in res.items() if r is None]
    if missing:
        raise RuntimeError(f'tuning grid not finished (fold {t["fold"]}, seed {seed0}): {missing}')
    best = max(grid, key=lambda a: res[a['name']]['best_val_mf1'])   # ties -> earliest grid entry (list order)
    out = {k: v for k, v in arm.items() if k != 'tune'}
    out.update(lr=best.get('lr', C['LR']), ls=best.get('ls', C['LABEL_SMOOTHING']), selected_from=best['name'],
               tuning={n: r['best_val_mf1'] for n, r in res.items()})
    return out


def _reuse_tuning_run(C, arm, fold, seed):
    """The selected grid run already is this arm's (fold, seed) run with identical settings: copy it, don't retrain."""
    src, dst = run_dir(C, arm['selected_from'], fold, seed), run_dir(C, arm['name'], fold, seed)
    os.makedirs(dst, exist_ok=True)
    for f in ('best.pt', 'preds_test.npz'):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    r = load_result(C, arm['selected_from'], fold, seed)
    r.update(arm=arm['name'], selected_from=arm['selected_from'], tuning=arm['tuning'], reused_tuning_run=True)
    json.dump(r, open(os.path.join(dst, 'result.json'), 'w'), indent=1)
    print(f'  {arm["name"]} fold{fold} seed{seed}: reused tuning run {arm["selected_from"]}')
    return r


def run_all(C, store, folds, device, run_folds=None, run_arms=None):
    """Arms in list order (= dependency order). arm['folds'] / arm['seeds'] restrict where an arm runs
    (tuning-grid arms: folds=[0], seeds='first')."""
    for arm in C['ARMS']:
        if run_arms and arm['name'] not in run_arms:
            continue
        seeds = C['SEEDS'][:1] if arm.get('seeds') == 'first' else C['SEEDS']
        for seed in seeds:
            for fold in (run_folds if run_folds is not None else range(C['K_FOLDS'])):
                if 'folds' in arm and fold not in arm['folds']:
                    continue
                run_arm(arm, fold, seed, C, store, folds, device)


def run_arm(arm, fold, seed, C, store, folds, device):
    name = arm['name']
    odir = run_dir(C, name, fold, seed)
    if os.path.exists(os.path.join(odir, 'result.json')):
        print(f'[skip] {name} fold{fold} seed{seed} (done)')
        return load_result(C, name, fold, seed)
    if 'tune' in arm:
        tune_fold = arm['tune']['fold']
        arm = resolve_tuned(C, arm)
        print(f'  {name}: selected {arm["selected_from"]} (lr {arm["lr"]}, ls {arm["ls"]}) by validation Macro-F1 '
              f'on fold {tune_fold}')
        if fold == tune_fold and seed == C['SEEDS'][0]:
            return _reuse_tuning_run(C, arm, fold, seed)
    os.makedirs(odir, exist_ok=True)
    t_start = time.time()
    tr, va, te = split_recs(folds, fold, C['K_FOLDS'])
    train_ds, val_ds, test_ds = SeqDS(store, tr), SeqDS(store, va), SeqDS(store, te)
    y_val, y_test = val_ds.labels(), test_ds.labels()
    set_seed(seed * 1000 + fold)

    # ---- source (for KD arms)
    cache, src_name, target_dim = None, None, None
    src_seed = arm.get('src_seed', seed)        # fixed source seed: only the student's training seed varies
    if arm['loss'] == 'kd':
        src_name = resolve_source(C, arm['src'], fold, seed)
        sres = load_result(C, src_name, fold, src_seed)
        if sres is None:
            raise RuntimeError(f'source {src_name} fold{fold} seed{src_seed} not trained yet')
        assert sres['fold'] == fold and sres['seed'] == src_seed, 'source model from another fold / seed'
        src = build_model(sres['kind'], sres['target_dim']).to(device)
        src.load_state_dict(torch.load(os.path.join(run_dir(C, src_name, fold, src_seed), 'best.pt'),
                                       map_location=device, weights_only=True))
        target_dim = NATIVE_DIM[sres['kind']]
        t0 = time.time()
        cache = cache_source(src, train_ds, C, device)
        print(f'  cached {src_name} outputs for {len(train_ds):,} windows ({time.time() - t0:.0f}s)')
        del src
        torch.cuda.empty_cache()

    model = build_model(arm['kind'], target_dim).to(device)
    p_total, p_infer = count_params(model)

    # ---- loss
    ls = arm.get('ls', C['LABEL_SMOOTHING'])
    if arm['loss'] == 'focal':
        y_tr = train_ds.labels()
        counts = np.bincount(y_tr, minlength=5).astype(np.float64)
        cw = torch.tensor(counts.sum() / (5.0 * counts + 1e-8), dtype=torch.float32, device=device)
        crit = FocalLoss(gamma=C['FOCAL_GAMMA'], class_weights=cw, label_smoothing=ls)
    elif arm['loss'] == 'ce':
        crit = nn.CrossEntropyLoss(label_smoothing=ls)
    else:
        crit = KDLossV11(temperature=arm['T'], w_ce=C['W_CE'], w_kl=C['W_KL'], w_mse=C['W_MSE'],
                         label_smoothing=ls)

    lr = arm.get('lr', C['LR'])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C['WD'])
    sch = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    st = dict(epoch=0, best=-1.0, best_epoch=0, patience=0, history=[])
    last_p, best_p = os.path.join(odir, 'last.pt'), os.path.join(odir, 'best.pt')
    if os.path.exists(last_p):                                # resume after a Colab disconnect
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sch'])
        st = ck['state']
        print(f'  resumed at epoch {st["epoch"]}')

    g = torch.Generator().manual_seed(seed * 1000 + fold + st['epoch'])
    loader = DataLoader(train_ds, batch_size=arm['bs'], shuffle=True, drop_last=True, generator=g,
                        num_workers=C['NUM_WORKERS'], pin_memory=True, persistent_workers=C['NUM_WORKERS'] > 0)
    print(f'  {name} fold{fold} seed{seed} | params {p_total:,} (inference {p_infer:,}) | '
          f'train {len(train_ds):,} val {len(val_ds):,} test {len(test_ds):,} | lr {lr} ls {ls} bs {arm["bs"]}')

    for ep in range(st['epoch'] + 1, C['EPOCHS'] + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, i in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx(device, C['USE_BF16']):
                o = model(x)
            o = {k: v.float() for k, v in o.items()}
            if cache is not None:
                i = i.to(device)
                tgt = {'logits': cache['logits'][i], 'cnn_feat': cache['cnn_feat'][i].float(),
                       'ctx_feat': cache['ctx_feat'][i].float()}
                loss = crit(o, tgt, y)['loss']
            else:
                loss = crit(o['logits'], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sch.step()
        vm = metrics(y_val, predict(model, val_ds, C, device).argmax(1))
        improved = vm['mf1'] > st['best']
        if improved:
            st.update(best=vm['mf1'], best_epoch=ep, patience=0)
            torch.save(model.state_dict(), best_p)
        else:
            st['patience'] += 1
        st['epoch'] = ep
        st['history'].append(dict(epoch=ep, loss=tot / max(nb, 1), val_acc=vm['acc'], val_mf1=vm['mf1'],
                                  val_n1=vm['f1'][1], sec=time.time() - t0))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), state=st), last_p)
        print(f'    ep{ep:3d} loss {tot / max(nb, 1):.4f} | val acc {vm["acc"]:.4f} mf1 {vm["mf1"]:.4f} '
              f'n1 {vm["f1"][1]:.4f} | {time.time() - t0:.0f}s {"*" if improved else ""}')
        if st['patience'] >= C['PATIENCE']:
            print(f'    early stop (best epoch {st["best_epoch"]})')
            break

    # ---- test with the best-validation checkpoint
    model.load_state_dict(torch.load(best_p, map_location=device, weights_only=True))
    logits = predict(model, test_ds, C, device)
    tm = metrics(y_test, logits.argmax(1))
    pred = dict(rec=test_ds.rec_ids(), center=test_ds.index[:, 1], y=y_test, logits=logits.astype(np.float16))
    tfm = None
    if store.has_full():
        full_ds = SeqDS(store, te, which='full')
        lf = predict(model, full_ds, C, device)
        tfm = metrics(full_ds.labels(), lf.argmax(1))
        pred.update(full_rec=full_ds.rec_ids(), full_center=full_ds.index[:, 1], full_y=full_ds.labels(),
                    full_logits=lf.astype(np.float16))
    np.savez_compressed(os.path.join(odir, 'preds_test.npz'), **pred)
    res = dict(arm=name, kind=arm['kind'], loss=arm['loss'], src=src_name,
               src_seed=src_seed if src_name else None, T=arm.get('T'), lr=lr, ls=ls, bs=arm['bs'],
               selected_from=arm.get('selected_from'), tuning=arm.get('tuning'),
               fold=fold, seed=seed, target_dim=target_dim, params_total=p_total, params_inference=p_infer,
               best_epoch=st['best_epoch'], best_val_mf1=st['best'], epochs_run=st['epoch'],
               n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds),
               test=tm, test_full_night=tfm, minutes=(time.time() - t_start) / 60, history=st['history'])
    json.dump(res, open(os.path.join(odir, 'result.json'), 'w'), indent=1)
    if os.path.exists(last_p):
        os.remove(last_p)
    print(f'  ==> {name} fold{fold}: acc {tm["acc"]:.4f} mf1 {tm["mf1"]:.4f} kappa {tm["kappa"]:.4f} '
          f'n1 {tm["f1"][1]:.4f} ({res["minutes"]:.0f} min)')
    return res


# ================================================================ aggregation & statistics
def collect(C):
    rows = []
    for a in C['ARMS']:
        for s in C['SEEDS']:
            for f in range(C['K_FOLDS']):
                r = load_result(C, a['name'], f, s)
                if r is None:
                    continue
                for split in ['test', 'test_full_night']:
                    m = r.get(split)
                    if m is None:
                        continue
                    row = dict(arm=a['name'], seed=s, fold=f, split=split, src=r['src'],
                               acc=m['acc'], mf1=m['mf1'], kappa=m['kappa'],
                               params_total=r['params_total'], params_inference=r['params_inference'],
                               best_epoch=r['best_epoch'], minutes=r['minutes'])
                    row.update({f'f1_{nm}': v for nm, v in zip(STAGES, m['f1'])})
                    rows.append(row)
    return pd.DataFrame(rows)


def summary(df, split='test'):
    d = df[df.split == split]
    cols = ['acc', 'mf1', 'kappa'] + [f'f1_{s}' for s in STAGES]
    g = d.groupby('arm')[cols]
    mean, sd = g.mean(), g.std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        scale = 1 if c == 'kappa' else 100
        out[c] = [f'{m * scale:.2f} ± {s * scale:.2f}' if scale == 100 else f'{m:.4f} ± {s:.4f}'
                  for m, s in zip(mean[c], sd[c].fillna(0))]
    out['n_runs'] = g.size()
    return out


def _subject_scores(y, p, subj):
    rows = []
    for s in np.unique(subj):
        m = subj == s
        present = np.unique(y[m])
        f1 = f1_score(y[m], p[m], labels=list(range(5)), average=None, zero_division=0)
        rows.append(dict(subject=s, acc=float((y[m] == p[m]).mean()),
                         mf1=float(f1_score(y[m], p[m], labels=present, average='macro', zero_division=0)),
                         kappa=float(cohen_kappa_score(y[m], p[m])),
                         n1=float(f1[1]) if (y[m] == 1).any() else np.nan, n=int(m.sum())))
    return pd.DataFrame(rows)


def subject_table(C, arm, seed, full_night=False):
    """Per-subject metrics pooled over both nights; Macro-F1 over stages present in that subject."""
    ys, ps, ss = [], [], []
    for f in range(C['K_FOLDS']):
        name = resolve_source(C, arm, f, seed)
        p = os.path.join(run_dir(C, name, f, seed), 'preds_test.npz')
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        k = 'full_' if full_night else ''
        ys.append(d[f'{k}y']); ps.append(d[f'{k}logits'].astype(np.float32).argmax(1))
        ss.append(np.array([r[:5] for r in d[f'{k}rec']]))
    return _subject_scores(np.concatenate(ys), np.concatenate(ps), np.concatenate(ss))


def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def subject_wilcoxon(C, pairs, metrics_=('mf1', 'acc', 'n1'), n_boot=10000, full_night=False):
    """SENSITIVITY analysis: paired per-subject metrics (Wilcoxon + bootstrap of the mean difference)."""
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    rows = []
    for a, b in pairs:
        ta = [subject_table(C, a, s, full_night) for s in C['SEEDS']]
        tb = [subject_table(C, b, s, full_night) for s in C['SEEDS']]
        if any(t is None for t in ta + tb):
            print(f'  (skip {a} vs {b}: runs incomplete)')
            continue
        A = pd.concat(ta).groupby('subject').mean(numeric_only=True)     # mean over seeds per subject
        B = pd.concat(tb).groupby('subject').mean(numeric_only=True)
        for m in metrics_:
            d = (A[m] - B[m]).dropna()
            boot = rng.choice(d.values, size=(n_boot, len(d)), replace=True).mean(1)
            w = wilcoxon(d.values) if (d != 0).any() else None
            rows.append(dict(contrast=f'{a} - {b}', metric=m, n_subjects=len(d),
                             mean_diff_pp=100 * d.mean(), ci95_low_pp=100 * np.percentile(boot, 2.5),
                             ci95_high_pp=100 * np.percentile(boot, 97.5), subjects_better=int((d > 0).sum()),
                             p_wilcoxon=w.pvalue if w is not None else np.nan))
    out = pd.DataFrame(rows)
    if len(out):
        out['q_bh'] = _bh(out['p_wilcoxon'].fillna(1).values)
    return out


# ================================================================ PRIMARY analysis (pre-specified)
# Metric: Macro-F1 of all pooled out-of-fold test predictions (5 folds; pooled over training seeds).
# Inference: subject-cluster bootstrap (whole subjects resampled with replacement, both nights kept together),
# 95% percentile CI of the difference, bootstrap p; BH-FDR within the primary family (one metric x all KD contrasts).
# Secondary metrics (accuracy, kappa, N1-F1) use the same bootstrap and form a separate BH family.
def _cm_metrics(cm):
    """cm: (..., 5, 5) rows = true, cols = predicted."""
    cm = cm.astype(np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rows, cols = cm.sum(-1), cm.sum(-2)
    n = cm.sum((-1, -2))
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(rows + cols > 0, 2 * tp / (rows + cols), 0.0)
        acc = tp.sum(-1) / n
        pe = (rows * cols).sum(-1) / n ** 2
        kappa = (acc - pe) / (1 - pe)
    return dict(mf1=f1.mean(-1), acc=acc, kappa=kappa, n1=f1[..., 1])


def subject_cms(C, arm, full_night=False, exclude_subjects=None):
    """{subject: 5x5 confusion matrix} over all folds and seeds, plus the label vector (to check alignment).
    exclude_subjects: subjects dropped from every arm (sensitivity analyses)."""
    cms, y_ref = {}, None
    k = 'full_' if full_night else ''
    excl = set(exclude_subjects or ())
    for s in C['SEEDS']:
        ys = []
        for f in range(C['K_FOLDS']):
            p = os.path.join(run_dir(C, arm, f, s), 'preds_test.npz')
            if not os.path.exists(p):
                return None, None
            d = np.load(p, allow_pickle=True)
            subj = np.array([r[:5] for r in d[f'{k}rec']])
            keep = ~np.isin(subj, list(excl))
            y = d[f'{k}y'].astype(np.int64)[keep]
            pr = d[f'{k}logits'].astype(np.float32).argmax(1)[keep]
            subj = subj[keep]
            for sj in np.unique(subj):
                m = subj == sj
                cm = np.zeros((5, 5), np.int64)
                np.add.at(cm, (y[m], pr[m]), 1)
                cms[sj] = cms.get(sj, 0) + cm
            ys.append(y)
        y_ref = np.concatenate(ys) if y_ref is None else y_ref
    return cms, y_ref


def cluster_contrasts(C, pairs, n_boot=10000, primary='mf1', full_night=False, seed=0, exclude_subjects=None):
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        A, ya = subject_cms(C, a, full_night, exclude_subjects)
        B, yb = subject_cms(C, b, full_night, exclude_subjects)
        if A is None or B is None:
            print(f'  (skip {a} vs {b}: runs incomplete)')
            continue
        assert np.array_equal(ya, yb), f'{a} and {b} were not evaluated on identical test epochs'
        subs = sorted(A)
        CA, CB = np.stack([A[s] for s in subs]), np.stack([B[s] for s in subs])
        W = rng.multinomial(len(subs), np.full(len(subs), 1 / len(subs)), size=n_boot)   # (n_boot, S) draws
        ma, mb = _cm_metrics(CA.sum(0)), _cm_metrics(CB.sum(0))
        ba = _cm_metrics(np.einsum('bs,sij->bij', W, CA))
        bb = _cm_metrics(np.einsum('bs,sij->bij', W, CB))
        for met in ('mf1', 'acc', 'kappa', 'n1'):
            d = ba[met] - bb[met]
            p = min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (n_boot + 1))
            rows.append(dict(contrast=f'{a} - {b}', metric=met, family='primary' if met == primary else 'secondary',
                             value_a=float(ma[met]), value_b=float(mb[met]), diff_pp=100 * float(ma[met] - mb[met]),
                             ci95_low_pp=100 * float(np.percentile(d, 2.5)),
                             ci95_high_pp=100 * float(np.percentile(d, 97.5)),
                             p_boot=p, n_subjects=len(subs), n_epochs=int(CA.sum())))
    out = pd.DataFrame(rows)
    if len(out):
        out['q_bh'] = np.nan
        for fam in out.family.unique():
            m = out.family == fam
            out.loc[m, 'q_bh'] = _bh(out.loc[m, 'p_boot'].values)
    return out


def tuned_dependent_arms(arms):
    """Arms whose test results depend on the tuned configuration: arms with 'tune' and every arm distilled from one."""
    dep = set()
    for a in arms:                                 # list order = dependency order
        if 'tune' in a or a.get('src') in dep:
            dep.add(a['name'])
    return dep


def hybrid_contrasts(ct_all, ct_excl, dependent):
    """Primary inference (chosen 2026-10-15): a contrast involving a tuned-dependent arm takes its row from the
    analysis without the tuning subjects (ct_excl), every other contrast from the full analysis (ct_all).
    Rows are copied unchanged; BH-FDR is recomputed within each family over the combined set."""
    def uses_dep(c):
        return any(x.strip() in dependent for x in c.split(' - '))
    out = pd.concat([ct_excl[ct_excl.contrast.map(uses_dep)].assign(subject_set='excl. tuning subjects'),
                     ct_all[~ct_all.contrast.map(uses_dep)].assign(subject_set='all subjects')])
    order = {c: i for i, c in enumerate(ct_all.contrast.unique())}
    out = out.sort_values(['family', 'contrast'], key=lambda s: s.map(order) if s.name == 'contrast' else s)
    out['q_bh'] = np.nan
    for fam in out.family.unique():
        m = out.family == fam
        out.loc[m, 'q_bh'] = _bh(out.loc[m, 'p_boot'].values)
    return out.reset_index(drop=True)


def pooled_table(C, arms, full_night=False, exclude_subjects=None):
    """Pooled out-of-fold metrics per arm (the numbers the primary analysis is about)."""
    rows = []
    for a in arms:
        cms, _ = subject_cms(C, a, full_night, exclude_subjects)
        if cms is None:
            continue
        m = _cm_metrics(sum(cms.values()))
        f1 = _cm_f1_per_class(sum(cms.values()))
        rows.append(dict(arm=a, mf1=100 * m['mf1'], acc=100 * m['acc'], kappa=m['kappa'],
                         **{f'f1_{s}': 100 * v for s, v in zip(STAGES, f1)}, n_subjects=len(cms)))
    return pd.DataFrame(rows).set_index('arm')


def _cm_f1_per_class(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    den = cm.sum(1) + cm.sum(0)
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def tuning_table(C):
    rows = []
    seed0 = C['SEEDS'][0]
    for a in C['ARMS']:
        if 'folds' not in a:
            continue
        for f in a['folds']:
            r = load_result(C, a['name'], f, seed0)
            if r is not None:
                rows.append(dict(arm=a['name'], fold=f, lr=r['lr'], ls=r.get('ls'), best_val_mf1=r['best_val_mf1'],
                                 best_epoch=r['best_epoch'], epochs_run=r['epochs_run']))
    return pd.DataFrame(rows)


# ================================================================ cross-dataset (zero-shot)
def eval_cross(C, arm, seed, tgt_store, tgt_recs, tgt_name, device):
    """Each fold model of `arm` (trained in C) predicts every target recording (trimmed targets). Resumable."""
    rows = []
    ds = SeqDS(tgt_store, tgt_recs)
    y = ds.labels()
    for f in range(C['K_FOLDS']):
        odir = run_dir(C, arm, f, seed)
        jp = os.path.join(odir, f'cross_{tgt_name}.json')
        if not os.path.exists(jp):
            r = load_result(C, arm, f, seed)
            if r is None:
                print(f'  (skip {arm} fold{f}: not trained)')
                continue
            model = build_model(r['kind'], r['target_dim']).to(device)
            model.load_state_dict(torch.load(os.path.join(odir, 'best.pt'), map_location=device, weights_only=True))
            logits = predict(model, ds, C, device)
            m = metrics(y, logits.argmax(1))
            np.savez_compressed(os.path.join(odir, f'cross_{tgt_name}.npz'), rec=ds.rec_ids(), center=ds.index[:, 1],
                                y=y, logits=logits.astype(np.float16))
            json.dump(m, open(jp, 'w'), indent=1)
        m = json.load(open(jp))
        rows.append(dict(arm=arm, seed=seed, fold=f, target=tgt_name, acc=m['acc'], mf1=m['mf1'], kappa=m['kappa'],
                         **{f'f1_{s}': v for s, v in zip(STAGES, m['f1'])}))
    return pd.DataFrame(rows)
