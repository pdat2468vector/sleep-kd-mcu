"""XAI v2 on the DREAMS K-complex and sleep-spindle databases (R1 attribution comments, R2.2).

Data facts (DREAMS documentation, Zenodo 10.5281/zenodo.2650142):
  * K-complexes: 10 excerpts of 30 min, healthy subjects, CZ-A1 @ 200 Hz.
  * Spindles: 8 excerpts of 30 min, patients, C3-A1 (excerpts 1, 3) or CZ-A1, 100 / 200 / 50 Hz.
  * Hypnogram_excerptN.txt: one value per 5 s, R&K: 5 W, 4 REM, 3 S1, 2 S2, 1 S3, 0 S4, -1 movement, -2/-3 unknown.
  * Visual_scoring1_excerptN.txt: expert-1 events (onset s, duration s); expert 1 scored every excerpt.
  * excerptN.txt: first line = name of the scored channel.
Preprocessing is identical to training (prep.B_BP filter per epoch, per-epoch per-channel z-score, fp16 storage);
the single scored channel is replicated into the 3 input slots (EEG, EEG, EOG), i.e. out of the training distribution.
"""
import os, re, glob, hashlib, subprocess, urllib.request
import numpy as np, pandas as pd, scipy.signal, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import prep, skd

DBS = ('kcomplexes', 'spindles')
EVENT_NAME = {'kcomplexes': 'K-complex', 'spindles': 'Spindle'}
ZENODO = {'kcomplexes': ('https://zenodo.org/api/records/2650142/files/DatabaseKcomplexes.rar/content',
                         '0bfc5bde8e36bcb1edcded50077423de'),
          'spindles': ('https://zenodo.org/api/records/2650142/files/DatabaseSpindles.rar/content',
                       '2f8a101194e133dd4324f21047dce579')}
RK_TO_AASM = {5: 0, 3: 1, 2: 2, 1: 3, 0: 3, 4: 4}      # S3 + S4 -> N3; -1 / -2 / -3 -> not a stage
HYP_S = 5                                             # seconds per hypnogram value
SIGMA_BAND = (11.0, 16.0)
B_SIG, A_SIG = scipy.signal.butter(N=4, Wn=SIGMA_BAND, btype='bandpass', fs=prep.FS)
CENTER = skd.SEQ_LEN // 2


# ================================================================ data
def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def download_dreams(raw_dir):
    """Download the two DREAMS archives from Zenodo (md5-checked) and extract them. Returns {db: folder}."""
    os.makedirs(raw_dir, exist_ok=True)
    out = {}
    for db, (url, md5) in ZENODO.items():
        rar = os.path.join(raw_dir, f'{db}.rar')
        if not (os.path.exists(rar) and _md5(rar) == md5):
            urllib.request.urlretrieve(url, rar)
        assert _md5(rar) == md5, f'{db}: md5 mismatch'
        dst = os.path.join(raw_dir, db)
        os.makedirs(dst, exist_ok=True)
        subprocess.run(['unrar', 'x', '-o+', '-inul', rar, dst + os.sep], check=True)
        out[db] = find_excerpt_dir(dst)
    return out


def find_excerpt_dir(root):
    hits = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(root, '**', 'excerpt*.edf'), recursive=True)})
    assert len(hits) == 1, f'expected one folder with excerpt*.edf under {root}, got {hits}'
    return hits[0]


def _two_col(path):
    rows = []
    for line in open(path, errors='ignore'):
        parts = line.split()
        if len(parts) >= 2 and not line.lstrip().startswith('['):
            rows.append((float(parts[0]), float(parts[1])))
    return rows


def _hypnogram(path):
    return np.array([int(float(l)) for l in open(path, errors='ignore') if l.strip() and not l.strip().startswith('[')])


def _epoch_stage(hyp_vals):
    """Majority AASM stage of the six 5-s values of one 30-s epoch; -1 if no valid value or a tie."""
    st = [RK_TO_AASM[v] for v in hyp_vals if v in RK_TO_AASM]
    if not st:
        return -1
    c = np.bincount(st, minlength=5)
    return int(c.argmax()) if (c == c.max()).sum() == 1 else -1


def process_excerpt(db, folder, n, out_dir):
    """One excerpt -> <out_dir>/<db>_excerptN.npz in the skd.Store format (+ event mask). Returns a metadata row."""
    import mne
    first = open(os.path.join(folder, f'excerpt{n}.txt'), errors='ignore').readline().strip()
    ch = first.strip('[]').strip()
    raw = mne.io.read_raw_edf(os.path.join(folder, f'excerpt{n}.edf'), preload=False, verbose='ERROR')
    extras = raw._raw_extras[0]
    native = dict(zip(raw.ch_names, extras['n_samps'] / extras['record_length'][0]))
    fs_native = float(native[ch])
    assert fs_native == raw.info['sfreq'], f'{db} {n}: scored channel is not at the file rate'
    x = raw.get_data(picks=[ch])[0] * 1e6                                   # µV at the native rate
    up, down = int(prep.FS), int(fs_native)
    g = np.gcd(up, down)
    if up != down:
        x = scipy.signal.resample_poly(x, up // g, down // g)
    n_ep = len(x) // prep.EP
    x = x[:n_ep * prep.EP].reshape(n_ep, 1, prep.EP)
    x = np.repeat(x, 3, axis=1)                                             # one channel -> 3 input slots
    x = scipy.signal.filtfilt(prep.B_BP, prep.A_BP, x, axis=2)              # identical to training (_finalize)
    x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-6)

    hyp = _hypnogram(os.path.join(folder, f'Hypnogram_excerpt{n}.txt'))
    per = prep.EPOCH_S // HYP_S
    y = np.array([_epoch_stage(hyp[i * per:(i + 1) * per]) for i in range(n_ep)], dtype=np.int64)

    events = _two_col(os.path.join(folder, f'Visual_scoring1_excerpt{n}.txt'))
    mask = np.zeros(n_ep * prep.EP, bool)
    for on, du in events:
        a, b = int(round(on * prep.FS)), int(round((on + du) * prep.FS))
        mask[max(a, 0):min(b, len(mask))] = True
    mask = mask.reshape(n_ep, prep.EP)
    ev_ep = np.where(mask.any(1) & ~mask.all(1))[0]                          # targets: epochs with an event

    rid = f'{db}_excerpt{n}'
    np.savez(os.path.join(out_dir, f'{rid}.npz'), x=x.astype(np.float16), y=y, mask=mask,
             centers_trim=ev_ep.astype(np.int64), centers_full=np.arange(n_ep, dtype=np.int64))
    stages = {s: int((y == i).sum()) for i, s in enumerate(skd.STAGES)}
    return dict(rec=rid, db=db, excerpt=int(n), channel=ch, fs_native=fs_native, resample=f'{up // g}/{down // g}',
                n_samples_native=int(raw.n_times), n_epochs=n_ep, n_events=len(events), n_event_epochs=len(ev_ep),
                event_samples_frac=float(mask[ev_ep].mean()) if len(ev_ep) else np.nan,
                hyp_values=len(hyp), **{f'stage_{s}': v for s, v in stages.items()}, stage_unscored=int((y < 0).sum()))


def process_all(dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for db in DBS:
        nums = sorted(int(m.group(1)) for p in glob.glob(os.path.join(dirs[db], 'excerpt*.edf'))
                      if (m := re.search(r'excerpt(\d+)\.edf$', p)))
        rows += [process_excerpt(db, dirs[db], k, out_dir) for k in nums]
    return pd.DataFrame(rows)


# ================================================================ attribution
def _target_logit(model, x, cls):
    return model(x)['logits'].gather(1, cls[:, None]).squeeze(1)


def ig_grid(kind, steps, lo=1e-6):
    """Path points alpha in [0, 1] and trapezoid weights. 'uniform': midpoint rule (the usual IG implementation).
    'log': 0 plus log-spaced points 1e-6..1 — the models normalise early (GroupNorm / LayerNorm), so the target logit
    changes almost entirely at very small alpha, which a uniform grid does not resolve (checked: completeness)."""
    if kind == 'uniform':
        return (np.arange(steps) + 0.5) / steps, np.full(steps, 1.0 / steps)
    a = np.concatenate([[0.0], np.logspace(np.log10(lo), 0, steps - 1)])
    w = np.zeros_like(a)
    d = np.diff(a)
    w[:-1] += d / 2
    w[1:] += d / 2
    return a, w


def attribute(model, x, ig_steps=64, ig_kind='log', ig_lo=1e-6):
    """x: (B, 21, 3, 3000) float32 on device. Target = logit of the class predicted for the clean input.
    Returns dict of (B, 3000) maps for the CENTRE epoch (|.| summed over the 3 input slots) + IG completeness
    |sum(IG) - (F(x) - F(0))| / |F(x) - F(0)| (sum over the whole 21-epoch input)."""
    use_cudnn = not hasattr(model, 'gru')            # cuDNN RNN backward needs train mode; we keep eval mode
    with torch.backends.cudnn.flags(enabled=use_cudnn):
        with torch.no_grad():
            logits = model(x)['logits']
            cls = logits.argmax(1)
            f_x = logits.gather(1, cls[:, None]).squeeze(1)
            f_0 = _target_logit(model, torch.zeros_like(x), cls)
        xg = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(_target_logit(model, xg, cls).sum(), xg)
        grad_map = g[:, CENTER].abs().sum(1)
        acc = torch.zeros_like(x)
        for a, w in zip(*ig_grid(ig_kind, ig_steps, ig_lo)):  # zero baseline
            xa = (x * float(a)).requires_grad_(True)
            gk, = torch.autograd.grad(_target_logit(model, xa, cls).sum(), xa)
            acc += float(w) * gk
        ig = x * acc
        ig_map = ig[:, CENTER].abs().sum(1)
        completeness = (ig.sum((1, 2, 3)) - (f_x - f_0)).abs() / ((f_x - f_0).abs() + 1e-8)
    return dict(gradient=grad_map.detach().cpu().numpy(), ig=ig_map.detach().cpu().numpy(),
                pred=cls.cpu().numpy(), ig_rel_err=completeness.detach().cpu().numpy())


def baselines(xc):
    """Model-free maps on the centre epoch actually given to the models (first input slot). xc: (B, 3000)."""
    env = np.abs(scipy.signal.hilbert(xc, axis=1))
    sig = np.abs(scipy.signal.hilbert(scipy.signal.filtfilt(B_SIG, A_SIG, xc, axis=1), axis=1))
    return {'abs_x': np.abs(xc), 'envelope': env, 'sigma_power': sig ** 2}


def _auroc(maps, mask):
    return np.array([roc_auc_score(m, s) for s, m in zip(maps, mask)])


def spec_ready(runs_dir, arm, k_folds, seed):
    """True when every fold model of this configuration has been trained (result.json + best.pt present)."""
    C = dict(RUNS_DIR=runs_dir)
    return all(os.path.exists(os.path.join(skd.run_dir(C, arm, f, seed), n))
               for f in range(k_folds) for n in ('result.json', 'best.pt'))


def load_model(runs_dir, arm, fold, seed, device):
    C = dict(RUNS_DIR=runs_dir)
    res = skd.load_result(C, arm, fold, seed)
    assert res is not None, f'{runs_dir}: {arm} fold{fold} seed{seed} not trained'
    m = skd.build_model(res['kind'], res['target_dim']).to(device)
    m.load_state_dict(torch.load(os.path.join(skd.run_dir(C, arm, fold, seed), 'best.pt'), map_location=device,
                                 weights_only=True))
    return m.eval(), res['kind']


def random_model(kind, seed, device):
    torch.manual_seed(seed)
    return skd.build_model(kind).to(device).eval()


def run_model(model, store, recs, mask_of, device, batch=32, ig_steps=64, ig_lo=1e-6):
    """Per-epoch AUROC rows for every target epoch of every excerpt."""
    rows = []
    for rec in recs:
        ds = skd.SeqDS(store, [rec])
        centers = ds.index[:, 1]
        out = {'gradient': [], 'ig': [], 'pred': [], 'ig_rel_err': []}
        for xb, _, _ in DataLoader(ds, batch_size=batch, shuffle=False):
            a = attribute(model, xb.to(device), ig_steps, 'log', ig_lo)
            for k in out:
                out[k].append(a[k])
        out = {k: np.concatenate(v) for k, v in out.items()}
        m = mask_of[rec][centers]
        y = store.recs[store.pos[rec]]['y'][centers]
        for meth in ('gradient', 'ig'):
            au = _auroc(out[meth], m)
            rows += [dict(rec=rec, epoch=int(c), method=meth, auroc=float(v), pred=int(p), stage=int(s),
                          ig_rel_err=float(e)) for c, v, p, s, e in zip(centers, au, out['pred'], y, out['ig_rel_err'])]
    return pd.DataFrame(rows)


def run_baselines(store, recs, mask_of):
    rows = []
    for rec in recs:
        r = store.recs[store.pos[rec]]
        centers = r['trim']
        xc = r['x'][centers, 0].astype(np.float32)
        m = mask_of[rec][centers]
        for name, maps in baselines(xc).items():
            rows += [dict(rec=rec, epoch=int(c), method=name, auroc=float(v)) for c, v in zip(centers, _auroc(maps, m))]
    return pd.DataFrame(rows)


# ================================================================ statistics (unit = excerpt; plan section 0.4)
def excerpt_level(df, by):
    """Mean AUROC over target epochs within an excerpt, then mean over the fold / random models."""
    e = df.groupby(by + ['model_id', 'rec'], as_index=False).auroc.mean()
    return e.groupby(by + ['rec'], as_index=False).auroc.mean()


def _boot_median_ci(v, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def wilcoxon_row(d, **info):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    lo, hi = _boot_median_ci(d)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(**info, n_excerpts=len(d), median=float(np.median(d)), ci95_low=float(lo), ci95_high=float(hi),
                n_positive=int((d > 0).sum()), p_wilcoxon=float(p))


def bh_by_family(tab):
    tab = tab.copy()
    tab['q_bh'] = np.nan
    for fam, idx in tab.groupby('family').groups.items():
        tab.loc[idx, 'q_bh'] = skd._bh(tab.loc[idx, 'p_wilcoxon'].values)
    return tab


BASELINES = {'kcomplexes': ['abs_x', 'envelope'], 'spindles': ['abs_x', 'envelope', 'sigma_power']}
METHODS = ('gradient', 'ig')


def planned_tests(specs, random_of):
    """The complete list of tests, fixed before any model is evaluated (plan section 0.4).
    Q1: trained-model AUROC vs 0.5. Q2: trained model vs each model-free baseline. Q3: trained vs random-init.
    One BH family per (question, event database)."""
    rows = []
    for db in DBS:
        for s in specs:
            for m in METHODS:
                rows.append(dict(family=f'Q1|{db}', db=db, spec=s, method=m, versus='chance (0.5)'))
                rows += [dict(family=f'Q2|{db}', db=db, spec=s, method=m, versus=b) for b in BASELINES[db]]
                rows.append(dict(family=f'Q3|{db}', db=db, spec=s, method=m, versus=random_of[s]))
    return pd.DataFrame(rows)


def run_tests(plan, exc_models, exc_base):
    """exc_models: excerpt-level AUROC with columns spec, method, db, rec, auroc (trained + random specs);
    exc_base: same for baselines (spec = baseline name, method = baseline name)."""
    M = exc_models.set_index(['spec', 'method', 'db', 'rec']).auroc
    B = exc_base.set_index(['method', 'db', 'rec']).auroc
    out = []
    for t in plan.itertuples():
        a = M.loc[(t.spec, t.method, t.db)]
        if t.versus == 'chance (0.5)':
            d = a - 0.5
        elif t.versus in B.index.get_level_values(0):
            d = a - B.loc[(t.versus, t.db)].reindex(a.index)
        else:
            d = a - M.loc[(t.versus, t.method, t.db)].reindex(a.index)
        assert d.notna().all() and len(d) == (10 if t.db == 'kcomplexes' else 8), (t, len(d))
        out.append(wilcoxon_row(d.values, family=t.family, db=t.db, spec=t.spec, method=t.method, versus=t.versus,
                                median_auroc=float(a.median())))
    return bh_by_family(pd.DataFrame(out))


def completeness_check(model, store, rec, device, n=32):
    """Median IG completeness error on the first n target epochs of one excerpt, uniform-50 vs log-64 grids."""
    ds = skd.SeqDS(store, [rec])
    x = torch.stack([ds[i][0] for i in range(min(n, len(ds)))]).to(device)
    return {f'{k}{s}': float(np.median(attribute(model, x, s, k)['ig_rel_err'])) for k, s in [('uniform', 50), ('log', 64)]}


def figure_epoch(store, mask_of, db, meta):
    """Pre-specified example: the excerpt with the most expert events; in it, the target epoch with the largest
    event fraction (ties -> earliest)."""
    rec = meta[meta.db == db].sort_values(['n_events', 'excerpt'], ascending=[False, True]).rec.iloc[0]
    r = store.recs[store.pos[rec]]
    frac = mask_of[rec][r['trim']].mean(1)
    return rec, int(r['trim'][int(np.argmax(frac))])


def plot_auroc(exc_m, exc_b, specs, random_specs, out_png):
    """Fig. 2b: excerpt-level AUROC (one dot per excerpt, bar = median) for every trained model, the randomly
    initialised models and the model-free baselines; one panel per event type; gradient and IG side by side."""
    import matplotlib.pyplot as plt
    rows = [(s, m) for s in list(specs) + list(random_specs) for m in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, db in zip(axes, DBS):
        items = [(f'{s}\n{m}', exc_m[(exc_m.spec == s) & (exc_m.method == m) & (exc_m.db == db)].auroc.values)
                 for s, m in rows]
        items += [(b, exc_b[(exc_b.method == b) & (exc_b.db == db)].auroc.values) for b in BASELINES[db]]
        for i, (lab, v) in enumerate(items):
            col = 'tab:gray' if 'random' in lab else ('tab:green' if '\n' not in lab else
                                                      ('tab:blue' if lab.endswith('ig') else 'tab:orange'))
            ax.bar(i, np.median(v), color=col, alpha=0.35, width=0.7)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), v, s=8, color=col, zorder=3)
        ax.axhline(0.5, color='k', lw=0.8, ls='--')
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([l for l, _ in items], rotation=90, fontsize=6)
        ax.set_title(f'{EVENT_NAME[db]} (n = {len(items[0][1])} excerpts)', fontsize=9)
    axes[0].set_ylabel('AUROC (excerpt mean over fold models)', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.replace('.png', '.pdf'))
    return fig


def plot_examples(panels, out_png):
    """panels: list of dict(title, x (3000,), mask (3000,), maps {label: (3000,)}). Time axis in seconds."""
    import matplotlib.pyplot as plt
    t = np.arange(prep.EP) / prep.FS
    nrow = max(len(p['maps']) for p in panels) + 1
    fig, axes = plt.subplots(nrow, len(panels), figsize=(6.5 * len(panels), 1.25 * nrow), sharex=True, squeeze=False)
    for j, p in enumerate(panels):
        spans = np.flatnonzero(np.diff(np.r_[0, p['mask'].astype(int), 0]))
        for i, (lab, y) in enumerate([('Input (z-scored)', p['x'])] + list(p['maps'].items())):
            ax = axes[i, j]
            for a, b in zip(spans[::2], spans[1::2]):
                ax.axvspan(a / prep.FS, b / prep.FS, color='tab:orange', alpha=0.25, lw=0)
            ax.plot(t, y, lw=0.6, color='k' if i == 0 else 'tab:blue')
            ax.set_ylabel(lab, fontsize=7, rotation=0, ha='right', va='center')
            ax.tick_params(labelsize=7)
            ax.set_yticks([])
        axes[0, j].set_title(p['title'], fontsize=9)
        axes[-1, j].set_xlabel('Time within the 30-s epoch (s)', fontsize=8)
        axes[-1, j].set_xticks(np.arange(0, 31, 5))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.replace('.png', '.pdf'))
    return fig
