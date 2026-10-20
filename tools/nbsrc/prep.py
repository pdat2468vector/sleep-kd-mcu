# prep.py — download + preprocess Sleep-EDF Expanded (v1.0.0) and HMC (v1.1) per recording; subject-level folds.
import os, re, glob, random, time
import numpy as np
import pandas as pd
import scipy.signal
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

PHYSIONET = 'https://physionet.org/files/sleep-edfx/1.0.0'
S3_BUCKET = 'https://physionet-open.s3.amazonaws.com'          # official AWS Open Data mirror (much faster)
S3_PREFIX = 'sleep-edfx/1.0.0'
SUBSETS = ['sleep-cassette', 'sleep-telemetry']
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']   # slot 0, 1 = EEG; slot 2 = EOG
FS, EPOCH_S = 100, 30
EP = FS * EPOCH_S
STAGE_MAP = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
             'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}   # M / ? -> -1
# Same filter as the original preprocess.py: Butterworth order 5, 0.5-35 Hz, zero-phase, per epoch
B_BP, A_BP = scipy.signal.butter(N=5, Wn=[0.5, 35.0], btype='bandpass', fs=FS)

# HMC (Haaglanden Medisch Centrum sleep staging database), same channel mapping as the original extract_hmc_to_numpy.py
HMC_FS = 256
HMC_CHANNELS = {'F4': ['EEG F4-M1', 'EEG F4-A1'], 'O2': ['EEG O2-M1', 'EEG O2-A1'],
                'E1': ['EOG E1-M2', 'EOG E1-A2'], 'E2': ['EOG E2-M2', 'EOG E2-A2']}   # slots: F4-M1, O2-M1, E1-E2
HMC_STAGES = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, 'Sleep stage N3': 3, 'Sleep stage R': 4}


# ---------------------------------------------------------------- download
def _list_edf(url):
    html = requests.get(url, timeout=120).text
    return sorted(set(re.findall(r'href="([^"/?]+\.edf)"', html)))


def _fetch(url, path, retries=4):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 'skip'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = path + '.part'
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(tmp, path)
            return 'ok'
        except Exception as e:
            if attempt == retries - 1:
                return f'FAIL {e}'
            time.sleep(5 * (attempt + 1))


def _s3_keys(prefix):
    keys, token = [], None
    while True:
        params = {'list-type': 2, 'prefix': prefix}
        if token:
            params['continuation-token'] = token
        xml = requests.get(S3_BUCKET + '/', params=params, timeout=120).text
        keys += re.findall(r'<Key>([^<]+)</Key>', xml)
        m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', xml)
        if not m:
            return keys
        token = m.group(1)


def _list_s3(sub):
    return [k.rsplit('/', 1)[1] for k in _s3_keys(f'{S3_PREFIX}/{sub}/') if k.endswith('.edf')]


def download_sleep_edfx(raw_dir, n_threads=8, source='s3'):
    jobs = []
    for sub in SUBSETS:
        if source == 's3':
            base, names = f'{S3_BUCKET}/{S3_PREFIX}/{sub}/', _list_s3(sub)
        else:
            base, names = f'{PHYSIONET}/{sub}/', _list_edf(f'{PHYSIONET}/{sub}/')
        jobs += [(base + f, os.path.join(raw_dir, sub, f)) for f in names]
    print(f'{len(jobs)} EDF files listed ({source})')
    status = {}
    with ThreadPoolExecutor(n_threads) as ex:
        futs = {ex.submit(_fetch, u, p): p for u, p in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            status[futs[fu]] = fu.result()
            if k % 50 == 0:
                print(f'  {k}/{len(jobs)}')
    fails = {p: s for p, s in status.items() if s.startswith('FAIL')}
    print(f'done | failed: {len(fails)}')
    for p, s in fails.items():
        print('  ', p, s)
    return fails


def pair_recordings(raw_dir):
    psgs = sorted(glob.glob(os.path.join(raw_dir, '**', '*-PSG.edf'), recursive=True))
    hyps = {os.path.basename(h)[:7]: h
            for h in glob.glob(os.path.join(raw_dir, '**', '*-Hypnogram.edf'), recursive=True)}
    pairs = []
    for p in psgs:
        rec = os.path.basename(p).split('-')[0]            # e.g. SC4001E0
        h = hyps.get(rec[:7])
        if h is None:
            print('no hypnogram for', rec)
            continue
        pairs.append((rec, p, h))
    return pairs


# ---------------------------------------------------------------- shared epoch processing
def subject_of(rec):
    return rec[:5]                                          # SC4ss / ST7ss


def _finalize(rec, x, y, out_dir, trim_wake_min, store_full_night, reject_ptp_uv, pad):
    """x: (n,3,3000) raw µV @100 Hz, y: (n,) int8 (-1 = not a target). Filter, z-score, trim, save.
    Returns the per-recording summary fields shared by all datasets."""
    n = len(y)
    ptp = np.ptp(x, axis=2).max(axis=1).astype(np.float32)  # raw µV, max over channels (reported, optional exclusion)
    x = scipy.signal.filtfilt(B_BP, A_BP, x, axis=2)
    x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-6)

    labelled = y >= 0
    if reject_ptp_uv is not None:
        labelled &= ptp <= reject_ptp_uv
    sleep = np.where((y >= 1) & (y <= 4))[0]
    if len(sleep) == 0:
        raise ValueError(f'{rec}: no sleep epochs')
    if trim_wake_min is None:
        ts, te = 0, n
    else:
        m = int(trim_wake_min * 60 / EPOCH_S)
        ts, te = max(0, sleep[0] - m), min(n, sleep[-1] + m + 1)
    centers_full = np.where(labelled)[0]
    centers_trim = centers_full[(centers_full >= ts) & (centers_full < te)]

    if store_full_night:
        lo, hi = 0, n
    else:                                                    # keep only the trimmed span + context margin
        lo, hi = max(0, ts - pad), min(n, te + pad)
        centers_trim = centers_trim - lo
        centers_full = np.zeros(0, dtype=np.int64)
    np.savez(os.path.join(out_dir, f'{rec}.npz'), x=x[lo:hi].astype(np.float16), y=y[lo:hi], ptp=ptp[lo:hi],
             centers_trim=centers_trim.astype(np.int64), centers_full=centers_full.astype(np.int64),
             lo=lo, n_total=n)

    yt = y[ts:te][labelled[ts:te]]
    yf = y[labelled]
    row = dict(n_epochs_total=n, trim_start=ts, trim_end=te, n_stored=hi - lo, n_unlabelled=int((y < 0).sum()),
               n_ptp_over_150uv=int((ptp > 150).sum()))
    for c, nm in enumerate(['W', 'N1', 'N2', 'N3', 'REM']):
        row[f'trim_{nm}'] = int((yt == c).sum())
        row[f'full_{nm}'] = int((yf == c).sum())
    return row


# ---------------------------------------------------------------- Sleep-EDF
def process_recording(rec, psg, hyp, out_dir, trim_wake_min=30, store_full_night=False,
                      reject_ptp_uv=None, pad=10):
    import mne
    raw = mne.io.read_raw_edf(psg, preload=False, verbose='ERROR')
    missing = [c for c in CHANNELS if c not in raw.ch_names]
    if missing:
        raise ValueError(f'{rec}: missing channels {missing}')
    raw.pick(CHANNELS)
    raw.reorder_channels(CHANNELS)
    raw.load_data(verbose='ERROR')
    sfreq_orig = float(raw.info['sfreq'])
    if sfreq_orig != FS:
        raw.resample(FS, verbose='ERROR')
    raw.set_annotations(mne.read_annotations(hyp), emit_warning=False)

    data = raw.get_data() * 1e6                             # volts -> microvolts, (3, N)
    n = data.shape[1] // EP
    x = data[:, :n * EP].reshape(3, n, EP).transpose(1, 0, 2).astype(np.float32)
    y = np.full(n, -1, dtype=np.int8)
    for on, du, de in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        s = int(round((on - raw.first_time) / EPOCH_S))
        e = s + int(round(du / EPOCH_S))
        y[max(s, 0):min(e, n)] = STAGE_MAP.get(de, -1)

    row = dict(rec=rec, subject=subject_of(rec), cohort=rec[:2], night=int(rec[5]), sfreq_orig=sfreq_orig)
    row.update(_finalize(rec, x, y, out_dir, trim_wake_min, store_full_night, reject_ptp_uv, pad))
    return row


def preprocess_all(raw_dir, out_dir, meta_csv, **kw):
    os.makedirs(out_dir, exist_ok=True)
    old = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else pd.DataFrame()
    done = set(old['rec']) if len(old) else set()
    rows = old.to_dict('records') if len(old) else []
    pairs = pair_recordings(raw_dir)
    print(f'{len(pairs)} recordings paired | already processed: {len(done)}')
    for k, (rec, psg, hyp) in enumerate(pairs, 1):
        if rec in done and os.path.exists(os.path.join(out_dir, f'{rec}.npz')):
            continue
        t0 = time.time()
        rows.append(process_recording(rec, psg, hyp, out_dir, **kw))
        pd.DataFrame(rows).to_csv(meta_csv, index=False)     # checkpoint progress
        print(f'  [{k}/{len(pairs)}] {rec} ({time.time() - t0:.1f}s)')
    meta = pd.DataFrame(rows).sort_values('rec').reset_index(drop=True)
    meta.to_csv(meta_csv, index=False)
    return meta


# ---------------------------------------------------------------- HMC
def parse_hmc_scoring(txt_path):
    """(onset_s, label) for every 30-s 'Sleep stage *' row of SNxxx_sleepscoring.txt (lights on/off etc. ignored)."""
    out = []
    for line in open(txt_path, encoding='utf-8', errors='ignore').read().splitlines()[1:]:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5 and parts[4] in HMC_STAGES and abs(float(parts[3]) - EPOCH_S) < 0.1:
            out.append((float(parts[2]), HMC_STAGES[parts[4]]))
    return sorted(out)


def process_hmc_recording(rec, edf, txt, out_dir, trim_wake_min=30, store_full_night=False,
                          reject_ptp_uv=None, pad=10):
    import mne
    raw = mne.io.read_raw_edf(edf, preload=False, verbose='ERROR')
    names = {c.strip().lower(): c for c in raw.ch_names}
    pick = {}
    for slot, cands in HMC_CHANNELS.items():
        hit = [names[c.lower()] for c in cands if c.lower() in names]
        if not hit:
            raise ValueError(f'{rec}: missing {slot} ({cands}); available {raw.ch_names}')
        pick[slot] = hit[0]
    sfreq_orig = float(raw.info['sfreq'])
    if abs(sfreq_orig - HMC_FS) > 0.5:
        raise ValueError(f'{rec}: sfreq {sfreq_orig}')
    raw.pick([pick[s] for s in ('F4', 'O2', 'E1', 'E2')])
    d = raw.get_data(picks=[pick[s] for s in ('F4', 'O2', 'E1', 'E2')]) * 1e6
    sig = np.stack([d[0], d[1], d[2] - d[3]])               # F4-M1, O2-M1, E1-E2 (derived horizontal EOG)

    stages = parse_hmc_scoring(txt)
    ep256 = HMC_FS * EPOCH_S
    segs, labs, onsets = [], [], []
    for on, lab in stages:
        s = int(round(on * HMC_FS))
        if s >= 0 and s + ep256 <= sig.shape[1]:
            segs.append(sig[:, s:s + ep256]); labs.append(lab); onsets.append(on)
    x = scipy.signal.resample_poly(np.stack(segs), up=25, down=64, axis=2).astype(np.float32)   # 256 -> 100 Hz
    assert x.shape[2] == EP
    y = np.array(labs, dtype=np.int8)
    n_gaps = int((np.diff(onsets) != EPOCH_S).sum())         # non-contiguous scored epochs (context would jump)

    row = dict(rec=rec, subject=rec, cohort='HMC', night=1, sfreq_orig=sfreq_orig, n_gaps=n_gaps,
               channels='|'.join(pick[s] for s in ('F4', 'O2', 'E1', 'E2')))
    row.update(_finalize(rec, x, y, out_dir, trim_wake_min, store_full_night, reject_ptp_uv, pad))
    return row


def list_hmc(version='1.1'):
    """Recording ids (SNxxx) that have both the PSG EDF and the scoring txt on the S3 mirror."""
    keys = _s3_keys(f'hmc-sleep-staging/{version}/recordings/')
    edf = {m.group(1) for k in keys for m in [re.search(r'/(SN\d+)\.edf$', k)] if m}
    txt = {m.group(1) for k in keys for m in [re.search(r'/(SN\d+)_sleepscoring\.txt$', k)] if m}
    recs = sorted(edf & txt)
    if not recs:
        raise RuntimeError(f'no HMC recordings found for version {version} ({len(keys)} keys listed)')
    return recs


def download_process_hmc(raw_dir, out_dir, meta_csv, version='1.1', n_threads=4, keep_raw=False, **kw):
    """Download each recording from the S3 mirror, process it, delete the raw EDF (unless keep_raw). Resumable."""
    os.makedirs(out_dir, exist_ok=True)
    prefix = f'hmc-sleep-staging/{version}/recordings/'
    recs = list_hmc(version)
    old = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else pd.DataFrame()
    rows = old.to_dict('records') if len(old) else []
    done = {r['rec'] for r in rows if os.path.exists(os.path.join(out_dir, f"{r['rec']}.npz"))}
    todo = [r for r in recs if r not in done]
    print(f'HMC v{version}: {len(recs)} recordings | already processed: {len(done)} | to do: {len(todo)}')

    def get(rec):
        for f in (f'{rec}.edf', f'{rec}_sleepscoring.txt'):
            st = _fetch(f'{S3_BUCKET}/{prefix}{f}', os.path.join(raw_dir, f))
            if st.startswith('FAIL'):
                return st
        return 'ok'

    with ThreadPoolExecutor(n_threads) as ex:                # downloads run ahead of processing
        futs = [(rec, ex.submit(get, rec)) for rec in todo]
        for k, (rec, fu) in enumerate(futs, 1):
            st = fu.result()
            if st != 'ok':
                print(f'  [{k}/{len(todo)}] {rec}: download {st}')
                continue
            t0 = time.time()
            edf, txt = os.path.join(raw_dir, f'{rec}.edf'), os.path.join(raw_dir, f'{rec}_sleepscoring.txt')
            rows = [r for r in rows if r['rec'] != rec] + [process_hmc_recording(rec, edf, txt, out_dir, **kw)]
            pd.DataFrame(rows).to_csv(meta_csv, index=False)
            if not keep_raw:
                os.remove(edf)
            print(f'  [{k}/{len(todo)}] {rec} ({time.time() - t0:.1f}s)')
    meta = pd.DataFrame(rows).sort_values('rec').reset_index(drop=True)
    meta.to_csv(meta_csv, index=False)
    return meta


# ---------------------------------------------------------------- subject-level folds
def make_subject_folds(meta, k=5, seed=42):
    """Shuffle subjects (seed) within each cohort, assign round-robin -> every night of a subject in one fold.
    Cohorts are processed in sorted order (Sleep-EDF: SC then ST; HMC: one cohort) with a single RNG."""
    rng = random.Random(seed)
    fold_of = {}
    for cohort in sorted(meta.cohort.unique()):
        subs = sorted(meta.loc[meta.cohort == cohort, 'subject'].unique())
        rng.shuffle(subs)
        for j, s in enumerate(subs):
            fold_of[s] = j % k
    meta = meta.copy()
    meta['fold'] = meta['subject'].map(fold_of)
    assert meta.groupby('subject')['fold'].nunique().max() == 1, 'subject spans >1 fold'
    folds = {f: sorted(meta.loc[meta.fold == f, 'rec']) for f in range(k)}
    return meta, folds


def fold_role_lists(meta, k=5):
    """Explicit train / validation / test subjects and recordings for every CV fold (test = i, val = i+1 mod k)."""
    out = {}
    for i in range(k):
        roles = {'test': [i], 'validation': [(i + 1) % k], 'train': [j for j in range(k) if j not in (i, (i + 1) % k)]}
        out[f'fold{i}'] = {r: dict(source_folds=fs,
                                   subjects=sorted(meta.loc[meta.fold.isin(fs), 'subject'].unique().tolist()),
                                   recordings=sorted(meta.loc[meta.fold.isin(fs), 'rec'].tolist()))
                           for r, fs in roles.items()}
    return out
