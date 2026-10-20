"""Generate the Sleep-EDF result tables of the manuscript from the exported Colab runs (no hand-copied numbers).
usage: python make_results_tables.py <export_dir> <hybrid_csv> <revision.tex>
Every table sits in revision.tex between '% BEGIN GENERATED <name>' and '% END GENERATED <name>'; this script
rewrites those blocks only. Numbers: pooled out-of-fold predictions of all 100 subjects (descriptive), per-arm
subject-cluster bootstrap 95% CI of Macro-F1 (10,000 resamples, seed 0); contrasts = primary analysis (hybrid 100/79)
read from <hybrid_csv> (tools/run_hybrid_local.py / notebook cell 3f).
(PYTHONPATH must contain the code dir written by build_notebook.py)"""
import os, re, sys
import numpy as np, pandas as pd
import skd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_names import NAME, GROUPS   # noqa: E402

EXP, HYB, TEX = sys.argv[1:4]
C = dict(K_FOLDS=5, RUNS_DIR=os.path.join(EXP, 'runs'), SEEDS=[42])
MAIN = [('Teacher', 'T_ce'), ('Linear Assistant', 'A_kd_Tce'), ('Edge Student', 'E_staged_Tce')]


def arm_stats(arm, n_boot=10000, seed=0):
    cms, _ = skd.subject_cms(C, arm)
    subs = sorted(cms)
    CM = np.stack([cms[s] for s in subs])
    tot = CM.sum(0)
    m = skd._cm_metrics(tot)
    f1 = skd._cm_f1_per_class(tot)
    W = np.random.default_rng(seed).multinomial(len(subs), np.full(len(subs), 1 / len(subs)), size=n_boot)
    b = skd._cm_metrics(np.einsum('bs,sij->bij', W, CM))['mf1']
    return dict(mf1=100 * m['mf1'], lo=100 * np.percentile(b, 2.5), hi=100 * np.percentile(b, 97.5),
                acc=100 * m['acc'], kappa=m['kappa'], f1=100 * np.asarray(f1), n_sub=len(subs), n_ep=int(tot.sum()))


S = {a: arm_stats(a) for _, arms in GROUPS for a in arms}
pooled = pd.read_csv(os.path.join(EXP, 'results', 'pooled_oof_metrics.csv')).set_index('arm')
for a, s in S.items():   # must equal the Colab pooled table (cell 3b)
    assert abs(s['mf1'] - pooled.loc[a, 'mf1']) < 1e-9 and abs(s['acc'] - pooled.loc[a, 'acc']) < 1e-9, a
    assert s['n_sub'] == 100, a
print('per-arm pooled metrics == Colab pooled_oof_metrics.csv; n_epochs', {s['n_ep'] for s in S.values()})
assert 0 <= S['T_ce']['mf1'] - S['E_staged_Tce']['mf1'] < 0.6     # Introduction: "within 0.6 percentage points"


def f2(v):
    return f'{v:.2f}'


def ci(s):
    return f"{f2(s['mf1'])} [{f2(s['lo'])}, {f2(s['hi'])}]"


blocks = {}
rows = [f"{lab} & {ci(S[a])} & {f2(S[a]['acc'])} & {S[a]['kappa']:.3f} & {f2(S[a]['f1'][1])} \\\\" for lab, a in MAIN]
blocks['overall_perf'] = '\n'.join([r'\begin{tabular}{lcccc}', r'\toprule',
    r"\textbf{Model} & \textbf{Macro-F1 (\%) [95\% CI]} & \textbf{Accuracy (\%)} & \textbf{Cohen's $\kappa$} & \textbf{N1-F1 (\%)} \\",
    r'\midrule', *rows, r'\bottomrule', r'\end{tabular}'])

rows = []
for g, arms in GROUPS:
    rows.append(rf'\multicolumn{{5}}{{l}}{{\textit{{{g}}}}} \\')
    rows += [f"\\quad {NAME[a]} & {ci(S[a])} & {f2(S[a]['acc'])} & {S[a]['kappa']:.3f} & {f2(S[a]['f1'][1])} \\\\"
             for a in arms]
blocks['ablation_student'] = '\n'.join([r'\begin{tabular}{lcccc}', r'\toprule',
    r"\textbf{Training variant} & \textbf{Macro-F1 (\%) [95\% CI]} & \textbf{Accuracy (\%)} & \textbf{Cohen's $\kappa$} & \textbf{N1-F1 (\%)} \\",
    r'\midrule', *rows, r'\bottomrule', r'\end{tabular}'])

rows = [f"{st} & " + ' & '.join(f2(S[a]['f1'][i]) for _, a in MAIN) + r' \\'
        for i, st in enumerate(['Wake (W)', 'N1', 'N2', 'N3', 'REM'])]
blocks['per_stage_f1'] = '\n'.join([r'\begin{tabular}{lccc}', r'\toprule',
    r'\textbf{Stage} & \textbf{Teacher} & \textbf{Linear Assistant} & \textbf{Edge Student} \\', r'\midrule',
    *rows, r'\bottomrule', r'\end{tabular}'])

H = pd.read_csv(HYB)
SHORT = {a: NAME[a] for a in NAME}


def lab(contrast):
    a, b = [x.strip() for x in contrast.split(' - ')]
    return f'{SHORT[a]} $-$ {SHORT[b]}'


def q(v):
    return '$<$0.001' if v < 0.001 else f'{v:.3f}'


rows = []
for c in H.contrast.unique():
    p = H[(H.contrast == c) & (H.metric == 'mf1')].iloc[0]
    n1 = H[(H.contrast == c) & (H.metric == 'n1')].iloc[0]
    rows.append(f"{lab(c)} & {int(p.n_subjects)} & ${p.diff_pp:+.2f}$ [${p.ci95_low_pp:+.2f}$, ${p.ci95_high_pp:+.2f}$] & "
                f"{q(p.q_bh)} & ${n1.diff_pp:+.2f}$ [${n1.ci95_low_pp:+.2f}$, ${n1.ci95_high_pp:+.2f}$] & {q(n1.q_bh)} \\\\")
blocks['contrasts'] = '\n'.join([r'\begin{tabular}{lccccc}', r'\toprule',
    r'\textbf{Contrast (A $-$ B)} & \textbf{$n$} & \textbf{$\Delta$Macro-F1 (pp) [95\% CI]} & \textbf{$q$} & \textbf{$\Delta$N1-F1 (pp) [95\% CI]} & \textbf{$q$} \\',
    r'\midrule', *rows, r'\bottomrule', r'\end{tabular}'])

# ---------------------------------------------------------------- benchmark rows "Ours" + number macros for the text
PARAMS = {'A_kd_Tce': '290,501', 'E_staged_Tce': r'$\mathbf{17,318}$'}
blocks['sota_ours'] = '\n'.join(
    f"{lab} (Ours) & Raw & Fpz-Cz, Pz-Oz + EOG & {PARAMS[a]} & 5-fold$^{{e}}$ & {f2(S[a]['acc'])} & "
    f"{f2(S[a]['mf1'])} & {f2(S[a]['f1'][1])} " + r'\\'
    for lab, a in [('Linear Assistant', 'A_kd_Tce'), ('Edge Student', 'E_staged_Tce')])
macros = []
for tag, a in [('Teacher', 'T_ce'), ('Asst', 'A_kd_Tce'), ('Edge', 'E_staged_Tce')]:
    s = S[a]
    macros += [rf'\newcommand{{\res{tag}MF}}{{{f2(s["mf1"])}}}', rf'\newcommand{{\res{tag}MFlo}}{{{f2(s["lo"])}}}',
               rf'\newcommand{{\res{tag}MFhi}}{{{f2(s["hi"])}}}', rf'\newcommand{{\res{tag}Acc}}{{{f2(s["acc"])}}}',
               rf'\newcommand{{\res{tag}Kappa}}{{{s["kappa"]:.3f}}}', rf'\newcommand{{\res{tag}NOne}}{{{f2(s["f1"][1])}}}']
blocks['result_macros'] = '\n'.join(macros)

# ---------------------------------------------------------------- CKA v2 (notebook R2_cka_v2.ipynb; primary = 79 subjects)
CT = pd.read_csv(os.path.join(EXP, 'results', 'cka_tests.csv'))
CT = CT[CT.subjects.str.startswith('excl')]
CS = pd.read_csv(os.path.join(EXP, 'results', 'cka_per_subject_summary.csv'))
assert set(CT.n_subjects) == {79} and set(CS.n) == {79}
PAIRS = ['Teacher ~ Assistant', 'Teacher ~ Edge-Staged', 'Teacher ~ Edge-Direct', 'Teacher ~ Edge-Scratch',
         'Assistant ~ Edge-Staged', 'Assistant ~ Edge-Direct', 'Assistant ~ Edge-Scratch',
         'Edge-Staged ~ Edge-Direct', 'Edge-Staged ~ Edge-Scratch', 'Edge-Direct ~ Edge-Scratch']
vr = CT[CT.family.str.startswith('vs_random')]
assert len(vr) == 20 and (vr.q_bh < 0.001).all() and (vr.ci95_low > 0).all()   # caption: all above random, q < 0.001


def cka_cell(level, pair):
    r = CS[(CS.level == level) & (CS.pair == pair)].iloc[0]
    return f'{r["median"]:.3f} [{r.q25:.3f}, {r.q75:.3f}]'


def vr_cell(level, pair):
    r = vr[(vr.level == level) & (vr.a == pair)].iloc[0]
    return f'$+{r["median"]:.3f}$ [{r.ci95_low:.3f}, {r.ci95_high:.3f}]'


def arrow(p):
    return p.replace('~', r'$\leftrightarrow$')


rows = [rf'\multicolumn{{5}}{{l}}{{\textit{{Trained models}}}} \\']
for p in PAIRS:
    rows.append(f"\\quad {arrow(p)} & {cka_cell('spatial', p)} & {vr_cell('spatial', p)} & "
                f"{cka_cell('temporal', p)} & {vr_cell('temporal', p)} \\\\")
rows.append(rf'\multicolumn{{5}}{{l}}{{\textit{{Baseline: second model randomly initialized}}}} \\')
for p in ['Teacher ~ random assistant', 'Teacher ~ random edge', 'Assistant ~ random edge',
          'Edge-Staged ~ random edge', 'Edge-Direct ~ random edge']:
    lab = arrow(p).replace('random assistant', 'Assistant (random)').replace('random edge', 'Edge (random)')
    rows.append(f"\\quad {lab} & {cka_cell('spatial', p)} & -- & {cka_cell('temporal', p)} & -- \\\\")
pw = CT[CT.family.str.startswith('pathway')]
rows.append(r'\midrule')
rows.append(r'\multicolumn{5}{l}{\textit{Pre-specified comparisons: $\Delta$ = A $-$ B (median of per-subject differences) [95\% CI]; $q$}} \\')
for a, b in [('Assistant ~ Edge-Staged', 'Assistant ~ Edge-Direct'), ('Teacher ~ Edge-Direct', 'Teacher ~ Edge-Staged'),
             ('Assistant ~ Edge-Staged', 'Assistant ~ Edge-Scratch'), ('Assistant ~ Edge-Direct', 'Assistant ~ Edge-Scratch')]:
    cells = []
    for lv in ('spatial', 'temporal'):
        r = pw[(pw.level == lv) & (pw.a == a) & (pw.b == b)].iloc[0]
        cells += [f'${r["median"]:+.4f}$ [${r.ci95_low:+.4f}$, ${r.ci95_high:+.4f}$]', q(r.q_bh)]
    lab = f'({arrow(a)}) $-$ ({arrow(b)})'
    rows.append(f'\\quad {lab} & ' + ' & '.join(cells) + r' \\')
blocks['cka_alignment'] = '\n'.join([r'\begin{tabular}{lcccc}', r'\toprule',
    r'& \multicolumn{2}{c}{\textbf{Spatial (CNN embedding)}} & \multicolumn{2}{c}{\textbf{Temporal (context representation)}} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'\textbf{Model pair} & \textbf{CKA, median [IQR]} & \textbf{$\Delta$ vs.\ random [95\% CI]} & \textbf{CKA, median [IQR]} & \textbf{$\Delta$ vs.\ random [95\% CI]} \\',
    r'\midrule', *rows, r'\bottomrule', r'\end{tabular}'])

# ---------------------------------------------------------------- HMC in-domain + zero-shot cross-dataset (Table 5)
C_HMC = dict(K_FOLDS=5, RUNS_DIR=os.path.join(EXP, 'runs_hmc'), SEEDS=[42])
TIERS = [('Teacher', 'T_ce'), ('Linear Assistant', 'A_kd_Tce'), ('Edge Student', 'E_staged_Tce')]


def stats_in(Cx, arm, n_boot=10000, seed=0):
    cms, _ = skd.subject_cms(Cx, arm)
    subs = sorted(cms)
    CM = np.stack([cms[s] for s in subs])
    m = skd._cm_metrics(CM.sum(0))
    W = np.random.default_rng(seed).multinomial(len(subs), np.full(len(subs), 1 / len(subs)), size=n_boot)
    b = skd._cm_metrics(np.einsum('bs,sij->bij', W, CM))['mf1']
    return dict(mf1=100 * m['mf1'], lo=100 * np.percentile(b, 2.5), hi=100 * np.percentile(b, 97.5),
                acc=100 * m['acc'], kappa=m['kappa'], n1=100 * m['n1'], n_sub=len(subs))


H = {a: stats_in(C_HMC, a) for _, a in TIERS}
hp = pd.read_csv(os.path.join(EXP, 'results', 'hmc_pooled_oof_metrics.csv')).set_index('arm')
for a, s in H.items():
    assert abs(s['mf1'] - hp.loc[a, 'mf1']) < 1e-9 and s['n_sub'] == 151, a
X = pd.read_csv(os.path.join(EXP, 'results', 'cross_dataset_fold_level.csv'))
assert (X.groupby(['direction', 'arm']).size() == 5).all()
rows = [r'\multicolumn{5}{l}{\textit{In-domain: five-fold patient-level cross-validation on HMC (pooled out-of-fold, 151 patients)}} \\']
rows += [f"\\quad {lab} & {f2(H[a]['mf1'])} [{f2(H[a]['lo'])}, {f2(H[a]['hi'])}] & {f2(H[a]['acc'])} & {H[a]['kappa']:.3f} & {f2(H[a]['n1'])} \\\\"
         for lab, a in TIERS]
for d, txt in [('SleepEDF->HMC', r'Zero-shot: trained on Sleep-EDF, tested on all of HMC (mean $\pm$ SD over the five fold models)'),
               ('HMC->SleepEDF', r'Zero-shot: trained on HMC, tested on all of Sleep-EDF (mean $\pm$ SD over the five fold models)')]:
    rows.append(r'\midrule')
    rows.append(rf'\multicolumn{{5}}{{l}}{{\textit{{{txt}}}}} \\')
    for lab, a in TIERS:
        v = X[(X.direction == d) & (X.arm == a)]
        cell = lambda c, k=100, dd=2: f'{k * v[c].mean():.{dd}f} $\\pm$ {k * v[c].std(ddof=1):.{dd}f}'
        rows.append(f"\\quad {lab} & {cell('mf1')} & {cell('acc')} & {cell('kappa', 1, 3)} & {cell('f1_N1')} \\\\")
blocks['gen_tradeoff'] = '\n'.join([r'\begin{tabular}{lcccc}', r'\toprule',
    r"\textbf{Setting / model} & \textbf{Macro-F1 (\%)} & \textbf{Accuracy (\%)} & \textbf{Cohen's $\kappa$} & \textbf{N1-F1 (\%)} \\",
    r'\midrule', *rows, r'\bottomrule', r'\end{tabular}'])
for tag, a in [('Teacher', 'T_ce'), ('Asst', 'A_kd_Tce'), ('Edge', 'E_staged_Tce')]:
    macros += [rf'\newcommand{{\hmc{tag}MF}}{{{f2(H[a]["mf1"])}}}', rf'\newcommand{{\hmc{tag}Acc}}{{{f2(H[a]["acc"])}}}']
    for d, dt in [('SleepEDF->HMC', 'EH'), ('HMC->SleepEDF', 'HE')]:
        v = X[(X.direction == d) & (X.arm == a)]
        macros += [rf'\newcommand{{\x{dt}{tag}MF}}{{{100 * v.mf1.mean():.2f}}}', rf'\newcommand{{\x{dt}{tag}MFsd}}{{{100 * v.mf1.std(ddof=1):.2f}}}',
                   rf'\newcommand{{\x{dt}{tag}Acc}}{{{100 * v.acc.mean():.2f}}}']

# ---------------------------------------------------------------- XAI v2 on DREAMS (Table 7)
XT = pd.read_csv(os.path.join(EXP, 'results', 'xai_tests.csv'))
XE = pd.read_csv(os.path.join(EXP, 'results', 'xai_excerpt_level_models.csv'))
XB = pd.read_csv(os.path.join(EXP, 'results', 'xai_excerpt_level_baselines.csv'))
assert len(XT) == 90
SPECS = ['Assistant (Sleep-EDF)', 'Edge-Staged (Sleep-EDF)', 'Edge-Direct (Sleep-EDF)', 'Assistant (HMC)', 'Edge-Staged (HMC)']
RAND = {'Assistant': 'Assistant (random init)', 'Edge': 'Edge (random init)'}
BEST = {'kcomplexes': 'envelope', 'spindles': 'sigma_power'}


def med_iqr(v):
    return f'{np.median(v):.3f} [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]'


def tcell(db, spec, meth, versus):
    r = XT[(XT.db == db) & (XT.spec == spec) & (XT.method == meth) & (XT.versus == versus)].iloc[0]
    return f'${r["median"]:+.3f}$ ({q(r.q_bh)})'


rows = []
for db, title in [('kcomplexes', 'K-complexes (10 excerpts)'), ('spindles', 'Sleep spindles (8 excerpts)')]:
    best = BEST[db].replace('_', ' ')
    rows.append(rf'\multicolumn{{6}}{{l}}{{\textit{{{title}}}}} \\')
    for sp in SPECS:
        rnd = RAND[sp.split()[0].split('-')[0]]
        g = XE[(XE.db == db) & (XE.spec == sp) & (XE.method == 'gradient')].auroc
        i = XE[(XE.db == db) & (XE.spec == sp) & (XE.method == 'ig')].auroc
        rows.append(f'\\quad {sp} & {med_iqr(g)} & {med_iqr(i)} & {tcell(db, sp, "ig", rnd)} & '
                    f'{tcell(db, sp, "ig", "abs_x")} & {tcell(db, sp, "ig", BEST[db])} \\\\')
    for sp in RAND.values():
        g = XE[(XE.db == db) & (XE.spec == sp) & (XE.method == 'gradient')].auroc
        i = XE[(XE.db == db) & (XE.spec == sp) & (XE.method == 'ig')].auroc
        rows.append(f'\\quad {sp} & {med_iqr(g)} & {med_iqr(i)} & -- & -- & -- \\\\')
    names = {'abs_x': r'$|x|$', 'envelope': 'Hilbert envelope', 'sigma_power': 'Sigma-band (11--16 Hz) power'}
    for b in (['abs_x', 'envelope'] + (['sigma_power'] if db == 'spindles' else [])):
        v = XB[(XB.db == db) & (XB.method == b)].auroc
        rows.append(f'\\quad Baseline: {names[b]} & \\multicolumn{{2}}{{c}}{{{med_iqr(v)}}} & -- & -- & -- \\\\')
    if db == 'kcomplexes':
        rows.append(r'\midrule')
blocks['xai_dreams'] = '\n'.join([r'\begin{tabular}{lccccc}', r'\toprule',
    r'& \multicolumn{2}{c}{\textbf{AUROC, median [IQR] over excerpts}} & \multicolumn{3}{c}{\textbf{IG: $\Delta$ median AUROC ($q$)}} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}',
    r'\textbf{Model (training data)} & \textbf{Gradient} & \textbf{IG} & \textbf{vs.\ random init} & \textbf{vs.\ $|x|$} & \textbf{vs.\ best baseline$^{a}$} \\',
    r'\midrule', *rows, r'\bottomrule', r'\end{tabular}'])
kc = XT[(XT.db == 'kcomplexes') & (XT.method == 'ig')]
sp_ = XT[(XT.db == 'spindles') & (XT.method == 'ig')]
assert (kc[kc.versus == 'chance (0.5)'].q_bh < 0.05).all() and (kc[kc.versus == 'envelope'].q_bh < 0.05).all()
assert (kc[kc.versus == 'envelope']['median'] < 0).all() and (sp_[sp_.versus == 'sigma_power']['median'] < 0).all()
assert (sp_[sp_.versus == 'sigma_power'].q_bh < 0.05).all()
ka = kc[kc.versus == 'abs_x']
assert (ka[ka.q_bh < 0.05]['median'] < 0).all()            # text: "lower than |x| in N of 5, not different in the others"
sa = sp_[sp_.versus == 'abs_x']
assert ((sa.q_bh < 0.05) & (sa['median'] > 0)).all()       # text: spindle IG slightly above |x| for every model
kc_ig = kc[kc.versus == 'chance (0.5)'].median_auroc
kc_rand = [np.median(XE[(XE.db == 'kcomplexes') & (XE.spec == s) & (XE.method == 'ig')].auroc) for s in RAND.values()]
kc_vr = kc[kc.versus.str.contains('random')]['median']
sp_all = XT[(XT.db == 'spindles') & (XT.versus == 'chance (0.5)')].median_auroc
macros += [rf'\newcommand{{\xaiKcIgLo}}{{{kc_ig.min():.3f}}}', rf'\newcommand{{\xaiKcIgHi}}{{{kc_ig.max():.3f}}}',
           rf'\newcommand{{\xaiKcRandLo}}{{{min(kc_rand):.3f}}}', rf'\newcommand{{\xaiKcRandHi}}{{{max(kc_rand):.3f}}}',
           rf'\newcommand{{\xaiKcVrLo}}{{{kc_vr.min():.3f}}}', rf'\newcommand{{\xaiKcVrHi}}{{{kc_vr.max():.3f}}}',
           rf'\newcommand{{\xaiKcAbs}}{{{np.median(XB[(XB.db == "kcomplexes") & (XB.method == "abs_x")].auroc):.3f}}}',
           rf'\newcommand{{\xaiKcEnv}}{{{np.median(XB[(XB.db == "kcomplexes") & (XB.method == "envelope")].auroc):.3f}}}',
           rf'\newcommand{{\xaiSpLo}}{{{sp_all.min():.3f}}}', rf'\newcommand{{\xaiSpHi}}{{{sp_all.max():.3f}}}',
           rf'\newcommand{{\xaiSpSigma}}{{{np.median(XB[(XB.db == "spindles") & (XB.method == "sigma_power")].auroc):.3f}}}',
           rf'\newcommand{{\xaiKcIgVsAbsN}}{{{int((kc[kc.versus == "abs_x"].q_bh < 0.05).sum())}}}']
# ---------------------------------------------------------------- training-seed robustness (Edge Students, Teacher-CE lineage)
SP = pd.read_csv(os.path.join(EXP, 'results', 'seed_robustness_pooled_per_seed.csv'))
SC = pd.read_csv(os.path.join(EXP, 'results', 'seed_robustness_contrasts.csv'), dtype={'seeds': str})
assert sorted(SP.seed.unique()) == [42, 43, 44] and (SP.n_subjects == 79).all() and len(SC) == 60
CTAG = {'E_staged_Tce - E_direct_Tce': 'SD', 'E_staged_Tce - E_scratch': 'SS', 'E_direct_Tce - E_scratch': 'DS'}


def seed_row(contrast, metric, seeds='42+43+44', subjects=79):
    r = SC[(SC.contrast == contrast) & (SC.metric == metric) & (SC.seeds == seeds) & (SC.subjects == subjects)]
    assert len(r) == 1, (contrast, metric, seeds, subjects)
    return r.iloc[0]


rng_mf = SP.groupby('arm').mf1.agg(lambda v: v.max() - v.min())
p79 = SC[(SC.seeds == '42+43+44') & (SC.subjects == 79)]
for c, tag in CTAG.items():
    for met, mt in (('mf1', 'MF'), ('n1', 'NOne')):
        r = seed_row(c, met)
        macros += [rf'\newcommand{{\seed{tag}{mt}}}{{{r.diff_pp:+.2f}}}', rf'\newcommand{{\seed{tag}{mt}lo}}{{{r.ci95_low_pp:+.2f}}}',
                   rf'\newcommand{{\seed{tag}{mt}hi}}{{{r.ci95_high_pp:+.2f}}}', rf'\newcommand{{\seed{tag}{mt}q}}{{{q(r.q_bh)}}}']
    r = seed_row(c, 'n1', subjects=100)
    macros += [rf'\newcommand{{\seed{tag}NOneAll}}{{{r.diff_pp:+.2f}}}', rf'\newcommand{{\seed{tag}NOneAllq}}{{{q(r.q_bh)}}}']
sd_seed = [seed_row('E_staged_Tce - E_direct_Tce', 'mf1', seeds=s).diff_pp for s in ('42', '43', '44')]
macros += [rf'\newcommand{{\seedSDMFs{k}}}{{{v:+.2f}}}' for k, v in zip(('i', 'ii', 'iii'), sd_seed)]
macros += [rf'\newcommand{{\seedRangeMF}}{{{rng_mf.max():.2f}}}', rf'\newcommand{{\seedMaxAbsMF}}{{{p79[p79.metric == "mf1"].diff_pp.abs().max():.2f}}}',
           rf'\newcommand{{\seedMFqMin}}{{{p79[p79.metric == "mf1"].q_bh.min():.2f}}}']
# claims made in the text
assert (p79[p79.metric == 'mf1'].q_bh >= 0.05).all()                              # no credible Macro-F1 difference
assert p79[p79.metric == 'mf1'].diff_pp.abs().max() < rng_mf.max()               # seed range exceeds every pooled difference
ss = seed_row('E_staged_Tce - E_scratch', 'n1')
assert ss.ci95_low_pp > 0 and ss.q_bh >= 0.05                                    # N1: CI above 0, not credible after BH
assert (seed_row('E_direct_Tce - E_scratch', 'n1').q_bh >= 0.05) and (seed_row('E_staged_Tce - E_direct_Tce', 'n1').q_bh >= 0.05)
for c in ('E_staged_Tce - E_scratch', 'E_direct_Tce - E_scratch'):                # 100 subjects: N1 credible, positive
    r = seed_row(c, 'n1', subjects=100)
    assert r.q_bh < 0.05 and r.diff_pp > 0, c
assert seed_row('E_staged_Tce - E_direct_Tce', 'n1', subjects=100).q_bh >= 0.05
assert min(sd_seed) < 0 < max(sd_seed)                                           # sign of Staged - Direct changes with seed
HY = pd.read_csv(HYB)
h = HY[(HY.contrast == 'E_staged_Tce - E_direct_Tce') & (HY.metric == 'mf1')].iloc[0]  # seed-42 row == primary analysis
assert abs(h.diff_pp - sd_seed[0]) < 1e-9 and h.n_subjects == 79
print('seed-robustness checks passed; seed range of Macro-F1 per arm:', rng_mf.round(2).to_dict())

# ---------------------------------------------------------------- quantization on the full test set (R2_quant_eval)
QP = pd.read_csv(os.path.join(EXP, 'results', 'quant_eval_pooled.csv'))
QC = pd.read_csv(os.path.join(EXP, 'results', 'quant_eval_contrasts.csv'))
QA = pd.read_csv(os.path.join(EXP, 'results', 'quant_eval_agreement.csv')).set_index('target')
S79, S100 = '79 subjects (primary)', '100 subjects (sensitivity)'
num = ['diff_pp', 'ci95_low_pp', 'ci95_high_pp', 'p_boot', 'q_bh']
a_ = QC[QC.contrast == 'esp32 - fp32'].reset_index(drop=True)
b_ = QC[QC.contrast == 'esp32s3 - fp32'].reset_index(drop=True)
assert np.array_equal(a_[num].values, b_[num].values) and (QA.loc['esp32'] == QA.loc['esp32s3']).all()   # text: identical
# the notebook's copy of the main evaluation == the main analysis (E_staged_Tce, 79 and 100 subjects)
px79 = pd.read_csv(os.path.join(EXP, 'results', 'sensitivity_exclF1_pooled.csv')).set_index('arm').loc['E_staged_Tce']
me = QP[QP.model == 'main_eval'].set_index('subjects')
assert abs(me.loc[S79, 'mf1'] - px79['mf1']) < 1e-9 and abs(me.loc[S100, 'mf1'] - pooled.loc['E_staged_Tce', 'mf1']) < 1e-9


def qrow(metric, subjects=S79):
    r = a_[(a_.metric == metric) & (a_.subjects == subjects)]
    assert len(r) == 1
    return r.iloc[0]


def pfmt(v):
    return '<0.001' if v < 0.001 else f'{v:.3f}'


for tag, met in (('MF', 'mf1'), ('NOne', 'f1_N1'), ('NThree', 'f1_N3'), ('REM', 'f1_REM')):
    for stag, sset in (('', S79), ('All', S100)):
        r = qrow(met, sset)
        macros += [rf'\newcommand{{\qe{tag}{stag}}}{{{r.diff_pp:+.2f}}}', rf'\newcommand{{\qe{tag}{stag}lo}}{{{r.ci95_low_pp:+.2f}}}',
                   rf'\newcommand{{\qe{tag}{stag}hi}}{{{r.ci95_high_pp:+.2f}}}',
                   rf'\newcommand{{\qe{tag}{stag}p}}{{{pfmt(r.p_boot)}}}', rf'\newcommand{{\qe{tag}{stag}q}}{{{pfmt(r.q_bh)}}}']
ag = QA.loc['esp32']
macros += [rf'\newcommand{{\qeAgree}}{{{ag.agreement_pct:.1f}}}', rf'\newcommand{{\qeChanged}}{{{int(ag.changed_epochs):,}}}',
           rf'\newcommand{{\qeToCorrect}}{{{int(ag.changed_to_correct):,}}}', rf'\newcommand{{\qeToWrong}}{{{int(ag.changed_to_wrong):,}}}',
           rf'\newcommand{{\qeEpochs}}{{{int(ag.epochs):,}}}',
           rf'\newcommand{{\qeAgreeMain}}{{{ag.agreement_fp32_vs_main_eval_pct:.2f}}}']
# claims made in the text
m79, m100 = qrow('mf1'), qrow('mf1', S100)
assert m79.ci95_low_pp < 0 < m79.ci95_high_pp and m79.p_boot >= 0.05            # primary: CI includes 0
assert (a_[(a_.subjects == S79) & (a_.family == 'secondary')].q_bh >= 0.05).all()   # no credible secondary change (79)
assert m100.ci95_high_pp < 0 and m100.p_boot < 0.05                                 # 100 subjects: small credible decrease
for met in ('f1_N1', 'f1_N3'):                                                      # N1, N3: no credible change in either set
    assert qrow(met).q_bh >= 0.05 and qrow(met, S100).q_bh >= 0.05
st = a_[(a_.subjects == S100) & a_.metric.str.startswith('f1_')]
assert st.loc[st.diff_pp.idxmin(), 'metric'] == 'f1_REM' and qrow('f1_REM', S100).q_bh < 0.05   # REM: largest, credible
assert -0.55 < m79.ci95_low_pp and -0.55 < m100.ci95_low_pp            # Discussion: CI excludes decreases > ~0.5 pp
st79 = a_[(a_.subjects == S79) & a_.metric.str.startswith('f1_')]
assert st79.loc[st79.diff_pp.idxmin(), 'metric'] == 'f1_REM'               # Discussion: largest change in REM (both sets)
print('quantization checks passed: dMF1 79 =', round(m79.diff_pp, 3), '| 100 =', round(m100.diff_pp, 3))

blocks['result_macros'] = '\n'.join(macros)
print('HMC/XAI checks passed; IG vs |x| significant for', int((kc[kc.versus == 'abs_x'].q_bh < 0.05).sum()), 'of 5 models')

if not TEX.endswith('.tex'):                     # standalone (code release): one file per block in a folder
    os.makedirs(TEX, exist_ok=True)
    for name, body in blocks.items():
        open(os.path.join(TEX, f'{name}.tex'), 'w', encoding='utf-8').write(body + '\n')
    print('wrote blocks to', TEX, ':', list(blocks))
    sys.exit(0)
tex = open(TEX, encoding='utf-8').read()
for name, body in blocks.items():
    pat = re.compile(rf'(% BEGIN GENERATED {name}\n).*?(% END GENERATED {name})', re.S)
    assert len(pat.findall(tex)) == 1, f'markers for {name} missing or duplicated'
    tex = pat.sub(lambda m: m.group(1) + body + '\n' + m.group(2), tex)
open(TEX, 'w', encoding='utf-8').write(tex)
print('updated blocks:', list(blocks))
