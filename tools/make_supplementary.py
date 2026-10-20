"""Generate the Supplementary Information (LaTeX + data files) from the exported Colab results — no hand-copied numbers.
usage: python make_supplementary.py <export_dir> <hybrid_csv> <out_dir>
writes <out_dir>/Supplementary_Information.tex, Supplementary_Data_1_SleepEDF_folds.csv, Supplementary_Data_2_HMC_folds.csv
(sections for HMC, XAI and hardware are added when those results exist)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_names import NAME, GROUPS, ARMS, contrast_label   # noqa: E402

EXP, HYB, OUT = sys.argv[1:4]
R = os.path.join(EXP, 'results')
os.makedirs(OUT, exist_ok=True)
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
T = []   # LaTeX chunks


def q(v):
    return '$<$0.001' if v < 0.001 else f'{v:.3f}'


def sgn(v, d=2):
    return f'${v:+.{d}f}$'


def table(caption, label, colspec, header, rows, size=r'\footnotesize'):
    T.append('\n'.join([r'\begin{table}[!htbp]', r'\centering', size, rf'\caption{{{caption}}}', rf'\label{{{label}}}',
                        r'\begin{adjustbox}{max width=\textwidth}', rf'\begin{{tabular}}{{{colspec}}}', r'\toprule', header + r' \\',
                        r'\midrule', *rows, r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}', '']))


def grouped(rowfn, ncol):
    rows = []
    for g, arms in GROUPS:
        rows.append(rf'\multicolumn{{{ncol}}}{{l}}{{\textit{{{g}}}}} \\')
        rows += [rf'\quad {NAME[a]} & ' + rowfn(a) + r' \\' for a in arms]
    return rows


# ---------------------------------------------------------------- S1 folds (+ data files)
man = pd.read_csv(os.path.join(R, 'fold_manifest_subject_level.csv'))
man.rename(columns={'rec': 'recording'}).to_csv(os.path.join(OUT, 'Supplementary_Data_1_SleepEDF_folds.csv'), index=False)
hman = pd.read_csv(os.path.join(R, 'hmc_fold_manifest.csv'))
hman.rename(columns={'rec': 'recording'}).to_csv(os.path.join(OUT, 'Supplementary_Data_2_HMC_folds.csv'), index=False)
inv = pd.read_csv(os.path.join(R, 'run_inventory.csv'))
inv = inv[(inv.dataset == 'SleepEDF') & (inv.seed == 42)]   # seeds 43/44 (Edge arms only) are the seed-robustness analysis
n_test = inv[inv.arm == 'T_ce'].set_index('fold').n_test
rows = []
for f in range(5):
    m = man[man.fold == f]
    rows.append(f'{f} & {m.subject.nunique()} ({m[m.cohort == "SC"].subject.nunique()} / {m[m.cohort == "ST"].subject.nunique()}) & '
                f'{len(m)} & {int(n_test[f]):,} & {int((hman.fold == f).sum())} \\\\')
assert man.subject.nunique() == 100 and len(man) == 197 and len(hman) == 151
table('Subject-level folds (seed 42; subjects shuffled within each cohort, then assigned in round-robin order). For fold '
      'index $i$, fold $i$ is the test set, fold $(i+1) \\bmod 5$ the validation set and the other three folds the training '
      'set. The fold of every subject and recording is listed in Supplementary Data 1 (Sleep-EDF) and 2 (HMC).',
      'tab:s_folds', 'ccccc', r'\textbf{Fold} & \textbf{Sleep-EDF subjects (SC / ST)} & \textbf{Sleep-EDF recordings} & '
      r'\textbf{Sleep-EDF scored test epochs} & \textbf{HMC patients}', rows)

# ---------------------------------------------------------------- S2 Teacher-CE tuning grid
tun = pd.read_csv(os.path.join(R, 'teacher_ce_tuning.csv')).sort_values('best_val_mf1', ascending=False)
sel = json.load(open(os.path.join(EXP, 'runs', 'seed42', 'T_ce', 'fold0', 'result.json')))['selected_from']
rows = [f"{r.lr:g} & {r.ls:g} & {100 * r.best_val_mf1:.2f} & {r.best_epoch} & {r.epochs_run} & {'selected' if r.arm == sel else ''} \\\\"
        for r in tun.itertuples()]
table('Teacher-CE configuration search (fold 0 only, one seed). Selection by validation Macro-F1 of fold 0 (the subjects '
      'of fold 1); test data were not used. Maximum 80 epochs, early stopping after 20 epochs without improvement.',
      'tab:s_tuning', 'cccccc', r'\textbf{Learning rate} & \textbf{Label smoothing} & \textbf{Validation Macro-F1 (\%)} & '
      r'\textbf{Best epoch} & \textbf{Epochs run} &', rows)

# ---------------------------------------------------------------- S3 fold-level mean +- SD
fl = pd.read_csv(os.path.join(R, 'fold_level_results.csv'))
fl = fl[(fl.split == 'test') & fl.arm.isin(ARMS)]


def msd(a, col, scale=100, d=2):
    v = fl[fl.arm == a][col] * scale
    assert len(v) == 5, a
    return f'{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}'


table('Test performance per fold (mean $\\pm$ SD over the five fold models; descriptive). The main text reports pooled '
      'out-of-fold metrics.', 'tab:s_foldsd', 'lcccc',
      r'\textbf{Training variant} & \textbf{Macro-F1 (\%)} & \textbf{Accuracy (\%)} & \textbf{Cohen\textquotesingle s $\kappa$} & \textbf{N1-F1 (\%)}',
      grouped(lambda a: ' & '.join([msd(a, 'mf1'), msd(a, 'acc'), msd(a, 'kappa', 1, 3), msd(a, 'f1_N1')]), 5))

# ---------------------------------------------------------------- S4 convergence
rows = []
for g, arms in GROUPS:
    rows.append(rf'\multicolumn{{6}}{{l}}{{\textit{{{g}}}}} \\')
    for a in arms:
        cells = []
        for f in range(5):
            r = inv[(inv.arm == a) & (inv.fold == f)].iloc[0]
            cells.append(f'{int(r.best_epoch)} / {int(r.epochs_run)}' + ('$^{*}$' if r.epochs_run >= 80 else ''))
        rows.append(rf'\quad {NAME[a]} & ' + ' & '.join(cells) + r' \\')
table('Training length per fold: epoch of the selected checkpoint / epochs run (maximum 80; early stopping after 20 epochs '
      'without improvement of the validation Macro-F1). $^{*}$Reached the 80-epoch limit.', 'tab:s_epochs', 'lccccc',
      r'\textbf{Training variant} & \textbf{Fold 0} & \textbf{Fold 1} & \textbf{Fold 2} & \textbf{Fold 3} & \textbf{Fold 4}', rows)

# ---------------------------------------------------------------- S5 pooled metrics without the 21 tuning subjects
px = pd.read_csv(os.path.join(R, 'sensitivity_exclF1_pooled.csv')).set_index('arm')
assert set(px.n_subjects) == {79}
table('Pooled out-of-fold metrics without the 21 subjects used to select the Teacher-CE configuration ($n = 79$ subjects).',
      'tab:s_pooled79', 'lcccccccc',
      r'\textbf{Training variant} & \textbf{Macro-F1} & \textbf{Acc.} & \textbf{$\kappa$} & \textbf{W} & \textbf{N1} & \textbf{N2} & \textbf{N3} & \textbf{REM}',
      grouped(lambda a: ' & '.join([f'{px.loc[a, "mf1"]:.2f}', f'{px.loc[a, "acc"]:.2f}', f'{px.loc[a, "kappa"]:.3f}']
                                   + [f'{px.loc[a, "f1_" + s]:.2f}' for s in STAGES]), 9))


# ---------------------------------------------------------------- S6 / S7 contrasts (all 100 subjects; hybrid secondary)
def contrast_rows(df):
    rows = []
    for c in df.contrast.unique():
        cells = [str(int(df[df.contrast == c].n_subjects.iloc[0]))]
        for met in ('mf1', 'acc', 'kappa', 'n1'):
            r = df[(df.contrast == c) & (df.metric == met)].iloc[0]
            cells += [f'{sgn(r.diff_pp)} [{sgn(r.ci95_low_pp)}, {sgn(r.ci95_high_pp)}]', q(r.q_bh)]
        rows.append(f'{contrast_label(c)} & ' + ' & '.join(cells) + r' \\')
    return rows


def contrast_table(df, caption, label, groups=None):
    rows = []
    for g, d in (groups or [(None, df)]):
        if g:
            rows.append(rf'\multicolumn{{10}}{{l}}{{\textit{{{g}}}}} \\')
        rows += contrast_rows(d)
    table(caption, label, 'lccccccccc', r'\textbf{Contrast (A $-$ B)} & \textbf{$n$} & \textbf{$\Delta$Macro-F1} & \textbf{$q$} & '
          r'\textbf{$\Delta$Accuracy} & \textbf{$q$} & \textbf{$\Delta\kappa\times100$} & \textbf{$q$} & \textbf{$\Delta$N1-F1} & \textbf{$q$}', rows)


contrast_table(pd.read_csv(HYB), 'Pre-specified contrasts, primary analysis (contrasts involving Teacher-CE or a model '
               'distilled from it use 79 subjects, the others 100): all metrics. $\\Delta$ in percentage points '
               '($\\kappa$: $\\times 100$) with subject-cluster bootstrap 95\\% CI; $q$: Benjamini--Hochberg within the '
               'primary (Macro-F1) and the secondary (accuracy, $\\kappa$, N1-F1) family.', 'tab:s_contrasts_primary')
contrast_table(pd.read_csv(os.path.join(R, 'primary_cluster_bootstrap.csv')), 'Sensitivity analysis: the same contrasts '
               'computed on all 100 subjects for every contrast.', 'tab:s_contrasts_all')

# ---------------------------------------------------------------- S8 per-subject Wilcoxon
wx = pd.read_csv(os.path.join(R, 'sensitivity_subject_wilcoxon.csv'))
rows = []
for c in wx.contrast.unique():
    cells = []
    for met in ('mf1', 'acc', 'n1'):
        r = wx[(wx.contrast == c) & (wx.metric == met)].iloc[0]
        cells += [f'{sgn(r.mean_diff_pp)} [{sgn(r.ci95_low_pp)}, {sgn(r.ci95_high_pp)}]', f'{int(r.subjects_better)}', q(r.q_bh)]
    rows.append(f'{contrast_label(c)} & ' + ' & '.join(cells) + r' \\')
table('Sensitivity analysis: per-subject metrics (Macro-F1 over the stages present in each subject) compared with two-sided '
      'Wilcoxon signed-rank tests over the 100 subjects. Mean difference (percentage points) with bootstrap 95\\% CI; '
      '"A better": number of subjects for which A is higher; $q$: Benjamini--Hochberg over all rows.', 'tab:s_wilcoxon',
      'lccccccccc', r'\textbf{Contrast (A $-$ B)} & \textbf{$\Delta$MF1} & \textbf{A better} & \textbf{$q$} & \textbf{$\Delta$Acc.} & '
      r'\textbf{A better} & \textbf{$q$} & \textbf{$\Delta$N1-F1} & \textbf{A better} & \textbf{$q$}', rows)

# ---------------------------------------------------------------- training-seed robustness (Edge Students, Teacher-CE lineage)
SEED_ARMS = ['E_scratch', 'E_direct_Tce', 'E_staged_Tce']
sp = pd.read_csv(os.path.join(R, 'seed_robustness_pooled_per_seed.csv'))
assert sorted(sp.seed.unique()) == [42, 43, 44] and (sp.n_subjects == 79).all() and len(sp) == 9
rows = []
for a in SEED_ARMS:
    for r in sp[sp.arm == a].sort_values('seed').itertuples():
        rows.append(f"{NAME[a] if r.seed == 42 else ''} & {r.seed} & {r.mf1:.2f} & {r.acc:.2f} & {r.kappa:.3f} & "
                    + ' & '.join(f'{getattr(r, "f1_" + st):.2f}' for st in STAGES) + r' \\')
    v = sp[sp.arm == a].mf1
    rows.append(rf'\quad \textit{{range over seeds}} & & {v.max() - v.min():.2f} & & & & & & & \\')
table('Training-seed robustness: pooled out-of-fold metrics of the three Edge Students of the Teacher-CE lineage for training '
      'seeds 42, 43 and 44 ($n = 79$ subjects; the 21 subjects used to select the Teacher-CE configuration excluded). Seeds 43 '
      'and 44 change only the initialization and batch order of the Edge Student; the folds and the distillation sources '
      '(the seed-42 Teacher-CE and Assistant-KD (Teacher-CE) of the same fold) are unchanged. Range: maximum minus minimum '
      'Macro-F1 over the three seeds.', 'tab:s_seed_pooled', 'llcccccccc',
      r'\textbf{Training variant} & \textbf{Seed} & \textbf{Macro-F1} & \textbf{Acc.} & \textbf{$\kappa$} & \textbf{W} & '
      r'\textbf{N1} & \textbf{N2} & \textbf{N3} & \textbf{REM}', rows)
sc = pd.read_csv(os.path.join(R, 'seed_robustness_contrasts.csv'), dtype={'seeds': str})
assert len(sc) == 60
grp = [('Seeds 42, 43 and 44 pooled, 79 subjects (pre-specified analysis)', sc[(sc.seeds == '42+43+44') & (sc.subjects == 79)]),
       ('Seeds 42, 43 and 44 pooled, all 100 subjects (sensitivity analysis)', sc[(sc.seeds == '42+43+44') & (sc.subjects == 100)])]
grp += [(f'Seed {s} only, 79 subjects', sc[sc.seeds == s]) for s in ('42', '43', '44')]
assert sum(len(d) for _, d in grp) == 60
contrast_table(sc, 'Training-seed robustness: contrasts between the Edge Students of Table~\\ref{tab:s_seed_pooled}. In the '
               'pooled analysis the predictions of the three seeds are pooled within each subject and subjects are '
               'resampled (10,000 subject-cluster bootstrap replicates). $\\Delta$ in percentage points ($\\kappa$: '
               '$\\times 100$) with 95\\% CI; $q$: Benjamini--Hochberg over the three contrasts within the primary '
               '(Macro-F1) and the secondary (accuracy, $\\kappa$, N1-F1) family, separately for each block.',
               'tab:s_seed_contrasts', groups=grp)

# ---------------------------------------------------------------- quantization on the full test set (R2_quant_eval)
qp = pd.read_csv(os.path.join(R, 'quant_eval_pooled.csv'))
QNAME = {'fp32': 'FP32 (strict single precision)', 'main_eval': 'Main evaluation (bfloat16 autocast)',
         'esp32': 'INT8 convolutions, ESP32 configuration', 'esp32s3': 'INT8 convolutions, ESP32-S3 configuration'}
rows = []
for sset in qp.subjects.unique():
    rows.append(rf'\multicolumn{{9}}{{l}}{{\textit{{{sset}}}}} \\')
    for r in qp[qp.subjects == sset].itertuples():
        rows.append(f'\\quad {QNAME[r.model]} & {r.mf1:.2f} & {r.acc:.2f} & {r.kappa / 100:.3f} & '
                    + ' & '.join(f'{getattr(r, "f1_" + st):.2f}' for st in STAGES) + r' \\')
table('Edge-Staged (Assistant-KD, Teacher-CE): pooled out-of-fold metrics of the FP32 model and of the deployed '
      'mixed-precision configuration (INT8 depthwise and pointwise convolutions, FP32 elsewhere; PC simulation of the '
      'quantized model of each fold, calibrated on training subjects of that fold).', 'tab:s_quant_pooled', 'lcccccccc',
      r'\textbf{Model} & \textbf{Macro-F1} & \textbf{Acc.} & \textbf{$\kappa$} & \textbf{W} & \textbf{N1} & \textbf{N2} & '
      r'\textbf{N3} & \textbf{REM}', rows)
qc = pd.read_csv(os.path.join(R, 'quant_eval_contrasts.csv'))
qc = qc[qc.contrast == 'esp32 - fp32']
QM = {'mf1': 'Macro-F1', 'acc': 'Accuracy', 'kappa': "Cohen's $\\kappa$ ($\\times 100$)", 'f1_W': 'F1 W', 'f1_N1': 'F1 N1',
      'f1_N2': 'F1 N2', 'f1_N3': 'F1 N3', 'f1_REM': 'F1 REM'}
rows = []
for sset in qc.subjects.unique():
    rows.append(rf'\multicolumn{{4}}{{l}}{{\textit{{{sset}}}}} \\')
    for r in qc[qc.subjects == sset].itertuples():
        rows.append(f'\\quad {QM[r.metric]} & {sgn(r.diff_pp)} [{sgn(r.ci95_low_pp)}, {sgn(r.ci95_high_pp)}] & {q(r.p_boot)} & '
                    f'{q(r.q_bh) if r.family == "secondary" else "--"} \\\\')
table('Effect of quantization (INT8 configuration $-$ FP32, percentage points) with paired subject-cluster bootstrap 95\\% '
      'CI (10,000 replicates). Macro-F1 is the pre-specified primary outcome; $q$: Benjamini--Hochberg over the seven '
      'secondary metrics. The ESP32 and ESP32-S3 configurations gave identical predictions, so one set of results is shown.',
      'tab:s_quant_contrasts', 'lccc', r'\textbf{Metric} & \textbf{$\Delta$ [95\% CI]} & \textbf{$p$} & \textbf{$q$}', rows)
tr = pd.read_csv(os.path.join(R, 'quant_eval_transitions_esp32.csv'), index_col=0)
rows = [f'{i} & ' + ' & '.join(f'{int(v):,}' for v in tr.loc[i].values) + r' \\' for i in tr.index]
table('Labels of all test epochs predicted by the FP32 model (rows) and by the INT8 configuration (columns).',
      'tab:s_quant_transitions', 'lccccc', r'\textbf{FP32 $\backslash$ INT8} & ' + ' & '.join(rf'\textbf{{{c}}}' for c in tr.columns), rows)

# ---------------------------------------------------------------- S9 CKA (all 100 subjects; fold level)
ct = pd.read_csv(os.path.join(R, 'cka_tests.csv'))
ct = ct[ct.subjects.str.startswith('all')]
assert set(ct.n_subjects) == {100}
rows = []
for r in ct.itertuples():
    rows.append(f"{r.family.split('|')[0].replace('_', ' ')} & {r.level} & {r.a.replace('~', '--')} & {r.b.replace('~', '--')} & "
                f"{r.median_a:.3f} & {r.median_b:.3f} & {sgn(r.median, 4)} [{sgn(r.ci95_low, 4)}, {sgn(r.ci95_high, 4)}] & {q(r.q_bh)} \\\\")
table('Linear CKA tests on all 100 subjects (sensitivity analysis; the main text uses the 79 subjects not involved in '
      'selecting the Teacher-CE configuration). Medians of per-subject CKA; $\\Delta$: median of per-subject differences '
      'A $-$ B with bootstrap 95\\% CI; Wilcoxon signed-rank tests, Benjamini--Hochberg within each family.',
      'tab:s_cka100', 'llllcccc', r'\textbf{Family} & \textbf{Level} & \textbf{A} & \textbf{B} & \textbf{Median A} & '
      r'\textbf{Median B} & \textbf{$\Delta$ [95\% CI]} & \textbf{$q$}', rows, size=r'\scriptsize')
cf = pd.read_csv(os.path.join(R, 'cka_fold_level_summary.csv'))
rows = [f"{r.level} & {r.pair.replace('~', '--')} & {r.mean:.3f} $\\pm$ {r.std:.3f} \\\\" for r in cf.itertuples()]
table('Linear CKA computed over all test epochs of a fold (mean $\\pm$ SD over the five folds; descriptive).',
      'tab:s_ckafold', 'llc', r'\textbf{Level} & \textbf{Model pair} & \textbf{CKA}', rows, size=r'\scriptsize')

# ---------------------------------------------------------------- S11-S12 HMC
TIER = [('Teacher', 'T_ce'), ('Linear Assistant', 'A_kd_Tce'), ('Edge Student', 'E_staged_Tce')]
hf = pd.read_csv(os.path.join(R, 'hmc_fold_level_results.csv'))
hf = hf[hf.split == 'test']
rows = []
for lab, a in TIER:
    v = hf[hf.arm == a]
    assert len(v) == 5, a
    rows.append(f'{lab} & ' + ' & '.join(f'{k * v[c].mean():.{d}f} $\\pm$ {k * v[c].std(ddof=1):.{d}f}' for c, k, d in
                                       [('mf1', 100, 2), ('acc', 100, 2), ('kappa', 1, 3)] + [(f'f1_{s}', 100, 2) for s in STAGES]) + r' \\')
table('HMC in-domain test performance per fold (mean $\\pm$ SD over the five fold models; descriptive). Models: '
      'Teacher-CE, Assistant-KD (Teacher-CE), Edge-Staged (Assistant-KD, Teacher-CE).', 'tab:s_hmcfold', 'lcccccccc',
      r'\textbf{Model} & \textbf{Macro-F1} & \textbf{Acc.} & \textbf{$\kappa$} & \textbf{W} & \textbf{N1} & \textbf{N2} & \textbf{N3} & \textbf{REM}', rows)
xf = pd.read_csv(os.path.join(R, 'cross_dataset_fold_level.csv'))
rows = []
for d, dl in [('SleepEDF->HMC', 'Sleep-EDF $\\rightarrow$ HMC'), ('HMC->SleepEDF', 'HMC $\\rightarrow$ Sleep-EDF')]:
    for lab, a in TIER:
        v = xf[(xf.direction == d) & (xf.arm == a)]
        assert len(v) == 5
        rows.append(f'{dl} & {lab} & ' + ' & '.join(f'{100 * v[f"f1_{s}"].mean():.2f} $\\pm$ {100 * v[f"f1_{s}"].std(ddof=1):.2f}'
                                                   for s in STAGES) + r' \\')
table('Zero-shot cross-dataset transfer: per-stage F1 (\\%, mean $\\pm$ SD over the five fold models; every fold model '
      'predicts all scored epochs of the target dataset).', 'tab:s_cross', 'llccccc',
      r'\textbf{Direction} & \textbf{Model} & \textbf{W} & \textbf{N1} & \textbf{N2} & \textbf{N3} & \textbf{REM}', rows)

# ---------------------------------------------------------------- S13-S15 XAI on DREAMS
xm = pd.read_csv(os.path.join(R, 'xai_dreams_excerpts.csv'))
assert len(xm) == 18
rows = [f"{'K-complexes' if r.db == 'kcomplexes' else 'Spindles'} & {r.excerpt} & {r.channel} & {r.fs_native:g} & {r.resample} & "
        f"{r.n_events} & {r.n_event_epochs} & {r.stage_W}/{r.stage_N1}/{r.stage_N2}/{r.stage_N3}/{r.stage_REM} \\\\" for r in xm.itertuples()]
table('DREAMS excerpts used for the attribution analysis (60 epochs of 30 s each). Events: expert-1 annotations; target '
      'epochs: epochs containing part of an event. Stages: majority of the six 5-s hypnogram values per epoch (R\\&K, '
      'S3 and S4 merged into N3); used for description only. The scored channel is resampled to 100 Hz with the given '
      'up/down ratio and copied into all three input slots.', 'tab:s_dreams', 'lccccccc',
      r'\textbf{Database} & \textbf{Excerpt} & \textbf{Channel} & \textbf{Native Hz} & \textbf{Resampling} & \textbf{Events} & '
      r'\textbf{Target epochs} & \textbf{W/N1/N2/N3/REM epochs}', rows)
xt = pd.read_csv(os.path.join(R, 'xai_tests.csv'))
assert len(xt) == 90
for db, ev, n in [('kcomplexes', 'K-complexes', 10), ('spindles', 'sleep spindles', 8)]:
    rows = [f"{r.family.split('|')[0]} & {r.spec} & {'IG' if r.method == 'ig' else 'Gradient'} & "
            f"{r.versus.replace('_', ' ').replace('abs x', '$|x|$')} & {r.median_auroc:.3f} & {sgn(r.median, 3)} [{sgn(r.ci95_low, 3)}, {sgn(r.ci95_high, 3)}] & "
            f"{int(r.n_positive)}/{int(r.n_excerpts)} & {q(r.q_bh)} \\\\" for r in xt[xt.db == db].itertuples()]
    table(f'Pre-specified attribution tests, {ev} ($n = {n}$ excerpts; {len(rows)} of the 90 tests). Q1: AUROC vs.\\ 0.5; '
          'Q2: trained model vs.\\ model-free reference map; Q3: trained vs.\\ randomly initialized model of the same '
          'architecture. $\\Delta$: median of per-excerpt differences with bootstrap 95\\% CI; positive: excerpts with '
          '$\\Delta > 0$; two-sided Wilcoxon signed-rank tests, Benjamini--Hochberg within each question and event type.',
          f'tab:s_xaitests_{db}', 'lllccccc', r'\textbf{Q} & \textbf{Model} & \textbf{Method} & \textbf{Versus} & '
          r'\textbf{Median AUROC} & \textbf{$\Delta$ [95\% CI]} & \textbf{Positive} & \textbf{$q$}', rows, size=r'\scriptsize')
comp = pd.read_csv(os.path.join(R, 'xai_ig_completeness.csv'))
sens = pd.read_csv(os.path.join(R, 'xai_ig_grid_sensitivity_models.csv'))
stest = pd.read_csv(os.path.join(R, 'xai_ig_grid_sensitivity_tests.csv'))
n_changed = int(stest.conclusion_changed.sum())
rows = [f'{r.spec} & {r.rec.replace("_", " ")} & {r.uniform50:.3f} & {r.log64:.3f} \\\\' for r in comp.itertuples()]
rows += [r'\midrule', r'\multicolumn{4}{l}{\textit{Fold models with median error $> 0.10$ on the 64-point grid, repeated '
         r'with 256 points from $10^{-9}$: median error 64 $\rightarrow$ 256 points; mean epoch AUROC 64 $\rightarrow$ 256 points}} \\']
rows += [f'{r.model_id.replace("|", ", ")} & {r.completeness_64:.3f} $\\rightarrow$ {r.completeness_256:.3f} & '
         f'{r.auroc_64:.4f} $\\rightarrow$ {r.auroc_256:.4f} & max $|\\Delta|$ {r.max_abs_auroc_diff:.4f} \\\\' for r in sens.itertuples()]
table('Completeness of Integrated Gradients, $|\\sum \\mathrm{IG} - (F(x) - F(0))| / |F(x) - F(0)|$ (median over the '
      f'first 32 target epochs; fold-0 models). Top: uniform 50-point midpoint grid vs.\\ the logarithmic 64-point grid used '
      f'in the analysis. Bottom: models repeated with a finer grid; replacing their IG results changed the conclusion '
      f'($q < 0.05$ or not) of {n_changed} of the {len(stest)} tests.', 'tab:s_igcomp', 'llcc',
      r'\textbf{Model} & \textbf{Excerpt / grid} & \textbf{Uniform, 50} & \textbf{Logarithmic, 64}', rows, size=r'\scriptsize')

doc = '\n'.join([r'\documentclass[11pt]{article}', r'\usepackage[margin=2cm]{geometry}', r'\usepackage{booktabs,graphicx,textcomp,adjustbox}',
                 r'\usepackage[T1]{fontenc}', r'\usepackage{lmodern}',r'\renewcommand{\thetable}{S\arabic{table}}',
                 r'\title{Supplementary Information\\[2pt]\large Ultra-Lightweight Sleep Stage Classification on a Microcontroller: Hardware-Validated Compression and a '
                 r'Controlled Evaluation of Teacher-Assistant Distillation}',
                 r'\date{}', r'\begin{document}', r'\maketitle',
                 r'All values are generated from the exported cross-validation runs (five-fold subject-level cross-validation on '
                 f'Sleep-EDF Expanded, {man.subject.nunique()} subjects, {int(n_test.sum()):,} scored epochs). '
                 r'Supplementary Data 1 and 2 list the fold of every '
                 r'subject and recording.', '', *T, r'\end{document}'])
open(os.path.join(OUT, 'Supplementary_Information.tex'), 'w', encoding='utf-8').write(doc)
print(f'{len(T)} tables ->', OUT)
