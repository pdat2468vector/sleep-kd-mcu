"""Build the two Colab notebooks of the round-2 re-run, plus a matching code dir for local smoke tests.
usage: python build_notebook.py <repo_root> <nbsrc_dir> <out_dir> <out_code_dir>
  -> <out_dir>/R2_subject_level_rerun.ipynb   (Sleep-EDF: data, folds, all KD arms, primary statistics)
  -> <out_dir>/R2_hmc_crossdataset.ipynb      (HMC: data, folds, in-domain arms, zero-shot EDF<->HMC)"""
import json, os, sys

REPO, SRC, OUT_DIR, OUT_CODE = sys.argv[1:5]
CODE_DIR = '/content/skd_code'

RET_OLD = """        return {
            'logits'        : logits,
            'cnn_feat_proj' : cnn_feat_proj,
            'ctx_feat_proj' : ctx_feat_proj,
        }"""
RET_NEW = """        return {
            'logits'        : logits,
            'cnn_feat'      : cnn_feat,       # native features (used when this model is a KD source)
            'ctx_feat'      : ctx_feat,
            'cnn_feat_proj' : cnn_feat_proj,
            'ctx_feat_proj' : ctx_feat_proj,
        }"""


def read(p):
    return open(p, encoding='utf-8').read()


files = {}   # relative path -> content
for f in ['__init__.py', 'blocks_v11.py', 'teacher_v11.py']:
    files[f'model_v11/{f}'] = read(os.path.join(REPO, 'model_v11', f))
for f in ['student_v11.py', 'student_v2.py']:
    s = read(os.path.join(REPO, 'model_v11', f))
    assert s.count(RET_OLD) == 1, f
    files[f'model_v11/{f}'] = s.replace(RET_OLD, RET_NEW)
files['losses_v11.py'] = read(os.path.join(REPO, 'losses_v11.py'))
files['prep.py'] = read(os.path.join(SRC, 'prep.py'))
files['skd.py'] = read(os.path.join(SRC, 'skd.py'))
files['xai.py'] = read(os.path.join(SRC, 'xai.py'))
files['cka.py'] = read(os.path.join(SRC, 'cka.py'))
files['qsim.py'] = read(os.path.join(SRC, 'qsim.py'))

for rel, content in files.items():                      # local copy for smoke tests
    p = os.path.join(OUT_CODE, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w', encoding='utf-8').write(content)


class NB:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n')))

    def code(self, s):
        self.cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip('\n')))

    def code_files(self):
        self.code("!pip -q install mne thop")
        self.md("## 0. Code (model gốc của project + module chung)\n"
                "Các file `model_v11/*` và `losses_v11.py` được chép nguyên văn từ project. Thay đổi duy nhất: "
                "`StudentSleepNetV11` và `StudentSleepNetV2` trả thêm `cnn_feat`/`ctx_feat` (feature gốc) để làm model nguồn cho KD.")
        for rel, content in files.items():
            self.code(f'%%writefile {CODE_DIR}/{rel}\n' + content)
        self.code(r"""
import sys, json, importlib, shutil, time
sys.path.insert(0, CODE_DIR)
import torch, numpy as np, pandas as pd
import prep, skd
importlib.reload(prep); importlib.reload(skd)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '',
      '| bf16:', device.type == 'cuda' and torch.cuda.is_bf16_supported())
""")

    def save(self, path, gpu='A100'):
        nb = dict(nbformat=4, nbformat_minor=5, cells=self.cells,
                  metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                language_info=dict(name='python'), accelerator='GPU',
                                colab=dict(provenance=[], gpuType=gpu)))
        for c in nb['cells']:
            c['source'] = c['source'].splitlines(keepends=True)
        os.makedirs(os.path.abspath(OUT_DIR), exist_ok=True)
        json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('notebook:', path, '| cells:', len(self.cells))


COMMON_CONFIG = r"""
from google.colab import drive
drive.mount('/content/drive')
import os

DRIVE_ROOT = '/content/drive/MyDrive/sleep_kd_r2'
CODE_DIR   = '/content/skd_code'
# Sleep-EDF
RAW_DIR    = '/content/sleep-edfx'                 # EDF gốc (local, tải lại được)
DATA_DIR   = f'{DRIVE_ROOT}/data_sleepedf'         # npz theo từng recording (lưu Drive)
LOCAL_DATA = '/content/data_sleepedf'              # bản copy local để đọc nhanh
META_CSV   = f'{DRIVE_ROOT}/recordings.csv'
FOLDS_JSON = f'{DRIVE_ROOT}/folds_subject_level.json'
RUNS_DIR   = f'{DRIVE_ROOT}/runs'
RESULTS    = f'{DRIVE_ROOT}/results'
for d in [DRIVE_ROOT, DATA_DIR, RUNS_DIR, RESULTS, f'{CODE_DIR}/model_v11']:
    os.makedirs(d, exist_ok=True)

# ---- Quyết định giao thức (ghi rõ trong Methods)
K_FOLDS          = 5
SPLIT_SEED       = 42    # seed CHIA FOLD (cố định cho mọi phân tích) — khác seed huấn luyện ở SEEDS
TRIM_WAKE_MIN    = 30    # epoch mục tiêu = epoch có nhãn trong [epoch ngủ đầu − 30 phút, epoch ngủ cuối + 30 phút]
STORE_FULL_NIGHT = False # True: lưu cả đêm để báo cáo thêm kết quả không cắt Wake
REJECT_PTP_UV    = None  # None = không loại epoch nào

# ---- Huấn luyện
SEEDS     = [42]         # seed HUẤN LUYỆN (khởi tạo + thứ tự batch); fold giữ nguyên theo SPLIT_SEED
RUN_FOLDS = [0, 1, 2, 3, 4]
RUN_ARMS  = None         # None = tất cả; hoặc list tên arm để chạy một phần
TRAIN_CFG = dict(EPOCHS=80, PATIENCE=20, LR=1e-3, WD=1e-4, LABEL_SMOOTHING=0.05, FOCAL_GAMMA=2.0,
                 W_CE=0.3, W_KL=0.5, W_MSE=0.2, EVAL_BS=256, NUM_WORKERS=min(8, os.cpu_count()), USE_BF16=True)
"""

PROTOCOL_MD = r"""
**Thống nhất giao thức (khác code cũ — cần ghi vào Methods):**
- Mọi nhánh: AdamW (wd 1e-4), CosineAnnealingWarmRestarts(10, 2, 1e-6), tối đa 80 epoch, early stopping 20 theo **val Macro-F1**, clip 1.0; **chỉ một tiêu chí chọn checkpoint** (val Macro-F1). LR 1e-3 và label smoothing 0.05, trừ Teacher-CE được tune.
- KD: w_CE 0.3 / w_KL 0.5 / w_MSE 0.2; T = 5 (Teacher→Assistant), T = 4 cho **mọi** nhánh →Edge (Direct và Staged cùng T và trọng số).
- Feature matching khớp **feature gốc** của model nguồn (Teacher 256-d, Assistant 128-d) qua adaptor tuyến tính chỉ dùng khi train → adaptor của Edge-Direct (→256) lớn hơn Edge-Staged (→128); lúc suy luận hai Edge giống hệt nhau (17,318 tham số).
- Fold chia theo **subject** với `SPLIT_SEED` = 42 (xáo subject trong từng cohort, chia vòng tròn); seed huấn luyện (`SEEDS`) là yếu tố riêng.

**Tiền xử lý:** 3 kênh → 100 Hz → Butterworth bậc 5, 0.5–35 Hz, zero-phase **theo từng epoch** → z-score **theo từng epoch, từng kênh**. Không notch, không loại epoch.
**Cắt Wake:** epoch mục tiêu là các epoch có nhãn nằm trong khoảng từ 30 phút (60 epoch) trước epoch ngủ đầu tiên đến 30 phút sau epoch ngủ cuối cùng; epoch ngoài khoảng chỉ dùng làm ngữ cảnh (tối đa 10 epoch mỗi bên). Cửa sổ 21 epoch không bao giờ nối hai recording (biên được lặp lại).

**Chạy lại an toàn khi Colab mất kết nối:** dữ liệu, checkpoint `last.pt`/`best.pt`, dự đoán và kết quả lưu trên Drive; chạy lại cell nào thì phần đã xong được bỏ qua, phần dở chạy tiếp từ epoch cuối.
"""

# =============================================================================== notebook 1: Sleep-EDF
nb = NB()
nb.md(r"""
# Sleep-EDF — chạy lại theo **subject-level 5-fold CV** (revision vòng 2)

**Lý do:** CV cũ chia theo *recording* → 83/97 subject có 2 đêm bị tách sang 2 fold (xem `revision/P0_subject_split/`).

**Ma trận đối chứng**

| Câu hỏi | Nhánh |
|---|---|
| Edge chỉ học nhãn thật | `E_scratch` |
| KD trực tiếp từ Teacher (công thức gốc) | `E_direct_Tfocal` |
| Teacher → Assistant → Edge (pipeline đề xuất) | `A_kd_Tfocal` → `E_staged_Tfocal` |
| Assistant có ích **không cần Teacher**? (R1-b) | `A_scratch` → `E_from_Ascratch` |
| Cách train Teacher ảnh hưởng ra sao? (R1-a) | `T_focal` vs `T_ce` |
| Staged còn hơn Direct khi Teacher đã được tune? (R1-a,c) | `E_direct_Tce` vs `E_staged_Tce` (qua `A_kd_Tce`) |

**Tune Teacher-CE (R1-a):** lưới LR {1e-3, 3e-4, 1e-4} × label smoothing {0, 0.05} (`Ttune_ce_*`), chỉ train trên **fold 0** với seed đầu tiên; chọn cấu hình có **val Macro-F1** cao nhất (không dùng test), rồi `T_ce` dùng cấu hình đó cho cả 5 fold (fold 0 dùng lại đúng run đã tune).
""" + PROTOCOL_MD + r"""
**Thống kê (chốt trước khi xem kết quả):**
- Metric chính: **Macro-F1 của toàn bộ dự đoán out-of-fold gộp** (5 fold; gộp các seed huấn luyện).
- Suy luận: **bootstrap theo cụm subject** (lấy mẫu lại cả subject, giữ 2 đêm cùng nhau), 10,000 lần → CI 95% của chênh lệch và p bootstrap.
- **Family chính** = Macro-F1 × mọi so sánh trong `PRIMARY_CONTRASTS`, hiệu chỉnh BH-FDR. Accuracy, κ, N1-F1 = family phụ (BH riêng).
- Phân tích độ nhạy: Wilcoxon trên Macro-F1 theo từng subject. Mean ± SD theo fold chỉ để mô tả.
""")
nb.code("# ===================== CONFIG — chỉ sửa ở đây =====================" + COMMON_CONFIG + r"""
ARMS = [   # thứ tự = thứ tự phụ thuộc; nhóm "lõi" đứng trước
    dict(name='T_focal',          kind='teacher',   loss='focal', bs=64),
    dict(name='A_scratch',        kind='assistant', loss='ce',    bs=32),
    dict(name='A_kd_Tfocal',      kind='assistant', loss='kd', src='T_focal',     T=5.0, bs=32),
    dict(name='E_scratch',        kind='edge',      loss='ce',    bs=64),
    dict(name='E_direct_Tfocal',  kind='edge',      loss='kd', src='T_focal',     T=4.0, bs=64),
    dict(name='E_staged_Tfocal',  kind='edge',      loss='kd', src='A_kd_Tfocal', T=4.0, bs=64),
    dict(name='E_from_Ascratch',  kind='edge',      loss='kd', src='A_scratch',   T=4.0, bs=64),
] + [   # ---- lưới tune Teacher-CE: chỉ fold 0, seed đầu tiên; chọn theo validation
    dict(name=f'Ttune_ce_lr{lr:g}_ls{ls:g}', kind='teacher', loss='ce', lr=lr, ls=ls, bs=64, folds=[0], seeds='first')
    for lr in (1e-3, 3e-4, 1e-4) for ls in (0.05, 0.0)
] + [
    dict(name='T_ce',             kind='teacher',   loss='ce', bs=64, tune=dict(prefix='Ttune_ce_', fold=0)),
    dict(name='A_kd_Tce',         kind='assistant', loss='kd', src='T_ce',        T=5.0, bs=32),
    dict(name='E_direct_Tce',     kind='edge',      loss='kd', src='T_ce',        T=4.0, bs=64),
    dict(name='E_staged_Tce',     kind='edge',      loss='kd', src='A_kd_Tce',    T=4.0, bs=64),
]

PRIMARY_CONTRASTS = [   # family chính (Macro-F1) — CHỐT TRƯỚC, không thêm/bớt sau khi xem kết quả
    ('E_staged_Tfocal', 'E_direct_Tfocal'),   # staged vs direct (Teacher gốc)
    ('E_staged_Tfocal', 'E_from_Ascratch'),   # đóng góp của bước Teacher→Assistant (R1-b)
    ('E_from_Ascratch', 'E_scratch'),         # Assistant không có Teacher có giúp Edge không
    ('E_staged_Tfocal', 'E_scratch'),
    ('E_direct_Tfocal', 'E_scratch'),
    ('E_staged_Tce',    'E_direct_Tce'),      # staged vs direct với Teacher đã tune (R1-a,c)
    ('E_direct_Tce',    'E_direct_Tfocal'),   # ảnh hưởng của chất lượng Teacher
    ('T_ce',            'T_focal'),
    ('A_kd_Tfocal',     'A_scratch'),
]
C = dict(TRAIN_CFG, K_FOLDS=K_FOLDS, RUNS_DIR=RUNS_DIR, ARMS=ARMS, SEEDS=SEEDS)
print('config OK |', len(ARMS), 'arms')
""")
nb.code_files()

nb.md("## 1. Dữ liệu\n"
      "Mục 1a–1c chạy được trên **runtime CPU**: EDF gốc nằm ở đĩa local của runtime, còn kết quả tiền xử lý "
      "được lưu lên Drive. Sau khi 1b xong, đổi sang GPU rồi chạy lại các cell CONFIG → code → import, "
      "bỏ qua 1a/1b, sau đó chạy 1c và mục 2.\n\n"
      "### 1a. Tải Sleep-EDF Expanded v1.0.0 từ PhysioNet (~8 GB, 394 file EDF)")
nb.code(r"""
import sys; sys.path.insert(0, CODE_DIR)
import pandas as pd, prep          # tự import, không phụ thuộc cell import ở trên
if os.path.exists(META_CSV) and len(pd.read_csv(META_CSV)) >= 197:
    print('Dữ liệu đã được tiền xử lý đủ trên Drive — bỏ qua bước tải.')
else:
    fails = prep.download_sleep_edfx(RAW_DIR, n_threads=8)   # chạy lại nếu có file lỗi; file đã có sẽ được bỏ qua
""")
nb.md("### 1b. Tiền xử lý từng recording → `DATA_DIR/<rec>.npz` (tiếp tục được nếu bị ngắt)")
nb.code(r"""
import sys; sys.path.insert(0, CODE_DIR)
import pandas as pd, prep
meta = prep.preprocess_all(RAW_DIR, DATA_DIR, META_CSV, trim_wake_min=TRIM_WAKE_MIN,
                           store_full_night=STORE_FULL_NIGHT, reject_ptp_uv=REJECT_PTP_UV)
print(len(meta), 'recordings |', meta.subject.nunique(), 'subjects |',
      meta.groupby('cohort').rec.count().to_dict())
print('sampling rates gốc:', meta.sfreq_orig.value_counts().to_dict())
print('epoch có biên độ đỉnh-đỉnh > 150 µV (chỉ để báo cáo):', int(meta.n_ptp_over_150uv.sum()))
meta.head()
""")
nb.md("### 1c. Chia 5 fold **theo subject** — kiểm tra không subject nào nằm ở 2 fold; xuất danh sách train/val/test")
nb.code(r"""
meta = pd.read_csv(META_CSV)
meta, folds = prep.make_subject_folds(meta, k=K_FOLDS, seed=SPLIT_SEED)
json.dump({str(k): v for k, v in folds.items()}, open(FOLDS_JSON, 'w'), indent=1)
meta[['subject', 'cohort', 'rec', 'night', 'fold']].sort_values(['fold', 'subject', 'rec']) \
    .to_csv(f'{RESULTS}/fold_manifest_subject_level.csv', index=False)
json.dump(prep.fold_role_lists(meta, K_FOLDS), open(f'{RESULTS}/fold_roles_train_val_test.json', 'w'), indent=1)
assert meta.groupby('subject').fold.nunique().max() == 1
stage_cols = [f'trim_{s}' for s in skd.STAGES]
summ = meta.groupby('fold').agg(subjects=('subject', 'nunique'), recordings=('rec', 'count'),
                                SC=('cohort', lambda c: (c == 'SC').sum()), ST=('cohort', lambda c: (c == 'ST').sum()),
                                **{s: (s, 'sum') for s in stage_cols})
summ.loc['total'] = summ.sum()
summ
""")

nb.md("## 2. Huấn luyện\n### 2a. Nạp dữ liệu vào RAM (copy Drive → local trước để đọc nhanh)")
nb.code(r"""
t0 = time.time()
shutil.copytree(DATA_DIR, LOCAL_DATA, dirs_exist_ok=True)
folds = {int(k): v for k, v in json.load(open(FOLDS_JSON)).items()}
store = skd.Store(LOCAL_DATA, sorted(r for v in folds.values() for r in v))
print(f'{len(store.recs)} recordings in RAM | full-night centres stored: {store.has_full()} | {time.time() - t0:.0f}s')
""")
nb.md("### 2b. Số tham số và MACs (đo thật — số dùng trong bài; cột thop chỉ để đối chiếu với bản cũ)")
nb.code(r"""
from thop import profile
from torch.utils.flop_counter import FlopCounterMode
rows = []
x1 = torch.randn(1, skd.SEQ_LEN, 3, 3000)
for kind, tdim in [('teacher', None), ('assistant', 256), ('edge', 128)]:
    m = skd.build_model(kind, tdim)
    total, infer = skd.count_params(m)
    macs_thop, thop_params = profile(m, inputs=(x1,), verbose=False)
    # MACs đo bằng FlopCounter của PyTorch: train mode để không đi fast path (fused) của Transformer,
    # trừ adaptor (chỉ dùng khi train), cộng GRU tính theo công thức (counter không thấy op GRU)
    m.train()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x1)
    ad = sum(l.in_features * l.out_features for n, l in m.named_modules()
             if 'adaptor' in n and isinstance(l, torch.nn.Linear))
    gru = skd.SEQ_LEN * 3 * (m.gru.input_size * m.gru.hidden_size + m.gru.hidden_size ** 2) if kind == 'edge' else 0
    rows.append(dict(model=kind, params_total=total, params_inference=infer, params_thop=int(thop_params),
                     MACs_inference_M=round((fc.get_total_flops() / 2 - ad + gru) / 1e6, 2),
                     MACs_thop_M=round(macs_thop / 1e6, 2)))
params_df = pd.DataFrame(rows); params_df.to_csv(f'{RESULTS}/param_counts.csv', index=False)
params_df
""")
nb.md("### 2c. Train tất cả nhánh (theo thứ tự trong `ARMS`)\n"
      "Chạy lại cell này sau mỗi lần mất kết nối — phần đã xong sẽ được bỏ qua. "
      "`T_ce` chỉ chạy được sau khi đủ 6 run `Ttune_ce_*` ở fold 0.")
nb.code(r"""
skd.run_all(C, store, folds, device, run_folds=RUN_FOLDS, run_arms=RUN_ARMS)
""")

nb.md("## 3. Kết quả\n### 3a. Tune Teacher-CE (validation fold 0) và bảng mô tả theo fold")
nb.code(r"""
tune = skd.tuning_table(C); tune.to_csv(f'{RESULTS}/teacher_ce_tuning.csv', index=False)
display(tune.sort_values('best_val_mf1', ascending=False))
df = skd.collect(C)
df.to_csv(f'{RESULTS}/fold_level_results.csv', index=False)
main_arms = [a['name'] for a in ARMS if 'folds' not in a]
tab = skd.summary(df[df.arm.isin(main_arms)], 'test')       # mean ± SD qua fold (mô tả)
tab.to_csv(f'{RESULTS}/summary_fold_mean_sd.csv')
tab
""")
nb.md("### 3b. Phân tích chính — Macro-F1 gộp out-of-fold, bootstrap theo cụm subject, BH trong family chính")
nb.code(r"""
main_arms = [a['name'] for a in ARMS if 'folds' not in a]   # tự định nghĩa, không phụ thuộc cell 3a
pooled = skd.pooled_table(C, main_arms); pooled.to_csv(f'{RESULTS}/pooled_oof_metrics.csv')
display(pooled.round(2))
ct = skd.cluster_contrasts(C, PRIMARY_CONTRASTS, n_boot=10000)
ct.to_csv(f'{RESULTS}/primary_cluster_bootstrap.csv', index=False)
display(ct[ct.family == 'primary'].round(4))
ct[ct.family == 'secondary'].round(4)
""")
nb.md("### 3c. Phân tích độ nhạy — Wilcoxon trên Macro-F1 theo từng subject")
nb.code(r"""
sens = skd.subject_wilcoxon(C, PRIMARY_CONTRASTS)
sens.to_csv(f'{RESULTS}/sensitivity_subject_wilcoxon.csv', index=False)
sens.round(4)
""")
nb.md("### 3d. Lưu cấu hình đã dùng (đính kèm khi trả lời reviewer)")
nb.code(r"""
cfg = dict(C, ARMS=ARMS, PRIMARY_CONTRASTS=PRIMARY_CONTRASTS, SPLIT_SEED=SPLIT_SEED, TRIM_WAKE_MIN=TRIM_WAKE_MIN,
           STORE_FULL_NIGHT=STORE_FULL_NIGHT, REJECT_PTP_UV=REJECT_PTP_UV, channels=prep.CHANNELS,
           filter='Butterworth order 5, 0.5-35 Hz, filtfilt per epoch', normalisation='per-epoch per-channel z-score',
           torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
json.dump(cfg, open(f'{RESULTS}/run_config.json', 'w'), indent=1, default=str)
print(os.listdir(RESULTS))
""")
nb.md(r"""
### 3e. Phân tích độ nhạy: loại các subject của nhóm F1
Lưới tune Teacher-CE chỉ chạy ở run fold 0 và chọn cấu hình theo validation của run đó, tức **các subject nhóm F1**.
Nhóm F1 lại là **tập test của run fold 1**, nên kết quả test fold 1 của `T_ce`, `A_kd_Tce`, `E_direct_Tce`,
`E_staged_Tce` có thể bị thiên lệch nhẹ do chọn cấu hình. Phân tích này lặp lại đúng phân tích chính (3b) sau khi
**loại toàn bộ subject nhóm F1 khỏi mọi nhánh** (các cặp so sánh vẫn cùng một tập subject). Phân tích được bổ sung
sau khi phát hiện vấn đề trong thiết kế tune; báo cáo song song với phân tích chính.
""")
nb.code(r"""
folds = {int(k): v for k, v in json.load(open(FOLDS_JSON)).items()}
f1_subjects = sorted({r[:5] for r in folds[1]})
print(len(f1_subjects), 'subject bị loại (nhóm F1):', f1_subjects)
main_arms = [a['name'] for a in ARMS if 'folds' not in a]
pooled_x = skd.pooled_table(C, main_arms, exclude_subjects=f1_subjects)
pooled_x.to_csv(f'{RESULTS}/sensitivity_exclF1_pooled.csv')
display(pooled_x.round(2))
ct_x = skd.cluster_contrasts(C, PRIMARY_CONTRASTS, n_boot=10000, exclude_subjects=f1_subjects)
ct_x.to_csv(f'{RESULTS}/sensitivity_exclF1_cluster_bootstrap.csv', index=False)
display(ct_x[ct_x.family == 'primary'].round(4))
ct_x[ct_x.family == 'secondary'].round(4)
""")
nb.md(r"""
### 3f. Phân tích chính (chốt 2026-10-15): lai 100 / 79 subject
So sánh có dính tới nhánh phụ thuộc cấu hình đã tune (`T_ce` và mọi nhánh distill từ nó) lấy kết quả **khi loại nhóm F1**
(3e); các so sánh còn lại lấy kết quả **đủ subject** (3b). Các dòng được chép nguyên từ 3b/3e, chỉ tính lại BH-FDR
trong từng family trên tập so sánh gộp. Cột `n_subjects` cho biết mỗi so sánh dùng bao nhiêu subject. Cần chạy 3b và 3e trước.
""")
nb.code(r"""
dep = skd.tuned_dependent_arms(ARMS)
print('nhánh phụ thuộc cấu hình đã tune:', sorted(dep))
ct_h = skd.hybrid_contrasts(ct, ct_x, dep)
ct_h.to_csv(f'{RESULTS}/primary_hybrid_cluster_bootstrap.csv', index=False)
display(ct_h[ct_h.family == 'primary'].round(4))
ct_h[ct_h.family == 'secondary'].round(4)
""")
nb.save(os.path.join(OUT_DIR, 'R2_subject_level_rerun.ipynb'))

# =============================================================================== notebook 2: HMC + cross-dataset
nb = NB()
nb.md(r"""
# HMC — in-domain (subject-level 5-fold) và zero-shot EDF ↔ HMC (revision vòng 2)

Dựng lại HMC từ dữ liệu gốc với **cùng giao thức** như notebook Sleep-EDF (`R2_subject_level_rerun.ipynb`), rồi đánh giá chéo hai chiều:
- **Sleep-EDF → HMC:** mỗi model fold của các nhánh EDF dự đoán toàn bộ HMC.
- **HMC → Sleep-EDF:** mỗi model fold của các nhánh HMC dự đoán toàn bộ Sleep-EDF.

**Dữ liệu:** HMC sleep staging database **v1.1** (151 recording, mỗi recording một bệnh nhân) từ bản sao AWS của PhysioNet. Bản thảo cũ trích dẫn v1.0.1 (154 recording) nhưng code cũ thực tế dùng v1.1 — đổi `HMC_VERSION` nếu quyết định khác. Kênh: F4-M1, O2-M1, E1−E2 (EOG ngang suy ra), 256 → 100 Hz (`resample_poly` 25/64, theo từng epoch như code cũ), nhãn AASM từ `SNxxx_sleepscoring.txt`.

**Cần có trước:** notebook Sleep-EDF đã chạy xong 1b–1c (dữ liệu EDF trên Drive) và các nhánh EDF trong `CROSS_ARMS` đã train xong (cho chiều EDF→HMC).
""" + PROTOCOL_MD)
nb.code("# ===================== CONFIG — chỉ sửa ở đây =====================" + COMMON_CONFIG + r"""
# HMC
HMC_VERSION = '1.1'
HMC_RAW     = '/content/hmc_raw'                    # EDF gốc (xoá ngay sau khi xử lý)
HMC_DATA    = f'{DRIVE_ROOT}/data_hmc'
HMC_LOCAL   = '/content/data_hmc'
HMC_META    = f'{DRIVE_ROOT}/recordings_hmc.csv'
HMC_FOLDS   = f'{DRIVE_ROOT}/folds_hmc_subject_level.json'
HMC_RUNS    = f'{DRIVE_ROOT}/runs_hmc'
for d in [HMC_DATA, HMC_RUNS, HMC_RAW]:
    os.makedirs(d, exist_ok=True)

ARMS_HMC = [   # model chính (chốt 2026-10-15): chuỗi Teacher-CE đã tune -> Assistant -> Edge (staged)
    # lr / label smoothing của T_ce KHÔNG tune lại trên HMC: lấy đúng cấu hình đã chọn trên Sleep-EDF (đọc ở cell 2b)
    dict(name='T_ce',         kind='teacher',   loss='ce', bs=64),
    dict(name='A_kd_Tce',     kind='assistant', loss='kd', src='T_ce',     T=5.0, bs=32),
    dict(name='E_staged_Tce', kind='edge',      loss='kd', src='A_kd_Tce', T=4.0, bs=64),
]
CROSS_ARMS = ['T_ce', 'A_kd_Tce', 'E_staged_Tce']   # đánh giá zero-shot cả hai chiều
C_HMC = dict(TRAIN_CFG, K_FOLDS=K_FOLDS, RUNS_DIR=HMC_RUNS, ARMS=ARMS_HMC, SEEDS=SEEDS)
C_EDF = dict(TRAIN_CFG, K_FOLDS=K_FOLDS, RUNS_DIR=RUNS_DIR, ARMS=[], SEEDS=SEEDS)
print('config OK')
""")
nb.code_files()
nb.md("## 1. Dữ liệu HMC (chạy được trên runtime CPU; ~15.7 GB tải về, mỗi file EDF bị xoá ngay sau khi xử lý)")
nb.code(r"""
import sys; sys.path.insert(0, CODE_DIR)
import pandas as pd, prep
hmeta = prep.download_process_hmc(HMC_RAW, HMC_DATA, HMC_META, version=HMC_VERSION, trim_wake_min=TRIM_WAKE_MIN,
                                  store_full_night=STORE_FULL_NIGHT, reject_ptp_uv=REJECT_PTP_UV)
print(len(hmeta), 'recordings | sfreq:', hmeta.sfreq_orig.value_counts().to_dict(),
      '| recordings có epoch không liên tục:', int((hmeta.n_gaps > 0).sum()))
print('kênh dùng:', hmeta.channels.value_counts().to_dict())
hmeta.head()
""")
nb.md("### 1b. Chia 5 fold theo subject (seed cố định) và xuất danh sách train/val/test")
nb.code(r"""
hmeta = pd.read_csv(HMC_META)
hmeta, hfolds = prep.make_subject_folds(hmeta, k=K_FOLDS, seed=SPLIT_SEED)
json.dump({str(k): v for k, v in hfolds.items()}, open(HMC_FOLDS, 'w'), indent=1)
hmeta[['subject', 'rec', 'fold']].sort_values(['fold', 'rec']).to_csv(f'{RESULTS}/hmc_fold_manifest.csv', index=False)
json.dump(prep.fold_role_lists(hmeta, K_FOLDS), open(f'{RESULTS}/hmc_fold_roles_train_val_test.json', 'w'), indent=1)
stage_cols = [f'trim_{s}' for s in skd.STAGES]
hsum = hmeta.groupby('fold').agg(subjects=('subject', 'nunique'), **{s: (s, 'sum') for s in stage_cols})
hsum.loc['total'] = hsum.sum()
hsum
""")
nb.md("## 2. HMC in-domain\n### 2a. Nạp HMC vào RAM")
nb.code(r"""
t0 = time.time()
shutil.copytree(HMC_DATA, HMC_LOCAL, dirs_exist_ok=True)
hfolds = {int(k): v for k, v in json.load(open(HMC_FOLDS)).items()}
hmc_recs = sorted(r for v in hfolds.values() for r in v)
hstore = skd.Store(HMC_LOCAL, hmc_recs)
print(f'{len(hstore.recs)} HMC recordings in RAM | {time.time() - t0:.0f}s')
""")
nb.md("### 2b. Train các nhánh HMC")
nb.code(r"""
# T_ce dùng đúng lr / label smoothing đã chọn trên Sleep-EDF (validation fold 0), không tune lại trên HMC
tce = skd.load_result(C_EDF, 'T_ce', 0, SEEDS[0])
assert tce is not None, 'T_ce trên Sleep-EDF chưa train (cần runs/ của notebook EDF trên Drive)'
for a in ARMS_HMC:
    if a['name'] == 'T_ce':
        a.update(lr=tce['lr'], ls=tce['ls'])
print(f"T_ce trên HMC: lr {tce['lr']}, label smoothing {tce['ls']} (chọn trên Sleep-EDF: {tce['selected_from']})")
skd.run_all(C_HMC, hstore, hfolds, device, run_folds=RUN_FOLDS)
""")
nb.md("### 2c. Kết quả HMC in-domain (mô tả theo fold + Macro-F1 gộp out-of-fold)")
nb.code(r"""
hdf = skd.collect(C_HMC); hdf.to_csv(f'{RESULTS}/hmc_fold_level_results.csv', index=False)
display(skd.summary(hdf, 'test'))
hp = skd.pooled_table(C_HMC, [a['name'] for a in ARMS_HMC]); hp.to_csv(f'{RESULTS}/hmc_pooled_oof_metrics.csv')
hp.round(2)
""")
nb.md("## 3. Zero-shot cross-dataset\n### 3a. Nạp Sleep-EDF vào RAM")
nb.code(r"""
t0 = time.time()
shutil.copytree(DATA_DIR, LOCAL_DATA, dirs_exist_ok=True)
efolds = {int(k): v for k, v in json.load(open(FOLDS_JSON)).items()}
edf_recs = sorted(r for v in efolds.values() for r in v)
estore = skd.Store(LOCAL_DATA, edf_recs)
print(f'{len(estore.recs)} Sleep-EDF recordings in RAM | {time.time() - t0:.0f}s')
""")
nb.md("### 3b. Đánh giá hai chiều (mỗi model fold dự đoán **toàn bộ** dataset đích; mean ± SD qua 5 model fold)")
nb.code(r"""
rows = []
for seed in SEEDS:
    for arm in CROSS_ARMS:
        rows.append(skd.eval_cross(C_EDF, arm, seed, hstore, hmc_recs, 'HMC', device).assign(direction='SleepEDF->HMC'))
        rows.append(skd.eval_cross(C_HMC, arm, seed, estore, edf_recs, 'SleepEDF', device).assign(direction='HMC->SleepEDF'))
xdf = pd.concat(rows, ignore_index=True); xdf.to_csv(f'{RESULTS}/cross_dataset_fold_level.csv', index=False)
cols = ['acc', 'mf1', 'kappa'] + [f'f1_{s}' for s in skd.STAGES]
xsum = xdf.groupby(['direction', 'arm'])[cols].agg(['mean', 'std']).round(4)
xsum.to_csv(f'{RESULTS}/cross_dataset_summary.csv')
xsum
""")
nb.save(os.path.join(OUT_DIR, 'R2_hmc_crossdataset.ipynb'))

# =============================================================================== notebook 3: export / checks
nb = NB()
nb.md(r"""
# Xuất số liệu và kiểm tra kết quả (chỉ đọc, chạy trên runtime CPU)

Notebook này **không train gì**. Nó đọc `result.json` và `preds_test.npz` trên Drive để:
1. Lập danh mục mọi run (Sleep-EDF và HMC): xong, thiếu, đang dở.
2. Lập bảng **val Macro-F1** theo nhánh (dùng để chốt Teacher chính và Edge nạp lên ESP32 — **không dùng số test**).
3. Kiểm tra hội tụ (run chạm số epoch tối đa, model đoán dồn vào một lớp).
4. Kiểm tra nhất quán (mọi nhánh cùng tập epoch test; số epoch test mỗi fold khớp `AUDITED_NUMBERS`).
5. Đóng gói `results_export.zip` (không gồm checkpoint `.pt`) để đưa về project — mọi con số trong bài sẽ được script tính lại từ gói này.
""")
nb.code(r"""
import os, json, glob, hashlib, zipfile
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass
DRIVE_ROOT = os.environ.get('SKD_ROOT', '/content/drive/MyDrive/sleep_kd_r2')
RESULTS = f'{DRIVE_ROOT}/results'
os.makedirs(RESULTS, exist_ok=True)
DATASETS = {'SleepEDF': f'{DRIVE_ROOT}/runs', 'HMC': f'{DRIVE_ROOT}/runs_hmc'}
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
MAX_EPOCHS = 80
# số epoch test mỗi fold theo AUDITED_NUMBERS (tính độc lập từ dữ liệu gốc)
EXPECTED_TEST = {'SleepEDF': [53332, 48522, 47935, 44604, 43543], 'HMC': [28476, 27492, 27644, 25808, 26591]}
print({k: os.path.isdir(v) for k, v in DATASETS.items()})
""")
nb.md("## 1. Danh mục run")
nb.code(r"""
rows, in_progress = [], []
for ds, root in DATASETS.items():
    for d in sorted(glob.glob(f'{root}/seed*/*/fold*')):
        parts = os.path.normpath(d).split(os.sep)
        seed, arm, fold = parts[-3][4:], parts[-2], int(parts[-1][4:])
        rp = f'{d}/result.json'
        if not os.path.exists(rp):
            if os.path.exists(f'{d}/last.pt'):
                in_progress.append((ds, arm, fold))
            continue
        r = json.load(open(rp)); t = r['test']
        rows.append(dict(dataset=ds, seed=int(seed), arm=arm, fold=fold, kind=r['kind'], loss=r['loss'], src=r['src'],
                         T=r.get('T'), lr=r['lr'], ls=r.get('ls'), bs=r['bs'], selected_from=r.get('selected_from'),
                         best_val_mf1=r['best_val_mf1'], best_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                         minutes=round(r['minutes'], 1), n_train=r['n_train'], n_val=r['n_val'], n_test=r['n_test'],
                         params_inference=r['params_inference'], test_acc=t['acc'], test_mf1=t['mf1'],
                         test_kappa=t['kappa'], test_n1=t['f1'][1]))
inv = pd.DataFrame(rows)
inv.to_csv(f'{RESULTS}/run_inventory.csv', index=False)
print('runs done:', len(inv), '| in progress (có last.pt):', in_progress)
cov = inv.pivot_table(index=['dataset', 'arm'], columns='fold', values='seed', aggfunc='count').fillna(0).astype(int)
cov
""")
nb.md("## 2. Validation Macro-F1 theo nhánh (dùng để chốt lựa chọn — không dùng test)\n"
      "Cột `mean_folds1_4` loại fold 0 vì ở fold 0 `T_ce` là cấu hình được chọn từ lưới tune trên chính tập validation đó "
      "(val bị đẩy cao do chọn lọc).")
nb.code(r"""
v = inv[~inv.arm.str.startswith('Ttune_') & (inv.seed == 42)].pivot_table(index=['dataset', 'kind', 'arm'], columns='fold', values='best_val_mf1')
v['mean_5'] = v[[c for c in v.columns if isinstance(c, (int, np.integer))]].mean(1)
v['mean_folds1_4'] = v[[c for c in v.columns if isinstance(c, (int, np.integer)) and c > 0]].mean(1)
v['n_folds'] = v[[c for c in v.columns if isinstance(c, (int, np.integer))]].notna().sum(1)
v = (v * [100 if isinstance(c, (int, np.integer)) or c.startswith('mean') else 1 for c in v.columns]).round(2)
v.to_csv(f'{RESULTS}/validation_mf1_by_arm.csv')
display(v)
tune = inv[inv.arm.str.startswith('Ttune_')][['dataset', 'arm', 'lr', 'ls', 'best_val_mf1', 'best_epoch', 'epochs_run']]
tune.sort_values('best_val_mf1', ascending=False)
""")
nb.md("## 3. Kiểm tra hội tụ và dự đoán")
nb.code(r"""
# (a) run chạm số epoch tối đa mà epoch tốt nhất nằm ở cuối -> có thể thiếu ngân sách epoch
cap = inv[(inv.epochs_run >= MAX_EPOCHS) & (inv.best_epoch >= MAX_EPOCHS - 10)]
print('run chạm trần epoch và vẫn đang cải thiện:', len(cap)); display(cap[['dataset', 'arm', 'fold', 'best_epoch', 'epochs_run']])
# (b) phân bố lớp dự đoán trên test: phát hiện model đoán dồn vào một lớp
pr = []
for ds, root in DATASETS.items():
    for _, r in inv[inv.dataset == ds].iterrows():
        p = f'{root}/seed{r.seed}/{r.arm}/fold{r.fold}/preds_test.npz'
        d = np.load(p)
        pred = d['logits'].astype(np.float32).argmax(1)
        frac = np.bincount(pred, minlength=5) / len(pred)
        true = np.bincount(d['y'].astype(int), minlength=5) / len(d['y'])
        pr.append(dict(dataset=ds, arm=r.arm, fold=r.fold, **{f'pred_{s}': round(100 * f, 2) for s, f in zip(STAGES, frac)},
                       **{f'true_{s}': round(100 * f, 2) for s, f in zip(STAGES, true)},
                       min_pred_pct=round(100 * frac.min(), 2), nan_logits=bool(np.isnan(d['logits'].astype(np.float32)).any())))
pr = pd.DataFrame(pr); pr.to_csv(f'{RESULTS}/prediction_distribution.csv', index=False)
print('run có lớp dự đoán < 1%:', int((pr.min_pred_pct < 1).sum()), '| run có NaN:', int(pr.nan_logits.sum()))
pr.sort_values('min_pred_pct').head(10)
""")
nb.md("## 4. Kiểm tra nhất quán")
nb.code(r"""
issues = []
for ds, root in DATASETS.items():
    sub = inv[inv.dataset == ds]
    for (seed, fold), g in sub.groupby(['seed', 'fold']):
        ref = None
        for _, r in g.iterrows():
            d = np.load(f'{root}/seed{seed}/{r.arm}/fold{fold}/preds_test.npz')
            key = (d['rec'].tolist(), d['center'].tolist(), d['y'].tolist())
            if ref is None:
                ref, ref_arm = key, r.arm
            elif key != ref:
                issues.append(f'{ds} seed{seed} fold{fold}: {r.arm} test epochs differ from {ref_arm}')
        exp = EXPECTED_TEST[ds][fold]
        bad = g[g.n_test != exp]
        if len(bad):
            issues.append(f'{ds} fold{fold}: n_test {sorted(set(bad.n_test))} != expected {exp} ({sorted(bad.arm)})')
print('OK — mọi nhánh cùng tập epoch test và khớp AUDITED_NUMBERS' if not issues else '\n'.join(issues))
""")
nb.md("## 5. Đóng gói để đưa về project")
nb.code(r"""
out = f'{DRIVE_ROOT}/results_export.zip'
n = 0
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for ds, root in DATASETS.items():
        for p in glob.glob(f'{root}/**/*', recursive=True):
            if os.path.isfile(p) and (p.endswith(('result.json', 'preds_test.npz')) or os.path.basename(p).startswith('cross_')):
                z.write(p, os.path.relpath(p, DRIVE_ROOT)); n += 1
    for p in glob.glob(f'{RESULTS}/*') + glob.glob(f'{DRIVE_ROOT}/*.csv') + glob.glob(f'{DRIVE_ROOT}/*.json'):
        if os.path.isfile(p):
            z.write(p, os.path.relpath(p, DRIVE_ROOT)); n += 1
sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
print(f'{n} files | {os.path.getsize(out) / 1e6:.1f} MB | sha256 {sha}')
print('Tải file này về và đặt vào revision/R2_subject_level/ ; gửi kèm dòng sha256 ở trên.')
""")
nb.md("## 6. Đóng gói checkpoint cho Zenodo (chỉ `best.pt`, không lấy `last.pt`)")
nb.code(r"""
out = f'{DRIVE_ROOT}/checkpoints_export.zip'
n = 0
with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:          # .pt is already compressed
    for ds, root in DATASETS.items():
        for p in sorted(glob.glob(f'{root}/seed*/*/fold*/best.pt')):
            z.write(p, 'checkpoints/' + os.path.relpath(p, DRIVE_ROOT)); n += 1
sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
print(f'{n} checkpoints | {os.path.getsize(out) / 1e6:.1f} MB | sha256 {sha}')
""")
nb.save(os.path.join(OUT_DIR, 'R2_export_numbers.ipynb'))

# =============================================================================== notebook 4: XAI v2 on DREAMS
nb = NB()
nb.md(r"""
# XAI v2 — DREAMS K-complex / spindle (R1: đơn vị thống kê, baseline, kiểm tra model ngẫu nhiên; R2.2)

**Chạy SAU khi** notebook Sleep-EDF đã xong và notebook HMC đã train xong (cần `runs/` và `runs_hmc/` trên Drive). GPU (A100).

**Dữ liệu** (tài liệu DREAMS, Zenodo 10.5281/zenodo.2650142; notebook tự tải và kiểm tra md5):
- K-complex: 10 excerpt 30 phút, người khỏe, CZ-A1 200 Hz. Spindle: 8 excerpt 30 phút, bệnh nhân, C3-A1 (excerpt 1, 3) hoặc CZ-A1, 100/200/50 Hz.
- Hypnogram: **1 giá trị / 5 s**, R&K: 5 W, 4 REM, 3 S1, 2 S2, 1 S3, 0 S4, −1 movement, −2/−3 unknown (code cũ đọc sai: 7.5 s và đảo S1/S3). Nhãn giai đoạn chỉ dùng để mô tả.
- Sự kiện: chuyên gia 1 (`Visual_scoring1`, chấm đủ mọi excerpt).
- Tiền xử lý **giống hệt lúc train**: resample về 100 Hz → Butterworth 0.5–35 Hz theo epoch → z-score theo epoch; kênh được chấm **nhân bản vào cả 3 ô đầu vào** (EEG, EEG, EOG) → ngoài phân phối lúc train. Code cũ không lọc.
- Cửa sổ 21 epoch quanh mỗi epoch mục tiêu, trong cùng excerpt, biên lặp epoch đầu/cuối (như `skd.SeqDS`). Epoch mục tiêu = epoch 30 s có ít nhất một mẫu thuộc sự kiện.

**Attribution:** logit của lớp model dự đoán cho epoch trung tâm; gradient và Integrated Gradients (baseline 0); lấy epoch trung tâm, |·|, cộng 3 ô đầu vào; model ở **eval mode** (code cũ để train mode → dropout bật). IG tích phân trên **lưới α log (0 và 63 điểm 1e-6…1, hình thang)** vì model chuẩn hoá sớm (GroupNorm/LayerNorm) nên logit đổi gần như hoàn toàn ở α rất nhỏ; lưới đều 50 điểm không thỏa completeness (cell 3b đo cả hai).

**AUROC theo epoch** (bản đồ attribution so với mặt nạ sự kiện theo mẫu) → trung bình trong excerpt → trung bình qua 5 model fold. **Đơn vị thống kê = excerpt** (n = 10 / 8).

**Kiểm định (chốt trước, cell 2):** Q1 AUROC vs 0.5; Q2 model vs baseline không cần model (|x|, envelope Hilbert; thêm công suất dải sigma 11–16 Hz cho spindle); Q3 model đã train vs cùng kiến trúc khởi tạo ngẫu nhiên (5 seed). Wilcoxon signed-rank (cặp) trên excerpt; median + CI bootstrap 95%; BH-FDR trong từng family (câu hỏi × loại sự kiện). Không có ý nghĩa thống kê ≠ "tương đương". Chỉ diễn giải về liên hệ attribution – đoạn được gán nhãn.
""")
nb.code("# ===================== CONFIG — chỉ sửa ở đây =====================" + COMMON_CONFIG + r"""
# XAI v2
HMC_RUNS    = f'{DRIVE_ROOT}/runs_hmc'
DREAMS_RAW  = '/content/dreams_raw'
DREAMS_DATA = f'{DRIVE_ROOT}/data_dreams'
XAI_DIR     = f'{RESULTS}/xai_v2'                 # kết quả theo epoch (resume được)
for d in [DREAMS_DATA, XAI_DIR]:
    os.makedirs(d, exist_ok=True)
IG_STEPS, IG_GRID, N_RANDOM, XAI_BS = 64, 'log', 5, 32
SPECS = {   # model chính (plan 0.6) + Edge-Direct (R1: thiếu trong Hình 2b)
    'Assistant (Sleep-EDF)':   (RUNS_DIR, 'A_kd_Tce'),
    'Edge-Staged (Sleep-EDF)': (RUNS_DIR, 'E_staged_Tce'),
    'Edge-Direct (Sleep-EDF)': (RUNS_DIR, 'E_direct_Tce'),
    'Assistant (HMC)':         (HMC_RUNS, 'A_kd_Tce'),
    'Edge-Staged (HMC)':       (HMC_RUNS, 'E_staged_Tce'),
}
RANDOM_SPECS = {'assistant': 'Assistant (random init)', 'edge': 'Edge (random init)'}
RANDOM_OF = {s: RANDOM_SPECS['assistant' if s.startswith('Assistant') else 'edge'] for s in SPECS}
print('config OK')
""")
nb.code_files()
nb.code("!which unrar > /dev/null || apt-get -qq install -y unrar")
nb.md("## 1. Dữ liệu DREAMS (CPU được) — bảng metadata theo excerpt (B1)")
nb.code(r"""
import xai, glob; importlib.reload(xai)
meta_csv = f'{RESULTS}/xai_dreams_excerpts.csv'
if not os.path.exists(meta_csv):
    dirs = xai.download_dreams(DREAMS_RAW)
    xai.process_all(dirs, DREAMS_DATA).to_csv(meta_csv, index=False)
meta = pd.read_csv(meta_csv)
assert len(meta) == 18 and (meta.n_epochs == 60).all() and (meta.hyp_values == 360).all()
meta
""")
nb.md("## 2. Danh sách kiểm định chốt trước (in ra TRƯỚC khi chạy model)")
nb.code(r"""
plan = xai.planned_tests(list(SPECS), RANDOM_OF)
plan.to_csv(f'{RESULTS}/xai_planned_tests.csv', index=False)
print(plan.groupby('family').size().to_string())
plan
""")
nb.md("## 3. Attribution\n### 3a. Nạp dữ liệu + baseline không cần model")
nb.code(r"""
store = skd.Store(DREAMS_DATA, list(meta.rec))
mask_of = {r: np.load(f'{DREAMS_DATA}/{r}.npz')['mask'] for r in meta.rec}
base_csv = f'{XAI_DIR}/baselines.csv'
if not os.path.exists(base_csv):
    xai.run_baselines(store, list(meta.rec), mask_of).to_csv(base_csv, index=False)
base = pd.read_csv(base_csv)
base.groupby([base.rec.str.split('_').str[0], 'method']).auroc.mean().round(3)
""")
nb.md("### 3b. Kiểm tra completeness của IG (lưới đều 50 điểm vs lưới log 64 điểm), model fold 0\n"
      "Model nào chưa train đủ 5 fold (ví dụ HMC đang train) thì **bỏ qua**; chạy lại 3b–3c sau khi train xong, "
      "các model đã làm được giữ nguyên.")
nb.code(r"""
READY = {s: xai.spec_ready(runs, arm, K_FOLDS, SEEDS[0]) for s, (runs, arm) in SPECS.items()}
print('chưa train xong (bỏ qua lần này):', [s for s, ok in READY.items() if not ok] or 'không có')
comp_csv = f'{RESULTS}/xai_ig_completeness.csv'
rows = pd.read_csv(comp_csv).to_dict('records') if os.path.exists(comp_csv) else []
done = {r['spec'] for r in rows}
for s, (runs, arm) in SPECS.items():
    if not READY[s] or s in done:
        continue
    m, _ = xai.load_model(runs, arm, 0, SEEDS[0], device)
    for rec in ['kcomplexes_excerpt1', 'spindles_excerpt1']:
        rows.append(dict(spec=s, rec=rec, **xai.completeness_check(m, store, rec, device)))
comp = pd.DataFrame(rows); comp.to_csv(comp_csv, index=False)
comp.round(4)
""")
nb.md("### 3c. Chạy attribution cho mọi model fold và model ngẫu nhiên (resume theo từng model)")
nb.code(r"""
def _fname(mid):
    return f"{XAI_DIR}/{''.join(ch if ch.isalnum() else '_' for ch in mid)}.csv"

jobs = [(s, f'{s}|fold{f}', ('trained', runs, arm, f)) for s, (runs, arm) in SPECS.items() for f in range(K_FOLDS)
        if READY[s]]
jobs += [(s, f'{s}|seed{k}', ('random', kind, k)) for kind, s in RANDOM_SPECS.items() for k in range(N_RANDOM)]
for spec, mid, src in jobs:
    p = _fname(mid)
    if os.path.exists(p):
        continue
    t0 = time.time()
    if src[0] == 'trained':
        model, _ = xai.load_model(src[1], src[2], src[3], SEEDS[0], device)
    else:
        model = xai.random_model(src[1], 10_000 + src[2], device)
    df = xai.run_model(model, store, list(meta.rec), mask_of, device, batch=XAI_BS, ig_steps=IG_STEPS)
    df.assign(spec=spec, model_id=mid).to_csv(p, index=False)
    print(f'{mid}: {len(df)} rows | IG completeness median {df.ig_rel_err.median():.4f} | {time.time() - t0:.0f}s')
""")
nb.md("## 4. Thống kê theo excerpt (Q1–Q3, BH theo family)")
nb.code(r"""
ep = pd.concat([pd.read_csv(p) for p in glob.glob(f'{XAI_DIR}/*.csv') if not p.endswith('baselines.csv')])
ep['db'] = ep.rec.str.split('_').str[0]
missing = (set(SPECS) | set(RANDOM_SPECS.values())) - set(ep.spec)
assert not missing, f'chưa có attribution cho {sorted(missing)} — đợi train xong rồi chạy lại 3b, 3c, rồi cell này'
exc_m = xai.excerpt_level(ep, ['spec', 'method', 'db'])
b = base.assign(model_id='none', spec=base.method, db=base.rec.str.split('_').str[0])
exc_b = xai.excerpt_level(b, ['spec', 'method', 'db'])
exc_m.to_csv(f'{RESULTS}/xai_excerpt_level_models.csv', index=False)
exc_b.to_csv(f'{RESULTS}/xai_excerpt_level_baselines.csv', index=False)
res = xai.run_tests(plan, exc_m, exc_b)
res.to_csv(f'{RESULTS}/xai_tests.csv', index=False)
desc = (pd.concat([exc_m, exc_b]).groupby(['db', 'spec', 'method']).auroc
        .agg(median='median', q25=lambda v: v.quantile(.25), q75=lambda v: v.quantile(.75), n='size').round(3))
desc.to_csv(f'{RESULTS}/xai_auroc_summary.csv')
display(desc)
pred = (ep[ep.method == 'ig'].groupby(['db', 'spec']).pred.value_counts(normalize=True).unstack(fill_value=0)
        .sort_index(axis=1).round(3))
pred.columns = [skd.STAGES[c] for c in pred.columns]
pred.to_csv(f'{RESULTS}/xai_predicted_stage_event_epochs.csv'); display(pred)
res.round(4)
""")
nb.md("### 4b. Hình 2b — AUROC theo excerpt cho mọi model (kể cả Edge-Direct), model ngẫu nhiên và baseline")
nb.code(r"""
xai.plot_auroc(exc_m, exc_b, SPECS, RANDOM_SPECS.values(), f'{RESULTS}/xai_fig2b_auroc.png');
""")
nb.md(r"""
### 4c. Độ nhạy của IG với lưới tích phân (chỉ chạy lại model có completeness kém)
Model fold nào có sai số completeness trung vị > 0.10 được chạy lại IG với lưới mịn hơn (256 điểm, α từ 1e-9). So sánh
completeness, AUROC theo epoch, và chạy lại các kiểm định sau khi thay kết quả IG của các model đó. Nếu không kiểm định nào
đổi kết luận (q < 0.05 hay không) thì giữ kết quả chính (lưới 64 điểm) và báo cáo bảng này trong Supplementary.
""")
nb.code(r"""
FINE_STEPS, FINE_LO, FLAG = 256, 1e-9, 0.10
FINE_DIR = f'{XAI_DIR}_fine'; os.makedirs(FINE_DIR, exist_ok=True)
comp_m = ep[(ep.method == 'ig') & ep.spec.isin(list(SPECS))].groupby(['spec', 'model_id']).ig_rel_err.median()
flagged = [(s, mid) for (s, mid), v in comp_m.items() if v > FLAG]
print('model chạy lại:', [(mid, round(comp_m[(s, mid)], 3)) for s, mid in flagged])
for s, mid in flagged:
    p = f"{FINE_DIR}/{''.join(ch if ch.isalnum() else '_' for ch in mid)}.csv"
    if os.path.exists(p):
        continue
    t0 = time.time()
    runs, arm = SPECS[s]
    m, _ = xai.load_model(runs, arm, int(mid.split('fold')[1]), SEEDS[0], device)
    df = xai.run_model(m, store, list(meta.rec), mask_of, device, batch=XAI_BS, ig_steps=FINE_STEPS, ig_lo=FINE_LO)
    df.assign(spec=s, model_id=mid).to_csv(p, index=False)
    print(f'{mid}: completeness median {df.ig_rel_err.median():.4f} | {time.time() - t0:.0f}s')
fine = pd.concat([pd.read_csv(p) for p in glob.glob(f'{FINE_DIR}/*.csv')])
fine['db'] = fine.rec.str.split('_').str[0]
old = ep[(ep.method == 'ig') & ep.model_id.isin(fine.model_id.unique())]
fi = fine[fine.method == 'ig']
j = old.merge(fi, on=['model_id', 'rec', 'epoch'], suffixes=('_64', '_256'))
cmp = j.groupby('model_id').agg(completeness_64=('ig_rel_err_64', 'median'), completeness_256=('ig_rel_err_256', 'median'),
                                auroc_64=('auroc_64', 'mean'), auroc_256=('auroc_256', 'mean'),
                                max_abs_auroc_diff=('auroc_64', lambda v: float(np.abs(v - j.loc[v.index, 'auroc_256']).max())))
display(cmp.round(4))
ep2 = pd.concat([ep[~((ep.method == 'ig') & ep.model_id.isin(fi.model_id.unique()))], fi])
res2 = xai.run_tests(plan, xai.excerpt_level(ep2, ['spec', 'method', 'db']), exc_b)
d = res.merge(res2, on=['family', 'db', 'spec', 'method', 'versus'], suffixes=('', '_fine'))
d['conclusion_changed'] = (d.q_bh < 0.05) != (d.q_bh_fine < 0.05)
cmp.to_csv(f'{RESULTS}/xai_ig_grid_sensitivity_models.csv')
d.to_csv(f'{RESULTS}/xai_ig_grid_sensitivity_tests.csv', index=False)
print('số kiểm định đổi kết luận (q < 0.05):', int(d.conclusion_changed.sum()), '/', len(d))
d[d.conclusion_changed | d.spec.isin({s for s, _ in flagged})][['family', 'spec', 'method', 'versus', 'median_auroc',
   'median_auroc_fine', 'q_bh', 'q_bh_fine', 'conclusion_changed']].round(4)
""")
nb.md("## 5. Hình 2 vẽ lại (B5) — ví dụ chọn theo quy tắc cố định, model fold 0, trục thời gian (s)")
nb.code(r"""
panels = []
for db in xai.DBS:
    rec, c = xai.figure_epoch(store, mask_of, db, meta)
    i = int(np.flatnonzero(store.recs[store.pos[rec]]['trim'] == c)[0])
    x = skd.SeqDS(store, [rec])[i][0][None].to(device)
    maps = {}
    for s in [k for k in SPECS if 'Sleep-EDF' in k]:
        m, _ = xai.load_model(*SPECS[s], 0, SEEDS[0], device)
        maps['IG: ' + s.split(' (')[0]] = xai.attribute(m, x, IG_STEPS, IG_GRID)['ig'][0]
    xc = store.recs[store.pos[rec]]['x'][c, 0].astype(np.float32)
    maps['|x| (baseline)'] = np.abs(xc)
    panels.append(dict(title=f"{xai.EVENT_NAME[db]}: {rec.split('_')[1]}, epoch {c}", x=xc, mask=mask_of[rec][c], maps=maps))
xai.plot_examples(panels, f'{RESULTS}/xai_fig2_examples.png');
""")
nb.save(os.path.join(OUT_DIR, 'R2_xai_dreams.ipynb'))

# =============================================================================== notebook 5: CKA v2
nb = NB()
nb.md(r"""
# CKA v2 — biểu diễn trên Sleep-EDF (R1: "CKA baselines"; plan C1, C3)

**Chạy SAU khi** notebook Sleep-EDF đã xong (không cần HMC). GPU.

- **Model** (Sleep-EDF, cùng fold): Teacher-CE, Assistant-KD (Teacher-CE), Edge-Staged (Teacher-CE), Edge-Direct (Teacher-CE), Edge-Scratch. **Mốc:** cùng kiến trúc, khởi tạo ngẫu nhiên (1 model / kiến trúc / fold).
- **Biểu diễn** tại epoch trung tâm: *spatial* = nhúng CNN (`cnn_feat`), *temporal* = biểu diễn ngữ cảnh đưa vào đầu phân loại (`ctx_feat`). Model ở eval mode.
- **Linear CKA** (Kornblith 2019) tính **theo từng subject** trên các epoch test của subject đó (hai model đều là model của fold giữ subject đó làm test). **Đơn vị thống kê = subject.**
- **Kiểm định chốt trước (cell 2):** `vs_random` — mỗi cặp đã train (X, Y) so với (X, Y khởi tạo ngẫu nhiên); `pathway` — các so sánh bài báo diễn giải (Assistant~Staged vs Assistant~Direct, Teacher~Direct vs Teacher~Staged, Assistant~Staged/Direct vs Assistant~Scratch). Wilcoxon cặp trên subject, median + CI bootstrap, BH theo family (loại kiểm định × mức biểu diễn).
- **Subject:** mọi cặp đều có model thuộc nhánh Teacher-CE → phân tích chính **bỏ 21 subject nhóm F1** (dùng để tune Teacher-CE; quy tắc lai, plan 0.3), còn 79; bản đủ 100 subject là phân tích độ nhạy.
- Bảng CKA theo fold (toàn bộ epoch test của fold, mean ± SD qua 5 fold) chỉ để mô tả, so với bảng cũ.
- **UMAP** (C3): fold 0, 5,000 epoch test chọn ngẫu nhiên (seed 0), chỉ để minh hoạ.
""")
nb.code("# ===================== CONFIG — chỉ sửa ở đây =====================" + COMMON_CONFIG + r"""
# CKA v2
CKA_MODELS = {'Teacher': 'T_ce', 'Assistant': 'A_kd_Tce', 'Edge-Staged': 'E_staged_Tce',
              'Edge-Direct': 'E_direct_Tce', 'Edge-Scratch': 'E_scratch'}
KIND_OF = {'Teacher': 'teacher', 'Assistant': 'assistant', 'Edge-Staged': 'edge', 'Edge-Direct': 'edge', 'Edge-Scratch': 'edge'}
CKA_DIR = f'{RESULTS}/cka_v2'
os.makedirs(CKA_DIR, exist_ok=True)
UMAP_N, UMAP_MODELS = 5000, ['Teacher', 'Assistant', 'Edge-Staged']
print('config OK')
""")
nb.code_files()
nb.code("!pip -q install umap-learn")
nb.md("## 1. Nạp Sleep-EDF vào RAM")
nb.code(r"""
import cka, xai, glob; importlib.reload(cka)
t0 = time.time()
shutil.copytree(DATA_DIR, LOCAL_DATA, dirs_exist_ok=True)
folds = {int(k): v for k, v in json.load(open(FOLDS_JSON)).items()}
store = skd.Store(LOCAL_DATA, sorted(r for v in folds.values() for r in v))
f1_subjects = sorted({r[:5] for r in folds[1]})
print(f'{len(store.recs)} recordings in RAM | {time.time() - t0:.0f}s | nhóm F1 (tune Teacher-CE): {len(f1_subjects)} subject')
""")
nb.md("## 2. Danh sách kiểm định chốt trước (in ra TRƯỚC khi tính CKA)")
nb.code(r"""
plan = cka.planned_tests(list(CKA_MODELS), KIND_OF)
plan.to_csv(f'{RESULTS}/cka_planned_tests.csv', index=False)
print(plan.groupby('family').size().to_string())
plan
""")
nb.md("## 3. Trích biểu diễn và tính CKA theo fold (resume theo fold)")
nb.code(r"""
for fold in range(K_FOLDS):
    ps, pf = f'{CKA_DIR}/per_subject_fold{fold}.csv', f'{CKA_DIR}/per_fold_fold{fold}.csv'
    if os.path.exists(ps) and os.path.exists(pf):
        continue
    t0 = time.time()
    ds = skd.SeqDS(store, folds[fold])
    subj = np.array([r[:5] for r in ds.rec_ids()])
    feats = {}
    for name, arm in CKA_MODELS.items():
        m, _ = cka.trained_model(RUNS_DIR, arm, fold, SEEDS[0], device)
        feats[name] = cka.features(m, ds, device)
    for kind in sorted(set(KIND_OF.values())):
        feats[f'random {kind}'] = cka.features(cka.random_model(kind, 20_000 + fold, device), ds, device)
    if fold == 0:   # UMAP subsample (descriptive)
        idx = np.sort(np.random.default_rng(0).choice(len(ds), min(UMAP_N, len(ds)), replace=False))
        np.savez(f'{CKA_DIR}/umap_fold0.npz', idx=idx, y=ds.labels()[idx],
                 **{n: feats[n]['temporal'][idx] for n in UMAP_MODELS})
    s, f = cka.fold_cka(feats, subj, list(CKA_MODELS), KIND_OF)
    s.assign(fold=fold).to_csv(ps, index=False); f.assign(fold=fold).to_csv(pf, index=False)
    print(f'fold {fold}: {len(ds)} epochs, {len(np.unique(subj))} subjects | {time.time() - t0:.0f}s')
""")
nb.md("## 4. Thống kê (chính: 79 subject; độ nhạy: 100 subject) và bảng mô tả")
nb.code(r"""
ps = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{CKA_DIR}/per_subject_fold*.csv'))])
pf = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{CKA_DIR}/per_fold_fold*.csv'))])
assert ps.subject.nunique() == len({r[:5] for v in folds.values() for r in v}) and pf.fold.nunique() == K_FOLDS
res = cka.run_tests(plan, ps, exclude_subjects=f1_subjects).assign(subjects='excl. tuning subjects (primary)')
res_all = cka.run_tests(plan, ps).assign(subjects='all 100 (sensitivity)')
pd.concat([res, res_all]).to_csv(f'{RESULTS}/cka_tests.csv', index=False)
desc = (ps[~ps.subject.isin(f1_subjects)].groupby(['level', 'pair']).cka
        .agg(median='median', q25=lambda v: v.quantile(.25), q75=lambda v: v.quantile(.75), n='size').round(3))
desc.to_csv(f'{RESULTS}/cka_per_subject_summary.csv')
fold_tab = pf.groupby(['level', 'pair']).cka.agg(['mean', 'std']).round(3)
fold_tab.to_csv(f'{RESULTS}/cka_fold_level_summary.csv')
display(desc); display(fold_tab)
res.round(4)
""")
nb.md("## 5. UMAP (C3) — fold 0, mô tả")
nb.code(r"""
import umap, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
d = np.load(f'{CKA_DIR}/umap_fold0.npz')
y = d['y']; colors = plt.cm.tab10(np.arange(5))
fig, axes = plt.subplots(1, len(UMAP_MODELS), figsize=(4.2 * len(UMAP_MODELS), 4))
for ax, n in zip(axes, UMAP_MODELS):
    e = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(d[n])
    for c in range(5):
        ax.scatter(*e[y == c].T, s=2, color=colors[c], label=skd.STAGES[c])
    ax.set_title(n); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(markerscale=5, fontsize=8)
fig.tight_layout(); fig.savefig(f'{RESULTS}/cka_umap_per_model.png', dpi=200); fig.savefig(f'{RESULTS}/cka_umap_per_model.pdf')
k = min(d[n].shape[1] for n in UMAP_MODELS)          # common dimension for the joint embedding
Z = [PCA(k, random_state=0).fit_transform(d[n]) for n in UMAP_MODELS]
Z = np.concatenate([(z - z.mean(0)) / (z.std(0) + 1e-8) for z in Z])
e = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z)
lab_m = np.repeat(np.arange(len(UMAP_MODELS)), len(y)); lab_y = np.tile(y, len(UMAP_MODELS))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for c in range(5):
    axes[0].scatter(*e[lab_y == c].T, s=2, color=colors[c], label=skd.STAGES[c])
for i, n in enumerate(UMAP_MODELS):
    axes[1].scatter(*e[lab_m == i].T, s=2, label=n)
for ax, t in zip(axes, ['by sleep stage', 'by model']):
    ax.set_title(f'Joint embedding ({t})'); ax.set_xticks([]); ax.set_yticks([]); ax.legend(markerscale=5, fontsize=8)
fig.tight_layout(); fig.savefig(f'{RESULTS}/cka_umap_joint.png', dpi=200); fig.savefig(f'{RESULTS}/cka_umap_joint.pdf')
""")
nb.save(os.path.join(OUT_DIR, 'R2_cka_v2.ipynb'))

# =============================================================================== notebook 6: hardware export (D4)
nb = NB()
nb.md(r"""
# Xuất dữ liệu cho triển khai phần cứng (D4) — chạy CPU được

**Chạy SAU notebook Sleep-EDF.** Tạo thư mục `hw_export/` trên Drive (rồi nén zip để tải về `revision/E_hardware/R2/inputs/`):
- `model_best.pt`, `model_result.json`: Edge Student chính (`E_staged_Tce`, fold 0).
- `calib_epochs.npy`: 1,000 epoch (200 / giai đoạn, seed 42) từ subject **train** của fold 0 — dùng để lượng tử hoá INT8 (không dùng subject test).
- `test_windows.npz`: 100 cửa sổ test **phân tầng** (20 / giai đoạn, seed 42) từ subject **test** fold 0; mỗi cửa sổ = 11 epoch nhân quả (10 epoch trước + epoch đích, trong cùng recording, biên lặp epoch đầu), đúng dữ liệu đã chuẩn hoá (fp16 như lúc train).
- `pc_reference.json`: logit FP32 của PyTorch cho từng cửa sổ, tính **hai cách** — (a) epoch trung tâm của cửa sổ 21 epoch như lúc đánh giá; (b) bước cuối của chuỗi 11 epoch như trên chip. Hai cách phải trùng nhau (Edge Student là nhân quả); notebook kiểm tra điều này.
""")
nb.code("# ===================== CONFIG =====================" + COMMON_CONFIG + r"""
HW_DIR, HW_ARM, HW_FOLD, N_CAL_PER_CLASS, N_TEST_PER_CLASS, HW_SEED = f'{DRIVE_ROOT}/hw_export', 'E_staged_Tce', 0, 200, 20, 42
os.makedirs(HW_DIR, exist_ok=True)
print('config OK')
""")
nb.code_files()
nb.md("## 1. Nạp dữ liệu fold 0 và model")
nb.code(r"""
import glob, hashlib, zipfile
shutil.copytree(DATA_DIR, LOCAL_DATA, dirs_exist_ok=True)
folds = {int(k): v for k, v in json.load(open(FOLDS_JSON)).items()}
train_recs, val_recs, test_recs = skd.split_recs(folds, HW_FOLD)
store = skd.Store(LOCAL_DATA, train_recs + test_recs)
C = dict(RUNS_DIR=RUNS_DIR)
res = skd.load_result(C, HW_ARM, HW_FOLD, SEEDS[0])
model = skd.build_model(res['kind'], res['target_dim'])
model.load_state_dict(torch.load(f"{skd.run_dir(C, HW_ARM, HW_FOLD, SEEDS[0])}/best.pt", map_location='cpu', weights_only=True))
model.eval()
shutil.copy(f"{skd.run_dir(C, HW_ARM, HW_FOLD, SEEDS[0])}/best.pt", f'{HW_DIR}/model_best.pt')
json.dump({k: res[k] for k in ('arm', 'fold', 'seed', 'kind', 'target_dim', 'params_inference', 'best_epoch')},
          open(f'{HW_DIR}/model_result.json', 'w'), indent=1)
print(HW_ARM, 'fold', HW_FOLD, '| train', len(train_recs), 'recs | test', len(test_recs), 'recs')
""")
nb.md("## 2. Tập calibration (train) và 100 cửa sổ test phân tầng (test)")
nb.code(r"""
def targets(recs):
    return [(store.pos[r], int(c), int(store.recs[store.pos[r]]['y'][c])) for r in recs for c in store.recs[store.pos[r]]['trim']]

def stratified(rows, n_per_class, seed):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(5):
        cand = [t for t in rows if t[2] == k]
        out += [cand[i] for i in sorted(rng.choice(len(cand), n_per_class, replace=False))]
    return out

cal = stratified(targets(train_recs), N_CAL_PER_CLASS, HW_SEED)
calib = np.stack([store.recs[ri]['x'][c].astype(np.float32) for ri, c, _ in cal])
np.save(f'{HW_DIR}/calib_epochs.npy', calib)
test = stratified(targets(test_recs), N_TEST_PER_CLASS, HW_SEED)
win11 = np.stack([store.recs[ri]['x'][np.clip(np.arange(c - 10, c + 1), 0, len(store.recs[ri]['y']) - 1)] for ri, c, _ in test])
win21 = np.stack([store.recs[ri]['x'][np.clip(np.arange(c - 10, c + 11), 0, len(store.recs[ri]['y']) - 1)] for ri, c, _ in test])
labels = np.array([y for _, _, y in test])
recs = np.array([store.recs[ri]['id'] for ri, _, _ in test]); centers = np.array([c for _, c, _ in test])
np.savez(f'{HW_DIR}/test_windows.npz', x=win11, y=labels, rec=recs, center=centers)
print('calibration', calib.shape, np.bincount([y for *_, y in cal]), '| test', win11.shape, np.bincount(labels),
      '| test subjects', len({r[:5] for r in recs}))
""")
nb.md("## 3. Logit tham chiếu FP32 trên PC (cửa sổ 21 epoch vs chuỗi nhân quả 11 epoch)")
nb.code(r"""
@torch.no_grad()
def causal11(m, x):                      # x: (B, 11, 3, 3000) -> logits after the last (11th) step
    B, L = x.shape[:2]
    f = m.cnn(x.reshape(B * L, 3, 3000)).view(B, L, -1)
    g, _ = m.gru(f)
    return m.classifier(m.gru_norm(m.gru_dropout(g[:, -1])))

with torch.no_grad():
    l21 = model(torch.from_numpy(win21.astype(np.float32)))['logits'].numpy()
    l11 = causal11(model, torch.from_numpy(win11.astype(np.float32))).numpy()
diff = float(np.abs(l21 - l11).max())
print(f'max |logit(21-epoch centre) - logit(11-step causal)| = {diff:.2e}')
assert diff < 1e-4, 'Edge output is not causal?'
ref = dict(arm=HW_ARM, fold=HW_FOLD, seed=HW_SEED, n=len(labels), labels=labels.tolist(), rec=recs.tolist(),
           center=centers.tolist(), logits_21=l21.tolist(), logits_11=l11.tolist(), pred=l11.argmax(1).tolist(),
           max_abs_diff_21_vs_11=diff, pc_correct=int((l11.argmax(1) == labels).sum()))
json.dump(ref, open(f'{HW_DIR}/pc_reference.json', 'w'))
print('PC FP32 correct on the 100 windows:', ref['pc_correct'], '/ 100')
""")
nb.md("## 4. Nén để tải về (đặt nội dung vào `revision/E_hardware/R2/inputs/`)")
nb.code(r"""
out = f'{DRIVE_ROOT}/hw_export.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(glob.glob(f'{HW_DIR}/*')):
        z.write(p, os.path.basename(p))
print(f'{out} | {os.path.getsize(out) / 1e6:.1f} MB | sha256 {hashlib.sha256(open(out, "rb").read()).hexdigest()}')
""")
nb.save(os.path.join(OUT_DIR, 'R2_hw_export.ipynb'))

# =============================================================================== notebook 7: seed robustness (R1 "multiple seeds")
nb = NB()
nb.md(r"""
# Độ ổn định theo seed của các nhánh Edge chính (R1: "multiple seeds")

**Chạy SAU notebook Sleep-EDF** (cần `runs/seed42/T_ce`, `runs/seed42/A_kd_Tce` trên Drive). GPU L4 là đủ.

- Chạy lại **3 nhánh Edge**: `E_scratch` (chỉ học từ nhãn), `E_direct_Tce` (distill trực tiếp từ Teacher-CE),
  `E_staged_Tce` (distill từ Assistant-KD (Teacher-CE), model Edge chính) với **seed huấn luyện 43 và 44** (chốt trước khi xem kết quả).
- Nguồn distill **cố định = model seed 42 của cùng fold** (`src_seed = 42`): chỉ khởi tạo và thứ tự batch của Edge thay đổi;
  chia fold không đổi (`SPLIT_SEED` = 42) → không rò rỉ dữ liệu. Kết quả là độ ổn định của Edge **khi Teacher/Assistant cố định**.
- Giao thức train giống hệt seed 42 (cùng `TRAIN_CFG`, T = 4, trọng số KD).
- Chạy song song 2 phiên Colab: phiên 1 đặt `RUN_SEEDS = [43]`, phiên 2 `RUN_SEEDS = [44]`; mỗi run resume được.

**Phân tích (chốt trước):** 3 so sánh Macro-F1 — Staged − Direct, Staged − Scratch, Direct − Scratch (Teacher-CE) — trên dữ liệu gộp
3 seed (42, 43, 44), bootstrap theo cụm subject, BH trong 3 so sánh; accuracy, κ, N1-F1 là family phụ. Vì cả 3 so sánh đều có nhánh
thuộc dòng Teacher-CE, dùng **79 subject** (bỏ nhóm F1, quy tắc lai); bản 100 subject là độ nhạy. Kèm kết quả từng seed riêng.
""")
nb.code("# ===================== CONFIG =====================" + COMMON_CONFIG + r"""
SEED_ARMS = [
    dict(name='E_scratch',    kind='edge', loss='ce', bs=64),
    dict(name='E_direct_Tce', kind='edge', loss='kd', src='T_ce',     T=4.0, bs=64, src_seed=42),
    dict(name='E_staged_Tce', kind='edge', loss='kd', src='A_kd_Tce', T=4.0, bs=64, src_seed=42),
]
NEW_SEEDS = [43, 44]
RUN_SEEDS = [43, 44]          # chạy song song: phiên 1 = [43], phiên 2 = [44]
SEED_CONTRASTS = [('E_staged_Tce', 'E_direct_Tce'), ('E_staged_Tce', 'E_scratch'), ('E_direct_Tce', 'E_scratch')]
C_NEW = dict(TRAIN_CFG, K_FOLDS=K_FOLDS, RUNS_DIR=RUNS_DIR, ARMS=SEED_ARMS, SEEDS=RUN_SEEDS)
print('config OK')
""")
nb.code_files()
nb.md("## 1. Nạp dữ liệu và kiểm tra nguồn distill seed 42")
nb.code(r"""
t0 = time.time()
shutil.copytree(DATA_DIR, LOCAL_DATA, dirs_exist_ok=True)
folds = {int(k): v for k, v in json.load(open(FOLDS_JSON)).items()}
store = skd.Store(LOCAL_DATA, sorted(r for v in folds.values() for r in v))
for src in ('T_ce', 'A_kd_Tce'):
    for f in range(K_FOLDS):
        d = skd.run_dir(dict(RUNS_DIR=RUNS_DIR), src, f, 42)
        assert os.path.exists(f'{d}/best.pt') and os.path.exists(f'{d}/result.json'), f'missing {src} fold{f} seed42'
print(f'{len(store.recs)} recordings in RAM | nguồn seed 42 đủ 5 fold | {time.time() - t0:.0f}s')
""")
nb.md("## 2. Train (resume được; mỗi phiên chạy các seed trong `RUN_SEEDS`)")
nb.code(r"""
skd.run_all(C_NEW, store, folds, device, run_folds=RUN_FOLDS)
""")
nb.md("## 3. Phân tích (chạy khi đủ cả seed 43 và 44)")
nb.code(r"""
f1_subjects = sorted({r[:5] for r in folds[1]})
arms = [a['name'] for a in SEED_ARMS]
missing = [(a, s, f) for a in arms for s in [42] + NEW_SEEDS for f in range(K_FOLDS)
           if skd.load_result(dict(RUNS_DIR=RUNS_DIR), a, f, s) is None]
assert not missing, f'chưa xong: {missing}'
for a in arms:                       # same protocol as seed 42; distillation source = seed-42 model of the same fold
    for f in range(K_FOLDS):
        r42 = skd.load_result(dict(RUNS_DIR=RUNS_DIR), a, f, 42)
        for s in NEW_SEEDS:
            r = skd.load_result(dict(RUNS_DIR=RUNS_DIR), a, f, s)
            assert all(r[k] == r42[k] for k in ('kind', 'loss', 'src', 'T', 'lr', 'ls', 'bs', 'n_train', 'n_test')), (a, s, f)
            assert r['src_seed'] == (42 if r['src'] else None), (a, s, f)
print('giao thức seed 43/44 == seed 42; nguồn distill = model seed 42 cùng fold')
C_ALL = dict(TRAIN_CFG, K_FOLDS=K_FOLDS, RUNS_DIR=RUNS_DIR, SEEDS=[42] + NEW_SEEDS)
per_seed = pd.concat([skd.pooled_table(dict(C_ALL, SEEDS=[s]), arms, exclude_subjects=f1_subjects).assign(seed=s)
                      for s in [42] + NEW_SEEDS])
per_seed.to_csv(f'{RESULTS}/seed_robustness_pooled_per_seed.csv')
display(per_seed[['seed', 'mf1', 'acc', 'kappa', 'f1_N1', 'n_subjects']].round(2))
ct79 = skd.cluster_contrasts(C_ALL, SEED_CONTRASTS, n_boot=10000, exclude_subjects=f1_subjects).assign(subjects=79, seeds='42+43+44')
ct100 = skd.cluster_contrasts(C_ALL, SEED_CONTRASTS, n_boot=10000).assign(subjects=100, seeds='42+43+44')
cts = [skd.cluster_contrasts(dict(C_ALL, SEEDS=[s]), SEED_CONTRASTS, n_boot=10000, exclude_subjects=f1_subjects)
       .assign(subjects=79, seeds=str(s)) for s in [42] + NEW_SEEDS]
allc = pd.concat([ct79, ct100] + cts)
allc.to_csv(f'{RESULTS}/seed_robustness_contrasts.csv', index=False)
allc[allc.metric.isin(['mf1', 'n1'])][['seeds', 'subjects', 'contrast', 'metric', 'diff_pp', 'ci95_low_pp', 'ci95_high_pp', 'q_bh']].round(4)
""")
nb.save(os.path.join(OUT_DIR, 'R2_seed_robustness.ipynb'), gpu='L4')

# =============================================================================== notebook 8: quantization evaluation (D3/D4)
nb = NB()
nb.md(r"""
# Lượng tử hoá: FP32 vs cấu hình mixed-precision đã triển khai — toàn bộ 5 fold, 100 subject

**Chạy SAU notebook Sleep-EDF.** GPU L4 là đủ (mô phỏng nhẹ; phần lượng tử hoá chạy CPU).

**Cấu hình mô phỏng = đúng firmware** (`revision/E_hardware/R2/firmware`): chỉ depthwise + pointwise conv của 4 khối DWSep
(EEG 0–1, EOG 0–1) chạy INT8 (ESP-PPQ, mô phỏng của đúng chip đích); GroupNorm + GELU, pooling, projection, GRU,
LayerNorm và classifier chạy FP32. Đầu vào mỗi khối được làm tròn như firmware (half away from zero, kẹp [−128, 127]).
Code lượng tử hoá/mô phỏng (`qsim.py`) dùng chung với `quantize_export.py` (file nạp lên chip).

**Calibration theo từng fold, chỉ subject train:** 1.000 epoch phân tầng (200 / giai đoạn, seed 42) từ subject train của fold
(cùng quy tắc với `R2_hw_export`), lấy 512 epoch (seed 42) như `quantize_export.py`. Test fold không dùng để chọn scale.
Mỗi fold lượng tử hoá model `E_staged_Tce` (seed 42) **của chính fold đó**.

**Tham chiếu FP32:** PyTorch FP32 thuần (tắt TF32, không autocast) — khác với đánh giá chính (bf16 autocast); độ lệch giữa
hai cách được báo cáo riêng (mô tả).

**Phân tích (chốt trước khi chạy):**
- Chính: ΔMacro-F1 = (cấu hình `esp32`) − FP32, dự đoán out-of-fold gộp, bootstrap cụm subject 10.000 lần (cặp), 95% CI.
  79 subject (quy tắc lai: `E_staged_Tce` thuộc dòng Teacher-CE); 100 subject = độ nhạy.
- Phụ: Δ accuracy, κ và F1 từng giai đoạn (W, N1, N2, N3, REM), BH trong 7 kiểm định này.
- Cấu hình `esp32s3` báo cáo cùng cách (nền tảng thứ hai).
- Mô tả: tỷ lệ epoch có nhãn trùng FP32, phân bố max |Δlogit|, bảng chuyển nhãn FP32 → INT8.
- Không có ngưỡng tương đương định trước: kết luận dựa trên khoảng tin cậy, không tuyên bố "không mất độ chính xác".
- Đối chiếu với chip: fold 0 xuất thêm logit mô phỏng cho 100 cửa sổ đã chốt trong `R2_hw_export` (cùng tập `host_run.py` gửi lên chip).
""")
nb.code("# ===================== CONFIG =====================" + COMMON_CONFIG + r"""
QE_ARM, QE_SEED, QE_TARGETS = 'E_staged_Tce', 42, ['esp32', 'esp32s3']
N_CAL_PER_CLASS, N_CAL, CAL_SEED = 200, 512, 42
QE_DIR = f'{RESULTS}/quant_eval'
QE_WORK = '/content/qe_work'
HW_DIR = f'{DRIVE_ROOT}/hw_export'          # 100 cửa sổ đã chốt + calibration fold 0 (để đối chiếu)
EXPECTED_SUBJECTS, EXPECTED_EPOCHS = 100, 237936   # AUDITED_NUMBERS
N_PER_STAGE_500 = 100                              # tập 500 cửa sổ đối chiếu chip (mục 2b)
PPQ_PY = '/content/ppq312/bin/python'              # Python 3.12 có esp-ppq (tạo ở cell dưới)
os.makedirs(QE_DIR, exist_ok=True)
print('config OK')
""")
nb.md("ESP-PPQ chỉ chạy trên Python ≤ 3.12 (Colab hiện là 3.13): cell dưới tạo một môi trường Python 3.12 riêng "
      "(`uv`, ~2–3 phút) **chỉ để lượng tử hoá** (cùng phiên bản với máy local: esp-ppq 1.3.9, onnx 1.17.0, torch 2.5.1 CPU). "
      "Mô phỏng toàn tập test chạy trong kernel chính bằng `qsim.TorchQuantEdge` (PyTorch thuần), đã kiểm chứng trùng từng bit "
      "với bộ thực thi ESP-PPQ; notebook kiểm tra lại điều này ở mỗi fold.")
nb.code("!pip -q install uv\n!uv python install 3.12 2>&1 | tail -n 1\n!uv venv -q --python 3.12 /content/ppq312\n"
        "!uv pip install -q --python /content/ppq312/bin/python --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cpu \"torch==2.5.1+cpu\" esp-ppq==1.3.9 onnx==1.17.0 2>&1 | tail -n 5\n"
        "!/content/ppq312/bin/python -c \"import sys, esp_ppq, onnx, torch; print('Python', sys.version.split()[0], "
        "'| esp-ppq OK | onnx', onnx.__version__, '| torch', torch.__version__)\"")
nb.code_files()
nb.md("## 1. Nạp dữ liệu")
nb.code(r"""
import glob
import qsim
importlib.reload(qsim)
qsim.strict_fp32()
t0 = time.time()
shutil.copytree(DATA_DIR, LOCAL_DATA, dirs_exist_ok=True)
folds = {int(k): v for k, v in json.load(open(FOLDS_JSON)).items()}
store = skd.Store(LOCAL_DATA, sorted(r for v in folds.values() for r in v))
f1_subjects = sorted({r[:5] for r in folds[1]})
C = dict(RUNS_DIR=RUNS_DIR)


def targets(recs):                       # identical to R2_hw_export
    return [(store.pos[r], int(c), int(store.recs[store.pos[r]]['y'][c])) for r in recs for c in store.recs[store.pos[r]]['trim']]


def stratified(rows, n_per_class, seed):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(5):
        cand = [t for t in rows if t[2] == k]
        out += [cand[i] for i in sorted(rng.choice(len(cand), n_per_class, replace=False))]
    return out


print(f'{len(store.recs)} recordings | {time.time() - t0:.0f}s')
""")
nb.md("## 2. Lượng tử hoá theo fold + mô phỏng toàn bộ tập test (resume được: mỗi fold một file)")
nb.code(r"""
import copy, subprocess
assert os.path.exists(PPQ_PY), f'{PPQ_PY} chưa có: chạy cell tạo môi trường Python 3.12 ở trên'
for f in range(K_FOLDS):
    out_npz = f'{QE_DIR}/fold{f}.npz'
    if os.path.exists(out_npz):
        print(f'[skip] fold {f}'); continue
    t0 = time.time()
    train_recs, val_recs, test_recs = skd.split_recs(folds, f, K_FOLDS)
    res = skd.load_result(C, QE_ARM, f, QE_SEED)
    assert res['fold'] == f and res['seed'] == QE_SEED
    model = skd.build_model(res['kind'], res['target_dim'])
    model.load_state_dict(torch.load(f'{skd.run_dir(C, QE_ARM, f, QE_SEED)}/best.pt', map_location='cpu', weights_only=True))
    model.eval()
    cal = stratified(targets(train_recs), N_CAL_PER_CLASS, CAL_SEED)
    assert not {store.recs[ri]['id'][:5] for ri, _, _ in cal} & {r[:5] for r in test_recs + val_recs}
    calib = np.stack([store.recs[ri]['x'][c].astype(np.float32) for ri, c, _ in cal])
    if f == 0 and os.path.exists(f'{HW_DIR}/calib_epochs.npy'):
        assert np.array_equal(calib, np.load(f'{HW_DIR}/calib_epochs.npy')), 'fold-0 calibration != R2_hw_export'
        print('fold 0 calibration == R2_hw_export (deployed model)')
    cal512 = qsim.calibration_subset(calib, N_CAL, CAL_SEED)
    model.to(device)

    # test epochs in the order of skd.SeqDS / preds_test.npz
    rec_ids, centers, ys = [], [], []
    for r in test_recs:
        R = store.recs[store.pos[r]]
        rec_ids += [r] * len(R['trim']); centers.append(R['trim']); ys.append(R['y'][R['trim']])
    rec_ids, centers, ys = np.array(rec_ids), np.concatenate(centers), np.concatenate(ys)

    def all_logits(feat_fn):
        return np.concatenate([qsim.recording_logits(feat_fn, model, store.recs[store.pos[r]]['x'], store.recs[store.pos[r]]['trim'], device)
                               for r in test_recs])

    out = dict(rec=rec_ids, center=centers, y=ys, logits_fp32=all_logits(lambda x: qsim.fp32_features(model, x)))
    # check: per-epoch-feature route == the model's own forward on 21-epoch windows (strict FP32)
    ds = skd.SeqDS(store, test_recs)
    pick = np.random.default_rng(f).choice(len(ds), min(256, len(ds)), replace=False)
    with torch.no_grad():
        direct = model(torch.stack([ds[i][0] for i in pick]).to(device))['logits'].cpu().numpy()
    d = float(np.abs(direct - out['logits_fp32'][pick]).max())
    assert d < 1e-4, d
    info = dict(fold=f, arm=QE_ARM, seed=QE_SEED, n_test=len(ys), n_calibration=len(cal512),
                calibration_subjects=sorted({store.recs[ri]['id'][:5] for ri, _, _ in cal}), max_diff_route_vs_forward=d)
    # comparison with the main evaluation (bf16 autocast), descriptive
    pt = np.load(f'{skd.run_dir(C, QE_ARM, f, QE_SEED)}/preds_test.npz', allow_pickle=True)
    assert np.array_equal(pt['rec'], rec_ids) and np.array_equal(pt['center'], centers) and np.array_equal(pt['y'], ys)
    out['logits_main_eval'] = pt['logits'].astype(np.float32)
    info['agreement_fp32_vs_main_eval'] = float((pt['logits'].astype(np.float32).argmax(1) == out['logits_fp32'].argmax(1)).mean())
    # quantization with ESP-PPQ in the Python 3.12 environment (CPU); ESP-PPQ-executor features of 256 test epochs for the check
    wd = f'{QE_WORK}/fold{f}'
    os.makedirs(wd, exist_ok=True)
    np.save(f'{wd}/cal512.npy', cal512)
    chk = np.stack([store.recs[ri]['x'][c].astype(np.float32) for ri, c in ds.index[pick]])
    np.save(f'{wd}/check.npy', chk)
    r = subprocess.run([PPQ_PY, f'{CODE_DIR}/qsim.py', 'quantize', '--code-dir', CODE_DIR,
                        '--model', f'{skd.run_dir(C, QE_ARM, f, QE_SEED)}/best.pt', '--target-dim', str(res['target_dim'] or 0),
                        '--calib', f'{wd}/cal512.npy', '--targets', *QE_TARGETS, '--work', f'{wd}/work',
                        '--export', f'{QE_DIR}/espdl/fold{f}', '--out', f'{wd}/params.json', '--check', f'{wd}/check.npy'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    params = json.load(open(f'{wd}/params.json'))
    info['quantizer'] = {k: params[k] for k in ('esp_ppq', 'onnx', 'torch', 'n_calibration')}
    for tgt in QE_TARGETS:
        binfo = params[tgt]
        # check: pure-PyTorch simulation vs ESP-PPQ executor on the same epochs (both CPU). Bit-identical with the same
        # torch version (verified locally); here the two environments use different torch versions, so float results
        # may differ in the last digits (~1e-6). A wrong exponent or rounding would give differences >= ~1e-2.
        ref = np.load(f'{wd}/params.json.check_{tgt}.npy')
        mine = qsim.TorchQuantEdge(copy.deepcopy(model).cpu(), binfo, torch.device('cpu')).features(torch.from_numpy(chk)).numpy()
        dmax = float(np.abs(mine - ref).max())
        assert dmax < 1e-3, f'TorchQuantEdge != ESP-PPQ executor (max |d| {dmax:.3e})'
        sim = qsim.TorchQuantEdge(model, binfo, device)
        out[f'logits_{tgt}'] = all_logits(sim.features)
        info[tgt] = dict(blocks=binfo, check_max_abs_diff_vs_ppq=dmax, check_identical=bool(dmax == 0), agreement_vs_fp32=float((out[f'logits_{tgt}'].argmax(1) == out['logits_fp32'].argmax(1)).mean()))
        if f == 0 and os.path.exists(f'{HW_DIR}/test_windows.npz'):     # the 100 windows sent to the chip
            X = torch.from_numpy(np.load(f'{HW_DIR}/test_windows.npz')['x'].astype(np.float32)).to(device)
            B, L = X.shape[:2]
            with torch.no_grad():
                lq = qsim.causal_head(model, sim.features(X.reshape(B * L, 3, 3000)).view(B, L, -1)).cpu().numpy()
            json.dump(dict(target=tgt, logits_sim=lq.tolist(), pred_sim=lq.argmax(1).tolist()),
                      open(f'{QE_DIR}/sim_reference_fold0_{tgt}.json', 'w'))
        print(f"fold {f} {tgt}: agreement with FP32 {100 * info[tgt]['agreement_vs_fp32']:.2f}% | exponents "
              f"{ {k: v['exponents'].get('input') for k, v in binfo.items()} }")
    np.savez_compressed(out_npz, **out)
    json.dump(info, open(f'{QE_DIR}/fold{f}_info.json', 'w'), indent=1)
    print(f'==> fold {f}: {len(ys):,} test epochs | FP32 vs main eval agreement {100 * info["agreement_fp32_vs_main_eval"]:.2f}% '
          f'| {time.time() - t0:.0f}s')
""")
nb.md("## 2b. Tập 500 cửa sổ cho đối chiếu chip – mô phỏng (chốt trước khi chạy trên chip)\n"
      "Cùng quy tắc với 100 cửa sổ của `R2_hw_export`: subject **test** của fold 0 (model đã nạp lên chip), phân tầng "
      "**100 / giai đoạn** (seed 42), mỗi cửa sổ 11 epoch nhân quả; logit FP32 PyTorch (CPU), kiểm tra cửa sổ 21 epoch = chuỗi "
      "nhân quả 11 epoch. Tải `hw_export_500.zip` và giải nén vào `revision/E_hardware/R2/inputs/`.")
nb.code(r"""
import zipfile, hashlib
tr0, va0, te0 = skd.split_recs(folds, 0, K_FOLDS)
res0 = skd.load_result(C, QE_ARM, 0, QE_SEED)
m0 = skd.build_model(res0['kind'], res0['target_dim'])
m0.load_state_dict(torch.load(f'{skd.run_dir(C, QE_ARM, 0, QE_SEED)}/best.pt', map_location='cpu', weights_only=True))
m0.eval()
sel = stratified(targets(te0), N_PER_STAGE_500, 42)
w11 = np.stack([store.recs[ri]['x'][np.clip(np.arange(c - 10, c + 1), 0, len(store.recs[ri]['y']) - 1)] for ri, c, _ in sel])
w21 = np.stack([store.recs[ri]['x'][np.clip(np.arange(c - 10, c + 11), 0, len(store.recs[ri]['y']) - 1)] for ri, c, _ in sel])
lab = np.array([y_ for _, _, y_ in sel])
rid = np.array([store.recs[ri]['id'] for ri, _, _ in sel]); cen = np.array([c for _, c, _ in sel])
with torch.no_grad():
    l21 = np.concatenate([m0(torch.from_numpy(w21[i:i + 50].astype(np.float32)))['logits'].numpy() for i in range(0, len(sel), 50)])
    l11 = np.concatenate([qsim.causal_head(m0, m0.cnn(torch.from_numpy(w11[i:i + 50].reshape(-1, 3, 3000).astype(np.float32)))
                                           .view(-1, 11, m0.embed_dim)).numpy() for i in range(0, len(sel), 50)])
d2111 = float(np.abs(l21 - l11).max())
assert d2111 < 1e-4, d2111
np.savez(f'{HW_DIR}/test_windows_500.npz', x=w11, y=lab, rec=rid, center=cen)
json.dump(dict(arm=QE_ARM, fold=0, seed=QE_SEED, n=len(lab), rule='test subjects of fold 0, stratified 100 per stage, seed 42',
               labels=lab.tolist(), rec=rid.tolist(), center=cen.tolist(), logits_21=l21.tolist(), logits_11=l11.tolist(),
               pred=l11.argmax(1).tolist(), max_abs_diff_21_vs_11=d2111, pc_correct=int((l11.argmax(1) == lab).sum())),
          open(f'{HW_DIR}/pc_reference_500.json', 'w'))
out = f'{DRIVE_ROOT}/hw_export_500.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in ('test_windows_500.npz', 'pc_reference_500.json'):
        z.write(f'{HW_DIR}/{n}', n)
print(f'500 windows: {len(lab)} | per stage {np.bincount(lab)} | {len(set(r[:5] for r in rid))} test subjects | '
      f'PC FP32 correct {int((l11.argmax(1) == lab).sum())} | max |21 vs 11| {d2111:.1e}')
print(f'{out} | {os.path.getsize(out) / 1e6:.1f} MB | sha256 {hashlib.sha256(open(out, "rb").read()).hexdigest()}')
""")
nb.md("## 3. Phân tích (chốt trước)")
nb.code(r"""
parts = [np.load(f'{QE_DIR}/fold{f}.npz', allow_pickle=True) for f in range(K_FOLDS)]
D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
subj = np.array([r[:5] for r in D['rec']])
y = D['y'].astype(np.int64)
pred = {k[7:]: D[k].astype(np.float32).argmax(1) for k in D if k.startswith('logits_')}
assert len(np.unique(subj)) == EXPECTED_SUBJECTS and len(y) == EXPECTED_EPOCHS, (len(np.unique(subj)), len(y))
SETS = [('79 subjects (primary)', f1_subjects), ('100 subjects (sensitivity)', [])]

rows, pooled = [], []
for sname, excl in SETS:
    keep = ~np.isin(subj, excl)
    cms = {k: qsim.subject_cms(y[keep], p[keep], subj[keep])[1] for k, p in pred.items()}
    for k, cm in cms.items():
        m = qsim.cm_metrics(cm.sum(0))
        pooled.append(dict(subjects=sname, model=k, n_subjects=len(cm), **{mk: 100 * float(v) for mk, v in m.items()}))
    for tgt in QE_TARGETS:
        for r in qsim.paired_bootstrap(cms[tgt], cms['fp32'], n_boot=10000, seed=0):
            rows.append(dict(subjects=sname, contrast=f'{tgt} - fp32', **r))
pooled, ct = pd.DataFrame(pooled), pd.DataFrame(rows)
pooled.to_csv(f'{RESULTS}/quant_eval_pooled.csv', index=False)
ct.to_csv(f'{RESULTS}/quant_eval_contrasts.csv', index=False)
display(pooled.round(3))
ct[['subjects', 'contrast', 'metric', 'family', 'diff_pp', 'ci95_low_pp', 'ci95_high_pp', 'p_boot', 'q_bh', 'n_subjects']].round(4)
""")
nb.code(r"""
# descriptive: label agreement with FP32, logit differences, transitions FP32 -> INT8
desc = []
for tgt in QE_TARGETS:
    dl = np.abs(D[f'logits_{tgt}'].astype(np.float32) - D['logits_fp32'].astype(np.float32)).max(1)
    desc.append(dict(target=tgt, epochs=len(y), agreement_pct=100 * float((pred[tgt] == pred['fp32']).mean()),
                     changed_epochs=int((pred[tgt] != pred['fp32']).sum()),
                     changed_to_correct=int(((pred[tgt] != pred['fp32']) & (pred[tgt] == y)).sum()),
                     changed_to_wrong=int(((pred[tgt] != pred['fp32']) & (pred['fp32'] == y)).sum()),
                     max_abs_dlogit_median=float(np.median(dl)), max_abs_dlogit_p95=float(np.percentile(dl, 95)),
                     max_abs_dlogit_max=float(dl.max()),
                     agreement_fp32_vs_main_eval_pct=100 * float((pred['fp32'] == pred['main_eval']).mean())))
desc = pd.DataFrame(desc); desc.to_csv(f'{RESULTS}/quant_eval_agreement.csv', index=False)
display(desc.round(4))
STG = ['W', 'N1', 'N2', 'N3', 'REM']
tr = pd.crosstab(pd.Categorical([STG[i] for i in pred['fp32']], STG), pd.Categorical([STG[i] for i in pred['esp32']], STG),
                 rownames=['FP32'], colnames=['esp32 INT8'], dropna=False)
tr.to_csv(f'{RESULTS}/quant_eval_transitions_esp32.csv')
tr
""")
nb.save(os.path.join(OUT_DIR, 'R2_quant_eval.ipynb'), gpu='L4')
