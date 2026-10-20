# qsim.py — PC simulation of the deployed mixed-precision Edge Student (D4 / quantization evaluation).
# Deployed configuration (revision/E_hardware/R2/firmware): the depthwise + pointwise convolutions of the four DWSep
# blocks (EEG 0-1, EOG 0-1) run in INT8 (ESP-DL, quantized with ESP-PPQ); GroupNorm + GELU, pooling, projection, GRU,
# LayerNorms and classifier run in FP32. Nothing else is quantized here.
import copy, inspect, os, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BLOCKS = [('eeg_block_0', 'eeg_branch', 0), ('eog_block_0', 'eog_branch', 0),
          ('eeg_block_1', 'eeg_branch', 1), ('eog_block_1', 'eog_branch', 1)]
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def strict_fp32():
    """No TF32 / non-deterministic kernels: the reference must be genuine FP32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConvOnly(nn.Module):        # depthwise + pointwise only (GroupNorm and GELU run in FP32 on the chip)
    def __init__(self, blk):
        super().__init__()
        self.dw, self.pw = blk.dw, blk.pw

    def forward(self, x):
        return self.pw(self.dw(x))


def calibration_subset(cal, n_cal=512, seed=42):
    """Same subsample as revision/E_hardware/R2/quantize_export.py (512 of the stratified training epochs)."""
    rng = np.random.default_rng(seed)
    return cal[np.sort(rng.choice(len(cal), min(n_cal, len(cal)), replace=False))]


@torch.no_grad()
def calibration_inputs(model, cal):
    """Input of each block on the calibration epochs; block 1 sees the FP32 output of the full block 0
    (incl. GroupNorm + GELU), as in the deployed export."""
    cnn = copy.deepcopy(model.cnn).cpu().eval()
    eeg, eog = cal[:, :2], cal[:, 2:3]
    return {'eeg_block_0': eeg, 'eog_block_0': eog,
            'eeg_block_1': cnn.eeg_branch[0](torch.from_numpy(eeg)).numpy(),
            'eog_block_1': cnn.eog_branch[0](torch.from_numpy(eog)).numpy()}


def graph_exponents(graph):
    """log2 of every power-of-two scale in a quantized block (input, weights, intermediate, output)."""
    out = {}
    for op in graph.operations.values():
        for cfg, var in op.config_with_variable:
            if cfg.scale is not None:
                e = np.log2(cfg.scale.detach().cpu().numpy().astype(np.float64).ravel())
                assert np.allclose(e, np.round(e)), f'{var.name}: scale is not a power of two'
                out[var.name] = [int(v) for v in np.round(e)]
    return out


def quantize_blocks(model, cal, target, work_dir, export_dir):
    """Quantize the four convolution blocks with ESP-PPQ (INT8, target chip), export the .espdl files to export_dir.
    Returns {block: quantized graph} and per-block info (shapes, exponents, rounding)."""
    from esp_ppq.api import espdl_quantize_onnx
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    ins = calibration_inputs(model, cal)
    graphs, info = {}, {}
    for name, branch, i in BLOCKS:
        blk = getattr(model.cnn, branch)[i]
        data = np.ascontiguousarray(ins[name], dtype=np.float32)
        shape = [1] + list(data.shape[1:])
        onnx_path = os.path.join(work_dir, f'{name}.onnx')
        kw = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(ConvOnly(copy.deepcopy(blk)).cpu().eval(), torch.zeros(shape), onnx_path, input_names=['input'],
                          output_names=['output'], opset_version=13, **kw)
        loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=1, shuffle=False)
        g = espdl_quantize_onnx(onnx_import_file=onnx_path, espdl_export_file=os.path.join(export_dir, f'{name}.espdl'),
                                calib_dataloader=loader, calib_steps=len(data), input_shape=shape, target=target,
                                num_of_bits=8, export_test_values=False, device='cpu', error_report=False)
        for side in ('.info', '.json'):          # ESP-PPQ side files: keep them out of the export folder
            p = os.path.join(export_dir, name + side)
            if os.path.exists(p):
                shutil.move(p, os.path.join(work_dir, f'{name}_{target}{side}'))   # may cross file systems (Drive)
        rounding = sorted({str(cfg.rounding) for op in g.operations.values() for cfg, _ in op.config_with_variable})
        graphs[name] = g
        info[name] = dict(input_shape=shape, groupnorm_groups=blk.norm.num_groups, exponents=graph_exponents(g),
                          rounding=rounding)
    return graphs, info


class QuantEdge:
    """Edge Student with the four convolution blocks executed by the ESP-PPQ simulator of the target chip.
    The block input is quantized like the firmware (round half away from zero, clip to [-128, 127]) before the
    simulator, so the simulator's own input quantization is the identity."""

    def __init__(self, model, graphs, device):
        from esp_ppq.executor import TorchExecutor
        self.m, self.device = model.to(device).eval(), device
        self.ex = {k: TorchExecutor(graph=g, device=str(device)) for k, g in graphs.items()}
        self.in_scale = {}
        for k, g in graphs.items():
            cfg = [c for op in g.operations.values() for c, v in op.config_with_variable if v.name == 'input']
            self.in_scale[k] = float(cfg[0].scale.ravel()[0])

    def _block(self, name, branch, i, x):
        s = self.in_scale[name]
        v = x / s
        q = torch.clamp(torch.sign(v) * torch.floor(torch.abs(v) + 0.5), -128, 127) * s
        y = self.ex[name].forward(inputs=q)[0]
        blk = getattr(self.m.cnn, branch)[i]
        return blk.act(blk.norm(y))

    @torch.no_grad()
    def features(self, x):                 # x: (N, 3, 3000) float32 on device -> (N, 64)
        cnn = self.m.cnn
        eeg = self._block('eeg_block_1', 'eeg_branch', 1, self._block('eeg_block_0', 'eeg_branch', 0, x[:, :2]))
        eog = self._block('eog_block_1', 'eog_branch', 1, self._block('eog_block_0', 'eog_branch', 0, x[:, 2:3]))
        assert eeg.shape[-1] == eog.shape[-1]
        return cnn.proj(cnn.pool(torch.cat([eeg, eog], 1)).squeeze(-1))


def _q_half_up(x, e):                        # ESP-PPQ ROUND_HALF_UP, symmetric INT8, power-of-two scale 2**e
    s = 2.0 ** e
    return torch.clamp(torch.floor(x / s + 0.5), -128, 127) * s


def _q_half_away(x, e):                      # firmware input rounding (roundf: half away from zero)
    s = 2.0 ** e
    v = x / s
    return torch.clamp(torch.sign(v) * torch.floor(torch.abs(v) + 0.5), -128, 127) * s


def check_params(params):
    """params: {block: {'exponents': {var: [e]}, 'rounding': [...]}} as returned by quantize_blocks (info)."""
    for b, v in params.items():
        assert v['rounding'] == ['RoundingPolicy.ROUND_HALF_UP'], (b, v['rounding'])
        ex = v['exponents']
        assert set(ex) == {'input', 'dw.weight', '/dw/Conv_output_0', 'pw.weight', 'output'}, (b, sorted(ex))
        assert all(len(e) == 1 for e in ex.values()), f'{b}: per-channel scales are not supported here'


class TorchQuantEdge:
    """Pure-PyTorch re-implementation of the ESP-PPQ simulation used by QuantEdge (no esp_ppq needed): per block,
    input -> INT8 (firmware rounding), depthwise conv with INT8 weights -> INT8 (half up), pointwise conv with INT8
    weights -> INT8 output (half up); then FP32 GroupNorm + GELU. Verified against QuantEdge (ESP-PPQ executor)."""

    def __init__(self, model, params, device):
        check_params(params)
        self.m, self.device = model.to(device).eval(), device
        self.e = {b: {k: int(v[0]) for k, v in p['exponents'].items()} for b, p in params.items()}
        self.w = {}
        for name, branch, i in BLOCKS:
            blk, e = getattr(self.m.cnn, branch)[i], self.e[name]
            self.w[name] = (_q_half_up(blk.dw.weight.detach(), e['dw.weight']), _q_half_up(blk.pw.weight.detach(), e['pw.weight']))

    def _block(self, name, branch, i, x):
        blk, e = getattr(self.m.cnn, branch)[i], self.e[name]
        wd, wp = self.w[name]
        y = torch.nn.functional.conv1d(_q_half_away(x, e['input']), wd, None, blk.dw.stride, blk.dw.padding,
                                       blk.dw.dilation, blk.dw.groups)
        y = _q_half_up(y, e['/dw/Conv_output_0'])
        y = _q_half_up(torch.nn.functional.conv1d(y, wp, None, blk.pw.stride, blk.pw.padding, blk.pw.dilation,
                                                  blk.pw.groups), e['output'])
        return blk.act(blk.norm(y))

    @torch.no_grad()
    def features(self, x):
        cnn = self.m.cnn
        eeg = self._block('eeg_block_1', 'eeg_branch', 1, self._block('eeg_block_0', 'eeg_branch', 0, x[:, :2]))
        eog = self._block('eog_block_1', 'eog_branch', 1, self._block('eog_block_0', 'eog_branch', 0, x[:, 2:3]))
        assert eeg.shape[-1] == eog.shape[-1]
        return cnn.proj(cnn.pool(torch.cat([eeg, eog], 1)).squeeze(-1))


@torch.no_grad()
def fp32_features(model, x):
    return model.cnn(x)


@torch.no_grad()
def head(model, seq):                       # seq: (B, L, 64) -> logits of the centre epoch (GRU is causal)
    g, _ = model.gru(seq)
    return model.classifier(model.gru_norm(g[:, seq.shape[1] // 2]))


@torch.no_grad()
def causal_head(model, seq):                # seq: (B, 11, 64) -> logits after the last step (as on the chip)
    g, _ = model.gru(seq)
    return model.classifier(model.gru_norm(g[:, -1]))


@torch.no_grad()
def recording_logits(feat_fn, model, x, centers, device, half=10, chunk=1024):
    """Logits for the target epochs `centers` of one recording, from per-epoch features computed once.
    Windows are clamped to the recording (edge-replicate padding), exactly as skd.SeqDS."""
    n = len(x)
    lo, hi = max(0, int(centers.min()) - half), min(n - 1, int(centers.max()) + half)
    feats = torch.empty(hi - lo + 1, model.cnn.proj[0].out_features, device=device)
    for s in range(lo, hi + 1, chunk):
        e = min(hi + 1, s + chunk)
        feats[s - lo:e - lo] = feat_fn(torch.from_numpy(x[s:e].astype(np.float32)).to(device))
    idx = np.clip(centers[:, None] + np.arange(-half, half + 1)[None], 0, n - 1) - lo
    out = []
    for s in range(0, len(idx), chunk):
        out.append(head(model, feats[torch.from_numpy(idx[s:s + chunk]).to(device)]).float().cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- statistics (paired subject-cluster bootstrap)
def cm_metrics(cm):
    """cm: (..., 5, 5) -> dict of arrays: mf1, acc, kappa, f1_<stage>."""
    cm = cm.astype(np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rows, cols = cm.sum(-1), cm.sum(-2)
    n = cm.sum((-1, -2))
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(rows + cols > 0, 2 * tp / np.where(rows + cols > 0, rows + cols, 1), 0.0)
        acc = tp.sum(-1) / n
        pe = (rows * cols).sum(-1) / n ** 2
        kappa = (acc - pe) / (1 - pe)
    out = dict(mf1=f1.mean(-1), acc=acc, kappa=kappa)
    out.update({f'f1_{s}': f1[..., k] for k, s in enumerate(STAGES)})
    return out


def subject_cms(y, pred, subj):
    subs = np.unique(subj)
    cms = np.zeros((len(subs), 5, 5), np.int64)
    for k, s in enumerate(subs):
        m = subj == s
        np.add.at(cms[k], (y[m], pred[m]), 1)
    return subs, cms


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def paired_bootstrap(cm_a, cm_b, n_boot=10000, seed=0):
    """A - B for every metric; subjects resampled with replacement (same replicate subjects for A and B).
    Returns rows with the point difference (pp), 95% percentile CI and two-sided bootstrap p (+1 correction)."""
    rng = np.random.default_rng(seed)
    S = len(cm_a)
    W = rng.multinomial(S, np.full(S, 1 / S), size=n_boot)
    ma, mb = cm_metrics(cm_a.sum(0)), cm_metrics(cm_b.sum(0))
    ba = cm_metrics(np.einsum('bs,sij->bij', W, cm_a))
    bb = cm_metrics(np.einsum('bs,sij->bij', W, cm_b))
    rows = []
    for met in ma:
        d = ba[met] - bb[met]
        p = min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (n_boot + 1))
        rows.append(dict(metric=met, family='primary' if met == 'mf1' else 'secondary', value_a=100 * float(ma[met]),
                         value_b=100 * float(mb[met]), diff_pp=100 * float(ma[met] - mb[met]),
                         ci95_low_pp=100 * float(np.percentile(d, 2.5)), ci95_high_pp=100 * float(np.percentile(d, 97.5)),
                         p_boot=p, n_subjects=S, n_epochs=int(cm_a.sum())))
    sec = [r for r in rows if r['family'] == 'secondary']
    for r, q in zip(sec, bh([r['p_boot'] for r in sec])):
        r['q_bh'] = float(q)
    rows[0]['q_bh'] = rows[0]['p_boot']           # single primary test
    return rows


# ---------------------------------------------------------------- CLI (run with a Python that has esp-ppq, e.g. 3.12)
def _cli():
    """python qsim.py quantize --code-dir D --model best.pt --target-dim N --calib cal.npy --targets esp32 esp32s3
                              --work W --export E --out params.json [--check check.npy]
    Quantizes the four blocks for each target (ESP-PPQ), writes {target: {block: info}} to --out and, with --check,
    the ESP-PPQ-executor features of those epochs (CPU) to <out>.check_<target>.npy for comparison with TorchQuantEdge."""
    import argparse, json, sys
    from importlib.metadata import version
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['quantize'])
    ap.add_argument('--code-dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--target-dim', type=int, default=0)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--targets', nargs='+', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--export', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    from model_v11.student_v2 import StudentSleepNetV2
    strict_fp32()
    model = StudentSleepNetV2(embed_dim=64, gru_hidden=32, gru_layers=1, v1_embed_dim=a.target_dim or 128,
                              num_classes=5, seq_len=21, dropout=0.1)
    model.load_state_dict(torch.load(a.model, map_location='cpu', weights_only=True))
    model.eval()
    cal = np.load(a.calib).astype(np.float32)
    out = dict(esp_ppq=version('esp-ppq'), onnx=version('onnx'), torch=torch.__version__, n_calibration=len(cal))
    for t in a.targets:
        graphs, info = quantize_blocks(model, cal, t, os.path.join(a.work, t), os.path.join(a.export, t))
        check_params(info)
        out[t] = info
        if a.check:
            x = torch.from_numpy(np.load(a.check).astype(np.float32))
            np.save(f'{a.out}.check_{t}.npy', QuantEdge(model, graphs, torch.device('cpu')).features(x).numpy())
    json.dump(out, open(a.out, 'w'), indent=1)
    print('quantized:', a.targets, '| esp-ppq', out['esp_ppq'], '| onnx', out['onnx'], '| torch', out['torch'])


if __name__ == '__main__':
    _cli()
