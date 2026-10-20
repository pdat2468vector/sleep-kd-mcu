"""D4 step 2 (local PC): quantize the 4 convolution blocks of the Edge Student to INT8 (.espdl, ESP-PPQ) and export the
FP32 parameters (GroupNorm, projection, GRU, classifier) for the firmware; simulate the exported mixed-precision model
on the 100 pre-specified test windows (sim_reference.json) so that host_run.py can compare the chip with the simulation
produced by this very quantization run.

usage: python quantize_export.py --target esp32|esp32s3 [--inputs inputs] [--n-cal 512]
  inputs/ = contents of hw_export.zip (notebook R2_hw_export.ipynb): model_best.pt, model_result.json, calib_epochs.npy,
            test_windows.npz, pc_reference.json
  writes firmware/model/ (the files embedded into the firmware), manifest.json and sim_reference.json

Partition (same as the round-1 deployment): each depthwise-separable block = depthwise conv + pointwise conv run in
INT8 by ESP-DL; GroupNorm + GELU, pooling, projection, GRU and classifier run in FP32 C code on the chip.
The quantization and simulation code is shared with the Colab notebook R2_quant_eval (tools/nbsrc/qsim.py)."""
import argparse, hashlib, json, os, shutil, sys
from importlib.metadata import version
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# works in the project tree (revision/E_hardware/R2) and in the code release (hardware/)
for root in (os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', '..')):
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, 'model_v11')):
        sys.path.insert(0, root)
        for nb in (os.path.join(root, 'tools', 'nbsrc'), os.path.join(root, 'revision', 'R2_subject_level', 'tools', 'nbsrc')):
            if os.path.isfile(os.path.join(nb, 'qsim.py')):
                sys.path.insert(0, nb)
        break
from model_v11.student_v2 import StudentSleepNetV2   # noqa: E402
import qsim   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--target', required=True, choices=['esp32', 'esp32s3'])
ap.add_argument('--inputs', default=os.path.join(HERE, 'inputs'))
ap.add_argument('--out', default=os.path.join(HERE, 'firmware', 'model'))
ap.add_argument('--n-cal', type=int, default=512)
ap.add_argument('--sim-only', action='store_true',
                help='do not touch the flashed files: re-quantize into work_onnx/, check that every exponent equals the '
                     'manifest of --out, and (re)write the simulation references for every window set in --inputs')
args = ap.parse_args()
qsim.strict_fp32()
WINDOW_SETS = [s for s in ('', '_500') if os.path.exists(os.path.join(args.inputs, f'test_windows{s}.npz'))]

res = json.load(open(os.path.join(args.inputs, 'model_result.json')))
model = StudentSleepNetV2(embed_dim=64, gru_hidden=32, gru_layers=1, v1_embed_dim=res['target_dim'] or 128,
                          num_classes=5, seq_len=21, dropout=0.1)
model.load_state_dict(torch.load(os.path.join(args.inputs, 'model_best.pt'), map_location='cpu', weights_only=True))
model.eval()
cnn = model.cnn

work = os.path.join(HERE, 'work_onnx')
export_dir = os.path.join(work, 'sim_only_export') if args.sim_only else args.out
shutil.rmtree(export_dir, ignore_errors=True)
os.makedirs(export_dir)

# ---------------------------------------------------------------- INT8 convolution blocks (calibration: train subjects only)
cal = qsim.calibration_subset(np.load(os.path.join(args.inputs, 'calib_epochs.npy')).astype(np.float32), args.n_cal, 42)
graphs, info = qsim.quantize_blocks(model, cal, args.target, work, export_dir)


# ---------------------------------------------------------------- simulation on the pre-specified test windows
def sim_reference(suffix):
    win = np.load(os.path.join(args.inputs, f'test_windows{suffix}.npz'))
    ref = json.load(open(os.path.join(args.inputs, f'pc_reference{suffix}.json')))
    X = torch.from_numpy(win['x'].astype(np.float32))                    # (n, 11, 3, 3000)
    sim = qsim.QuantEdge(model, graphs, torch.device('cpu'))
    l_fp, l_q = [], []
    with torch.no_grad():
        for i in range(0, len(X), 50):
            x = X[i:i + 50]
            B, L = x.shape[:2]
            l_fp.append(qsim.causal_head(model, qsim.fp32_features(model, x.reshape(B * L, 3, 3000)).view(B, L, -1)).numpy())
            l_q.append(qsim.causal_head(model, sim.features(x.reshape(B * L, 3, 3000)).view(B, L, -1)).numpy())
    l_fp, l_q = np.concatenate(l_fp), np.concatenate(l_q)
    L_ref = np.array(ref['logits_11'])
    assert np.abs(l_fp - L_ref).max() < 1e-4, 'FP32 path differs from the PyTorch reference of the export notebook'
    y = np.array(ref['labels'])
    return dict(target=args.target, window_set=suffix or '_100', n=len(X), logits_sim=l_q.tolist(),
                pred_sim=l_q.argmax(1).tolist(), sim_correct=int((l_q.argmax(1) == y).sum()),
                fp32_correct=int((L_ref.argmax(1) == y).sum()),
                agreement_sim_vs_fp32=int((l_q.argmax(1) == L_ref.argmax(1)).sum()),
                max_abs_logit_diff_sim_vs_fp32=dict(median=float(np.median(np.abs(l_q - L_ref).max(1))),
                                                    max=float(np.abs(l_q - L_ref).max())),
                note='ESP-PPQ simulation of the exported INT8 blocks (block inputs rounded half away from zero, as in '
                     'the firmware) + FP32 rest; causal 11-step windows as on the chip')


def report(r):
    print(f"simulation, {r['n']} windows: agreement with FP32 {r['agreement_sim_vs_fp32']}/{r['n']} | correct: sim "
          f"{r['sim_correct']}, FP32 {r['fp32_correct']} | max |dlogit| median "
          f"{r['max_abs_logit_diff_sim_vs_fp32']['median']:.3f}, max {r['max_abs_logit_diff_sim_vs_fp32']['max']:.3f}")


if args.sim_only:
    flashed = json.load(open(os.path.join(args.out, 'manifest.json')))
    assert flashed['target'] == args.target, 'manifest of --out is for another target'
    for b, v in info.items():
        assert v['exponents'] == flashed['blocks'][b]['exponents'], f'{b}: exponents differ from the flashed model'
    print('re-quantization reproduces every exponent of the flashed model (INT8 weights follow from the same FP32 '
          'weights and exponents)')
    for s in WINDOW_SETS:
        r = sim_reference(s)
        p = os.path.join(args.out, f'sim_reference{s}.json')
        if os.path.exists(p):
            old = np.array(json.load(open(p))['logits_sim'])
            print(f'  {p}: existing reference reproduced, max |d| = {np.abs(old - np.array(r["logits_sim"])).max():.2e}')
        json.dump(r, open(p, 'w'), indent=1)
        report(r)
    sys.exit(0)


# ---------------------------------------------------------------- FP32 parameters
def save(arr, fname):
    arr = arr.detach().cpu().numpy().astype('<f4')
    arr.tofile(os.path.join(args.out, fname))
    return list(arr.shape)


shapes = {}
for prefix, branch in (('eeg', cnn.eeg_branch), ('eog', cnn.eog_branch)):
    for i in (0, 1):
        shapes[f'{prefix}_{i}_gn_weight.bin'] = save(branch[i].norm.weight, f'{prefix}_{i}_gn_weight.bin')
        shapes[f'{prefix}_{i}_gn_bias.bin'] = save(branch[i].norm.bias, f'{prefix}_{i}_gn_bias.bin')
lin, ln = cnn.proj[0], cnn.proj[1]
for n, t in [('proj_linear_weight', lin.weight), ('proj_linear_bias', lin.bias), ('proj_ln_weight', ln.weight),
             ('proj_ln_bias', ln.bias)]:
    shapes[n + '.bin'] = save(t, n + '.bin')
for pname, p in model.gru.named_parameters():
    shapes[f'gru_{pname}.bin'] = save(p, f'gru_{pname}.bin')
for n, t in [('gru_norm_weight', model.gru_norm.weight), ('gru_norm_bias', model.gru_norm.bias),
             ('cls_ln_weight', model.classifier[0].weight), ('cls_ln_bias', model.classifier[0].bias),
             ('cls_linear_weight', model.classifier[1].weight), ('cls_linear_bias', model.classifier[1].bias)]:
    shapes[n + '.bin'] = save(t, n + '.bin')
eps = dict(groupnorm=[cnn.eeg_branch[0].norm.eps], proj_ln=ln.eps, gru_norm=model.gru_norm.eps,
           cls_ln=model.classifier[0].eps)
assert all(abs(v - 1e-5) < 1e-12 for v in [eps['groupnorm'][0], eps['proj_ln'], eps['gru_norm'], eps['cls_ln']]), eps

simrefs = {s: sim_reference(s) for s in WINDOW_SETS}
files = sorted(os.listdir(args.out))
manifest = dict(target=args.target, model=res, n_calibration_epochs=len(cal), calibration='train subjects of fold 0',
                esp_ppq=version('esp-ppq'), torch=torch.__version__, blocks=info, fp32_shapes=shapes, layer_norm_eps=eps,
                files={f: dict(bytes=os.path.getsize(os.path.join(args.out, f)),
                               sha256=hashlib.sha256(open(os.path.join(args.out, f), 'rb').read()).hexdigest())
                       for f in files},
                espdl_bytes=sum(os.path.getsize(os.path.join(args.out, f)) for f in files if f.endswith('.espdl')),
                fp32_bytes=sum(os.path.getsize(os.path.join(args.out, f)) for f in files if f.endswith('.bin')))
manifest['model_bytes_total'] = manifest['espdl_bytes'] + manifest['fp32_bytes']
json.dump(manifest, open(os.path.join(args.out, 'manifest.json'), 'w'), indent=1)
for s, r in simrefs.items():
    json.dump(r, open(os.path.join(args.out, f'sim_reference{s}.json'), 'w'), indent=1)
print(f"{args.target}: {len(files)} files | INT8 {manifest['espdl_bytes']:,} B + FP32 {manifest['fp32_bytes']:,} B "
      f"= {manifest['model_bytes_total']:,} B")
for r in simrefs.values():
    report(r)
