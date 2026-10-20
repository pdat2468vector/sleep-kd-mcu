"""D4 step 4 (local PC, board connected): run all on-chip measurements and save the raw logs.

usage: python host_run.py --port COM5 --target esp32 [--baud 460800] [--stress 1000] [--map firmware/build/*.map]
       [--set _500]   (the 500-window chip-vs-simulation set; use --stress 0 to skip the stress test)

Steps: (1) INFO (chip, build, heap, model bytes); (2) the 100 causal 11-epoch test windows from inputs/test_windows.npz,
compared with the PyTorch FP32 reference inputs/pc_reference.json and with the ESP-PPQ simulation of the flashed
model files (firmware/model/sim_reference.json, written by quantize_export.py); (3) determinism: the first 3 windows are run again
and must give bit-identical logits; (4) continuous streaming test of N epochs.
Everything the board prints is written to logs/<target>_<timestamp>/serial_log.txt; parsed results to results.json."""
import argparse, glob, json, os, re, struct, sys, time
import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--port', required=True)
ap.add_argument('--target', required=True, choices=['esp32', 'esp32s3'])
ap.add_argument('--baud', type=int, default=460800)
ap.add_argument('--stress', type=int, default=1000)
ap.add_argument('--inputs', default=os.path.join(HERE, 'inputs'))
ap.add_argument('--map', default=None, help='linker map of the flashed build (for the Flash audit)')
ap.add_argument('--model-dir', default=os.path.join(HERE, 'firmware', 'model'),
                help='folder with manifest.json / sim_reference*.json of the FLASHED model (one per target when two '
                     'boards run at the same time)')
ap.add_argument('--set', default='', choices=['', '_500'],
                help="window set: '' = the 100 windows of R2_hw_export, '_500' = the 500 windows of R2_quant_eval 2b")
args = ap.parse_args()

OUT = os.path.join(HERE, 'logs', f"{args.target}{args.set}_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(OUT)
LOG = open(os.path.join(OUT, 'serial_log.txt'), 'w', encoding='utf-8')
win = np.load(os.path.join(args.inputs, f'test_windows{args.set}.npz'))
X = win['x'].astype('<f4')                                     # (n, 11, 3, 3000): exactly the model input values
ref = json.load(open(os.path.join(args.inputs, f'pc_reference{args.set}.json')))
manifest = json.load(open(os.path.join(args.model_dir, 'manifest.json')))
simref = json.load(open(os.path.join(args.model_dir, f'sim_reference{args.set}.json')))
assert simref['target'] == args.target and len(simref['logits_sim']) == len(X) == len(ref['labels'])
assert manifest['target'] == args.target, 'firmware/model was exported for another target'
ser = serial.Serial(args.port, args.baud, timeout=1)


def readline(expect=None, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode('utf-8', 'replace').strip()
        LOG.write(f'{time.time():.3f} {line}\n')
        if expect is None or line.startswith(expect):
            return line
    raise TimeoutError(f'no line starting with {expect!r} within {timeout}s')


def send_epoch(arr):
    b = arr.tobytes()
    for i in range(0, len(b), 2048):
        ser.write(b[i:i + 2048])
        time.sleep(0.001)


def reset_board():
    ser.dtr = False; ser.rts = True; time.sleep(0.2); ser.rts = False
    readline('READY', timeout=30)
    ser.reset_input_buffer()


TRANSPORT_ERRORS = []      # windows repeated because a line arrived corrupted over UART (raw text stays in serial_log)


def run_window_once(w):
    ser.write(b'W')
    for k in range(11):
        assert readline('RDY').split()[1] == str(k)
        send_epoch(X[w, k])
    r = json.loads(readline('RES ', timeout=300)[4:])
    assert len(r['logits']) == 5 and all(len(r[k]) == 11 for k in ('step_us', 'create_us', 'conv_us', 'fp32_gn_us'))
    return r


def run_window(w, attempts=3):
    for a in range(attempts):
        try:
            return run_window_once(w)
        except (AssertionError, ValueError, IndexError, KeyError, TimeoutError) as e:
            TRANSPORT_ERRORS.append(dict(window=int(w), attempt=a, error=f'{type(e).__name__}: {e}'[:200]))
            LOG.write(f'{time.time():.3f} HOST transport error on window {w} (attempt {a}): {e!r}\n'); LOG.flush()
            print(f'window {w}: transport error ({type(e).__name__}), board reset and window repeated')
            reset_board()
    raise RuntimeError(f'window {w}: {attempts} transport errors in a row')


reset_board()
for a in range(3):                          # the INFO line was once not received after a reset: ask again
    ser.write(b'H')
    try:
        info = json.loads(readline('INFO ', timeout=20)[5:])
        break
    except (TimeoutError, ValueError) as e:
        TRANSPORT_ERRORS.append(dict(window=-1, attempt=a, error=f'INFO: {type(e).__name__}'))
        reset_board()
else:
    raise RuntimeError('no INFO from the board')
print('INFO', info)
assert info['target'] == args.target and info['psram_total'] == 0, 'wrong target or PSRAM in use'
assert info['espdl_bytes'] + info['fp32_bytes'] == manifest['model_bytes_total'], 'embedded model != manifest'
for k, b in (('eeg1_input_exp', 'eeg_block_1'), ('eog1_input_exp', 'eog_block_1')):    # firmware with INT8 intermediates
    if k in info:
        assert [info[k]] == manifest['blocks'][b]['exponents']['input'], f'{k} on the chip != manifest'

res = []
t_all = time.time()
for w in range(len(X)):
    r = run_window(w)
    r['window'] = w
    res.append(r)
    print(f"window {w:3d}: esp {int(np.argmax(r['logits']))} sim {simref['pred_sim'][w]} pc {ref['pred'][w]} label {ref['labels'][w]} "
          f"| step {np.mean(r['step_us']) / 1000:.1f} ms | min_free {r['min_free']}")
repeat = [run_window(w) for w in range(3)]

stress_lines, stress = [], None
if args.stress > 0:
    ser.write(b'S' + struct.pack('<I', args.stress))
    while True:
        line = readline(timeout=3600)
        if line.startswith('STR '):
            stress_lines.append([int(v) for v in line.split()[1:]])
        elif line.startswith('STRESS '):
            stress = json.loads(line[7:])
            break

# ---------------------------------------------------------------- summary
L_esp = np.array([r['logits'] for r in res]); L_pc = np.array(ref['logits_11']); L_sim = np.array(simref['logits_sim'])
p_esp, p_pc, y = L_esp.argmax(1), L_pc.argmax(1), np.array(ref['labels'])
step = np.array([r['step_us'] for r in res]) / 1000.0
summary = dict(
    target=args.target, info=info, model_manifest=manifest, n_windows=len(res), all_ok=all(r['ok'] for r in res),
    agreement=int((p_esp == p_pc).sum()), esp_correct=int((p_esp == y).sum()), pc_correct=int((p_pc == y).sum()),
    disagreements=[dict(window=int(w), label=int(y[w]), pc=int(p_pc[w]), esp=int(p_esp[w])) for w in np.flatnonzero(p_esp != p_pc)],
    max_abs_logit_diff=dict(median=float(np.median(np.abs(L_esp - L_pc).max(1))), max=float(np.abs(L_esp - L_pc).max())),
    agreement_vs_sim=int((p_esp == L_sim.argmax(1)).sum()), sim_correct=int((L_sim.argmax(1) == y).sum()),
    max_abs_logit_diff_vs_sim=dict(median=float(np.median(np.abs(L_esp - L_sim).max(1))), max=float(np.abs(L_esp - L_sim).max())),
    windows_logits_equal_sim_fp32=int((L_esp.astype(np.float32) == L_sim.astype(np.float32)).all(1).sum()),
    disagreements_vs_sim=[dict(window=int(w), label=int(y[w]), sim=int(L_sim[w].argmax()), esp=int(p_esp[w]))
                          for w in np.flatnonzero(p_esp != L_sim.argmax(1))],
    step_ms=dict(mean=float(step.mean()), sd=float(step.std(ddof=1)), min=float(step.min()), max=float(step.max())),
    window_ms=dict(mean=float(step.sum(1).mean()), sd=float(step.sum(1).std(ddof=1))),
    create_ms_mean=float(np.mean([r['create_us'] for r in res]) / 1000),
    conv_ms_mean=float(np.mean([r['conv_us'] for r in res]) / 1000),
    fp32_gn_ms_mean=float(np.mean([r['fp32_gn_us'] for r in res]) / 1000),
    heap=dict(boot_free=info['boot_free'], after_static_alloc_free=info['alloc_free'],
              static_buffer_bytes=info['static_buffer_bytes'], min_free_after_windows=min(r['min_free'] for r in res),
              largest_free_after_windows=min(r['largest'] for r in res)),
    determinism_bit_identical=all(np.array_equal(np.array(a['logits'], np.float32), np.array(b['logits'], np.float32))
                                  for a, b in zip(res[:3], repeat)),
    stress=stress, stress_progress=stress_lines, host_seconds=time.time() - t_all,
    transport_errors=TRANSPORT_ERRORS)
summary['heap']['peak_dynamic_bytes'] = info['alloc_free'] - summary['heap']['min_free_after_windows']
if args.map:
    mp = open(glob.glob(args.map)[0], errors='ignore').read()
    sym = {m.group(2) + '_' + m.group(3): int(m.group(1), 16)
           for m in re.finditer(r'^\s+0x([0-9a-f]+)\s+_binary_(\w+?)_(start|end)\s*$', mp, re.M)}
    names = sorted({k.rsplit('_', 1)[0] for k in sym})
    summary['flash_linked_model_bytes'] = sum(sym[n + '_end'] - sym[n + '_start'] for n in names)
json.dump(dict(summary=summary, windows=res, repeat=repeat), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
print(json.dumps({k: summary[k] for k in ('agreement', 'esp_correct', 'pc_correct', 'max_abs_logit_diff', 'agreement_vs_sim',
                                          'sim_correct', 'max_abs_logit_diff_vs_sim', 'windows_logits_equal_sim_fp32', 'step_ms',
                                          'window_ms', 'heap', 'determinism_bit_identical')}, indent=1))
print('stress:', stress)
print('saved ->', OUT)
