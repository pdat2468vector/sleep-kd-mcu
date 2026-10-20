"""Test host_run.py without hardware: a fake serial device implements the firmware protocol (H / W / S) with the
PyTorch FP32 Edge Student, so the expected result is 100/100 agreement and bit-identical repeats.
usage: python test_host_fake_device.py <workdir>   (PYTHONPATH must contain the code dir written by build_notebook.py)"""
import json, os, shutil, struct, sys, types
import numpy as np, torch
import skd

W = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
shutil.rmtree(W, ignore_errors=True)
os.makedirs(os.path.join(W, 'inputs')); os.makedirs(os.path.join(W, 'firmware', 'model'))
shutil.copy(os.path.join(HERE, 'host_run.py'), W)

torch.manual_seed(0)
model = skd.build_model('edge', 128).eval()
rng = np.random.default_rng(0)
X = rng.standard_normal((6, 11, 3, 3000)).astype(np.float16)
y = rng.integers(0, 5, 6)


@torch.no_grad()
def step(x, h):
    f = model.cnn(torch.from_numpy(x[None].astype(np.float32)))
    out, h = model.gru(f[:, None], h)
    return model.classifier(model.gru_norm(out[:, -1]))[0].numpy(), h


ref = []
for w in range(len(X)):
    h = None
    for k in range(11):
        lg, h = step(X[w, k], h)
    ref.append(lg)
np.savez(os.path.join(W, 'inputs', 'test_windows.npz'), x=X, y=y)
json.dump(dict(labels=y.tolist(), logits_11=np.array(ref).tolist(), pred=np.array(ref).argmax(1).tolist()),
          open(os.path.join(W, 'inputs', 'pc_reference.json'), 'w'))
json.dump(dict(target='esp32', model_bytes_total=68260), open(os.path.join(W, 'firmware', 'model', 'manifest.json'), 'w'))
json.dump(dict(target='esp32', logits_sim=np.array(ref).tolist(), pred_sim=np.array(ref).argmax(1).tolist()),   # fake device = FP32
          open(os.path.join(W, 'firmware', 'model', 'sim_reference.json'), 'w'))


class FakeSerial:
    def __init__(self, *a, **k):
        self.out, self.inbuf, self.mode, self.k, self.h, self.ts = [], b'', None, 0, None, []
        self.dtr = False
        self._rts = False

    @property
    def rts(self):
        return self._rts

    @rts.setter
    def rts(self, v):
        if self._rts and not v:
            self.out = [b'garbage boot text\n', b'READY\n']
            self.mode, self.inbuf = None, b''
        self._rts = v

    def reset_input_buffer(self):
        self.out = []

    def readline(self):
        return self.out.pop(0) if self.out else b''

    def write(self, b):
        self.inbuf += b
        while self.inbuf:
            if self.mode is None:
                c, self.inbuf = self.inbuf[:1], self.inbuf[1:]
                if c == b'H':
                    self.out.append(b'INFO ' + json.dumps(dict(target='esp32', psram_total=0, espdl_bytes=11280,
                                                               fp32_bytes=56980, boot_free=300000, alloc_free=150000,
                                                               static_buffer_bytes=208800)).encode() + b'\n')
                elif c == b'W':
                    self.mode, self.k, self.h, self.ts = 'W', 0, None, []
                    self.out.append(b'RDY 0\n')
                elif c == b'S':
                    self.mode = 'S'
            elif self.mode == 'W':
                if len(self.inbuf) < 36000:
                    return
                x = np.frombuffer(self.inbuf[:36000], '<f4').reshape(3, 3000).astype(np.float16)
                self.inbuf = self.inbuf[36000:]
                self.lg, self.h = step(x, self.h)
                self.k += 1
                if self.k < 11:
                    self.out.append(f'RDY {self.k}\n'.encode())
                else:
                    self.mode = None
                    r = dict(ok=1, logits=[float(f'{v:.9g}') for v in self.lg], step_us=[300000] * 11,
                             create_us=[20000] * 11, conv_us=[200000] * 11, fp32_gn_us=[60000] * 11,
                             free=140000, min_free=120000, largest=60000)
                    line = b'RES ' + json.dumps(r).encode() + b'\n'
                    FakeSerial.n_res = getattr(FakeSerial, 'n_res', 0) + 1
                    if FakeSerial.n_res == 3:                   # one corrupted line, as seen once on the real link
                        line = line[:60] + line[40:]
                    self.out.append(line)
            elif self.mode == 'S':
                if len(self.inbuf) < 4:
                    return
                n = struct.unpack('<I', self.inbuf[:4])[0]
                self.inbuf, self.mode = self.inbuf[4:], None
                self.out += [f'STR {e} 140000 120000 60000 300000\n'.encode() for e in range(100, n + 1, 100)]
                self.out.append(b'STRESS ' + json.dumps(dict(n=n, fails=0, mean_us=300000.0)).encode() + b'\n')


sys.modules['serial'] = types.SimpleNamespace(Serial=FakeSerial)
sys.argv = ['host_run.py', '--port', 'FAKE', '--target', 'esp32', '--stress', '300']
os.chdir(W)
exec(compile(open(os.path.join(W, 'host_run.py'), encoding='utf-8').read(), 'host_run.py', 'exec'),
     {'__name__': '__main__', '__file__': os.path.join(W, 'host_run.py')})
s = json.load(open(next(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(W, 'logs')) for f in fs
                        if f == 'results.json')))['summary']
assert s['agreement'] == len(X) and s['agreement_vs_sim'] == len(X) and s['determinism_bit_identical'] and s['stress']['n'] == 300, s
assert [e['window'] for e in s['transport_errors']] == [2], s['transport_errors']      # corrupted line -> window repeated
print('HOST FAKE-DEVICE TEST OK')
