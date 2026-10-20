"""Generate the hardware table and number macros of the manuscript from the on-chip measurement logs (no hand-copied
numbers). usage: python make_hw_tables.py <revision.tex>

Which runs are official is listed in logs/OFFICIAL_RUNS.json ({target: {"100": <log dir> | null, "500": ...}}).
Missing runs (e.g. the ESP32 board not yet measured) appear as \\hwTBD{...} in the table and in the macros.
Writes the blocks '% BEGIN/END GENERATED hw_profile' (table body) and 'hw_macros' (\\newcommand definitions), and
asserts the qualitative statements made in the text."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = sys.argv[1]
OFFICIAL = json.load(open(os.path.join(HERE, 'logs', 'OFFICIAL_RUNS.json')))
TARGETS = [('esp32', 'Esp', 'ESP32'), ('esp32s3', 'SThree', 'ESP32-S3')]
CHIP = {1: ('ESP32', 'Xtensa LX6'), 9: ('ESP32-S3', 'Xtensa LX7')}


def load(target, wset):
    d = OFFICIAL.get(target, {}).get(wset)
    if not d:
        return None
    r = json.load(open(os.path.join(HERE, 'logs', d, 'results.json')))
    s = r['summary']
    assert s['target'] == target and s['n_windows'] == int(wset) and s['all_ok'], (target, wset)
    return r


def esp_dl_version(target):
    p = os.path.join(HERE, 'build_artifacts', target, 'dependencies.lock')
    m = re.search(r'espressif/esp-dl:\s*\n(?:\s+.*\n)*?\s+version:\s*([\d.]+)', open(p).read())
    return m.group(1)


def tbd(what):
    return rf'\hwTBD{{{what}}}'


def sci(v):
    m, e = f'{v:.1e}'.split('e')
    return rf'${m} \times 10^{{{int(e)}}}$'


R = {t: {w: load(t, w) for w in ('100', '500')} for t, _, _ in TARGETS}
macros, cols = [], {}
for t, tag, name in TARGETS:
    r100, r500 = R[t]['100'], R[t]['500']
    c = {}
    if r100 is None:
        for k in ('chip', 'core', 'rev', 'clock', 'idf', 'espdl', 'ppq', 'psram', 'flash', 'flash_espdl', 'flash_fp32',
                  'static', 'peakdyn', 'minfree', 'largest', 'lat', 'lat_range', 'create', 'conv', 'gn', 'win', 'rt',
                  'agree_sim', 'dlogit_sim', 'agree_fp', 'correct', 'determ', 'stress', 'drift', 'heap_stress'):
            c[k] = tbd(name)
    else:
        s = r100['summary']
        info, st = s['info'], s['stress']
        assert info['psram_total'] == 0 and s['flash_linked_model_bytes'] == s['model_manifest']['model_bytes_total']
        assert s['determinism_bit_identical'] and st['fails'] == 0 and st['free_start'] == st['free_end']
        assert s['agreement_vs_sim'] == s['n_windows']                       # text: chip == simulation on every window
        assert s['fp32_gn_ms_mean'] > s['conv_ms_mean']                       # text: FP32 GroupNorm/GELU > INT8 conv
        assert s['create_ms_mean'] + s['fp32_gn_ms_mean'] > s['conv_ms_mean']  # Discussion: instantiation + FP32 > INT8 conv
        assert st['free_start'] == st['free_end'] and st['fails'] == 0        # Discussion: no failure, no memory growth
        chip, core = CHIP[info['chip_model']]
        lat = s['step_ms']
        c.update(chip=chip, core=f"{info['cores']} $\\times$ {core}",
                 rev=f"v{info['revision'] // 100}.{info['revision'] % 100}",          # esp_chip_info: major * 100 + minor
                 clock=f"{info['cpu_mhz']}\\,MHz", idf=info['idf'].lstrip('v'), espdl=esp_dl_version(t),
                 ppq=s['model_manifest']['esp_ppq'], psram='not used (0 B)',
                 flash=f"{s['flash_linked_model_bytes']:,}", flash_espdl=f"{info['espdl_bytes']:,}", flash_fp32=f"{info['fp32_bytes']:,}",
                 static=f"{s['heap']['static_buffer_bytes']:,}", peakdyn=f"{s['heap']['peak_dynamic_bytes']:,}",
                 minfree=f"{s['heap']['min_free_after_windows']:,}", largest=f"{s['heap']['largest_free_after_windows']:,}",
                 lat=f"{lat['mean']:.1f} $\\pm$ {lat['sd']:.1f}", lat_range=f"{lat['min']:.1f}--{lat['max']:.1f}",
                 create=f"{s['create_ms_mean']:.1f}", conv=f"{s['conv_ms_mean']:.1f}", gn=f"{s['fp32_gn_ms_mean']:.1f}",
                 win=f"{s['window_ms']['mean'] / 1000:.2f}", rt=f"{30000 / lat['mean']:.0f}",
                 agree_sim=f"{s['agreement_vs_sim']} / {s['n_windows']}",
                 dlogit_sim=f"{sci(s['max_abs_logit_diff_vs_sim']['median'])} / {sci(s['max_abs_logit_diff_vs_sim']['max'])}",
                 agree_fp=f"{s['agreement']} / {s['n_windows']}",
                 correct=f"{s['esp_correct']} / {s['sim_correct']} / {s['pc_correct']}",
                 determ='identical logits (3 / 3)',
                 stress=f"{st['n']:,} steps, {st['fails']} failures",
                 drift=f"{st['first100_mean_us'] / 1000:.2f} / {st['last100_mean_us'] / 1000:.2f}",
                 heap_stress=f"{st['free_start']:,} / {st['free_end']:,}")
        macros += [rf'\newcommand{{\hw{tag}Lat}}{{{lat["mean"]:.1f}}}', rf'\newcommand{{\hw{tag}LatSD}}{{{lat["sd"]:.1f}}}',
                   rf'\newcommand{{\hw{tag}RT}}{{{30000 / lat["mean"]:.0f}}}',
                   rf'\newcommand{{\hw{tag}Win}}{{{s["window_ms"]["mean"] / 1000:.2f}}}',
                   rf'\newcommand{{\hw{tag}WinRT}}{{{30000 / s["window_ms"]["mean"]:.0f}}}',   # epoch / 11-step decision
                   # hardware figure caption: remaining FP32 time of a step, free heap after start-up, allocator overhead
                   rf'\newcommand{{\hw{tag}Other}}{{{lat["mean"] - s["create_ms_mean"] - s["conv_ms_mean"] - s["fp32_gn_ms_mean"]:.1f}}}',
                   rf'\newcommand{{\hw{tag}BootFree}}{{{s["heap"]["boot_free"]:,}}}',
                   rf'\newcommand{{\hw{tag}Overhead}}{{{s["heap"]["boot_free"] - s["heap"]["after_static_alloc_free"] - s["heap"]["static_buffer_bytes"]:,}}}',
                   rf'\newcommand{{\hw{tag}Conv}}{{{s["conv_ms_mean"]:.1f}}}', rf'\newcommand{{\hw{tag}GN}}{{{s["fp32_gn_ms_mean"]:.1f}}}',
                   rf'\newcommand{{\hw{tag}Create}}{{{s["create_ms_mean"]:.1f}}}',
                   rf'\newcommand{{\hw{tag}Flash}}{{{s["flash_linked_model_bytes"]:,}}}',
                   rf'\newcommand{{\hw{tag}Static}}{{{s["heap"]["static_buffer_bytes"]:,}}}',
                   rf'\newcommand{{\hw{tag}PeakDyn}}{{{s["heap"]["peak_dynamic_bytes"]:,}}}',
                   rf'\newcommand{{\hw{tag}N}}{{{s["n_windows"]}}}', rf'\newcommand{{\hw{tag}AgreeSim}}{{{s["agreement_vs_sim"]}}}',
                   rf'\newcommand{{\hw{tag}AgreeFP}}{{{s["agreement"]}}}',
                   rf'\newcommand{{\hw{tag}DlogitSim}}{{{sci(s["max_abs_logit_diff_vs_sim"]["max"])}}}',
                   rf'\newcommand{{\hw{tag}StressN}}{{{st["n"]:,}}}',
                   rf'\newcommand{{\hw{tag}DriftFirst}}{{{st["first100_mean_us"] / 1000:.2f}}}',
                   rf'\newcommand{{\hw{tag}DriftLast}}{{{st["last100_mean_us"] / 1000:.2f}}}']
    if r500 is None:
        c['agree_sim500'] = tbd(f'{name}, 500 windows')
        macros.append(rf'\newcommand{{\hw{tag}AgreeSimFive}}{{\hwTBD{{{name}, 500 windows}}}}')
    else:
        s5 = r500['summary']
        c['agree_sim500'] = (f"{s5['agreement_vs_sim']} / {s5['n_windows']} "
                             f"(max $|\\Delta|$ {sci(s5['max_abs_logit_diff_vs_sim']['max'])})")
        macros.append(rf'\newcommand{{\hw{tag}AgreeSimFive}}{{{s5["agreement_vs_sim"]} of {s5["n_windows"]}}}')
    if r100 is None:
        macros += [rf'\newcommand{{\hw{tag}{k}}}{{\hwTBD{{{name}}}}}' for k in
                   ('Lat', 'LatSD', 'RT', 'Win', 'Conv', 'GN', 'Create', 'Flash', 'Static', 'PeakDyn', 'N', 'AgreeSim',
                    'AgreeFP', 'DlogitSim', 'StressN', 'DriftFirst', 'DriftLast')]
    cols[t] = c

# text claim (Methods, Temporal Block): on the 100 and 500 test windows, the 11-step causal formulation reproduces the
# center-epoch logits of the full 21-epoch formulation exactly (PyTorch FP32, deployed fold-0 model)
for ref in ('pc_reference.json', 'pc_reference_500.json'):
    pr = json.load(open(os.path.join(HERE, 'inputs', ref)))
    assert pr['max_abs_diff_21_vs_11'] == 0.0 and pr['n'] in (100, 500), ref

# text claim: on every measured chip, chip-FP32 label differences are exactly the simulation-FP32 differences
for t, _, _ in TARGETS:
    for w in ('100', '500'):
        r = R[t][w]
        if r is None:
            continue
        s = r['summary']
        assert s['agreement_vs_sim'] == s['n_windows'], (t, w, s['disagreements_vs_sim'])

ROWS = [('A', 'Platform and build'),
        ('chip', 'Microcontroller (revision)'), ('core', 'CPU cores'), ('clock', 'Clock'), ('psram', 'External PSRAM'),
        ('idf', 'ESP-IDF / ESP-DL / ESP-PPQ'),
        ('B', 'Memory'),
        ('flash', 'Model data in Flash (B)'), ('flash_split', '\\quad INT8 blocks / FP32 parameters (B)'),
        # static buffers, peak dynamic heap, minimum free heap and the latency breakdown are shown in the hardware
        # figure (figures/src/make_hw_figure.py, same logs), so they are not repeated in the table
        ('largest', 'Largest free block after inference (B)'),
        ('C', 'Latency (esp\\_timer, one step = one 30-s epoch)'),
        ('lat', 'Step latency, mean $\\pm$ SD (ms)'), ('lat_range', '\\quad range (ms)'),
        ('win', '11-step window (s)'), ('rt', 'Epoch duration / step latency'),
        ('D', 'Agreement'),
        ('agree_sim', 'Labels equal to PC simulation, 100 windows'), ('dlogit_sim', '\\quad max $|\\Delta\\mathrm{logit}|$ vs.\\ simulation, median / max'),
        ('agree_sim500', 'Labels equal to PC simulation, 500 windows'),
        ('agree_fp', 'Labels equal to PyTorch FP32, 100 windows'), ('correct', 'Correct vs.\\ expert, chip / simulation / FP32'),
        ('determ', 'Repeated windows'),
        ('E', 'Continuous operation (GRU state not reset)'),
        ('stress', 'Consecutive steps'), ('drift', '\\quad mean latency, first / last 100 steps (ms)'),
        ('heap_stress', '\\quad free heap, start / end (B)')]
body = [r'\begin{tabular}{lll}', r'\toprule', r'\textbf{Quantity} & \textbf{ESP32} & \textbf{ESP32-S3} \\', r'\midrule']
for key, label in ROWS:
    if len(key) == 1:
        body.append(rf'\multicolumn{{3}}{{l}}{{\textit{{{label}}}}} \\')
        continue
    cells = []
    for t, _, name in TARGETS:
        c = cols[t]
        if key == 'chip':
            v = c['chip'] if c['chip'].startswith('\\hwTBD') else f"{c['chip']} ({c['rev']})"
        elif key == 'idf':
            v = c['idf'] if c['idf'].startswith('\\hwTBD') else f"{c['idf']} / {c['espdl']} / {c['ppq']}"
        elif key == 'flash_split':
            v = c['flash_espdl'] if c['flash_espdl'].startswith('\\hwTBD') else f"{c['flash_espdl']} / {c['flash_fp32']}"
        elif key == 'parts':
            v = c['create'] if c['create'].startswith('\\hwTBD') else f"{c['create']} / {c['conv']} / {c['gn']}"
        elif key == 'rt':
            v = c['rt'] if c['rt'].startswith('\\hwTBD') else f"{c['rt']}$\\times$"
        elif key == 'energy':
            v = tbd(f'{name}, PPK2')
        else:
            v = c[key]
        cells.append(v)
    body.append(f'\\quad {label} & ' + ' & '.join(cells) + r' \\')
body += [r'\bottomrule', r'\end{tabular}']
blocks = {'hw_profile': '\n'.join(body), 'hw_macros': '\n'.join(macros)}

if not TEX.endswith('.tex'):                     # standalone (code release): one file per block in a folder
    os.makedirs(TEX, exist_ok=True)
    for name, b in blocks.items():
        open(os.path.join(TEX, f'{name}.tex'), 'w', encoding='utf-8').write(b + '\n')
    print('wrote blocks to', TEX, ':', list(blocks))
    sys.exit(0)
tex = open(TEX, encoding='utf-8').read()
for name, b in blocks.items():
    pat = re.compile(rf'(% BEGIN GENERATED {name}\n).*?(% END GENERATED {name})', re.S)
    assert len(pat.findall(tex)) == 1, f'markers for {name} missing or duplicated'
    tex = pat.sub(lambda m: m.group(1) + b + '\n' + m.group(2), tex)
open(TEX, 'w', encoding='utf-8').write(tex)
print('updated blocks:', list(blocks), '| official runs:', {t: {w: bool(R[t][w]) for w in R[t]} for t in R})
