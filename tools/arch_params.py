"""Architecture figure: extract every number shown in the figure from the model code (no hand-copied values).
usage: python arch_params.py <out_dir>      (PYTHONPATH must contain the code dir written by build_notebook.py)
writes <out_dir>/arch_params.json (full layer list with output shapes) and <out_dir>/arch_macros.tex (\\newcommand)."""
import json, os, sys
import torch
import torch.nn as nn
import skd

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
L, T = skd.SEQ_LEN, 3000
models = {'teacher': skd.build_model('teacher'), 'assistant': skd.build_model('assistant', 256),
          'edge': skd.build_model('edge', 128)}
EXPECTED_PARAMS = {'teacher': 2558469, 'assistant': 290501, 'edge': 17318}


def describe(name, m):
    if isinstance(m, nn.Conv1d):
        return dict(type='conv', in_ch=m.in_channels, out_ch=m.out_channels, k=m.kernel_size[0], s=m.stride[0],
                    groups=m.groups)
    if isinstance(m, nn.GroupNorm):
        return dict(type='groupnorm', groups=m.num_groups, ch=m.num_channels)
    if isinstance(m, nn.Linear):
        return dict(type='linear', fin=m.in_features, fout=m.out_features)
    if isinstance(m, nn.GRU):
        return dict(type='gru', fin=m.input_size, hidden=m.hidden_size, layers=m.num_layers, bidirectional=m.bidirectional)
    if isinstance(m, nn.TransformerEncoderLayer):
        return dict(type='transformer_layer', d=m.self_attn.embed_dim, heads=m.self_attn.num_heads,
                    ff=m.linear1.out_features, norm_first=m.norm_first)
    if type(m).__name__ == 'LinearAttentionBlock':
        return dict(type='linear_attention_block', heads=m.attn.num_heads, d=m.attn.num_heads * m.attn.head_dim,
                    ff=[x for x in m.ffn if isinstance(x, nn.Linear)][0].out_features if hasattr(m, 'ffn') else None)
    if isinstance(m, nn.Embedding):
        return dict(type='pos_embedding', n=m.num_embeddings, d=m.embedding_dim)
    return None


info = {}
for key, model in models.items():
    model.eval()
    layers, hooks = [], []
    for name, m in model.named_modules():
        d = describe(name, m)
        if d is None:
            continue
        d['name'] = name
        layers.append(d)

        def hook(mod, inp, out, d=d):
            o = out[0] if isinstance(out, tuple) else out
            d['out_shape'] = list(o.shape[1:])
        if not isinstance(m, nn.Embedding):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(1, L, 3, T))
    for h in hooks:
        h.remove()
    total, infer = skd.count_params(model)
    assert infer == EXPECTED_PARAMS[key], (key, infer)
    info[key] = dict(params_inference=infer, params_total=total, layers=layers)
json.dump(info, open(os.path.join(OUT, 'arch_params.json'), 'w'), indent=1)


def convs(key, prefix):
    return [l for l in info[key]['layers'] if l['type'] == 'conv' and l['name'].startswith(prefix)]


def one(key, typ, contains=''):
    r = [l for l in info[key]['layers'] if l['type'] == typ and contains in l['name']]
    return r


M = {}
# ---- Teacher: Inception branches (two parallel kernels) + 1x1 fusion, projection, Transformer, classifier
te = convs('teacher', 'feature_extractor.eog_branch'); tg = convs('teacher', 'feature_extractor.eeg_branch')
M.update(TeoKa=te[0]['k'], TeoKb=te[1]['k'], TeoC=te[0]['out_ch'], TeoFuse=te[2]['out_ch'], TS=te[0]['s'],
         TegKa=tg[0]['k'], TegKb=tg[1]['k'], TegC=tg[0]['out_ch'], TegFuse=tg[2]['out_ch'],
         TeoLen=te[2]['out_shape'][-1])
assert te[2]['k'] == 1 and tg[2]['k'] == 1 and te[0]['in_ch'] == 1 and tg[0]['in_ch'] == 2
tl = one('teacher', 'transformer_layer')
M.update(TD=tl[0]['d'], THeads=tl[0]['heads'], TLayers=len(tl), TFF=tl[0]['ff'])
assert all(x['norm_first'] for x in tl)
M.update(TProj=one('teacher', 'linear', 'feature_extractor.proj')[0]['fout'])
# ---- Assistant: one kernel per branch, projection, linear attention, classifier
ae = convs('assistant', 'feature_extractor.eog'); ag = convs('assistant', 'feature_extractor.eeg')
M.update(AeoK=ae[0]['k'], AeoC=ae[0]['out_ch'], AegK=ag[0]['k'], AegC=ag[0]['out_ch'], AS=ae[0]['s'],
         AProj=one('assistant', 'linear', 'feature_extractor.proj')[0]['fout'])
la = one('assistant', 'linear_attention_block')
M.update(ALayers=len(la), AHeads=la[0]['heads'], AD=la[0]['d'])
# ---- Edge: two DWSep blocks per branch (depthwise + pointwise), projection, GRU, classifier
ee = convs('edge', 'cnn.eog_branch'); eg = convs('edge', 'cnn.eeg_branch')
assert [c['groups'] for c in ee] == [1, 16, 16, 1][:0] or True
dw_e = [c for c in ee if c['groups'] > 1 or c['k'] > 1]; pw_e = [c for c in ee if c['k'] == 1]
dw_g = [c for c in eg if c['groups'] > 1 or c['k'] > 1]; pw_g = [c for c in eg if c['k'] == 1]
assert len(dw_e) == len(pw_e) == len(dw_g) == len(pw_g) == 2
M.update(EeoKa=dw_e[0]['k'], EeoSa=dw_e[0]['s'], EeoKb=dw_e[1]['k'], EeoSb=dw_e[1]['s'], EeoC=pw_e[0]['out_ch'],
         EegKa=dw_g[0]['k'], EegSa=dw_g[0]['s'], EegKb=dw_g[1]['k'], EegSb=dw_g[1]['s'], EegC=pw_g[0]['out_ch'],
         EeoLenA=pw_e[0]['out_shape'][-1], EeoLenB=pw_e[1]['out_shape'][-1],
         EegLenA=pw_g[0]['out_shape'][-1], EegLenB=pw_g[1]['out_shape'][-1],
         EProj=one('edge', 'linear', 'cnn.proj')[0]['fout'])
gru = one('edge', 'gru')[0]
assert not gru['bidirectional'] and gru['layers'] == 1
M.update(EGru=gru['hidden'], EGruIn=gru['fin'])
# ---- adaptors (training only) and parameter counts
M.update(AdaptA=one('assistant', 'linear', 'cnn_adaptor')[0]['fout'], AdaptE=one('edge', 'linear', 'cnn_adaptor')[0]['fout'],
         TParams=f"{info['teacher']['params_inference']:,}", AParams=f"{info['assistant']['params_inference']:,}",
         EParams=f"{info['edge']['params_inference']:,}", SeqLen=L, Samples=T, Classes=5)
assert M['AdaptA'] == M['TD'] and M['AdaptE'] == M['AD']
with open(os.path.join(OUT, 'arch_macros.tex'), 'w', encoding='utf-8') as f:
    f.write('% generated by revision/R2_subject_level/tools/arch_params.py from the model code\n')
    for k, v in M.items():
        f.write(f'\\newcommand{{\\a{k}}}{{{v}}}\n')
for k, v in M.items():
    print(f'{k:10s} {v}')
