"""Matched single-seed LoRa PAMF-Tri shape and loss screen."""
import argparse
from functools import partial
import json
from pathlib import Path
from statistics import mean
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_ssl_research.u_gradprotect_objective import protected_loss
from .adaptive_repair import load_u, practical_quality
from .comparison import seed_all
from .data import TARGET, VIEWS
from .full_packet_data import LIMIT
from .full_supervised import batches, handles, role
from .full_tri_repl_lora import initialized, labeled_loss, unlabeled_loss
from .packet_runner import schedule
from .physical import ChirpResponse
from .refine_runner import start
from .runner import activate


RUN = '20261125-phase1-lora-pamf-tri-shape-loss-m8-r01'
SEED = 2026092702
EPOCHS = 4
STEPS = 88
U_BATCH = 64
ROWS = {
    'control': {},
    'as_double': {'physical_weight': 2.},
    'kl_half': {'kl_weight': .5},
    'u_half': {'u_weight': 1.5},
    'three_chirps': {'chirps': (0, 2, 4)},
    'signature_half': {'signature_half': True},
    'aligned_pool12': {'align_bins': 12},
    'fusion192': {'fusion_dim': 192},
}
LOCAL_ROWS = ('control', 'three_chirps')


class ExpandFeatures(nn.Linear):
    """Continue the existing feature coordinates and learn additional ones."""

    def __init__(self, width, device):
        super().__init__(160, width - 160, bias=False, device=device)
        with torch.no_grad():
            self.weight.mul_(.1)

    def forward(self, x):
        return torch.cat((x, super().forward(x)), dim=1)


def expanded_raw_features(encoder, x):
    return encoder.fusion_tail(type(encoder).features(encoder, x))


def resize_alignment(model, bins):
    old = model.encoder[11]
    channels = old.in_features // 8
    replacement = nn.Linear(channels * bins, old.out_features, device=old.weight.device)
    with torch.no_grad():
        kernel = old.weight.reshape(old.out_features, channels, 8)
        replacement.weight.copy_(F.interpolate(kernel, size=bins, mode='linear', align_corners=False)
                                 .flatten(1) * (8 / bins))
        replacement.bias.copy_(old.bias)
    model.encoder[9] = nn.AdaptiveAvgPool1d(bins)
    model.encoder[11] = replacement


def configure(model, row):
    if 'chirps' in row:
        model.raw.views = len(row['chirps'])
    raw = model.raw.encoder
    if row.get('signature_half'):
        for view in raw.sequence_pool.views:
            view.tuning_half_path = True
    if bins := row.get('align_bins'):
        resize_alignment(model, bins)
    if width := row.get('fusion_dim'):
        device = model.head.weight.device
        raw.fusion_tail = ExpandFeatures(width, device)
        raw.features = MethodType(expanded_raw_features, raw)
        model.encoder.add_module('fusion_tail', ExpandFeatures(width, device))
        for head in (raw.core.id_backbone.cls_head, model.head):
            head.weight = nn.Parameter(F.pad(head.weight.detach(), (0, width - 160)))
        model.head.in_features = width
    return model


def iq(x, row):
    return x[:, row['chirps']] if 'chirps' in row else x


def model_for(rt, root, contract, row, device):
    if rt.get('local_parent'):
        from .full_adaptive import load_supervised_continuation
        model = load_supervised_continuation(Path(rt['local_parent']), contract, SEED, device)
    else:
        model = initialized(root, contract, SEED, device, full=True)
    return configure(model, row)


def shape_trace(model, x):
    raw = model.raw.encoder
    seen = {}
    hooks = []
    for name, layer in (
        ('shallow', raw.core.id_backbone.time_down),
        ('middle', raw.core.id_backbone.t2.pool),
        ('aligned', model.encoder[9]),
    ):
        def capture(_, args, out, name=name):
            seen[name] = list(out.shape[1:])
        hooks.append(layer.register_forward_hook(capture))
    def capture_signature(_, args):
        seen['signature'] = list(args[0].shape[1:])
    hooks.append(raw.sequence_pool.register_forward_pre_hook(capture_signature))
    try:
        with torch.no_grad():
            score, feature = model.eval()(x)
        seen.update(input=list(x.shape[1:]), fused=list(feature.shape[1:]), logits=list(score.shape[1:]),
                    signature_views=len(raw.sequence_pool.views),
                    signature_path_points=4 if getattr(raw.sequence_pool.views[0], 'tuning_half_path', False) else 8,
                    parameters=sum(p.numel() for p in model.parameters()))
    finally:
        for hook in hooks:
            hook.remove()
    if seen['signature_views'] != 3 or seen['logits'] != [10]:
        raise ValueError('LoRa Tri view or classifier shape differs')
    return seen


def smoke(rt):
    root = Path(rt['run_root'])
    source, views, contract = handles(rt)
    model = model_for(rt, root, contract, ROWS['control'], 'cpu')
    raw = np.load(views/'L_s.npy', mmap_mode='r', allow_pickle=False)
    with np.load(source/'L_s_metadata.npz', allow_pickle=False) as labels:
        y, rx = labels['y'], labels['rx_i']
        idx = sorted(next(batches(y, rx, np.random.default_rng(721)))[:8])
    a, b = tensor(raw[0, idx]), tensor(raw[1, idx])
    y, rx = tensor(y[idx]), tensor(rx[idx])
    results = {}
    from copy import deepcopy
    for name, row in ROWS.items():
        f = configure(deepcopy(model), row)
        x, p = iq(a, row), iq(b, row)
        trace = shape_trace(f, x)
        f.train()
        options = mechanism_options(row)
        labeled, _ = labeled_loss(f, x, p, y, rx, 1, mechanism=options)
        u, _, mask, terms = unlabeled_loss(f, x, rx, 1, row.get('u_weight', 3.),
                                          forced_pseudo=y, mechanism=options)
        safe, _, _ = protected_loss(labeled, u, trainable(f))
        safe.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(f.parameters(), 5., error_if_nonfinite=True))
        if not (mask.all() and terms['forwards'] == 6 and terms['partners'] and grad > 0.):
            raise ValueError(f'No-query gradient smoke failed for {name}')
        results[name] = dict(trace, gradient=grad)
    write(root/'smoke.json', dict(status='PASS', rows=results, U_reads=0, target_reads=0))


def screen_views(source, views, contract, device):
    with np.load(source/'V_metadata.npz', allow_pickle=False) as meta:
        y, rx = meta['y'], meta['rx_i']
        if meta['ids'].tolist() != contract['role_ids']['V']:
            raise ValueError('Source validation order differs')
    rng = np.random.default_rng(20261125)
    index = np.sort(np.concatenate([
        rng.choice(np.flatnonzero((y == cls) & (rx == receiver)), 24, replace=False)
        for cls in range(10) for receiver in range(5)
    ]))
    raw = np.load(views/'V.npy', mmap_mode='r', allow_pickle=False)
    return tensor(raw[:, index], device), tensor(y[index], device), tensor(rx[index], device)


@torch.no_grad()
def measure(f, raw, y, rx, row):
    values = {}
    f.eval()
    for view, v in zip(VIEWS, raw, strict=True):
        q = torch.cat([f(iq(b, row))[0] for b in v.split(64)])
        ok = q.argmax(dim=1) == y
        values[view] = dict(accuracy=float(ok.float().mean()), nll=float(F.cross_entropy(q,y)),
                            by_rx=[float(ok[rx==r].float().mean()) for r in range(5)],
                            by_tx=[float(ok[y==c].float().mean()) for c in range(10)])
    return values


def mechanism_options(row):
    return {key: row[key] for key in ('kl_weight', 'contrast_weight', 'physical_weight') if key in row}


def trainable(f):
    return [(name, param) for name, param in f.named_parameters() if param.requires_grad]


def train(rt, name):
    row = ROWS[name]
    root = Path(rt['run_root'])
    source, views, contract = handles(rt)
    out = root/'F'/name
    out.mkdir(parents=True, exist_ok=False)
    device = 'cuda:0'
    seed_all(SEED)
    f = model_for(rt, root, contract, row, device)
    # The architecture variants consume extra initialization draws. Training
    # samples, permutations, and style draws start from the same seed.
    seed_all(SEED)
    x, yn, rn, _ = role(source, views, contract, 'L_s', device)
    y, rx = tensor(yn, device), tensor(rn, device)
    v, vy, vr = screen_views(source, views, contract, device)
    u, ur = load_u(source, views, contract, device)
    practical = tensor(np.load(views/'U_s_practical.npy', mmap_mode='r', allow_pickle=False), device)
    valid = practical_quality(ChirpResponse(fft_factor=4).to(device), practical)
    opt = torch.optim.Adam(f.parameters(), lr=6e-5, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS*STEPS, eta_min=1e-5)
    named = trainable(f)
    rng = np.random.default_rng(721)
    u_rng = torch.Generator(device=device).manual_seed(SEED)
    loss_options = mechanism_options(row)
    step = 0
    write(out/'initial.json', dict(views=measure(f, v, vy, vr, row), shapes=shape_trace(f, iq(x[0,:2],row))))
    with (out/'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, EPOCHS+1):
            order = torch.randperm(len(u), device=device, generator=u_rng)[:STEPS*U_BATCH].reshape(STEPS,U_BATCH)
            loss_l = loss_u = accepted = partners = flags = 0
            for ids, pos in zip(batches(yn, rn, rng), order, strict=True):
                i = tensor(ids, device)
                step += 1
                scene = (i+step+epoch)%6+1
                f.train()
                opt.zero_grad(set_to_none=True)
                lab, _ = labeled_loss(f, iq(x[0,i], row), iq(x[scene,i], row), y[i], rx[i], step,
                                      mechanism=loss_options)
                unlab, _, keep, detail = unlabeled_loss(
                    f, iq(u[pos], row), ur[pos], step, row.get('u_weight', 3.),
                    practical=iq(practical[pos], row), practical_valid=valid[pos], mechanism=loss_options)
                safeguarded, _, _ = protected_loss(lab, unlab, named)
                safeguarded.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 5., error_if_nonfinite=True)
                opt.step()
                sched.step()
                loss_l += float(lab.detach())
                loss_u += float(unlab.detach())
                accepted += int(keep.sum())
                partners += detail['partners']
                flags += detail['practical']
            record = dict(epoch=epoch, step=step, L_loss=loss_l/STEPS, U_loss=loss_u/STEPS,
                          U_seen=STEPS*U_BATCH, U_accepted=accepted, U_partners=partners,
                          U_practical=flags, V=measure(f, v, vy, vr, row))
            log.write(json.dumps(record, allow_nan=False)+'\n')
            log.flush()
            print('LORA_TRI_SCREEN',name,epoch,step,
                  record['V']['clean']['accuracy'],round(scene_score(record['V']),6),accepted,flush=True)
    if step != EPOCHS*STEPS:
        raise ValueError('Source budget differs')
    torch.save(f.state_dict(), out/'model.pt')
    write(out/'complete.json', dict(status='SOURCE_FROZEN', seed=SEED, method=name, settings=row,
                                  epochs=EPOCHS, steps=step, target_reads=0, U_truth_reads=0))


def scene_score(views):
    return mean(views[k]['accuracy'] for k in VIEWS[1:])


def select(root):
    results = {}
    for name in ROWS:
        folder = root/'F'/name
        stamp = read(folder/'complete.json')
        if (stamp['status'], stamp['steps'], stamp['target_reads'], stamp['U_truth_reads']) != (
                'SOURCE_FROZEN', EPOCHS*STEPS, 0, 0):
            raise ValueError(f'Source checkpoint not frozen: {name}')
        logs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        if [r['step'] for r in logs] != [STEPS*i for i in range(1, EPOCHS+1)]:
            raise ValueError(f'Incomplete source log: {name}')
        results[name] = logs[-1]['V']
    ref = results['control']
    eligible = [n for n,v in results.items() if n != 'control'
                and v['clean']['accuracy'] >= ref['clean']['accuracy'] - .02
                and min(v['clean']['by_rx']) >= min(ref['clean']['by_rx']) - .03]
    eligible.sort(key=lambda n: (scene_score(results[n]), results[n][VIEWS[-1]]['accuracy'],
                                 results[n]['clean']['accuracy']), reverse=True)
    promoted = ['control', *[n for n in eligible if scene_score(results[n]) > scene_score(ref)][:2]]
    write(root/'selection.json', dict(status='SOURCE_SELECTED', names=promoted, source=results, target_reads=0))
    return promoted


@torch.no_grad()
def predict(rt, name):
    root = Path(rt['run_root'])
    if name not in read(root/'selection.json')['names']:
        raise ValueError('Target inference requires a source-frozen eligible row')
    _, _, contract = handles(rt)
    target = Path(rt['target_views'])
    if target != Path(rt['parent_root'])/'target_views' or read(target/'complete.json') != dict(
            status='TARGET_READY',protocol=contract['protocol'],target_truth_read=False,
            receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS)):
        raise ValueError('Registered blind target cache required')
    out = root/'predictions'/name
    out.mkdir(parents=True, exist_ok=False)
    f = model_for(rt, root, contract, ROWS[name], 'cuda:0').eval()
    f.load_state_dict(torch.load(root/'F'/name/'model.pt',map_location='cuda:0',weights_only=True),strict=True)
    for rx in TARGET:
        a = np.load(target/f'{rx}.npy', mmap_mode='r', allow_pickle=False)
        if a.shape != (len(VIEWS),LIMIT,6,2,256) or a.dtype != np.float32:
            raise ValueError('Unexpected blind IQ shape')
        for v, samples in zip(VIEWS, a, strict=True):
            codes = []
            for batch in tensor(samples,'cuda:0').split(64):
                logits = f(iq(batch, ROWS[name]))[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite LoRa score')
                codes.extend(logits.argmax(1).tolist())
            np.save(out/f'{rx}__{v}__direct.npy', np.asarray(codes,dtype=np.int16))
        print('LORA_TRI_BLIND',name,rx,flush=True)
    write(out/'complete.json', dict(method=name,seed=SEED,status='PREDICTIONS_FROZEN',
          target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS)))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    launch = partial(start, module=__spec__.name, extra_keys=('local_parent',) if rt.get('local_parent') else ())
    try:
        write(state, dict(status='SOURCE_SMOKE'))
        if launch(rt, 'smoke', 'all').wait():
            raise RuntimeError('No-query smoke failed')
        write(state, dict(status='SOURCE_TRAINING'), exclusive=False)
        rows = LOCAL_ROWS if rt.get('local_parent') else ROWS
        schedule(rt, [('train',n) for n in rows], launch)
        if rt.get('local_parent'):
            write(state, dict(status='SOURCE_DIAGNOSTIC_COMPLETE'), exclusive=False)
            return
        names = select(root)
        write(state, dict(status='PREDICTING', names=names), exclusive=False)
        schedule(rt, [('predict',n) for n in names], launch)
        write(state, dict(status='PREDICTIONS_FROZEN', names=names), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('dispatch','smoke','train','predict'))
    parser.add_argument('runtime', type=Path)
    parser.add_argument('name', nargs='?', default='')
    args = parser.parse_args()
    rt = read(args.runtime)
    if args.kind == 'dispatch':
        dispatch(rt)
    else:
        activate(rt)
        if args.kind == 'smoke':
            smoke(rt)
        else:
            {'train':train,'predict':predict}[args.kind](rt,args.name)
