"""Compare frozen-model CE/PL and beta=0 branch relations on source V only."""
import contextlib
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

release = Path('/home/szu2070436088/2510044040/CV-SincNet/releases/20261110-phase1-pamf-ptbr-f0-s2026092701-m2-r01-release')
sys.path.insert(0, str(release))
from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_completion.runner import activate
from experiments.cric_dura_comparison.f_data import SourceDataset
from experiments.pamf_paper_completion.model import load_reader
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_physics_branch.geometry import physical_branch_loss, bounded_echo, tangent
from experiments.pamf_physics_branch.objective import echo_parameters

root = Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261110-phase1-pamf-ptbr-f0-s2026092701-m2-r01')
c = read(root / 'configs/source_diagnostic__baseline_pl-s2026092701.json')
activate(c)
contract = read(c['source_contract'])
with contextlib.redirect_stdout(sys.stderr):
    reader, _ = load_reader(c['source_output'], contract, c['paper_row'], 'cuda:0')
reader.eval().requires_grad_(True)
data = SourceDataset(c['source_cache'], 'V', contract)
meta = data.metadata
parser = argparse.ArgumentParser()
parser.add_argument('--all-v', action='store_true')
parser.add_argument('--beta', type=float, choices=(0., .5), default=0.)
args = parser.parse_args()
cells = sorted(set(zip(meta['y'].tolist(), meta['rx_i'].tolist(), meta['day_i'].tolist())))
repeats = 300 if args.all_v else 8
indices = [np.flatnonzero((meta['y'] == y) & (meta['rx_i'] == rx) & (meta['day_i'] == day))[:repeats]
           for y, rx, day in cells]
if len(cells) != 90 or any(len(i) != repeats for i in indices) or len(data) != 27000:
    raise ValueError('Exact stratified source V population required')
count = 90 * repeats
core = reader.core.id_backbone
modules = dict(time=[core.t_proj], frequency=[core.f_proj, core.freq_stats_proj],
               pa=[core.pa_proj, core.pa_stats_proj], path=[reader.sequence_pool],
               reference=[reader.response_projection])
groups = {name: list(dict.fromkeys(p for module in paths for p in module.parameters()))
          for name, paths in modules.items()}
if any(not values for values in groups.values()):
    raise ValueError('Empty real branch parameter group')
params = list(dict.fromkeys(p for values in groups.values() for p in values))
positions = {id(p): i for i, p in enumerate(params)}
names = ('ce', 'pl') + tuple(kind + '_' + branch for kind in ('L', 'U') for branch in groups)
acc = {name: [torch.zeros_like(p, device='cpu', dtype=torch.float64) for p in params] for name in names}
valid, accepted, wrong = {'L': 0, 'U': 0}, 0, 0
for repeat in range(repeats):
    part = np.asarray([row[repeat] for row in indices], dtype=np.int64)
    x = data.x.index_select(0, torch.as_tensor(part, dtype=torch.long)).to('cuda:0')
    y = torch.as_tensor(meta['y'][part].tolist(), dtype=torch.long, device='cuda:0')
    rx = torch.as_tensor(meta['rx_i'][part].tolist(), dtype=torch.long, device='cuda:0')
    day = torch.as_tensor(meta['day_i'][part].tolist(), dtype=torch.long, device='cuda:0')
    record = torch.as_tensor(part, dtype=torch.long, device='cuda:0')
    with capture_branches(core, reader.sequence_pool, reader.response_projection) as features:
        logits = reader.classify_features(reader.features(x))
    if args.beta:
        coefficients, delays = echo_parameters(len(x), x.device, x.dtype, 202611200 + 90 * repeat)
        transformed = bounded_echo(x, coefficients, delays)
        with torch.no_grad():
            with capture_branches(core, reader.sequence_pool, reader.response_projection) as changed:
                reader.features(transformed)
            directions = {branch: tangent(features[branch], changed[branch])[0] for branch in features}
    else:
        directions = {branch: torch.zeros_like(value) for branch, value in features.items()}
    conf, pseudo = logits.detach().softmax(1).max(1)
    selected = conf >= .95
    accepted += int(selected.sum())
    wrong += int((selected & (pseudo != y)).sum())
    objectives = {'ce': F.cross_entropy(logits, y, reduction='sum') / count,
                  'pl': (F.cross_entropy(logits, pseudo, reduction='none') * selected).sum() / count}
    for kind, labels, mask in (('L', y, torch.ones_like(selected)), ('U', pseudo, selected)):
        for branch, value in features.items():
            values = {branch: value}
            loss, counts = physical_branch_loss(values, {branch: directions[branch]}, labels, rx, day,
                                                record, mask, beta=args.beta, single_receiver=False)
            # The registered .05 relation weight applies to the mean of five branches.
            objectives[kind + '_' + branch] = .01 * loss / repeats
            if branch == 'time':
                valid[kind] += counts['valid_anchors']
    for j, name in enumerate(names):
        grads = torch.autograd.grad(objectives[name], params, retain_graph=j < len(names) - 1, allow_unused=True)
        for i, grad in enumerate(grads):
            if grad is not None:
                acc[name][i] += grad.detach().to('cpu', dtype=torch.float64)

def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / denominator) if denominator else None

result = {}
for branch, group in groups.items():
    indexes = [positions[id(p)] for p in group]
    vectors = {name: torch.cat([acc[name][i].flatten() for i in indexes])
               for name in ('ce', 'pl', 'L_' + branch, 'U_' + branch)}
    result[branch] = dict(parameter_count=vectors['ce'].numel(),
        gradient_norms={name: float(value.norm()) for name, value in vectors.items()},
        ce_vs_L=cosine(vectors['ce'], vectors['L_' + branch]),
        pl_vs_U=cosine(vectors['pl'], vectors['U_' + branch]),
        ce_vs_pl=cosine(vectors['ce'], vectors['pl']),
        L_vs_U=cosine(vectors['L_' + branch], vectors['U_' + branch]),
        ce_pl_sum_norm=float((vectors['ce'] + vectors['pl']).norm()),
        relation_sum_norm=float((vectors['L_' + branch] + vectors['U_' + branch]).norm()),
        ce_pl_sum_vs_relation_sum=cosine(vectors['ce'] + vectors['pl'],
                                        vectors['L_' + branch] + vectors['U_' + branch]))
if any(p.grad is not None for p in reader.parameters()):
    raise ValueError('Probe must not apply or retain parameter gradients')
print(json.dumps(dict(status='SOURCE_RELATION_GRADIENT_PROBE_COMPLETE', source_only=True, role='V', count=count,
    batch_rule=str(repeats) + ' batches of one packet per source TX/RX/day cell', accepted=accepted, accepted_wrong=wrong,
    valid_anchors=valid, target_reads=0, U_reads=0, optimizer_updates=0, model_mode='eval', beta=args.beta,
    relation_weight_per_branch=.01, scope='source V diagnostic proxy; not actual train-mode objective',
    branches=result), allow_nan=False))
