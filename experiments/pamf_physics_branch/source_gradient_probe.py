"""Source V only; frozen F21; gradients measured, never applied or saved as weights."""
import contextlib
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

root = Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261110-phase1-pamf-ptbr-f0-s2026092701-m2-r01')
c = read(root / 'configs/source_diagnostic__baseline_pl-s2026092701.json')
activate(c)
contract = read(c['source_contract'])
with contextlib.redirect_stdout(sys.stderr):
    reader, _ = load_reader(c['source_output'], contract, c['paper_row'], 'cuda:0')
reader.eval().requires_grad_(True)
data = SourceDataset(c['source_cache'], 'V', contract)
meta = data.metadata
cells = sorted(set(zip(meta['y'].tolist(), meta['rx_i'].tolist(), meta['day_i'].tolist())))
indices = np.asarray([int(i) for y, rx, day in cells
    for i in np.flatnonzero((meta['y'] == y) & (meta['rx_i'] == rx) & (meta['day_i'] == day))[:8]], dtype=np.int64)
if len(cells) != 90 or len(indices) != 720:
    raise ValueError('Source V stratification differs')
core = reader.core.id_backbone
modules = dict(time=core.t_proj, frequency_main=core.f_proj, frequency_stats=core.freq_stats_proj,
               pa_main=core.pa_proj, pa_stats=core.pa_stats_proj, path=reader.sequence_pool,
               reference=reader.response_projection, shared_time_down=core.time_down, shared_t2=core.t2)
groups = {name: list(module.parameters()) for name, module in modules.items()}
empty_groups = [name for name, group in groups.items() if not group]
groups = {name: group for name, group in groups.items() if group}
params = list(dict.fromkeys(p for group in groups.values() for p in group))
lookup = {id(p): i for i, p in enumerate(params)}
if not params:
    raise ValueError('No actual branch parameters')
components = ('true_all', 'true_accepted', 'true_rejected', 'pl_correct', 'pl_wrong')
acc = [[torch.zeros_like(p, device='cpu', dtype=torch.float64) for p in params] for _ in components]
accepted = pseudo_wrong = 0
for start in range(0, len(indices), 32):
    part = indices[start:start + 32]
    x = data.x.index_select(0, torch.as_tensor(part, dtype=torch.long)).to('cuda:0')
    y = torch.as_tensor(meta['y'][part].tolist(), dtype=torch.long, device='cuda:0')
    logits = reader.classify_features(reader.features(x))
    confidence, pseudo = logits.detach().softmax(1).max(1)
    mask = confidence >= .95
    accepted += int(mask.sum())
    pseudo_wrong += int((mask & (pseudo != y)).sum())
    true_loss = F.cross_entropy(logits, y, reduction='none')
    pseudo_loss = F.cross_entropy(logits, pseudo, reduction='none')
    good = mask & (pseudo == y)
    wrong = mask & (pseudo != y)
    losses = (true_loss.sum(), (true_loss * mask).sum(), (true_loss * ~mask).sum(),
              (pseudo_loss * good).sum(), (pseudo_loss * wrong).sum())
    for j, loss in enumerate(losses):
        grads = torch.autograd.grad(loss, params, retain_graph=j < len(losses) - 1, allow_unused=True)
        for i, g in enumerate(grads):
            if g is not None:
                acc[j][i] += g.detach().to('cpu', dtype=torch.float64) / len(indices)
result = {}
for name, group in groups.items():
    indexes = [lookup[id(p)] for p in group]
    vectors = {key: torch.cat([acc[j][i].flatten() for i in indexes]) for j, key in enumerate(components)}
    vectors['pl_all'] = vectors['pl_correct'] + vectors['pl_wrong']
    if (vectors['true_all'] - vectors['true_accepted'] - vectors['true_rejected']).norm() > 1e-5 * vectors['true_all'].norm() + 1e-9:
        raise ValueError('True-label gradient decomposition failed')
    def cosine(a, b):
        x, y = vectors[a], vectors[b]
        den = float(x.norm() * y.norm())
        return float(torch.dot(x, y) / den) if den else None
    result[name] = dict(parameters=vectors['true_all'].numel(),
                        gradient_norms={k: float(v.norm()) for k, v in vectors.items()},
                        true_vs_pl=cosine('true_all', 'pl_all'),
                        true_vs_true_accepted=cosine('true_all', 'true_accepted'),
                        true_vs_pl_correct=cosine('true_all', 'pl_correct'),
                        true_vs_pl_wrong=cosine('true_all', 'pl_wrong'))
if any(p.grad is not None for p in reader.parameters()):
    raise ValueError('Probe unexpectedly populated parameter gradients')
print(json.dumps(dict(status='SOURCE_GRADIENT_PROBE_COMPLETE', source_only=True, role='V', count=len(indices),
    sample_rule='first 8 per source TX/RX/day', accepted=accepted, accepted_wrong=pseudo_wrong, threshold=.95,
    ce='clean hard true-label CE', pl='clean hard pseudo-label CE, mean over all 720',
    model_mode='eval; same inputs and logits; no update', target_reads=0, U_reads=0, optimizer_updates=0,
    scope='descriptive source V surrogate; not the complete multi-view L loss or actual U training dynamics',
    groups=result, groups_without_parameters=empty_groups), allow_nan=False))
