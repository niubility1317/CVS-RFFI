"""Read-only reference-branch gradient probe across existing source contracts."""
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
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_physics_branch.geometry import bounded_echo, physical_branch_loss, tangent
from experiments.pamf_physics_branch.objective import echo_parameters

root = Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261110-phase1-pamf-ptbr-f0-s2026092701-m2-r01')
base = read(root / 'configs/source_diagnostic__baseline_pl-s2026092701.json')
context = read(release / 'context.json')
activate(base)
settings = ('F0', 'F1_rx_repartition_r01', 'F2_rx_repartition_r01', 'ManyRx',
            'label01', 'label05', 'm1-natural', 'm2-natural', 'm3-natural', 'm4-natural', 'm6-natural')
rows = {}
for setting in settings:
    handle = context['handles'][setting]
    entry = context['existing'][setting + '-ordinary_PL-s2026092701']
    contract = read(handle['source_contract'])
    with contextlib.redirect_stdout(sys.stderr):
        reader, _ = load_reader(entry['source_output'], contract, base['paper_row'], 'cuda:0')
    reader.eval().requires_grad_(True)
    data = SourceDataset(handle['source_cache'], 'V', contract)
    meta = data.metadata
    cells = sorted(set(zip(meta['y'].tolist(), meta['rx_i'].tolist(), meta['day_i'].tolist())))
    picks = [np.flatnonzero((meta['y'] == y) & (meta['rx_i'] == rx) & (meta['day_i'] == day))[:8]
             for y, rx, day in cells]
    if not cells or any(len(x) != 8 for x in picks):
        raise ValueError('At least eight source V packets per observed TX/RX/day cell required')
    single_rx = len(set(meta['rx_i'].tolist())) == 1
    selected_count = 8 * len(cells)
    params = list(reader.response_projection.parameters())
    names = ('ce', 'pl', 'relation_L', 'relation_U')
    grad = {name: [torch.zeros_like(p, device='cpu', dtype=torch.float64) for p in params] for name in names}
    totals = dict(accepted=0, accepted_wrong=0, anchors_L=0, anchors_U=0, delay_changes=0)
    for repeat in range(4):
        part = np.asarray([item[i] for item in picks for i in (2*repeat, 2*repeat+1)], dtype=np.int64)
        x = data.x.index_select(0, torch.as_tensor(part, dtype=torch.long)).to('cuda:0')
        y = torch.as_tensor(meta['y'][part].tolist(), dtype=torch.long, device='cuda:0')
        rx = torch.as_tensor(meta['rx_i'][part].tolist(), dtype=torch.long, device='cuda:0')
        day = torch.as_tensor(meta['day_i'][part].tolist(), dtype=torch.long, device='cuda:0')
        ids = torch.as_tensor(part, dtype=torch.long, device='cuda:0')
        core = reader.core.id_backbone
        with capture_branches(core, reader.sequence_pool, reader.response_projection) as feature:
            logits = reader.classify_features(reader.features(x))
        coefficient, delay = echo_parameters(len(x), x.device, x.dtype, 202611200 + repeat * 1000)
        changed_x = bounded_echo(x, coefficient, delay)
        with torch.no_grad():
            with capture_branches(core, reader.sequence_pool, reader.response_projection) as changed:
                reader.features(changed_x)
            direction, _ = tangent(feature['reference'], changed['reference'])
            before = reader.response.components(x)['selected_delay_samples']
            after = reader.response.components(changed_x)['selected_delay_samples']
            totals['delay_changes'] += int((before != after).sum())
        confidence, pseudo = logits.detach().softmax(1).max(1)
        mask = confidence >= .95
        totals['accepted'] += int(mask.sum())
        totals['accepted_wrong'] += int((mask & (pseudo != y)).sum())
        values, directions = {'reference': feature['reference']}, {'reference': direction}
        relation_l, l_stats = physical_branch_loss(values, directions, y, rx, day, ids,
            torch.ones_like(mask), beta=.5, single_receiver=single_rx)
        relation_u, u_stats = physical_branch_loss(values, directions, pseudo, rx, day, ids,
            mask, beta=.5, single_receiver=single_rx)
        totals['anchors_L'] += l_stats['valid_anchors']
        totals['anchors_U'] += u_stats['valid_anchors']
        losses = (F.cross_entropy(logits, y, reduction='sum') / selected_count,
                  (F.cross_entropy(logits, pseudo, reduction='none') * mask).sum() / selected_count,
                  .01 * relation_l / 4, .01 * relation_u / 4)
        for j, name in enumerate(names):
            values = torch.autograd.grad(losses[j], params, retain_graph=j < len(names)-1, allow_unused=True)
            for i, value in enumerate(values):
                if value is not None:
                    grad[name][i] += value.detach().to('cpu', dtype=torch.float64)
    vectors = {name: torch.cat([v.flatten() for v in values]) for name, values in grad.items()}
    direct, relation = vectors['ce'] + vectors['pl'], vectors['relation_L'] + vectors['relation_U']
    direct_norm, relation_norm = float(direct.norm()), float(relation.norm())
    if any(p.grad is not None for p in reader.parameters()):
        raise ValueError('Probe cannot apply or retain parameter gradients')
    rows[setting] = dict(V_total=len(data), selected=selected_count, cells=len(cells), single_receiver=single_rx,
        direct_ce_pl_norm=direct_norm, reference_relation_norm=relation_norm,
        relation_to_direct=relation_norm/direct_norm if direct_norm else None,
        cosine=float(torch.dot(direct, relation)/(direct_norm*relation_norm)) if direct_norm and relation_norm else None,
        **totals)
    del reader, data, x, logits, feature, changed, losses
    torch.cuda.empty_cache()
print(json.dumps(dict(status='SOURCE_CROSS_SETTING_PROBE_COMPLETE', source_only=True, target_reads=0, U_reads=0,
    seed=2026092701, optimizer_updates=0, selected_rule='first eight V packets per observed TX/RX/day cell',
    batch_rule='four balanced batches, two packets per cell', beta=.5,
    scope='frozen original PL models, eval mode, reference parameters only; local proxy, not training trajectory',
    rows=rows), allow_nan=False))
