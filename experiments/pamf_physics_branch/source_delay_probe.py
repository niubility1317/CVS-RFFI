"""Source-only diagnostic of reference delay jumps under the fixed echo proxy."""
import argparse
import json
from collections import defaultdict

import torch

from experiments.cric_dura_comparison.f_data import SourceDataset
from experiments.cric_dura_comparison.io import read
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_paper_completion.model import load_reader
from experiments.pamf_physics_branch.geometry import bounded_echo, tangent
from experiments.pamf_physics_branch.objective import echo_parameters


def probe(config, role='V', device='cuda:0'):
    if role not in ('L_s', 'V') or any(config.get(k) for k in
            ('truth', 'target_truth', 'tests', 'views_root', 'target_inputs')):
        raise ValueError('Only declared labeled source roles are allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(config)
    contract = read(config['source_contract'])
    reader, _ = load_reader(config['source_output'], contract, config['paper_row'], device)
    reader.eval()
    data = SourceDataset(config['source_cache'], role, contract)
    meta = data.metadata
    metrics = defaultdict(lambda: defaultdict(float))
    names = ('time', 'frequency', 'pa', 'path', 'reference')
    for begin in range(0, len(data), 128):
        end = min(begin + 128, len(data))
        x = data.x[begin:end].to(device)
        coef, delay = echo_parameters(len(x), x.device, x.dtype, 202611200 + begin)
        augmented = bounded_echo(x, coef, delay)
        with torch.no_grad():
            core, sequence, reference = reader.core.id_backbone, reader.sequence_pool, reader.response_projection
            with capture_branches(core, sequence, reference) as raw:
                f0 = reader.features(x)
            with capture_branches(core, sequence, reference) as altered:
                f1 = reader.features(augmented)
            p0 = reader.classify_features(f0).softmax(1)
            p1 = reader.classify_features(f1).softmax(1)
            r0, r1 = reader.response.components(x), reader.response.components(augmented)
            changed = r0['selected_delay_samples'] != r1['selected_delay_samples']
            confidence, pred = p0.max(1)
            pred1 = p1.argmax(1)
            drift = {n: tangent(raw[n], altered[n])[1] for n in names}
            y = torch.as_tensor(meta['y'][begin:end], device=device, dtype=torch.long)
            rx = torch.as_tensor(meta['rx_i'][begin:end], device=device, dtype=torch.long)
            day = torch.as_tensor(meta['day_i'][begin:end], device=device, dtype=torch.long)
            conditions = [('overall', 'ALL', torch.ones_like(changed, dtype=torch.bool))]
            for key, value in (('receiver', rx), ('day', day), ('transmitter', y)):
                conditions.extend((key, str(int(v)), value == v) for v in value.unique().tolist())
            for dimension, value, mask in conditions:
                for change_name, change_mask in (('stable', ~changed), ('changed', changed)):
                    mask_group = mask & change_mask
                    out = metrics[(dimension, value, change_name)]
                    out['count'] += int(mask_group.sum())
                    out['correct'] += int(((pred == y) & mask_group).sum())
                    out['accepted'] += int(((confidence >= .95) & mask_group).sum())
                    out['accepted_wrong'] += int(((confidence >= .95) & (pred != y) & mask_group).sum())
                    out['prediction_flips'] += int(((pred != pred1) & mask_group).sum())
                    out['quality_sum'] += float(r0['quality'][mask_group].sum())
                    out['coherence_sum'] += float(r0['coherence'][mask_group].sum())
                    for n in names:
                        out['drift_' + n + '_sum'] += float(drift[n][mask_group].sum())
    rows = []
    for (dimension, value, change), sums in sorted(metrics.items()):
        count = int(sums['count'])
        row = dict(dimension=dimension, stratum=value, delay=change, **sums)
        row['count'] = count
        for key in ('correct', 'accepted', 'accepted_wrong', 'prediction_flips'):
            row[key] = int(row[key])
        if count:
            for key in ('quality', 'coherence', *('drift_' + n for n in names)):
                row[key + '_mean'] = row[key + '_sum'] / count
        rows.append(row)
    if sum(r['count'] for r in rows if r['dimension'] == 'overall') != contract['counts'][role]:
        raise ValueError('Source role coverage mismatch')
    return dict(status='SOURCE_DELAY_PROBE_COMPLETE', role=role, row_id=config['row_id'],
                source_only=True, target_reads=0, U_reads=0, fitted_updates=0,
                echo_seed_rule='202611200 + batch_start', rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--role', choices=('L_s', 'V'), default='V')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(probe(read(args.config), args.role, args.device), allow_nan=False))
