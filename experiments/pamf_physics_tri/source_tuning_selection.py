"""Fixed selection rule for the source-development temperature pilot."""
import math

from . import source_tuning_plan as plan


def select(scores):
    rows = plan.candidate_rows()
    by_id = {s['row_id']: s for s in scores}
    if len(by_id) != len(scores) or set(by_id) != {r['row_id'] for r in rows}:
        raise ValueError('Exactly one source-development score per registered candidate is required')
    for score in scores:
        for key in ('clean_accuracy', 'physical_accuracy'):
            if not math.isfinite(score[key]) or not 0 <= score[key] <= 1:
                raise ValueError('Source-development accuracies must be finite proportions')
    return max(rows, key=lambda r: (
        (by_id[r['row_id']]['clean_accuracy'] + by_id[r['row_id']]['physical_accuracy']) / 2,
        r['temperature'] == .2, -r['temperature']))


def score_source(source, device):
    """Evaluate the frozen default head on held-out source L and fixed simulated views."""
    from pathlib import Path
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset, FixedEpochLoader
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.t1_practical import Practical
    from experiments.pamf_paper_completion.model import load_reader
    from .source_development import partition_context, verify_parent_partition
    from .source_tuning_plan import validate

    source = Path(source)
    done = read(source / 'complete.json')
    cfg = done['config']
    validate(cfg)
    if (done.get('status') != 'SOURCE_FROZEN' or not done.get('source_development_only')
            or done.get('target_access') or done.get('U_truth_read') or len(done.get('segments', [])) != 3):
        raise ValueError('Completed source-development trajectory required')
    for segment in done['segments']:
        verify_parent_partition(cfg, segment['config'])
    contract = read(cfg['source_contract'])
    reader, _ = load_reader(source, contract, cfg['paper_row'], device)
    labeled = SourceDataset(cfg['source_cache'], 'L_s', contract)
    base = dict(train_loader=FixedEpochLoader(labeled, 128, cfg['model_seed']), split_info={})
    part = cfg['source_development']
    _, dev = partition_context(base, part['train_ids'], part['dev_ids'], model_seed=cfg['model_seed'])
    loader = torch.utils.data.DataLoader(dev, batch_size=128, shuffle=False, num_workers=0)
    physical = Practical(cfg)
    correct_clean = correct_physical = count = 0
    with torch.no_grad():
        for step, (x, y, _, meta) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            changed = physical(x, meta, epoch=999, step=step)
            clean_logits = reader.classify_features(reader.features(x))
            physical_logits = reader.classify_features(reader.features(changed))
            if not bool(torch.isfinite(clean_logits).all() and torch.isfinite(physical_logits).all()):
                raise ValueError('Finite source-development predictions required')
            correct_clean += int((clean_logits.argmax(1) == y).sum())
            correct_physical += int((physical_logits.argmax(1) == y).sum())
            count += len(y)
    if count != len(dev) or not count:
        raise ValueError('Complete source-development coverage required')
    return dict(row_id=cfg['row_id'], count=count, clean_accuracy=correct_clean / count,
                physical_accuracy=correct_physical / count, physical_scenarios=dict(physical.coverage),
                selection_data='held-out source L only', target_reads=0, model_updates=0)
