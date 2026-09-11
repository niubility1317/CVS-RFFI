"""Source-only candidate noise estimates and competitor selection.

No numerical tuning defaults are frozen here. Estimation may run before source
validation; application requires the caller's explicit source-frozen flag.
"""
from dataclasses import dataclass
from collections import defaultdict
import math
import torch


@dataclass(frozen=True)
class DecisionCalibrationConfig:
    quantile: float = None
    min_groups: int = None
    group_size: int = None
    max_delta: float = None
    top_k: int = None
    min_pair_support: int = None
    boundary_margin: float = None

    def validate(self):
        if any(getattr(self, key) is None for key in self.__dataclass_fields__):
            raise ValueError('source calibration parameters are UNFROZEN; all must be explicit')
        if (not 0 <= self.quantile <= 1 or self.min_groups < 1 or self.group_size < 1
                or self.max_delta < 0 or self.top_k < 1 or self.min_pair_support < 1
                or not math.isfinite(self.max_delta) or not math.isfinite(self.boundary_margin)):
            raise ValueError('invalid source calibration parameters')


def _inputs(logits, labels, records):
    x = torch.as_tensor(logits).detach().double().cpu()
    y = torch.as_tensor(labels).detach().long().cpu()
    if x.ndim != 2 or x.shape[1] < 2 or len(records) != len(x) or y.shape != (len(x),):
        raise ValueError('invalid source calibration dimensions')
    if not torch.isfinite(x).all() or (y < 0).any() or (y >= x.shape[1]).any():
        raise ValueError('invalid source calibration values')
    if len({r.physical_sample_id for r in records}) != len(records):
        raise ValueError('calibration requires distinct physical records, not augmented duplicates')
    return x, y, x.gather(1,y[:,None]) - x


def estimate_source_noise(logits, labels, records, config, *, source_role):
    """L_s repeats: same TX/RX/day/condition, disjoint equally sized groups.

    Caller freezes model/augmentation for the supplied logits. Each difference
    uses two groups of group_size physical records; leftovers are disclosed.
    Sparse conditional estimates shrink by complete fallback to class-pair,
    then global distributions. Unavailable global evidence remains None.
    """
    config.validate()
    if source_role != 'L_s':
        raise ValueError('noise calibration permits source L_s only')
    x, y, margins = _inputs(logits, labels, records)
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[(int(y[i]), r.rx_id, r.day_id, r.condition_id)].append(i)
    conditional, pooled, global_diffs = defaultdict(list), defaultdict(list), []
    group_evidence = []
    used_records = 0
    for key in sorted(groups, key=repr):
        idx = sorted(groups[key], key=lambda i: repr(records[i].physical_sample_id))
        width = 2 * config.group_size
        for start in range(0, len(idx)-width+1, width):
            a, b = idx[start:start+config.group_size], idx[start+config.group_size:start+width]
            difference = (margins[a].mean(0)-margins[b].mean(0)).abs()
            used_records += width
            group_evidence.append(dict(cell=key, group_a=tuple(records[i].physical_sample_id for i in a),
                group_b=tuple(records[i].physical_sample_id for i in b), records_per_group=config.group_size))
            for competitor in range(x.shape[1]):
                if competitor == key[0]:
                    continue
                value = float(difference[competitor])
                conditional[(key[0],competitor,key[2],key[3])].append(value)
                pooled[(key[0],competitor)].append(value)
                global_diffs.append(value)
    def summary(values, independent_groups=None):
        values = torch.tensor(values,dtype=torch.float64)
        count = len(values) if independent_groups is None else independent_groups
        return dict(groups=count, comparison_values=len(values),variance=float(values.var(unbiased=False)) if len(values) else None,
                    delta=min(config.max_delta,float(values.quantile(config.quantile)))
                          if count>=config.min_groups else None,
                    absolute_margin_differences=values.tolist())
    return dict(source_role=source_role, conditional={k:summary(v) for k,v in conditional.items()},
        pair={k:summary(v) for k,v in pooled.items()}, global_noise=summary(global_diffs,len(group_evidence)),
        matched_groups=group_evidence, used_physical_records=used_records,
        unused_physical_records=len(records)-used_records, num_classes=x.shape[1],
        config=config, source_frozen=False)


def select_source_competitors(logits, labels, records, config, *, source_role):
    """Single source V: rank competing classes by observed boundary risk."""
    config.validate()
    if source_role != 'V':
        raise ValueError('competitor selection permits single source V only')
    x,y,margins = _inputs(logits,labels,records)
    selected, evidence = {}, {}
    for target in range(x.shape[1]):
        rows = y == target
        support = int(rows.sum())
        ranked = []
        for competitor in range(x.shape[1]):
            if competitor == target:
                continue
            boundary = int((margins[rows,competitor] <= config.boundary_margin).sum())
            confusion = int(((x.argmax(1) == competitor) & rows).sum())
            evidence[(target,competitor)] = dict(physical_records=support,
                boundary_records=boundary, confusion_records=confusion)
            ranked.append((-boundary,-confusion,competitor))
        selected[target] = tuple(row[2] for row in sorted(ranked)[:config.top_k]) if support >= config.min_pair_support else ()
    return dict(source_role=source_role,selected=selected,evidence=evidence,
                config=config,num_classes=x.shape[1],source_frozen=False)


def calibrated_decision_tensors(noise, competitors, labels, day_ids, condition_ids, *, device=None, dtype=None):
    """Build detached [N,C] delta/weights only after source-side freezing.

    Setting source_frozen is a caller-owned source validation decision; this
    module neither infers it from target performance nor adds approval gates.
    """
    if not noise.get('source_frozen') or (competitors is not None and not competitors.get('source_frozen')):
        raise ValueError('source calibration is UNFROZEN and cannot enable the mechanism')
    if noise['source_role'] != 'L_s' or (competitors is not None and competitors['source_role'] != 'V'):
        raise ValueError('invalid source roles')
    if competitors is not None and noise['num_classes'] != competitors['num_classes']:
        raise ValueError('source calibration class mismatch')
    ys = torch.as_tensor(labels).detach().cpu().tolist()
    if len(day_ids) != len(ys) or len(condition_ids) != len(ys):
        raise ValueError('missing query source condition metadata')
    delta = torch.full((len(ys),noise['num_classes']),float('nan'),device=device,dtype=dtype or torch.float32)
    weights = torch.zeros_like(delta)
    fallbacks = {}
    for i,(y,day,condition) in enumerate(zip(ys,day_ids,condition_ids)):
        selected = competitors['selected'].get(y,()) if competitors is not None else range(noise['num_classes'])
        for b in selected:
            if b == y:
                continue
            choices = [('conditional',noise['conditional'].get((y,b,day,condition))),
                       ('pair',noise['pair'].get((y,b))),('global',noise['global_noise'])]
            choice = next(((kind,row) for kind,row in choices if row and row['delta'] is not None),None)
            if choice is None:
                fallbacks[(i,b)] = dict(reason='insufficient_global_noise_evidence',groups=0)
                continue
            kind,row = choice
            delta[i,b],weights[i,b] = row['delta'],1.0
            fallbacks[(i,b)] = dict(reason=kind,groups=row['groups'],delta=row['delta'])
    return delta.detach(),weights.detach(),dict(fallbacks=fallbacks,
        available_comparisons=int(weights.sum()),source_roles=('L_s','V') if competitors is not None else ('L_s',))
