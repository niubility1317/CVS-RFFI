"""Paired-seed descriptive effects and explicit scenario/cost completeness."""
import math
import random

SCENARIOS = ('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')


def paired_joint_effect(rows, metric='accuracy', *, scenarios=SCENARIOS,
                        bootstrap_samples=2000, seed=1337, confidence=.95,
                        joint_variant='U4_additive'):
    """Rows: {variant,seed,metrics:{scenario:{metric:value}},costs:{...}}.

    Bootstrap independent paired training seeds, never query samples. Missing
    variants/scenarios are reported, not imputed. One seed has no valid CI.
    Select the actual registered U4_additive or U4_bilinear row explicitly;
    these are distinct capacity comparisons and must never be pooled.
    """
    if bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError('invalid bootstrap parameters')
    if joint_variant not in ('U4_additive','U4_bilinear'):
        raise ValueError('joint_variant must be a registered U4_additive or U4_bilinear row')
    indexed = {}
    for row in rows:
        key = (str(row['variant']),row['seed'])
        if key in indexed:
            raise ValueError('duplicate variant/seed row')
        indexed[key] = row
    variants = ('U1','U2','U3',joint_variant)
    seeds = sorted({s for v,s in indexed if v in variants},key=str)
    paired = [s for s in seeds if all((v,s) in indexed for v in variants)]
    rng = random.Random(seed)
    effects = {}
    for scenario in scenarios:
        values,missing = [],[]
        for s in paired:
            numbers = []
            for v in variants:
                entry = indexed[v,s].get('metrics',{}).get(scenario,{})
                value = entry.get(metric) if isinstance(entry,dict) else entry
                numbers.append(value)
            if any(value is None or not math.isfinite(float(value)) for value in numbers):
                missing.append(s)
                continue
            u1,u2,u3,u4 = map(float,numbers)
            values.append((s,u4-u2-u3+u1))
        estimate = sum(v for _,v in values)/len(values) if values else None
        interval = None
        if len(values)>1:
            samples = sorted(sum(rng.choice(values)[1] for _ in values)/len(values) for _ in range(bootstrap_samples))
            def quantile(q):
                pos = q*(len(samples)-1)
                lo,hi = math.floor(pos),math.ceil(pos)
                return samples[lo]+(pos-lo)*(samples[hi]-samples[lo])
            interval = [quantile((1-confidence)/2),quantile((1+confidence)/2)]
        effects[scenario] = dict(mean=estimate,confidence_interval=interval,paired_seed_count=len(values),
                                 per_seed=[dict(seed=s,effect=v) for s,v in values],missing_metric_seeds=missing,
                                 interpretation='paired_seed_estimate' if len(values)>1 else 'descriptive_only')
    costs = []
    required_costs = ('training_seconds','peak_memory_bytes','optimizer_steps','physical_record_exposures')
    for (variant,s),row in indexed.items():
        supplied = row.get('costs',{})
        costs.append(dict(variant=variant,seed=s,values=supplied,
                          missing=[key for key in required_costs if supplied.get(key) is None]))
    return dict(metric=metric,joint_variant=joint_variant,effect_formula=f'{joint_variant}-U2-U3+U1',scenarios=effects,
                unpaired_seeds=[s for s in seeds if s not in paired],costs=costs,
                all_costs_complete=bool(costs) and all(not row['missing'] for row in costs),
                complete_four_scenarios=len(scenarios)==4 and all(e['paired_seed_count']==len(paired) and len(paired)>0 for e in effects.values()),
                claim='No automatic promotion; assess identity performance, costs and paired uncertainty.')


def paired_joint_effects_by_variant(rows, metric='accuracy', **kwargs):
    """Return separate effects for both actual registered U4 capacities."""
    rows = list(rows)
    return {variant: paired_joint_effect(rows,metric,joint_variant=variant,**kwargs)
            for variant in ('U4_additive','U4_bilinear')}
