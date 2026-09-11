"""Validate and freeze actual source perturbation schedules; never train.

Count matching preserves phase, actual selected counts and scene counts.
Exact matching additionally preserves sample-to-scene/channel assignments by
permuting only equal ordered-ID batches. No available permutation is reported
as nontrivial=False, not as an executed ordering experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import sys

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path: sys.path.insert(0,str(CODE))

try:
    from scripts.build_core90_game_replay import load_donor
except ModuleNotFoundError:
    from build_core90_game_replay import load_donor

SCENES = ('leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')


def channel_config_signature(config):
    from dataclasses import asdict
    from types import SimpleNamespace
    from cvsrffi.eval import make_sat_config
    args=SimpleNamespace(**config)
    return json.loads(json.dumps({scene:asdict(make_sat_config(scene,args)) for scene in SCENES}))


def validate_exposure_schedule(schedule, *, recipient_seed, data_contract, sample_stream=None, channel_config=None):
    if schedule.get('schema') != 'core90_exposure_schedule_v2': raise ValueError('unsupported exposure schema')
    if schedule.get('source_only') is not True or schedule.get('target_evaluated') is not False:
        raise ValueError('explicit source-only exposure required')
    if (type(recipient_seed) is not int or recipient_seed < 0 or schedule.get('recipient_seed') != recipient_seed
            or type(schedule.get('donor_seed')) is not int or schedule['donor_seed'] < 0
            or schedule['donor_seed'] == recipient_seed):
        raise ValueError('independent cross-seed donor required')
    if not isinstance(data_contract, dict) or not data_contract or schedule.get('data_contract') != data_contract:
        raise ValueError('exposure data contract mismatch')
    if channel_config is not None and schedule.get('channel_config') != channel_config:
        raise ValueError('exposure physical channel configuration mismatch')
    records = schedule.get('records')
    if not isinstance(records, list) or not records or type(schedule.get('horizon')) is not int or schedule['horizon'] != len(records):
        raise ValueError('incomplete exposure horizon')
    if sample_stream is not None and len(sample_stream) != len(records): raise ValueError('recipient stream horizon mismatch')
    counts = {'pre_E80_bn': Counter(), 'E80_plus_direct_ce': Counter()}
    previous_epoch = 0
    for step, r in enumerate(records):
        if r.get('step') != step or type(r.get('epoch')) is not int or r['epoch'] < max(1, previous_epoch):
            raise ValueError('exposure steps/epochs not complete and ordered')
        previous_epoch = r['epoch']
        ids, mask = r.get('sample_ids'), r.get('selected_mask')
        if not isinstance(ids, list) or not ids or any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
            raise ValueError('invalid opaque sample IDs')
        if not isinstance(mask, list) or len(mask) != len(ids) or any(type(v) is not bool for v in mask):
            raise ValueError('selected mask must exactly match sample IDs')
        if r.get('scenario') not in SCENES or type(r.get('channel_seed')) is not int or r['channel_seed'] < 0:
            raise ValueError('actual scene/channel seed required')
        if sample_stream is not None and list(sample_stream[step]) != ids: raise ValueError('recipient sample stream mismatch')
        counts['pre_E80_bn' if r['epoch'] < 80 else 'E80_plus_direct_ce'][r['scenario']] += sum(mask)
    result = copy.deepcopy(schedule)
    result['actual_counts'] = {phase: {scene: count[scene] for scene in SCENES} for phase, count in counts.items()}
    return result


def permute_exposure_schedule(schedule, *, seed, mode='count', ordering='random'):
    if mode not in ('count', 'exact') or ordering not in ('random', 'uniform', 'replay'):
        raise ValueError('unsupported exposure permutation')
    result = validate_exposure_schedule(schedule, recipient_seed=schedule.get('recipient_seed'), data_contract=schedule.get('data_contract'))
    groups = defaultdict(list)
    for i, row in enumerate(result['records']):
        key = (row['epoch'] >= 80, tuple(row['sample_ids']) if mode == 'exact' else len(row['sample_ids']))
        groups[key].append(i)
    rng = random.Random(seed)
    originals = copy.deepcopy(result['records'])
    for positions in groups.values():
        donors = list(positions)
        if ordering == 'random': rng.shuffle(donors)
        elif ordering == 'uniform':
            buckets = defaultdict(list)
            for i in donors: buckets[(originals[i]['scenario'], sum(originals[i]['selected_mask']))].append(i)
            # Spread each actual scene/count category over normalized ranks.
            donors = [i for _, _, i in sorted(( (j+.5)/len(bucket), str(key), i)
                      for key, bucket in buckets.items() for j,i in enumerate(bucket))]
        for i, donor in zip(positions, donors):
            for key in ('scenario', 'selected_mask', 'channel_seed'):
                result['records'][i][key] = copy.deepcopy(originals[donor][key])
            result['records'][i]['donor_step'] = originals[donor].get('donor_step', donor)
    result.update(matching_scope='actual_count_only' if mode == 'count' else 'actual_sample_assignment',
                  exact_sample_assignment=mode == 'exact', ordering=ordering, schedule_seed=seed,
                  nontrivial=any(any(r[k] != originals[i][k] for k in ('scenario','selected_mask','channel_seed'))
                                 for i,r in enumerate(result['records'])))
    checked = validate_exposure_schedule(result, recipient_seed=result['recipient_seed'], data_contract=result['data_contract'])
    if checked['actual_counts'] != schedule.get('actual_counts', validate_exposure_schedule(schedule,
            recipient_seed=schedule['recipient_seed'], data_contract=schedule['data_contract'])['actual_counts']):
        raise AssertionError('actual exposure changed during permutation')
    return checked


def build_exposure_schedules(donor_dir, recipient_seed, recipient_config, records, *, schedule_seed=0, mode='count'):
    config, completion, actions = load_donor(donor_dir, recipient_seed, version=2, recipient_config=recipient_config)
    channel_config=channel_config_signature(config)
    if channel_config!=channel_config_signature(recipient_config):
        raise ValueError('donor/recipient physical channel configuration mismatch')
    if type(config.get('game_data_order_seed')) is not int or config['game_data_order_seed'] < 0:
        raise ValueError('exposure matching requires an explicit shared data-order seed')
    if len(records) != len(actions) or any(r.get('epoch') != a.get('epoch') for r,a in zip(records,actions)):
        raise ValueError('exposure stream does not match complete donor horizon')
    base = dict(schema='core90_exposure_schedule_v2', source_only=True, target_evaluated=False,
                donor_seed=config['seed'], recipient_seed=recipient_seed, data_contract=config['game_data_contract'],
                horizon=len(records), records=records, donor_completion=completion,
                channel_config=channel_config,
                status='FROZEN_NOT_EXECUTED', recipient_data_order_seed=recipient_config.get('game_data_order_seed'))
    return {name: permute_exposure_schedule(base, seed=schedule_seed, mode=mode, ordering=name)
            for name in ('replay', 'uniform', 'random')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('donor-dir', 'recipient-config', 'exposure-jsonl', 'output-dir'): p.add_argument('--'+name, required=True)
    p.add_argument('--schedule-seed', type=int, default=0)
    p.add_argument('--mode', choices=('count','exact'), default='count')
    args = p.parse_args(argv)
    recipient = json.loads(Path(args.recipient_config).read_text(encoding='utf-8-sig'))
    records = [json.loads(l) for l in Path(args.exposure_jsonl).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    schedules = build_exposure_schedules(args.donor_dir, recipient['seed'], recipient, records,
                                        schedule_seed=args.schedule_seed, mode=args.mode)
    output = Path(args.output_dir)
    if any((output/(name+'.json')).exists() for name in schedules): raise FileExistsError('schedule exists')
    output.mkdir(parents=True, exist_ok=True)
    for name, schedule in schedules.items():
        with (output/(name+'.json')).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(schedule, stream, indent=2, allow_nan=False); stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
