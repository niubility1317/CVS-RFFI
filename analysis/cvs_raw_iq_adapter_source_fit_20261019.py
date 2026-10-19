"""Frozen raw-IQ adapter source readout; reuse identical TC/TS observations.

Four own FINAL30 models, one real-checkpoint smoke, no training or target path.
The original fixed L180 panel and full-physics named streams remain unchanged.
"""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

RUN = '20261019-diagnostic-F-rawIQAdapter-sourceFit-e30-s2026092701-m6-r01'
REFERENCE_RUN = '20261019-phase1-F-rawIQ-dualTeacher-e20plus10-s2026092701-m4-r01'
REFERENCE_COMMIT = '6ea2f71f6654f4ceeadd460905a9d719b5dc1b89'
CACHE_RUN = '20261019-diagnostic-F-dualAdapter-sourceFit-e30-s2026092701-m8-r01'
ARMS = ('NOAS_ADAPTER', 'S1', 'S2', 'S3')
VIEWS = ('raw', 'observed_last', 'fresh_mid_d0', 'fresh_mid_d1', 'fresh_low_d0', 'fresh_low_d1')
SEED = 2026092701
MAX_NN = 4321
KEYS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
HEAD_KEY = 'id_backbone.encoder.core.id_backbone.cls_head.weight'
ORIGIN_KEYS = ('local_step', 'cumulative_step', 'local_epoch', 'epoch', 'slot',
               'pid', 'physical_id', 'namespace', 'scene', 'step_scene_counts_verified')
RECORD_KEYS = ('view', 'panel_index', 'physical_id', 'sample_id', 'TX', 'RX', 'day',
               'scene', 'namespace', 'session_id', 'seed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap(runtime):
    rt = read(runtime)
    require(rt.get('run_id') == RUN, 'Fixed diagnostic run required')
    ref = Path(rt['reference_release'])
    require((ref / 'release_commit.txt').read_text().strip() == REFERENCE_COMMIT,
            'Pinned completed raw-IQ release differs')
    module_at('_raw_sourcefit_parent', ref / 'experiments/cvs_raw_iq_adapter30/runner.py').bootstrap()
    return rt


def checked_panel(selected, base, roles, opaque, classes):
    require(len(selected) == 180 and len({r['physical_id'] for r in selected}) == 180
            and Counter((r['TX'], r['RX'], r['day']) for r in selected) == Counter(
                {(t, r, d): 2 for t in range(6) for r in (1, 3, 4, 6, 8) for d in (1, 2, 3)}),
            'Reuse original balanced physical L180 without reselection')
    require(list(base.tx_list) == list(classes) and base.transform is None
            and base.normalize and not base.center, 'Original normalized raw preprocessing required')
    legal = set(roles['L_s'])
    for row in selected:
        pid = row['pid']
        require(len(pid) == 5 and all(type(x) is int for x in pid)
                and 0 <= pid[4] < len(base.index), 'Invalid physical L index before IQ access')
        item = base.index[pid[4]]
        require(tuple(pid) == (item.eq_i, item.rx_i, item.day_i, item.sig_i, pid[4])
                and opaque(item) == row['physical_id'] and row['physical_id'] in legal
                and (item.tx_i, item.rx_i, item.day_i) == (row['TX'], row['RX'], row['day'])
                and item.eq_i == 1 and item.rx_i in (1, 3, 4, 6, 8) and item.day_i in (1, 2, 3),
                'Only original visible-label source L physical IDs allowed')


def checked_reuse(panel, complete, readouts, arrays, selected, origins, channel):
    """Check prediction/input identity from existing source artifacts, no NN/hash."""
    require(complete['status'] == 'SOURCE_ONLY_ZERO_UPDATE_COMPLETE' and not complete['target_access'],
            'Prior source readout must be complete')
    require(panel['selected'] == selected and panel['views'] == list(VIEWS)
            and panel['source_raw_reads'] == 180 and panel['target_access'] is False,
            'Cached physical panel/order differs')
    require(len(panel['origins']) == len(origins) == 180, 'Exact 180 last origins required')
    for old, new in zip(panel['origins'], origins):
        require(all(old[k] == new[k] for k in ORIGIN_KEYS), 'Cached last As origin differs')
    previous = panel['channel']
    for key in ('views', 'shape', 'training_scenes', 'channel_seeds', 'fresh_namespaces',
                'sessions', 'native_normalizations', 'extra_RMS', 'output_normalization',
                'generated_packets', 'observed_last_packets', 'fresh_packets', 'raw_packets'):
        require(previous[key] == channel[key], 'Cached channel input policy differs: ' + key)
    require(previous['channel_audit']['configurations'] == channel['channel_audit']['configurations'],
            'Cached physical configurations differ')
    require(len(previous['records']) == len(channel['records']) == 900, 'All 900 observation identities required')
    for old, new in zip(previous['records'], channel['records']):
        require(all(old[k] == new[k] for k in RECORD_KEYS), 'Cached named observation differs')
    expected = dict(TX=[r['TX'] for r in selected], ids=[r['physical_id'] for r in selected],
                    RX=[r['RX'] for r in selected], day=[r['day'] for r in selected])
    for key, values in expected.items():
        require(np.array_equal(np.asarray(arrays[key]), np.asarray(values)), 'Cached prediction order differs: ' + key)
    for name, origin in (('TC', 'fixed E20'), ('TS', 'fixed E200')):
        meta = readouts[name]
        logits = np.asarray(arrays[name + '_logits'])
        require(meta['model'] == name and meta['origin'] == origin and meta['views'] == list(VIEWS)
                and meta['n_per_view'] == 180 and meta['NN_packets'] == 1080
                and meta['registered_state_unchanged'] is True and meta['target_access'] is False
                and meta['optimizer_updates'] == 0, 'Cached frozen teacher identity differs')
        require(logits.shape == (180, 6, 6) and np.isfinite(logits).all(), 'Finite cached logits180x6x6 required')
        correct = (logits.argmax(-1) == np.asarray(expected['TX'])[:, None]).sum(0).tolist()
        require(correct == meta['correct'], 'Cached source logits disagree with original readout')
    joint = np.asarray(arrays['TC_joint'])
    require(joint.shape == (180, 6, 160) and np.isfinite(joint).all(), 'Finite cached TC joint180x6x160 required')
    return dict(status='REUSED_IDENTICAL_NAMED_SOURCE_OBSERVATIONS', run_id=CACHE_RUN,
                models=['TC', 'TS'], new_NN_packets=0, physical_IDs=180, matched_origins=180,
                matched_generated_observations=900, raw_preprocessing='original normalized Dataset;transformNone;centerFalse',
                historical_IQ_byte_comparison=False, target_access=False)


def same_sequences(paths):
    """All four source-only logged L/U streams; no labels or IQ are opened."""
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(p, 'rt', encoding='utf8')) for p in paths]
        n = 0
        while True:
            lines = [s.readline() for s in streams]
            if not any(lines):
                break
            require(all(lines), 'Truncated raw-row L/U sequence')
            records = [json.loads(line) for line in lines]
            require(all(r == records[0] for r in records[1:]), 'Raw rows used different source L/U sequences')
            n += 1
    require(n == 4440, 'Complete 2220 L plus2220 U batches required')
    return dict(rows=4, matching_logged_batches_per_row=n, L_batches=2220, U_batches=2220, IQ_reads=0, NN=0)


def run(runtime, output):
    rt = bootstrap(runtime)
    import torch
    from experiments.cvs_raw_iq_adapter30 import design as d, model as dm
    from experiments.cvs_F_nontrue20 import design as pd
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_fullsource_concat20.channel import Practical, serial
    from experiments.cvs_fullsource_concat20 import design as cd
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    from cvsrffi.game_tracking.data import opaque_id
    helper = module_at('_rawfit_panel', Path(__file__).with_name('cvs_dual_teacher_adapter_source_panel_20261019.py'))
    metrics = module_at('_rawfit_metrics', Path(__file__).with_name('cvs_dual_teacher_adapter_source_metrics_20261019.py'))
    require(os.geteuid() != 0 and d.RUN == REFERENCE_RUN and Path(d.PROJECT) == Path(rt['project']),
            'Fixed raw-IQ ordinary-account source runtime required')
    out = Path(output)
    require(out == Path(rt['project']) / 'runs' / RUN, 'Registered immutable diagnostic output required')
    cache = Path(rt.get('frozen_reference_root', str(Path(d.PROJECT) / 'runs' / CACHE_RUN)))
    require(cache == Path(d.PROJECT) / 'runs' / CACHE_RUN, 'Only original source-only teacher cache allowed')
    oldpanel, oldcomplete = read(cache / 'source_panel.json'), read(cache / 'complete.json')
    cached_readouts = {n: read(cache / 'models' / n / 'readout.json') for n in ('TC', 'TS')}
    with np.load(cache / 'predictions.npz', allow_pickle=False) as saved:
        cached = {k: saved[k].copy() for k in ('TC_logits', 'TC_joint', 'TS_logits', 'TX', 'ids', 'RX', 'day')}
    selected = oldpanel['selected']
    rows = {a: d.config(next(r for r in d.rows() if r['arm'] == a)) for a in ARMS}
    sources = {a: Path(rows[a]['output_root']) for a in ARMS}
    roles = d.read(d.SOURCE)['role_ids']
    require(set(roles) == {'L_s', 'U_s', 'V'} and all(len(roles[k]) == n for k, n in
            [('L_s', 6300), ('U_s', 56700), ('V', 27000)]), 'Original source roles required')
    sequence_audit = same_sequences([sources[a] / 'training_sequence.jsonl.gz' for a in ARMS])
    origins, origin_audit = helper.recover_origins(sources['S3'] / 'training_sequence.jsonl.gz',
                                                 sources['S3'] / 'mechanism_steps.jsonl.gz', selected)
    # Close the native Dataset scope before constructing ANY checkpoint model.
    args = SimpleNamespace(**read(d.PARENT_SOURCE / 'actual_native_args.json'))
    require(args.a1_source_screen_only and not args.baseline_ckpt and not args.teacher_ckpt
            and not args.use_concat_sat_channel_aug and args.wisig_out_len == 256,
            'Original source-only Dataset builder required')
    with installed(pd.config(next(r for r in pd.rows() if r['arm'] == 'B0'))) as native:
        ctx = native._build_ssdg_wisig_data(args, torch.device('cpu'))
    require(not ctx['named_test_loaders'] and role_ids_from_native(ctx) == roles, 'No target loaders; unchanged source roles')
    base = ctx['train_loader'].dataset.base
    checked_panel(selected, base, roles, opaque_id, d.CLASSES)
    raw = []
    for row in selected:
        x, _, _, meta = base[row['pid'][4]]
        require(tuple(int(meta[k]) for k in KEYS) == tuple(row['pid']) and x.shape == (2, 256)
                and x.dtype == torch.float32 and bool(torch.isfinite(x).all()), 'Actual raw L preprocessing differs')
        raw.append(x)
    raw = torch.stack(raw)
    x, channel = helper.build_views(raw, selected, origins, cd, Practical, serial)
    reuse = checked_reuse(oldpanel, oldcomplete, cached_readouts, cached, selected, origins, channel)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'execution.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, started_at=time.time(),
                                      target_access=False, optimizer_updates=0, EMA_updates=0, head_fit_updates=0))
    torch.set_num_threads(2)
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    require(gpu in (4, 6, 7), 'Registered GPU4/6/7 required')
    with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
        state = capacity()[gpu]
        require(len(state['pids'] | {os.getpid()}) <= 2 and state['free_mb'] >= 12000, 'Total-owner GPU cap2/free12GB')
        torch.cuda.init()
        require(len(capacity()[gpu]['pids'] | {os.getpid()}) <= 2, 'GPU capacity changed')
    labels = cached['TX']
    nn = 0
    head_errors, state_checks, arrays = {}, {}, dict(cached)
    def asnp(t):
        return np.asarray(t.detach().cpu().tolist(), dtype=np.float64)
    parent = dm.parent_payload(rows['S3'])
    parent_head = parent['model'][HEAD_KEY].detach().cpu().clone()
    require(tuple(parent_head.shape) == (6, 160), 'Actual original six-class parent head required')
    del parent
    results = dict(TC=dict(logits=cached['TC_logits'], joint=cached['TC_joint'], head=asnp(parent_head), scale=30.,
                           origin='reused fixed B0 FINAL20; identical source observations'))
    def forward(m, value, name):
        nonlocal nn
        zz, jj = [], []
        with torch.no_grad():
            for start in range(0, len(value), 128):
                xb = value[start:start + 128].to('cuda:0')
                val = m.forward_identity_only(xb)
                z, j = val['tx_logits'], val['id_feat_joint']
                nn += len(xb)
                require(z.shape == (len(xb), 6) and j.shape == (len(xb), 160)
                        and bool(torch.isfinite(z).all()) and bool(torch.isfinite(j).all()), 'Actual finite default logits/joint160 required')
                error = float((dm.classifier(m).classify(j) - z).abs().max())
                require(error <= 2e-5, 'Actual joint/default head forward disagrees')
                head_errors[name] = max(head_errors.get(name, 0.), error)
                zz.append(asnp(z)); jj.append(asnp(j))
        return np.concatenate(zz), np.concatenate(jj)
    with numerical_context(d.FULL_FP32_POLICY):
        for name in ARMS:
            m = dm.own_model(rows[name], 'cuda:0')
            require(not m.training and all(not p.requires_grad and p.grad is None for p in m.parameters()),
                    'Readout models must remain eval/stopgrad')
            before = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            h = dm.classifier(m)
            require(float(h.scale) == 30. and torch.equal(h.weight.detach().cpu(), parent_head),
                    'Exactly the frozen original B0 head required for margin comparison')
            if name == ARMS[0]:
                forward(m, raw[:1], name)
                write(out / 'source_smoke.json', dict(status='PASS', source_packets=1, checkpoint=name + ' own FINAL30',
                                                      target_access=False, continue_immediately=True))
                print('SOURCE_SMOKE_PASS; continuing frozen source readout', flush=True)
            logits, joint = forward(m, x.reshape(1080, 2, 256), name)
            logits, joint = logits.reshape(180, 6, 6), joint.reshape(180, 6, 160)
            state_checks[name] = all(torch.equal(before[k], v.detach().cpu()) for k, v in m.state_dict().items())
            require(state_checks[name] and all(p.grad is None for p in m.parameters()), 'Frozen registered state changed')
            results[name] = dict(logits=logits, joint=joint, head=asnp(h.weight), scale=30., origin='own fixed FINAL30;parent20+segment10')
            arrays[name + '_logits'], arrays[name + '_joint'] = logits, joint
            write(out / (name + '_readout.json'), dict(model=name, views=list(VIEWS), origin=results[name]['origin'],
                  NN_packets=1080, smoke_packets=1 if name == ARMS[0] else 0,
                  correct=(logits.argmax(-1) == labels[:, None]).sum(0).tolist(), n_per_view=180,
                  registered_state_unchanged=True, target_access=False, optimizer_updates=0))
            print(json.dumps(dict(model=name, NN_packets=nn, registered_state_unchanged=True)), flush=True)
            del m, before
            torch.cuda.empty_cache()
    require(nn == MAX_NN, 'Exact four-model plus first checkpoint smoke budget differs')
    results['TS'] = dict(logits=cached['TS_logits'], joint=None, head=None, scale=5., origin='reused legal CSIL fixed E200;identical observations')
    result = metrics.analyze(results, labels, list(VIEWS))
    result.update(run_id=RUN, status='SOURCE_ONLY_ZERO_UPDATE_COMPLETE', model_seed_count=1,
                  NN_packets=nn, generated_channel_packets=900, source_raw_reads=180, new_V_NN=0,
                  optimizer_updates=0, EMA_updates=0, head_fit_updates=0, target_access=False,
                  teacher_prediction_reuse=reuse, frozen_reference_root=str(cache),
                  registered_state_unchanged=state_checks, head_forward_max_abs_error=head_errors,
                  sequence_audit=sequence_audit, parent_epochs=20, segment_epochs=10, cumulative_epochs=30,
                  dropout_only_repeated=False, INIT_ZERO_repeated=False,
                  limitations=['Fixed180 seen L, not unseen-RX evidence; no V NN',
                    'observed_last uses deterministic named-stream reconstruction, not historical IQ byte comparison',
                    'NoAs student did not train on As; training As were teacher shadows',
                    'Mixed observed scenes versus all-L fixed-scene fresh views is not a matched channel effect',
                    'Compare seen/new only on same-pid observed-mid/low subsets, with actual subgroup n',
                    'Two channel draws are repeated observations, not model seeds; n1 SD is null',
                    'Frozen output/feature tension does not prove parameter-gradient conflict',
                    'Raw4256 recipe remains closed; this diagnostic does not select a new training recipe'])
    write(out / 'source_panel.json', dict(selected=selected, origins=origins, origin_audit=origin_audit,
          channel=channel, views=list(VIEWS), source_raw_reads=180, target_access=False))
    np.savez_compressed(out / 'predictions.npz', **arrays)
    write(out / 'analysis.json', result)
    write(out / 'complete.json', dict(status=result['status'], NN_packets=nn, reused_teacher_NN=0,
                                      target_access=False, finished_at=time.time()))
    print(json.dumps(dict(status=result['status'], NN_packets=nn, reused_teacher_NN=0)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.runtime, args.output)


if __name__ == '__main__':
    main()
