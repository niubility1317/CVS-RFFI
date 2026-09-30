"""Two CPU lanes: fixed LocalRidge B0/C0 support registration diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time

from run_d92_branch_support_probe import read, write, launch

ROOT = Path(__file__).resolve().parents[1]
KS = [1, 5, 10, 20]
NEW_COUNTS = [0, 2, 5, 10, 20]
SCENARIOS = ['practical_high', 'practical_mid', 'practical_low_urban']
CHANNEL = dict(route='residual', mode='post_sync', equalization_enabled=False, fs_hz=25000000)
CONFIG_NAMES = {c: f'configs/d92_registration_diagnostic_{c}_20260930.json' for c in ('rx3', 'rx1')}
DIAGNOSTIC_CONFIG = dict(schema='d92_registration_diagnostic_v1',
    method='D92-LocalRidge-registration-diagnostic',
    base_algorithm=read(ROOT/'configs/d92_branch_local_ridge_frozen_20260929.json')['algorithm'],
    fits=['B0_old_support', 'C0_all_registered_support'],
    c_old='offline_restriction_of_fixed_C0_scores_no_refit',
    decomposition='B0_correct-C0_correct=(B0_correct-C_old_correct)+(C_old_correct-C0_correct)',
    tie_break='physical_class_id_lexicographic',
    n0='reuse_B0_state_and_scores_as_C0', k1='numerical_only_no_fit_no_holdout',
    oof='per_class_physical_id_sort_position_mod_3_then_pool_held_rows_per_parent',
    proxy='all_per_class_sorted_anchor_positions_then_parent_equal_mean',
    query_access=False, source_sample_access=False, optimizer_steps=0,
    selection='fixed_no_parameter_search', channel=CHANNEL, allowed_scenarios=SCENARIOS)
STATUS = 'REGISTRATION_DIAGNOSTIC_COMPLETE'
IDENTITY_KEYS = {'split_id', 'receiver', 'scenario', 'k', 'support_seed', 'registered_classes', 'new_count'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_selection(selection, producer_matrix):
    require(set(selection) == {'receiver_scenes', 'support_seed', 'ks', 'new_counts', 'splits'}, 'Selection schema mismatch')
    require(selection['ks'] == KS and selection['new_counts'] == NEW_COUNTS, 'Pilot K/new-count drift')
    pairs = selection['receiver_scenes']
    require(isinstance(pairs, list) and len(pairs) == 2
        and all(isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p) for p in pairs), 'Two explicit receiver/scenario pairs required')
    expected_pairs = sorted(itertools.product(producer_matrix['receivers'], producer_matrix['scenarios']))[:2]
    require([tuple(p) for p in pairs] == expected_pairs, 'Pilot receiver/scenario selection drift')
    seed = selection['support_seed']
    require(type(seed) is int and seed in producer_matrix['support_seeds'], 'Invalid selected support seed')
    require(set(producer_matrix['scenarios']) == set(SCENARIOS), 'Practical scenario contract mismatch')
    require(producer_matrix['ks'] == KS and producer_matrix['new_counts'] == NEW_COUNTS, 'Producer axes mismatch')
    rows = selection['splits']
    require(isinstance(rows, list) and len(rows) == 40, 'Exactly 40 selected splits required')
    expected = {(r, s, k, n, seed) for r, s in expected_pairs for k in KS for n in NEW_COUNTS}
    seen = set(); ids = set(); old = None
    for row in rows:
        require(set(row) == IDENTITY_KEYS, 'Selected identity schema mismatch')
        require(isinstance(row['split_id'], str) and row['split_id'] and row['split_id'] not in ids, 'Duplicate/empty selected split ID')
        require(all(type(row[k]) is int for k in ('k', 'new_count', 'support_seed')), 'Invalid selected numeric identity')
        classes = row['registered_classes']
        require(isinstance(classes, list) and classes and all(isinstance(c, str) and c for c in classes)
            and len(classes) == len(set(classes)), 'Invalid selected class mapping')
        count = row['new_count']; prior = classes[:len(classes)-count] if count else classes
        require(len(prior) == 6, 'Current frozen diagnostic requires six old classes')
        if old is None: old = prior
        require(prior == old, 'Old class mapping drift')
        cell = tuple(row[k] for k in ('receiver', 'scenario', 'k', 'new_count', 'support_seed'))
        require(cell in expected and cell not in seen, 'Missing/duplicate selected matrix cell')
        seen.add(cell); ids.add(row['split_id'])
    require(seen == expected, 'Incomplete pilot matrix')
    return old


def validate_spec(spec):
    rows = spec['rows']; probe = spec['probe']; execution = spec['execution']; permissions = spec['permissions']
    require(permissions['query_use'] == 'none; query IQ/labels/truth/scores never read'
        and all(permissions.get(k) is False for k in ('source_samples','source_per_record_features','summary_inputs','old_target_scores_for_adaptation','cross_run_result_tuning'))
        and probe['query_access'] is False and probe['reuse_support_cache'] is True, 'Support-only permissions mismatch')
    require(execution['cpu_lanes'] == 2 and execution['blas_threads_per_lane'] == 2
        and execution['export_device'] is None and execution['export_batch_size'] is None, 'Two CPU lanes / BLAS2 required')
    require(probe['algorithm'] == DIAGNOSTIC_CONFIG and probe['channel'] == CHANNEL, 'Frozen algorithm/channel mismatch')
    require(set(probe['cohorts']) == {'rx3','rx1'}, 'Both cohorts required')
    available = probe['available_model_seeds']; selected = probe['selected_model_seeds']
    require(isinstance(available, list) and len(available) >= 2 and all(type(v) is int for v in available)
        and len(available) == len(set(available)) and selected == sorted(available)[:2], 'Two smallest model seeds required')
    common = set.intersection(*(set(c['matrix']['support_seeds']) for c in probe['cohorts'].values()))
    require(bool(common), 'No common support seed')
    for name, co in probe['cohorts'].items():
        matrix = co['matrix']
        require(set(matrix) == {'receivers','scenarios','ks','new_counts','support_seeds'}, 'Producer matrix schema mismatch')
        for axis in matrix.values():
            require(isinstance(axis,list) and axis and len(axis)==len(set(axis)), 'Invalid producer matrix axis')
        validate_selection(co['selection'], matrix)
        require(co['selection']['support_seed'] == min(common), 'Smallest common support seed required')
        require(co['expected_split_count'] == len(list(itertools.product(*matrix.values())))
            and co['selected_split_count'] == 40, 'Producer/pilot split count mismatch')
        require(co['capsule_id'].startswith('residual-noeq-'), 'Residual-noeq capsule required')
        require(co['evaluation_config'] == str(PurePosixPath(spec['code']['cwd'])/CONFIG_NAMES[name]), 'Evaluation config escaped release')
    require(len(rows) == 4 and len({r['row_id'] for r in rows}) == 4, 'Exactly four unique model/cohort rows required')
    root = Path(execution['remote_run_root']).resolve(); pairs = set(); checkpoint_by_model = {}
    for row in rows:
        co = probe['cohorts'][row['cohort']]; out = Path(row['output_root']).resolve()
        require(out.parent == root and out.name == row['row_id'], 'Row output escaped run root')
        require(row['data_overrides'] == dict(capsule=co['capsule'],capsule_id=co['capsule_id'],expected_split_count=co['expected_split_count']), 'Cohort binding mismatch')
        require(row['seeds']['support'] == co['selection']['support_seed'], 'Explicit support seed mismatch')
        cache = Path(row['support_features']).resolve()
        require(cache != root and root not in cache.parents and cache not in root.parents, 'Cache/output overlap')
        capsule = Path(co['capsule']).resolve()
        require(capsule != root and root not in capsule.parents and capsule not in root.parents, 'Capsule/output overlap')
        digest = row['expected_checkpoint_sha256']
        require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'Invalid checkpoint binding')
        model = row['seeds']['model']; pairs.add((row['cohort'],model))
        require(checkpoint_by_model.setdefault(model,digest)==digest, 'Cross-cohort model checkpoint mismatch')
    require(pairs == {(co,m) for co in ('rx3','rx1') for m in selected}, 'Model/cohort coverage mismatch')
    require(probe['expected_parents']==160 and probe['expected_true_k1_parents']==40
        and probe['expected_oof_parents']==120 and probe['expected_proxy_anchors']==1400
        and probe['expected_sequence_paths']==1760 and probe['expected_head_fits']==3168, 'Pilot budget mismatch')


def command(spec, row):
    co=spec['probe']['cohorts'][row['cohort']]
    return [sys.executable,'-u',str(Path(spec['code']['cwd'])/'tools/evaluate_d92_registration_diagnostic.py'),
        '--support-features',row['support_features'],'--capsule',co['capsule'],
        '--output',str(Path(row['output_root'])/'probe'),'--config',co['evaluation_config'],
        '--expected-capsule-id',co['capsule_id'],'--expected-checkpoint-sha256',row['expected_checkpoint_sha256'],
        '--expected-model-seed',str(row['seeds']['model'])]


def verify_marker(path, spec, row):
    marker=read(path); co=spec['probe']['cohorts'][row['cohort']]
    expected=dict(status=STATUS,capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],
        model_seed=row['seeds']['model'],algorithm=DIAGNOSTIC_CONFIG,selection=co['selection'],
        episodes=40,k1_episodes=10,oof_episodes=30,proxy_anchor_count=350,sequence_paths=440,
        head_fit_count=792,optimizer_steps=0,query_rows_used=0,source_rows_used=0)
    require(all(marker.get(k)==v for k,v in expected.items()), 'Incomplete or incorrectly bound registration diagnostic')
    require(type(marker.get('factorization_count')) is int and 0<=marker['factorization_count']<=792, 'Invalid actual factorization count')
    for key in ('episodes','k1_episodes','oof_episodes','proxy_anchor_count','sequence_paths','head_fit_count','optimizer_steps','query_rows_used','source_rows_used'):
        require(type(marker.get(key)) is int, 'Invalid counter type: '+key)
    require(marker.get('producer_matrix') == co['matrix'], 'Producer matrix marker mismatch')
    return marker


def run(spec, commit, launch_fn=launch):
    validate_spec(spec)
    for co in spec['probe']['cohorts'].values():
        expected=dict(algorithm=DIAGNOSTIC_CONFIG,producer_matrix=co['matrix'],selection=co['selection'])
        require(read(co['evaluation_config'])==expected, 'Evaluator config/spec mismatch')
    root=Path(spec['execution']['remote_run_root']);root.mkdir(parents=True,exist_ok=False)
    rows=spec['rows'];state={r['row_id']:dict(status='PENDING') for r in rows};lock=threading.Lock()
    write(root/'startup.json',dict(spec=spec,commit=commit,pid=os.getpid(),argv=sys.argv,
        cpu_lanes=2,blas_threads_per_lane=2,query_access=False,source_sample_access=False,
        checkpoint_loaded=False,gpu_use=False,optimizer_steps=0,started=time.time()))
    def update(row,status,**fields):
        with lock:
            state[row['row_id']]=dict(status=status,updated=time.time(),**fields)
            write(root/'state.json',state)
        print(json.dumps(dict(row=row['row_id'],status=status,**fields)),flush=True)
    counters=('episodes','k1_episodes','oof_episodes','proxy_anchor_count','sequence_paths','head_fit_count','factorization_count')
    def work(row):
        try:
            out=Path(row['output_root']);out.mkdir(exist_ok=False);update(row,'PROBING_SUPPORT')
            launch_fn(command(spec,row),out/'probe.log',Path(spec['code']['cwd']),'probe')
            marker=verify_marker(out/'probe/probe_complete.json',spec,row)
            update(row,STATUS,**{k:marker[k] for k in counters})
        except Exception as exc:
            update(row,'FAILED',error_type=type(exc).__name__,error=str(exc))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(work,rows))
    complete=all(v['status']==STATUS for v in state.values())
    write(root/'complete.json',dict(status=STATUS if complete else 'FAILED',commit=commit,model_rows=4,
        completed_rows=sum(v['status']==STATUS for v in state.values()),
        **{k:sum(v.get(k,0) for v in state.values()) for k in counters},optimizer_steps=0,
        query_access=False,source_sample_access=False,checkpoint_loaded=False,gpu_use=False,finished=time.time()))
    if not complete:raise RuntimeError('Registration diagnostic lane failed; healthy lanes retained, no retry')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--commit',required=True)
    args=parser.parse_args();run(read(args.spec),args.commit)
