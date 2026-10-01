"""Close the complete GroupBarrier support artifact set before diagnostic joins.

Only the explicitly supplied new run directory is read. No cache, packet,
capsule, source sample, query, external truth, fitter, or scorer is opened.
All accuracy values describe support OOF/proxy, never query performance.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'code')]
from run_d92_group_barrier_joint_probe import validate_spec, evaluator_config, KS, NEW_COUNTS
from evaluate_d92_group_barrier_joint_probe import COUNTERS, PEAK_COUNTERS, METRICS, scalars as _source_scalar_projection

SCHEMA = 'd92_group_barrier_joint_local_ridge_v1'
METHOD = 'D92-GroupBarrierJointLocalRidge-v1'
STATUS = 'GROUP_BARRIER_JOINT_PROBE_COMPLETE'
SCOPE = 'SUPPORT_ONLY_GROUP_BARRIER_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION'
NATIVE_A_SCOPE = 'CURRENT_LEGAL_OLD_HELD_ONLY_ORIGINAL_SIX_CLASS_COMPETITION'
ANALYSIS_SCHEMA = 'd92_group_barrier_joint_probe_analysis_v1'
PATHS = ('R0', 'R_GROUP_BARRIER_seq')
IDENTITY = ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'new_count')
METRIC_NAMES = ('A_old_accuracy', 'B_old_accuracy', 'C_old_accuracy', 'C_new_accuracy',
    'adaptation_gain_B_minus_A', 'support_adaptation_B_minus_B0', 'total_old_accuracy_drop',
    'C_abs_new_old_gap', 'C_h', 'C_old_minus_R0', 'C_new_minus_R0',
    'old_winner_changed_fraction', 'old_displaced_by_new_fraction')
REQUIRED_ARTIFACTS = ('startup.json', 'probe_complete.json', 'state_manifest.json',
    'fit_trace.jsonl', 'compact.jsonl', 'compact.csv', 'fit_stages.jsonl', 'fit_stages.csv',
    'training_events.jsonl', 'training_events_compact.jsonl', 'training_events_compact.csv',
    'fixed_predictions.jsonl', 'training.log')
MAX_STATE_COUNTERS = frozenset(('resident_numeric_state_bytes','deployment_numeric_state_bytes',
    'persistent_state_bytes','forward_cache_bytes','peak_rss_bytes','peak_process_rss_bytes','peak_gpu_memory_bytes'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))
                for line in stream if line.strip()]


def safe_path(root, relative):
    value = PurePosixPath(relative)
    require(isinstance(relative, str) and relative and not value.is_absolute()
            and '..' not in value.parts and '\\' not in relative, 'Unsafe artifact path')
    path = root.joinpath(*value.parts)
    require(path.resolve().is_relative_to(root.resolve()), 'Artifact leaves run directory')
    require(not path.is_symlink(), 'Artifact symlink is not a closed local file')
    return path


def _refs(value):
    if isinstance(value, dict):
        if 'path' in value and 'arrays' in value and 'namespace' in value:
            yield value
        for item in value.values():
            yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


class GroupStateResolver:
    """Bind existing refs and NPZ coordinates, with no invented receipt chain."""
    def __init__(self, directory, marker, *, verify_numeric_states=True):
        self.root = Path(directory)
        manifest = read_json(self.root/'artifact_manifest.json')
        require(manifest.get('schema') == 'd92_group_barrier_joint_artifacts_v1'
                and manifest.get('method') == METHOD and manifest.get('status') == STATUS,
                'Wrong Group artifact manifest')
        items = manifest['files']; paths = [x['path'] for x in items]
        require(len(paths) == len(set(paths)), 'Duplicate artifact inventory')
        actual = {x.relative_to(self.root).as_posix() for x in self.root.rglob('*')
                  if x.is_file() and x.name != 'artifact_manifest.json'}
        require(set(paths) == actual and set(REQUIRED_ARTIFACTS) <= actual,
                'Artifact inventory missing, extra, or incomplete')
        require('probe_failed.json' not in actual, 'Partial failed run cannot become complete')
        for item in items:
            path = safe_path(self.root, item['path'])
            require(path.is_file() and path.stat().st_size == item['file_bytes'], 'Artifact byte inventory differs')
        state = read_json(self.root/'state_manifest.json')
        require(state.get('schema') == 'd92_group_barrier_joint_state_archive_v1'
                and state.get('method') == METHOD and state.get('status') == 'COMPLETE', 'State archive is not complete Group schema')
        require(state.get('prediction_formula') == 'frozen_actual_B_old_conditional_plus_new_only_Ridge_and_Bernoulli_barrier_gate', 'Wrong Group prediction formula')
        files = state['files']; self.entries = {x['path']: x for x in files}
        require(len(files) == len(self.entries) == state['file_count'] == marker['state_archive_file_count'], 'State archive count differs')
        require(set(self.entries) == {x for x in actual if x.startswith('state_arrays/')}, 'State NPZ inventory differs')
        require(sum(x['file_bytes'] for x in files) == state['total_file_bytes'] == marker['state_archive_file_bytes'], 'State file byte sum differs')
        require(sum(a['nbytes'] for x in files for a in x['arrays'].values()) == state['numeric_array_bytes'] == marker['state_archive_numeric_bytes'], 'State numeric byte sum differs')
        self.used = set(); self.verify = verify_numeric_states; self.cache = {}; self.gate_readbacks=[]
        for value in files:
            require(not value.get('failed_numeric_state') and all(a.get('all_finite') is True and a.get('nonfinite_count') == 0 for a in value['arrays'].values()), 'Failed/nonfinite state in complete archive')
            if self.verify:
                item=gate_diagnostic(self.load(value),state_path=value['path'])
                if item: self.gate_readbacks.append(item)

    def bind(self, ref, *, allow_scalar_projection=False):
        require(isinstance(ref, dict) and ref.get('path') in self.entries, 'Missing state reference')
        actual = self.entries[ref['path']]
        require(isinstance(ref.get('arrays'),dict) and set(ref['arrays']) == set(actual['arrays']), 'State array names differ')
        for key in ('key', 'namespace', 'file_bytes'):
            require(ref.get(key) == actual.get(key), 'State reference identity differs: '+key)
        # The callback returns the canonical StateArchive descriptor. The
        # affine/Group success Recorder replaces only arrays metadata with its
        # shape/dtype/nbytes view. Both complete source-defined forms are known.
        basic=dict(actual,arrays={name:{key:meta[key] for key in ('shape','dtype','nbytes')}
                                  for name,meta in actual['arrays'].items()})
        complete=all(isinstance(meta,dict) and all(key in meta for key in ('shape','dtype','nbytes'))
                     for meta in ref['arrays'].values())
        if complete:
            for name, meta in ref['arrays'].items():
                require(all(meta[key] == actual['arrays'][name][key] for key in ('shape','dtype','nbytes')), 'State coordinates differ')
            require(ref==actual or ref==basic, 'Unknown or altered complete state reference descriptor')
        else:
            require(allow_scalar_projection, 'State coordinates missing from full-stream reference')
            # compact_event first recursively applies the production scalars
            # function. Nested audit/preparation refs can lose list-valued
            # shapes; this is a log projection, never a new numerical state.
            # Match the whole source-defined projection, including summaries,
            # identity, bytes and archive time; mixed/unknown forms fail.
            require(ref==_source_scalar_projection(actual) or ref==_source_scalar_projection(basic),
                    'Unknown or altered scalar state reference projection')
        self.used.add(ref['path'])
        return actual

    def load(self, ref):
        value = self.entries[ref['path']]
        if value['path'] not in self.cache:
            with np.load(safe_path(self.root, value['path']), allow_pickle=False) as archive:
                require(set(archive.files) == set(value['arrays']), 'NPZ names differ')
                arrays = {name: archive[name].copy() for name in archive.files}
            for name, array in arrays.items():
                meta = value['arrays'][name]
                require(array.dtype.kind in 'fbiu' and np.isfinite(array).all()
                        and list(array.shape) == meta['shape'] and str(array.dtype) == meta['dtype']
                        and array.nbytes == meta['nbytes'], 'NPZ numeric coordinates differ')
                array.setflags(write=False)
            # Retaining every archived state would turn an analysis into an
            # unbounded resident-memory workload. Keep at most one NPZ.
            self.cache = {value['path']:arrays}
        return self.cache[value['path']]

    def close_references(self, records, *, scalar_projected_records=None):
        """Full trace/events stay complete; only fit_stages supplies projections."""
        for ref in _refs(records):
            self.bind(ref)
        for ref in _refs(scalar_projected_records):
            self.bind(ref,allow_scalar_projection=True)
        require(self.used == set(self.entries), 'Unreferenced state archive artifact')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _same_number(a, b):
    return a == b or (type(a) in (int, float) and type(b) in (int, float)
                     and math.isclose(a, b, rel_tol=3e-13, abs_tol=3e-13))


def aggregate_costs(records, keys=COUNTERS):
    """Counts/seconds SUM; measured buffer peaks MAX. Missing is never zero."""
    sums, maxima, missing = {}, {}, {}
    for key in keys:
        values = [record[key] for record in records if key in record]
        require(all(_number(value) for value in values), 'Invalid measured cost '+key)
        missing[key] = len(records)-len(values)
        is_max=key in PEAK_COUNTERS or key in MAX_STATE_COUNTERS
        destination = maxima if is_max else sums
        destination[key] = None if len(values) != len(records) else (max(values, default=0) if is_max else sum(values))
    return dict(sum=sums, max=maxima, unavailable_record_counts={k: v for k, v in missing.items() if v},
                scope='actual_present_counters; missing is N/A; trial/failed-trial work remains in stage ledgers')


def _predictions(scores, classes):
    return [min(classes[j] for j in np.flatnonzero(row == row.max())) for row in scores]


def _accuracy(preds, ids, labels):
    return None if not ids else sum(preds[i] == labels[pid] for i, pid in enumerate(ids))/len(ids)


def _mean(values):
    return None if not values or any(x is None for x in values) else sum(values)/len(values)


def _difference(a, b):
    return None if a is None or b is None else a-b


def _matrix(value, n, q):
    array = np.asarray(value, dtype=np.float64)
    if n == 0 and not value:
        array = np.empty((0, q))
    require(array.shape == (n, q) and np.isfinite(array).all(), 'Fixed score shape/finite mismatch')
    return array


def assess_entry(entry, old, classes, new_count, fixed, resolver=None):
    """Join only stored legal held support labels after fixed predictions close."""
    bids, cids, labels = entry['b_ids'], entry['c_ids'], entry['held_labels']
    require(len(bids) == len(set(bids)) and len(cids) == len(set(cids))
            and set(bids) <= set(cids) and set(labels) == set(cids), 'Held physical IDs differ')
    require(entry['b_classes'] == old and entry['c_classes'] == classes, 'Held registry differs')
    require(set(bids) == {pid for pid in cids if labels[pid] in old}
            and set(labels.values()) <= set(classes), 'Old held physical role differs')
    require(set(bids).isdisjoint(entry['b_training_ids']) and set(cids).isdisjoint(entry['c_training_ids'])
            and set(entry['b_training_ids']) <= set(entry['c_training_ids']), 'Train/held physical overlap')
    require(len(entry['b_training_ids'])==6*entry['train_k']
            and len(entry['c_training_ids'])==len(classes)*entry['train_k']
            and len(bids)==6*entry['held_k'] and len(cids)==len(classes)*entry['held_k']
            and all(sum(label==name for label in labels.values())==entry['held_k'] for name in classes), 'Balanced physical class train/held counts differ')
    require(set(entry['paths']) == set(PATHS), 'Both independent paths are required')
    if not cids:
        require(entry['prediction_status'] == 'NO_HELD_PREDICTIONS' and not bids and not fixed, 'K1 has predictions')
        require(all(x is None for path in entry['paths'].values() for x in path['metrics'].values()), 'No-held accuracy must be N/A')
        return {name: dict.fromkeys(METRIC_NAMES) for name in PATHS}, {}
    require(entry['prediction_status'] == 'FIXED_BEFORE_SUPPORT_TRUTH_JOIN', 'Predictions were not fixed before support join')
    wanted = {name+'_'+side for name in PATHS for side in ('B', 'C')}
    if entry.get('ground_A') is not None:
        wanted.add('A')
    require(set(fixed) == wanted, 'Fixed prediction streams incomplete or duplicated')
    require(entry.get('stages'), 'Actual baseline stage binding unavailable')
    expected_binding={key:entry['stages'][0].get(key) for key in ('run_id','row_id','split_id')}
    require(all(isinstance(value,str) and value for value in expected_binding.values()), 'Actual baseline parent binding unavailable')
    for stream in fixed.values():
        require(all(stream.get(key)==value for key,value in expected_binding.items()), 'Fixed prediction stream parent binding differs')
        require(_fixed_outer_coords(stream)==_coords(entry)
                and stream.get('parent_k')==entry['parent_k'] and stream.get('train_k')==entry['train_k'],
                'Fixed prediction stream outer path coordinates differ')
    values = {}; raw = {}
    archived = None
    if resolver:
        resolver.bind(entry['fixed_prediction_state_ref']); archived = resolver.load(entry['fixed_prediction_state_ref'])
    for name in PATHS:
        path = entry['paths'][name]
        bs = _matrix(path['b_scores'], len(bids), len(old)); cs = _matrix(path['c_scores'], len(cids), len(classes))
        bp = _predictions(bs, old)
        cp = ([path['c_predictions'][pid] for pid in cids] if name == PATHS[1] else _predictions(cs, classes))
        require(all(p in classes for p in cp), 'Prediction outside registry')
        for side, ids, registry, scores, preds in (('B', bids, old, bs, bp), ('C', cids, classes, cs, cp)):
            stream = fixed[name+'_'+side]
            require(stream.get('status') == 'FIXED_BEFORE_SUPPORT_TRUTH_JOIN' and 'held_labels' not in stream
                    and stream['physical_ids'] == ids and stream['classes'] == registry
                    and stream['predictions'] == preds and np.array_equal(_matrix(stream['scores'],len(ids),len(registry)), scores), 'Fixed stream differs from paired trace')
            if archived is not None:
                require(np.array_equal(archived[name+'_'+side.lower()], scores), 'Fixed NPZ scores differ')
        cmap = dict(zip(cids, cp)); cold = [cmap[pid] for pid in bids]
        oldcols=[classes.index(label) for label in old]
        cindex={pid:i for i,pid in enumerate(cids)}
        op=bp if name==PATHS[1] else _predictions(cs[[cindex[pid] for pid in bids]][:,oldcols],old)
        nids = [pid for pid in cids if pid not in set(bids)]; npreds = [cmap[pid] for pid in nids]
        if name == PATHS[1]:
            # Score rounding may collapse old-common-offset differences. The
            # saved public structural argmax preserves the actual B old winner.
            require(all(cmap[pid] == bp[i] for i,pid in enumerate(bids) if cmap[pid] in old), 'Group C changed frozen old conditional winner')
        if new_count == 0:
            require(entry['c_reuses_b0'] is True and entry['c_reuses_b_candidates'] is True
                    and cids == bids and np.array_equal(cs,bs) and cp == bp, 'new0 must exactly reuse actual path B')
        bacc = _accuracy(bp,bids,labels); oacc = _accuracy(cold,bids,labels); nacc = _accuracy(npreds,nids,labels)
        aacc = None
        ground = entry.get('ground_A')
        if ground is not None:
            require(ground['physical_ids'] == bids and set(ground['classes']) == set(old)
                    and ground.get('score_dtype') == 'float32', 'Ground A is not the same physical old held')
            av = _matrix(ground['scores'],len(bids),len(old)).astype(np.float32)
            ap = [ground['classes'][int(j)] for j in np.argmax(av,axis=1)]
            require(ground['predictions'] == ap and fixed['A']['predictions'] == ap
                    and fixed['A']['physical_ids'] == bids and fixed['A']['classes'] == ground['classes']
                    and np.array_equal(_matrix(fixed['A']['scores'],len(bids),len(old)),av), 'Native A fixed stream differs')
            if resolver:
                resolver.bind(ground['scores_state_ref']); require(np.array_equal(resolver.load(ground['scores_state_ref'])['scores'],av), 'Native A NPZ differs')
            aacc = _accuracy(ap,bids,labels)
        values[name] = dict(A_old_accuracy=aacc,B_old_accuracy=bacc,C_old_accuracy=oacc,C_new_accuracy=nacc,
            adaptation_gain_B_minus_A=_difference(bacc,aacc),support_adaptation_B_minus_B0=None,
            total_old_accuracy_drop=_difference(bacc,oacc),
            C_abs_new_old_gap=None if nacc is None else abs(nacc-oacc),
            C_h=None if nacc is None else (0. if nacc+oacc == 0 else 2*nacc*oacc/(nacc+oacc)),
            C_old_minus_R0=None,C_new_minus_R0=None,
            old_winner_changed_fraction=None if not bids else sum(op[i] != bp[i] for i in range(len(bids)))/len(bids),
            old_displaced_by_new_fraction=None if not bids else sum(p not in old for p in cold)/len(bids))
        raw[name] = dict(b_ids=bids,c_ids=cids,labels=labels,bp=bp,cp=cp,op=op,
                         ap=None if ground is None else ground['predictions'])
    for name in PATHS:
        value=values[name]; reference=values['R0']
        value['support_adaptation_B_minus_B0']=_difference(value['B_old_accuracy'],reference['B_old_accuracy'])
        value['C_old_minus_R0']=_difference(value['C_old_accuracy'],reference['C_old_accuracy'])
        value['C_new_minus_R0']=_difference(value['C_new_accuracy'],reference['C_new_accuracy'])
        for key in METRIC_NAMES:
            saved=entry['paths'][name]['metrics'].get(key)
            require(saved is None if value[key] is None else _same_number(saved,value[key]), 'Saved support metric differs: '+key)
    return values, raw


def _coords(entry):
    return (entry['scope'],entry.get('fold'),entry.get('trial'))


def _fixed_outer_coords(record):
    """A's native inference scope overwrites outer scope in the producer callback.

    Only its known native scope is interpreted, with unchanged fold/trial
    coordinates. No physical IDs or labels are used to guess an outer path.
    The original record and native scope stay intact for subsequent checks.
    """
    if record.get('stream')!='A':
        require(record.get('scope') in ('support_oof','support_oneshot_proxy'), 'Unknown B/C fixed stream outer scope')
        return _coords(record)
    require(record.get('scope')==NATIVE_A_SCOPE, 'Unknown native A inference scope')
    fold,trial=record.get('fold'),record.get('trial')
    require((type(fold) is int and fold>=0 and trial is None)
            or (fold is None and type(trial) is int and trial>=0), 'Native A has ambiguous/missing outer fold/trial')
    return ('support_oof' if fold is not None else 'support_oneshot_proxy',fold,trial)


def index_fixed_predictions(records, *, run_id, row_id):
    """Index actual callbacks, keeping native A records unchanged and unique."""
    groups=defaultdict(dict)
    streams={'A'}|{path+'_'+side for path in PATHS for side in ('B','C')}
    for record in records:
        require(record.get('status')=='FIXED_BEFORE_SUPPORT_TRUTH_JOIN' and 'held_labels' not in record
                and record.get('run_id')==run_id and record.get('row_id')==row_id,
                'Fixed prediction stream has wrong access/row binding')
        require(record.get('stream') in streams and isinstance(record.get('split_id'),str)
                and record['split_id'], 'Unknown fixed prediction stream/parent')
        key=(record['split_id'],)+_fixed_outer_coords(record)
        require(record['stream'] not in groups[key], 'Duplicate fixed prediction stream')
        groups[key][record['stream']]=record
    return dict(groups)


def _entries(record):
    if record['k'] == 1:
        require(record['folds'] == [] and record['oof'] is None and record['oneshot_proxy'] is None
                and isinstance(record['full_support'],dict), 'K1 must contain only no-held full-support path')
        return [record['full_support']]
    require(record['full_support'] is None and len(record['folds']) == min(record['k'],3)
            and record['oneshot_proxy']['trial_count'] == record['k']
            and len(record['oneshot_proxy']['trials']) == record['k'], 'OOF/proxy path coverage incomplete')
    return record['folds']+record['oneshot_proxy']['trials']


def _stage_costs(records):
    groups=defaultdict(list)
    for record in records:
        for entry in _entries(record):
            for category in ('stages','preparations','candidate_stages'):
                for audit in entry[category]:
                    groups[category+'/'+audit['state']].append(audit)
    output={}
    for name, audits in sorted(groups.items()):
        keys=sorted({key for audit in audits for key,value in audit.items()
                     if _number(value) and (key in COUNTERS or key.endswith(('_seconds','_bytes'))
                         or key in ('factorization_calls','effective_degrees_of_freedom_extra_triangular_solves'))})
        parameter_fields=('trainable_parameter_count','active_parameter_count','declared_coordinate_scalar_count',
            'analytic_new_coefficient_count','analytic_new_intercept_count','analytic_gate_coefficient_count','analytic_gate_intercept_count')
        parameters={}
        for field in parameter_fields:
            values=[x[field] for x in audits if field in x]
            parameters[field]=dict(minimum=min(values) if values else None,maximum=max(values) if values else None,
                available_stage_count=len(values),unavailable_stage_count=len(audits)-len(values))
        output[name]=dict(record_count=len(audits),costs=aggregate_costs(audits,keys),parameter_counts=parameters,
                         executed_core_schemas=sorted({x.get('executed_core_schema',x.get('schema','N/A')) for x in audits}))
    return output


def gate_diagnostic(arrays, *, state_path):
    """Read back the barrier equations; this is no exact hard-optimum certificate."""
    if 'gate_K' not in arrays:
        return None
    K=arrays['gate_K']; alpha=arrays['gate_alpha']; b=float(arrays['gate_b']); t=arrays['gate_targets']
    old=arrays['gate_old_indices'].astype(np.int64); bounds=arrays['gate_lower_bounds']; zeta=float(arrays['gate_zeta'])
    f=K@alpha+b; slack=f[old,None]-bounds
    require(np.isfinite(slack).all() and np.all(slack>0), 'Barrier state has nonpositive slack')
    e=np.exp(-np.abs(f)); p=np.where(f>=0,1/(1+e),e/(1+e))
    q=np.where(t==1,-np.where(f>=0,e/(1+e),1/(1+e)),p)
    mu=zeta/slack; qeff=q.copy(); qeff[old]-=mu.sum(axis=1)
    deff=e/(1+e)**2; deff[old]+=np.sum(zeta/slack**2,axis=1)
    logistic=float(np.logaddexp(0.,np.where(t==1,-f,f)).sum()); ridge=.5*float(alpha@(K@alpha))
    extra={}
    if 'gate_D_eff' in arrays:
        extra['stored_D_eff_readback_max_abs']=float(np.max(np.abs(deff-arrays['gate_D_eff'])))
    if {'K','new_indices','new_alpha','new_intercept','Ynew'}<=set(arrays):
        ni=arrays['new_indices'].astype(np.int64); na=arrays['new_alpha']; nb=arrays['new_intercept']
        extra['new_ridge_equation_residual']=float(np.max(np.abs((arrays['K'][np.ix_(ni,ni)]+np.eye(len(ni)))@na+nb-arrays['Ynew'])))
        extra['new_free_intercept_residual']=float(np.max(np.abs(na.sum(axis=0))))
    extra['adjoint_covector_norms']={name:float(np.linalg.norm(arrays[name]))
        for name in ('gate_K_upstream','gate_L_upstream','bounds_upstream','new_T','new_intercept_adjoint','K_upstream','L_upstream','g_U') if name in arrays}
    return dict(state_path=state_path,physical_count=len(t),constraint_count=bounds.size,zeta=zeta,
        zeta_law_residual=abs(zeta-len(t)*1e-4/bounds.size),minimum_slack=float(slack.min()),
        stored_slack_readback_max_abs=float(np.max(np.abs(slack-arrays['gate_slacks']))),
        stored_function_readback_max_abs=float(np.max(np.abs(f-arrays['gate_f']))),
        stored_qeff_readback_max_abs=float(np.max(np.abs(qeff-arrays['gate_qeff']))),
        canonical_residual=float(np.max(np.abs(alpha+qeff))),intercept_residual=abs(float(qeff.sum())),
        sum_alpha_residual=abs(float(alpha.sum())),primal_gradient_residual=float(np.max(np.abs(K@(alpha+qeff)))),
        logistic_loss_sum=logistic,ridge_penalty=ridge,primal_objective=logistic+ridge,
        barrier_term=-zeta*float(np.log(slack).sum()),mu_positive=bool(np.all(mu>0)),
        complementarity_sum=float(np.sum(mu*slack)),
        theoretical_center_path_gap=bounds.size*zeta,theoretical_average_gap=1e-4,
        exact_dual_certificate=False,**extra,
        evidence_scope='floating readback of finite-barrier approximate equilibrium; mu*slack and theoretical gap are not measured exact dual certificates')


def _metrics_from_raw(raw):
    """Pool disjoint OOF held records, rather than average unequal folds."""
    pooled={}
    for name in PATHS:
        parts=[x[name] for x in raw]; bids=sum((x['b_ids'] for x in parts),[]); cids=sum((x['c_ids'] for x in parts),[])
        require(len(bids)==len(set(bids)) and len(cids)==len(set(cids)), 'OOF held physical ID repeated')
        labels={pid:label for x in parts for pid,label in x['labels'].items()}; bp=sum((x['bp'] for x in parts),[]); cp=sum((x['cp'] for x in parts),[])
        ap=None if any(x['ap'] is None for x in parts) else sum((x['ap'] for x in parts),[])
        cmap=dict(zip(cids,cp)); cold=[cmap[pid] for pid in bids]; nids=[pid for pid in cids if pid not in set(bids)]; op=sum((x['op'] for x in parts),[])
        b=_accuracy(bp,bids,labels); c=_accuracy(cold,bids,labels); n=_accuracy([cmap[pid] for pid in nids],nids,labels); a=None if ap is None else _accuracy(ap,bids,labels)
        pooled[name]=dict(A_old_accuracy=a,B_old_accuracy=b,C_old_accuracy=c,C_new_accuracy=n,
            adaptation_gain_B_minus_A=_difference(b,a),support_adaptation_B_minus_B0=None,
            total_old_accuracy_drop=_difference(b,c),C_abs_new_old_gap=None if n is None else abs(n-c),
            C_h=None if n is None else (0. if n+c==0 else 2*n*c/(n+c)),C_old_minus_R0=None,C_new_minus_R0=None,
            old_winner_changed_fraction=sum(op[i]!=bp[i] for i in range(len(bids)))/len(bids),
            old_displaced_by_new_fraction=sum(p not in set(labels[pid] for pid in bids) for p in cold)/len(bids))
    for name in PATHS:
        for dest,key in (('support_adaptation_B_minus_B0','B_old_accuracy'),('C_old_minus_R0','C_old_accuracy'),('C_new_minus_R0','C_new_accuracy')):
            pooled[name][dest]=_difference(pooled[name][key],pooled['R0'][key])
    return pooled


def _check_lineage(record,entry,run_id,row_id,resolver=None):
    new=record['new_count']; expected=('B_MARGIN',) if new==0 else ('B_MARGIN','C_GROUP_BARRIER_seq')
    require(tuple(x['state'] for x in entry['candidate_stages'])==expected
            and tuple(x['state'] for x in entry['stages'])==(('B0',) if new==0 else ('B0','C0'))
            and tuple(x['state'] for x in entry['preparations'])==(('B',) if new==0 else ('B','C')), 'Actual independent path stage coverage differs')
    for category in ('stages','preparations','candidate_stages'):
        for audit in entry[category]:
            require(audit.get('run_id')==run_id and audit.get('row_id')==row_id and audit.get('split_id')==record['split_id']
                    and _coords(audit)==_coords(entry) and audit.get('parent_k')==record['k']
                    and audit.get('train_k')==entry['train_k'], 'Cross parent/fold/model lineage')
            ids=entry['b_training_ids'] if audit['state'] in ('B0','B','B_MARGIN') else entry['c_training_ids']
            require(audit['training_physical_ids']==ids, 'Actual stage training physical IDs differ')
    bstage=entry['candidate_stages'][0]
    require(bstage.get('mode')=='B' and bstage.get('schema')=='d92_margin_joint_local_ridge_v1'
            and bstage.get('method')=='D92-MarginJointLocalRidge-v1', 'B is not actual Margin B mode/schema')
    if resolver:
        for stage in entry['stages']+entry['candidate_stages']:
            ref=stage['final_state_ref']; resolver.bind(ref); namespace=json.loads(ref['namespace'])
            require(namespace.get('state')==stage['state'] and namespace.get('run_id')==run_id
                    and namespace.get('row_id')==row_id and namespace.get('split_id')==record['split_id']
                    and _coords(namespace)==_coords(entry), 'State reference crosses path/parent')
    if new:
        cstage=entry['candidate_stages'][1]
        require(cstage.get('mode')=='C_seq' and cstage.get('inherited_from_B') is True, 'C did not inherit actual path B')
        require(cstage.get('schema')==SCHEMA and cstage.get('method')==METHOD, 'C schema is not independent Group barrier')
        prep=entry['preparations'][1]
        require(prep.get('inherited_state') is True, 'C preparation is not true sequential inheritance')
        require(prep['final_problem'].get('prior_source')=='FROZEN_CURRENT_ACTUAL_B'
                and prep['final_problem'].get('prior_ref')==bstage.get('final_state_ref'), 'C final prior is not this path actual B')
        if resolver and resolver.verify:
            ba=resolver.load(bstage['final_state_ref']); ca=resolver.load(cstage['final_state_ref'])
            require({'K','new_indices','Ynew','new_alpha','new_intercept','new_schur_z','new_schur_s','new_chol','new_rhs',
                     'gate_K','gate_b','gate_lower_bounds','gate_zeta','anchor_U','prior_B_U'}<=set(ca), 'C full new head/free constants/gate arrays missing')
            require(np.array_equal(ca['anchor_U'],ba['U']) and np.array_equal(ca['prior_B_U'],ba['U']), 'C arrays do not inherit actual B U')
            require(all('prior_B_'+name in ca and np.array_equal(ca['prior_B_'+name],value)
                        for name,value in ba.items()), 'C full frozen B state differs from this path actual B')
            expected_old=np.array([entry['c_training_ids'].index(pid) for pid in entry['b_training_ids']],dtype=np.int64)
            n=len(entry['c_training_ids']); target=np.zeros(n); target[expected_old]=1
            require(ca['gate_K'].shape==(n,n) and np.array_equal(ca['gate_old_indices'],expected_old)
                    and np.array_equal(ca['gate_targets'],target)
                    and ca['gate_lower_bounds'].shape==(len(expected_old),new)
                    and ca['new_alpha'].shape==(n-len(expected_old),new) and ca['new_intercept'].shape==(new,), 'Gate/new head physical training coordinates differ')


def analyze_run(spec, run_root, *, expected_runtime_commit, verify_numeric_states=True):
    """Return a complete, query-blind diagnostic; never fit or select parameters."""
    validate_spec(spec); root=Path(run_root)
    require(isinstance(expected_runtime_commit,str) and len(expected_runtime_commit)==40
            and all(x in '0123456789abcdef' for x in expected_runtime_commit), 'Explicit actual runtime OID required')
    startup=read_json(root/'startup.json'); complete=read_json(root/'complete.json')
    rows=spec['rows']; row_ids={r['row_id'] for r in rows}
    require(len(rows)==4 and len(row_ids)==4 and complete.get('status')==STATUS
            and complete.get('workload_complete') is True and complete.get('completed_rows')==4
            and set(complete['rows'])==row_ids, 'All four complete model rows required')
    for doc in (startup,complete):
        require(doc.get('schema')==SCHEMA and doc.get('method')==METHOD
                and doc.get('run_id')==spec['run_id'] and doc.get('group_id')==spec['group_id']
                and doc.get('runtime_commit')==expected_runtime_commit and doc.get('code_commit')==spec['code']['commit'], 'Root runtime/run binding differs')
        require(doc.get('query_access') is False and doc.get('truth_read') is False
                and doc.get('source_sample_access') is False and doc.get('scorer_invoked') is False, 'Forbidden data access in marker')
    require(startup.get('resolved_spec')==spec, 'Actual supervisor spec differs')
    parent_tables=[]; entry_tables=[]; costs=[]; per_row=[]; gate_states=[]; gate_logs=[]
    paired_old={}; paired_old_held={}; total_parents=0
    for row in rows:
        row_id=row['row_id']; directory=root/row_id; probe=directory/'probe'
        row_start=read_json(directory/'row_startup.json'); config=read_json(directory/'resolved_config.json')
        co=spec['probe']['cohorts'][row['cohort']]; expected_config=evaluator_config(spec,row['cohort'])
        require(config==expected_config and row_start.get('runtime_commit')==expected_runtime_commit
                and row_start.get('run_id')==spec['run_id'] and row_start.get('row_id')==row_id
                and row_start.get('model_seed')==row['seeds']['model']
                and row_start.get('checkpoint_sha256')==row['expected_checkpoint_sha256']
                and row_start.get('capsule_id')==co['capsule_id'], 'Actual row startup differs')
        marker=read_json(probe/'probe_complete.json'); first=read_json(probe/'startup.json')
        binding=dict(schema=SCHEMA,method=METHOD,run_id=spec['run_id'],row_id=row_id,
                     release_commit=expected_runtime_commit,model_seed=row['seeds']['model'],
                     checkpoint_sha256=row['expected_checkpoint_sha256'],capsule_id=co['capsule_id'],
                     group_barrier_resources=spec['probe']['group_barrier_resources'],query_rows_used=0,source_rows_used=0,truth_read=False)
        for doc in (first,marker):
            require(all(doc.get(k)==v for k,v in binding.items()), 'Probe/runtime row binding differs')
        require(first['config']==config and marker['status']==STATUS and marker['workload_complete'] is True
                and first['pid']==marker['pid']==row_start['pid'] and marker['scope']==SCOPE
                and marker['selection']==co['selection'] and marker['producer_matrix']==co['matrix']
                and marker['algorithm']==config['algorithm'], 'Probe marker/config differs')
        require(first.get('ground_A_binding')==marker.get('ground_A_binding')
                and first.get('ground_packet')==marker.get('ground_packet')==row.get('ground_packet'), 'Native A packet binding differs')
        if row.get('ground_packet'):
            packet=marker['ground_A_binding']
            require(packet.get('status')=='MATCHED_SOURCE_ONLY_PACKET' and packet.get('path')==row['ground_packet']
                    and packet.get('checkpoint_sha256')==row['expected_checkpoint_sha256']
                    and packet.get('model_seed')==row['seeds']['model'], 'Native A matched checkpoint/model binding differs')
        require(complete['rows'][row_id]['status']==STATUS and complete['rows'][row_id]['pid']==row_start['pid'], 'Supervisor row not complete')
        resolver=GroupStateResolver(probe,marker,verify_numeric_states=verify_numeric_states)
        traces=read_jsonl(probe/'fit_trace.jsonl'); compact=read_jsonl(probe/'compact.jsonl')
        fixed=read_jsonl(probe/'fixed_predictions.jsonl'); events=read_jsonl(probe/'training_events.jsonl')
        selection={x['split_id']:x for x in co['selection']['splits']}
        require(len(traces)==len(compact)==40 and {x['split_id'] for x in traces}==set(selection)
                and len({x['split_id'] for x in traces})==40 and {x['split_id'] for x in compact}==set(selection), 'Complete 40-parent row required')
        resolver.close_references([traces,events],scalar_projected_records=read_jsonl(probe/'fit_stages.jsonl'))
        frozen=index_fixed_predictions(fixed,run_id=spec['run_id'],row_id=row_id)
        consumed=set(); compact_by={x['split_id']:x for x in compact}
        for record in traces:
            sid=record['split_id']; selected=selection[sid]; small=compact_by[sid]
            require(all(record.get(k)==selected[k] for k in IDENTITY)
                    and record['classes']==sorted(selected['registered_classes']) and record['old_classes']==sorted(selected['registered_classes'][:6])
                    and record['support_count']==record['k']*len(record['classes'])
                    and record['old_class_count']==6 and record['new_class_count']==record['new_count']
                    and record['scope']==SCOPE and record['query_rows_used']==record['source_rows_used']==0
                    and record['inheritance_binding']==dict(run_id=spec['run_id'],row_id=row_id,split_id=sid)
                    and record['group_barrier_resources']==binding['group_barrier_resources'], 'Parent identity/permission differs')
            require(all(small.get(k)==record[k] for k in IDENTITY)
                    and all(_same_number(small.get(k),record[k]) for k in COUNTERS[4:]), 'Compact parent counters differ')
            entries=_entries(record); parsed=[]; raws=[]
            all_old=set(); all_ids=set(); holdmap={}
            for entry in entries:
                require(entry['parent_k']==record['k'] and entry['train_k']+entry['held_k']==record['k'], 'Physical path K differs')
                _check_lineage(record,entry,spec['run_id'],row_id,resolver)
                if entry.get('ground_A') is not None:
                    require(entry['ground_A']['classes']==marker['ground_A_binding'].get('ordered_classes'),
                            'A ordered native class registry differs from matched packet')
                elif entry['c_ids'] and row.get('ground_packet'):
                    raise ValueError('Matched native A packet lacks paired old-held predictions')
                key=(sid,)+_coords(entry); frozen_entry=frozen.get(key,{})
                if key in frozen: consumed.add(key)
                metrics,raw=assess_entry(entry,record['old_classes'],record['classes'],record['new_count'],frozen_entry,resolver)
                old_pair_key=(row_id,record['receiver'],record['scenario'],record['k'],record['support_seed'])+_coords(entry)
                old_held={pid:entry['held_labels'][pid] for pid in entry['b_ids']}
                require(paired_old_held.setdefault(old_pair_key,old_held)==old_held, 'Paired old held physical IDs/labels changed across new counts')
                all_old.update(entry['b_training_ids']); all_old.update(entry['b_ids']); all_ids.update(entry['c_training_ids']); all_ids.update(entry['c_ids'])
                holdmap.update(entry['held_labels'])
                parsed.append((entry,metrics)); raws.append(raw)
                dims=dict(row_id=row_id,model_seed=row['seeds']['model'],cohort=row['cohort'],**{k:record[k] for k in IDENTITY})
                for path in PATHS:
                    entry_tables.append(dict(dims,path=path,scope=entry['scope'],fold=entry['fold'],trial=entry['trial'],
                        old_held_physical_count=len(entry['b_ids']),new_held_physical_count=len(entry['c_ids'])-len(entry['b_ids']),
                        metric_unit='fraction; differences are fraction-point differences',**metrics[path]))
                for stage in entry['candidate_stages']:
                    if stage['state']=='C_GROUP_BARRIER_seq':
                        for audit in _walk_gate_audits(stage):
                            gate_logs.append(dict(dims,scope=entry['scope'],fold=entry['fold'],trial=entry['trial'],audit=audit,
                                certificate_scope='logged finite-barrier approximate equilibrium, not old QP or exact hard-optimal certificate'))
            require(len(all_ids)==record['support_count'] and len(all_old)==6*record['k'], 'Parent physical support count differs')
            paired_key=(row_id,record['receiver'],record['scenario'],record['k'],record['support_seed'])
            require(paired_old.setdefault(paired_key,all_old)==all_old, 'Old physical supports changed across new counts')
            if record['k']==1:
                require(not holdmap and record['heldout_unavailable_reason']=='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT', 'K1 held labels are forbidden')
                modes={scope:{path:dict.fromkeys(METRIC_NAMES) for path in PATHS}
                       for scope in ('support_oof','support_oneshot_proxy')}
            else:
                assignment=record['physical_fold_assignment']; amap={x['physical_id']:x for x in assignment}
                require(len(amap)==len(assignment)==record['support_count'] and set(amap)==all_ids
                        and all(x['class_id']==holdmap[x['physical_id']] for x in assignment), 'Physical fold assignment differs')
                for entry in record['folds']:
                    require(set(entry['c_ids'])=={pid for pid,x in amap.items() if x['fold']==entry['fold']}, 'OOF physical held differs')
                anchors=[pid for entry in record['oneshot_proxy']['trials'] for pid in entry['c_training_ids']]
                require(len(anchors)==len(set(anchors))==record['support_count'] and set(anchors)==all_ids
                        and all(entry['train_k']==1 for entry in record['oneshot_proxy']['trials']), 'Proxy physical anchors are missing/repeated')
                oofraw=[raw for (entry,_),raw in zip(parsed,raws) if entry['scope']=='support_oof']
                modes={'support_oof':_metrics_from_raw(oofraw),
                       'support_oneshot_proxy':{path:{key:_mean([metrics[path][key] for entry,metrics in parsed if entry['scope']=='support_oneshot_proxy']) for key in METRIC_NAMES} for path in PATHS}}
                for path in PATHS:
                    for key in METRIC_NAMES:
                        for name,saved in (('support_oof',record['oof']['paths'][path]['metrics']),('support_oneshot_proxy',record['oneshot_proxy']['parent_mean_metrics'][path])):
                            v=modes[name][path][key]; require(saved.get(key) is None if v is None else _same_number(saved.get(key),v), 'Parent pooled/proxy metric differs')
            for scope,metrics in modes.items():
                for path in PATHS:
                    parent_tables.append(dict(dims,path=path,scope=scope,old_class_count=6,new_class_count=record['new_count'],
                        heldout_available=record['k']>1,
                        actual_A_unavailable_reason=None if metrics[path]['A_old_accuracy'] is not None else ('K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if record['k']==1 else 'NO_EXPLICIT_MATCHED_SOURCE_ONLY_GROUND_PACKET'),
                        metric_unit='fraction; multiply differences by 100 for percentage points',**metrics[path]))
        require(consumed==set(frozen), 'Fixed streams from unknown parent/path')
        rowcost=aggregate_costs(traces,COUNTERS[4:]); expected=dict(rowcost['sum'],**rowcost['max'])
        expected.update(episodes=40,k1_episodes=sum(x['k']==1 for x in traces),oof_episodes=sum(x['k']>1 for x in traces),
                        proxy_anchor_count=sum(0 if x['k']==1 else x['k'] for x in traces))
        require(all(_same_number(marker.get(k),v) for k,v in expected.items()), 'Row measured counter sums/maxima differ')
        require(all(marker.get(k)==v for k,v in spec['probe']['budget']['per_row'][row_id]['exact'].items()), 'Actual row structural workload is incomplete')
        for key in COUNTERS:
            require(_same_number(complete['rows'][row_id].get(key),marker[key]), 'Supervisor row measured costs differ')
        if verify_numeric_states:
            for item in resolver.gate_readbacks:
                gate_states.append(dict(item,state_path=row_id+'/probe/'+item['state_path']))
        payload=marker['payload_audit']; packet=marker.get('ground_A_binding',{})
        require(payload.get('new_source_payload_bytes')==0 and payload.get('new_ground_statistics_bytes')==0, 'Extra ground data/stat input')
        deployments=[entry['minimum_deployment_numeric_state_bytes']['R_GROUP_BARRIER_seq']
            for record in traces for entry in _entries(record) if 'minimum_deployment_numeric_state_bytes' in entry]
        path_count=sum(len(_entries(record)) for record in traces)
        per_row.append(dict(row_id=row_id,model_seed=row['seeds']['model'],cohort=row['cohort'],parents=40,
            costs=rowcost,phase_costs=_stage_costs(traces),hardware=first.get('hardware'),
            wall_seconds=marker.get('wall_seconds'),peak_process_rss_bytes=marker.get('peak_process_rss_bytes'),
            peak_gpu_memory_bytes=marker.get('peak_gpu_memory_bytes'),resident_numeric_state_bytes=marker.get('persistent_state_bytes'),
            maximum_minimal_deployment_numeric_state_bytes=max(deployments) if len(deployments)==path_count else None,
            minimal_deployment_available_path_count=len(deployments),minimal_deployment_unavailable_path_count=path_count-len(deployments),
            state_archive_file_bytes=marker['state_archive_file_bytes'],state_archive_numeric_bytes=marker['state_archive_numeric_bytes'],
            native_packet_file_bytes=packet.get('packet_file_bytes'),native_head_weight_bytes=packet.get('native_head_weight_bytes'),
            extra_ground_data_stat_payload_bytes=0,code_wire_payload_bytes=None,incremental_transfer_bytes=payload.get('incremental_transfer_bytes'),
            star_training_seconds=None,star_inference_seconds=None,star_peak_memory_bytes=None,
            unknown_measurement_reason='CPU support diagnostic; native packet file bytes are existing local storage, not measured wire or star cost'))
        costs.append(marker); total_parents+=40
    require(total_parents==160, 'All 160 parents required')
    combined=aggregate_costs(costs)
    require(all(_same_number(complete.get(k),v) for k,v in dict(combined['sum'],**combined['max']).items()), 'Root measured SUM/MAX counters differ')
    grouped=_group_tables(parent_tables)
    return dict(schema=ANALYSIS_SCHEMA,method=METHOD,status='COMPLETE_SUPPORT_DIAGNOSTIC',runtime_commit=expected_runtime_commit,
        run_id=spec['run_id'],row_count=4,parent_count=160,scope=SCOPE,query_performance=False,
        frozen_algorithm=spec['probe']['algorithm'],actual_group_barrier_resources=spec['probe']['group_barrier_resources'],
        parameter_selection=False,automatic_promotion=False,query_read=False,external_truth_read=False,
        interpretation='Fixed support OOF/proxy diagnostic; no query performance or performance target certification. K1 has no independent held support. new0 has no new accuracy, H, or gap.',
        path_lineage={'R0':'independently fitted old B0 then independently refitted all-class C0; no sequential inherited adapter',
                      'R_GROUP_BARRIER_seq':'actual unchanged Margin B; Group C starts from and freezes this same path B old conditional function'},
        metrics=parent_tables,entry_metrics=entry_tables,tables=grouped,costs=combined,rows=per_row,
        gate_numeric_readbacks=gate_states,gate_logged_audits=gate_logs,
        gate_evidence='All-pair positive barrier equilibrium; no independent-active-set/strict-complementarity QP certificate. Theoretical N*1e-4 gap and mu*slack are not exact measured dual certificates.',
        numeric_state_verification=verify_numeric_states)


def _walk_gate_audits(value):
    if isinstance(value,dict):
        if 'gate_fit_audit' in value:
            yield value['gate_fit_audit']
        # final_objective and gradient/trial objects contain distinct measured
        # evaluations. They are retained as evidence, never summed as costs.
        for key,item in value.items():
            if key not in ('preparation','preparation_ref'):
                yield from _walk_gate_audits(item)
    elif isinstance(value,list):
        for item in value:
            yield from _walk_gate_audits(item)


def _group_tables(rows):
    output={}
    for name,dimensions in (('row',('row_id',)),('receiver_scenario_modelseed',('receiver','scenario','model_seed')),
                            ('K_by_new',('k','new_count')),('row_K_by_new',('row_id','k','new_count'))):
        grouped=defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in dimensions+('scope','path'))].append(row)
        values=[]
        for identity,items in sorted(grouped.items()):
            value=dict(zip(dimensions+('scope','path'),identity)); value['parent_count']=len(items)
            for metric in METRIC_NAMES:
                available=[x[metric] for x in items if x[metric] is not None]
                value[metric]=None if not available else sum(available)/len(available)
                value[metric+'_available_parent_count']=len(available)
            value['aggregation']='equal parent mean; available coverage explicit; H/gap are mean per-parent nonlinear metrics'
            values.append(value)
        output[name]=values
    return output


def write_analysis(result, output):
    """Write only a fresh analysis directory after complete closure succeeds."""
    output=Path(output); require(not output.exists(), 'Analysis output exists; artifacts are preserved')
    output.mkdir(parents=True)
    with (output/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2); stream.write('\n')
    tables={'parents':result['metrics'],'paths':result['entry_metrics'],**result['tables']}
    for name,rows in tables.items():
        with (output/(name+'.csv')).open('x',encoding='utf-8',newline='') as stream:
            keys=list(dict.fromkeys(k for row in rows for k in row)); writer=csv.DictWriter(stream,fieldnames=keys)
            writer.writeheader(); writer.writerows([{k:('N/A' if v is None else v) for k,v in row.items()} for row in rows])
    with (output/'report.md').open('x',encoding='utf-8') as stream:
        stream.write('# GroupBarrier support 诊断\n\n')
        stream.write(f"已闭合 {result['row_count']} 行、{result['parent_count']} 个 parent。实际 runtime：`{result['runtime_commit']}`。\n\n")
        stream.write('本报告只描述 support OOF 与 one-shot proxy，不代表 query 性能，不选参数、不自动晋级。A、B、C 按同 parent、同旧类物理 held 配对；K=1 的准确率均为 N/A，new0 的新类准确率、H 与新旧差距均为 N/A。\n\n')
        stream.write('R0 独立拟合 B0/C0；R_GROUP_BARRIER_seq 的 B 是实际 Margin B，C 继承该 B，并冻结旧类条件函数。原始同 parent 表与完整 K×新增类数表见 CSV。指标单位为比例，gain/forget 差乘 100 后为百分点。\n\n')
        stream.write('SUM 计数与耗时、MAX 实际 factor buffer 峰值分开保存。新 Ridge、gate 前向/伴随、准备与实际 B 的成本均保留；训练 events 的累计 audit 不重复计费。CPU 测量和本地 native packet 文件字节不代替星载测量或传输字节；未测量项为 N/A。\n\n')
        stream.write('barrier 日志与 NPZ 原方程读回单独保存。理论 gap 和 μ×slack 不是精确实测 dual 证书，旧 QP 的独立活跃集、严格互补检查不适用于该 gate。\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','run-root','output','expected-runtime-commit'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    require(not Path(args.output).resolve().is_relative_to(Path(args.run_root).resolve()),
            'Analysis output must be outside the closed original run artifacts')
    result=analyze_run(read_json(args.spec),args.run_root,expected_runtime_commit=args.expected_runtime_commit)
    write_analysis(result,args.output)


if __name__=='__main__':
    main()
