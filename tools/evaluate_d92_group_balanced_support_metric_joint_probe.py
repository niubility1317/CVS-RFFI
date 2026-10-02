"""Independent prototype-frame support paths; actual candidate B and no query I/O.

The independent candidate requires root validation and publication. This
entry accepts support caches only; it never opens query or source samples.
"""
import argparse
from collections.abc import Mapping
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from cvsrffi import d92_branch_interaction as interaction
from cvsrffi.d92_branch_local_ridge import fit_branch_local_ridge
from cvsrffi.d92_branch_support_probe import FROZEN_CONFIG as CACHE_VALIDATION_CONFIG
from cvsrffi.d92_group_balanced_support_metric_joint_local_ridge import (
    FROZEN_CONFIG, prepare_group_balanced_support_metric_joint_training, fit_group_balanced_support_metric_joint_local_ridge,
    WORK_SUM_KEYS, WORK_MAX_KEYS,
)
from cvsrffi import d92_group_balanced_support_metric_joint_local_ridge as candidate_core
from cvsrffi.d92_support_metric_basis import build_support_metric_basis
from evaluate_d92_registration_diagnostic import (
    CHANNEL, SCENARIOS, BRANCHES, split_identity,
    diagnose_evidence, measured_metrics as baseline_metrics, _safe,
)
from evaluate_d92_branch_support_probe import load_support, read, write, check, scalars, csv_record, peak_rss
from evaluate_d92_branch_local_ridge_probe import stage_csv_from_jsonl
from run_d92_margin_joint_probe import validate_selection, selected_tasks
from cvsrffi.d92_proto_frame_ground_geometry import load_proto_frame_ground_geometry
from export_d92_ground_classifier_a_packet import load_packet
import torch

STATUS = 'GROUP_BALANCED_SUPPORT_METRIC_JOINT_PROBE_COMPLETE'
SCOPE = 'SUPPORT_ONLY_GROUP_BALANCED_SUPPORT_METRIC_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION'
SCHEMA = 'd92_group_balanced_support_metric_joint_support_probe_v1'
METHOD = FROZEN_CONFIG['method']
IMPLEMENTATION_STATUS = 'SOURCE_GROUP_BALANCED_SUPPORT_METRIC_GGN1_SUPPORT_DIAGNOSTIC_NOT_QUERY_PERFORMANCE'
PATHS = ('R0', 'R_GROUP_BALANCED_SUPPORT_METRIC_seq')
PROBE_CONFIG = deepcopy(FROZEN_CONFIG)
RESOURCE_KEYS = frozenset(('max_newton_iterations','max_line_search_trials','max_factor_buffer_bytes',
    'max_integer_bits','max_fraction_operations','max_secular_iterations'))
COUNTERS = ('episodes','k1_episodes','oof_episodes','proxy_anchor_count',
    'sequence_paths','baseline_head_fit_count','baseline_factorization_count','baseline_triangular_solve_count',
    'candidate_preparation_count','candidate_stage_count','final_candidate_head_fit_count',
    'optimizer_steps','trial_count','accepted_trial_count','rejected_trial_count',
    'ggn_step_count','ggn_parameter_direction_count','final_score_evaluation_count','final_score_physical_count',
    'ground_A_score_evaluation_count','ground_A_score_physical_count','ground_A_inference_seconds',
    'structural_predict_evaluation_count','structural_predict_physical_count','structural_predict_seconds',
    'peak_resident_numeric_state_bytes','row_basis_construction_count','peak_effective_adapter_rank')
PEAK_COUNTERS = frozenset(('peak_resident_numeric_state_bytes','peak_effective_adapter_rank'))
COUNTERS = tuple(dict.fromkeys(COUNTERS+tuple(WORK_SUM_KEYS)+tuple(WORK_MAX_KEYS)))
PEAK_COUNTERS = PEAK_COUNTERS.union(WORK_MAX_KEYS)
SUM_COUNTERS = tuple(key for key in COUNTERS if key not in PEAK_COUNTERS)
WORK_KEYS = frozenset(WORK_SUM_KEYS).union(WORK_MAX_KEYS)


def validate_resources(resources):
    check(isinstance(resources,dict) and set(resources)==RESOURCE_KEYS
        and all(type(value) is int and value>0 for value in resources.values()),
        'Explicit positive integer SupportMetric resources required')
    check(resources['max_secular_iterations']<=128,'max_secular_iterations must be at most 128')
    return resources


def oid(value):
    return isinstance(value,str) and len(value)==40 and all(c in '0123456789abcdef' for c in value)


METRICS = ('A_old_accuracy', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_columns_accuracy',
    'C_old_accuracy', 'C_new_accuracy', 'C_h', 'adaptation_gain_B_minus_A', 'support_adaptation_B_minus_B0',
    'old_order_change', 'new_competition_loss', 'total_old_accuracy_drop', 'C_abs_new_old_gap',
    'C_new_minus_old', 'C_old_minus_R0', 'C_new_minus_R0', 'old_order_recovery_fraction',
    'total_recovery_fraction', 'old_winner_changed_fraction', 'old_displaced_by_new_fraction') + tuple(
        'correctness_B_Cold_C_'+bits for bits in ('000', '100', '010', '110', '011', '111'))
MARGIN_FIELDS = ('margin_mean', 'margin_minus_R0', 'winner_changed_from_R0_fraction',
    'R0_correct_to_wrong_fraction', 'R0_wrong_to_correct_fraction')
METRICS += tuple(stage+'_'+field for stage in ('B_old', 'C_old', 'C_new') for field in MARGIN_FIELDS)



class StateArchive:
    """Exclusive numeric archives: JSON references retain every array coordinate."""
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root/'state_arrays'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files = []

    def __call__(self, key, arrays):
        return self._save(key, arrays, allow_nonfinite=False)

    def failure(self, key, arrays):
        """Retain failed numeric states, including nonfinite coordinates, as NPZ."""
        return self._save(key, arrays, allow_nonfinite=True)

    def _save(self, key, arrays, *, allow_nonfinite):
        check(isinstance(key, str) and bool(key), 'State archive key must be nonempty text')
        check(isinstance(arrays, Mapping) and bool(arrays), 'State archive requires numeric arrays')
        numeric, metadata, summaries = {}, {}, {}
        for name, value in arrays.items():
            check(isinstance(name, str) and name and '/' not in name, 'Invalid state array name')
            array = np.asarray(value)
            check(array.dtype.kind in 'fbiu', 'Nonnumeric archived state')
            finite = bool(np.isfinite(array).all())
            check(allow_nonfinite or finite, 'Nonfinite archived successful state')
            check(array.dtype.kind != 'f' or array.dtype in (np.dtype('float32'),np.dtype('float64')), 'State floats must be float32 native A or float64 model')
            numeric[name] = np.array(array, copy=True)
            metadata[name] = dict(shape=list(array.shape), dtype=str(array.dtype), nbytes=int(array.nbytes),
                all_finite=finite, nonfinite_count=int(np.sum(~np.isfinite(array))))
            norm = float(np.linalg.norm(array.reshape(-1))) if finite else None
            summaries[name] = dict(norm=norm if norm is not None and math.isfinite(norm) else None,
                minimum=float(np.min(array)) if finite and array.size else None,
                maximum=float(np.max(array)) if finite and array.size else None)
        relative = 'state_arrays/'+str(len(self.files)).zfill(8)+'.npz'
        path = self.root/relative
        tick = time.perf_counter()
        with path.open('xb') as stream:
            np.savez_compressed(stream, **numeric)
        ref = json_native(dict(key=key.rsplit('/', 1)[-1], namespace=key.rsplit('/', 1)[0] if '/' in key else None,
            path=relative, arrays=metadata, array_summaries=summaries, file_bytes=path.stat().st_size,
            archive_seconds=time.perf_counter()-tick, failed_numeric_state=allow_nonfinite))
        self.files.append(ref)
        return deepcopy(ref)

    def finalize(self, status):
        phases = {}
        for ref in self.files:
            phase = json.loads(ref['namespace'])['state'] if ref['namespace'] else 'unscoped'
            group = phases.setdefault(phase, dict(file_count=0, file_bytes=0, numeric_array_bytes=0, archive_seconds=0.))
            group['file_count'] += 1; group['file_bytes'] += ref['file_bytes']
            group['numeric_array_bytes'] += sum(a['nbytes'] for a in ref['arrays'].values())
            group['archive_seconds'] += ref['archive_seconds']
        manifest = dict(schema='d92_group_balanced_support_metric_joint_state_archive_v1', method=METHOD,
            prediction_formula='frozen_actual_ProtoFrame_B_old_conditional_plus_new_only_Ridge_and_Bernoulli_barrier_gate', status=status, files=self.files,
            file_count=len(self.files), total_file_bytes=sum(ref['file_bytes'] for ref in self.files),
            numeric_array_bytes=sum(a['nbytes'] for ref in self.files for a in ref['arrays'].values()),
            archive_seconds=sum(ref['archive_seconds'] for ref in self.files), by_phase=phases)
        with (self.root/'state_manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(json_native(manifest), stream, ensure_ascii=False, allow_nan=False)
        return manifest


def json_native(value, path='$'):
    """Convert only NumPy/container data types at the public output boundary."""
    if isinstance(value, np.generic): return json_native(value.item(), path)
    if isinstance(value, np.ndarray): return json_native(value.tolist(), path)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            native_key = json_native(key, path+'<key>')
            if not isinstance(native_key, str): raise TypeError(path+': JSON object key must be a string')
            result[native_key] = json_native(item, path+'.'+native_key)
        return result
    if isinstance(value, (list, tuple)):
        return [json_native(item, path+'['+str(index)+']') for index, item in enumerate(value)]
    if value is None or type(value) in (str, bool, int): return value
    if type(value) is float:
        if not math.isfinite(value): raise ValueError(path+': nonfinite JSON number')
        return value
    raise TypeError(path+': unsupported JSON value '+type(value).__module__+'.'+type(value).__name__)


def held_observations(ids, classes, scores, labels, predictions=None):
    """Post-fit outer-held diagnostics only; never called by the fitting API."""
    values = np.asarray(scores, dtype=np.float64)
    check(values.shape == (len(ids), len(classes)), 'Held score shape mismatch')
    check(np.isfinite(values).all(), 'Nonfinite held scores')
    result = {}
    for pid, row in zip(ids, values):
        target = classes.index(labels[pid]); pred = min(classes[j] for j in np.flatnonzero(row == np.max(row)))
        if predictions is not None:pred=predictions[pid];check(pred in classes,'Fixed predicted class unregistered')
        wrong = [value for j, value in enumerate(row) if j != target]
        result[pid] = dict(predicted_class=pred, correct=pred == labels[pid],
            true_class_margin=float(row[target]-max(wrong)) if wrong else None)
    return result


def held_comparisons(evidence, reference, old):
    result = {}; metrics = {}
    for stage, ids, classes, scores, ref_scores in (
        ('B_old', evidence['b_ids'], evidence['b_classes'], evidence['b_scores'], reference['b_scores']),
        ('C_old', [pid for pid in evidence['c_ids'] if evidence['held_labels'][pid] in old],
            evidence['c_classes'], evidence['c_scores'], reference['c_scores']),
        ('C_new', [pid for pid in evidence['c_ids'] if evidence['held_labels'][pid] not in old],
            evidence['c_classes'], evidence['c_scores'], reference['c_scores'])):
        all_ids = evidence['b_ids'] if stage == 'B_old' else evidence['c_ids']
        actual = held_observations(all_ids, classes, scores, evidence['held_labels'],evidence.get('c_predictions') if stage!='B_old' else None)
        baseline = held_observations(all_ids, classes, ref_scores, evidence['held_labels'])
        pairs = [dict(physical_id=pid, actual=actual[pid], R0=baseline[pid]) for pid in ids]
        result[stage] = dict(evidence_scope='OUTER_HELD_AFTER_ALL_UPDATES', records=pairs)
        n = len(pairs); margins = [p['actual']['true_class_margin'] for p in pairs]
        refs = [p['R0']['true_class_margin'] for p in pairs]
        known = n > 0 and all(v is not None for v in margins+refs)
        metrics.update({stage+'_margin_mean': sum(margins)/n if known else None,
            stage+'_margin_minus_R0': sum(a-b for a, b in zip(margins, refs))/n if known else None,
            stage+'_winner_changed_from_R0_fraction': sum(p['actual']['predicted_class'] != p['R0']['predicted_class'] for p in pairs)/n if n else None,
            stage+'_R0_correct_to_wrong_fraction': sum(p['R0']['correct'] and not p['actual']['correct'] for p in pairs)/n if n else None,
            stage+'_R0_wrong_to_correct_fraction': sum(not p['R0']['correct'] and p['actual']['correct'] for p in pairs)/n if n else None})
    return result, metrics


def path_metrics(diagnostic, reference, actual_A=None):
    old, new = diagnostic['C_full_acc'], diagnostic['Cnew_acc']; base = baseline_metrics(diagnostic)
    result = dict(A_old_accuracy=actual_A, B0_old_accuracy=reference['B_acc'], B_old_accuracy=diagnostic['B_acc'],
        C_old_columns_accuracy=diagnostic['C_old_columns_acc'], C_old_accuracy=old, C_new_accuracy=new,
        C_h=None if new is None else diagnostic['C_harmonic_mean'], adaptation_gain_B_minus_A=None if actual_A is None else diagnostic['B_acc']-actual_A,
        support_adaptation_B_minus_B0=diagnostic['B_acc']-reference['B_acc'],
        C_abs_new_old_gap=None if new is None else abs(new-old), C_new_minus_old=None if new is None else new-old,
        C_old_minus_R0=old-reference['C_full_acc'], C_new_minus_R0=None if new is None else new-reference['Cnew_acc'],
        **{key: base[key] for key in ('old_order_change', 'new_competition_loss', 'total_old_accuracy_drop',
            'old_order_recovery_fraction', 'total_recovery_fraction', 'old_winner_changed_fraction', 'old_displaced_by_new_fraction')})
    for bits in ('000', '100', '010', '110', '011', '111'):
        result['correctness_B_Cold_C_'+bits] = sum(''.join(str(int(row[key])) for key in
            ('B_correct', 'C_old_columns_correct', 'C_full_correct')) == bits for row in diagnostic['old_records'])/diagnostic['old_held_count']
    return result


def assess_paths(evidence, old):
    diagnostics = {name: diagnose_fixed_support_metric(value,old) if name=='R_GROUP_BALANCED_SUPPORT_METRIC_seq' else diagnose_evidence(value, old) for name,value in evidence.items()}
    result = {}
    for name, value in diagnostics.items():
        ground=evidence[name].get('ground_A')
        actual_A=None
        if ground is not None:
            check(ground['physical_ids']==evidence[name]['b_ids'], 'A/B old-held physical IDs differ')
            check(set(ground['classes'])==set(old) and len(ground['classes'])==len(old), 'A original six-class registry differs')
            actual_A=sum(pred==evidence[name]['held_labels'][pid] for pid,pred in
                zip(ground['physical_ids'],ground['predictions']))/len(ground['physical_ids'])
        comparisons, margins = held_comparisons(evidence[name], evidence['R0'], old)
        result[name] = dict(diagnostic=value, held_comparisons=comparisons,
            metrics=dict(path_metrics(value, diagnostics['R0'],actual_A), **margins))
    return result


def diagnose_fixed_support_metric(evidence,old):
    """Use public structural C decisions, preserving frozen raw B old ordering."""
    from d92_registration_score_diagnostics import _transitions,_quantity
    bi,ci=evidence['b_ids'],evidence['c_ids'];truth=evidence['held_labels'];cc=evidence['c_classes']
    check(set(truth)==set(ci) and set(bi)=={pid for pid in ci if truth[pid] in old},'B/C held physical pairing mismatch')
    B=held_observations(bi,old,evidence['b_scores'],truth)
    C=held_observations(ci,cc,evidence['c_scores'],truth,evidence['c_predictions'])
    oldrows=[];newrows=[]
    for pid in sorted(bi):
        bp,cp=B[pid]['predicted_class'],C[pid]['predicted_class'];bo=bp==truth[pid];cf=cp==truth[pid]
        check(cp not in old or cp==bp,'Structured C old winner differs from actual frozen B')
        check(not(cf and not bo),'Structured C old winner differs from actual frozen B')
        oldrows.append(dict(physical_id=pid,true_class=truth[pid],B_predicted_class=bp,C_old_columns_predicted_class=bp,
            C_full_predicted_class=cp,B_correct=bo,C_old_columns_correct=bo,C_full_correct=cf,
            old_winner_changed=False,C_old_winner_displaced_by_new=cp not in old))
    for pid in sorted(set(ci)-set(bi)):
        newrows.append(dict(physical_id=pid,true_class=truth[pid],C_full_predicted_class=C[pid]['predicted_class'],C_full_correct=C[pid]['correct']))
    n=len(bi);bn=sum(r['B_correct'] for r in oldrows);cn=sum(r['C_full_correct'] for r in oldrows)
    nc=sum(r['C_full_correct'] for r in newrows);new=nc/len(newrows) if newrows else None;oa=cn/n
    ordering=_transitions([r['B_correct'] for r in oldrows],[r['B_correct'] for r in oldrows])
    competition=_transitions([r['B_correct'] for r in oldrows],[r['C_full_correct'] for r in oldrows])
    return dict(schema='d92_support_metric_fixed_decision_diagnostics_v1',decision_rule='PUBLIC_STRUCTURAL_PREDICT_FROZEN_B_OLD_ORDER',
        old_held_count=n,B_acc=bn/n,C_old_columns_acc=bn/n,C_full_acc=oa,Cnew_acc=new,
        C_harmonic_mean=None if new is None else (2*oa*new/(oa+new) if oa+new else 0.),
        old_records=oldrows,new_records=newrows,old_winner_changed_count=0,
        old_winner_displaced_by_new_count=sum(r['C_old_winner_displaced_by_new'] for r in oldrows),
        decomposition=dict(old_order_change=_quantity(0,n),new_competition_loss=_quantity(bn-cn,n),total_old_accuracy_drop=_quantity(bn-cn,n)),
        transitions=dict(old_order_change=ordering,new_competition=competition,total=competition),
        limitation='Support diagnostic only; C_old_columns is the frozen B conditional counterfactual; no query routing')


def parent_mean(trials):
    return {name: {metric: None if trials[0]['paths'][name]['metrics'][metric] is None else
        sum(t['paths'][name]['metrics'][metric] for t in trials)/len(trials) for metric in METRICS} for name in PATHS}


def pooled_assess(entries, labels, classes, old):
    evidence = {}
    for name in PATHS:
        brows, crows, apreds, ascores, cpreds, aclasses = {}, {}, {}, {}, {}, None
        for entry in entries:
            check(not set(brows).intersection(entry['b_ids']) and not set(crows).intersection(entry['c_ids']), 'Duplicate physical held ID')
            brows.update(zip(entry['b_ids'], entry['paths'][name]['b_scores']))
            crows.update(zip(entry['c_ids'], entry['paths'][name]['c_scores']))
            cpreds.update(entry['paths'][name].get('c_predictions',{}))
            ground=entry.get('ground_A')
            if ground is not None:
                check(aclasses is None or aclasses==ground['classes'],'A original class order changed')
                aclasses=ground['classes'];apreds.update(zip(ground['physical_ids'],ground['predictions']))
                ascores.update(zip(ground['physical_ids'],ground['scores']))
        evidence[name] = dict(b_ids=sorted(brows), b_classes=old, b_scores=[brows[i] for i in sorted(brows)],
            c_ids=sorted(crows), c_classes=classes, c_scores=[crows[i] for i in sorted(crows)], held_labels=labels)
        if name=='R_GROUP_BALANCED_SUPPORT_METRIC_seq':evidence[name]['c_predictions']=cpreds
        if aclasses is not None:
            check(set(apreds)==set(brows),'Incomplete paired A old-held coverage')
            evidence[name]['ground_A']=dict(physical_ids=sorted(brows),classes=aclasses,
                predictions=[apreds[i] for i in sorted(brows)],scores=[ascores[i] for i in sorted(brows)])
    return assess_paths(evidence, old)


_write_native_json = write


def write(path, value):
    return _write_native_json(path, json_native(value))


def compact_event(event):
    """Keep measured scalars and complete NPZ references without copying matrices."""
    event = json_native(event)
    result = scalars(event)
    for key, value in event.items():
        if key in ('state_ref', 'head_ref', 'head_refs', 'prior_ref') or key.endswith('_state_ref') or key.endswith('_head_ref'):
            result[key] = value
    for key in ('objective', 'audit', 'preparation', 'initial_objective', 'final_objective', 'final_fit'):
        value = event.get(key)
        if isinstance(value, dict):
            result[key] = compact_event(value)
    for key in ('class_ce_means','class_ce_sums','class_ce_counts','class_weights',
            'theta','anchor_theta','gradient','direction','metric_fisher','metric_M','metric_H','physical_gram',
            'held_ce_sums','held_ce_counts','classes','old_classes'):
        if key in event: result[key] = event[key]
    for key in ('inner_folds','folds','trials'):
        if isinstance(event.get(key), list):
            result[key] = [compact_event(value) for value in event[key]]
    for key in ('actual_work','operation_audits','limits','rms_audit','quadratic_step_audit',
            'support_metric_step_audit','step_audit','basis_audit','basis_certificate_ref','row_basis_ref'):
        if key in event:result[key]=event[key]
    return result


def validate_spec(spec):
    from run_d92_group_balanced_support_metric_joint_probe import validate_spec as validate
    return validate(spec)


def verify_marker(path, spec, row):
    from run_d92_group_balanced_support_metric_joint_probe import verify_marker as verify
    return verify(path, spec, row)


def _empty_work():
    return dict.fromkeys(tuple(WORK_SUM_KEYS)+tuple(WORK_MAX_KEYS),0)


def _work_merge(target, source):
    """Merge actual numeric ledger entries under the explicit core SUM/MAX ABI."""
    check(isinstance(source, Mapping) and set(source)==WORK_KEYS,
        'Complete declared actual core work ledger required')
    for key,value in source.items():
        check(key in WORK_SUM_KEYS or key in WORK_MAX_KEYS,'Undeclared actual core work: '+str(key))
        check(isinstance(key,str) and type(value) in (int,float) and math.isfinite(value) and value>=0,
            'Invalid measured core work: '+str(key))
        if key in WORK_MAX_KEYS:
            target[key]=max(target.get(key,0),value)
        else:
            target[key]=target.get(key,0)+value


def _account_work(result, audit, phase):
    check(isinstance(audit,Mapping) and isinstance(audit.get('actual_work'),Mapping)
        and isinstance(audit.get('operation_audits'),(tuple,list)), 'Complete actual core ledger required')
    _work_merge(result[phase+'_actual_work'],audit['actual_work'])
    _work_merge(result['actual_work'],audit['actual_work'])
    for key,value in audit['actual_work'].items():
        result[key]=max(result[key],value) if key in WORK_MAX_KEYS else result[key]+value


def build_row_basis(*,prototype_frame,resources,context,state_callback=None):
    """One explicit frozen-Q factory owner; never uses support records or labels."""
    validate_resources(resources);ledger=candidate_core._Ledger()
    coords=dict(context,split_id=None,scope='row_frozen_ground_geometry',fold=None,trial=None,
        parent_k=None,train_k=None,state='ROW_GROUP_BALANCED_SUPPORT_METRIC_BASIS')
    namespace=json.dumps(json_native(coords),sort_keys=True,separators=(',',':'))
    basis=None
    try:
        basis=candidate_core._invoke(ledger,'basis',build_support_metric_basis,Q=prototype_frame.Q,
            max_integer_bits=resources['max_integer_bits'],max_fraction_operations=resources['max_fraction_operations'])
        arrays=candidate_core._basis_arrays(basis)
        arrays.update(rank=np.asarray(basis.rank,dtype=np.int64))
        ref=None if state_callback is None else state_callback(namespace+'/basis',arrays)
        encoded=basis.certificate_json();certificate=dict(path=None,utf8_bytes=len(encoded.encode('utf-8')),
            file_bytes=None,archive_seconds=None,scope='ORDINARY_METHOD_ARTIFACT_NOT_AUTHORIZATION')
        archive_root=getattr(state_callback,'root',None)
        if archive_root is not None:
            path=Path(archive_root)/'basis_certificate.json'
            certificate_started=time.perf_counter()
            with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(encoded)
            certificate.update(path='basis_certificate.json',file_bytes=path.stat().st_size,
                archive_seconds=time.perf_counter()-certificate_started)
        audit=dict(schema='d92_support_metric_row_basis_v1',method=METHOD,status='COMPLETE',
            row_basis_ref=ref,row_basis_rank=basis.rank,basis_certificate=certificate,
            row_basis_construction_count=1,construction_audit=basis.audit_dict(),
            input_role='FROZEN_GROUND_Q_ONLY_NO_SUPPORT_TEACHER',context=coords,**ledger.audit())
        if ref is None:audit['in_memory_certificate_json']=encoded
        return basis,json_native(audit)
    except Exception as exc:
        failed=exc.audit_dict() if callable(getattr(exc,'audit_dict',None)) else getattr(exc,'audit',{})
        ref=None;arrays=dict(getattr(exc,'arrays',{}));archive_error=None
        if basis is not None:arrays.update(candidate_core._basis_arrays(basis))
        if arrays and state_callback is not None:
            try:ref=getattr(state_callback,'failure',state_callback)(namespace+'/basis_failure',arrays)
            except Exception as secondary:archive_error=str(secondary)
        audit=dict(status='TECHNICAL_FAILURE',failed_operation='ROW_BASIS_CONSTRUCTION',
            failure_code=getattr(exc,'code',str(exc)),failed_basis_audit=failed,failure_state_ref=ref,
            context=coords,**ledger.audit())
        if archive_error is not None:audit['failure_archive_error']=archive_error
        if basis is not None:audit['completed_basis_certificate_json']=basis.certificate_json()
        if hasattr(exc,'exact_state'):
            audit['basis_failure_exact_state']=candidate_core.basis_module._exact_encode(candidate_core._plain(exc.exact_state))
        error=candidate_core.GroupBalancedSupportMetricFailure(getattr(exc,'code',str(exc)),audit,arrays)
        error.registration_context=json_native(audit);raise error from exc


def _basis_refs(value,owner):
    """Archive the exact certificate once; bind repeated core descriptors to it."""
    if isinstance(value,Mapping):
        result={key:_basis_refs(item,owner) for key,item in value.items() if key!='basis_certificate'}
        if 'basis_certificate' in value:
            result['basis_certificate_ref']=deepcopy(owner['basis_certificate'])
            result['row_basis_ref']=deepcopy(owner['row_basis_ref'])
        return result
    if isinstance(value,(tuple,list)):return [_basis_refs(item,owner) for item in value]
    return value


def _stage_account(result, audit):
    check(audit.get('final_head_complete') is True, 'Complete final candidate head required')
    check(type(audit.get('optimizer_steps')) is int and 0<=audit['optimizer_steps']<=1,
        'One GGN accepted-update ceiling required')
    trials=audit.get('trials');check(isinstance(trials,(tuple,list)) and len(trials)==audit.get('trial_count')
        and len(trials)<=12,'Actual trial ledger incomplete')
    accepted=sum(row.get('accepted') is True for row in trials)
    check(accepted==audit['optimizer_steps'],'Accepted update/trial ledger differs')
    steps=sum(row.get('operation')=='support_metric_step' for row in audit['operation_audits'])
    check(steps<=1,'One GGN direction ceiling required')
    result['candidate_stage_count']+=1
    result['final_candidate_head_fit_count']+=1
    result['optimizer_steps']+=audit['optimizer_steps']
    result['trial_count']+=len(trials)
    result['accepted_trial_count']+=accepted
    result['rejected_trial_count']+=len(trials)-accepted
    result['ggn_step_count']+=steps
    rank=audit.get('effective_parameter_rank')
    check(type(rank) is int and 0<=rank<=5,'Actual effective U parameter rank required')
    result['ggn_parameter_direction_count']+=rank*steps
    result['peak_effective_adapter_rank']=max(result['peak_effective_adapter_rank'],rank)
    value=audit.get('resident_numeric_state_bytes')
    check(type(value) is int and value>=0,'Actual retained candidate numeric bytes required')
    result['peak_resident_numeric_state_bytes']=max(result['peak_resident_numeric_state_bytes'],value)
    _account_work(result,audit,'stage')


def _score_candidate_records(state, features):
    rows=[];work=_empty_work();audits=[];started=time.perf_counter()
    for index in range(len(features['z_id'])):
        one={key:value[index:index+1] for key,value in features.items()}
        score,audit=state.score_with_audit(**one)
        check(np.asarray(score).shape==(1,len(state.classes)),'Single physical score shape mismatch')
        rows.append(np.asarray(score)[0])
        check(isinstance(audit.get('actual_work'),Mapping),'Actual score ledger required')
        _work_merge(work,audit['actual_work']);audits.append(audit)
    values=np.asarray(rows,dtype=np.float64).reshape(len(rows),len(state.classes))
    check(np.isfinite(values).all(),'Nonfinite candidate held scores')
    return values,dict(physical_record_count=len(rows),actual_score_calls=len(rows),
        score_seconds=time.perf_counter()-started,actual_work=work,operation_audits=[
            dict(physical_position=index,audit=audit) for index,audit in enumerate(audits)],
        aggregation='SUM_WORK_MAX_PEAK_PER_PHYSICAL_SINGLE_RECORD')


def training_text(record):
    """CVS-style text uses actual event values; unavailable components stay N/A."""
    obj=record.get('objective',record.get('audit',{}))
    if not isinstance(obj,Mapping):obj={}
    final=obj.get('final_objective')
    if isinstance(final,Mapping):obj=dict(final,**obj)
    gates=[item['audit'] for item in obj.get('operation_audits',())
        if item.get('operation')=='gate_forward' and isinstance(item.get('audit'),Mapping)]
    latest=gates[-1] if gates else {}
    def shown(value):return 'N/A' if value is None else str(value)
    return ('GROUP_BALANCED_SUPPORT_METRIC '+str(record.get('event'))+' state='+str(record.get('state'))+
        ' RMSCE='+shown(obj.get('RMSCE',obj.get('final_objective',{}).get('RMSCE') if isinstance(obj.get('final_objective'),Mapping) else None))+
        ' loss_total='+shown(obj.get('loss_total'))+' loss_proximal='+shown(obj.get('loss_proximal'))+
        ' ridge_weight='+str(FROZEN_CONFIG['ridge_coefficient'])+' damping='+str(FROZEN_CONFIG['damping'])+
        ' temperature=1 nominal_adapter_coordinates='+str(FROZEN_CONFIG['nominal_coordinates'])+
        ' gradient_trainable_adapter_coordinates='+shown(obj.get('trainable_parameter_count'))+
        ' effective_U_rank='+shown(record.get('effective_parameter_rank',obj.get('effective_parameter_rank')))+
        ' coordinate_scope=PHYSICAL_U_ONLY'+
        ' theta='+shown(obj.get('theta'))+' anchor_theta='+shown(obj.get('anchor_theta'))+
        ' gradient='+shown(record.get('gradient_norm',obj.get('gradient_norm')))+
        ' direction='+shown(record.get('direction_norm',obj.get('direction_norm')))+
        ' GGN_multiplier='+shown(obj.get('quadratic_multiplier'))+
        ' class_CE_means='+shown(record.get('class_ce_means'))+
        ' class_CE_sums='+shown(record.get('class_ce_sums'))+
        ' class_CE_counts='+shown(record.get('class_ce_counts'))+
        ' class_weights='+shown(record.get('class_weights'))+
        ' class_weight_scope=RMS_CLASS_MEAN_DERIVATIVE_AT_THIS_EVALUATION'+
        ' metric_status='+shown(record.get('step_audit',obj.get('support_metric_step_audit',{})).get('status'))+
        ' lr='+shown(record.get('step_size'))+' accepted='+shown(record.get('accepted'))+
        ' observed_increase='+shown(record.get('observed_objective_increase'))+
        ' real_Armijo_rhs='+shown(record.get('armijo_rhs'))+
        ' real_Armijo_holds='+shown(record.get('real_inequality_holds'))+
        ' objective_seconds='+shown(obj.get('objective_seconds'))+' fit_seconds='+shown(obj.get('fit_seconds'))+
        ' resident_bytes='+shown(obj.get('resident_numeric_state_bytes'))+
        ' actual_updated_coordinates='+shown(obj.get('actual_updated_coordinate_count'))+
        ' gate_heads_in_event='+str(len(gates))+' gate_measurement_scope=LAST_COMPLETED_GATE_OPERATION_IN_THIS_EVENT'+
        ' weighted_gate_objective='+shown(latest.get('objective'))+
        ' weighted_gate_logistic='+shown(latest.get('logistic_loss_sum'))+
        ' gate_ridge='+shown(latest.get('ridge_penalty'))+' gate_barrier='+shown(latest.get('barrier_term'))+
        ' gate_old_mass='+shown(latest.get('old_supervision_mass'))+' gate_new_mass='+shown(latest.get('new_supervision_mass'))+
        ' gate_weight_constructions='+shown(latest.get('weight_constructions_completed'))+
        ' gate_weight_seconds='+shown(latest.get('weight_construction_seconds'))+
        ' gate_canonical_residual='+shown(latest.get('canonical_residual'))+
        ' gate_intercept_residual='+shown(latest.get('intercept_residual'))+
        ' gate_wall_seconds='+shown(latest.get('wall_seconds'))+
        ' current_stage_fitted_head_parameters='+shown(record.get('current_stage_fitted_head_parameters'))+
        ' head_parameter_scope=CURRENT_STAGE_RIDGE_OR_NEW_RIDGE_PLUS_NEWTON_GATE_EXCLUDES_REUSED_ACTUAL_B'+
        ' radius='+str(FROZEN_CONFIG['radius'])+
        ' source_validation=N/A objective_scope=INNER_SUPPORT_TRAINING_NOT_VALIDATION')


def _small_measurements(arrays):
    names={'rms_class_ce_means':'class_ce_means','rms_class_ce_sums':'class_ce_sums',
        'rms_class_ce_counts':'class_ce_counts','rms_class_weights':'class_weights',
        'theta':'theta','gradient':'gradient','direction':'direction','physical_gram':'physical_gram',
        'fisher':'metric_fisher','metric':'metric_M','hessian':'metric_H'}
    result={}
    for source,target in names.items():
        if source in arrays and arrays[source] is not None:
            value=np.asarray(arrays[source])
            if value.size<=26 and np.isfinite(value).all():result[target]=value.tolist()
    if 'new_alpha' in arrays and 'new_intercept' in arrays and 'gate_alpha' in arrays and 'gate_b' in arrays:
        result['current_stage_fitted_head_parameters']=sum(int(np.asarray(arrays[k]).size)
            for k in ('new_alpha','new_intercept','gate_alpha','gate_b'))
    elif 'alpha' in arrays and 'intercept' in arrays:
        result['current_stage_fitted_head_parameters']=sum(int(np.asarray(arrays[k]).size) for k in ('alpha','intercept'))
    return result


def probe_group_balanced_support_metric_joint(*, z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids,
                       classes, old_classes, max_newton_iterations, max_line_search_trials, max_factor_buffer_bytes,
                       max_integer_bits,max_fraction_operations,max_secular_iterations,
                       log_callback=None, event_callback=None, context=None, state_callback=None, ground_head=None,
                       prediction_callback=None, prototype_frame=None,support_metric_basis=None,row_basis_owner=None):
    started = time.perf_counter()
    support_metric_resources = dict(max_newton_iterations=max_newton_iterations,max_line_search_trials=max_line_search_trials,
        max_factor_buffer_bytes=max_factor_buffer_bytes,max_integer_bits=max_integer_bits,
        max_fraction_operations=max_fraction_operations,max_secular_iterations=max_secular_iterations)
    validate_resources(support_metric_resources)
    check(prototype_frame is not None, 'Explicit frozen prototype frame required')
    context = dict(context or {})
    # A direct synthetic/in-memory invocation creates its own inheritance scope.
    # The executable cache entry requires explicit launch-owner run/row binding.
    token = uuid.uuid4().hex
    context.setdefault('run_id', 'in_memory_support_probe_'+token)
    context.setdefault('row_id', 'in_memory_support_parent_'+token)
    check(all(isinstance(context[key], str) and bool(context[key]) for key in ('run_id', 'row_id')),
        'Explicit nonempty run/row inheritance binding required')
    b, a, labels, ids, canonical, _, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    check(bool(old), 'Old registry required'); classes, old = list(canonical), sorted(old)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: support_ids[i]))
    raw = {key: np.asarray(value)[order] for key, value in zip(BRANCHES, (z_id, fft, t_emb, f_emb, pa_local))}
    label_by_id = {pid: classes[int(y)] for pid, y in zip(ids, labels)}
    is_old = np.asarray([label_by_id[pid] in old for pid in ids]); folds = 0 if k == 1 else min(k, 3)
    result = dict(schema=SCHEMA, method=METHOD, inheritance_binding=dict(context), support_metric_resources=support_metric_resources,
        classes=classes, old_classes=old, support_count=len(ids), old_class_count=len(old),
        new_class_count=len(classes)-len(old), fold_count=folds, physical_fold_assignment=[],
        numerical=interaction._diagonal_stats(b, a), folds=[], oof=None, oneshot_proxy=None, full_support=None,
        **dict.fromkeys(COUNTERS[4:], 0), persistent_state_bytes=0,
        actual_work=_empty_work(),row_basis_actual_work=_empty_work(),preparation_actual_work=_empty_work(),
        stage_actual_work=_empty_work(),score_actual_work=_empty_work(),
        prototype_geometry_audit=json_native(dict(prototype_frame.audit)),
        nominal_adapter_parameter_count=5,structural_predict_actual_work=None,
        work_aggregation='SUM/MAX',
        row_basis_construction_status='PENDING_EXPLICIT_FROZEN_Q_FACTORY',
        row_basis_namespace=dict(run_id=context['run_id'],row_id=context['row_id'],split_id=None,scope='row_frozen_ground_geometry',
            fold=None,trial=None,parent_k=None,train_k=None,state='ROW_GROUP_BALANCED_SUPPORT_METRIC_BASIS'),
        structural_predict_work_unavailable_reason='PUBLIC_PREDICT_RETURNS_NO_INTERNAL_LEDGER',
        path_definitions=dict(R0='ORIGINAL_FEATURE_LOCAL_RIDGE_B_OLD_AND_C_FULL_REGISTRY_REFIT',
            R_GROUP_BALANCED_SUPPORT_METRIC_seq='ACTUAL_GROUP_BALANCED_SUPPORT_METRIC_B_TO_C_SEQUENTIAL_INHERITANCE_WITH_FROZEN_B_OLD_CONDITIONAL'),
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if support_metric_basis is None:
        support_metric_basis,row_basis_owner=build_row_basis(prototype_frame=prototype_frame,
            resources=support_metric_resources,context=context,state_callback=state_callback)
        result['row_basis_construction_count']=1
        _account_work(result,row_basis_owner,'row_basis')
    check(isinstance(support_metric_basis,candidate_core.basis_module.SupportMetricBasis),
        'Explicit immutable support metric basis required')
    check(isinstance(row_basis_owner,Mapping),'Explicit row basis owner metadata required')
    check(row_basis_owner.get('status')=='COMPLETE' and row_basis_owner.get('row_basis_rank')==support_metric_basis.rank
        and all(row_basis_owner.get('context',{}).get(key)==context[key] for key in ('run_id','row_id')),
        'Row frozen basis owner identity differs')
    result.update(row_basis_ref=deepcopy(row_basis_owner['row_basis_ref']),row_basis_rank=support_metric_basis.rank,
        basis_certificate=deepcopy(row_basis_owner['basis_certificate']),row_basis_audit=deepcopy(row_basis_owner),
        row_basis_construction_status='COMPLETE')
    positions = [np.flatnonzero(labels == i) for i in range(len(classes))]
    assignments = np.full(len(ids), -1, dtype=int)
    if folds:
        for indices in positions: assignments[indices] = np.arange(k) % folds
        result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=label_by_id[pid], fold=int(assignments[i])) for i, pid in enumerate(ids)]
    completed = []

    def path(keep, scope, index):
        reuse = classes == old; has_held = bool((~keep).any())
        entry = dict(scope=scope, fold=index if scope == 'support_oof' else None,
            trial=index if scope == 'support_oneshot_proxy' else None, parent_k=k,
            train_k=int(keep.sum())//len(classes), held_k=int((~keep).sum())//len(classes),
            b_training_ids=[pid for i, pid in enumerate(ids) if keep[i] and is_old[i]],
            c_training_ids=[pid for i, pid in enumerate(ids) if keep[i]],
            b_ids=[pid for i, pid in enumerate(ids) if not keep[i] and is_old[i]],
            c_ids=[pid for i, pid in enumerate(ids) if not keep[i]], b_classes=old, c_classes=classes,
            held_labels={pid: label_by_id[pid] for i, pid in enumerate(ids) if not keep[i]},
            row_basis_ref=deepcopy(row_basis_owner['row_basis_ref']),row_basis_rank=support_metric_basis.rank,
            stages=[], preparations=[], candidate_stages=[], training_events=[], c_reuses_b0=reuse,
            c_reuses_b_candidates=reuse, paths={}, outer_features_state_ref=None)
        coords = dict(context or {}, **{key: entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')})
        bases, prepared, candidates = {}, {}, {};measurements={}
        def emit_event(row, stage):
            value = _basis_refs(dict(row),row_basis_owner)
            key=value.get('state_ref',{}).get('key') if isinstance(value.get('state_ref'),Mapping) else None
            observed=dict(measurements.get((stage,key),{}))
            if str(row.get('event','')).endswith('GRADIENT'):
                observed=dict(measurements.get((stage,'initial'),{}),**observed)
            if str(row.get('event','')).endswith('FINAL'):
                final=value.get('audit',{});accepted=[r for r in final.get('trials',[]) if r.get('accepted')]
                source=accepted[0]['state_ref'] if accepted else final.get('initial_state_ref')
                if isinstance(source,Mapping):observed=dict(measurements.get((stage,source.get('key')),{}),**observed)
            value.update(observed,effective_parameter_rank=support_metric_basis.rank,
                nominal_parameter_count=5,class_statistics_scope='CURRENT_ACTUAL_OOF_FORWARD',
                class_weights_unavailable_reason=None if 'class_weights' in observed else 'NO_GRADIENT_WEIGHTS_AT_THIS_FORWARD')
            # Retain the actually executed candidate core schema separately from row schema.
            value['executed_core_schema']=row.get('schema');value['executed_core_method']=row.get('method')
            value.update({key: item for key, item in coords.items() if key != 'trial'},
                schema=SCHEMA, method=METHOD, outer_trial=coords['trial'], state=stage, source_validation=None,
                source_validation_reason='SOURCE_ACCESS_FORBIDDEN', objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
            entry['training_events'].append(value)
            if event_callback: event_callback(value)
        def archive_callback(stage):
            if state_callback is None: return None
            namespace = json.dumps(json_native(dict(coords, state=stage)), sort_keys=True, separators=(',', ':'))
            def save(key,arrays):
                measurements[(stage,key)]=_small_measurements(arrays)
                # Only the core's actual failed-state keys may retain nonfinite values.
                # Accepted/completed states still use strict successful archival.
                saver=getattr(state_callback,'failure',state_callback) if key in ('failure','preparation_failure') else state_callback
                return saver(namespace+'/'+key,arrays)
            return save
        try:
            outer_archive = archive_callback('OUTER_SUPPORT_HELD')
            if outer_archive is not None and has_held:
                entry['outer_features_state_ref'] = outer_archive('features', {key: np.asarray(value[~keep], dtype=np.float64) for key, value in raw.items()})
            for name, train, held, registry in (('B0', keep & is_old, ~keep & is_old, old), ('C0', keep, ~keep, classes)):
                if name == 'C0' and reuse: bases[name] = bases['B0']; continue
                tids = [pid for i, pid in enumerate(ids) if train[i]]
                ys = np.asarray([registry.index(label_by_id[pid]) for pid in tids], dtype=np.int64)
                tick = time.perf_counter()
                state = fit_branch_local_ridge(**{key: value[train] for key, value in raw.items()},
                    support_labels=ys, support_ids=tids, classes=registry, old_classes=old, arm='local_ridge')
                top = state.audit_dict(); audit = top['final_fit']
                audit.update(coords, state=name, training_physical_ids=tids, all_states_estimated_from_trainfold_only=True,
                    held_physical_count=int(held.sum()), class_count=len(registry), learning_rate=None,
                    learning_rate_reason='Closed-form baseline; no optimizer', source_validation=None,
                    source_validation_reason='SOURCE_ACCESS_FORBIDDEN', score_seconds=0., persistent_state_bytes=top['persistent_state_bytes'])
                ref = archive_callback(name)
                audit['final_state_ref'] = None if ref is None else ref('baseline', dict(
                    original_train_b=state.support_background, original_train_a=state.support_auxiliary,
                    alpha=state.alpha, reference_kernel=state.reference_kernel,
                    reference_self=np.asarray(state.reference_self), center_mean=state.center_mean,
                    center_grand=np.asarray(state.center_grand),
                    tau=np.asarray([] if state.bandwidth_tau is None else [state.bandwidth_tau], dtype=np.float64),
                    gamma=np.asarray([] if state.trace_scale is None else [state.trace_scale], dtype=np.float64)))
                entry['stages'].append(audit); result['baseline_head_fit_count'] += 1
                result['baseline_factorization_count'] += audit['factorization_calls']
                result['baseline_triangular_solve_count'] += 2*audit['factorization_calls']+audit['effective_degrees_of_freedom_extra_triangular_solves']
                audit['triangular_count_scope']='SUCCESSFULLY_RETURNED_ORIGINAL_BASELINE_ONLY; FAILED_ORIGINAL_CONTROL_WORK_NOT_INFERRED'
                bases[name] = dict(state=state, train=train, held=held, registry=registry, labels=ys, ids=tids, scores=np.empty((0, len(registry))))
                if held.any():
                    score_start = time.perf_counter(); result['final_score_evaluation_count'] += 1
                    result['final_score_physical_count'] += int(held.sum())
                    bases[name]['scores'] = state.score(**{key: value[held] for key, value in raw.items()})
                    audit['score_seconds'] = time.perf_counter()-score_start
                audit['fit_and_score_seconds'] = time.perf_counter()-tick
                if log_callback: log_callback(dict(event='BASE_FIT', **audit))
            for prep_name, name, mode in (('B', 'B_GROUP_BALANCED_SUPPORT_METRIC', 'B'), ('C', 'C_GROUP_BALANCED_SUPPORT_METRIC_seq', 'C_seq')):
                if prep_name == 'C' and reuse: candidates[name] = candidates['B_GROUP_BALANCED_SUPPORT_METRIC']; continue
                base = bases[prep_name+'0']
                prepared[prep_name] = prepare_group_balanced_support_metric_joint_training(**{key: value[base['train']] for key, value in raw.items()},
                    support_labels=base['labels'], support_ids=base['ids'], classes=base['registry'], old_classes=old,
                    inherited=None if prep_name == 'B' else candidates['B_GROUP_BALANCED_SUPPORT_METRIC']['state'], context=dict(coords, stage=prep_name),
                    prototype_frame=prototype_frame,
                    support_metric_basis=support_metric_basis,
                    **support_metric_resources,
                    log_callback=lambda row, current=prep_name: emit_event(row, current+'_prepare'),
                    state_callback=archive_callback(prep_name+'_prepare'))
                prep = _basis_refs(prepared[prep_name].audit_dict(),row_basis_owner)
                prep.update(coords, state=prep_name, training_physical_ids=base['ids'],
                    train_physical_count=len(base['ids']), class_count=len(base['registry']),
                    inherited_adapter_from=None if prep_name == 'B' else 'B_GROUP_BALANCED_SUPPORT_METRIC', objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
                entry['preparations'].append(prep); result['candidate_preparation_count']+=1
                _account_work(result,prep,'preparation')
                if log_callback: log_callback(dict(event='GROUP_BALANCED_SUPPORT_METRIC_PREPARATION', **prep))
                state = fit_group_balanced_support_metric_joint_local_ridge(prepared[prep_name], mode=mode,
                    log_callback=lambda row, current=name: emit_event(row, current), state_callback=archive_callback(name))
                audit = _basis_refs(state.audit_dict(),row_basis_owner)
                audit['executed_core_schema']=audit.get('schema');audit['executed_core_method']=audit.get('method')
                audit['persistent_state_bytes'] = audit['resident_numeric_state_bytes']
                audit.update(coords, state=name, preparation_ref=prep_name, mode=mode,
                    training_physical_ids=base['ids'], train_physical_count=len(base['ids']), class_count=len(base['registry']),
                    held_physical_count=int(base['held'].sum()), source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
                    score_seconds=0., score_workload=None, score_workload_unavailable_reason='NO_OUTER_HELD_INFERENCE' if not base['held'].any() else None,
                    objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION')
                entry['candidate_stages'].append(audit); candidates[name] = dict(state=state, scores=np.empty((0, len(base['registry']))))
                _stage_account(result,audit)
                if base['held'].any():
                    score_start = time.perf_counter(); result['final_score_evaluation_count'] += int(base['held'].sum())
                    result['final_score_physical_count'] += int(base['held'].sum())
                    held_features={key:value[base['held']] for key,value in raw.items()}
                    candidates[name]['scores'], audit['score_workload'] = _score_candidate_records(state,held_features)
                    _account_work(result,audit['score_workload'],'score')
                    audit['score_seconds'] = time.perf_counter()-score_start
                    if name=='C_GROUP_BALANCED_SUPPORT_METRIC_seq':
                        tick=time.perf_counter();preds=[str(state.predict(**{key:value[i:i+1] for key,value in held_features.items()})[0]) for i in range(int(base['held'].sum()))]
                        candidates[name]['predictions']=dict(zip(entry['c_ids'],map(str,preds)))
                        result['structural_predict_evaluation_count']+=int(base['held'].sum());result['structural_predict_physical_count']+=int(base['held'].sum())
                        elapsed=time.perf_counter()-tick;result['structural_predict_seconds']+=elapsed
                        audit['structural_predict_seconds']=elapsed;audit['structural_predict_workload']=None
                        audit['structural_predict_workload_unavailable_reason']='PUBLIC_PREDICT_RECOMPUTES_GEOMETRY_WITHOUT_RETURNING_AUDIT; SCORE_WORKLOAD_COVERS_ONLY_SCORE_CALL'
                if log_callback: log_callback(dict(event='CANDIDATE_FIT', **audit))
            ground=None
            if ground_head is not None and has_held:
                check(set(ground_head.classes)==set(old) and len(ground_head.classes)==6,'Ground A must compete over original six old columns')
                ahold=(~keep)&is_old;tick=time.perf_counter();rows=[]
                # Native float32 source head receives only this path's legal old-held records.
                for vector in raw['z_id'][ahold]:
                    scores=ground_head.score(z_id=torch.tensor([np.asarray(vector,dtype=np.float32).tolist()],dtype=torch.float32),
                        feature_contract=ground_head.metadata.feature_contract)
                    rows.append(scores.detach().cpu().tolist()[0])
                av=np.asarray(rows,dtype=np.float32);check(av.shape==(int(ahold.sum()),6) and np.isfinite(av).all(),'Native Ground A score shape/finite mismatch')
                result['ground_A_score_evaluation_count']+=len(rows);result['ground_A_score_physical_count']+=len(rows)
                elapsed=time.perf_counter()-tick;result['ground_A_inference_seconds']+=elapsed
                ground=dict(physical_ids=entry['b_ids'],classes=list(ground_head.classes),scores=av.tolist(),
                    predictions=[ground_head.classes[int(j)] for j in np.argmax(av,axis=1)],score_dtype='float32',
                    inference_seconds=elapsed,scope='CURRENT_LEGAL_OLD_HELD_ONLY_ORIGINAL_SIX_CLASS_COMPETITION')
                ref=archive_callback('A_GROUND_FIXED_PREDICTIONS')
                ground['scores_state_ref']=None if ref is None else ref('scores',dict(scores=av))
            entry['ground_A']=ground
            entry['actual_A_unavailable_reason']=None if ground is not None else ('K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if not has_held else 'NO_EXPLICIT_MATCHED_SOURCE_ONLY_GROUND_PACKET')
            common = {key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')}
            evidence = dict(R0=dict(common, b_scores=bases['B0']['scores'].tolist(), c_scores=bases['C0']['scores'].tolist()),
                R_GROUP_BALANCED_SUPPORT_METRIC_seq=dict(common, b_scores=candidates['B_GROUP_BALANCED_SUPPORT_METRIC']['scores'].tolist(), c_scores=candidates['C_GROUP_BALANCED_SUPPORT_METRIC_seq']['scores'].tolist()))
            if has_held:
                # new0 reuses the exact B object, whose raw canonical score argmax is authoritative.
                evidence['R_GROUP_BALANCED_SUPPORT_METRIC_seq']['c_predictions']=candidates['C_GROUP_BALANCED_SUPPORT_METRIC_seq'].get('predictions',
                    {pid:old[int(np.argmax(row))] for pid,row in zip(entry['c_ids'],candidates['B_GROUP_BALANCED_SUPPORT_METRIC']['scores'])})
            if has_held:
                fixed=[]
                if ground is not None:fixed.append(dict(stream='A',**ground))
                for name,value in evidence.items():
                    value['ground_A']=ground
                    for side in ('b','c'):
                        values=np.asarray(value[side+'_scores']);registry=value[side+'_classes']
                        predictions=[value['c_predictions'][pid] for pid in value['c_ids']] if side=='c' and 'c_predictions' in value else [min(registry[j] for j in np.flatnonzero(r==np.max(r))) for r in values]
                        fixed.append(dict(stream=name+'_'+side.upper(),physical_ids=value[side+'_ids'],classes=registry,
                            scores=values.tolist(),predictions=predictions))
                # The persistence callback sees no held labels and flushes before any diagnostic join.
                for frozen in fixed:
                    if prediction_callback:prediction_callback(json_native(dict(coords,outer_scope=coords['scope'],**frozen,status='FIXED_BEFORE_SUPPORT_TRUTH_JOIN')))
                ref=archive_callback('FIXED_SUPPORT_PREDICTIONS')
                entry['fixed_prediction_state_ref']=None if ref is None else ref('scores',
                    {name+'_'+side:np.asarray(value[side+'_scores'],dtype=np.float64) for name,value in evidence.items() for side in ('b','c')})
            entry['prediction_status'] = 'FIXED_BEFORE_SUPPORT_TRUTH_JOIN' if has_held else 'NO_HELD_PREDICTIONS'
            if has_held:
                fresh = assess_paths(evidence, old)
                entry['paths'] = {name: dict(b_scores=value['b_scores'], c_scores=value['c_scores'],c_predictions=value.get('c_predictions',{}), **fresh[name]) for name, value in evidence.items()}
            else:
                entry['paths'] = {name: dict(b_scores=[], c_scores=[], diagnostic=None, held_comparisons=None,
                    metrics=dict.fromkeys(METRICS), evidence_scope='FULL_SUPPORT_HEAD_NO_INDEPENDENT_HELD') for name in PATHS}
            final_audit = candidates['C_GROUP_BALANCED_SUPPORT_METRIC_seq']['state'].audit_dict()
            entry['deployment_C_state_bytes'] = dict(R_GROUP_BALANCED_SUPPORT_METRIC_seq=final_audit['resident_numeric_state_bytes'])
            entry['minimum_deployment_numeric_state_bytes'] = dict(R_GROUP_BALANCED_SUPPORT_METRIC_seq=final_audit['deployment_numeric_state_bytes'])
            result['persistent_state_bytes'] = max(result['persistent_state_bytes'], entry['deployment_C_state_bytes']['R_GROUP_BALANCED_SUPPORT_METRIC_seq'])
        except Exception as exc:
            audit_method = getattr(exc, 'audit_dict', None)
            failed_fit = audit_method() if callable(audit_method) else getattr(exc, 'audit', None)
            arrays = getattr(exc, 'arrays', None)
            arrays_method = getattr(exc, 'to_arrays', None)
            if arrays is None and callable(arrays_method):
                arrays = arrays_method()
            failed_arrays_ref = failed_fit.get('failure_state_ref') if isinstance(failed_fit, Mapping) else None
            if failed_arrays_ref is None and isinstance(arrays, Mapping) and arrays and state_callback is not None:
                saver = getattr(state_callback, 'failure', state_callback)
                namespace = json.dumps(json_native(dict(coords, state='FAILED_FIT')), sort_keys=True, separators=(',', ':'))
                failed_arrays_ref = saver(namespace+'/failed_numeric_state', arrays)
            measured_failed_counters = failed_fit.get('actual_work') if isinstance(failed_fit, Mapping) else None
            exc.registration_context = _safe(_basis_refs(dict(current_path=entry, failed_fit=failed_fit,
                failed_fit_audit_unavailable_reason=None if failed_fit is not None else 'EXCEPTION_HAS_NO_MEASURED_FIT_AUDIT',
                failed_numeric_state_ref=failed_arrays_ref, failed_fit_counters=measured_failed_counters or None,
                counters_scope='COMPLETED_BASELINES_PREPARATIONS_AND_CANDIDATE_STAGES_BEFORE_FAILURE',
                failed_fit_counters_scope='SEPARATE_PARTIAL_FIT_AUDIT_NOT_ADDED_TO_COMPLETED_COUNTERS',
                workload_complete=False, support_metric_resources=support_metric_resources,
                completed_preparations={name: value.audit_dict() for name, value in prepared.items()},
                completed_candidate_states={name: value['state'].audit_dict() for name, value in candidates.items()},
                completed_paths=completed, counters={key: result[key] for key in COUNTERS[4:]},
                row_basis_audit=row_basis_owner,row_basis_actual_work=result['row_basis_actual_work'],
                completed_actual_work=result['actual_work'], failed_actual_work=measured_failed_counters),row_basis_owner))
            raise
        result['sequence_paths'] += 1; completed.append(entry)
        return entry

    if k == 1:
        result['full_support'] = path(np.ones(len(ids), dtype=bool), 'support_full_k1', None)
    else:
        for fold in range(folds): result['folds'].append(path(assignments != fold, 'support_oof', fold))
        result['oof'] = dict(paths=pooled_assess(result['folds'], label_by_id, classes, old), aggregation='one_record_per_physical_held_id')
        trials = []
        for trial in range(k):
            keep = np.zeros(len(ids), dtype=bool)
            for indices in positions: keep[indices[trial]] = True
            trials.append(path(keep, 'support_oneshot_proxy', trial))
        result['oneshot_proxy'] = dict(trials=trials, trial_count=k, parent_mean_metrics=parent_mean(trials),
            aggregation='all_anchors_mean_within_parent_then_equal_parent', proxy_train_k=1)
    result['fit_seconds'] = time.perf_counter()-started
    return result


def compact_record(record):
    keys = ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'new_count', 'support_count',
        'old_class_count', 'new_class_count', 'fold_count', 'fit_seconds', 'persistent_state_bytes', 'heldout_unavailable_reason')+COUNTERS[4:]
    return json_native(dict({key: record[key] for key in keys},
        oof=None if record['oof'] is None else {name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=None if record['oneshot_proxy'] is None else record['oneshot_proxy']['parent_mean_metrics'],
        full_support_head_fitted=record['full_support'] is not None,
        proxy_anchor_count=0 if record['oneshot_proxy'] is None else record['oneshot_proxy']['trial_count'],
        schema=SCHEMA, method=METHOD, scope=SCOPE, query_rows_used=0, source_rows_used=0,
        work_aggregation='SUM/MAX',
        actual_work=record['actual_work'],preparation_actual_work=record['preparation_actual_work'],
        row_basis_actual_work=record['row_basis_actual_work'],row_basis_ref=record['row_basis_ref'],
        row_basis_rank=record['row_basis_rank'],basis_certificate=record['basis_certificate'],
        stage_actual_work=record['stage_actual_work'],score_actual_work=record['score_actual_work'],
        structural_predict_actual_work=None,structural_predict_work_unavailable_reason=record['structural_predict_work_unavailable_reason']))


def ground_packet_binding(packet,checkpoint_sha256,model_seed,old):
    if packet is None:return None,dict(status='N/A',reason='NO_EXPLICIT_MATCHED_SOURCE_ONLY_GROUND_PACKET')
    head=load_packet(packet);metadata=read(Path(packet)/'metadata.json')
    check(head.metadata.checkpoint_sha256==checkpoint_sha256,'Ground/cache checkpoint identity mismatch')
    check(metadata['existing_source_only_provenance'].get('model_seed')==model_seed,'Ground/cache model seed mismatch')
    check(len(head.classes)==6 and set(head.classes)==set(old),'Ground A six original classes mismatch')
    return head,dict(status='MATCHED_SOURCE_ONLY_PACKET',path=str(packet),checkpoint_sha256=checkpoint_sha256,
        model_seed=model_seed,ordered_classes=list(head.classes),head_metadata=metadata['head_metadata'],
        packet_file_bytes=sum(p.stat().st_size for p in Path(packet).iterdir() if p.is_file()),
        native_head_weight_bytes=(Path(packet)/'head_weight.float32.bin').stat().st_size,
        new_transfer_bytes=None,transfer_unavailable_reason='No transfer performed or measured by scorer')


def evaluate(*, support_features, capsule, output, config, expected_capsule_id,
             expected_checkpoint_sha256, expected_model_seed, run_id, row_id,release_commit,
             ground_summary,ground_summary_already_deployed,ground_packet=None):
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    check(isinstance(ground_summary,(str,Path)) and bool(str(ground_summary))
        and type(ground_summary_already_deployed) is bool,
        'Explicit ground-summary and boolean already-deployed flag required')
    check(set(config) == {'algorithm', 'producer_matrix', 'selection', 'support_metric_resources'} and config['algorithm'] == PROBE_CONFIG == FROZEN_CONFIG,
        'Fixed GROUP_BALANCED_SUPPORT_METRIC_JOINT code/config mismatch')
    validate_resources(config['support_metric_resources'])
    check(oid(release_commit),'Explicit actual runtime release commit required')
    check(all(isinstance(value, str) and bool(value) for value in (run_id, row_id)),
        'Explicit nonempty run/row inheritance binding required')
    validate_selection(config['selection']); manifest = read(Path(capsule)/'manifest.json')
    check(all(manifest.get('channel', {}).get(key) == value for key, value in CHANNEL.items())
        and manifest.get('scenarios') == SCENARIOS, 'Practical residual channel mismatch')
    begin = time.perf_counter()
    cache_started=time.perf_counter()
    arrays, tasks, old, producer, extraction, provenance = load_support(
        support_features=support_features, capsule=capsule, expected_capsule_id=expected_capsule_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256, expected_model_seed=expected_model_seed,
        config=dict(algorithm=CACHE_VALIDATION_CONFIG, matrix=config['producer_matrix']))
    cache_load_seconds=time.perf_counter()-cache_started
    check(len(old) == 6, 'Fixed six-old-class pilot required'); chosen = selected_tasks(tasks, config['selection'], old)
    ground_head,ground_binding=ground_packet_binding(ground_packet,expected_checkpoint_sha256,expected_model_seed,old)
    frame_started=time.perf_counter()
    frame,frame_binding=load_proto_frame_ground_geometry(ground_summary,expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_classes=old,already_deployed=ground_summary_already_deployed)
    frame_binding=json_native(frame_binding)
    frame_binding['total_geometry_binding_seconds']=time.perf_counter()-frame_started
    selected_physical={split['split_id']:list(split['support_ids']) for split,_,_ in chosen}
    binding = dict(run_id=run_id, row_id=row_id, capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256,
        model_seed=expected_model_seed, release_commit=release_commit, ground_packet=ground_packet,
        ground_summary=str(ground_summary),ground_summary_already_deployed=ground_summary_already_deployed,
        channel=deepcopy(manifest['channel']), scenarios=manifest['scenarios'])
    payload = dict(feature_cache_reused=True, checkpoint_loaded=False, native_physical_forward_count_this_run=0,
        new_source_payload_bytes=0, new_ground_statistics_bytes=0, support_feature_array_bytes=producer['feature_array_bytes'],
        support_feature_file_bytes=producer['feature_file_bytes'], cache_load_seconds=cache_load_seconds,
        ground_geometry_payload_audit=frame_binding['ground_payload_audit'],newly_generated_ground_statistics_bytes=0,
        deployment_package_bytes=None, incremental_transfer_bytes=None, native_wire_bytes=None,
        unmeasured_reason='Reader reports component file bytes separately; no native transport or deployment measured')
    startup = dict(binding, schema=SCHEMA, method=METHOD, implementation_status=IMPLEMENTATION_STATUS,
        support_metric_resources=deepcopy(config['support_metric_resources']),
        scope=SCOPE, config=config, episodes=len(chosen), producer_episodes=len(tasks),
        selected_support_physical_ids=selected_physical,
        support_features=str(support_features), provenance=provenance, payload_audit=payload,
        query_rows_used=0, source_rows_used=0, query_iq_access=False, truth_read=False,
        checkpoint_loaded=False, encoder_updated=False, adapted_state_inherited=True,
        actual_A=None, ground_A_binding=ground_binding,ground_geometry_binding=frame_binding,
        work_aggregation='SUM/MAX',
        nominal_adapter_parameter_count=5,prototype_geometry_role='FROZEN_REFERENCE_ONLY_NOT_TEACHER',
        row_basis_construction_status='PENDING_EXPLICIT_FROZEN_Q_FACTORY',
        row_basis_namespace=dict(run_id=run_id,row_id=row_id,split_id=None,scope='row_frozen_ground_geometry',
            fold=None,trial=None,parent_k=None,train_k=None,state='ROW_GROUP_BALANCED_SUPPORT_METRIC_BASIS'),
        actual_A_unavailable_reason=ground_binding.get('reason','SCORES_NOT_YET_FIXED'),
        old_teacher_scope='C_INNER_PRIOR_FROM_OLD_INNER_TRAIN; C_FINAL_PRIOR_FROM_ACTUAL_PATH_B; INNER_HELD_IS_SUPERVISION_NOT_INDEPENDENT_VALIDATION',
        source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        objective_scope='INNER_SUPPORT_TRAINING_NOT_VALIDATION',
        argv=sys.argv, pid=os.getpid(), python=sys.executable,
        hardware=dict(platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(), dtype='float64', gpu_use=False),
        triangular_rhs_count_scope='sum_per_actual_solve_columns_r; elements=n*r; dense_work_units=n*n*r_not_measured_FLOPs',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        blas_environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    out.mkdir(parents=True, exist_ok=False); write(out/'startup.json', startup)
    startup = json_native(startup)
    print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    totals = dict.fromkeys(COUNTERS, 0); peak_state = 0
    total_work=_empty_work();row_basis_work=_empty_work();prep_work=_empty_work();stage_work=_empty_work();score_work=_empty_work()
    archive = StateArchive(out); archive_status = 'INCOMPLETE'
    try:
        try:
            basis,row_basis_owner=build_row_basis(prototype_frame=frame,resources=config['support_metric_resources'],
                context=dict(run_id=run_id,row_id=row_id),state_callback=archive)
        except Exception as exc:
            failed=exc.audit_dict() if callable(getattr(exc,'audit_dict',None)) else getattr(exc,'registration_context',{})
            write(out/'probe_failed.json',json_native(dict(status='GROUP_BALANCED_SUPPORT_METRIC_JOINT_PROBE_FAILED',**binding,
                failure_phase='ROW_BASIS_CONSTRUCTION',failure_context=failed,completed_episodes=0,
                completed_episode_counters=totals,counters_scope='NO_SUPPORT_EPISODES_STARTED',
                row_basis_failed_actual_work=failed.get('actual_work'),workload_complete=False,
                support_metric_resources=config['support_metric_resources'],query_rows_used=0,source_rows_used=0)))
            raise
        write(out/'row_basis.json',row_basis_owner)
        _work_merge(row_basis_work,row_basis_owner['actual_work']);_work_merge(total_work,row_basis_work)
        totals['row_basis_construction_count']=1;totals['peak_effective_adapter_rank']=basis.rank
        for key,value in row_basis_work.items():
            totals[key]=max(totals[key],value) if key in WORK_MAX_KEYS else totals[key]+value
        with (out/'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
             (out/'compact.jsonl').open('x', encoding='utf-8') as compact, \
             (out/'compact.csv').open('x', encoding='utf-8', newline='') as csvfile, \
             (out/'fit_stages.jsonl').open('x', encoding='utf-8') as stages, \
             (out/'training_events.jsonl').open('x', encoding='utf-8') as events, \
             (out/'training_events_compact.jsonl').open('x', encoding='utf-8') as compactevents, \
             (out/'fixed_predictions.jsonl').open('x',encoding='utf-8') as predictions, \
             (out/'training.log').open('x', encoding='utf-8') as textlog:
            textlog.write('STARTUP '+json.dumps(startup, allow_nan=False)+'\n'); textlog.flush()
            textlog.write('ROW_BASIS '+json.dumps(row_basis_owner,allow_nan=False)+'\n');textlog.flush()
            writer = None
            for split, positions, labels in chosen:
                def log(stage):
                    row = dict(compact_event(stage), schema=SCHEMA, method=METHOD, split_id=split['split_id'])
                    line = json.dumps(row, allow_nan=False)
                    stages.write(line+'\n'); stages.flush(); textlog.write(line+'\n'); textlog.flush(); print(line, flush=True)
                def event(row):
                    full = json_native(dict(row, split_id=split['split_id']))
                    events.write(json.dumps(full, allow_nan=False)+'\n'); events.flush()
                    small = compact_event(full); line = json.dumps(small, allow_nan=False)
                    compactevents.write(line+'\n'); compactevents.flush()
                    detailed=training_text(full)
                    textlog.write(detailed+'\n');textlog.write('GROUP_BALANCED_SUPPORT_METRIC_JOINT_TRAINING '+line+'\n'); textlog.flush(); print(detailed,flush=True);print('GROUP_BALANCED_SUPPORT_METRIC_JOINT_TRAINING '+line, flush=True)
                def fixed(row):
                    row=dict(row,split_id=split['split_id']);check('held_labels' not in row,'Truth leaked into fixed predictions')
                    predictions.write(json.dumps(json_native(row),allow_nan=False)+'\n');predictions.flush()
                try:
                    audit = probe_group_balanced_support_metric_joint(**{key: value[positions] for key, value in arrays.items()},
                        support_labels=labels, support_ids=split['support_ids'], classes=split['registered_classes'], old_classes=old,
                        **config['support_metric_resources'],
                        log_callback=log, event_callback=event, context=dict(run_id=run_id, row_id=row_id, split_id=split['split_id']),
                        state_callback=archive,ground_head=ground_head,prediction_callback=fixed,prototype_frame=frame,
                        support_metric_basis=basis,row_basis_owner=row_basis_owner)
                except Exception as exc:
                    write(out/'probe_failed.json', _safe(dict(status='GROUP_BALANCED_SUPPORT_METRIC_JOINT_PROBE_FAILED', **binding,
                        split_identity=split_identity(split, old), error_type=type(exc).__name__, error=str(exc),
                        failure_context=getattr(exc, 'registration_context', {}), completed_episodes=totals['episodes'],
                        completed_episode_counters=totals, counters_scope='COMPLETED_EPISODES_ONLY', workload_complete=False,
                        row_basis_audit=row_basis_owner,row_basis_actual_work=row_basis_work,
                        support_metric_resources=config['support_metric_resources'],
                        query_rows_used=0, source_rows_used=0)))
                    raise
                record = json_native(dict(audit, **split_identity(split, old), scope=SCOPE, query_rows_used=0, source_rows_used=0))
                trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush(); small = compact_record(record)
                compact.write(json.dumps(small, allow_nan=False)+'\n'); compact.flush()
                if writer is None: writer = csv.DictWriter(csvfile, fieldnames=list(small)); writer.writeheader()
                writer.writerow(csv_record(small)); csvfile.flush()
                for key in COUNTERS[4:]:
                    totals[key] = max(totals[key], audit[key]) if key in PEAK_COUNTERS else totals[key]+audit[key]
                for target,field in ((total_work,'actual_work'),(prep_work,'preparation_actual_work'),(stage_work,'stage_actual_work'),(score_work,'score_actual_work')):
                    _work_merge(target,audit[field])
                totals['episodes'] += 1; totals['k1_episodes'] += int(split['k'] == 1)
                totals['oof_episodes'] += int(split['k'] > 1); totals['proxy_anchor_count'] += small['proxy_anchor_count']
                peak_state = max(peak_state, audit['persistent_state_bytes'])
                print(json.dumps(dict(event='SUPPORT_PARENT_COMPLETE', **small, completed=totals['episodes']), allow_nan=False), flush=True)
            archive_status = 'COMPLETE'
    finally:
        for name in ('fit_stages', 'training_events_compact'):
            if (out/(name+'.jsonl')).exists(): stage_csv_from_jsonl(out/(name+'.jsonl'), out/(name+'.csv'))
        state_manifest = archive.finalize(archive_status)
    marker = dict(binding, **totals, status=STATUS, schema=SCHEMA, method=METHOD,
        support_metric_resources=deepcopy(config['support_metric_resources']), counters_scope='ALL_COMPLETED_EPISODES', workload_complete=True,
        implementation_status=IMPLEMENTATION_STATUS, scope=SCOPE, algorithm=PROBE_CONFIG,
        selection=config['selection'], producer_matrix=config['producer_matrix'], payload_audit=payload,
        selected_support_physical_ids=selected_physical,
        query_rows_used=0, source_rows_used=0, truth_read=False, pid=os.getpid(),ground_A_binding=ground_binding,
        ground_geometry_binding=frame_binding,actual_work=total_work,preparation_actual_work=prep_work,
        row_basis_actual_work=row_basis_work,row_basis_ref=row_basis_owner['row_basis_ref'],row_basis_rank=basis.rank,
        row_basis_audit=row_basis_owner,basis_certificate=row_basis_owner['basis_certificate'],row_basis='row_basis.json',
        stage_actual_work=stage_work,score_actual_work=score_work,
        actual_work_aggregation={key:'MAX' if key in WORK_MAX_KEYS else 'SUM' for key in total_work},
        work_aggregation='SUM/MAX',
        nominal_adapter_parameter_count=5,structural_predict_actual_work=None,
        structural_predict_work_unavailable_reason='PUBLIC_PREDICT_RETURNS_NO_INTERNAL_LEDGER',
        independent_analysis_status='NEW_GROUP_BALANCED_SUPPORT_METRIC_SUPPORT_ANALYSIS_REQUIRED; NO_QUERY_RISK_CERTIFICATE',
        persistent_state_bytes=peak_state,
        persistent_state_scope='maximum_retained_C_resident_numeric_state_including_actual_B_not_minimal_deployment',
        triangular_rhs_count_scope='sum_per_actual_solve_columns_r; elements=n*r; dense_work_units=n*n*r_not_measured_FLOPs',
        state_archive_file_count=state_manifest['file_count'], state_archive_file_bytes=state_manifest['total_file_bytes'],
        state_archive_numeric_bytes=state_manifest['numeric_array_bytes'], state_archive_seconds=state_manifest['archive_seconds'],
        state_manifest='state_manifest.json',
        peak_gpu_memory_bytes=None, gpu_memory_reason='CPU only', wall_seconds=time.perf_counter()-begin,
        peak_process_rss_bytes=peak_rss())
    marker = json_native(marker)
    marker['artifact_manifest'] = 'artifact_manifest.json'
    write(out/'probe_complete.json', marker)
    write(out/'artifact_manifest.json', dict(schema='d92_group_balanced_support_metric_joint_artifacts_v1', method=METHOD, status=STATUS,
        files=[dict(path=path.relative_to(out).as_posix(), file_bytes=path.stat().st_size)
               for path in sorted(out.rglob('*')) if path.is_file()],
        inventory_excludes='artifact_manifest.json itself'))
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('support-features', 'capsule', 'output', 'config', 'expected-capsule-id', 'expected-checkpoint-sha256', 'run-id', 'row-id','release-commit'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--ground-packet')
    parser.add_argument('--ground-summary',required=True)
    parser.add_argument('--ground-summary-already-deployed',choices=('true','false'),required=True)
    parser.add_argument('--expected-model-seed', type=int, required=True)
    args = vars(parser.parse_args()); args['config'] = read(args['config'])
    args['ground_summary_already_deployed']=args['ground_summary_already_deployed']=='true'
    evaluate(**args)

if __name__ == '__main__': main()
