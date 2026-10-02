"""Truth-last independent analysis of a complete GroupBalancedSupportMetric support run.

Only JSON artifacts from this run and declared numeric state NPZs are read.
No production fitter, model, cache reader or ground loader is imported.
"""
import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import time

import numpy as np

SCHEMA='d92_group_balanced_support_metric_joint_support_probe_v1'
METHOD='D92-GroupBalancedSupportMetric-GGN1-LocalRidge'
STATUS='GROUP_BALANCED_SUPPORT_METRIC_JOINT_PROBE_COMPLETE'
SCOPE='SUPPORT_ONLY_GROUP_BALANCED_SUPPORT_METRIC_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION'
SUMMARY_STATUS='COMPLETE_GROUP_BALANCED_SUPPORT_METRIC_JOINT_SUPPORT_ANALYSIS_VERIFIED'
SUMMARY_SCHEMA='d92_group_balanced_support_metric_joint_support_analysis_v1'
PATHS=('R0','R_GROUP_BALANCED_SUPPORT_METRIC_seq')
ALGORITHM=dict(schema='d92_group_balanced_support_metric_joint_local_ridge_v1',method=METHOD,nominal_coordinates=5,
    coordinates='EXACT_STORED_Q_RANK_PHYSICAL_U',kappa=.25,ridge_coefficient=1.,
    objective='RMS_class_mean_OOF_CE',damping='ACTUAL_U_GRAM_PLUS_CLASS_BALANCED_PREDICTION_FISHER',
    radius=.5,max_updates=1,max_trials=12,
    initial_step=1.,backtrack=.5,armijo=1e-4,float_comparison_multiplier=128,
    folds='physical_ID_class_rank_mod_min_K_3',prototype_role='FROZEN_GEOMETRY_ONLY',
    C_anchor='ACTUAL_CURRENT_B_U_THETA',old_conditional='FROZEN_ACTUAL_B_FUNCTION',
    kernel='HALF_ORIGINAL_HALF_TANGENT_DISTANCE_FIXED_ORIGINAL_OLD_TRAIN_TAU_GAMMA',
    barrier_average_objective_gap=1e-4,phase1_frozen=True,source_examples=False,query_fit=False,
    parameter_search=False,encoder_backward=False,coordinate_lift_from_prior_methods=False,
    evidence_level='FLOAT64_SUBPROBLEM_DIAGNOSTIC_NOT_COMPLETE_HEAD_CERTIFICATE',
    gate_supervision='GROUP_BALANCED_CLASS_BALANCED_CURRENT_TRAIN',
    gate_weight_rule='N/(2*C_group*n_class)',
    gate_weight_scope='CURRENT_FOLD_LEGAL_TRAIN_LABELS_ONLY')
PRIMITIVE_SUM=('frame_construction_count','physical_evaluation_count','factorization_attempts','factorizations_completed',
    'triangular_calls','triangular_rhs_columns','triangular_rhs_elements','triangular_dense_work_units',
    'distance_evaluation_count','distance_pair_count','distance_qr_batch_count','distance_qr_pair_count',
    'kernel_evaluation_count','kernel_pair_count','jacobian_evaluation_count','spectral_check_count',
    'eigendecomposition_count','secular_iteration_count','secular_evaluation_count','seconds')
GATE_SUM=('factorization_attempts','factorizations_completed','condition_estimation_calls','triangular_calls',
    'triangular_rhs_columns','triangular_rhs_elements','triangular_dense_work_units','spectral_checks',
    'spectral_cubic_dimension_units','objective_evaluations','logistic_record_evaluations','barrier_constraint_evaluations',
    'line_search_trials','newton_iterations','accepted_steps','round_off_residual_acceptances',
    'factorization_seconds','triangular_seconds','spectral_seconds','objective_seconds','wall_seconds',
    'weight_construction_attempts','weight_constructions_completed','weight_physical_record_count','weight_class_count',
    'weighted_logistic_record_evaluations','weighted_entropy_record_evaluations','weight_construction_seconds')
PRIMITIVE_OPS=('original_geometry','tangent_transform','adapted_geometry','raw_kernel','fixed_old_distances','ridge','rmsce','ggn_step')
OPS=PRIMITIVE_OPS+('fixed_old_geometry','gate_forward','gate_jvp')
WORK_SUM=tuple(op+'_calls' for op in OPS)+tuple(op+'_'+k for op in PRIMITIVE_OPS for k in PRIMITIVE_SUM)+tuple(
    op+'_'+k for op in ('gate_forward','gate_jvp') for k in GATE_SUM)+('fixed_old_geometry_wall_seconds',)
WORK_MAX=tuple(op+'_'+k for op in ('gate_forward','gate_jvp') for k in ('peak_factor_buffer_bytes','peak_explicit_temporary_bytes'))
BASIS_SUM=('fraction_operations_attempted','fraction_operations_completed','integer_operations_attempted',
    'integer_operations_completed','integer_guard_checks','binary64_decodes','float_conversions','nextafter_steps',
    'square_root_enclosures','columns_started','columns_completed','numeric_array_payload_bytes',
    'logical_rational_integer_payload_bytes','certificate_utf8_bytes','wall_seconds')
BASIS_MAX=('max_observed_integer_bits','max_intermediate_bit_bound')
STEP_SUM=('fisher_construction_attempts','fisher_constructions_completed','physical_support_count','score_jvp_scalar_count',
    'symmetry_check_count','spectral_check_attempts','spectral_checks_completed','factorization_attempts',
    'factorizations_completed','triangular_calls','triangular_calls_completed','triangular_rhs_columns',
    'triangular_rhs_elements','triangular_dense_work_units','secular_evaluation_count','secular_iteration_count',
    'inward_rescale_count','diagnostic_readback_count','fisher_seconds','spectral_seconds','factor_seconds',
    'triangular_seconds','diagnostic_seconds','seconds')
WORK_SUM+=('basis_calls','basis_binding_calls','basis_binding_wall_seconds',
    'basis_binding_physical_gram_evaluation_count','support_metric_step_calls')+tuple('basis_'+k for k in BASIS_SUM)+tuple(
    'support_metric_step_'+k for k in STEP_SUM)
WORK_MAX+=tuple('basis_'+k for k in BASIS_MAX)+('support_metric_step_peak_single_factor_input_output_bytes',)
OPS+=('basis','basis_binding','support_metric_step')
WORK_KEYS=set(WORK_SUM+WORK_MAX)
HIGH_COUNTERS=('episodes','k1_episodes','oof_episodes','proxy_anchor_count','sequence_paths','baseline_head_fit_count',
    'baseline_factorization_count','baseline_triangular_solve_count','candidate_preparation_count','candidate_stage_count',
    'final_candidate_head_fit_count','optimizer_steps','trial_count','accepted_trial_count','rejected_trial_count',
    'ggn_step_count','ggn_parameter_direction_count','final_score_evaluation_count','final_score_physical_count',
    'ground_A_score_evaluation_count','ground_A_score_physical_count','ground_A_inference_seconds',
    'structural_predict_evaluation_count','structural_predict_physical_count','structural_predict_seconds','peak_resident_numeric_state_bytes',
    'row_basis_construction_count','peak_effective_adapter_rank')
COUNTERS=HIGH_COUNTERS+WORK_SUM+WORK_MAX
PEAKS=set(WORK_MAX)|{'peak_resident_numeric_state_bytes','peak_effective_adapter_rank'}
COORDS=('run_id','row_id','split_id','scope','fold','trial','parent_k','train_k')
IDENTITY=('split_id','receiver','scenario','k','support_seed','new_count')
STREAMS=('R0_B','R0_C','R_GROUP_BALANCED_SUPPORT_METRIC_seq_B','R_GROUP_BALANCED_SUPPORT_METRIC_seq_C')
EPS=np.finfo(np.float64).eps


def check(condition,message):
    if not condition:raise ValueError(message)


def _pairs(items):
    result={}
    for key,value in items:
        check(key not in result,'Duplicate JSON key '+key);result[key]=value
    return result


def loads(text):
    def reject(value):raise ValueError('Nonfinite JSON literal '+value)
    return json.loads(text,object_pairs_hook=_pairs,parse_constant=reject)


def read(path):return loads(Path(path).read_text(encoding='utf-8'))
def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield loads(line)


def _number(value,integer=False):
    return type(value) in ((int,) if integer else (int,float)) and math.isfinite(value) and value>=0


def _equal(actual,expected,label):
    if type(expected) is float or type(actual) is float:
        check(type(actual) in (int,float) and type(expected) in (int,float) and
            abs(actual-expected)<=128*EPS*max(1.,abs(actual),abs(expected)),label)
    elif isinstance(expected,dict):
        check(isinstance(actual,dict) and set(actual)==set(expected),label+' keys')
        for key in expected:_equal(actual[key],expected[key],label+'.'+key)
    elif isinstance(expected,list):
        check(isinstance(actual,list) and len(actual)==len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):_equal(a,b,label+'.'+str(i))
    else:check(actual==expected,label)


def _empty_work():return dict.fromkeys(WORK_SUM+WORK_MAX,0)
def _empty_counters():return dict.fromkeys(COUNTERS,0)
def _merge(target,source,peaks=PEAKS):
    for key,value in source.items():
        check(_number(value),'Invalid measured work '+key)
        target[key]=max(target.get(key,0),value) if key in peaks else target.get(key,0)+value


def verify_work(audit):
    """Rebuild actual operation SUM/MAX ledger, including physical score wrappers."""
    work=audit.get('actual_work');ops=audit.get('operation_audits')
    check(isinstance(work,dict) and set(work)==WORK_KEYS and isinstance(ops,list),'Incomplete actual work schema')
    result=_empty_work()
    if ops and 'physical_position' in ops[0]:
        check([r['physical_position'] for r in ops]==list(range(len(ops))),'Physical score positions differ')
        for item in ops:_merge(result,verify_work(item['audit']),set(WORK_MAX))
        check(audit['physical_record_count']==audit['actual_score_calls']==len(ops),'Physical score count differs')
    else:
        for item in ops:
            op=item['operation'];check(op in OPS,'Undeclared operation '+str(op));detail=item['audit']
            check(isinstance(detail,dict),'Missing operation audit');result[op+'_calls']+=1
            for key,value in detail.items():
                name=op+'_'+key
                if name in WORK_KEYS:
                    check(_number(value),'Invalid operation counter '+name)
                    result[name]=max(result[name],value) if name in WORK_MAX else result[name]+value
            if op=='ridge':
                c=detail['class_count'];n=detail['train_count'];d=detail['jvp_directions']
                check(d in (0,5) and detail['factorization_attempts']==detail['factorizations_completed']==1 and
                    detail['spectral_check_count']==1,'Completed Ridge factor/PSD work differs')
                check(detail['triangular_calls']==(4 if d else 2) and detail['triangular_rhs_columns']==2*(c+1)*(1+d)
                    and detail['triangular_rhs_elements']==n*detail['triangular_rhs_columns'] and
                    detail['triangular_dense_work_units']==n*n*detail['triangular_rhs_columns'],'Complete free-intercept RHS accounting differs')
                check(detail['intercept_regularized'] is False,'Free intercept changed')
    _equal(work,result,'Operation ledger closure')
    return result


def compact_event(event):
    def scalars(value):
        if not isinstance(value,dict):return value
        return {k:scalars(v) for k,v in value.items() if isinstance(v,dict) or v is None or type(v) in (str,int,float,bool)}
    result=scalars(event)
    for key,value in event.items():
        if key in ('state_ref','head_ref','head_refs','prior_ref') or key.endswith(('_state_ref','_head_ref')):result[key]=value
    for key in ('objective','audit','preparation','initial_objective','final_objective','final_fit'):
        if isinstance(event.get(key),dict):result[key]=compact_event(event[key])
    for key in ('class_ce_means','class_ce_sums','class_ce_counts','class_weights',
            'theta','anchor_theta','gradient','direction','metric_fisher','metric_M','metric_H','physical_gram',
            'held_ce_sums','held_ce_counts','classes','old_classes'):
        if key in event:result[key]=event[key]
    for key in ('inner_folds','folds','trials'):
        if isinstance(event.get(key),list):result[key]=[compact_event(v) for v in event[key]]
    for key in ('actual_work','operation_audits','limits','rms_audit','quadratic_step_audit',
            'support_metric_step_audit','step_audit','basis_audit','basis_certificate_ref','row_basis_ref'):
        if key in event:result[key]=event[key]
    return result


def validate_spec(spec):
    check(spec['schema']==SCHEMA and spec['group_id']=='d92-group-balanced-support-metric-joint-support','Independent source schema/group differs')
    p=spec['probe'];check(p['algorithm']==ALGORITHM,'Frozen algorithm differs')
    resources=p['support_metric_resources']
    check(set(resources)=={'max_integer_bits','max_fraction_operations','max_newton_iterations',
        'max_line_search_trials','max_factor_buffer_bytes','max_secular_iterations'} and
        all(type(v) is int and v>0 for v in resources.values()) and resources['max_secular_iterations']<=128,
        'Explicit resources differ')
    rows=spec['rows'];check(len(rows)==4 and len({r['row_id'] for r in rows})==4,'Exactly four distinct declared rows required')
    models={}
    for row in rows:
        check(isinstance(row['row_id'],str) and row['row_id'] not in ('','.','..') and '/' not in row['row_id'] and '\\' not in row['row_id'],
            'Invalid declared row ID')
        identity=(row['expected_checkpoint_sha256'],row.get('ground_packet'),row['ground_summary'],row['ground_summary_already_deployed'])
        check(models.setdefault(row['seeds']['model'],identity)==identity,'Same model source identity differs across cohorts')
    check({(r['cohort'],r['seeds']['model']) for r in rows}==set(itertools.product(('rx3','rx1'),(2026092701,2026092702))),
        'Fixed two-model two-cohort matrix differs')
    check(set(p['cohorts'])=={'rx3','rx1'},'Unexpected cohort')
    for co in p['cohorts'].values():
        sel=co['selection'];check(sel['ks']==[1,5,10,20] and sel['new_counts']==[0,2,5,10,20],'Full K/new axes required')
        pairs=sel['receiver_scenes'];check(len(pairs)==2 and len(set(map(tuple,pairs)))==2,'Two explicit receiver/scene strata required')
        expected={(rx,sc,k,q,sel['support_seed']) for rx,sc in pairs for k in sel['ks'] for q in sel['new_counts']}
        got=set();ids=set();old=None
        for item in sel['splits']:
            key=tuple(item[k] for k in ('receiver','scenario','k','new_count','support_seed'))
            check(key in expected and key not in got and item['split_id'] not in ids,'Duplicate/unexpected physical parent')
            registry=item['registered_classes'];check(len(registry)==6+item['new_count'] and len(set(registry))==len(registry),'Invalid class registry')
            if old is None:old=registry[:6]
            check(registry[:6]==old,'Old registry changed within row');got.add(key);ids.add(item['split_id'])
        check(got==expected and len(ids)==40,'Incomplete forty-parent row')
    return spec


class Archives:
    def __init__(self,root,run_id,row_id):
        self.root=Path(root);self.run_id=run_id;self.row_id=row_id;self.used=set();self.refs={};self.namespaces={}
        self.weighted_gate_readbacks=[]
        self.manifest=read(self.root/'state_manifest.json');m=self.manifest
        check(m['status']=='COMPLETE' and m['schema']=='d92_group_balanced_support_metric_joint_state_archive_v1' and m['method']==METHOD,'Incomplete numeric archive')
        size=logical=0;seconds=0.;phases={}
        for ref in m['files']:
            relative=PurePosixPath(ref['path'])
            check(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='state_arrays' and
                relative.suffix=='.npz' and ref['path'] not in self.refs,'Invalid/duplicate numeric archive path')
            path=self.root/ref['path'];check(path.resolve().is_relative_to(self.root.resolve()) and not path.is_symlink(),'State archive escaped row')
            ns=loads(ref['namespace']);check(ns['run_id']==run_id and ns['row_id']==row_id,'State run/row identity differs')
            check(ref['failed_numeric_state'] is False,'Failed numeric state cannot certify completion')
            identity=(ref['namespace'],ref['key']);check(identity not in self.namespaces,'Duplicate state namespace/key')
            self.namespaces[identity]=ref['path'];self.refs[ref['path']]=ref
            check(path.stat().st_size==ref['file_bytes'],'Numeric archive physical bytes differ')
            arrays=self.load(ref)
            count=sum(v.nbytes for v in arrays.values());logical+=count;size+=path.stat().st_size
            check(_number(ref['archive_seconds']),'Invalid archive timing');seconds+=ref['archive_seconds']
            phase=phases.setdefault(ns['state'],dict(file_count=0,file_bytes=0,numeric_array_bytes=0,archive_seconds=0.))
            _merge(phase,dict(file_count=1,file_bytes=ref['file_bytes'],numeric_array_bytes=count,archive_seconds=ref['archive_seconds']),set())
        check(set(self.refs)=={p.relative_to(self.root).as_posix() for p in (self.root/'state_arrays').glob('*.npz')},'Undeclared/missing numeric archive')
        _equal(dict(file_count=len(self.refs),total_file_bytes=size,numeric_array_bytes=logical,archive_seconds=seconds,by_phase=phases),
            {k:m[k] for k in ('file_count','total_file_bytes','numeric_array_bytes','archive_seconds','by_phase')},'Archive totals')

    def load(self,ref,names=None):
        path=self.root/ref['path']
        with np.load(path,allow_pickle=False) as saved:
            check(len(saved.files)==len(set(saved.files)) and set(saved.files)==set(ref['arrays']),'NPZ numeric closure differs')
            values={}
            for key in saved.files:
                a=saved[key];meta=ref['arrays'][key]
                check(a.dtype.kind in 'fbiu' and (a.dtype.kind!='f' or a.dtype in (np.dtype('float32'),np.dtype('float64')))
                    and np.isfinite(a).all(),'Nonfinite/object/unsupported state array')
                check(meta==dict(shape=list(a.shape),dtype=str(a.dtype),nbytes=a.nbytes,all_finite=True,nonfinite_count=0),'State array metadata differs')
                if names is None or key in names:values[key]=a
        return values

    def reference(self,ref,coords=None,state=None,key=None):
        check(isinstance(ref,dict) and ref.get('path') in self.refs,'Missing declared state reference')
        check(ref==self.refs[ref['path']],'State reference metadata differs from manifest')
        ns=loads(ref['namespace'])
        if coords is not None:check(all(ns.get(k)==v for k,v in coords.items()),'State physical namespace differs')
        if state is not None:check(ns['state']==state,'State phase differs')
        if key is not None:check(ref['key']==key,'State key differs')
        self.used.add(ref['path']);return ref

    def walk(self,value,coords):
        if isinstance(value,dict):
            if {'path','namespace','arrays','file_bytes','key'}<=set(value):
                if value==getattr(self,'row_basis_ref',None):self.reference(value,self.row_basis_coords,'ROW_GROUP_BALANCED_SUPPORT_METRIC_BASIS','basis')
                else:self.reference(value,coords)
            else:
                for item in value.values():self.walk(item,coords)
        elif isinstance(value,list):
            for item in value:self.walk(item,coords)

    def close(self):check(self.used==set(self.refs),'Unreferenced numeric state; full archive closure required')


def _scores(value,ids,classes):
    result=np.asarray(value,dtype=np.float64)
    if not ids and result.shape==(0,):result=result.reshape(0,len(classes))
    check(result.shape==(len(ids),len(classes)),'Fixed score matrix shape differs')
    check(np.isfinite(result).all(),'Nonfinite fixed scores');return result


def _predictions(scores,classes,native=False):
    if native:return [classes[int(i)] for i in np.argmax(scores,axis=1)]
    return [min(classes[i] for i in np.flatnonzero(row==np.max(row))) for row in scores]


def _coordinates(row_id,run_id,parent,path):
    return dict(run_id=run_id,row_id=row_id,split_id=parent['split_id'],
        **{key:path[key] for key in ('scope','fold','trial','parent_k','train_k')})


def _array_close(actual,expected,label):
    actual=np.asarray(actual);expected=np.asarray(expected)
    check(actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),label+' shape/finite')
    # Arithmetic readback, not an interval enclosure or independent head JVP.
    scale=max(1.,float(np.max(np.abs(expected),initial=0)),float(np.max(np.abs(actual),initial=0)))
    tolerance=128*EPS*max((1,)+actual.shape)*scale
    check(float(np.max(np.abs(actual-expected),initial=0))<=tolerance,label)


def verify_basis_arrays(arrays,rank):
    check(type(rank) is int and 0<=rank<=5,'Invalid declared exact basis rank')
    Q=arrays['original_dictionary_Q'];U=arrays['U'];lo=arrays['U_lower'];hi=arrays['U_upper'];err=arrays['U_error_bounds']
    check(Q.shape==(160,5) and U.shape==lo.shape==hi.shape==err.shape==(160,rank),'Physical U basis shape differs')
    check(all(a.dtype==np.dtype('float64') and np.isfinite(a).all() for a in (Q,U,lo,hi,err)),
        'Physical basis precision/finite differs')
    check(np.all(lo<=U) and np.all(U<=hi) and np.all(err>=0),'Physical U enclosure differs')
    outward=np.maximum(np.abs(np.spacing(lo)),np.abs(np.spacing(hi)))+EPS*np.maximum(np.abs(lo),np.abs(hi))
    check(np.all(np.maximum(U-lo,hi-U)<=err+outward), 'Stored U component error bound differs')
    columns=arrays['exact_column_indices']
    check(columns.shape==(rank,) and columns.dtype.kind in 'iu' and len(set(columns.tolist()))==rank and
        np.all((columns>=0)&(columns<5)) and columns.tolist()==sorted(columns.tolist()),'Exact column selection differs')
    if 'physical_gram' in arrays:_array_close(arrays['physical_gram'],U.T@U,'Actual physical Gram differs')
    return U


def verify_row_basis(owner,archives,resources):
    check(owner['schema']=='d92_support_metric_row_basis_v1' and owner['method']==METHOD and owner['status']=='COMPLETE'
        and owner['row_basis_construction_count']==1 and owner['input_role']=='FROZEN_GROUND_Q_ONLY_NO_SUPPORT_TEACHER',
        'Incomplete or wrong-scope row basis owner')
    coords=dict(run_id=archives.run_id,row_id=archives.row_id,split_id=None,scope='row_frozen_ground_geometry',
        fold=None,trial=None,parent_k=None,train_k=None,state='ROW_GROUP_BALANCED_SUPPORT_METRIC_BASIS')
    check(owner['context']==coords,'Row basis namespace differs')
    if hasattr(archives,'row_basis_owner'):
        _equal(owner,archives.row_basis_owner,'Same row basis owner changed')
        return archives.row_basis_work
    ref=archives.reference(owner['row_basis_ref'],coords,'ROW_GROUP_BALANCED_SUPPORT_METRIC_BASIS','basis')
    arrays=archives.load(ref);rank=owner['row_basis_rank'];verify_basis_arrays(arrays,rank)
    check(arrays['rank'].shape==() and arrays['rank'].dtype.kind in 'iu' and int(arrays['rank'])==rank,'Basis scalar rank differs')
    audit=owner['construction_audit']
    check(audit['status']=='COMPLETE' and audit['exact_rank']==rank and audit['enclosure_bits']==128 and
        audit['max_integer_bits']==resources['max_integer_bits'] and audit['max_fraction_operations']==resources['max_fraction_operations'],
        'Basis arithmetic resource/rank contract differs')
    check(audit['max_observed_integer_bits']<=resources['max_integer_bits'] and
        audit['max_intermediate_bit_bound']<=resources['max_integer_bits'] and
        audit['fraction_operations_attempted']<=resources['max_fraction_operations'],'Completed basis exceeded arithmetic resources')
    certificate=owner['basis_certificate'];check(certificate['path']=='basis_certificate.json' and
        certificate['scope']=='ORDINARY_METHOD_ARTIFACT_NOT_AUTHORIZATION' and _number(certificate['archive_seconds']),
        'Declared basis certificate/actual archive timing differs')
    path=archives.root/certificate['path'];check(not path.is_symlink(),'Basis certificate escaped row')
    encoded=path.read_bytes();decoded=encoded.decode('utf-8');value=loads(decoded)
    check(len(encoded)==certificate['file_bytes']==certificate['utf8_bytes']==audit['certificate_utf8_bytes'],
        'Basis certificate actual bytes differ')
    check(value['schema']=='d92_support_metric_basis_v1' and value['shape']==[160,5] and value['rank']==rank and
        value['enclosure_bits']==128 and value['input_scope']=='EXACT_STORED_BINARY64_NOT_PRE_ROUNDING_PROTOTYPES' and
        value['fraction_encoding']=='signed_hex_numerator_positive_hex_denominator','Basis certificate metadata differs')
    exact=value['exact_state']
    check(exact['column_indices']==arrays['exact_column_indices'].tolist() and
        len(exact['orthogonal_residuals'])==len(exact['norm_squared'])==len(value['normalizations'])==rank,
        'Exact basis certificate rank/column closure differs')
    numeric=sum(arrays[k].nbytes for k in ('original_dictionary_Q','U','U_lower','U_upper','U_error_bounds'))
    check(audit['numeric_array_payload_bytes']==numeric,'Basis ndarray byte scope differs')
    work=verify_work(owner);check(work['basis_calls']==1 and len(owner['operation_audits'])==1 and
        owner['operation_audits'][0]==dict(operation='basis',audit=audit),'Basis construction ledger differs')
    archives.row_basis_owner=owner;archives.row_basis_ref=ref;archives.row_basis_coords=coords
    archives.row_basis_arrays=arrays;archives.row_basis_work=work
    return work


def verify_objective_arrays(arrays,rank,train,registry,label_mapping,reported_loss):
    scores=arrays['scores'];labels=arrays['labels']
    check(scores.shape==(len(train),len(registry)) and labels.shape==(len(train),) and labels.dtype.kind in 'iu',
        'OOF objective physical shape differs')
    check(np.all((labels>=0)&(labels<len(registry))),'OOF objective labels outside registry')
    if label_mapping:check(labels.tolist()==[registry.index(label_mapping[p]) for p in train],'OOF label physical order differs')
    counts=np.bincount(labels,minlength=len(registry));check(np.all(counts>0),'Missing OOF class')
    shift=scores-scores.max(axis=1,keepdims=True);exp=np.exp(shift);p=exp/exp.sum(axis=1,keepdims=True)
    ce=np.asarray([np.logaddexp(0.,np.logaddexp.reduce(np.delete(row-row[y],y))) for row,y in zip(scores,labels)])
    means=np.bincount(labels,weights=ce,minlength=len(registry))/counts
    maximum=float(means.max());rms=maximum*float(np.sqrt(np.mean((means/maximum)**2))) if maximum>0 else 0.
    check(rms>0,'Invalid saved multiclass RMSCE')
    _array_close(np.asarray(reported_loss),np.asarray(rms),'RMSCE readback differs')
    _array_close(arrays['probabilities'],p,'Objective probabilities differ')
    if 'score_jacobian' not in arrays:return
    J=arrays['score_jacobian'];check(J.shape==scores.shape+(rank,),'Complete r-direction JVP shape differs')
    residual=p.copy()
    for i,y in enumerate(labels):residual[i,y]=-np.sum(p[i,np.arange(len(registry))!=y])
    per_sample=np.einsum('nc,ncr->nr',residual,J);classJ=np.zeros((len(registry),rank))
    np.add.at(classJ,labels,per_sample);classJ/=counts[:,None]
    weights=means/(len(registry)*rms);gradient=weights@classJ
    centered=J-np.einsum('nc,ncr->nr',p,J)[:,None,:]
    first=np.einsum('n,nc,ncr,ncs->rs',weights[labels]/counts[labels],p,centered,centered)
    unit=(means/maximum)/np.linalg.norm(means/maximum)
    outer=(np.eye(len(registry))-np.outer(unit,unit))/len(registry)/rms
    curvature=first+classJ.T@outer@classJ
    _array_close(arrays['gradient'],gradient,'RMSCE gradient readback differs')
    _array_close(arrays['curvature'],curvature,'RMSCE GGN differs; Fisher is not its replacement')


def verify_metric_direction(initial,saved,step_audit,rank):
    check(rank>0 and saved['direction'].shape==(rank,),'Missing metric direction')
    for source,target in (('gradient','gradient'),('curvature','ggn'),('physical_gram','physical_gram'),
                          ('score_jacobian','score_jvp'),('probabilities','probabilities'),('labels','labels')):
        _array_close(saved[target],initial[source],'Metric step input differs: '+source)
    J=initial['score_jacobian'];p=initial['probabilities'];y=initial['labels'];C=p.shape[1]
    counts=np.bincount(y,minlength=C);weights=1./(C*counts[y]);centered=J-np.einsum('nc,ncr->nr',p,J)[:,None,:]
    F=np.einsum('n,nc,ncr,ncs->rs',weights,p,centered,centered)
    M=initial['physical_gram']+F;H=initial['curvature']+M
    for name,value in (('fisher',F),('metric',M),('hessian',H),('centered_jvp',centered),('sample_weights',weights),('class_counts',counts)):
        _array_close(saved[name],value,'Saved metric construction differs: '+name)
    # KKT is the readback of the actually solved stored subproblem. Rebuilt F is
    # separately compared above; do not substitute its different reduction order.
    M=saved['metric'];H=saved['hessian']
    factor=saved['metric_chol'];check(factor.shape==(rank,rank) and np.all(np.diag(factor)>0),'Metric factor missing')
    _array_close(factor,np.tril(factor),'Metric factor not lower triangular')
    _array_close(factor@factor.T,M,'Metric factor residual differs')
    d=saved['direction'];lam=float(saved['multiplier']);check(math.isfinite(lam) and lam>=0,'Invalid metric multiplier')
    g=initial['gradient'];Md=M@d;res=H@d+g+lam*Md;value=float(d@Md);slack=.25-value
    scale=max(1.,float(np.linalg.norm(g)),float(np.linalg.norm(H))*float(np.linalg.norm(d)),abs(lam)*float(np.linalg.norm(Md)))
    residual=float(np.linalg.norm(res));comp=abs(lam*slack);comp_scale=max(1.,lam*.25);tol=128*EPS*max(1,rank)
    check(0<=value<=.25 and residual/scale<=tol and comp/comp_scale<=tol,'Original-coordinate metric KKT readback failed')
    _array_close(saved['kkt_residual'],res,'Saved metric stationarity residual differs')
    for key,actual in dict(metric_ball_value=value,metric_ball_slack=slack,stationarity_residual_norm=residual,
        stationarity_residual_scale=scale,stationarity_relative_residual=residual/scale,
        complementarity_residual=comp,complementarity_relative_residual=comp/comp_scale,
        directional_derivative=float(g@d),quadratic_value=float(g@d+.5*d@H@d),
        numerical_diagnostic_relative_tolerance=tol).items():
        _array_close(np.asarray(step_audit[key]),np.asarray(actual),'Step diagnostic readback differs: '+key)
    check(step_audit['direction_error_bound'] is None and step_audit['precise_original_problem_certificate'] is False
        and step_audit['gradient_zero_certified'] is False,'Float64 diagnostic promoted to complete certificate')
    return d,M


def verify_weighted_gate(arrays,labels,old_columns,audit):
    """Independent saved-equation readback; no gate fit, kernel or JVP replay.

    Weights are reconstructed from this head's legal physical training labels.
    This is finite-precision equation evidence, not an interval head certificate
    or a claim that an approximately feasible dual value is a rigorous bound.
    """
    labels=np.asarray(labels);n=len(labels);old_columns=np.asarray(old_columns,dtype=int)
    check(labels.shape==(n,) and labels.dtype.kind in 'iu' and n>0,'Gate legal labels missing')
    check(all(arrays[key].dtype.kind in 'iu' for key in ('gate_labels','gate_class_labels','gate_class_counts','gate_old_indices')),
        'Gate physical labels/counts must retain integer array types')
    check(np.array_equal(arrays['gate_labels'],labels),'Gate labels differ from physical train labels')
    classes,counts=np.unique(labels,return_counts=True)
    class_targets=np.isin(classes,old_columns).astype(float)
    targets=np.isin(labels,old_columns).astype(float);old=np.flatnonzero(targets==1)
    check(0<len(old)<n and np.any(class_targets==0) and np.any(class_targets==1),'Gate requires both train groups')
    weights=np.empty(n,dtype=np.float64)
    for c,count,t in zip(classes,counts,class_targets):
        weights[labels==c]=n/(2*int(np.count_nonzero(class_targets==t))*int(count))
    for key,expected in (('gate_class_labels',classes),('gate_class_counts',counts),
                         ('gate_class_targets',class_targets),('gate_targets',targets),('gate_old_indices',old)):
        check(np.array_equal(arrays[key],expected),'Gate class/group binding differs: '+key)
    _array_close(arrays['gate_weights'],weights,'Gate current-train class-balanced weights differ')
    _array_close(np.asarray([weights[targets==t].sum() for t in (1,0)]),np.full(2,n/2),
        'Gate old/new group mass differs')
    K=arrays['K'];alpha=arrays['gate_alpha'];b=arrays['gate_b'];bounds=arrays['lower_bounds']
    check(K.shape==(n,n) and alpha.shape==(n,) and b.shape==() and bounds.shape==(len(old),int(np.count_nonzero(class_targets==0))),
        'Weighted gate physical shape differs')
    _array_close(arrays['gate_K'],K,'Gate kernel binding differs')
    _array_close(arrays['gate_lower_bounds'],bounds,'Gate all-pair bounds binding differs')
    zeta=float(arrays['gate_zeta']);_equal(zeta,float(n*1e-4/bounds.size),'Fixed barrier scale differs')
    Ka=K@alpha;f=Ka+float(b);slack=f[old,None]-bounds
    check(np.all(slack>0),'Weighted gate is not strictly feasible')
    _array_close(arrays['gate_f'],f,'Gate logits differ')
    _array_close(arrays['gate_slacks'],slack,'Gate slacks differ')
    e=np.exp(-np.abs(f));check(np.all(e>0),'Weighted gate logistic curvature underflow')
    positive=np.where(f>=0,1/(1+e),e/(1+e));negative=np.where(f>=0,e/(1+e),1/(1+e))
    q=weights*np.where(targets==1,-negative,positive);D=weights*e/((1+e)**2)
    mu=zeta/slack;barrier_D=mu/slack
    q[old]-=mu.sum(axis=1);D[old]+=barrier_D.sum(axis=1)
    _array_close(arrays['gate_qeff'],q,'Weighted gate residual vector differs')
    _array_close(arrays['gate_D_eff'],D,'Weighted gate effective curvature differs')
    ridge=.5*float(alpha@Ka);logistic=float(np.sum(weights*np.logaddexp(0.,np.where(targets==1,-f,f))))
    barrier=-zeta*float(np.log(slack).sum());h=alpha+q
    values=dict(ridge_penalty=ridge,logistic_loss_sum=logistic,barrier_term=barrier,
        primal_objective=ridge+logistic,objective=ridge+logistic+barrier,
        canonical_residual=float(np.linalg.norm(h,np.inf)),intercept_residual=abs(float(q.sum())),
        sum_alpha_residual=abs(float(alpha.sum())),primal_gradient_residual=float(np.linalg.norm(K@h,np.inf)),
        minimum_slack=float(slack.min()),complementarity_gap=float(np.sum(mu*slack)))
    check(audit['status']=='COMPLETE' and audit['schema']=='d92_group_balanced_barrier_gate_v1'
        and audit['constraint_count']==bounds.size,'Incomplete weighted gate audit')
    check(audit['weight_rule']==ALGORITHM['gate_weight_rule'] and audit['weight_scope']==ALGORITHM['gate_weight_scope']
        and audit['weight_derivative_scope']=='FIXED_CURRENT_TRAIN_LABELS_AND_COUNTS; dw=0',
        'Weighted gate declared supervision scope differs')
    check(audit['old_class_count']==int(np.count_nonzero(class_targets==1))
        and audit['new_class_count']==int(np.count_nonzero(class_targets==0)),'Weighted gate audit group class counts differ')
    for key,value in dict(old_supervision_mass=float(weights[targets==1].sum()),
            new_supervision_mass=float(weights[targets==0].sum()),total_supervision_mass=float(weights.sum())).items():
        _equal(audit[key],value,'Weighted gate audit mass differs: '+key)
    for key,value in values.items():_array_close(np.asarray(audit[key]),np.asarray(value),'Weighted gate readback: '+key)
    # Match the actual solver's arithmetic-error model, not an accuracy threshold.
    f_error=8*EPS*max(1,n)*(np.abs(K)@np.abs(alpha)+abs(float(b)))
    q_error=D*f_error
    q_error[old]+=np.sum(barrier_D*(4*EPS*(np.abs(f[old,None])+np.abs(bounds))),axis=1)
    q_error+=8*EPS*(np.abs(q)+weights)
    roundoff=float(np.max(q_error));scale=max(1.,float(np.linalg.norm(alpha,np.inf)),float(np.linalg.norm(q,np.inf)))
    tolerance=max(512*EPS*max(1,n)*scale,roundoff)
    for key,value in dict(evaluated_gradient_roundoff_bound=roundoff,stationarity_tolerance=float(tolerance),
            intercept_tolerance=float(n*tolerance),primal_gradient_tolerance=float(max(1.,np.linalg.norm(K,np.inf))*tolerance)).items():
        _array_close(np.asarray(audit[key]),np.asarray(value),'Weighted gate arithmetic tolerance: '+key)
    check(values['canonical_residual']<=tolerance and values['intercept_residual']<=n*tolerance
        and values['sum_alpha_residual']<=n*tolerance,'Weighted gate stationarity/intercept equations fail')
    return dict(values,stationarity_tolerance=float(tolerance),intercept_tolerance=float(n*tolerance),
        evaluated_gradient_roundoff_bound=roundoff,physical_count=n,class_count=len(classes),old_mass=float(weights[targets==1].sum()),
        new_mass=float(weights[targets==0].sum()),evidence_scope='SAVED_WEIGHTED_GATE_FLOAT64_EQUATIONS_NOT_COMPLETE_HEAD_CERTIFICATE')


def _verify_weighted_objective(arrays,prep,train,full_labels,old_columns,audits):
    by_id=dict(zip(train,full_labels));readbacks=[]
    for fold in prep['folds']:
        prefix='fold_'+str(fold['fold'])+'_'
        saved={key[len(prefix):]:value for key,value in arrays.items() if key.startswith(prefix)}
        labels=np.asarray([by_id[p] for p in fold['train_ids']],dtype=int)
        audit=next(audits,None);check(audit is not None,'Missing weighted gate operation')
        readbacks.append(dict(inner_fold=fold['fold'],**verify_weighted_gate(saved,labels,old_columns,audit)))
    return readbacks


def _verify_stage(stage,prep,path,coords,archives,resources,b_state=None,label_mapping=None):
    name=stage['state'];is_b=name=='B_GROUP_BALANCED_SUPPORT_METRIC';registry=path['b_classes'] if is_b else path['c_classes']
    train=path['b_training_ids'] if is_b else path['c_training_ids']
    check(stage['status']=='COMPLETED' and stage['method']==METHOD and stage['executed_core_schema']==ALGORITHM['schema']
        and stage['final_head_complete'] is True,'Incomplete actual candidate head')
    check(stage['training_physical_ids']==train and stage['classes']==registry and stage['train_physical_count']==len(train),
        'Candidate train/class binding differs')
    check(stage['config']==ALGORITHM and stage['limits']==resources and stage['nominal_parameter_count']==5,'Stage algorithm/resources differ')
    check(stage['preparation_ref']==('B' if is_b else 'C') and prep['training_physical_ids']==train,'Preparation/stage differs')
    rank=stage['effective_parameter_rank'];eligible=path['train_k']>1 and rank>0
    check(type(rank) is int and 0<=rank<=5 and prep['effective_parameter_rank']==rank,'Stage/preparation basis rank differs')
    check(stage['trainable_parameter_count']==(rank if eligible else 0),'Actual trainable parameter scope differs')
    check(stage['no_held']==(path['train_k']==1) and stage['mode']==('B' if is_b else 'C_seq'),'ParentK confused with actual trainK')
    for obj in (prep,stage):
        check(all(obj.get(k)==v for k,v in coords.items()),'Preparation/stage physical identity differs')
        verify_work(obj)
    check(prep['basis_construction_charged_here'] is False and prep['actual_work']['basis_calls']==0,
        'Row-owned basis construction charged again within preparation')
    check(prep['actual_work']['basis_binding_calls']==1 and prep['actual_work']['basis_binding_physical_gram_evaluation_count']==1,
        'Actual preparation binding/Gram work differs')
    ba=prep['basis_audit']
    check(ba['status']=='COMPLETE' and ba['exact_rank']==rank and ba['max_integer_bits']==resources['max_integer_bits']
        and ba['max_fraction_operations']==resources['max_fraction_operations'],'Frozen basis audit/resources differ')
    _equal(ba,archives.row_basis_owner['construction_audit'],'Preparation basis construction provenance differs')
    check(prep['row_basis_ref']==archives.row_basis_ref and prep['basis_certificate_ref']==archives.row_basis_owner['basis_certificate'],
        'Preparation replaced row basis artifact')
    folds=prep.get('folds',[]);expected_count=0 if path['train_k']==1 else min(path['train_k'],3)
    check(len(folds)==expected_count,'Complete inner physical fold coverage differs')
    for index,fold in enumerate(folds):
        check(fold['fold']==index and len(fold['train_ids'])==len(set(fold['train_ids'])) and
            len(fold['held_ids'])==len(set(fold['held_ids'])),'Duplicate/misordered inner physical fold')
        check(set(fold['train_ids']).isdisjoint(fold['held_ids']) and set(fold['train_ids'])|set(fold['held_ids'])==set(train),
            'Inner physical partition differs')
        if label_mapping:
            members={pid:i%expected_count for c in registry for i,pid in enumerate(sorted(p for p in train if label_mapping[p]==c))}
            check(fold['held_ids']==sorted(p for p in train if members[p]==index) and
                fold['train_ids']==sorted(p for p in train if members[p]!=index),'Inner train-only physical-ID fold rule differs')
            check(fold['old_inner_train_ids']==[p for p in fold['train_ids'] if label_mapping[p] in path['b_classes']],
                'Inner old teacher physical IDs differ')
        check(set(fold['old_inner_train_ids'])<=set(fold['train_ids']),'Old teacher escaped inner train')
        if not is_b:
            ref=archives.reference(fold['prior_ref'],coords,'C_prepare')
            saved=archives.load(ref,names={'theta','K','alpha','intercept','head_theta_padded','physical_gram',
                'original_dictionary_Q','U','U_lower','U_upper','U_error_bounds','exact_column_indices'})
            check(saved['K'].shape==(len(fold['old_inner_train_ids']),)*2 and saved['alpha'].shape[1]==6,'Inner teacher shape differs')
            check(np.array_equal(saved['theta'],b_state['theta']),'Inner teacher does not use frozen actual B theta')
            verify_basis_arrays(saved,rank)
            check(np.array_equal(saved['U'],b_state['U']) and np.array_equal(saved['original_dictionary_Q'],b_state['original_dictionary_Q']),
                'Inner teacher does not use actual B physical basis')
            _array_close(saved['head_theta_padded'],np.pad(saved['theta'],(0,5-rank)),'Inner teacher padded coordinates differ')
    final=archives.reference(stage['final_state_ref'],coords,name,'final');arrays=archives.load(final)
    gate_audits=iter([op['audit'] for op in stage['operation_audits'] if op['operation']=='gate_forward'])
    if not is_b:
        full_labels=arrays['labels'];old_columns=[registry.index(c) for c in path['b_classes']]
        check(full_labels.shape==(len(train),) and full_labels.dtype.kind in 'iu'
            and np.all((full_labels>=0)&(full_labels<len(registry))),'Final legal gate label shape differs')
        check(np.array_equal(np.bincount(full_labels,minlength=len(registry)),np.full(len(registry),path['train_k'])),
            'Final gate training class counts differ')
        if label_mapping:check(full_labels.tolist()==[registry.index(label_mapping[p]) for p in train],
            'Final gate labels differ from physical train mapping')
    verify_basis_arrays(arrays,rank)
    for key in ('original_dictionary_Q','U','U_lower','U_upper','U_error_bounds','exact_column_indices'):
        check(np.array_equal(arrays[key],archives.row_basis_arrays[key]),'Final head uses different row basis: '+key)
    check(arrays['theta'].shape==(rank,) and arrays['head_theta_padded'].shape==(5,),'U-coordinate final state differs')
    anchor=np.asarray(stage['anchor_theta']);check(anchor.shape==(rank,),'U-coordinate anchor missing')
    _array_close(arrays['head_theta_padded'],np.pad(arrays['theta'],(0,5-rank)),'Padded head coordinates differ')
    check(stage['parameter_buffer_bytes']==8*rank,'Actual coordinate buffer bytes differ')
    check(stage['evidence_level']==ALGORITHM['evidence_level'] and stage['complete_head_jvp_error_bound'] is None,
        'Unsupported complete-head certification')
    check(stage['actual_updated_coordinate_count']==int(np.count_nonzero(arrays['theta']-anchor)),'Updated coordinate count differs')
    check(np.array_equal(arrays['theta'],np.asarray(stage['theta'])),'Saved/audited final theta differs')
    check(_number(stage['resident_numeric_state_bytes'],True) and stage['deployment_numeric_state_bytes']==stage['resident_numeric_state_bytes'],
        'Retained numeric state byte scope differs')
    if is_b:
        check(np.array_equal(anchor,np.zeros(rank)) and arrays['alpha'].shape==(len(train),6) and arrays['intercept'].shape==(6,),
            'Independent B zero anchor/free intercept differs')
    else:
        q=len(registry)-6;n=len(train);p=q*path['train_k']
        check(b_state is not None and np.array_equal(anchor,b_state['theta']),'C did not inherit actual B coordinates')
        check(np.array_equal(arrays['U'],b_state['U']) and np.array_equal(arrays['original_dictionary_Q'],b_state['original_dictionary_Q']),
            'C physical basis differs from actual B')
        check(arrays['new_alpha'].shape==(p,q) and arrays['new_intercept'].shape==(q,) and arrays['gate_alpha'].shape==(n,)
            and arrays['gate_b'].shape==() and arrays['gate_zeta'].shape==(),'Complete C heads/free constants differ')
        check(arrays['gate_slacks'].shape==(6*path['train_k'],q) and np.all(arrays['gate_slacks']>0),'Incomplete/invalid all-pair gate')
        old_positions=np.asarray([i for i,pid in enumerate(train) if pid in set(path['b_training_ids'])],dtype=int)
        check(np.array_equal(arrays['old_indices'],old_positions) and np.array_equal(arrays['gate_old_indices'],old_positions)
            and np.array_equal(arrays['gate_targets'],np.isin(np.arange(n),old_positions).astype(float)),'Gate physical old/new groups differ')
        check(arrays['lower_bounds'].shape==(len(old_positions),q),'Full all-pair bounds shape differs')
        for key,value in b_state.items():
            check('actual_B_'+key in arrays and np.array_equal(arrays['actual_B_'+key],value),'Actual B numeric inheritance differs: '+key)
    trials=stage['trials'];check(len(trials)==stage['trial_count']<=12 and stage['optimizer_steps'] in (0,1),'GGN/trial budget differs')
    accepted=0
    for i,trial in enumerate(trials):
        check(trial['trial']==i and trial['step_size']==.5**i,'Fixed halving sequence differs')
        check(type(trial['accepted']) is bool and trial['accepted']==(trial['loss_after']<=trial['armijo_rhs']+trial['comparison_tolerance']),
            'Recorded floating Armijo acceptance differs')
        check(trial['real_inequality_holds']==(trial['loss_after']<=trial['armijo_rhs']) and
            trial['observed_objective_increase']==(trial['loss_after']>trial['loss_before']),'Real/roundoff objective distinction differs')
        accepted+=trial['accepted'];check(not accepted or i==len(trials)-1,'Trials continued after acceptance')
    check(accepted==stage['optimizer_steps'],'Actual accepted update differs')
    if not eligible:
        check(not trials and stage['initial_objective'] is None and stage['final_objective'] is None and
            np.array_equal(arrays['theta'],anchor),'K1/rank0 fabricated OOF/update')
    else:
        check(stage['initial_objective']['RMSCE']>0 and stage['final_objective']['RMSCE']>0,'Missing genuine training objective')
        check(stage['initial_objective']['loss_proximal']==stage['final_objective']['loss_proximal']==0,'Damping became training loss')
        initial=archives.load(archives.reference(stage['initial_state_ref'],coords,name,'initial'))
        verify_basis_arrays(initial,rank)
        check(np.array_equal(initial['U'],arrays['U']),'Initial/final U changed within stage')
        check(initial['gradient'].shape==(rank,) and initial['curvature'].shape==(rank,rank), 'Complete physical gradient/GGN shape differs')
        verify_objective_arrays(initial,rank,train,registry,label_mapping,stage['initial_objective']['RMSCE'])
        if not is_b:
            for readback in _verify_weighted_objective(initial,prep,train,full_labels,old_columns,gate_audits):
                archives.weighted_gate_readbacks.append(dict(coords,state=name,head_phase='initial',
                    state_archive=stage['initial_state_ref']['path'],**readback))
        check(np.array_equal(initial['theta'],anchor),'Initial metric anchor differs')
        ns=dict(coords,state=name)
        # Match the actual manifest namespace rather than reconstructing JSON formatting.
        matches=[ref for ref in archives.refs.values() if ref['key']=='metric_direction' and loads(ref['namespace'])==ns]
        check(len(matches)==1,'Missing/duplicate metric direction state')
        direction_arrays=archives.load(archives.reference(matches[0],coords,name,'metric_direction'))
        check(stage['support_metric_step_audit']['max_secular_iterations']==resources['max_secular_iterations'] and
            stage['support_metric_step_audit']['secular_iteration_count']<=resources['max_secular_iterations'],
            'Metric step resource contract differs')
        direction,M=verify_metric_direction(initial,direction_arrays,stage['support_metric_step_audit'],rank)
        check(stage['quadratic_multiplier']==float(direction_arrays['multiplier']),'Audited multiplier differs')
        if np.any(direction):
            check(bool(trials) and (accepted==1 or len(trials)==12),'Nonzero metric direction lacks actual bounded trial evidence')
        else:check(not trials and accepted==0,'Zero metric direction fabricated trial/update')
        last=anchor;final_loss=stage['initial_objective']['RMSCE']
        for index,trial in enumerate(trials):
            saved=archives.load(archives.reference(trial['state_ref'],coords,name,'trial_'+str(index)))
            check(saved['theta'].shape==(rank,) and saved['scores'].shape==(len(train),len(registry))
                and saved['labels'].shape==(len(train),),'Trial objective physical shape differs')
            attempt=anchor+trial['step_size']*direction;delta=attempt-anchor
            _array_close(saved['theta'],attempt,'Trial physical coordinates differ')
            verify_objective_arrays(saved,rank,train,registry,label_mapping,trial['loss_after'])
            if not is_b:
                for readback in _verify_weighted_objective(saved,prep,train,full_labels,old_columns,gate_audits):
                    archives.weighted_gate_readbacks.append(dict(coords,state=name,head_phase='trial_'+str(index),
                        state_archive=trial['state_ref']['path'],**readback))
            rhs=stage['initial_objective']['RMSCE']+1e-4*float(initial['gradient']@delta)
            _equal(trial['loss_before'],stage['initial_objective']['RMSCE'],'Trial changed frozen anchor objective')
            _array_close(np.asarray(trial['armijo_rhs']),np.asarray(rhs),'Actual delta Armijo differs')
            tolerance=128*EPS*max(1.,abs(trial['loss_before']),abs(trial['loss_after']),abs(rhs))
            _equal(float(trial['comparison_tolerance']/tolerance),1.,'Trial arithmetic tolerance changed')
            _array_close(np.asarray(trial['metric_ball_value']),np.asarray(delta@M@delta),'Trial metric ball differs')
            if trial['accepted']:last=attempt;final_loss=trial['loss_after']
        _array_close(arrays['theta'],last,'Final theta not final accepted state')
        _equal(stage['final_objective']['RMSCE'],final_loss,'Final objective not final accepted readback')
    if not is_b:
        gate_audit=next(gate_audits,None);check(gate_audit is not None,'Missing final weighted gate audit')
        readback=verify_weighted_gate(arrays,full_labels,old_columns,gate_audit)
        archives.weighted_gate_readbacks.append(dict(coords,state=name,head_phase='final',inner_fold=None,
            state_archive=stage['final_state_ref']['path'],**readback))
    check(next(gate_audits,None) is None,'Unexpected/unread weighted gate operation')
    return arrays


def verify_parent(parent,identity,physical,*,row_id,run_id,archives,stage_stream,prediction_stream,resources,ground_binding):
    """No accuracy is computed here; validate all physical fixed evidence first."""
    check(parent['schema']==SCHEMA and parent['method']==METHOD and parent['scope']==SCOPE and
        parent['query_rows_used']==parent['source_rows_used']==0,'Parent scope/access differs')
    check(all(parent[k]==identity[k] for k in IDENTITY),'Parent identity differs')
    check(parent['inheritance_binding']==dict(run_id=run_id,row_id=row_id,split_id=identity['split_id']),
        'Parent run/row inheritance binding differs')
    construction=verify_row_basis(parent['row_basis_audit'],archives,resources)
    check(parent['row_basis_ref']==archives.row_basis_ref and parent['row_basis_rank']==archives.row_basis_owner['row_basis_rank'] and
        parent['basis_certificate']==archives.row_basis_owner['basis_certificate'] and parent['row_basis_construction_status']=='COMPLETE',
        'Parent row basis identity differs')
    classes=sorted(identity['registered_classes']);old=sorted(identity['registered_classes'][:6]);k=identity['k'];q=identity['new_count']
    check(parent['classes']==classes and parent['old_classes']==old and parent['support_count']==k*len(classes)
        and parent['old_class_count']==6 and parent['new_class_count']==q,'Parent registry/count differs')
    physical=sorted(physical);check(len(physical)==len(set(physical))==k*len(classes),'Support physical identity count differs')
    mapping={};expected_paths=[]
    if k==1:
        check(parent['fold_count']==0 and parent['folds']==[] and parent['oof'] is None and parent['oneshot_proxy'] is None
            and parent['physical_fold_assignment']==[] and parent['full_support'] is not None,'K1 held evidence fabricated')
        expected_paths=[('support_full_k1',None,None,set(physical),set())];paths=[parent['full_support']]
    else:
        check(parent['fold_count']==min(k,3) and parent['full_support'] is None,'OOF fold structure differs')
        assignment=parent['physical_fold_assignment'];check([r['physical_id'] for r in assignment]==physical,'Physical assignment order differs')
        mapping={r['physical_id']:r['class_id'] for r in assignment}
        grouped={c:sorted(pid for pid in physical if mapping[pid]==c) for c in classes}
        check(all(len(v)==k for v in grouped.values()),'Unbalanced physical class registry')
        membership={pid:i%min(k,3) for ids in grouped.values() for i,pid in enumerate(ids)}
        check(all(r['fold']==membership[r['physical_id']] for r in assignment),'Physical ID fold rule differs')
        for f in range(min(k,3)):
            held={pid for pid in physical if membership[pid]==f}
            expected_paths.append(('support_oof',f,None,set(physical)-held,held))
        for trial in range(k):
            train={ids[trial] for ids in grouped.values()}
            expected_paths.append(('support_oneshot_proxy',None,trial,train,set(physical)-train))
        check(parent['oneshot_proxy']['trial_count']==k and parent['oneshot_proxy']['proxy_train_k']==1,'All anchor proxy required')
        paths=parent['folds']+parent['oneshot_proxy']['trials']
    check(len(paths)==len(expected_paths),'Missing/extra physical path')
    counters=_empty_counters();phase={name:_empty_work() for name in ('row_basis','preparation','stage','score')};saved_paths=[];old_physical=None
    own=parent['row_basis_construction_count'];check(type(own) is int and own in (0,1),'Invalid parent basis ownership count')
    counters['row_basis_construction_count']=own
    if own:phase['row_basis']=construction.copy()
    for path,(scope,fold,trial,train,held) in zip(paths,expected_paths):
        check(path['scope']==scope and path['fold']==fold and path['trial']==trial and path['parent_k']==k and
            path['train_k']==len(train)//len(classes) and path['held_k']==len(held)//len(classes),'Path coordinates/trainK differ')
        check(path['c_training_ids']==sorted(train) and path['c_ids']==sorted(held) and path['b_classes']==old and path['c_classes']==classes,
            'Path physical partition/registry differs')
        check(set(path['b_training_ids'])<=train and set(path['b_ids'])<=held and len(path['b_training_ids'])==6*path['train_k']
            and len(path['b_ids'])==6*path['held_k'],'Old physical partition differs')
        current=set(path['b_training_ids'])|set(path['b_ids'])
        if old_physical is None:old_physical=current
        check(current==old_physical,'Old physical population changed across paths')
        if k>1:
            check(current=={pid for pid,c in mapping.items() if c in old} and path['held_labels']=={pid:mapping[pid] for pid in sorted(held)},
                'Legal held label/registry physical binding differs')
        else:check(path['held_labels']=={},'K1 held labels fabricated')
        coords=_coordinates(row_id,run_id,parent,path);archives.walk(path,coords)
        reuse=q==0;check(path['c_reuses_b0'] is reuse and path['c_reuses_b_candidates'] is reuse,'new0 reuse declaration differs')
        check(set(path['paths'])==set(PATHS),'Missing baseline/candidate fixed evidence')
        expected_bases=['B0'] if reuse else ['B0','C0'];expected_states=['B_GROUP_BALANCED_SUPPORT_METRIC'] if reuse else ['B_GROUP_BALANCED_SUPPORT_METRIC','C_GROUP_BALANCED_SUPPORT_METRIC_seq']
        check([s['state'] for s in path['stages']]==expected_bases and [s['state'] for s in path['preparations']]==(['B'] if reuse else ['B','C'])
            and [s['state'] for s in path['candidate_stages']]==expected_states,'Actual head stages differ')
        logs=[]
        for base in path['stages']:
            b=base['state']=='B0';ids=path['b_training_ids'] if b else path['c_training_ids'];c=6 if b else len(classes)
            check(base['training_physical_ids']==ids and all(base.get(key)==value for key,value in coords.items()),'Baseline identity differs')
            arrays=archives.load(archives.reference(base['final_state_ref'],coords,base['state'],'baseline'))
            check(arrays['alpha'].shape==(len(ids),c),'Baseline head array shape differs')
            counters['baseline_head_fit_count']+=1;counters['baseline_factorization_count']+=base['factorization_calls']
            counters['baseline_triangular_solve_count']+=2*base['factorization_calls']+base['effective_degrees_of_freedom_extra_triangular_solves']
            if held:counters['final_score_evaluation_count']+=1;counters['final_score_physical_count']+=len(path['b_ids'] if b else path['c_ids'])
            logs.append(dict(event='BASE_FIT',**base))
        B=None;bref=None;parameter_counts=[]
        for prep,stage in zip(path['preparations'],path['candidate_stages']):
            arrays=_verify_stage(stage,prep,path,coords,archives,resources,B,mapping)
            is_b=stage['state']=='B_GROUP_BALANCED_SUPPORT_METRIC'
            parameter_counts.append(dict(state=stage['state'],nominal_adapter_coordinates=5,
                effective_physical_rank=stage['effective_parameter_rank'],
                trainable_adapter_coordinates=stage['trainable_parameter_count'],updated_adapter_coordinates=stage['actual_updated_coordinate_count'],
                analytic_ridge_coefficients=int(arrays['alpha' if is_b else 'new_alpha'].size),
                analytic_ridge_intercepts=int(arrays['intercept' if is_b else 'new_intercept'].size),
                gate_coefficients=0 if is_b else int(arrays['gate_alpha'].size),gate_intercepts=0 if is_b else 1,
                resident_numeric_state_bytes=stage['resident_numeric_state_bytes'],
                metric_stationarity_relative_residual=stage.get('support_metric_step_audit',{}).get('stationarity_relative_residual'),
                metric_complementarity_relative_residual=stage.get('support_metric_step_audit',{}).get('complementarity_relative_residual'),
                metric_ball_value=stage.get('support_metric_step_audit',{}).get('metric_ball_value'),
                complete_head_jvp_error_bound=None,evidence_level=ALGORITHM['evidence_level']))
            if stage['state']=='B_GROUP_BALANCED_SUPPORT_METRIC':B=arrays;bref=stage['final_state_ref']
            else:check(stage['final_prior_ref']==bref and prep['full_prior_ref']==bref,'Final C old function is not actual path B')
            for name,audit in (('preparation',prep),('stage',stage)):_merge(phase[name],verify_work(audit),set(WORK_MAX))
            counters['candidate_preparation_count']+=1;counters['candidate_stage_count']+=1;counters['final_candidate_head_fit_count']+=1
            counters['optimizer_steps']+=stage['optimizer_steps'];counters['trial_count']+=stage['trial_count']
            counters['accepted_trial_count']+=stage['optimizer_steps'];counters['rejected_trial_count']+=stage['trial_count']-stage['optimizer_steps']
            steps=sum(v['operation']=='support_metric_step' for v in stage['operation_audits']);check(steps<=1,'Multiple GGN updates')
            counters['ggn_step_count']+=steps;counters['ggn_parameter_direction_count']+=stage['effective_parameter_rank']*steps
            counters['peak_resident_numeric_state_bytes']=max(counters['peak_resident_numeric_state_bytes'],stage['resident_numeric_state_bytes'])
            counters['peak_effective_adapter_rank']=max(counters['peak_effective_adapter_rank'],stage['effective_parameter_rank'])
            if held:
                work=verify_work(stage['score_workload']);_merge(phase['score'],work,set(WORK_MAX))
                n=len(path['b_ids'] if stage['state']=='B_GROUP_BALANCED_SUPPORT_METRIC' else path['c_ids'])
                check(stage['score_workload']['physical_record_count']==n,'Single-record actual score coverage differs')
                counters['final_score_evaluation_count']+=n;counters['final_score_physical_count']+=n
                if stage['state']=='C_GROUP_BALANCED_SUPPORT_METRIC_seq':
                    check(stage['structural_predict_workload'] is None and bool(stage['structural_predict_workload_unavailable_reason']),
                        'Unknown public predict internals were fabricated')
                    counters['structural_predict_evaluation_count']+=n;counters['structural_predict_physical_count']+=n
                    counters['structural_predict_seconds']+=stage['structural_predict_seconds']
            else:check(stage['score_workload'] is None,'K1 outer score evidence fabricated')
            logs.extend((dict(event='GROUP_BALANCED_SUPPORT_METRIC_PREPARATION',**prep),dict(event='CANDIDATE_FIT',**stage)))
        for log in logs:
            expected=dict(compact_event(log),schema=SCHEMA,method=METHOD,split_id=parent['split_id'])
            check(next(stage_stream,None)==expected,'Actual preparation/fit stage stream order or contents differ')
        fixed={};ground=path['ground_A'];streams=(('A',) if ground is not None else ())+STREAMS if held else ()
        check((ground is not None)==(bool(held) and ground_binding['status']=='MATCHED_SOURCE_ONLY_PACKET'),'Actual A presence differs')
        if not held:
            check(path['prediction_status']=='NO_HELD_PREDICTIONS' and path['outer_features_state_ref'] is None,'K1 fixed predictions fabricated')
            check(all(v is None for result in path['paths'].values() for v in result['metrics'].values()),'K1 metrics fabricated')
        else:
            check(path['prediction_status']=='FIXED_BEFORE_SUPPORT_TRUTH_JOIN','Predictions not fixed')
            arrays=archives.load(archives.reference(path['fixed_prediction_state_ref'],coords,'FIXED_SUPPORT_PREDICTIONS','scores'))
            for stream in streams:
                record=next(prediction_stream,None);check(isinstance(record,dict),'Missing fixed prediction stream')
                check('held_labels' not in record and record['status']=='FIXED_BEFORE_SUPPORT_TRUTH_JOIN' and record['stream']==stream,'Fixed prediction status/truth scope differs')
                check(all(record.get(key)==value for key,value in coords.items() if key!='scope') and record['outer_scope']==scope,'Fixed physical prediction namespace differs')
                if stream=='A':
                    ids=path['b_ids'];registry=ground_binding['ordered_classes'];scores=_scores(ground['scores'],ids,registry)
                    check(record['scope']=='CURRENT_LEGAL_OLD_HELD_ONLY_ORIGINAL_SIX_CLASS_COMPETITION' and record['score_dtype']=='float32',
                        'A native scope/dtype differs')
                    check(ground['physical_ids']==ids and ground['classes']==registry and set(registry)==set(old),'Paired native A identity differs')
                    expected=_predictions(scores,registry,True)
                    ar=archives.load(archives.reference(ground['scores_state_ref'],coords,'A_GROUND_FIXED_PREDICTIONS','scores'))
                    check(np.array_equal(ar['scores'],scores.astype(np.float32)),'Native A archived scores differ')
                    check(record['scores_state_ref']==ground['scores_state_ref'] and record['inference_seconds']==ground['inference_seconds']
                        and _number(ground['inference_seconds']),'Native A work/reference binding differs')
                    counters['ground_A_score_evaluation_count']+=len(ids);counters['ground_A_score_physical_count']+=len(ids)
                    counters['ground_A_inference_seconds']+=ground['inference_seconds']
                else:
                    name,side=stream.rsplit('_',1);small=side.lower();ids=path[small+'_ids'];registry=path[small+'_classes']
                    scores=_scores(path['paths'][name][small+'_scores'],ids,registry)
                    check(record['scope']==scope and np.array_equal(arrays[name+'_'+small],scores),'Fixed archived scores differ')
                    expected=_predictions(scores,registry)
                    if name=='R_GROUP_BALANCED_SUPPORT_METRIC_seq' and side=='C':
                        expected=[path['paths'][name]['c_predictions'][pid] for pid in ids]
                        bpred=dict(zip(path['b_ids'],fixed['R_GROUP_BALANCED_SUPPORT_METRIC_seq_B']['predictions']))
                        for pid,pred,values in zip(ids,expected,scores):
                            check(pred in registry,'C predicted unregistered class')
                            tolerance=128*EPS*max(1.,float(np.max(np.abs(values))))
                            check(values[registry.index(pred)]>=float(np.max(values))-tolerance,'Public C prediction incompatible with full-class scores')
                            if pid in bpred and pred in old:check(pred==bpred[pid],'C old winner differs from actual paired B')
                check(record['physical_ids']==ids and record['classes']==registry and record['predictions']==expected
                    and np.array_equal(_scores(record['scores'],ids,registry),scores),'Fixed stream/trace scores or decisions differ')
                fixed[stream]=record
            if reuse:
                for name in PATHS:
                    check(fixed[name+'_B']['scores']==fixed[name+'_C']['scores'] and fixed[name+'_B']['predictions']==fixed[name+'_C']['predictions'],
                        'new0 did not exactly reuse B predictions')
        counters['sequence_paths']+=1
        saved_paths.append(dict(path=path,fixed=fixed,parameter_counts=parameter_counts))
    for name,value in phase.items():_equal(parent[name+'_actual_work'],value,'Parent '+name+' work closure')
    total=_empty_work()
    for value in phase.values():_merge(total,value,set(WORK_MAX))
    _equal(parent['actual_work'],total,'Parent total work closure');counters.update(total)
    for key in COUNTERS[4:]:_equal(parent[key],counters[key],'Parent actual counter '+key)
    check(parent['structural_predict_actual_work'] is None and bool(parent['structural_predict_work_unavailable_reason']),'Unknown predict work missing/N/A differs')
    counters.update(episodes=1,k1_episodes=int(k==1),oof_episodes=int(k>1),proxy_anchor_count=k if k>1 else 0)
    return dict(parent=parent,paths=saved_paths,counters=counters,phases=phase,old_physical=sorted(old_physical))


def _oid(value):return isinstance(value,str) and len(value)==40 and all(c in '0123456789abcdef' for c in value)


def _metric_view(verified):
    """Discard large nested training audits after full verification, keep fixed
    held decisions and their legal label binding for the deferred truth join."""
    parent=verified['parent']
    small={key:parent[key] for key in IDENTITY+('old_classes','classes','oof','oneshot_proxy')}
    if parent['oof'] is not None:small['oof']={'paths':{name:{'metrics':parent['oof']['paths'][name]['metrics']} for name in PATHS}}
    if parent['oneshot_proxy'] is not None:
        small['oneshot_proxy']={'parent_mean_metrics':parent['oneshot_proxy']['parent_mean_metrics']}
    paths=[]
    for value in verified['paths']:
        path=value['path']
        item={key:path[key] for key in ('held_labels','c_classes','b_ids','c_ids','scope','fold','trial','parent_k','train_k')}
        item['paths']={name:{'metrics':path['paths'][name]['metrics']} for name in PATHS}
        paths.append(dict(path=item,fixed=value['fixed'],parameter_counts=value['parameter_counts']))
    return dict(parent=small,paths=paths)


def verify_row(root,spec,row,commit):
    """Resolve only the declared completed row artifacts; never open input paths."""
    out=Path(root)/row['row_id']/'probe';startup=read(out/'startup.json');marker=read(out/'probe_complete.json')
    co=spec['probe']['cohorts'][row['cohort']];resources=spec['probe']['support_metric_resources']
    binding=dict(run_id=spec['run_id'],row_id=row['row_id'],capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],release_commit=commit,
        ground_packet=row.get('ground_packet'),ground_summary=row['ground_summary'],
        ground_summary_already_deployed=row['ground_summary_already_deployed'],schema=SCHEMA,method=METHOD,scope=SCOPE)
    for value in (startup,marker):
        check(all(value.get(k)==v for k,v in binding.items()),'Row source/runtime identity differs')
        check(value['support_metric_resources']==resources and value['query_rows_used']==value['source_rows_used']==0
            and value['truth_read'] is False and value['nominal_adapter_parameter_count']==5,'Row input/parameter scope differs')
    check(marker['status']==STATUS and marker['workload_complete'] is True and marker['counters_scope']=='ALL_COMPLETED_EPISODES',
        'Incomplete row cannot be analyzed')
    check(type(startup['pid']) is int and startup['pid']>0 and marker['pid']==startup['pid'],'Row PID binding differs')
    config=dict(algorithm=ALGORITHM,producer_matrix=co['matrix'],selection=co['selection'],support_metric_resources=resources)
    check(startup['config']==config and marker['algorithm']==ALGORITHM and marker['selection']==co['selection']
        and marker['producer_matrix']==co['matrix'],'Resolved full row config differs')
    check(startup['support_features']==row['support_features'] and startup['episodes']==40,'Declared cache/parent count differs')
    for key in ('query_iq_access','checkpoint_loaded','encoder_updated'):check(startup[key] is False,'Forbidden row access '+key)
    check(startup['adapted_state_inherited'] is True and startup['source_validation'] is None and
        startup['source_validation_reason']=='SOURCE_ACCESS_FORBIDDEN' and
        startup['prototype_geometry_role']=='FROZEN_REFERENCE_ONLY_NOT_TEACHER','Ground/teacher scope differs')
    for field in ('payload_audit','ground_A_binding','ground_geometry_binding','selected_support_physical_ids'):
        _equal(marker[field],startup[field],'Stable row '+field)
    payload=startup['payload_audit']
    check(payload['feature_cache_reused'] is True and payload['checkpoint_loaded'] is False and
        payload['native_physical_forward_count_this_run']==payload['new_source_payload_bytes']==payload['newly_generated_ground_statistics_bytes']==0,
        'Unexpected encoder/source payload')
    provenance=startup['provenance'];old=co['selection']['splits'][0]['registered_classes'][:6]
    check(provenance['checkpoint_sha256']==row['expected_checkpoint_sha256'] and provenance['model_seed']==row['seeds']['model'] and
        set(provenance['classes'])==set(old) and provenance['verdict']=='MATCHED_SOURCE_ONLY_SCRATCH' and
        provenance['source_role_comparison']=='EXACT_MATCH' and provenance['checkpoint_epoch']==200 and
        provenance['checkpoint_inheritance']==[] and provenance['target_access_before_freeze'] is False,'Existing source-only cache verdict differs')
    geometry=startup['ground_geometry_binding']
    check(geometry['operation']=='load_existing_ground_center_fixed_proto_frame' and
        geometry['center_policy']=='phase1_offline_global_maximin_fixed_before_target_access' and
        all(geometry[k] is False for k in ('source_samples_read','source_per_record_features_read','checkpoint_loaded',
            'residual_domain_reconstruction','prototype_updated','prototype_teacher_targets','support_labels_read','query_read')),
        'Frozen geometry-only input scope differs')
    ground=startup['ground_A_binding']
    if row.get('ground_packet') is None:
        check(ground['status']=='N/A','Undeclared Ground A evidence')
    else:
        check(ground['status']=='MATCHED_SOURCE_ONLY_PACKET' and ground['path']==row['ground_packet'] and
            ground['checkpoint_sha256']==row['expected_checkpoint_sha256'] and ground['model_seed']==row['seeds']['model'],
            'Native A source binding differs')
    selected=startup['selected_support_physical_ids'];identities=co['selection']['splits']
    check(set(selected)=={v['split_id'] for v in identities},'Selected physical parent coverage differs')
    archive=Archives(out,spec['run_id'],row['row_id']);stages=iter(jsonl(out/'fit_stages.jsonl'));fixed=iter(jsonl(out/'fixed_predictions.jsonl'))
    trace=iter(jsonl(out/'fit_trace.jsonl'));parents=[];total=_empty_counters()
    check(marker['row_basis']=='row_basis.json','Row basis descriptor path differs')
    owner=read(out/'row_basis.json');construction=verify_row_basis(owner,archive,resources)
    for key in ('row_basis_audit','row_basis_ref','row_basis_rank','basis_certificate'):
        expected=owner if key=='row_basis_audit' else owner[key]
        _equal(marker[key],expected,'Row basis marker differs: '+key)
    check(startup['row_basis_construction_status']=='PENDING_EXPLICIT_FROZEN_Q_FACTORY' and
        startup['row_basis_namespace']==archive.row_basis_coords,'Initial row basis declaration differs')
    total['row_basis_construction_count']=1;total['peak_effective_adapter_rank']=owner['row_basis_rank'];_merge(total,construction)
    phases={key:_empty_work() for key in ('row_basis','preparation','stage','score')};phases['row_basis']=construction.copy();old_bindings={}
    for identity in identities:
        parent=next(trace,None);check(parent is not None,'Missing full parent trace')
        verified=verify_parent(parent,identity,selected[identity['split_id']],row_id=row['row_id'],run_id=spec['run_id'],
            archives=archive,stage_stream=stages,prediction_stream=fixed,resources=resources,ground_binding=ground)
        check(verified['counters']['row_basis_construction_count']==0,'Row basis construction recharged per parent')
        parents.append(_metric_view(verified));_merge(total,verified['counters'])
        for key in phases:_merge(phases[key],verified['phases'][key],set(WORK_MAX))
        key=tuple(identity[k] for k in ('receiver','scenario','k','support_seed'))
        check(old_bindings.setdefault(key,verified['old_physical'])==verified['old_physical'],'Old physical support changed across new counts')
    check(next(trace,None) is None and next(stages,None) is None and next(fixed,None) is None,'Undeclared extra parent/stage/prediction')
    archive.close()
    _equal({k:marker[k] for k in COUNTERS},total,'Row measured counter closure')
    for name,value in phases.items():_equal(marker[name+'_actual_work'],value,'Row phase work closure')
    _equal(marker['actual_work'],{k:total[k] for k in WORK_KEYS},'Row work total closure')
    check(marker['actual_work_aggregation']=={k:'MAX' if k in WORK_MAX else 'SUM' for k in WORK_KEYS},'Work aggregation differs')
    for key in WORK_MAX:
        if key.endswith('peak_factor_buffer_bytes'):check(total[key]<=resources['max_factor_buffer_bytes'],'Owned gate factor buffer exceeds declared cap')
    for field,key in (('state_archive_file_count','file_count'),('state_archive_file_bytes','total_file_bytes'),
                      ('state_archive_numeric_bytes','numeric_array_bytes'),('state_archive_seconds','archive_seconds')):
        _equal(marker[field],archive.manifest[key],'Archive marker closure')
    check(marker['structural_predict_actual_work'] is None and bool(marker['structural_predict_work_unavailable_reason']),
        'Unmeasured public prediction work must remain N/A')
    check(marker['peak_gpu_memory_bytes'] is None and _number(marker['wall_seconds']) and
        (marker['peak_process_rss_bytes'] is None or _number(marker['peak_process_rss_bytes'],True)),'Measured resource scope differs')
    return dict(row=row,startup=startup,marker=marker,parents=parents,counters=total,archive=archive.manifest,row_basis=owner,
        weighted_gate_readbacks=archive.weighted_gate_readbacks,
        input_artifact_bytes=sum((out/name).stat().st_size for name in
            ('startup.json','probe_complete.json','fit_trace.jsonl','fit_stages.jsonl','fixed_predictions.jsonl','state_manifest.json',
             'row_basis.json','basis_certificate.json')))


def verify_run(*,spec,run_root,expected_runtime_commit):
    """Complete every fixed-prediction validation before returning any truth join."""
    validate_spec(spec);check(_oid(expected_runtime_commit),'Actual runtime commit must be explicit 40hex')
    root=Path(run_root);startup=read(root/'startup.json');complete=read(root/'complete.json')
    for value in (startup,complete):
        check(all(value[k]==v for k,v in dict(schema=SCHEMA,method=METHOD,run_id=spec['run_id'],group_id=spec['group_id'],
            runtime_commit=expected_runtime_commit,code_commit=spec['code']['commit']).items()),'Supervisor identity differs')
        check(all(value[k] is False for k in ('query_access','truth_read','source_sample_access','scorer_invoked')),'Supervisor input access differs')
    check(startup['status']=='GROUP_BALANCED_SUPPORT_METRIC_JOINT_SUPERVISOR_STARTED' and startup['resolved_spec']==spec,'Full resolved spec/startup differs')
    check(complete['status']==STATUS and complete['workload_complete'] is True and complete['automatic_retry'] is False
        and complete['model_rows']==complete['completed_rows']==4 and complete['pid']==startup['pid'],'Supervisor not fully complete')
    check(set(complete['rows'])=={r['row_id'] for r in spec['rows']},'Supervisor four-row closure differs')
    validated=[];total=_empty_counters();cohort_physical={}
    for row in spec['rows']:
        result=verify_row(root,spec,row,expected_runtime_commit);validated.append(result);_merge(total,result['counters'])
        saved=complete['rows'][row['row_id']]
        check(saved['status']==STATUS and saved['release_commit']==expected_runtime_commit and saved['pid']==result['marker']['pid'],
            'Supervisor/lane process binding differs')
        _equal({k:saved[k] for k in COUNTERS},result['counters'],'Supervisor row counters')
        _equal(saved['actual_work'],result['marker']['actual_work'],'Supervisor row work')
        _equal(complete['actual_work_by_row'][row['row_id']],saved['actual_work'],'Supervisor work-by-row closure')
        physical=result['startup']['selected_support_physical_ids']
        check(cohort_physical.setdefault(row['cohort'],physical)==physical,'Same cohort model rows use different physical support')
    _equal({k:complete[k] for k in COUNTERS},total,'Supervisor all-row actual cost closure')
    check(total['episodes']==160 and total['sequence_paths']==1800,'Incomplete full 160-parent/1800-path run')
    check(total['row_basis_construction_count']==4,'Basis must be constructed once per declared row')
    return dict(rows=validated,counters=total,complete=complete,startup=startup)


METRICS=('A_old_accuracy','B0_old_accuracy','B_old_accuracy','C_old_columns_accuracy','C_old_accuracy','C_new_accuracy','C_h',
    'adaptation_gain_B_minus_A','support_adaptation_B_minus_B0','total_old_accuracy_drop','old_order_change',
    'new_competition_loss','C_abs_new_old_gap','C_new_minus_old','C_old_minus_R0','C_new_minus_R0',
    'B_macro_f1','C_old_macro_f1','C_new_macro_f1','C_all_macro_f1')


def _accuracy(ids,predictions,truth):
    return sum(predictions[pid]==truth[pid] for pid in ids)/len(ids) if ids else None


def _macro(ids,predictions,truth,classes):
    if not ids:return None
    values=[]
    for label in classes:
        tp=sum(truth[i]==label and predictions[i]==label for i in ids)
        fp=sum(truth[i]!=label and predictions[i]==label for i in ids)
        fn=sum(truth[i]==label and predictions[i]!=label for i in ids)
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(values)/len(values)


def _difference(a,b):return None if a is None or b is None else a-b


def score_fixed(paths,name,old):
    """Only called after verify_run has certified every row; no model is invoked."""
    truth={};pred={k:{} for k in ('A','R0_B','R0_C',name+'_B',name+'_C')};oldcol={};registry=None
    for value in paths:
        path=value['path'];fixed=value['fixed']
        check(not(set(truth)&set(path['held_labels'])),'Repeated physical held observation inside pooled parent')
        truth.update(path['held_labels']);registry=path['c_classes']
        for stream in pred:
            if stream in fixed:pred[stream].update(zip(fixed[stream]['physical_ids'],fixed[stream]['predictions']))
        if name=='R_GROUP_BALANCED_SUPPORT_METRIC_seq':oldcol.update(pred[name+'_B'])
        elif fixed:
            record=fixed[name+'_C'];cols=[record['classes'].index(c) for c in old]
            values=np.asarray(record['scores'])[:,cols]
            oldcol.update(zip(record['physical_ids'],_predictions(values,old)))
    if not truth:return dict.fromkeys(METRICS,None)
    oi=sorted(i for i in truth if truth[i] in old);ni=sorted(set(truth)-set(oi))
    a=_accuracy(oi,pred['A'],truth) if pred['A'] else None;b0=_accuracy(oi,pred['R0_B'],truth)
    b=_accuracy(oi,pred[name+'_B'],truth);co=_accuracy(oi,pred[name+'_C'],truth);cn=_accuracy(ni,pred[name+'_C'],truth)
    cc=_accuracy(oi,oldcol,truth);r0o=_accuracy(oi,pred['R0_C'],truth);r0n=_accuracy(ni,pred['R0_C'],truth)
    h=None if cn is None else (2*co*cn/(co+cn) if co+cn else 0.)
    return dict(zip(METRICS,(a,b0,b,cc,co,cn,h,_difference(b,a),b-b0,b-co,b-cc,cc-co,
        None if cn is None else abs(cn-co),_difference(cn,co),co-r0o,_difference(cn,r0n),
        _macro(oi,pred[name+'_B'],truth,old),_macro(oi,pred[name+'_C'],truth,old),
        _macro(ni,pred[name+'_C'],truth,[c for c in registry if c not in old]),
        _macro(sorted(truth),pred[name+'_C'],truth,registry))))


def _mean(values):
    check(not values or all(v is None for v in values) or all(v is not None for v in values),'Mixed missing/observed parent evidence')
    return sum(values)/len(values) if values and values[0] is not None else None


def derive_tables(verified):
    parents=[];paths=[]
    for row in verified['rows']:
        metadata=dict(row_id=row['row']['row_id'],model_seed=row['row']['seeds']['model'],cohort=row['row']['cohort'])
        for item in row['parents']:
            parent=item['parent'];old=parent['old_classes'];base=dict(metadata,**{k:parent[k] for k in IDENTITY})
            for value in item['paths']:
                path=value['path']
                for name in PATHS:
                    metrics=score_fixed([value],name,old)
                    recorded=path['paths'][name]['metrics']
                    for key in set(recorded)&set(metrics):_equal(recorded[key],metrics[key],'Stored path metric readback '+key)
                    paths.append(dict(base,path=name,scope=path['scope'],fold=path['fold'],trial=path['trial'],
                        parent_k=path['parent_k'],train_k=path['train_k'],held_count=len(path['c_ids']),**metrics))
            for scope in ('support_oof','support_oneshot_proxy') if parent['k']>1 else ('support_full_k1',):
                selected=[value for value in item['paths'] if value['path']['scope']==scope]
                for name in PATHS:
                    if scope=='support_oneshot_proxy':
                        pieces=[score_fixed([value],name,old) for value in selected]
                        metrics={k:_mean([v[k] for v in pieces]) for k in METRICS}
                    else:metrics=score_fixed(selected,name,old)
                    if parent['k']>1:
                        recorded=(parent['oneshot_proxy']['parent_mean_metrics'][name] if scope=='support_oneshot_proxy'
                            else parent['oof']['paths'][name]['metrics'])
                        for key in set(recorded)&set(metrics):_equal(recorded[key],metrics[key],'Stored parent metric readback '+key)
                    parents.append(dict(base,path=name,scope=scope,anchor_count=len(selected),**metrics))
    tables={}
    for table,keys in dict(overall=('scope','path'),by_k_new_count=('scope','path','k','new_count'),
        by_row=('scope','path','row_id','model_seed','cohort'),
        by_receiver_scene=('scope','path','receiver','scenario'),by_model_seed=('scope','path','model_seed')).items():
        groups=defaultdict(list)
        for record in parents:groups[tuple(record[k] for k in keys)].append(record)
        values=[]
        for group,records in sorted(groups.items()):
            for metric in METRICS:
                observed=[r[metric] for r in records if r[metric] is not None];n=len(observed)
                mean=sum(observed)/n if n else None
                sd=math.sqrt(sum((v-mean)**2 for v in observed)/(n-1)) if n>1 else None
                values.append(dict(zip(keys,group),metric=metric,mean=mean,sd=sd,standard_error=sd/math.sqrt(n) if sd is not None else None,
                    parent_count=len(records),observed_parent_count=n))
        tables[table]=values
    groups=defaultdict(lambda:defaultdict(list))
    for r in parents:groups[(r['scope'],r['path'],r['k'],r['new_count'])][r['model_seed']].append(r)
    tables['paired_seed_means']=[]
    for key,byseed in sorted(groups.items()):
        check(len(byseed)==2,'Two declared model seeds required for seed mean/SD')
        for metric in METRICS:
            means={str(seed):_mean([v[metric] for v in values]) for seed,values in sorted(byseed.items())}
            vals=list(means.values());mean=_mean(vals)
            sd=math.sqrt(sum((v-mean)**2 for v in vals)) if mean is not None else None
            tables['paired_seed_means'].append(dict(zip(('scope','path','k','new_count'),key),metric=metric,
                seed_2026092701=means['2026092701'],seed_2026092702=means['2026092702'],mean=mean,sd=sd,seed_count=2))
    return parents,paths,tables


def parameter_table(verified):
    values=[]
    for row in verified['rows']:
        for parent in row['parents']:
            for value in parent['paths']:
                path=value['path']
                for counts in value['parameter_counts']:
                    values.append(dict(row_id=row['row']['row_id'],split_id=parent['parent']['split_id'],
                        **{k:path[k] for k in ('scope','fold','trial','parent_k','train_k')},**counts))
    return values


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)


def write_table(out,name,rows):
    fields=sorted({key for row in rows for key in row})
    with (out/(name+'.jsonl')).open('x',encoding='utf-8') as lines,(out/(name+'.csv')).open('x',encoding='utf-8',newline='') as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=fields);writer.writeheader()
        for row in rows:
            check(all(v is None or type(v) in (str,int,float,bool) for v in row.values()),'Non-scalar report field')
            lines.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');writer.writerow(row)


def analyze(*,spec,run_root=None,output,expected_runtime_commit):
    spec=read(spec) if isinstance(spec,(str,Path)) else spec;root=Path(run_root or spec['execution']['remote_run_root']);out=Path(output)
    check(not out.resolve().is_relative_to(root.resolve()),'Analysis output must not modify the original run')
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        verified=verify_run(spec=spec,run_root=root,expected_runtime_commit=expected_runtime_commit)
        parents,paths,tables=derive_tables(verified)
        parameters=parameter_table(verified)
        gate_readbacks=[v for row in verified['rows'] for v in row['weighted_gate_readbacks']]
        for name,rows in dict(parents=parents,paths=paths,parameters=parameters,weighted_gate_readbacks=gate_readbacks,**tables).items():write_table(out,name,rows)
        resources=dict(actual_counters=verified['counters'],aggregation={k:'MAX' if k in PEAKS else 'SUM' for k in COUNTERS},
            rows=[dict(row_id=r['row']['row_id'],wall_seconds=r['marker']['wall_seconds'],
                peak_process_rss_bytes=r['marker']['peak_process_rss_bytes'],peak_gpu_memory_bytes=None,
                resident_numeric_state_bytes=r['marker']['persistent_state_bytes'],state_archive_file_bytes=r['archive']['total_file_bytes'],
                state_archive_numeric_bytes=r['archive']['numeric_array_bytes'],state_archive_seconds=r['archive']['archive_seconds'],
                ground_A_binding=r['startup']['ground_A_binding'],ground_geometry_binding=r['startup']['ground_geometry_binding'],
                payload_audit=r['startup']['payload_audit'],row_basis=r['row_basis'],
                basis_construction_actual_work=r['marker']['row_basis_actual_work'],
                basis_certificate_file_bytes=r['row_basis']['basis_certificate']['file_bytes'],
                basis_certificate_archive_seconds=r['row_basis']['basis_certificate']['archive_seconds'],
                analysis_input_artifact_bytes=r['input_artifact_bytes']) for r in verified['rows']],
            structural_predict_internal_work=None,structural_predict_internal_work_reason='PUBLIC_PREDICT_RETURNS_NO_INTERNAL_LEDGER',
            source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
            timing_scope='CPU measurement; fit wall includes state callbacks/archive writes; operation times are separately measured and not disjoint total wall',
            factor_buffer_scope='Gate-owned explicit H/factor buffer and small step single-factor input/output peak are distinct; neither is process RSS or full resident state',
            basis_scope='Frozen Q construction once per row; preparation binding and actual Gram work separate; no construction recharging per parent or stage',
            parameter_scope='parameters.jsonl lists every actually fitted stage; analytic coefficients/intercepts are not gradient-trained coordinates; new0 has no second fit',
            bytes_scope='Physical files and logical numeric buffers reported separately; neither is measured transmission or deployment package',
            device_energy=None,spaceborne_cost=None)
        summary=dict(status=SUMMARY_STATUS,summary_schema=SUMMARY_SCHEMA,schema=SCHEMA,method=METHOD,scope=SCOPE,
            run_id=spec['run_id'],release_commit=expected_runtime_commit,algorithm=ALGORITHM,
            support_metric_resources=spec['probe']['support_metric_resources'],coverage=dict(rows=4,parents=160,sequence_paths=1800,
                metric_parent_rows=len(parents),metric_path_rows=len(paths)),statistics=tables,resources=resources,
            query_rows_used=0,source_rows_used=0,external_truth_read=False,model_calls=0,
            validation_scope='Complete saved metadata/numeric closure and fixed decisions; saved RMSCE/GGN/Fisher/actual Gram/metric KKT and legal-train weighted gate objective/stationarity/intercept readback; no independent kernel/JVP replay or complete-head interval certificate',
            numerical_evidence_level=ALGORITHM['evidence_level'],complete_head_interval_certificate=False,
            weighted_gate_readback_count=len(gate_readbacks),
            weighted_gate_readback_scope='Every completed initial/trial inner head and final C head; independent saved-equation analysis, excluded from training work',
            basis_evidence_scope='Declared exact-rank certificate metadata/column closure and stored U enclosure/Gram binding; no repetition of Fraction construction',
            C_decision_limit='Uses fixed public structural predictions; common-offset rounding ties are tolerated at float64 machine scale; old winner must match paired actual B',
            metric_scope='OOF pooled within physical parent; every proxy anchor scored then averaged within parent; equal parent aggregation; no query generalization claim',
            actual_A_scope='Original native six-class frozen head on the identical old-held physical IDs; absent packet remains N/A',
            automatic_promotion=False,conversion_seconds=time.perf_counter()-started,
            output_table_file_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),
            output_bytes_scope='Completed JSONL/CSV tables only; excludes summary.json and compact.json self-reference')
        write_json(out/'summary.json',summary)
        write_json(out/'compact.json',{k:summary[k] for k in ('status','schema','method','run_id','release_commit','coverage','query_rows_used','source_rows_used','validation_scope')})
        return summary
    except Exception as exc:
        write_json(out/'analysis_failed.json',dict(status='FAILED',error_type=type(exc).__name__,error=str(exc),
            run_id=spec.get('run_id'),expected_runtime_commit=expected_runtime_commit,automatic_retry=False,
            scope='Preserved partial analysis only; no complete certification'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','output','expected-runtime-commit'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--run-root');analyze(**vars(parser.parse_args()))


if __name__=='__main__':main()
