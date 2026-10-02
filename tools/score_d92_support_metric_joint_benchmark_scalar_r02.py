"""Independent truth-last scoring of every frozen SupportMetric A/B/C/R0 stream.

This module never fits, imports a model, loads a ground packet or opens features.
It reads capsule opaque IDs and only declared numeric training-state archives.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

import numpy as np

METHOD='D92-ProtoFrameSupportMetric-GGN1-LocalRidge'
PREDICTION_SCHEMA='d92_support_metric_joint_query_predictions_v1'
SCHEMA='d92_support_metric_joint_query_score_v1'
STATUS='SUPPORT_METRIC_JOINT_QUERY_SCORE_COMPLETE'
SCOPE='FROZEN_QUERY_BENCHMARK_TRUTH_LAST_THREE_STAGE_PAIRED'
KS=(1,5,10,20)
NEWS=(0,2,5,10,20)
METRICS=('A_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy','C_h',
    'adaptation_gain_B_minus_A','total_old_accuracy_drop','C_abs_new_old_gap',
    'A_old_macro_f1','B_old_macro_f1','C_old_macro_f1','C_new_macro_f1','C_macro_f1',
    'R0_B_old_accuracy','R0_C_old_accuracy','R0_C_new_accuracy','R0_C_h',
    'C_old_minus_R0','C_new_minus_R0','C_h_minus_R0')
IDENTITY=('run_id','row_id','capsule_id','checkpoint_sha256','model_seed','release_commit')
SPLIT_IDENTITY=('split_id','receiver','scenario','k','new_count','support_seed')
STREAMS={s:'predictions_'+s+'.jsonl' for s in ('A','B','C','R0_B','R0_C')}
PRIMITIVE_SUM=('frame_construction_count','physical_evaluation_count','factorization_attempts','factorizations_completed',
    'triangular_calls','triangular_rhs_columns','triangular_rhs_elements','triangular_dense_work_units',
    'distance_evaluation_count','distance_pair_count','distance_qr_batch_count','distance_qr_pair_count',
    'kernel_evaluation_count','kernel_pair_count','jacobian_evaluation_count','spectral_check_count',
    'eigendecomposition_count','secular_iteration_count','secular_evaluation_count','seconds')
GATE_SUM=('factorization_attempts','factorizations_completed','condition_estimation_calls','triangular_calls',
    'triangular_rhs_columns','triangular_rhs_elements','triangular_dense_work_units','spectral_checks',
    'spectral_cubic_dimension_units','objective_evaluations','logistic_record_evaluations','barrier_constraint_evaluations',
    'line_search_trials','newton_iterations','accepted_steps','round_off_residual_acceptances',
    'factorization_seconds','triangular_seconds','spectral_seconds','objective_seconds','wall_seconds')
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

WORK_SUM_KEYS=WORK_SUM
WORK_MAX_KEYS=WORK_MAX
PEAK_COUNTERS=frozenset(WORK_MAX)
PRIMITIVE_COUNTS=PRIMITIVE_SUM[:-1]
MODEL_SEEDS=(2026092701,2026092702)
COHORT_COUNTS={'rx3':900,'rx1':300}
RECEIVERS={'rx3':['19-1','8-14','8-7'],'rx1':['20-19']}
SCENARIOS=['practical_high','practical_mid','practical_low_urban']
SUPPORT_SEEDS=list(range(2026092711,2026092716))
FROZEN_RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160,max_integer_bits=65536,max_fraction_operations=65536,max_secular_iterations=128)
DECISION_SCHEMA='d92_support_metric_single_query_decision_v1'
DECISION_POLICY='WITHIN_GROUP_RAW_ARGMAX_THEN_GATE_LOGPROB_GAP_LEXICAL_TIE'
# Independent literal method contract, never imported from a fitting module.
FROZEN_ALGORITHM=dict(schema='d92_support_metric_joint_local_ridge_v1',method=METHOD,nominal_coordinates=5,
    coordinates='EXACT_STORED_Q_RANK_PHYSICAL_U',kappa=.25,ridge_coefficient=1.,
    objective='RMS_class_mean_OOF_CE',damping='ACTUAL_U_GRAM_PLUS_CLASS_BALANCED_PREDICTION_FISHER',
    radius=.5,max_updates=1,max_trials=12,
    initial_step=1.,backtrack=.5,armijo=1e-4,float_comparison_multiplier=128,
    folds='physical_ID_class_rank_mod_min_K_3',prototype_role='FROZEN_GEOMETRY_ONLY',
    C_anchor='ACTUAL_CURRENT_B_U_THETA',old_conditional='FROZEN_ACTUAL_B_FUNCTION',
    kernel='HALF_ORIGINAL_HALF_TANGENT_DISTANCE_FIXED_ORIGINAL_OLD_TRAIN_TAU_GAMMA',
    barrier_average_objective_gap=1e-4,phase1_frozen=True,source_examples=False,query_fit=False,
    parameter_search=False,encoder_backward=False,coordinate_lift_from_prior_methods=False,
    evidence_level='FLOAT64_SUBPROBLEM_DIAGNOSTIC_NOT_COMPLETE_HEAD_CERTIFICATE')
RAW_FEATURE_CONTRACT=dict(input_shape=[2,256],view='original_received_observation',view_count=1,
    identity_feature_key='feat_joint',branch_keys=['z_id','t_emb','f_emb','pa_local'],branch_dim=160,fft_dim=96,
    fft='historical_spectral_logmag_sketch',fft_norm_floor=1e-8,cache_dtype='float32',
    normalization='raw_native_aux_no_additional_branch_normalization')


def require(condition,message):
    if not condition:raise ValueError(message)


def _constant(value):
    raise ValueError('Nonfinite JSON number: '+value)


def _object(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Duplicate JSON key: '+key);result[key]=value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=_constant,object_pairs_hook=_object)


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def strings(values,empty=False):
    return isinstance(values,list) and (empty or bool(values)) and all(type(v) is str and v for v in values) and len(set(values))==len(values)


def _integer(value,minimum=0):
    return type(value) is int and value>=minimum


def validate_config(config):
    require(isinstance(config,dict) and set(config)=={'algorithm','support_metric_resources'},'Wrong fixed benchmark config')
    require(config['algorithm']==FROZEN_ALGORITHM,'Wrong frozen method algorithm')
    limits=config['support_metric_resources']
    require(isinstance(limits,dict) and set(limits)==set(FROZEN_RESOURCES)
        and all(_integer(v,1) for v in limits.values()),'Explicit SupportMetric resource limits required')
    require(limits==FROZEN_RESOURCES,'Registered SupportMetric resource limits changed')


EPS=np.finfo(np.float64).eps
ALGORITHM=FROZEN_ALGORITHM
check=require

def loads(text):return json.loads(text,parse_constant=_constant,object_pairs_hook=_object)

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

def _merge(target,source,peaks=PEAK_COUNTERS):
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

class Archives:
    def __init__(self,root,run_id,row_id):
        self.root=Path(root);self.run_id=run_id;self.row_id=row_id;self.used=set();self.refs={};self.namespaces={}
        self.manifest=read(self.root/'state_manifest.json');m=self.manifest
        check(m['status']=='COMPLETE' and m['schema']=='d92_support_metric_joint_state_archive_v1' and m['method']==METHOD,'Incomplete numeric archive')
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
                if value==getattr(self,'row_basis_ref',None):self.reference(value,self.row_basis_coords,'ROW_SUPPORT_METRIC_BASIS','basis')
                else:self.reference(value,coords)
            else:
                for item in value.values():self.walk(item,coords)
        elif isinstance(value,list):
            for item in value:self.walk(item,coords)

    def close(self):check(self.used==set(self.refs),'Unreferenced numeric state; full archive closure required')

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
        fold=None,trial=None,parent_k=None,train_k=None,state='ROW_SUPPORT_METRIC_BASIS')
    check(owner['context']==coords,'Row basis namespace differs')
    if hasattr(archives,'row_basis_owner'):
        _equal(owner,archives.row_basis_owner,'Same row basis owner changed')
        return archives.row_basis_work
    ref=archives.reference(owner['row_basis_ref'],coords,'ROW_SUPPORT_METRIC_BASIS','basis')
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


def _indices(value,count):
    return isinstance(value,list) and bool(value) and all(_integer(i) and i<count for i in value) and len(set(value))==len(value)


def load_capsule_metadata(capsule,expected_capsule_id):
    """Never read IQ or query labels: only manifest, splits and received.ids."""
    root=Path(capsule);manifest=read(root/'manifest.json')
    require(manifest.get('protocol_schema')=='p2_min_v1' and manifest.get('phase2_data_status')=='VALIDATED_ONCE'
        and manifest.get('capsule_id')==expected_capsule_id,'Capsule identity/protocol mismatch')
    with np.load(root/'received.npz',allow_pickle=False) as arrays:
        ids=arrays['ids']
        require(ids.ndim==1 and ids.dtype.kind in 'SU','Opaque capsule IDs must be text')
        physical=ids.astype(str).tolist()
    require(strings(physical),'Repeated or empty opaque capsule IDs')
    splits={};cells=set();old=None
    for path in sorted((root/'splits').glob('*.json')):
        s=read(path);sid=s['split_id'];classes=s['registered_classes']
        require(type(sid) is str and sid and sid not in splits and strings(classes),'Invalid capsule split registry')
        require(path.stem==sid and s.get('protocol_schema')=='p2_min_v1' and s.get('phase2_data_status')=='VALIDATED_ONCE'
            and s.get('capsule_id')==expected_capsule_id
            and not {'query_labels','query_truth','query_roles','query_class_counts'}&set(s),'Forbidden or mismatched split protocol metadata')
        require(_integer(s['k'],1) and s['k'] in KS and _integer(s['support_seed'])
            and len(classes)-6 in NEWS,'Wrong K/new/support seed axes')
        declared_old=classes[:6]
        if old is None:old=declared_old
        require(len(declared_old)==6 and declared_old==old,'Six-old class registry changed')
        require(all(type(s[k]) is str and s[k] for k in ('receiver','scenario')),'Invalid receiver/scenario')
        cell=(s['receiver'],s['scenario'],s['support_seed'],s['k'],len(classes)-6)
        require(cell not in cells,'Duplicate capsule matrix cell');cells.add(cell)
        si,qi=s['support_indices'],s['query_indices']
        require(_indices(si,len(physical)) and _indices(qi,len(physical)) and not set(si)&set(qi),'Invalid or overlapping support/query IDs')
        labels=s['support_labels']
        require(isinstance(labels,list) and len(si)==len(labels)==s['k']*len(classes)
            and all(_integer(y) and y<len(classes) for y in labels)
            and all(labels.count(j)==s['k'] for j in range(len(classes))),'Physical support K mismatch')
        support=[physical[i] for i in si];queries=[physical[i] for i in qi]
        if 'support_ids' in s:require(s['support_ids']==support,'Capsule support index/ID mismatch')
        old_support=[pid for pid,y in zip(support,labels) if classes[y] in old]
        new_support=[pid for pid,y in zip(support,labels) if classes[y] not in old]
        splits[sid]=dict(**{k:s[k] for k in ('split_id','receiver','scenario','k','support_seed')},
            new_count=len(classes)-6,declared_registered_classes=classes,registered_classes=sorted(classes),
            old_classes=sorted(old),support_ids=sorted(support),old_support_ids=sorted(old_support),new_support_ids=sorted(new_support),
            query_ids=queries,query_count=len(queries),
            _support_label_mapping={pid:classes[y] for pid,y in zip(support,labels)})
    pairs={(rx,scene,seed) for rx,scene,seed,_,_ in cells}
    expected={(rx,scene,seed,k,n) for rx,scene,seed in pairs for k in KS for n in NEWS}
    require(bool(splits) and cells==expected and manifest.get('split_count')==len(splits),'Incomplete capsule K/new matrix')
    return manifest,splits


def _source_identity(startup,complete,expected):
    source=complete['source_identity'];ground=complete['ground_packet_identity']
    require(startup['source_identity']==source and startup['ground_packet_identity']==ground,'Source identity changed between startup and completion')
    for item in (source,ground):
        require(item['checkpoint_sha256']==expected['checkpoint_sha256'] and item['model_seed']==expected['model_seed']
            and item['source_only_verdict']=='MATCHED_SOURCE_ONLY_SCRATCH','Source-only checkpoint identity mismatch')
    require(source['source_role_comparison']=='EXACT_MATCH' and type(source['checkpoint_epoch']) is int
        and source['checkpoint_epoch']==200 and source['checkpoint_inheritance']==[]
        and source['target_access_before_freeze'] is False,'Source provenance boundary mismatch')
    require(ground['ordered_classes']==complete['ordered_ground_classes'] and ground['feature_contract']==dict(
        checkpoint_sha256=expected['checkpoint_sha256'],cache_key='z_id',source_tensor='feat_joint',representation='raw',dtype='float32',feature_dim=160),
        'Ground/native feature or class mapping changed')
    require(Path(ground['path']).resolve()==Path(complete['run_binding']['ground_packet']).resolve(),
        'Ground packet path differs from the explicit current row source')
    require(source['cache_schema']=='d92_branch_received_features_v1' and source['feature_contract']==RAW_FEATURE_CONTRACT,
        'Wrong current five-branch raw feature source contract')
    require(type(ground['scale']) in (int,float) and not isinstance(ground['scale'],bool)
        and math.isfinite(ground['scale']) and ground['scale']>0
        and type(ground['norm_eps']) in (int,float) and not isinstance(ground['norm_eps'],bool)
        and math.isfinite(ground['norm_eps']) and ground['norm_eps']>0
        and _integer(ground['packet_total_file_bytes'],1),'Ground packet declared scale/bytes invalid')
    geometry=complete['ground_geometry_identity'];binding=complete['ground_geometry_binding']
    require(startup['ground_geometry_identity']==geometry and startup['ground_geometry_binding']==binding,
        'Frozen prototype geometry identity changed')
    require(geometry['checkpoint_sha256']==expected['checkpoint_sha256'] and geometry['model_seed']==expected['model_seed']
        and Path(geometry['path']).resolve()==Path(complete['run_binding']['ground_summary']).resolve()
        and type(geometry['already_deployed']) is bool
        and geometry['already_deployed']==complete['ground_summary_already_deployed']
        and geometry['input_role']=='FROZEN_PHASE1_CENTER_ONLY_GEOMETRY'
        and all(geometry[k] is False for k in ('source_examples','prototype_teacher_targets','checkpoint_loaded')),
        'Wrong frozen six-prototype geometry source/role')
    require(binding['operation']=='load_existing_ground_center_fixed_proto_frame'
        and all(binding[k] is False for k in ('source_samples_read','source_per_record_features_read','checkpoint_loaded',
            'residual_domain_reconstruction','prototype_updated','prototype_teacher_targets','support_labels_read','query_read'))
        and binding['newly_generated_ground_statistics_bytes']==0,
        'Forbidden prototype teacher/statistics/data access')
    frame=binding['frame_audit']
    require(frame['class_order']==complete['old_classes'] and frame['prototype_class_count']==6
        and frame['prototype_dimension']==160 and frame['dictionary_coordinate_count']==5,'Frozen frame class/dimension mismatch')
    payload=binding['ground_payload_audit']
    require(payload['already_deployed']==geometry['already_deployed']
        and _integer(payload['total_file_bytes']) and _integer(payload['incremental_transfer_bytes'])
        and payload['incremental_transfer_bytes']==(0 if geometry['already_deployed'] else payload['total_file_bytes'])
        and all(payload[k] is False for k in ('source_samples_read','checkpoint_file_read','dense_bank_persisted','dequantized_domain_cache')),
        'Ground geometry payload scope differs')


def _state_ref(ref,manifest,expected,state):
    require(isinstance(ref,dict) and ref.get('path') in manifest,'Head ref is not in current archive manifest')
    saved=manifest[ref['path']]
    require(set(ref)==set(saved) and all(ref[k]==saved[k] for k in saved if k!='arrays')
        and set(ref['arrays'])==set(saved['arrays']),'Head reference identity/inventory mismatch')
    for name,meta in saved['arrays'].items():
        require(ref['arrays'][name]==meta or ref['arrays'][name]=={k:meta[k] for k in ('shape','dtype','nbytes')},
            'Head reference numeric metadata mismatch')
        require(meta.get('all_finite') is True and type(meta.get('nonfinite_count')) is int
            and meta['nonfinite_count']==0,'Failed numeric metadata in complete head ref')
        shape=meta['shape'];dtype=np.dtype(meta['dtype'])
        require(isinstance(shape,list) and all(_integer(v) for v in shape) and dtype.kind in 'fiu b'.replace(' ','')
            and _integer(meta['nbytes']) and meta['nbytes']==math.prod(shape)*dtype.itemsize,
            'Invalid numeric archive shape/dtype/bytes')
    require(saved.get('failed_numeric_state') is False,'Failed state ref cannot count as completed actual B/C')
    require(saved.get('key')=='final','Completed stage must bind its actual final state ref')
    path=Path(ref['path'])
    require(not path.is_absolute() and len(path.parts)==2 and path.parts[0]=='state_arrays' and path.suffix=='.npz',
        'Head reference escaped current archive')
    namespace=json.loads(ref['namespace'],parse_constant=_constant,object_pairs_hook=_object)
    require(all(namespace.get(k)==expected[k] for k in ('run_id','row_id','split_id'))
        and namespace.get('scope')=='query_benchmark_support_training' and namespace.get('fold') is None
        and namespace.get('trial') is None and namespace.get('state')==state,'Head reference escaped actual current split')


def _validate_archive(root,manifest,binding,splits):
    """Verify every actual numeric file; metadata alone is not saved state."""
    require(isinstance(manifest['files'],list),'Invalid numeric archive inventory')
    require(all(_integer(manifest[k]) for k in ('file_count','total_file_bytes','numeric_array_bytes')),
        'Invalid numeric archive totals')
    refs={v['path']:v for v in manifest['files']}
    require(len(refs)==manifest['file_count']==len(manifest['files']),'Duplicate numeric archive refs')
    directory=root/'state_arrays';resolved=directory.resolve()
    require(directory.is_dir() and resolved==root.resolve()/'state_arrays','Numeric archive directory escaped row output')
    require(set(refs)=={p.relative_to(root).as_posix() for p in directory.glob('*.npz')},
        'Missing/undeclared physical numeric archive')
    numeric_bytes=0;file_bytes=0;phases={}
    for relative,ref in refs.items():
        path=Path(relative)
        require(not path.is_absolute() and len(path.parts)==2 and path.parts[0]=='state_arrays'
            and path.suffix=='.npz' and '..' not in path.parts,'Numeric archive escaped row output')
        physical=root/path
        require(physical.is_file() and physical.resolve().parent==resolved,'Missing or escaped actual numeric state file')
        actual_bytes=physical.stat().st_size
        require(_integer(ref['file_bytes'],1) and ref['file_bytes']==actual_bytes,'Numeric archive actual file bytes differ')
        require(ref['failed_numeric_state'] is False and isinstance(ref['arrays'],dict) and ref['arrays'],
            'Invalid/failed complete numeric archive')
        ns=json.loads(ref['namespace'],parse_constant=_constant,object_pairs_hook=_object)
        basis_namespace=ns.get('state')=='ROW_SUPPORT_METRIC_BASIS'
        require(ns['run_id']==binding['run_id'] and ns['row_id']==binding['row_id']
            and ((ns['split_id'] is None and ns['scope']=='row_frozen_ground_geometry'
                  and ns.get('parent_k') is None and ns.get('train_k') is None and ref['key']=='basis') if basis_namespace else
                 (ns['split_id'] in splits and ns['scope']=='query_benchmark_support_training'))
            and ns['fold'] is None and ns['trial'] is None and type(ns['state']) is str and ns['state'],
            'Actual numeric file namespace escaped current row/split')
        current_bytes=0
        with np.load(physical,allow_pickle=False) as arrays:
            require(len(arrays.files)==len(set(arrays.files)) and set(arrays.files)==set(ref['arrays']),
                'Actual numeric archive array inventory differs')
            for name,meta in ref['arrays'].items():
                a=arrays[name]
                require(isinstance(meta['shape'],list) and all(_integer(v) for v in meta['shape']) and _integer(meta['nbytes'])
                    and a.dtype.kind in 'fbiu' and (a.dtype.kind!='f' or a.dtype in (np.dtype('float32'),np.dtype('float64')))
                    and list(a.shape)==meta['shape'] and str(a.dtype)==meta['dtype'] and a.nbytes==meta['nbytes'],
                    'Actual numeric archive shape/dtype/nbytes differ')
                require(meta['all_finite'] is True and type(meta['nonfinite_count']) is int and meta['nonfinite_count']==0
                    and bool(np.isfinite(a).all()),'Nonfinite actual numeric state cannot certify completion')
                current_bytes+=a.nbytes
        file_bytes+=actual_bytes;numeric_bytes+=current_bytes
        group=phases.setdefault(ns['state'],dict(file_count=0,file_bytes=0,numeric_array_bytes=0,seconds=[]))
        group['file_count']+=1;group['file_bytes']+=actual_bytes;group['numeric_array_bytes']+=current_bytes
        group['seconds'].append(_seconds(ref['archive_seconds'],'archive_seconds'))
    require(manifest['total_file_bytes']==file_bytes and manifest['numeric_array_bytes']==numeric_bytes,
        'Physical/numeric archive totals differ')
    _seconds_sum_readback(manifest['archive_seconds'],[v['archive_seconds'] for v in refs.values()],'archive total seconds')
    require(set(manifest['by_phase'])==set(phases),'Numeric archive phase inventory differs')
    for phase,values in phases.items():
        declared=manifest['by_phase'][phase]
        require(all(declared[k]==values[k] for k in ('file_count','file_bytes','numeric_array_bytes')),'Actual archive phase totals differ')
        _seconds_sum_readback(declared['archive_seconds'],values['seconds'],'archive phase seconds')
    return refs


def _stream(root,stage,marker,splits,ground):
    path=root/STREAMS[stage];item=marker['streams'][stage]
    require(item['path']==STREAMS[stage] and item['file_bytes']==path.stat().st_size,'Fixed stream file identity/bytes mismatch')
    records=[];grouped=defaultdict(list)
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()),'Empty prediction record')
            record=json.loads(line,parse_constant=_constant,object_pairs_hook=_object)
            require(set(record)=={'split_id','query_id','classes','scores','prediction'},'Prediction carries forbidden or missing fields')
            sid=record['split_id'];require(sid in splits,'Unexpected prediction split')
            classes=ground if stage=='A' else splits[sid]['old_classes'] if stage in ('B','R0_B') else splits[sid]['registered_classes']
            require(record['classes']==classes and record['query_id'] in splits[sid]['query_ids'],'Prediction class/opaque ID binding mismatch')
            scores=record['scores']
            require(isinstance(scores,list) and len(scores)==len(classes)
                and all(type(v) in (int,float) and not isinstance(v,bool) and math.isfinite(v) for v in scores),
                'Invalid complete finite class scores')
            require(record['prediction'] in classes,'Prediction is outside all registered columns')
            if stage!='C' or splits[sid]['new_count']==0:
                winner=classes[max(range(len(scores)),key=scores.__getitem__)]
                require(record['prediction']==winner,'Wrong per-sample full-column argmax')
            records.append(record);grouped[sid].append(record)
    require(item['record_count']==len(records) and set(grouped)==set(splits),'Missing prediction stream coverage')
    for sid,split in splits.items():
        require([r['query_id'] for r in grouped[sid]]==split['query_ids'],'Duplicate, reordered or missing opaque queries')
    return records,dict(grouped)


def _vector(value,count,name):
    require(isinstance(value,list) and len(value)==count and all(type(v) in (int,float)
        and math.isfinite(v) for v in value),'Invalid decision vector: '+name)
    return np.asarray(value,dtype=np.float64)


def _numeric_readback(actual,expected,operation_count,name):
    # Binary64 forward arithmetic readback bound, independent of accuracy.
    expected=np.asarray(expected,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    scale=np.maximum(1.,np.abs(expected))
    bound=np.finfo(np.float64).eps*operation_count*scale
    require(actual.shape==expected.shape and np.all(np.isfinite(actual))
        and bool(np.all(np.abs(actual-expected)<=bound)),'Structural numeric readback differs: '+name)


def _seconds(value,name):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid measured seconds: '+name)
    return value


def _seconds_sum_readback(total,values,name):
    """Compare one clock scope only, allowing binary64 sequential SUM error."""
    total=_seconds(total,name)
    expected=math.fsum(_seconds(v,name) for v in values)
    count=max(1,len(values));eps=np.finfo(np.float64).eps
    require(count*eps<1,'Seconds readback accumulation bound is undefined')
    # gamma_n bounds nonnegative sequential addition; fsum adds one rounding.
    bound=(count*eps/(1-count*eps)+eps)*abs(expected)
    require(abs(total-expected)<=bound,'Measured seconds SUM readback differs: '+name)


def _work_seconds(work,name):
    require(isinstance(work,dict) and 'score_seconds' in work,'Missing returned score timing: '+name)
    for key,value in work.items():
        if key.endswith('_seconds'):_seconds(value,name+'.'+key)


def verify_structured_decision(certificate,c_record,b_record,split):
    require(isinstance(certificate,dict) and certificate.get('schema')==DECISION_SCHEMA,'Missing structural C decision certificate')
    require(certificate.get('prediction')==c_record['prediction'],'C prediction/certificate mismatch')
    if split['new_count']==0:
        require(set(certificate)=={'schema','policy','new0_reuses_actual_B','prediction'}
            and certificate['policy']=='EXACT_ACTUAL_B_REUSE' and certificate['new0_reuses_actual_B'] is True,
            'new0 structural certificate mismatch')
        require(c_record==b_record,'new0 changed actual B scores or prediction')
        return
    fields={'schema','policy','new0_reuses_actual_B','old_classes','new_classes','old_raw_scores','new_raw_scores',
        'gate_logit','old_log_probabilities','new_log_probabilities','old_winner','new_winner','group_gap','prediction'}
    require(set(certificate)==fields and certificate['policy']==DECISION_POLICY
        and certificate['new0_reuses_actual_B'] is False,'Wrong all-group structural decision policy')
    old=split['old_classes'];new=[v for v in split['registered_classes'] if v not in old]
    require(certificate['old_classes']==old and certificate['new_classes']==new,'Structural decision omitted registered columns')
    rawold=_vector(certificate['old_raw_scores'],len(old),'old raw')
    rawnew=_vector(certificate['new_raw_scores'],len(new),'new raw')
    # Actual B remains the old conditional classifier, not an independently fit teacher.
    require(certificate['old_raw_scores']==b_record['scores'],'C certificate did not use actual same-query B')
    lo=_vector(certificate['old_log_probabilities'],len(old),'old log probabilities')
    ln=_vector(certificate['new_log_probabilities'],len(new),'new log probabilities')
    def logsoftmax(values):
        shift=values-values.max();return shift-np.log(np.exp(shift).sum())
    _numeric_readback(lo,logsoftmax(rawold),len(old)+8,'old logsoftmax')
    _numeric_readback(ln,logsoftmax(rawnew),len(new)+8,'new logsoftmax')
    g=certificate['gate_logit'];gap=certificate['group_gap']
    require(type(g) in (int,float) and math.isfinite(g) and type(gap) in (int,float) and math.isfinite(gap),
        'Invalid gate scalar')
    oi=int(np.argmax(rawold));ni=int(np.argmax(rawnew));ow,nw=old[oi],new[ni]
    require(certificate['old_winner']==ow and certificate['new_winner']==nw,'Wrong raw within-group winner')
    require(gap==float(g+lo[oi]-ln[ni]),'Wrong structural group gap')
    expected=ow if gap>0 else nw if gap<0 else min(ow,nw)
    require(c_record['prediction']==expected,'Wrong public structured C decision')
    values={name:float(-np.logaddexp(0.,-g)+lo[i]) for i,name in enumerate(old)}
    values.update({name:float(-np.logaddexp(0.,g)+ln[i]) for i,name in enumerate(new)})
    _numeric_readback(c_record['scores'],[values[name] for name in split['registered_classes']],
        len(old)+len(new)+12,'all-column composed scores')


def _decision_stream(root,marker,splits,records):
    path=root/'query_score_work.jsonl';meta=marker['query_score_work']
    require(meta==dict(path=path.name,record_count=len(records['C']),file_bytes=path.stat().st_size),
        'Complete structural decision stream metadata differs')
    decisions=[]
    with path.open(encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            require(index<len(records['C']),'Unexpected structural decision')
            row=json.loads(line,parse_constant=_constant,object_pairs_hook=_object)
            require(set(row)=={'split_id','query_id','B_work','C_work','C_reused_B_scores','C_structural_decision',
                'C_structural_prediction','C_public_predict_calls','C_public_predict_seconds','C_public_predict_internal_work',
                'C_public_predict_internal_work_unavailable_reason','R0_score_work','R0_work_unavailable_reason',
                'R0_C_reused_B_scores','scope'},'Wrong query work schema')
            c=records['C'][index];b=records['B'][index]
            require((row.get('split_id'),row.get('query_id'))==(c['split_id'],c['query_id']),
                'Structural decision order/physical identity mismatch')
            verify_structured_decision(row.get('C_structural_decision'),c,b,splits[c['split_id']])
            reuse=splits[c['split_id']]['new_count']==0
            require(row['C_structural_prediction']==c['prediction'] and row['C_reused_B_scores'] is reuse
                and row['R0_C_reused_B_scores'] is reuse and row['R0_score_work'] is None
                and row['R0_work_unavailable_reason']=='ORIGINAL_SCORE_API_HAS_NO_OPERATION_LEDGER'
                and type(row['C_public_predict_calls']) is int and row['C_public_predict_calls']==int(not reuse)
                and type(row['C_public_predict_seconds']) in (int,float) and math.isfinite(row['C_public_predict_seconds'])
                and row['C_public_predict_seconds']>=0 and row['C_public_predict_internal_work'] is None,
                'Structured public inference work binding differs')
            require(isinstance(row['B_work'],dict) and (row['C_work'] is None if reuse else isinstance(row['C_work'],dict)),
                'Actual score work/reuse scope differs')
            _work_seconds(row['B_work'],'B_work')
            if not reuse:_work_seconds(row['C_work'],'C_work')
            for name in ('B_work','C_work'):
                if row[name] is not None:
                    _validate_work(row[name]['actual_work'],name)
                    verify_work(row[name])
                    require(row[name]['single_record_all_registered_classes'] is True,'Score work did not execute singleton all-column inference')
            require(row['C_public_predict_internal_work_unavailable_reason']==('EXACT_B_REUSE_NO_EXTRA_CALL' if reuse
                else 'PUBLIC_PREDICT_HAS_NO_WORK_AUDIT'),'Unknown public inference work reason differs')
            require(row['scope']=='ACTUAL_SINGLETON_SCORE_AND_SEPARATE_PUBLIC_PREDICT; NO_LABEL_OR_ROLE',
                'Query work scope differs')
            decisions.append(row)
    require(len(decisions)==len(records['C']),'Missing structural query decisions')
    return decisions


def _validate_work(value,name):
    require(isinstance(value,dict) and set(value)==WORK_KEYS,'Missing/unknown actual work key: '+name)
    for key,item in value.items():
        require(type(item) in (int,float) and math.isfinite(item) and item>=0
            and (key.endswith('_seconds') or type(item) is int),'Invalid actual work: '+name+'.'+key)
    return value


def _merge_work(values):
    for value in values:_validate_work(value,'actual_work')
    return {key:(max((v[key] for v in values),default=0) if key in PEAK_COUNTERS
        else math.fsum(v[key] for v in values) if key.endswith('_seconds') else sum(v[key] for v in values))
        for key in sorted(WORK_KEYS)}


def _work_readback(actual,values,name):
    _validate_work(actual,name)
    for key in WORK_KEYS:
        if key in PEAK_COUNTERS:require(actual[key]==max((v[key] for v in values),default=0),'Actual peak MAX differs: '+name+'.'+key)
        elif key.endswith('_seconds'):_seconds_sum_readback(actual[key],[v[key] for v in values],name+'.'+key)
        else:require(actual[key]==sum(v[key] for v in values),'Actual work SUM differs: '+name+'.'+key)


def _jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line,parse_constant=_constant,object_pairs_hook=_object) for line in stream]


def _candidate_numeric(stage,prep,split,archives,resources,b_state=None):
    """Saved-artifact readback only: no head fitting or Fraction reconstruction."""
    a=stage['audit'];p=prep['audit'];is_b=stage['state']=='B_SUPPORT_METRIC'
    train=split['old_support_ids'] if is_b else split['support_ids']
    classes=split['old_classes'] if is_b else split['registered_classes']
    rank=archives.row_basis_owner['row_basis_rank'];coords={k:stage[k] for k in
        ('run_id','row_id','split_id','scope','fold','trial','parent_k','train_k')}
    require(a['schema']==FROZEN_ALGORITHM['schema'] and a['method']==METHOD and a['status']=='COMPLETED'
        and a['final_head_complete'] is True and a['config']==FROZEN_ALGORITHM and a['limits']==resources,
        'Incomplete or changed SupportMetric stage')
    require(a['classes']==classes and a['training_physical_ids']==train and p['training_physical_ids']==train,
        'Actual registered classes/physical training records differ')
    # Core context carries the logical B/C mode; archive namespaces separately
    # carry B_prepare/C_prepare or B_SUPPORT_METRIC/C_SUPPORT_METRIC_seq.
    # Keep every physical coordinate exact rather than conflating these fields.
    core_context=dict(coords,stage='B' if is_b else 'C_seq')
    require(a['context']==p['context']==core_context,'Actual core context differs')
    require(a['mode']==('B' if is_b else 'C_seq') and a['effective_parameter_rank']==p['effective_parameter_rank']==rank
        and a['nominal_parameter_count']==5 and a['parameter_buffer_bytes']==8*rank,'Coordinate ABI/rank differs')
    eligible=split['k']>1 and rank>0
    require(a['no_held']==(split['k']==1) and a['trainable_parameter_count']==(rank if eligible else 0),
        'K1/rank0 trainable parameter scope differs')
    require(a['evidence_level']==FROZEN_ALGORITHM['evidence_level'] and a['complete_head_jvp_error_bound'] is None,
        'Unsupported complete-head certificate')
    for value in (a,p):verify_work(value)
    require(p['basis_construction_charged_here'] is False and p['actual_work']['basis_calls']==0
        and p['actual_work']['basis_binding_calls']==1 and p['actual_work']['basis_binding_physical_gram_evaluation_count']==1,
        'Once-built basis was replaced or binding/Gram work missing')
    _equal(p['basis_audit'],archives.row_basis_owner['construction_audit'],'Preparation basis provenance differs')
    for value in (stage,prep):
        require(value['row_basis_ref']==archives.row_basis_ref and value['row_basis_rank']==rank
            and value['basis_certificate_ref']==archives.row_basis_owner['basis_certificate'],
            'Stage/preparation replaced row-owned basis')
    final=archives.reference(a['final_state_ref'],coords,stage['state'],'final');arrays=archives.load(final)
    verify_basis_arrays(arrays,rank)
    for key in ('original_dictionary_Q','U','U_lower','U_upper','U_error_bounds','exact_column_indices'):
        require(np.array_equal(arrays[key],archives.row_basis_arrays[key]),'Final head basis differs: '+key)
    anchor=np.asarray(a['anchor_theta']);require(anchor.shape==arrays['theta'].shape==(rank,),'Coordinate shape differs')
    _array_close(arrays['head_theta_padded'],np.pad(arrays['theta'],(0,5-rank)),'Padded head coordinates differ')
    require(np.array_equal(arrays['theta'],np.asarray(a['theta'])) and
        a['actual_updated_coordinate_count']==int(np.count_nonzero(arrays['theta']-anchor)), 'Actual update count differs')
    require(_integer(a['resident_numeric_state_bytes']) and a['deployment_numeric_state_bytes']==a['resident_numeric_state_bytes'],
        'Retained state byte scope differs')
    labels=arrays['labels'];require(labels.shape==(len(train),) and labels.dtype.kind in 'iu'
        and np.all((labels>=0)&(labels<len(classes))) and all(np.count_nonzero(labels==j)==split['k'] for j in range(len(classes))),
        'Saved support labels/count registry differs')
    label_map=dict(zip(train,[classes[int(y)] for y in labels]))
    # Capsule legal support labels are the input binding, never outer/query truth.
    require(label_map=={pid:split['_support_label_mapping'][pid] for pid in train},'Saved support labels changed')
    folds=p['folds'];fold_count=0 if split['k']==1 else min(3,split['k'])
    require(len(folds)==fold_count,'Inner physical fold count differs')
    members={pid:i%fold_count for c in classes for i,pid in enumerate(sorted(x for x in train if label_map[x]==c))} if fold_count else {}
    for index,fold in enumerate(folds):
        require(fold['fold']==index and fold['held_ids']==sorted(x for x in train if members[x]==index)
            and fold['train_ids']==sorted(x for x in train if members[x]!=index)
            and fold['old_inner_train_ids']==[x for x in fold['train_ids'] if label_map[x] in split['old_classes']],
            'Inner held/train/prior physical binding differs')
        if not is_b:
            ref=archives.reference(fold['prior_ref'],coords,'C_prepare');prior=archives.load(ref)
            require(prior['K'].shape==(len(fold['old_inner_train_ids']),)*2 and prior['alpha'].shape[1]==6,
                'Inner old-only prior shape differs')
            require(np.array_equal(prior['theta'],b_state['theta']) and np.array_equal(prior['U'],b_state['U']),
                'Inner old prior changed actual B coordinates/basis')
            verify_basis_arrays(prior,rank)
    if is_b:
        require(np.array_equal(anchor,np.zeros(rank)) and arrays['alpha'].shape==(len(train),6)
            and arrays['intercept'].shape==(6,), 'Independent B/free intercept differs')
    else:
        q=split['new_count'];n=len(train);old=np.asarray([i for i,pid in enumerate(train) if pid in set(split['old_support_ids'])])
        require(np.array_equal(anchor,b_state['theta']) and arrays['new_alpha'].shape==(q*split['k'],q)
            and arrays['new_intercept'].shape==(q,) and arrays['gate_alpha'].shape==(n,)
            and arrays['gate_b'].shape==() and arrays['gate_zeta'].shape==(), 'Actual B inheritance/C free heads differ')
        require(arrays['gate_slacks'].shape==(6*split['k'],q) and np.all(arrays['gate_slacks']>0)
            and arrays['lower_bounds'].shape==(6*split['k'],q) and np.array_equal(arrays['old_indices'],old)
            and np.array_equal(arrays['gate_old_indices'],old)
            and np.array_equal(arrays['gate_targets'],np.isin(np.arange(n),old).astype(float)), 'All-pair gate physical groups differ')
        for key,value in b_state.items():
            require('actual_B_'+key in arrays and np.array_equal(arrays['actual_B_'+key],value),'Frozen actual B numeric function differs: '+key)
    trials=a['trials'];require(len(trials)==a['trial_count']<=12 and a['optimizer_steps'] in (0,1),'Trial/update budget differs')
    if not eligible:
        require(not trials and a['initial_objective'] is None and a['final_objective'] is None
            and a['optimizer_steps']==0 and np.array_equal(arrays['theta'],anchor),'K1/rank0 fabricated objective/update')
    else:
        initial=archives.load(archives.reference(a['initial_state_ref'],coords,stage['state'],'initial'))
        verify_basis_arrays(initial,rank);require(np.array_equal(initial['U'],arrays['U']) and np.array_equal(initial['theta'],anchor),
            'Initial metric basis/anchor differs')
        verify_objective_arrays(initial,rank,train,classes,label_map,a['initial_objective']['RMSCE'])
        require(a['initial_objective']['loss_proximal']==a['final_objective']['loss_proximal']==0,'Metric damping became proximal loss')
        matches=[v for v in archives.refs.values() if v['key']=='metric_direction' and loads(v['namespace'])==dict(coords,state=stage['state'])]
        require(len(matches)==1,'Missing/duplicate actual metric direction')
        saved=archives.load(archives.reference(matches[0],coords,stage['state'],'metric_direction'))
        require(a['support_metric_step_audit']['max_secular_iterations']==resources['max_secular_iterations']
            and a['support_metric_step_audit']['secular_iteration_count']<=resources['max_secular_iterations'],'Secular resources differ')
        direction,M=verify_metric_direction(initial,saved,a['support_metric_step_audit'],rank)
        require(a['quadratic_multiplier']==float(saved['multiplier']),'Metric multiplier differs')
        accepted=0;last=anchor;loss=a['initial_objective']['RMSCE']
        for i,trial in enumerate(trials):
            require(trial['trial']==i and trial['step_size']==.5**i and type(trial['accepted']) is bool,
                'Actual eta=1 halving sequence differs')
            attempt=anchor+trial['step_size']*direction;delta=attempt-anchor
            val=archives.load(archives.reference(trial['state_ref'],coords,stage['state'],'trial_'+str(i)))
            _array_close(val['theta'],attempt,'Trial coordinate readback differs')
            verify_objective_arrays(val,rank,train,classes,label_map,trial['loss_after'])
            rhs=a['initial_objective']['RMSCE']+1e-4*float(initial['gradient']@delta)
            tol=128*EPS*max(1.,abs(loss),abs(trial['loss_after']),abs(rhs))
            _equal(trial['loss_before'],a['initial_objective']['RMSCE'],'Trial anchor objective differs')
            _array_close(np.asarray(trial['armijo_rhs']),np.asarray(rhs),'Actual delta Armijo differs')
            _equal(float(trial['comparison_tolerance']/tol),1.,'Trial float tolerance differs')
            require(trial['accepted']==(trial['loss_after']<=rhs+trial['comparison_tolerance'])
                and trial['real_inequality_holds']==(trial['loss_after']<=trial['armijo_rhs'])
                and trial['observed_objective_increase']==(trial['loss_after']>trial['loss_before']), 'Armijo/roundoff readback differs')
            _array_close(np.asarray(trial['metric_ball_value']),np.asarray(delta@M@delta),'Trial metric norm differs')
            if trial['accepted']:accepted+=1;last=attempt;loss=trial['loss_after'];require(i==len(trials)-1,'Trials after accepted update')
        require(accepted==a['optimizer_steps'] and (bool(trials) and (accepted==1 or len(trials)==12) if np.any(direction)
            else not trials and accepted==0),'Metric direction/trial completion differs')
        _array_close(arrays['theta'],last,'Final theta differs from last accepted state')
        _equal(a['final_objective']['RMSCE'],loss,'Final objective differs from actual accepted readback')
    archives.walk(stage,coords);archives.walk(prep,coords)
    return arrays


def _training_ledger(root,marker,splits,refs,binding):
    """Read returned audits, not solver reruns or planned upper bounds."""
    preparation=_jsonl(root/'preparations.jsonl');stages=_jsonl(root/'fit_stages.jsonl')
    archives=Archives(root,binding['run_id'],binding['row_id'])
    require(not (root/'row_basis.json').is_symlink(),'Row basis owner escaped output')
    owner=read(root/'row_basis.json');limits=marker['support_metric_resources']
    basis_work=verify_row_basis(owner,archives,limits)
    for key in ('row_basis_ref','row_basis_rank','basis_certificate'):
        require(marker[key]==owner[key],'Completed row basis ownership differs: '+key)
    require(marker['row_basis']=='row_basis.json' and marker['row_basis_construction_status']=='COMPLETE',
        'Once-only row basis completion differs')
    require(marker['row_basis_audit']==owner and marker['row_basis_construction_count']==1
        and marker['row_basis_actual_work']==basis_work,'Once-only row basis actual accounting differs')
    candidate=[v for v in stages if v['state'] in ('B_SUPPORT_METRIC','C_SUPPORT_METRIC_seq')]
    baseline=[v for v in stages if v['state'] in ('R0_B','R0_C')]
    wanted={(sid,state) for sid,s in splits.items() for state in ('B_SUPPORT_METRIC',)+(('C_SUPPORT_METRIC_seq',) if s['new_count'] else ())}
    prep_wanted={(sid,'B_prepare' if state=='B_SUPPORT_METRIC' else 'C_prepare') for sid,state in wanted}
    r0_wanted={(sid,'R0_B' if state=='B_SUPPORT_METRIC' else 'R0_C') for sid,state in wanted}
    for rows,expected in ((preparation,prep_wanted),(candidate,wanted),(baseline,r0_wanted)):
        require(len(rows)==len(expected) and {(v['split_id'],v['state']) for v in rows}==expected,'Missing/duplicate actual training stage')
        for v in rows:
            require(v['run_id']==binding['run_id'] and v['row_id']==binding['row_id']
                and v['scope']=='query_benchmark_support_training' and v['fold'] is None and v['trial'] is None,
                'Returned training audit escaped actual current split')
            split=splits[v['split_id']]
            require(v['parent_k']==v['train_k']==split['k'],'Returned audit K mismatch')
            if v['state'] not in ('B_prepare','C_prepare'):
                _state_ref(v['audit']['final_state_ref'],refs,dict(binding,split_id=v['split_id']),v['state'])
    require(len(stages)==len(candidate)+len(baseline),'Unexpected training stage')
    _work_readback(marker['preparation_actual_work'],[v['audit']['actual_work'] for v in preparation],'preparation')
    _work_readback(marker['stage_actual_work'],[v['audit']['actual_work'] for v in candidate],'stage')
    r=marker['resources']
    require(r['original_R0_training_audits']==baseline,'R0 recorded audit list changed')
    totals=dict(completed_head_fits=len(baseline),factorization_calls=0,triangular_solve_calls=0)
    for v in baseline:
        a=v['audit'];split=splits[v['split_id']];fit=a['final_fit']
        require(a['status']=='COMPLETED' and v['mode']=='INDEPENDENT_R0_REFIT'
            and a['training_physical_ids']==(split['old_support_ids'] if v['state']=='R0_B' else split['support_ids']),
            'R0 was not independently fit on legal current support')
        f,e=fit['factorization_calls'],fit['effective_degrees_of_freedom_extra_triangular_solves']
        require(_integer(f) and _integer(e),'Invalid original R0 solve audit')
        totals['factorization_calls']+=f;totals['triangular_solve_calls']+=2*f+e
    recorded={v['split_id']:v for v in marker['splits']}
    names={'B_SUPPORT_METRIC':'b_state_ref','C_SUPPORT_METRIC_seq':'c_state_ref','R0_B':'r0_b_state_ref','R0_C':'r0_c_state_ref'}
    for v in candidate+baseline:
        require(v['audit']['final_state_ref']==recorded[v['split_id']][names[v['state']]],'Prediction did not bind returned actual final training state')
    actual_b={v['split_id']:v['audit']['final_state_ref'] for v in candidate if v['state']=='B_SUPPORT_METRIC'}
    c_prepared={v['split_id']:v for v in preparation if v['state']=='C_prepare'}
    for v in candidate:
        if v['state']=='C_SUPPORT_METRIC_seq':
            ref=actual_b[v['split_id']]
            require(v['audit'].get('final_prior_ref')==ref
                and c_prepared[v['split_id']]['audit'].get('full_prior_ref')==ref,
                'Actual C stage/preparation did not inherit returned same-split B final state')
    for v in candidate:
        split=splits[v['split_id']];a=v['audit']
        require(v['training_physical_ids']==(split['old_support_ids'] if v['state']=='B_SUPPORT_METRIC' else split['support_ids']),
            'Candidate fitted a different physical support set')
        require(_integer(a['optimizer_steps']) and a['optimizer_steps']<=1
            and _integer(a['trial_count']) and a['optimizer_steps']<=a['trial_count']<=12,'Actual GGN/Armijo budget changed')
    prep_by={(v['split_id'],v['state']):v for v in preparation};actual_states={}
    for v in candidate:
        sid=v['split_id'];is_b=v['state']=='B_SUPPORT_METRIC'
        saved=_candidate_numeric(v,prep_by[(sid,'B_prepare' if is_b else 'C_prepare')],splits[sid],archives,limits,
            None if is_b else actual_states[sid])
        if is_b:actual_states[sid]=saved
    for v in baseline:archives.walk(v,{k:v[k] for k in ('run_id','row_id','split_id','scope','fold','trial','parent_k','train_k')})
    archives.close()
    ggn=sum(op['operation']=='support_metric_step' for v in candidate for op in v['audit']['operation_audits'])
    require(marker['ggn_step_count']==ggn and marker['ggn_parameter_direction_count']==owner['row_basis_rank']*ggn
        and marker['peak_effective_adapter_rank']==owner['row_basis_rank'],'Actual r-direction metric accounting differs')
    require(marker['peak_resident_numeric_state_bytes']==max(v['audit']['resident_numeric_state_bytes'] for v in candidate),
        'Actual resident numeric peak MAX differs')
    require(r['original_R0_completed_training_counters']==totals,'R0 actual returned work SUM differs')
    require(marker['candidate_preparation_count']==len(preparation) and marker['candidate_stage_count']==len(candidate)
        and marker['final_candidate_head_fit_count']==len(candidate),'Actual preparation/head count differs')
    for key in ('optimizer_steps','trial_count'):
        require(marker[key]==sum(v['audit'][key] for v in candidate),'Actual optimizer/trial SUM differs')
    require(marker['accepted_trial_count']==marker['optimizer_steps']
        and marker['rejected_trial_count']==marker['trial_count']-marker['optimizer_steps'],'Trial acceptance accounting differs')
    parameter_evidence=[]
    for v in candidate:
        a=v['audit'];parameter_evidence.append(dict(split_id=v['split_id'],state=v['state'],
            **{key:a[key] for key in ('nominal_parameter_count','effective_parameter_rank','trainable_parameter_count',
                'actual_updated_coordinate_count','parameter_buffer_bytes','resident_numeric_state_bytes',
                'deployment_numeric_state_bytes','optimizer_steps','trial_count')},
            parameter_scope='ADAPTER_COORDINATES_ONLY; ANALYTIC_RIDGE_AND_FREE_GATE_HEADS_SEPARATE',
            complete_head_interval_certificate=False))
    return [basis_work]+[v['audit']['actual_work'] for v in preparation+candidate],parameter_evidence


def _fixed_resources(marker,decisions,training_work):
    resources=marker['resources']
    expected_aggregation={k:'MAX' if k in PEAK_COUNTERS else 'SUM' for k in WORK_KEYS}
    require(marker['actual_work_aggregation']==expected_aggregation and marker['work_aggregation']=='SUM/MAX'
        and resources['work_aggregation']=='SUM/MAX','Resource SUM/MAX scope differs')
    score_work=[row[name]['actual_work'] for row in decisions for name in ('B_work','C_work') if row[name] is not None]
    _work_readback(marker['score_actual_work'],score_work,'score')
    # Producer total is a sequential SUM of actual audit records, not a sum
    # of three pre-rounded phase subtotals. Use its full accumulation length.
    _work_readback(marker['actual_work'],training_work+score_work,'total')
    require(resources['actual_core_work']==marker['actual_work']
        and resources['counter_scope']=='COMPLETED_ROW_BASIS_PREP_STAGE_SCORE','Returned core work scope differs')
    calls=sum(row['C_public_predict_calls'] for row in decisions);n=len(decisions)
    require(isinstance(resources.get('query_score_calls'),dict)
        and all(_integer(v) for v in resources['query_score_calls'].values())
        and resources['query_score_calls']==dict(A=n,B=n,C=calls,R0_B=n,R0_C=calls)
        and _integer(resources.get('C_public_predict_calls'))
        and resources['C_public_predict_calls']==calls,'Actual singleton query inference count differs')
    measured=resources.get('query_score_seconds')
    require(isinstance(measured,dict) and set(measured)==set(STREAMS),'Missing query score seconds')
    for stage,value in measured.items():_seconds(value,'query_score_seconds.'+stage)
    _seconds_sum_readback(resources.get('C_public_predict_seconds'),
        [row['C_public_predict_seconds'] for row in decisions],'C_public_predict_seconds')
    require(resources['C_public_predict_internal_work'] is None
        and resources['C_public_predict_internal_work_unavailable_reason']=='PUBLIC_PREDICT_HAS_NO_WORK_AUDIT',
        'Unmeasured public predict internal work must stay N/A')
    aggregate_training_resources([dict(resources=resources)])


def load_fixed_predictions(*,predictions,capsule,config,expected_binding):
    """Validate one complete row without opening any truth path."""
    validate_config(config);root=Path(predictions)
    require(not any((root/name).exists() for name in ('prediction_failed.json','technical_failure.json')),'Failed predictions cannot be scored')
    startup=read(root/'startup.json');marker=read(root/'predictions_complete.json')
    require(marker['schema']==startup['schema']==PREDICTION_SCHEMA and marker['method']==startup['method']==METHOD
        and marker['status']=='COMPLETE','Incomplete fixed query predictions')
    require(re.fullmatch('[0-9a-f]{40}',expected_binding['release_commit']) is not None,'Actual runtime release commit required')
    for item in (startup,marker):
        require(all(item.get(k)==expected_binding[k] for k in IDENTITY),'Prediction current row/source identity mismatch')
        require(item.get('query_fit_access') is False and item.get('source_fit_access') is False and item.get('truth_read') is False,
            'Forbidden fitting/truth access')
        require(all(item.get(k) is False for k in ('checkpoint_loaded','encoder_called','cross_split_adapted_state_reuse')),
            'Forbidden checkpoint/encoder/cross-split adapted state use')
        require(item.get('C_decision_policy')==DECISION_POLICY and item.get('C_decision_certificate_schema')==DECISION_SCHEMA
            and item.get('fit_scope')=='CURRENT_SPLIT_LEGAL_SUPPORT_ONLY'
            and item.get('query_inference_scope')=='ONE_RECEIVED_RECORD_ALL_REGISTERED_COLUMNS'
            and item.get('A_tie_policy')=='first_original_native_head_column'
            and item.get('B_C_tie_policy')=='physical_class_id_ascending','Wrong singleton all-class prediction policy')
        require(item.get('algorithm')==config['algorithm'] and item.get('support_metric_resources')==config['support_metric_resources'],'Prediction config/resource mismatch')
        require(all(item.get(k) is False for k in ('query_role_used','query_count_used_for_decision','view_aggregation'))
            and item.get('technical_query_chunk_size')==1 and item.get('nominal_adapter_parameter_count')==5
            and item.get('prototype_geometry_role')=='FROZEN_REFERENCE_ONLY_NOT_TEACHER','Wrong fixed geometry/query role boundary')
    require(startup.get('config')==config,'Resolved predictor config mismatch')
    basis_context=dict(run_id=expected_binding['run_id'],row_id=expected_binding['row_id'],split_id=None,
        scope='row_frozen_ground_geometry',fold=None,trial=None,parent_k=None,train_k=None,state='ROW_SUPPORT_METRIC_BASIS')
    require(startup['row_basis_construction_status']=='PENDING_EXPLICIT_FROZEN_Q_FACTORY'
        and startup['row_basis_namespace']==marker['row_basis_namespace']==basis_context,
        'Row basis startup ownership changed')
    manifest,splits=load_capsule_metadata(capsule,expected_binding['capsule_id'])
    ground=marker['ordered_ground_classes']
    require(strings(ground) and len(ground)==6 and marker['old_classes']==sorted(ground)
        and all(set(ground)==set(s['old_classes']) for s in splits.values()),'Ground six native columns mismatch')
    _source_identity(startup,marker,expected_binding)
    archive=read(root/'state_manifest.json')
    require(archive['schema']=='d92_support_metric_joint_state_archive_v1' and archive['method']==METHOD and archive['status']=='COMPLETE',
        'Incomplete current numeric archive manifest')
    refs=_validate_archive(root,archive,expected_binding,splits)
    require(marker['resources']['state_archive_file_bytes']==archive['total_file_bytes']
        and marker['resources']['state_archive_numeric_bytes']==archive['numeric_array_bytes'],
        'Prediction resource/archive actual totals differ')
    actual={s['split_id']:s for s in marker['splits']}
    require(len(actual)==len(marker['splits'])==len(splits) and set(actual)==set(splits),'Incomplete prediction split metadata')
    require(marker['split_count']==marker['completed_split_count']==len(splits)
        and marker['query_record_count']==sum(s['query_count'] for s in splits.values()),'Wrong completed split/query counts')
    require(marker.get('actual_stage_count')==sum(1+int(s['new_count']>0) for s in splits.values())
        and marker.get('actual_baseline_stage_count')==marker['actual_stage_count']
        and marker.get('state_manifest')=='state_manifest.json','Actual B/C stage completion count differs')
    for sid,split in splits.items():
        item=actual[sid]
        require(all(item.get(k)==v for k,v in split.items() if not k.startswith('_')),'Prediction support/query/class metadata mismatch')
        require(item['B_classes']==split['old_classes'] and item['C_classes']==split['registered_classes']
            and item['R0_B_classes']==split['old_classes'] and item['R0_C_classes']==split['registered_classes']
            and item['support_only_fit'] is True and item['query_fit_access'] is False and item['source_fit_access'] is False,
            'Actual support-only fitted columns mismatch')
        ctx=dict(expected_binding,split_id=sid)
        require(item['row_basis_ref']==marker['row_basis_ref'] and item['row_basis_rank']==marker['row_basis_rank']
            and item['basis_certificate_ref']==marker['basis_certificate'],'Split changed row basis owner')
        _state_ref(item['b_state_ref'],refs,ctx,'B_SUPPORT_METRIC')
        _state_ref(item['r0_b_state_ref'],refs,ctx,'R0_B')
        require(item['c_inherited_from_b_state_ref']==item['b_state_ref'],'C did not inherit actual same-split B')
        reuse=split['new_count']==0
        require(item['c_reuses_b'] is reuse and item['r0_c_reuses_b'] is reuse,'new0 reuse boundary mismatch')
        if reuse:
            require(item['c_state_ref']==item['b_state_ref'] and item['r0_c_state_ref']==item['r0_b_state_ref'],
                'new0 must reuse the exact candidate and R0 B state refs')
        else:
            _state_ref(item['c_state_ref'],refs,ctx,'C_SUPPORT_METRIC_seq')
            _state_ref(item['r0_c_state_ref'],refs,ctx,'R0_C')
    require(set(marker['streams'])==set(STREAMS),'Missing or unexpected prediction streams')
    streams={};all_records={}
    for stage in STREAMS:
        records,streams[stage]=_stream(root,stage,marker,splits,ground);all_records[stage]=records
    order=[(r['split_id'],r['query_id']) for r in all_records['A']]
    require(all([(r['split_id'],r['query_id']) for r in all_records[s]]==order for s in STREAMS),
        'Five stream global opaque query order mismatch')
    for sid,split in splits.items():
        if split['new_count']==0:require(streams['R0_B'][sid]==streams['R0_C'][sid],'new0 changed actual R0 B scores/predictions')
    alias=[]
    with (root/'predictions.jsonl').open(encoding='utf-8') as stream:
        for line in stream:alias.append(json.loads(line,parse_constant=_constant,object_pairs_hook=_object))
    require(alias==all_records['C'],'Public C alias differs from frozen C stream')
    require(marker['compatibility_predictions']==dict(path='predictions.jsonl',alias_of='C',file_bytes=(root/'predictions.jsonl').stat().st_size),
        'Public C alias metadata mismatch')
    decisions=_decision_stream(root,marker,splits,all_records)
    training_work,parameter_evidence=_training_ledger(root,marker,splits,refs,expected_binding)
    _fixed_resources(marker,decisions,training_work)
    return dict(binding=dict(expected_binding),startup=startup,complete=marker,splits=splits,streams=streams,
        predictions_root=str(root),capsule_manifest=manifest,ground_classes=ground,decisions=decisions,
        parameter_evidence=parameter_evidence)


def _statistics(parents):
    dimensions=dict(overall=(),by_k_new_count=('k','new_count'),by_cohort=('cohort','k','new_count'),
        by_receiver_scene=('receiver','scenario','k','new_count'),
        by_model_row=('model_seed','row_id','k','new_count'),
        by_support_seed=('support_seed','k','new_count'),
        by_model_receiver_scene_support_seed=('model_seed','row_id','receiver','scenario','support_seed','k','new_count'))
    result={}
    for name,dims in dimensions.items():
        groups=defaultdict(list)
        for parent in parents:
            populations=['all']+(['new_present'] if parent['new_count'] else ['new0'])
            for population in populations:
                for metric in METRICS:groups[(population,)+tuple(parent[d] for d in dims)+(metric,)].append(parent['metrics'][metric])
        rows=[]
        for key,values in sorted(groups.items(),key=repr):
            measured=[v for v in values if v is not None]
            rows.append(dict(zip(('population',)+dims+('metric',),key),parent_count=len(values),measured_parent_count=len(measured),
                null_parent_count=len(values)-len(measured),mean=None if not measured else math.fsum(measured)/len(measured),
                minimum=None if not measured else min(measured),maximum=None if not measured else max(measured)))
        result[name]=rows
    # First average physical parents within a model seed; only then summarize
    # the two registered model-seed means. SD is descriptive sample SD, not CI.
    seed_tables={}
    for name,dims in dict(overall=(),by_k_new_count=('k','new_count'),
            by_receiver_scene=('receiver','scenario','k','new_count'),
            by_support_seed=('support_seed','k','new_count')).items():
        buckets=defaultdict(lambda:defaultdict(list))
        for parent in parents:
            for population in ['all','new_present' if parent['new_count'] else 'new0']:
                for metric in METRICS:
                    key=(population,)+tuple(parent[d] for d in dims)+(metric,)
                    buckets[key][parent['model_seed']].append(parent['metrics'][metric])
        records=[]
        for key,seeds in sorted(buckets.items(),key=repr):
            require(set(seeds)==set(MODEL_SEEDS),'Missing paired model seed statistics')
            means={str(seed):(None if all(v is None for v in values)
                else math.fsum(v for v in values if v is not None)/sum(v is not None for v in values))
                for seed,values in sorted(seeds.items())}
            measured=[v for v in means.values() if v is not None]
            mean=None if not measured else math.fsum(measured)/len(measured)
            sd=None if len(measured)<2 else math.sqrt(math.fsum((v-mean)**2 for v in measured)/(len(measured)-1))
            records.append(dict(zip(('population',)+dims+('metric',),key),model_seed_means=means,
                model_seed_count=len(measured),mean=mean,sample_standard_deviation=sd,
                aggregation='PARENT_MEAN_WITHIN_MODEL_SEED_THEN_EQUAL_MODEL_SEED_MEAN',
                uncertainty_scope='DESCRIPTIVE_TWO_MODEL_SEED_SD_NOT_CONFIDENCE_INTERVAL'))
        seed_tables[name]=records
    result['model_seed_mean_sd']=seed_tables
    return result


def _truth_parent(fixed,sid,truth):
    split=fixed['splits'][sid];old=set(split['old_classes']);classes=split['registered_classes']
    rows={stage:fixed['streams'][stage][sid] for stage in STREAMS};by_stage={s:{r['query_id']:r for r in v} for s,v in rows.items()}
    old_ids=[];new_ids=[];targets={}
    for pid in split['query_ids']:
        require(pid in truth,'Fixed opaque query is missing from truth')
        target=truth[pid];name=target['transmitter']
        require(target['pool_role']=='query' and target['receiver']==split['receiver'] and target['scene']==split['scenario']
            and name in classes and type(target['old']) is bool and target['old']==(name in old),'Independent truth binding mismatch')
        targets[pid]=name;(old_ids if name in old else new_ids).append(pid)
    require(all(any(targets[pid]==name for pid in split['query_ids']) for name in classes),'Missing registered-class query coverage')
    def accuracy(stage,pids):return None if not pids else sum(by_stage[stage][pid]['prediction']==targets[pid] for pid in pids)/len(pids)
    a,b,co,cn=accuracy('A',old_ids),accuracy('B',old_ids),accuracy('C',old_ids),accuracy('C',new_ids)
    def macro_f1(stage,pids,names):
        if not pids or not names:return None
        values=[]
        for name in names:
            tp=sum(targets[pid]==name and by_stage[stage][pid]['prediction']==name for pid in pids)
            fp=sum(targets[pid]!=name and by_stage[stage][pid]['prediction']==name for pid in pids)
            fn=sum(targets[pid]==name and by_stage[stage][pid]['prediction']!=name for pid in pids)
            denominator=2*tp+fp+fn
            values.append(0. if denominator==0 else 2*tp/denominator)
        return math.fsum(values)/len(values)
    metrics=dict(A_old_accuracy=a,B_old_accuracy=b,C_old_accuracy=co,C_new_accuracy=cn,
        C_h=None if cn is None else (0. if co+cn==0 else 2*co*cn/(co+cn)),
        adaptation_gain_B_minus_A=b-a,total_old_accuracy_drop=b-co,C_abs_new_old_gap=None if cn is None else abs(co-cn),
        A_old_macro_f1=macro_f1('A',old_ids,split['old_classes']),B_old_macro_f1=macro_f1('B',old_ids,split['old_classes']),
        C_old_macro_f1=macro_f1('C',old_ids,split['old_classes']),
        C_new_macro_f1=macro_f1('C',new_ids,[c for c in classes if c not in old]),
        C_macro_f1=macro_f1('C',split['query_ids'],classes))
    rb,ro,rn=accuracy('R0_B',old_ids),accuracy('R0_C',old_ids),accuracy('R0_C',new_ids)
    rh=None if rn is None else (0. if ro+rn==0 else 2*ro*rn/(ro+rn))
    metrics.update(R0_B_old_accuracy=rb,R0_C_old_accuracy=ro,R0_C_new_accuracy=rn,R0_C_h=rh,
        C_old_minus_R0=co-ro,C_new_minus_R0=None if cn is None else cn-rn,
        C_h_minus_R0=None if rh is None else metrics['C_h']-rh)
    if split['new_count']==0:
        require(not new_ids and metrics['C_new_accuracy'] is metrics['C_h'] is metrics['C_abs_new_old_gap'] is None,'new0 new/H/gap must be N/A')
        require(all(by_stage['B'][pid]['prediction']==by_stage['C'][pid]['prediction']
            and by_stage['B'][pid]['scores']==by_stage['C'][pid]['scores'] for pid in split['query_ids']),'new0 changed the actual B function')
    counts={name:sum(v==name for v in targets.values()) for name in classes}
    class_accuracy={stage:{name:sum(by_stage[stage][pid]['prediction']==name for pid in split['query_ids'] if targets[pid]==name)/counts[name]
        for name in (classes if stage in ('C','R0_C') else split['old_classes'])} for stage in STREAMS}
    metadata=next(v for v in fixed['complete']['splits'] if v['split_id']==sid)
    return dict(**fixed['binding'],**{k:split[k] for k in SPLIT_IDENTITY},old_class_count=6,registered_class_count=len(classes),
        ordered_ground_classes=fixed['ground_classes'],old_classes=split['old_classes'],registered_classes=classes,
        old_query_ids=old_ids,new_query_ids=new_ids,old_query_count=len(old_ids),new_query_count=len(new_ids),
        query_count=len(split['query_ids']),query_class_counts=counts,class_accuracy=class_accuracy,metrics=metrics,
        macro_f1_scope='OLD_AND_NEW_METRICS_USE_THEIR_PHYSICAL_QUERY_SUBSET; C_MACRO_F1_USES_ALL_REGISTERED_QUERIES',
        b_state_ref=metadata['b_state_ref'],c_state_ref=metadata['c_state_ref'],
        r0_b_state_ref=metadata['r0_b_state_ref'],r0_c_state_ref=metadata['r0_c_state_ref'])


def _mapped(path,spec,run_root):
    source=Path(path)
    if run_root is None:return source
    original=Path(spec['execution']['remote_run_root'])
    require(source.is_relative_to(original),'Declared output escaped current run root')
    return Path(run_root)/source.relative_to(original)


def aggregate_training_resources(rows):
    """Only returned work; public predict and R0 score internal work remain unknown."""
    resources=[v['resources'] for v in rows]
    actual=_merge_work([v['actual_core_work'] for v in resources])
    baseline={key:sum(v['original_R0_completed_training_counters'][key] for v in resources)
        for key in ('completed_head_fits','factorization_calls','triangular_solve_calls')}
    calls={s:sum(v['query_score_calls'][s] for v in resources) for s in STREAMS}
    seconds={s:math.fsum(v['query_score_seconds'][s] for v in resources) for s in STREAMS}
    return dict(actual_core_work=actual,actual_work_aggregation={k:'MAX' if k in PEAK_COUNTERS else 'SUM' for k in sorted(WORK_KEYS)},
        original_R0_completed_training_counters=baseline,query_score_calls=calls,query_score_seconds=seconds,
        C_public_predict_calls=sum(v['C_public_predict_calls'] for v in resources),
        C_public_predict_seconds=math.fsum(v['C_public_predict_seconds'] for v in resources),
        C_public_predict_internal_work=None,R0_query_internal_work=None,
        unavailable_work_reason='PUBLIC_PREDICT_AND_ORIGINAL_R0_SCORE_APIS_RETURN_NO_OPERATION_LEDGER',
        peak_counter_keys=sorted(PEAK_COUNTERS),missing_counter_policy='UNKNOWN_NOT_ZERO',
        measurements_by_row=rows,training_measurements_scope='REPORTED_PRODUCER_SCOPES; NO_SCORER_MEASUREMENT',
        deployment_numeric_state_bytes=None,incremental_network_transfer_bytes=None,energy=None,
        scopes='CORE_ROW_BASIS_PREP_STAGE_SCORE_SUM_MAX_SEPARATE_FROM_R0_FIT_AND_CALLER_WALL; NESTED_TIMES_NOT_ADDED',
        numeric_evidence_level=FROZEN_ALGORITHM['evidence_level'],complete_head_interval_certificate=False,
        row_basis_construction_scope='ONCE_PER_ROW; PREPARATION_BINDING_AND_ACTUAL_GRAM_SEPARATELY_CHARGED')


def validate_declared_matrix(spec):
    rows=spec['rows'];cohorts=spec['benchmark']['cohorts']
    require(set(cohorts)==set(COHORT_COUNTS),'Both frozen receiver cohorts required')
    require(isinstance(rows,list) and len(rows)==4 and strings([r['row_id'] for r in rows]),
        'Exactly four distinct model/cohort rows required; subset scoring forbidden')
    require({(r['cohort'],r['expected_model_seed']) for r in rows}
        ==set(itertools.product(COHORT_COUNTS,MODEL_SEEDS)),'Fixed model/cohort cross product required')
    models={};outputs=set()
    for row in rows:
        require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',row['row_id']) is not None
            and Path(row['output_root'])==Path(spec['execution']['remote_run_root'])/row['row_id'],
            'Row output escaped the declared current run')
        require(re.fullmatch('[0-9a-f]{64}',row['expected_checkpoint_sha256']) is not None,'Invalid checkpoint SHA')
        require(type(row['ground_summary_already_deployed']) is bool,'Explicit frozen geometry deployment role required')
        identity=(row['expected_checkpoint_sha256'],row['ground_packet'],row['ground_summary'],row['ground_summary_already_deployed'])
        require(models.setdefault(row['expected_model_seed'],identity)==identity,'Same model seed changed checkpoint/packet across cohorts')
        require(row['output_root'] not in outputs,'Duplicate row output path');outputs.add(row['output_root'])
    require(len({value[0] for value in models.values()})==2,'Distinct model seeds require distinct checkpoint identities')
    for name,co in cohorts.items():
        expected=dict(receivers=RECEIVERS[name],scenarios=SCENARIOS,ks=list(KS),new_counts=list(NEWS),support_seeds=SUPPORT_SEEDS)
        require(co.get('matrix')==expected and type(co.get('expected_split_count')) is int
            and co['expected_split_count']==COHORT_COUNTS[name],'Declared full receiver/scene/support-seed/K/new matrix differs')
    return rows,cohorts


def _physical_matrix(value,co):
    matrix=co['matrix']
    expected=set(itertools.product(matrix['receivers'],matrix['scenarios'],matrix['support_seeds'],matrix['ks'],matrix['new_counts']))
    actual={(s['receiver'],s['scenario'],s['support_seed'],s['k'],s['new_count']) for s in value['splits'].values()}
    require(actual==expected and len(value['splits'])==co['expected_split_count'],'Incomplete actual physical full query matrix')


def _validated_row_fixed(directory,spec,row,commit,preflight):
    validate_config(spec['benchmark']['config'])
    co=spec['benchmark']['cohorts'][row['cohort']]
    binding=dict(run_id=spec['run_id'],row_id=row['row_id'],capsule_id=co['expected_capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],release_commit=commit)
    fixed=load_fixed_predictions(predictions=directory,capsule=co['capsule'],config=spec['benchmark']['config'],expected_binding=binding)
    _physical_matrix(fixed,co)
    require(isinstance(preflight,dict) and preflight.get('status')=='SUPPORT_METRIC_QUERY_PREFLIGHT_COMPLETE',
        'Actual current predictor preflight required')
    required=set(IDENTITY)|{'schema','method','algorithm','support_metric_resources','run_binding','source_identity',
        'ground_packet_identity','ground_geometry_identity','ground_geometry_binding','ground_summary_already_deployed','splits'}
    require(required<=set(preflight),'Incomplete preflight public identity')
    def immutable(value):
        # Preflight and predict each read the same input. Their elapsed clocks
        # differ; only timing fields are removed from this identity comparison.
        if isinstance(value,dict):return {k:immutable(v) for k,v in value.items() if k!='seconds' and not k.endswith('_seconds')}
        if isinstance(value,list):return [immutable(v) for v in value]
        return value
    for key,value in preflight.items():
        if key!='status':require(immutable(fixed['startup'].get(key))==immutable(value),'Preflight/startup public identity changed: '+key)
    source_paths=dict(row_root=row['row_root'],branch_features=row['branch_features'],ground_packet=row['ground_packet'],
        ground_summary=row['ground_summary'],capsule=co['capsule'])
    rb=fixed['complete']['run_binding']
    require(all(Path(rb[k]).resolve()==Path(v).resolve() for k,v in source_paths.items())
        and Path(rb['prediction_output_root']).resolve()==Path(row['output_root']+'/predictions').resolve()
        and fixed['complete']['ground_summary_already_deployed']==row['ground_summary_already_deployed'],
        'Predictor explicit source/deployment/output binding mismatch')
    return fixed


def validate_row_output(directory,spec,row,commit,preflight):
    """Supervisor completion check. No truth path is read or interpreted."""
    require(spec['schema']=='d92_support_metric_joint_query_benchmark_v1','Wrong current benchmark spec')
    require(sum(r==row for r in spec['rows'])==1,'Completion row must be an exact declared row')
    return _validated_row_fixed(directory,spec,row,commit,preflight)['complete']


def score_benchmark(*,spec,output,run_root=None):
    """Finish every prediction validation and independent reread before truth."""
    output=Path(output)
    require(not output.exists(),'Exclusive score output required')
    require(spec['schema']=='d92_support_metric_joint_query_benchmark_v1','Wrong current benchmark spec')
    validate_config(spec['benchmark']['config']);rows,cohorts=validate_declared_matrix(spec)
    root=Path(spec['execution']['remote_run_root']) if run_root is None else Path(run_root)
    startup=read(root/'startup.json');complete=read(root/'complete.json')
    require(startup['schema']==complete['schema']==spec['schema'] and startup['status']=='SUPPORT_METRIC_QUERY_BENCHMARK_STARTED'
        and startup['resolved_spec']==complete['resolved_spec']==spec
        and complete['run_id']==startup['run_id']==spec['run_id']
        and complete['group_id']==startup['group_id']==spec['group_id'],'Current supervisor spec/run mismatch')
    require(complete['status']=='SUPPORT_METRIC_QUERY_BENCHMARK_PREDICTIONS_COMPLETE'
        and complete['declared_episode_count']==complete['completed_episode_count']==2400
        and complete['row_count']==complete['completed_row_count']==len(rows)
        and complete['all_predictions_fixed'] is True and complete['truth_read'] is False
        and complete['scorer_invoked'] is False and complete['automatic_retry'] is False
        and set(complete['rows'])==set(startup['rows'])=={r['row_id'] for r in rows}
        and all(v['status']=='COMPLETE' for v in complete['rows'].values()),
        'All declared rows must finish before truth access')
    runtime=complete['runtime_commit']
    require(runtime==startup['runtime_commit'] and re.fullmatch('[0-9a-f]{40}',runtime) is not None
        and complete['code_commit']==startup['code_commit']==spec['code']['commit'],'Runtime/preparation commit binding mismatch')
    fixed=[];cohort_physical={}
    for row in rows:
        co=cohorts[row['cohort']]
        binding=dict(run_id=spec['run_id'],row_id=row['row_id'],capsule_id=co['expected_capsule_id'],
            checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],release_commit=runtime)
        path=_mapped(str(Path(row['output_root'])/'predictions'),spec,run_root)
        lane=complete['rows'][row['row_id']]
        value=_validated_row_fixed(path,spec,row,runtime,lane['preflight'])
        require(cohort_physical.setdefault(row['cohort'],value['splits'])==value['splits'],
            'Same cohort model rows changed physical support/query split metadata')
        require(lane['marker']==value['complete'] and lane['release_commit']==runtime
            and lane['expected_model_seed']==row['expected_model_seed'] and lane['expected_checkpoint_sha256']==row['expected_checkpoint_sha256']
            and lane['expected_capsule_id']==co['expected_capsule_id'] and lane['output_root']==row['output_root']
            and lane['prediction_output']==str(Path(row['output_root'])/'predictions'),'Supervisor/lane fixed marker binding mismatch')
        source_paths=lane['source_paths']
        require(source_paths==dict(row_root=row['row_root'],branch_features=row['branch_features'],ground_packet=row['ground_packet'],
            ground_summary=row['ground_summary'],capsule=co['capsule']),
            'Supervisor source paths changed')
        run_binding=value['complete']['run_binding']
        require(all(Path(run_binding[k]).resolve()==Path(v).resolve() for k,v in source_paths.items())
            and Path(run_binding['prediction_output_root']).resolve()==Path(row['output_root']+'/predictions').resolve(),
            'Predictor source path/run output binding mismatch')
        fixed.append(value)
    require(sum(len(v['splits']) for v in fixed)==2400,'Exactly 2400 complete parents required before truth access')
    # Reopen every fixed stream and completion/source identity after all rows
    # validate. A truth reader cannot run from a partially verified prefix.
    for row,old in zip(rows,fixed):
        co=cohorts[row['cohort']]
        reread=load_fixed_predictions(predictions=old['predictions_root'],capsule=co['capsule'],config=spec['benchmark']['config'],expected_binding=old['binding'])
        require(reread==old,'Frozen predictions changed during independent reread')
    require(read(root/'startup.json')==startup and read(root/'complete.json')==complete,'Supervisor changed before truth join')
    resource_rows=[dict(binding=v['binding'],resources=v['complete']['resources']) for v in fixed]
    resources=aggregate_training_resources(resource_rows)
    parents=[];truths={};old_sets={};started=time.perf_counter()
    for row,value in zip(rows,fixed):
        path=str(cohorts[row['cohort']]['truth'])
        if path not in truths:truths[path]=read(path)
        for sid in value['splits']:
            parent=_truth_parent(value,sid,truths[path]);parent['cohort']=row['cohort']
            key=(row['row_id'],parent['receiver'],parent['scenario'],parent['k'],parent['support_seed'])
            paired=(sorted(parent['old_query_ids']),value['splits'][sid]['old_support_ids'])
            require(old_sets.setdefault(key,paired)==paired,'Old physical queries/support changed across registration counts')
            parents.append(parent)
    resources['scoring_truth_join_seconds']=time.perf_counter()-started
    result=dict(schema=SCHEMA,method=METHOD,status=STATUS,scope=SCOPE,run_id=spec['run_id'],group_id=spec['group_id'],
        release_commit=runtime,code_commit=spec['code']['commit'],algorithm=spec['benchmark']['config']['algorithm'],
        support_metric_resources=spec['benchmark']['config']['support_metric_resources'],row_count=len(rows),parent_count=len(parents),
        parents=parents,statistics=_statistics(parents),
        current_metadata=dict(spec=deepcopy(spec),supervisor_startup=startup,supervisor_complete=complete,
            rows=[dict(binding=v['binding'],source_identity=v['complete']['source_identity'],ground_packet_identity=v['complete']['ground_packet_identity'],
                ground_geometry_identity=v['complete']['ground_geometry_identity'],ground_geometry_binding=v['complete']['ground_geometry_binding'],
                row_basis_rank=v['complete']['row_basis_rank'],row_basis_ref=v['complete']['row_basis_ref'],
                row_basis_actual_work=v['complete']['row_basis_actual_work'],parameter_evidence=v['parameter_evidence'],
                streams=v['complete']['streams'],split_count=len(v['splits']),resources=v['complete']['resources'],
                predictor_device=v['startup'].get('device')) for v in fixed]),
        prediction_validation_complete_before_truth=True,prediction_streams_independently_reread=True,
        query_fit_access=False,source_fit_access=False,training_performed=False,model_called=False,selection_feedback_forbidden=True,
        data_reuse_statement='Repeated frozen benchmark data; not a fresh independent confirmation',
        resources=resources,automatic_promotion=False,
        score_units='FRACTIONS; SUBTRACTIONS_ARE_FRACTION_DIFFERENCES',
        metric_definitions=dict(total_old_accuracy_drop='B_old_accuracy-C_old_accuracy',
            adaptation_gain_B_minus_A='B_old_accuracy-A_old_accuracy',
            C_h='HARMONIC_MEAN_COMPUTED_WITHIN_EACH_PHYSICAL_PARENT_BEFORE_AGGREGATION',
            R0='INDEPENDENT_FULL_SUPPORT_R0_B_AND_R0_C_ON_IDENTICAL_PHYSICAL_QUERY_IDS; NOT_NATIVE_GROUND_A',
            C_abs_new_old_gap='ABS_C_OLD_MINUS_C_NEW_WITHIN_PARENT',
            K1='QUERY_BENCHMARK_IS_VALID; NOT_SUPPORT_HELD_DIAGNOSTICS',
            new0='C_EXACT_ACTUAL_B; C_NEW_ACCURACY/H/GAP/NEW_MACRO_F1_NULL'))
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result)
    require(read(output)==result,'Score artifact independent readback failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-root',type=Path)
    a=p.parse_args();result=score_benchmark(spec=read(a.spec),output=a.output,run_root=a.run_root)
    print(json.dumps(dict(status=result['status'],row_count=result['row_count'],parent_count=result['parent_count'])))


if __name__=='__main__':main()
