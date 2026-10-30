"""Frozen F0/G evidence, one U-only binary graph, then row-wise future Q."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import torch
from . import star_binary_graph_core as core
from . import star_reciprocal_run as reciprocal

previous=reciprocal.previous
registry=reciprocal.core
public=reciprocal.public
_read,_write,_npz=reciprocal._read,reciprocal._write,reciprocal._npz
_smoke,_encode,_margin=reciprocal._smoke,reciprocal._encode,reciprocal._margin
RUN_ID='20261030-phase3-star-binary-graph-s2026092701-r01'
SCHEMA='pamf_phase3_star_binary_graph_prediction_v1'
STATUS='STAR_BINARY_GRAPH_PREDICTIONS_COMPLETE'
U_STATUS='STAR_BINARY_GRAPH_U_STATE_FROZEN'
STATES=('DA0_REG0','DA1_REG0','DA0_REG1','DA1_REG1')
DAS=('DA0','DA1')
PARENT=reciprocal.PARENT
PARENT_COMMIT=reciprocal.PARENT_COMMIT
FIELDS={'run_id','code_commit','source_root','U_public_root','U_capsule_id','U_split_id','Q_public_root','Q_capsule_id','Q_split_id','output_root','batch_size','dura_checkpoint','dura_release'}
PRED_FIELDS={'ids','U_ids','state_names','prediction','closed_prediction','baseline_closed_prediction','knownness'}|{'joint_margins_'+s for s in STATES}
DIAG_FIELDS=reciprocal.DIAG_FIELDS
EVIDENCE_FIELDS={'ids','unit','logits','margins'}
FP64_RECONSTRUCTION_ATOL=1e-12


def _fp64_close(actual,expected):
    return (actual.dtype==expected.dtype==np.float64 and actual.shape==expected.shape
        and np.isfinite(actual).all() and np.isfinite(expected).all()
        and np.allclose(actual,expected,rtol=0.,atol=FP64_RECONSTRUCTION_ATOL))


def _graph_diagnostics_close(actual,expected):
    if not isinstance(actual,dict) or set(actual)!=set(expected):return False
    floating={'solve_residual_linf','energy_before','energy_after','f_min','f_max'}
    for key,value in expected.items():
        saved=actual[key]
        if key in floating and value is not None:
            if (type(saved) is not float or not np.isfinite(saved) or not np.isfinite(value)
                or abs(saved-value)>FP64_RECONSTRUCTION_ATOL):return False
        elif type(saved) is not type(value) or saved!=value:return False
    return True


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
        or not isinstance(c.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None
        or type(c.get('batch_size')) is not int or c['batch_size']!=128
        or any(not isinstance(c[k],str) or not c[k].strip() for k in FIELDS-{'batch_size'})):
        raise ValueError('Fixed independent binary-graph public thirteen-field config required; no truth/K/quotas')
    fixed=dict(U_capsule_id='p3dev-star-g10-6e1292b62ab09286ebf1a0df',U_split_id='83c5cc1fe2f2a1512dbc1287',
        Q_capsule_id='p3dev-star-g10-71a5a0dec23ac6ccc3b479c2',Q_split_id='05de65cd4818f3e5e33ced5c')
    if any(c[k]!=v for k,v in fixed.items()):raise ValueError('Only existing RX24-1 G10 U384/Q864 handles allowed')
    for role,folder in (('U','discovery'),('Q','evaluation')):
        p=Path(c[role+'_public_root'])
        if p.name!='public' or p.parent.name!=folder:raise ValueError('Separate public discovery/future roles required')
    source=Path(c['source_root'])
    if source.name!='source' or source.parent.name!=PARENT:raise ValueError('Exact frozen neighborhood aggregate source required')
    if any(Path(c['output_root']).resolve().is_relative_to(Path(c[k]).resolve()) for k in ('source_root','U_public_root','Q_public_root')):
        raise ValueError('Immutable output must not replace source or public inputs')
    return c


def _load_npz(path,fields):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=fields:raise ValueError('Public artifact fields differ: '+str(path))
        return {k:z[k].copy() for k in z.files}


def _save_registry(folder,da,state,diag,ids):
    _npz(folder/(da+'.npz'),{k:state[k] for k in ('centers','radii')})
    _npz(folder/(da+'_diagnostics.npz'),dict(ids=ids,**{k:diag[k] for k in DIAG_FIELDS-{'ids'}}))
    _write(folder/(da+'_diagnostics.json'),dict(components=diag['components'],registered_count=diag['registered_count']))


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable binary-graph output required')
    started=time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prior=previous.load_source(c['source_root'])
    if prior['marker'].get('code_commit')!=PARENT_COMMIT or prior['marker'].get('run_id')!=PARENT:
        raise ValueError('Exact original-F0 aggregate source identity required')
    backend=previous._target_backend(prior,c,device)
    before=copy.deepcopy(prior);snapshots=previous._snapshot(backend)
    def check():
        previous._check_models(snapshots)
        if not previous.relative_source._same(prior,before):raise ValueError('Frozen source prior changed')
    smoke=_smoke(backend,prior,device);check()
    out.mkdir(parents=True);folder=out/'adaptation';folder.mkdir();_write(out/'smoke.json',smoke)
    cfg=dict(c,U_expected_count=384,Q_expected_count=864)
    _,idsU,iq=public._open(cfg,'U');units=[];logits=[];margins=[]
    try:
        for start in range(0,len(idsU),128):
            b0,l0,b1,l1=_encode(backend,iq[start:start+128],device)
            units.append(reciprocal._unit(b1['fused160']));logits.append(l1);margins.append(_margin(b1,l1,b0,l0,prior))
            print('STAR_BINARY_GRAPH_U_PROGRESS',min(start+128,len(idsU)),len(idsU),flush=True)
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()
    unit=np.concatenate(units);logit=np.concatenate(logits);margin=np.concatenate(margins)
    graph,graph_diag=core.fit(unit,logit,margin,idsU)
    positive,negative=core.seeds(logit,margin)
    _npz(folder/'U_evidence.npz',dict(ids=idsU,unit=unit,logits=logit,margins=margin))
    _npz(folder/'graph.npz',{k:graph[k] for k in ('unit','ids','sigma','f')})
    _write(folder/'graph.json',dict(schema=core.STATE_SCHEMA,skipped=graph['skipped'],diagnostics=graph_diag))
    states={}
    for da,current in (('DA0',np.tanh(margin)),('DA1',graph['f'])):
        state,diag=registry.fit_registry(unit,current,margin,positive,idsU)
        states[da]=state;_save_registry(folder,da,state,diag,idsU)
    counts={da:len(s['centers']) for da,s in states.items()}
    check();frozen_graph=copy.deepcopy(graph);frozen_states=copy.deepcopy(states)
    _write(folder/'complete.json',dict(status=U_STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=c['code_commit'],
        U_count=384,Q_rows_read=0,truth_read=False,target_label_inputs=0,true_K_inputs=0,source_updates=0,
        threshold_updates=0,prototype_counts=counts,graph_diagnostics=graph_diag,registries_frozen_before_Q=True,
        U_memory_deployed=True,source_bank_deployed=False))
    del units,logits,margins,unit,logit,margin,positive,negative
    _,idsQ,iq=public._open(cfg,'Q')
    if set(idsU.tolist())&set(idsQ.tolist()):
        if hasattr(iq,'_mmap'):iq._mmap.close()
        raise ValueError('U and future Q IDs overlap')
    n=len(idsQ)
    p=dict(ids=idsQ,U_ids=idsU,state_names=np.asarray(STATES),prediction=np.empty((4,n),np.int64),
        closed_prediction=np.empty((2,n),np.int64),baseline_closed_prediction=np.empty(n,np.int64),knownness=np.empty((4,n),np.float64))
    q=dict(ids=idsQ,unit=np.empty((n,160),np.float32),logits=np.empty((n,6),np.float32),margins=np.empty(n,np.float64))
    for name in STATES:p['joint_margins_'+name]=np.empty((n,6+(counts[name[:3]] if name.endswith('REG1') else 0)),np.float64)
    try:
        for start in range(0,n,128):
            stop=min(start+128,n);b0,l0,b1,l1=_encode(backend,iq[start:stop],device)
            unit=reciprocal._unit(b1['fused160']);rawmargin=_margin(b1,l1,b0,l0,prior)
            q['unit'][start:stop]=unit;q['logits'][start:stop]=l1;q['margins'][start:stop]=rawmargin
            p['baseline_closed_prediction'][start:stop]=l1.argmax(1)
            current={'DA0':np.tanh(rawmargin),'DA1':core.margins(unit,rawmargin,graph)}
            for i,name in enumerate(STATES):
                da=name[:3];prediction,closed,knownness,joint=registry.infer(unit,l1,current[da],states[da],name.endswith('REG1'))
                p['prediction'][i,start:stop]=prediction;p['closed_prediction'][int(da=='DA1'),start:stop]=closed
                p['knownness'][i,start:stop]=knownness;p['joint_margins_'+name][start:stop]=joint
            check()
            if not previous.relative_source._same(graph,frozen_graph) or not previous.relative_source._same(states,frozen_states):
                raise ValueError('Frozen U memory/registry changed during Q')
            print('STAR_BINARY_GRAPH_Q_PROGRESS',stop,n,flush=True)
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()
    _npz(out/'Q_evidence.npz',q);_npz(out/'predictions.npz',p)
    costs=dict(F0_real_full_forward_rows=2496,F0_synthetic_full_forward_rows=4,G_real_forward_rows=1248,
        G_synthetic_forward_rows=2,default_head_smoke_feature_rows=2,source_IQ_or_feature_row_reads=0,
        source_F0_head_G_updates=0,optimizer_updates=0,new_neural_parameters=0,graph_solve_calls=int(not graph['skipped']),
        U_memory_rows=384,U_memory_numeric_bytes=int(sum(graph[k].nbytes for k in ('unit','sigma','f'))),
        prior_models_stored=4,prior_models_scored_per_evidence=4,prior_evidence_real_row_calls=1248,
        prior_evidence_synthetic_row_calls=2,F0_parameters=232993,G_parameters=266951,stored_readout_parameters=12932,
        resident_neural_parameters=512876,B1_latency_measured=False,ancestor_costs_preserved=True)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,states=list(STATES),
        primary_state='DA1_REG1',condition='P1',view_family='practical_cross_receiver',classification_path='P1',
        rejection_input_paths=['P1','P0'],count=864,U_count=384,U_day='2021-03-08',Q_day='2021-03-15',
        prototype_counts=counts,graph_diagnostics=graph_diag,source_frozen_before_U=True,registry_frozen_before_Q=True,
        U_graph_frozen_before_Q=True,U_memory_deployed=True,U_memory_rows=384,source_bank_deployed=False,
        model_parameters_buffers_unchanged=True,features_unchanged=True,identity_predictions_unchanged=True,
        truth_read=False,query_fit=False,algorithm_target_label_inputs=0,true_K_inputs=0,new_threshold_calibrations=0,
        formal_target_reads=0,source_parameters_updated=0,identity_head_updates=0,optimizer_updates=0,all_new_IQ_satellite=True,
        original_F0_identity=backend['reader_identity'],dura_identity=backend['dura_identity'],
        source_prior_run=PARENT,source_prior_code=PARENT_COMMIT,costs=costs,final_phase3_complete=False,
        full_goal_status='ACTIVE',seconds=time.perf_counter()-started)
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def _predictions(root):
    root=Path(root);done=_read(root/'complete.json');c=done.get('config');validate_config(c)
    if (done.get('status')!=STATUS or done.get('schema')!=SCHEMA or done.get('run_id')!=RUN_ID
        or done.get('code_commit')!=c['code_commit'] or done.get('states')!=list(STATES)
        or done.get('count')!=864 or done.get('U_count')!=384 or done.get('source_prior_code')!=PARENT_COMMIT
        or any(done.get(k) is not True for k in ('source_frozen_before_U','registry_frozen_before_Q','U_graph_frozen_before_Q','U_memory_deployed','features_unchanged','identity_predictions_unchanged','model_parameters_buffers_unchanged','all_new_IQ_satellite'))
        or any(done.get(k)!=0 for k in ('algorithm_target_label_inputs','true_K_inputs','new_threshold_calibrations','formal_target_reads','source_parameters_updated','identity_head_updates','optimizer_updates'))
        or done.get('truth_read') is not False or done.get('query_fit') is not False or done.get('source_bank_deployed') is not False):
        raise ValueError('Complete frozen four-state source-free target predictions required')
    frozen=_read(root/'adaptation/complete.json')
    if (frozen.get('status')!=U_STATUS or frozen.get('run_id')!=RUN_ID or frozen.get('code_commit')!=done['code_commit']
        or frozen.get('Q_rows_read')!=0 or frozen.get('truth_read') is not False or frozen.get('prototype_counts')!=done['prototype_counts']):
        raise ValueError('Graph and registries must freeze before any Q')
    p=_load_npz(root/'predictions.npz',PRED_FIELDS)
    if (not reciprocal._ids(p['ids'],864) or not reciprocal._ids(p['U_ids'],384)
        or set(p['ids'].tolist())&set(p['U_ids'].tolist()) or p['state_names'].tolist()!=list(STATES)
        or p['prediction'].shape!=(4,864) or p['prediction'].dtype!=np.int64
        or p['knownness'].shape!=(4,864) or p['knownness'].dtype!=np.float64 or not np.isfinite(p['knownness']).all()
        or p['closed_prediction'].shape!=(2,864) or p['closed_prediction'].dtype!=np.int64
        or p['baseline_closed_prediction'].shape!=(864,) or p['baseline_closed_prediction'].dtype!=np.int64):
        raise ValueError('Independent aligned four-state IDs and predictions required')
    evidence=_load_npz(root/'adaptation/U_evidence.npz',EVIDENCE_FIELDS)
    if not np.array_equal(evidence['ids'],p['U_ids']):raise ValueError('U memory alignment differs')
    graph,diag=core.fit(evidence['unit'],evidence['logits'],evidence['margins'],evidence['ids'])
    saved=_load_npz(root/'adaptation/graph.npz',{'unit','ids','sigma','f'});meta=_read(root/'adaptation/graph.json')
    if (any(saved[k].dtype!=graph[k].dtype or not np.array_equal(saved[k],graph[k]) for k in ('unit','ids','sigma'))
        or (not np.array_equal(saved['f'],graph['f']) if graph['skipped'] else not _fp64_close(saved['f'],graph['f']))
        or set(meta)!={'schema','skipped','diagnostics'} or meta['schema']!=core.STATE_SCHEMA
        or type(meta['skipped']) is not bool or meta['skipped']!=graph['skipped']
        or not _graph_diagnostics_close(meta['diagnostics'],diag)
        or done['graph_diagnostics']!=meta['diagnostics'] or frozen.get('graph_diagnostics')!=meta['diagnostics']):
        raise ValueError('Frozen U graph memory does not reconstruct')
    # Verify independently, then reconstruct deployment decisions from the exact
    # saved memory rather than substituting a platform-specific LAPACK result.
    graph=core._state(dict(schema=core.STATE_SCHEMA,skipped=meta['skipped'],**saved))
    states={};diagnostics={}
    for da in DAS:
        s=_load_npz(root/'adaptation'/(da+'.npz'),{'centers','radii'})
        states[da]=dict(schema=registry.REGISTRY_SCHEMA,**s)
        diagnostics[da]=_load_npz(root/'adaptation'/(da+'_diagnostics.npz'),DIAG_FIELDS)
        diagnostics[da].update(_read(root/'adaptation'/(da+'_diagnostics.json')))
        current=np.tanh(evidence['margins']) if da=='DA0' else graph['f']
        positive,_=core.seeds(evidence['logits'],evidence['margins'])
        rebuilt,rd=registry.fit_registry(evidence['unit'],current,evidence['margins'],positive,evidence['ids'])
        if (not previous.relative_source._same(rebuilt,states[da]) or done['prototype_counts'][da]!=len(s['centers'])
            or not np.array_equal(diagnostics[da]['ids'],p['U_ids'])
            or any(not np.array_equal(diagnostics[da][k],rd[k]) for k in DIAG_FIELDS-{'ids'})
            or diagnostics[da]['components']!=rd['components'] or diagnostics[da]['registered_count']!=rd['registered_count']):
            raise ValueError('U-only arbitrary-K registry does not reconstruct')
    q=_load_npz(root/'Q_evidence.npz',EVIDENCE_FIELDS)
    if not np.array_equal(q['ids'],p['ids']):raise ValueError('Q evidence alignment differs')
    current={'DA0':np.tanh(q['margins']),'DA1':core.margins(q['unit'],q['margins'],graph)}
    for i,name in enumerate(STATES):
        prediction,closed,knownness,joint=registry.infer(q['unit'],q['logits'],current[name[:3]],states[name[:3]],name.endswith('REG1'))
        actual=p['joint_margins_'+name]
        if (not _fp64_close(actual,joint) or not np.array_equal(p['prediction'][i],prediction)
            or not _fp64_close(p['knownness'][i],knownness)
            or not np.array_equal(p['knownness'][i]>=0,knownness>=0)
            or not np.array_equal(actual.max(1),p['knownness'][i])
            or not np.array_equal(np.where(actual.max(1)>=0,actual.argmax(1),-1),p['prediction'][i])
            or not np.array_equal(p['closed_prediction'][int(name.startswith('DA1'))],closed)
            or not np.array_equal(p['baseline_closed_prediction'],closed)):
            raise ValueError('Fixed identity, graph margins, joint decisions or old gate differ')
    return done,p,diagnostics,states


def main():
    print(json.dumps(dict(event='STAR_BINARY_GRAPH_RUN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run']);parser.add_argument('--config',required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args();run(_read(args.config),args.device)


if __name__=='__main__':main()
