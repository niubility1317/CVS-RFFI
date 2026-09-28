"""Audit complete BNNA training metadata without scores, truth or feature reads."""
import argparse
from collections import Counter
import csv
import inspect
from itertools import product, zip_longest
import json
import math
from pathlib import Path
import statistics
import subprocess

from collect_d92_mvkme_audit import (check, finite_tree, read_json, json_lines,
    nonnegative, close, stats, validate_risk)


def scalars(record):
    return {k:v for k,v in record.items() if v is None or isinstance(v,(str,int,float,bool))}


def final_summary(record):
    value=scalars(record)
    value.update({'final_'+k:v for k,v in scalars(record.get('final_losses') or {}).items()})
    return value


def validate_losses(value,n):
    for key in ('loss_cls','loss_view','loss_gate','loss_total'):
        nonnegative(value[key],key)
    check(value['loss_cls_weight']==1.,'Classification weight mismatch')
    close(value['loss_view_weight'],1/(n+1),'view weight from actual train K')
    close(value['loss_gate_weight'],1/(n+1),'gate weight from actual train K')
    close(value['loss_total'],value['loss_cls']+(value['loss_view']+value['loss_gate'])/(n+1),'weighted loss')
    for key in ('near_zero_prototype_count','near_zero_physical_mean_count'):
        nonnegative(value[key],key,integer=True)


def validate_training(training,updates,ids,n,c,scope,fold,identity_selected=False):
    check(training['scope']==scope and training['fold']==fold and training['train_k']==n
          and training['train_physical_count']==n*c,'Training scope/count mismatch')
    check(training['training_physical_ids']==sorted(ids) and training['identity_initialization'] is True,
          'Training physical IDs or from-zero initialization mismatch')
    nonnegative(training['fit_seconds'],'training fit seconds')
    rank=training['active_rank']
    check(type(rank) is int and 0<=rank<=8,'Invalid active rank')
    if identity_selected:
        check(rank==0 and training['status']=='IDENTITY_SELECTED' and training['steps']==0 and not updates
              and training['final_losses'] is None and training['final_gradient_norm'] is None,'Identity selected must not train')
        return
    check(training['basis_estimated_from_trainfold_only'] is True,'Basis used heldout or other-row state')
    trace=training['covariance_trace'];nonnegative(trace,'covariance trace')
    close(training['eigen_tolerance'],64*2.220446049250313e-16*trace,'eigen tolerance')
    check(training['covariance_bytes']==160*160*8,'Covariance byte mismatch')
    check(len(training['eigenvalues'])==rank and all(v>training['eigen_tolerance'] for v in training['eigenvalues']),
          'Active eigenvalue mismatch')
    check(training['eigenvalues']==sorted(training['eigenvalues'],reverse=True),'Eigenvalues not descending')
    nonnegative(training['basis_orthogonality_error'],'basis orthogonality')
    check(training['basis_orthogonality_error']<=1e-10,'Nonorthogonal basis')
    check((trace>1e-24 or rank==0),'Nonzero rank for zero view variation')
    expected=64 if rank else 0
    check(training['steps']==len(updates)==expected and training['status']==('TRAINING_BUDGET_COMPLETE' if rank else 'NO_VIEW_VARIATION'),
          'Fixed training budget or rank-zero exception mismatch')
    gate=training['final_gate']
    check(len(gate)==rank and all(0<=v<=.5 for v in gate),'Final gate outside bounds')
    validate_losses(training['final_losses'],n)
    close(training['final_losses']['loss_gate'],sum(v*v for v in gate)/rank if rank else 0,'final gate loss')
    nonnegative(training['final_gradient_norm'],'final gradient norm')
    for key,value in dict(training_feature_bytes=n*c*(4*160+96)*8,cached_nonlinearity_bytes=n*c*4*rank*8,
                          optimizer_state_bytes=rank*2*8).items():
        check(training[key]==value,'Training array bytes mismatch: '+key)
    bound=.5*math.sqrt(rank/160)
    close(training['residual_norm_upper_bound'],bound,'residual upper bound')
    nonnegative(training['measured_max_residual_norm'],'measured residual')
    check(training['measured_max_residual_norm']<=bound+1e-10,'Residual exceeds analytic bound')
    nonnegative(training['measured_max_angle_degrees'],'measured angle')
    nonnegative(training['angular_upper_bound_degrees'],'angle bound')
    nonnegative(training['subunit_identity_count'],'subunit identity count',integer=True)
    check(training['subunit_identity_count']<=n*c*4,'Subunit count exceeds physical views')
    maximum_angle=math.degrees(math.asin(.5 if training['subunit_identity_count'] else bound))
    check(training['angular_upper_bound_degrees']<=maximum_angle+1e-9,'Claimed angular bound exceeds formula')
    check(training['measured_max_angle_degrees']<=training['angular_upper_bound_degrees']+1e-5,'Angle exceeds bound')
    previous=None
    for index,step in enumerate(updates,1):
        check(step['scope']==scope and step['fold']==fold and step['step']==index
              and step['train_k']==n and step['train_physical_count']==n*c and step['active_rank']==rank,'Step scope/order/count mismatch')
        validate_losses(step,n)
        check(step['learning_rate']==.03 and step['loss_evaluation']=='before_update','Step LR/loss timing mismatch')
        for key in ('gradient_norm','step_seconds','elapsed_seconds'): nonnegative(step[key],key)
        for prefix in ('gate_before','gate_after'):
            check(0<=step[prefix+'_min']<=step[prefix+'_max']<=.5,'Gate step bounds violated')
        check(type(step['clipped_coordinates']) is int and 0<=step['clipped_coordinates']<=rank,'Clip count mismatch')
        if index==1:
            check(step['gate_before_min']==step['gate_before_max']==0,'Training did not start from identity')
        else:
            check(step['gate_before_min']==previous['gate_after_min'] and step['gate_before_max']==previous['gate_after_max'],
                  'Step gate continuity mismatch')
        previous=step
    if rank:
        close(min(gate),updates[-1]['gate_after_min'],'final gate after step64 min')
        close(max(gate),updates[-1]['gate_after_max'],'final gate after step64 max')


def validate_fit(row,compact,config):
    finite_tree(row);finite_tree(compact)
    k,c=row['k'],row['classes'];classes=row['registered_classes'];old=config['old_classes']
    check(row['method']=='D92-BNNA-v1' and row['config']==config['algorithm'],'Fit formula mismatch')
    check(type(k) is int and k in config['ks'] and type(c) is int and c-len(old) in config['new_counts'],'K/classes mismatch')
    check(len(classes)==len(set(classes))==c and classes[:len(old)]==old and row['old_classes']==old,'Class registry mismatch')
    check(row['new_count']==c-len(old),'New count mismatch')
    check(row['query_rows_used_for_fit']==row['source_rows_used_for_fit']==row['new_source_payload_bytes']==0
          and row['ground_summary_used'] is row['encoder_updated'] is row['cross_row_adapted_state_reuse'] is False,'Input permission violation')
    f=0 if k==1 else min(k,3)
    check(row['fold_count']==f and len(row['folds'])==f,'Fold count mismatch')
    assignments=row['physical_fold_assignment'];ids=[v['physical_id'] for v in assignments]
    check(len(ids)==len(set(ids))==k*c and all(isinstance(v,str) and v for v in ids),'Physical support ID mismatch')
    check(all(v['class_id'] in classes for v in assignments),'Unknown support class')
    for cls in classes:
        members=sorted((v for v in assignments if v['class_id']==cls),key=lambda v:v['physical_id'])
        check(len(members)==k and [v['fold'] for v in members]==([-1]*k if k==1 else [i%f for i in range(k)]),'Physical class folds mismatch')
    seen_folds=set();expected_steps=[]
    for fold in row['folds']:
        index=fold['fold'];check(type(index) is int and index in range(f) and index not in seen_folds,'Duplicate/unknown fold')
        seen_folds.add(index)
        train=sorted(v['physical_id'] for v in assignments if v['fold']!=index)
        held=sorted(v['physical_id'] for v in assignments if v['fold']==index)
        check(fold['train_ids']==train and fold['heldout_ids']==held and not set(train)&set(held),'Fold physical isolation mismatch')
        n,h=len(train)//c,len(held)//c
        check(fold['train_k']==n and fold['heldout_k']==h,'Fold actual K mismatch')
        updates=[s for s in row['steps'] if s['scope']=='fold' and s['fold']==index]
        validate_training(fold['training'],updates,train,n,c,'fold',index)
        expected_steps.extend(updates)
        for arm in ('identity','trained'): validate_risk(fold[arm+'_risk'],c,len(old))
    candidates=row['candidates']
    if k==1:
        check(row['selected']=='trained' and row['selection']=='k1_fixed_trained' and row['candidate_count']==0
              and candidates==[] and all(row['selected_'+n] is None for n in ('objective','macro_nll','old_nll','new_nll')),'K1 must not select or validate')
    else:
        check(row['selection']=='physical_support_oof' and row['candidate_count']==len(candidates)==2
              and {v['candidate'] for v in candidates}=={'identity','trained'},'OOF arm coverage mismatch')
        for candidate in candidates:
            validate_risk(candidate,c,len(old))
            for name in ('macro_nll','old_nll','new_nll'):
                if candidate[name] is not None:
                    close(candidate[name],sum(v[candidate['candidate']+'_risk'][name]*v['heldout_k']/k for v in row['folds']),'OOF physical weighted '+name)
        chosen=min(candidates,key=lambda v:(round(v['objective'],10),round(v['macro_nll'],10),v['candidate']!='identity'))
        check(row['selected']==chosen['candidate'],'Fixed OOF selection mismatch')
        for name in ('objective','macro_nll','old_nll','new_nll'): check(row['selected_'+name]==chosen[name],'Selected OOF loss mismatch')
    final_steps=[s for s in row['steps'] if s['scope']=='final']
    validate_training(row['final_fit'],final_steps,sorted(ids),k,c,'final',None,row['selected']=='identity')
    expected_steps.extend(final_steps)
    check(expected_steps==row['steps'] and row['optimizer_steps']==len(expected_steps),'Unexpected or duplicate optimizer steps')
    check(row['final_optimizer_steps']==row['final_fit']['steps'] and row['optimizer_status']==row['final_fit']['status'],'Final optimizer status mismatch')
    rank=row['active_rank'];check(rank==row['final_fit']['active_rank'],'Final rank mismatch')
    for key,value in dict(head_bytes=8*256*c,basis_bytes=8*160*rank,gate_bytes=8*rank,
                          persistent_state_bytes=8*(256*c+161*rank),support_feature_bytes=8*k*c*(4*160+96)).items():
        check(type(row[key]) is int and row[key]==value,'State byte mismatch: '+key)
    for key in ('split_id','k','classes','selected','selection','candidate_count','fold_count','selected_objective',
        'selected_macro_nll','selected_old_nll','selected_new_nll','fit_seconds','active_rank','head_bytes','basis_bytes','gate_bytes',
        'persistent_state_bytes','final_optimizer_steps','receiver','scenario','support_seed','new_count',
        'fit_call_seconds','query_score_seconds','prediction_write_seconds'):
        check(row[key]==compact[key],'Compact/trace mismatch: '+key)
    check(compact['optimizer_steps_recorded']==len(row['steps']) and compact['final_step']==(scalars(final_steps[-1]) if final_steps else None)
          and compact['final_fit']==final_summary(row['final_fit']),'Compact final state mismatch')
    check(compact['query_rows_used_for_fit']==compact['source_rows_used_for_fit']==0 and compact['source_validation'] is None
          and bool(compact['source_validation_reason']) and compact['epoch'] is None and bool(compact['epoch_reason']),'Compact permission/epoch mismatch')
    for key in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds'): nonnegative(compact[key],key)
    check(compact['fit_seconds']<=compact['fit_call_seconds']+1e-8,'Fit time exceeds wrapper')
    check(sum(compact[k] for k in ('fit_call_seconds','query_score_seconds','prediction_write_seconds'))<=compact['total_seconds']+1e-8,'Timing sum exceeds total')


def csv_matches(path,records):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        for actual,expected in zip_longest(reader,records):
            check(actual is not None and expected is not None,'CSV/JSONL row count mismatch')
            converted={k:'N/A' if v is None else json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else str(v) for k,v in expected.items()}
            check(actual==converted,'CSV/JSONL content mismatch')


def validate_resources(row,startup,marker,config):
    root=Path(row['output_root'])/'bnna_features'
    feature=read_json(root/'features_complete.json');provenance=read_json(root/'checkpoint_provenance.json')
    check(feature['status']=='BNNA_FEATURES_COMPLETE' and feature['schema']=='d92_bnna_received_views_v1'
          and feature['algorithm']==config['algorithm'] and feature['capsule_id']==config['capsule_id']
          and feature['checkpoint_sha256']==row['expected_checkpoint_sha256'] and feature['model_seed']==row['seeds']['model']
          and feature['classes']==config['old_classes'] and feature['dtype']=='float32','Feature binding mismatch')
    check(feature['query_used_for_fitting'] is feature['source_data_access'] is feature['truth_read'] is feature['encoder_updated'] is False
          and feature['native_eval'] is feature['native_buffers_unchanged'] is True
          and feature['view_count_per_observation']==4 and feature['new_ground_statistics_bytes']==0,'Feature input/freeze violation')
    check(startup['config']=={'algorithm':config['algorithm']} and startup['checkpoint_sha256']==row['expected_checkpoint_sha256']
          and startup['capsule_id']==config['capsule_id'] and startup['model_seed']==row['seeds']['model'],'Predictor startup binding mismatch')
    check(startup['query_fit_access'] is startup['truth_read'] is startup['source_data_access'] is startup['ground_summary_access'] is startup['cross_row_adapted_state_reuse'] is False,'Predictor permission violation')
    count=feature['count'];nonnegative(count,'physical count',integer=True,positive=True)
    check(feature['identity_views_shape']==[count,4,160] and feature['fft_shape']==[count,96]
          and feature['identity_views_bytes']==count*4*160*4 and feature['fft_bytes']==count*96*4
          and feature['feature_array_bytes']==count*2944,'Received cache shape/bytes mismatch')
    cache=root/'received_bnna_features.npz'
    check(feature['feature_file_bytes']==cache.stat().st_size,'Cache file bytes mismatch')
    payload=marker['payload_audit'];check(startup['payload_audit']==payload,'Startup/completion payload mismatch')
    check(payload.get('model_already_deployed') is None and payload.get('model_incremental_transfer_bytes') is None,
          'Model deployment cannot be inferred from checkpoint reuse')
    check(payload['new_ground_statistics_bytes']==0 and payload['model_file_bytes']==feature['model_file_bytes']
          and payload['received_feature_array_bytes']==feature['feature_array_bytes']
          and payload['received_feature_file_bytes']==feature['feature_file_bytes'],'Payload byte mismatch')
    check(provenance['checkpoint_sha256']==row['expected_checkpoint_sha256'] and provenance['model_file_bytes']==feature['model_file_bytes'],'Model provenance byte mismatch')
    nonnegative(feature['model_file_bytes'],'model package bytes',integer=True,positive=True)
    nonnegative(feature['feature_seconds'],'feature time')
    return dict(feature_extraction_seconds=feature['feature_seconds'],physical_observations=count,native_views_per_observation=4,
        recorded_native_view_count=count*4,received_cache_array_bytes=feature['feature_array_bytes'],received_cache_file_bytes=cache.stat().st_size,
        existing_frozen_model_file_bytes=feature['model_file_bytes'],model_incremental_transfer_bytes=None,
        model_transfer_unavailable_reason='Deployment state unknown; training checkpoint package is not a measured minimum inference package',
        feature_peak_process_rss_bytes=feature.get('peak_process_rss_bytes'),feature_peak_rss_unavailable_reason=feature.get('peak_process_rss_reason'),
        new_source_payload_bytes=0,ground_summary_used=False,cache_scope='Received-only local cache, not a ground payload')


def audit_run(config):
    root=Path(config['root']);check(read_json(root/'complete.json')['status']=='SCORED','Collection requires SCORED completion')
    state=read_json(root/'state.json')
    check(set(state)=={r['row_id'] for r in config['rows']} and all(v['status']=='PREDICTIONS_COMPLETE' for v in state.values()),'Incomplete model rows')
    expected=set(product(config['receivers'],config['scenarios'],config['ks'],config['new_counts'],config['support_seeds']))
    check(len(expected)==config['splits'],'Matrix split count mismatch');models=[]
    for row in config['rows']:
        folder=Path(row['output_root'])/'bnna';marker=read_json(folder/'predictions_complete.json')
        check(marker['status']=='PREDICTIONS_COMPLETE' and marker['split_count']==marker['predictions']==config['splits']
              and marker['capsule_id']==config['capsule_id'] and marker['truth_read'] is marker['source_data_access'] is False
              and marker['new_ground_statistics_bytes']==0,'Prediction completion mismatch')
        resources=validate_resources(row,read_json(folder/'startup.json'),marker,config)
        records=list(json_lines(folder/'compact.jsonl'));compact={v['split_id']:v for v in records}
        check(len(records)==len(compact)==config['splits'],'Missing/duplicate compact rows')
        csv_matches(folder/'compact.csv',records)
        csv_matches(folder/'training_steps.csv',json_lines(folder/'training_steps.jsonl'))
        step_stream=iter(json_lines(folder/'training_steps.jsonl'));seen=set();cells=set();items=[]
        for trace in json_lines(folder/'fit_trace.jsonl'):
            sid=trace['split_id'];check(sid in compact and sid not in seen,'Unbound or duplicate fit trace');seen.add(sid)
            validate_fit(trace,compact[sid],config)
            cell=tuple(trace[k] for k in ('receiver','scenario','k','new_count','support_seed'))
            check(cell in expected and cell not in cells,'Unexpected/duplicate cell');cells.add(cell)
            for step in trace['steps']:
                check(next(step_stream,None)==dict(split_id=sid,**scalars(step)),'Training-step stream differs from complete trace')
            items.append(compact[sid])
        check(next(step_stream,None) is None,'Extra unbound training steps')
        check(seen==set(compact) and cells==expected,'Incomplete fit matrix')
        check(all(v['total']==config['splits'] for v in records) and {v['completed'] for v in records}==set(range(1,config['splits']+1)),'Compact progress mismatch')
        models.append(dict(row_id=row['row_id'],fits=len(items),fixed_k1_fits=sum(v['k']==1 for v in items),support_cv_fits=sum(v['k']>1 for v in items),
            optimizer_steps=sum(v['optimizer_steps_recorded'] for v in items),final_optimizer_steps=sum(v['final_optimizer_steps'] for v in items),
            final_status_counts=dict(Counter(v['final_fit']['status'] for v in items)),selected_arm_counts=dict(Counter(v['selected'] for v in items)),
            resources=resources,timing={k:stats([v[k] for v in items]) for k in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds')},
            numeric_state_byte_ranges={k:stats([v[k] for v in items]) for k in ('head_bytes','basis_bytes','gate_bytes','persistent_state_bytes')},
            peak_process_rss_bytes=marker.get('peak_process_rss_bytes'),per_k=[dict(k=k,fits=sum(v['k']==k for v in items),
                optimizer_steps=sum(v['optimizer_steps_recorded'] for v in items if v['k']==k),
                selected_arm_counts=dict(Counter(v['selected'] for v in items if v['k']==k))) for k in config['ks']],
            trace_file=str(folder/'fit_trace.jsonl'),trace_file_bytes=(folder/'fit_trace.jsonl').stat().st_size,
            training_steps_file_bytes=(folder/'training_steps.jsonl').stat().st_size))
    return dict(status='VERIFIED',method='D92-BNNA-v1',models=models,total_fits=sum(v['fits'] for v in models),
        optimizer_steps=sum(v['optimizer_steps'] for v in models),fixed_k1_fits=sum(v['fixed_k1_fits'] for v in models),
        support_cv_fits=sum(v['support_cv_fits'] for v in models),query_rows_used_for_fit=0,source_rows_used_for_fit=0,new_source_payload_bytes=0,
        ground_summary_used=False,raw_traces_preserved=True,parameter_feedback_forbidden=True,optimizer_convergence_claim=False,
        scope='Complete fit/fold/step metadata consistency audit, not a mathematical reexecution; no scores, truth, predictions, checkpoints or feature values read')


def audit_config(spec):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
    from cvsrffi.stage2_d92_bnna import FROZEN_CONFIG
    from run_d92_confirmation import candidate_definition
    conf=spec['confirmation'];data=spec['data']
    check(candidate_definition(conf)['candidate_method']=='D92-BNNA-v1' and conf['candidate']==FROZEN_CONFIG,'Frozen BNNA only')
    check(len(spec['rows'])==4 and sorted(r['seeds']['model'] for r in spec['rows'])==list(range(2026092701,2026092705)),'Expected four frozen model seeds')
    check(data['k']==[1,5,10,20] and data['new_class_counts']==[0,2,5,10,20] and len(data['scenarios'])==3
          and len(data['support_seeds'])==5 and len(data['target_receivers']) in (1,3),'Expected full cohort matrix')
    count=math.prod(len(data[k]) for k in ('target_receivers','scenarios','k','new_class_counts','support_seeds'))
    check(count==conf['expected_split_count']==conf['splits_per_model'],'Spec split count mismatch')
    return dict(root=spec['execution']['remote_run_root'],rows=spec['rows'],splits=count,receivers=data['target_receivers'],
        scenarios=data['scenarios'],ks=data['k'],new_counts=data['new_class_counts'],support_seeds=data['support_seeds'],
        old_classes=conf['old_classes'],algorithm=conf['candidate'],capsule_id=conf['reuse_validated_capsule_id'])


def remote_script(config):
    functions=(check,finite_tree,read_json,json_lines,nonnegative,close,stats,validate_risk,scalars,final_summary,
               validate_losses,validate_training,validate_fit,csv_matches,validate_resources,audit_run)
    return ('import json,math,statistics,csv\nfrom pathlib import Path\nfrom collections import Counter\nfrom itertools import product,zip_longest\n'
            +'\n'.join(inspect.getsource(f) for f in functions)+'\nprint(json.dumps(audit_run('+repr(config)+'),allow_nan=False))\n')


def main():
    from read_d92_run import FLAGS
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    script=remote_script(audit_config(read_json(args.spec)));compile(script,'remote_audit','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,check=True)
    value=json.loads(result.stdout);finite_tree(value);check(value.get('status')=='VERIFIED','Audit incomplete')
    with args.output.open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='models'},allow_nan=False))


if __name__=='__main__': main()
