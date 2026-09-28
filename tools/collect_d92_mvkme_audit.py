"""Audit complete MVKME fit/resource metadata; never read target scores or truth."""
import argparse
from collections import Counter
import inspect
from itertools import product
import json
import math
from pathlib import Path
import statistics
import subprocess


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values(): finite_tree(child)
    elif isinstance(value, list):
        for child in value: finite_tree(child)
    elif type(value) in (int, float):
        check(math.isfinite(value), 'Nonfinite fit/resource metadata')


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream: value=json.load(stream)
    finite_tree(value)
    return value


def json_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            value=json.loads(line);finite_tree(value);yield value


def nonnegative(value, name, *, integer=False, positive=False):
    check(type(value) is int if integer else type(value) in (int,float), 'Invalid numeric type: '+name)
    check(math.isfinite(value) and (value>0 if positive else value>=0), 'Invalid numeric value: '+name)


def close(a,b,name):
    check(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10), 'Inconsistent '+name)


def stats(values):
    values=[v for v in values if v is not None]
    return dict(min=min(values),mean=statistics.mean(values),max=max(values),total=sum(values)) if values else None


def validate_risk(row,c,old_count):
    for key in ('objective','macro_nll'): nonnegative(row[key],key)
    groups=[]
    for key,count in (('old_nll',old_count),('new_nll',c-old_count)):
        if count:
            nonnegative(row[key],key);groups.append(row[key])
        else: check(row[key] is None, 'Absent group NLL must be null')
    close(row['objective'],max(groups),'max group NLL')
    close(row['macro_nll'],((row['old_nll'] or 0)*old_count+(row['new_nll'] or 0)*(c-old_count))/c,'macro NLL')


def validate_fit(row,compact,config):
    finite_tree(row);finite_tree(compact)
    k,c=row['k'],row['classes'];old_count=len(config['old_classes'])
    check(row['method']=='D92-MVKME-v1' and row['config']==config['algorithm'],'Fit formula mismatch')
    check(type(k) is int and k in config['ks'] and type(c) is int and c-old_count in config['new_counts'], 'Fit K/classes mismatch')
    classes=row['registered_classes']
    check(len(classes)==len(set(classes))==c and all(isinstance(v,str) and v for v in classes)
          and set(config['old_classes']).issubset(classes), 'Invalid physical class registry')
    check(row['old_classes']==config['old_classes'],'Old class membership mismatch')
    check(row['new_count']==c-old_count,'Registered new count mismatch')
    check(row['query_rows_used_for_fit']==row['source_rows_used_for_fit']==row['new_source_payload_bytes']==0
          and row['ground_summary_used'] is False and row['cross_row_adapted_state_reuse'] is False,'Source/query/summary fit violation')
    check(row['optimizer_status']=='closed_form_complete' and row['optimizer_steps'] is None
          and row['learning_rate'] is None and row['gradient_norm'] is None and bool(row['unavailable_reason']), 'Closed-form status mismatch')
    expected_bytes=dict(head_bytes=8*768*c,fourier_matrix_bytes=8*128*256,
        persistent_state_bytes=8*768*c+8*128*256,support_feature_cache_bytes=8*k*c*3*256,gram_bytes=8*(k*c)**2)
    for key,value in expected_bytes.items():
        check(type(row[key]) is int and row[key]==value, 'Array byte accounting mismatch: '+key)
    nonnegative(row['fit_seconds'],'fit_seconds')
    for key in ('k','classes','selected','selection','candidate_count','fold_count','selected_objective',
                'selected_macro_nll','selected_old_nll','selected_new_nll','fit_seconds','head_bytes',
                'fourier_matrix_bytes','persistent_state_bytes','receiver','scenario','support_seed','new_count',
                'fit_call_seconds','query_score_seconds','prediction_write_seconds'):
        check(compact[key]==row[key], 'Compact/trace mismatch: '+key)
    check(compact['query_rows_used_for_fit']==compact['source_rows_used_for_fit']==0
          and compact['learning_rate'] is None and compact['gradient'] is None
          and compact['source_validation'] is None and bool(compact['source_validation_reason'])
          and bool(compact['unavailable_reason']), 'Compact boundary/status mismatch')
    for key in ('fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds'):
        nonnegative(compact[key],key)
    check(compact['fit_call_seconds']+1e-8>=compact['fit_seconds'],'Core fit time exceeds measured fit call')
    check(compact['total_seconds']+1e-8>=compact['fit_call_seconds']+compact['query_score_seconds']+compact['prediction_write_seconds'],
          'Separated time components exceed measured row total')
    if compact.get('peak_process_rss_bytes') is not None:
        nonnegative(compact['peak_process_rss_bytes'],'RSS',integer=True)
    selected=row['selected']
    check(set(selected)=={'eta','gamma'} and selected['eta'] in (0,.5,1) and selected['gamma'] in (.01,.1,1), 'Selected grid mismatch')
    f=0 if k==1 else min(k,3)
    check(row['fold_count']==f,'Fold count mismatch')
    assignment=row['physical_fold_assignment']
    check(len(assignment)==k*c and len({v['physical_id'] for v in assignment})==k*c,'Duplicate/missing physical support ID')
    for entry in assignment:
        check(isinstance(entry['physical_id'],str) and bool(entry['physical_id']) and entry['class_id'] in classes,
              'Invalid physical assignment')
    for cls in classes:
        members=sorted((v for v in assignment if v['class_id']==cls),key=lambda v:v['physical_id'])
        check(len(members)==k,'Physical support K differs by class')
        check([v['fold'] for v in members]==([-1]*k if k==1 else [i%f for i in range(k)]), 'Physical class fold assignment mismatch')
    folds=[s for s in row['steps'] if s['event']=='PHYSICAL_SUPPORT_FOLD']
    finals=[s for s in row['steps'] if s['event']=='FINAL_SUPPORT_FIT']
    check(len(finals)==1 and len(row['steps'])==len(folds)+1,'Unexpected fit step events')
    final=finals[0]
    check(final['eta']==selected['eta'] and final['gamma']==selected['gamma'] and final['train_k']==k
          and final['train_physical_count']==final['gram_dimension']==k*c and final['regularizer']==k*selected['gamma'], 'Final fit mismatch')
    for step in row['steps']:
        check(step['temperature']==1.0 and step['score_scale']==10.0
              and step['training_scale']=='fixed; no data-estimated normalization','Unexpected fitted scale/temperature')
        for name in ('train_squared_error','regularization_loss','train_objective','elapsed_seconds'): nonnegative(step[name],name)
        close(step['train_objective'],step['train_squared_error']+step['regularization_loss'],'training objective')
    nonnegative(final['fit_seconds'],'final fit time')
    candidates=row['candidates']
    if k==1:
        check(selected==dict(eta=.5,gamma=.1) and row['selection']=='k1_preregistered_no_holdout'
              and row['candidate_count']==0 and candidates==[] and folds==[], 'K1 fixed rule mismatch')
        check(all(row['selected_'+name] is None for name in ('objective','macro_nll','old_nll','new_nll')), 'K1 must have no holdout loss')
    else:
        expected=set(product((0.0,.5,1.0),(.01,.1,1.0)))
        check(row['selection']=='physical_support_oof' and row['candidate_count']==len(candidates)==9
              and {(v['eta'],v['gamma']) for v in candidates}==expected,'Incomplete candidate grid')
        check(len(folds)==9*f and {(v['eta'],v['gamma'],v['fold']) for v in folds}
              =={(e,g,i) for e,g in expected for i in range(f)},'Incomplete candidate/fold grid')
        for step in folds:
            held=len(range(step['fold'],k,f));train=k-held
            check(step['train_k']==train and step['heldout_k']==held and step['train_physical_count']==step['gram_dimension']==train*c
                  and step['heldout_physical_count']==held*c and step['regularizer']==train*step['gamma']
                  and step['decomposition_shared_across_gammas'] is True,'Fold actual K/count/regularizer mismatch')
            for key in ('decomposition_seconds','solve_score_seconds'): nonnegative(step[key],key)
            validate_risk(step,c,old_count)
        for candidate in candidates:
            validate_risk(candidate,c,old_count)
            subset=[v for v in folds if v['eta']==candidate['eta'] and v['gamma']==candidate['gamma']]
            for name in ('macro_nll','old_nll','new_nll'):
                if candidate[name] is not None:
                    close(candidate[name],sum(v[name]*v['heldout_k']/k for v in subset), 'OOF weighted '+name)
        chosen=min(candidates,key=lambda v:(round(v['objective'],10),round(v['macro_nll'],10),-v['gamma'],v['eta']))
        check(selected=={name:chosen[name] for name in ('eta','gamma')},'Selected candidate violates fixed selection')
        for name in ('objective','macro_nll','old_nll','new_nll'):
            check(row['selected_'+name]==chosen[name],'Selected loss differs from candidate')


def validate_resources(row,startup,marker,config):
    feature_root=Path(row['output_root'])/'mv_features'
    feature=read_json(feature_root/'features_complete.json')
    provenance=read_json(feature_root/'checkpoint_provenance.json')
    payload=marker['payload_audit']
    check(startup['payload_audit']==payload and startup['config']==dict(algorithm=config['algorithm']), 'Prediction startup mismatch')
    check(feature['status']=='MULTIVIEW_FEATURES_COMPLETE' and feature['schema']=='d92_mvkme_received_blocks_v1'
          and feature['capsule_id']==config['capsule_id'] and feature['checkpoint_sha256']==row['expected_checkpoint_sha256']
          and feature['algorithm']==config['algorithm'] and feature['dtype']=='float32'
          and feature['classes']==config['old_classes'] and feature['model_seed']==row['seeds']['model'], 'Feature marker binding mismatch')
    check(feature['query_used_for_fitting'] is False and feature['source_data_access'] is False and feature['truth_read'] is False
          and feature['encoder_updated'] is False and feature['native_eval'] is True and feature['native_buffers_unchanged'] is True
          and feature['view_count_per_observation']==16 and feature['new_ground_statistics_bytes']==0, 'Feature access/freeze violation')
    check(startup['query_fit_access'] is False and startup['truth_read'] is False and startup['source_data_access'] is False
          and startup['ground_summary_access'] is False and startup['cross_row_adapted_state_reuse'] is False,'Prediction access violation')
    nonnegative(feature['count'],'feature count',integer=True,positive=True)
    check(feature['shape']==[feature['count'],3,256] and feature['feature_array_bytes']==feature['count']*3*256*4,'Cache array bytes mismatch')
    cache=feature_root/'received_mv_features.npz'
    check(feature['feature_file_bytes']==cache.stat().st_size,'Cache disk bytes mismatch')
    nonnegative(feature['model_file_bytes'],'model_file_bytes',integer=True,positive=True)
    check(provenance['checkpoint_sha256']==row['expected_checkpoint_sha256'] and provenance['model_file_bytes']==feature['model_file_bytes'], 'Model package metadata mismatch')
    check(payload['new_ground_statistics_bytes']==0 and payload['model_file_bytes']==feature['model_file_bytes']
          and payload['received_feature_array_bytes']==feature['feature_array_bytes']
          and payload['received_feature_file_bytes']==feature['feature_file_bytes'], 'Payload accounting mismatch')
    nonnegative(feature['feature_seconds'],'feature_seconds')
    feature_rss=feature.get('peak_process_rss_bytes')
    if feature_rss is not None: nonnegative(feature_rss,'feature RSS',integer=True)
    return dict(feature_extraction_seconds=feature['feature_seconds'],physical_observations=feature['count'],
        native_views_per_observation=16,recorded_native_view_count=feature['count']*16,
        received_cache_array_bytes=feature['feature_array_bytes'],received_cache_file_bytes=cache.stat().st_size,
        existing_frozen_model_file_bytes=feature['model_file_bytes'],new_source_payload_bytes=0,
        model_file_bytes_source='existing checkpoint package size recorded by bound exporter; no checkpoint file read by audit',
        new_ground_statistics_bytes=0,ground_summary_used=False,feature_peak_process_rss_bytes=feature_rss,
        feature_peak_rss_unavailable_reason=None if feature_rss is not None else feature.get('peak_process_rss_reason','Exporter did not record process RSS on this platform/version'),
        cache_scope='Received-only local derived cache; not a source transmission payload',
        model_incremental_transfer_bytes=None,model_transfer_unavailable_reason='Deployment state not specified; existing model package bytes are reported separately')


def audit_run(config):
    root=Path(config['root'])
    check(read_json(root/'complete.json')['status']=='SCORED','Collection requires SCORED completion')
    state=read_json(root/'state.json')
    check(set(state)=={r['row_id'] for r in config['rows']} and all(v['status']=='PREDICTIONS_COMPLETE' for v in state.values()),'Incomplete model rows')
    expected_cells=set(product(config['receivers'],config['scenarios'],config['ks'],config['new_counts'],config['support_seeds']))
    check(len(expected_cells)==config['splits'],'Expected matrix duplicate or size mismatch')
    models=[]
    for row in config['rows']:
        folder=Path(row['output_root'])/'mvkme';marker=read_json(folder/'predictions_complete.json')
        check(marker['status']=='PREDICTIONS_COMPLETE' and marker['split_count']==marker['predictions']==config['splits']
              and marker['capsule_id']==config['capsule_id'] and marker['truth_read'] is False
              and marker['source_data_access'] is False and marker['new_ground_statistics_bytes']==0,'Prediction marker mismatch')
        resources=validate_resources(row,read_json(folder/'startup.json'),marker,config)
        records=list(json_lines(folder/'compact.jsonl'));compact={r['split_id']:r for r in records}
        check(len(records)==len(compact)==config['splits'],'Missing or duplicate compact rows')
        seen=set();cells=set();items=[]
        for fit in json_lines(folder/'fit_trace.jsonl'):
            sid=fit['split_id'];check(sid in compact and sid not in seen,'Unbound or duplicate fit trace')
            seen.add(sid);validate_fit(fit,compact[sid],config)
            cell=tuple(fit[key] for key in ('receiver','scenario','k','new_count','support_seed'))
            check(cell in expected_cells and cell not in cells,'Unregistered or duplicate physical split cell');cells.add(cell)
            small=compact[sid]
            items.append(dict(k=fit['k'],classes=fit['classes'],fit_seconds=fit['fit_seconds'],
                fit_call_seconds=small['fit_call_seconds'],
                query_score_seconds=small['query_score_seconds'],prediction_write_seconds=small['prediction_write_seconds'],
                total_seconds=small['total_seconds'],head_bytes=fit['head_bytes'],fourier_matrix_bytes=fit['fourier_matrix_bytes'],
                persistent_state_bytes=fit['persistent_state_bytes'],gram_bytes=fit['gram_bytes'],
                peak_process_rss_bytes=small.get('peak_process_rss_bytes'),selection_key=json.dumps(fit['selected'],sort_keys=True)))
        check(seen==set(compact) and cells==expected_cells,'Incomplete physical fit matrix')
        for item in records:
            check(item['total']==config['splits'] and type(item['completed']) is int,'Compact progress count mismatch')
        check({v['completed'] for v in records}==set(range(1,config['splits']+1)),'Compact progress incomplete')
        rss=marker.get('peak_process_rss_bytes')
        if rss is not None: nonnegative(rss,'prediction RSS',integer=True)
        times={key:stats([v[key] for v in items]) for key in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds')}
        models.append(dict(row_id=row['row_id'],fits=len(items),fixed_k1_fits=sum(v['k']==1 for v in items),
            support_cv_fits=sum(v['k']>1 for v in items),resources=resources,timing=times,peak_process_rss_bytes=rss,
            peak_rss_unavailable_reason=None if rss is not None else 'Predictor did not record process RSS on this platform',
            numeric_state_byte_ranges={key:stats([v[key] for v in items]) for key in ('head_bytes','fourier_matrix_bytes','persistent_state_bytes','gram_bytes')},
            per_k=[dict(k=k,fits=sum(v['k']==k for v in items),folds_per_fit=0 if k==1 else min(k,3),
                candidates_per_fit=0 if k==1 else 9,
                fit_seconds=stats([v['fit_seconds'] for v in items if v['k']==k]),
                selected_parameter_frequencies=dict(Counter(v['selection_key'] for v in items if v['k']==k))) for k in config['ks']],
            trace_file=str(folder/'fit_trace.jsonl'),trace_file_bytes=(folder/'fit_trace.jsonl').stat().st_size,
            compact_file_bytes=(folder/'compact.jsonl').stat().st_size))
    return dict(status='VERIFIED',method='D92-MVKME-v1',models=models,total_fits=sum(v['fits'] for v in models),
        fixed_k1_fits=sum(v['fixed_k1_fits'] for v in models),support_cv_fits=sum(v['support_cv_fits'] for v in models),
        query_rows_used_for_fit=0,source_rows_used_for_fit=0,new_source_payload_bytes=0,ground_summary_used=False,
        raw_traces_preserved=True,parameter_feedback_forbidden=True,selection_frequencies_descriptive_only=True,
        optimizer_convergence_claim=False,scope='Complete physical-support fit/resource metadata only; no target scores, truth, predictions or feature values read',
        time_scope='Extraction, core fit, query score, prediction serialization/write reported separately; total includes overhead; never sum repeated shared decomposition times')


def audit_config(spec):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
    from cvsrffi.stage2_d92_mv_kme import FROZEN_CONFIG
    from run_d92_confirmation import candidate_definition
    check(candidate_definition(spec['confirmation'])['candidate_method']=='D92-MVKME-v1','MVKME only')
    data=spec['data'];conf=spec['confirmation']
    check(conf['candidate']==FROZEN_CONFIG,'Spec differs from frozen MVKME formula')
    count=len(data['target_receivers'])*len(data['scenarios'])*len(data['k'])*len(data['new_class_counts'])*len(data['support_seeds'])
    check(count==conf['expected_split_count']==conf['splits_per_model'],'Spec split count mismatch')
    return dict(root=spec['execution']['remote_run_root'],rows=spec['rows'],splits=count,
        receivers=data['target_receivers'],scenarios=data['scenarios'],ks=data['k'],new_counts=data['new_class_counts'],
        support_seeds=data['support_seeds'],old_classes=conf['old_classes'],algorithm=conf['candidate'],capsule_id=conf['reuse_validated_capsule_id'])


def remote_script(config):
    helpers=(check,finite_tree,read_json,json_lines,nonnegative,close,stats,validate_risk,validate_fit,validate_resources,audit_run)
    return ('import json,math,statistics\nfrom pathlib import Path\nfrom collections import Counter\nfrom itertools import product\n'
            +'\n'.join(inspect.getsource(function) for function in helpers)
            +'\nprint(json.dumps(audit_run('+repr(config)+'),allow_nan=False))\n')


def main():
    from read_d92_run import FLAGS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config=audit_config(read_json(args.spec));script=remote_script(config);compile(script,'remote_audit','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,check=True)
    value=json.loads(result.stdout);finite_tree(value);check(value.get('status')=='VERIFIED','Remote audit incomplete')
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:value for key,value in value.items() if key!='models'},allow_nan=False))


if __name__=='__main__': main()
