"""Read complete SGJoint fit diagnostics, never prediction scores or truth."""
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
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif type(value) in (int, float):
        check(math.isfinite(value), 'Nonfinite fit diagnostic')


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        value=json.load(stream)
    finite_tree(value)
    return value


def json_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            value=json.loads(line)
            finite_tree(value)
            yield value


def validate_payload(payload, ground):
    npz=Path(ground)/'int8_domain_class_center_lowrank_residual_radius_v2.npz'
    manifest=Path(ground)/'manifest.json'
    check(payload['npz_file_bytes']==npz.stat().st_size and payload['manifest_file_bytes']==manifest.stat().st_size,
          'Summary payload disk bytes mismatch')
    check(payload['total_file_bytes']==payload['npz_file_bytes']+payload['manifest_file_bytes'], 'Summary file total mismatch')
    check(payload['total_array_bytes']==payload['numeric_array_bytes']+payload['registry_schema_bytes'], 'Summary array total mismatch')
    check(sum(payload['per_array_bytes'].values())==payload['total_array_bytes'], 'Summary member bytes mismatch')
    check(type(payload['already_deployed']) is bool and payload['incremental_transfer_bytes']==
          (0 if payload['already_deployed'] else payload['total_file_bytes']), 'Summary incremental transfer mismatch')
    check(all(type(payload[k]) is int and payload[k]>=0 for k in ('numeric_array_bytes','registry_schema_bytes',
        'total_array_bytes','npz_file_bytes','manifest_file_bytes','total_file_bytes','incremental_transfer_bytes')),
        'Invalid summary byte count')
    check(payload['source_samples_read'] is False and payload['checkpoint_file_read'] is False
          and payload['dense_bank_persisted'] is False and payload['dequantized_domain_cache'] is False,
          'Summary access boundary violation')
    return dict(npz=str(npz),manifest=str(manifest),**{key:payload[key] for key in
        ('numeric_array_bytes','registry_schema_bytes','total_array_bytes','npz_file_bytes','manifest_file_bytes',
         'total_file_bytes','already_deployed','incremental_transfer_bytes')})


def validate_fit(row, compact, config, payload):
    k,c=row['k'],row['class_count'];old_count=len(config['old_classes'])
    check(row['method']=='D92-SGJoint-v1' and row['algorithm']==config['algorithm'], 'Fit method/formula mismatch')
    check(type(k) is int and k in config['ks'] and c-old_count in config['new_counts'], 'Unregistered fit dimensions')
    check(row['old_classes']==sorted(config['old_classes']) and row['support_rows']==k*c, 'Support/class count mismatch')
    check(row['query_rows_used_for_fit']==0 and row['source_runtime_access'] is False and row['teacher_used'] is False
          and row['encoder_updated'] is False and row['virtual_samples_generated']==0 and row['new_ground_payload_bytes']==0,
          'Source/query fit boundary violation')
    check(row['optimizer_steps']==0 and row['optimizer_status']=='CLOSED_FORM_WITH_SUPPORT_CV'
          and row['loss'] is None and bool(row['loss_reason']), 'Closed-form fit status mismatch')
    dimension=160 if row['selected']['alpha']==0 else 256
    check(row['feature_dim']==dimension and row['head_bytes']==8*c*(dimension+1) and row['summary_operator_bytes']==8*160*160
          and row['persistent_state_bytes']==row['head_bytes']+row['summary_operator_bytes']
          and row['support_matrix_bytes']==8*k*c*256 and row['covariance_matrix_bytes']==8*dimension*dimension,
          'Fit array byte accounting mismatch')
    summary=row['summary']
    check(summary['source_rows_read']==0 and summary['virtual_samples_generated']==0 and summary['dense_bank_persisted'] is False
          and summary['matrix_bytes']==row['summary_operator_bytes'] and summary['payload']==payload,
          'Frozen summary operator accounting mismatch')
    check(row['fit_seconds']>=0, 'Negative fit time')
    for key,value in dict(k=k,classes=c,selected=row['selected'],selection=row['selection'],fit_seconds=row['fit_seconds'],
            persistent_state_bytes=row['persistent_state_bytes'],summary_operator_bytes=row['summary_operator_bytes'],
            fold_count=row['fold_count'],candidate_count=len(row['candidate_trace'])).items():
        check(compact.get(key)==value, 'Compact/trace mismatch: '+key)
    check(compact['query_rows_used_for_fit']==compact['source_rows_used_for_fit']==0
          and compact['learning_rate'] is None and compact['gradient'] is None
          and compact['source_validation'] is None and bool(compact['source_validation_reason'])
          and bool(compact['unavailable_reason']), 'Compact access/status explanation mismatch')
    selected=row['selected']
    check(set(selected)=={'alpha','beta','shrinkage','temperature'} and
          all(type(value) in (int,float) and math.isfinite(value) for value in selected.values()), 'Invalid selected parameters')
    if k==1:
        check(selected==dict(alpha=1.0,beta=1,shrinkage=1.0,temperature=1.0)
              and row['selection']=='fixed_K1' and row['fold_count']==0 and row['folds']==[]
              and row['candidate_trace']==[] and row['selected_oof_risk'] is None, 'K1 fixed rule mismatch')
    else:
        f=min(k,3);expected=set(product((0.0,0.5,1.0,4.0),(0,1),(0.1,1.0)))
        candidates=row['candidate_trace']
        check(row['selection']=='row_physical_support_cross_validation' and row['fold_count']==f
              and len(row['folds'])==f and len(candidates)==16
              and {(v['alpha'],v['beta'],v['shrinkage']) for v in candidates}==expected, 'Incomplete candidate/fold coverage')
        held_ids=[]
        for index,fold in enumerate(row['folds']):
            held_per_class=len(range(index,k,f))
            check(fold['fold']==index and fold['held_rows']==held_per_class*c and fold['train_rows']==(k-held_per_class)*c
                  and len(fold['held_ids'])==fold['held_rows'], 'Physical held-support fold count mismatch')
            held_ids.extend(fold['held_ids'])
        check(len(held_ids)==len(set(held_ids))==k*c, 'Physical support folds overlap or omit rows')
        matches=[]
        for candidate in candidates:
            check(0.1<=candidate['temperature']<=10 and candidate['temperature_evaluations']>0
                  and candidate['elapsed_seconds']>=0 and len(candidate['folds'])==f, 'Invalid candidate temperature/time/folds')
            risk={name:candidate[name] for name in ('objective','macro_nll','old_nll','new_nll')}
            check(all(type(risk[name]) in (int,float) for name in ('objective','macro_nll','old_nll'))
                  and ((risk['new_nll'] is None) if c==old_count else type(risk['new_nll']) in (int,float)), 'Invalid OOF loss')
            for index,fold in enumerate(candidate['folds']):
                held_per_class=len(range(index,k,f))
                check(fold['fold']==index and fold['held_per_class']==held_per_class and fold['train_per_class']==k-held_per_class
                      and fold['held_rows']==held_per_class*c and fold['train_rows']==(k-held_per_class)*c
                      and fold['training_scale']>0 and fold['fit_elapsed_seconds']>=0, 'Candidate physical fold mismatch')
                for phase in ('held_before_temperature','held_after_temperature'):
                    check(all(type(fold[phase][name]) in (int,float) for name in ('objective','macro_nll','old_nll'))
                          and ((fold[phase]['new_nll'] is None) if c==old_count else type(fold[phase]['new_nll']) in (int,float)),
                          'Invalid measured fold NLL')
            if all(candidate[name]==selected[name] for name in selected):
                matches.append(candidate)
        check(len(matches)==1 and row['selected_oof_risk']=={name:matches[0][name] for name in
              ('objective','macro_nll','old_nll','new_nll')}, 'Selected parameters/loss not bound to candidate trace')
    risk=row['selected_oof_risk'] or {}
    for key in ('objective','macro_nll','old_nll','new_nll'):
        check(compact['selected_'+key]==risk.get(key), 'Compact selected loss mismatch')


def stats(values):
    values=[v for v in values if v is not None]
    return dict(min=min(values),mean=statistics.mean(values),max=max(values)) if values else None


def audit_run(config):
    root=Path(config['root'])
    check(read_json(root/'complete.json')['status']=='SCORED', 'Collection requires complete SCORED run')
    state=read_json(root/'state.json')
    check(set(state)=={r['row_id'] for r in config['rows']} and
          all(r['status']=='PREDICTIONS_COMPLETE' for r in state.values()), 'Incomplete model rows')
    models=[]
    for row in config['rows']:
        folder=Path(row['output_root'])/'sgjoint'
        marker=read_json(folder/'predictions_complete.json')
        check(marker['status']=='PREDICTIONS_COMPLETE' and marker['split_count']==marker['predictions']==config['splits']
              and marker['capsule_id']==config['capsule_id'] and marker['truth_read'] is False and marker['source_data_access'] is False,
              'Incomplete candidate prediction marker')
        startup=read_json(folder/'startup.json')
        process_rss=marker.get('peak_process_rss_bytes')
        check(process_rss is None or (type(process_rss) is int and process_rss>=0), 'Invalid recorded process RSS')
        payload=marker['payload_audit']
        check(startup['payload_audit']==payload and startup['config']['algorithm']==config['algorithm'], 'Startup payload/formula mismatch')
        payload_files=validate_payload(payload,Path(row.get('reuse_row_root',row['output_root']))/'ground')
        compact_rows=list(json_lines(folder/'compact.jsonl'))
        compact={r['split_id']:r for r in compact_rows}
        check(len(compact_rows)==len(compact)==config['splits'], 'Incomplete/duplicate compact rows')
        seen=set();counts=Counter();measurements={k:[] for k in config['ks']};state_bytes={}
        for fit in json_lines(folder/'fit_trace.jsonl'):
            sid=fit['split_id']
            check(sid in compact and sid not in seen, 'Duplicate or unbound fit trace row')
            seen.add(sid);validate_fit(fit,compact[sid],config,payload)
            k,c=fit['k'],fit['class_count'];counts[(k,c-len(config['old_classes']))]+=1
            selected=fit['selected'];risk=fit['selected_oof_risk'] or {}
            measurements[k].append(dict(fit_seconds=fit['fit_seconds'],temperature=selected['temperature'],
                support_matrix_bytes=fit['support_matrix_bytes'],covariance_matrix_bytes=fit['covariance_matrix_bytes'],
                selection_key=json.dumps({name:selected[name] for name in ('alpha','beta','shrinkage')},sort_keys=True),
                **{name:risk.get(name) for name in ('objective','old_nll','new_nll','macro_nll')}))
            byte_key=f"classes={c},features={fit['feature_dim']}"
            byte_record=state_bytes.setdefault(byte_key,dict(fits=0,class_count=c,feature_dim=fit['feature_dim'],
                head_bytes=fit['head_bytes'],summary_operator_bytes=fit['summary_operator_bytes'],
                persistent_state_bytes=fit['persistent_state_bytes'],covariance_matrix_bytes=fit['covariance_matrix_bytes']))
            byte_record['fits']+=1
        expected={(k,n):config['cells_per_k_new'] for k in config['ks'] for n in config['new_counts']}
        check(seen==set(compact) and len(seen)==config['splits'] and dict(counts)==expected, 'Incomplete K/new-class fit matrix')
        groups=[]
        for k,items in measurements.items():
            groups.append(dict(k=k,fits=len(items),fixed_k1_fits=len(items) if k==1 else 0,
                support_cv_fits=len(items) if k>1 else 0,
                candidates_per_fit=0 if k==1 else 16,folds_per_fit=0 if k==1 else min(k,3),
                fit_seconds=stats([r['fit_seconds'] for r in items]),temperature=stats([r['temperature'] for r in items]),
                support_matrix_bytes=stats([r['support_matrix_bytes'] for r in items]),
                covariance_matrix_bytes=stats([r['covariance_matrix_bytes'] for r in items]),
                selected_oof_loss={name:stats([r[name] for r in items]) for name in ('objective','old_nll','new_nll','macro_nll')},
                selected_parameter_frequencies=dict(Counter(r['selection_key'] for r in items))))
        fit_times=[r['fit_seconds'] for items in measurements.values() for r in items]
        models.append(dict(row_id=row['row_id'],fits=len(seen),fixed_k1_fits=sum(g['fixed_k1_fits'] for g in groups),
            support_cv_fits=sum(g['support_cv_fits'] for g in groups),
            total_fit_seconds=sum(fit_times),mean_fit_seconds=statistics.mean(fit_times),max_fit_seconds=max(fit_times),per_k=groups,
            peak_process_rss_bytes=process_rss,
            peak_process_rss_source='predictions_complete.json' if process_rss is not None else None,
            query_rows_used_for_fit=0,source_rows_read=0,
            state_bytes_by_classes=state_bytes,payload_files=payload_files,
            trace_file=str(folder/'fit_trace.jsonl'),trace_file_bytes=(folder/'fit_trace.jsonl').stat().st_size,
            compact_file_bytes=(folder/'compact.jsonl').stat().st_size))
    return dict(status='VERIFIED',method='D92-SGJoint-v1',models=models,total_fits=sum(m['fits'] for m in models),
        fixed_k1_fits=sum(m['fixed_k1_fits'] for m in models),support_cv_fits=sum(m['support_cv_fits'] for m in models),
        process_peak_rss_bytes=stats([m['peak_process_rss_bytes'] for m in models]),
        payload_byte_ranges={key:stats([m['payload_files'][key] for m in models]) for key in
            ('numeric_array_bytes','registry_schema_bytes','total_array_bytes','npz_file_bytes','manifest_file_bytes',
             'total_file_bytes','incremental_transfer_bytes')},
        query_rows_used_for_fit=0,source_rows_read=0,new_ground_statistics_bytes=0,
        scope='Complete closed-form fit and held-support CV diagnostics; no prediction scores, query truth or source samples read',
        optimizer_convergence_claim=False,selection_frequencies_descriptive_only=True,parameter_feedback_forbidden=True,
        raw_traces_preserved=True)


def audit_config(spec):
    from run_d92_confirmation import candidate_definition
    check(candidate_definition(spec['confirmation'])['candidate_method']=='D92-SGJoint-v1', 'SGJoint only')
    data=spec['data'];confirmation=spec['confirmation']
    cells=len(data['target_receivers'])*len(data['scenarios'])*len(data['support_seeds'])
    splits=cells*len(data['k'])*len(data['new_class_counts'])
    check(splits==confirmation['expected_split_count']==confirmation['splits_per_model'], 'Spec split count mismatch')
    return dict(root=spec['execution']['remote_run_root'],rows=spec['rows'],splits=splits,cells_per_k_new=cells,
        ks=data['k'],new_counts=data['new_class_counts'],old_classes=confirmation['old_classes'],
        algorithm=confirmation['candidate'],capsule_id=confirmation['reuse_validated_capsule_id'])


def remote_script(config):
    helpers=(check,finite_tree,read_json,json_lines,validate_payload,validate_fit,stats,audit_run)
    return ('import json,math,statistics\nfrom pathlib import Path\nfrom collections import Counter\nfrom itertools import product\n'
            +'\n'.join(inspect.getsource(function) for function in helpers)
            +'\nprint(json.dumps(audit_run('+repr(config)+'),allow_nan=False))\n')


def main():
    from read_d92_run import FLAGS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    config=audit_config(read_json(args.spec));script=remote_script(config);compile(script,'remote_audit','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,check=True)
    value=json.loads(result.stdout);finite_tree(value)
    check(value.get('status')=='VERIFIED','Remote audit did not complete')
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:value for key,value in value.items() if key!='models'}))


if __name__=='__main__':main()
