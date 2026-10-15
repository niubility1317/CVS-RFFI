"""Standard-library, read-only completed-row audit; no IQ, checkpoint or truth reads."""
import gzip,json,math,re
from pathlib import Path

VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def parse_steps(path,epochs=50,steps_per_epoch=222):
    count=0;sums={};clipping=[];epoch_counts=[0]*epochs
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            v=json.loads(line);count+=1;epoch=v['epoch'];metrics=v['metrics']
            if v['step']!=count or epoch!=(count-1)//steps_per_epoch+1 or not 1<=epoch<=epochs:
                raise ValueError('Nonconsecutive actual native step/epoch')
            epoch_counts[epoch-1]+=1
            for key,number in metrics.items():
                if isinstance(number,(int,float)) and not math.isfinite(number):
                    raise ValueError('Nonfinite numeric training metric: '+key)
                if (key.startswith('urban/') or key in ('train/loss','train/grad_before_clip','train/grad_total','train/grad_clip_limit','train/grad_clip_active')) and isinstance(number,(int,float)):
                    array=sums.setdefault(key,[[0.,0] for _ in range(epochs)]);array[epoch-1][0]+=number;array[epoch-1][1]+=1
            if any(metrics.get(k,0) for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad','train/pseudo_truth_available')):
                raise ValueError('Skipped/nonfinite update or hidden U truth exposure')
            if metrics.get('train/optimizer_step_applied')!=1:raise ValueError('Missing applied native optimizer update')
            if metrics.get('train/grad_clip_active'):
                clipping.append((metrics['train/grad_before_clip'],metrics['train/grad_total'],metrics['train/grad_clip_limit']))
    if count!=epochs*steps_per_epoch or epoch_counts!=[steps_per_epoch]*epochs:
        raise ValueError('Incomplete full training log')
    return dict(complete_steps=count,epoch_step_counts=epoch_counts,
        curves={k:[s/n if n else None for s,n in array] for k,array in sums.items()},
        clipping=dict(active_steps=len(clipping),clipped_steps=sum(a>limit for a,b,limit in clipping),
            max_before=max((a for a,b,c in clipping),default=None),max_after=max((b for a,b,c in clipping),default=None),
            limits=sorted({c for a,b,c in clipping})))


def audit_row(row):
    source=Path(row['source']);root=source.parent
    if not (root/'complete.json').is_file():return None
    done=read(source/'completion.json');frozen=read(source/'source_frozen.json');complete=read(root/'complete.json')
    if (complete['status']!='TRAINED_AND_SCORED' or done['config']!=row['config'] or frozen['config']!=row['config']
            or done['epoch']!=50 or done['optimizer_steps']!=11100 or done['logged_steps']!=11100 or done['target_access']
            or done['head_fit_updates'] or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors']
            or frozen['selection']!='final_only' or frozen['target_access']):
        raise ValueError('Completed row source/default-head/final50 lineage differs')
    pred=read(root/'prediction/complete.json');score=read(root/'scoring/complete.json')
    if (pred['status']!='PREDICTIONS_COMPLETE' or pred['count']!=168000 or pred['views']!=VIEWS or pred['truth_read']
            or pred['query_fit'] or not pred['all_parameters_frozen'] or pred['head_fit_updates']
            or score['status']!='SCORED_COMPLETE' or score['views']!=VIEWS or score['predictions']!=1176000
            or score['metric_records']!=126 or score['independent_recount']!='VERIFIED' or not score['truth_last']):
        raise ValueError('Seven-view prediction/truth-last completion metadata differs')
    parsed=parse_steps(source/'step_metrics.jsonl.gz')
    stdout=Path(row['log']).read_text(errors='replace')
    errors=[v for v in stdout.splitlines() if any(k in v for k in ('Traceback','Error:','CUDA out of memory','Killed'))]
    if errors:raise ValueError('Full stdout contains technical errors: '+repr(errors))
    warnings=[v for v in stdout.splitlines() if any(k in v for k in ('[WARN','Warning:','UserWarning','RuntimeWarning'))]
    mentions=[v for v in stdout.splitlines() if re.search(r'\b(?:nan|[+-]?inf|infinity)\b',v,re.IGNORECASE)]
    lifecycle=[v for v in stdout.splitlines() if re.search(r'\b(?:resume|resumed|rollback|early.stop)\b',v,re.IGNORECASE)]
    native=[json.loads(v) for v in (source/'epoch_metrics.jsonl').read_text().splitlines()]
    execution=[json.loads(v) for v in (source/'execution_epoch.jsonl').read_text().splitlines()]
    args=read(source/'resolved_native_args.json');args=args.get('args',args)
    if (len(native)!=50 or len(execution)!=50 or [int(v['epoch']) for v in native]!=list(range(1,51))
            or not all(v['val_tx_total']==27000 and v['stage_source_val_heavy_eval_ran']==1 for v in native)
            or not all(v['cache_hits']==106 and v['cache_misses']==0 and v['cache_allowed'] for v in execution)):
        raise ValueError('Complete native V/epoch execution evidence differs')
    if (not args['use_aug'] or not args['use_unlabeled'] or not args['use_ema_teacher'] or args['label_epochs']!=33 or args['pseudo_epochs']!=17
            or args['eval_max_batches']!=0 or args['eval_sat_channel'] or args['direct_metric_multiview_separate'] or args['phase2_export_prototypes']
            or done['final_source_metrics']['source_val_count']!=27000 or done['native_disabled_probe_calls']!=1):
        raise ValueError('Actual native generic/old-U/default-head/source-only recipe differs')
    inner=read(source/'inner_source_contract.json');mechanism=read(source/'training_mechanism_complete.json')
    if (inner['counts']!=dict(L_train=5580,L_dev=720,U_s=56700,V=27000) or inner['fold_seed']!=2026101503 or inner['L_dev_updates']
            or mechanism['status']!='VERIFIED' or mechanism['updates']!=11100 or mechanism['head_fit_updates'] or mechanism['U_truth_read']
            or mechanism['source_dev_forward_packets']!=70560):
        raise ValueError('Actual source fold/update/forward budget differs')
    counts=mechanism['counts'];augmented=row['arm']!='D0'
    if (counts['clean_L_packets']!=1420800 or counts['teacher_L_clean_packets']!=1420800 or counts['native_U_packets']!=963900
            or counts.get('satellite_L_packets',0)!=(1420800 if augmented else 0)):
        raise ValueError('Main view/teacher/native U sample budget differs')
    main=sum(counts.get(k,0) for k in ('clean_L_packets','satellite_L_packets','teacher_L_clean_packets'))+2*counts['native_U_packets']
    if main!=(6190200 if augmented else 4769400):raise ValueError('Declared actual main forwards differ')
    gradients=read(source/'gradient_gate_history.json')['records'];updates=read(source/'parameter_update_history.json')['records']
    if (not all(all(v<=1e-8 for v in r['auxiliary_boundary_norms'].values()) for r in gradients)
            or len(updates)!=50 or not all(sum(v['groups'][key]['update_norm'] for v in updates)>0 for key in updates[0]['groups'])):
        raise ValueError('Auxiliary W/gain boundary or actual original group updates differ')
    learning=read(source/'source_learning_progress.json')
    if [r['epoch'] for r in learning['records']]!=[1,5,6,10,15,20,25,30,33,34,35,40,45,50]:
        raise ValueError('Incomplete source learning observations')
    coverage=read(source/'training_coverage.json')
    if augmented:
        if ({(v['TX'],v['RX'],v['state']) for v in coverage['records'] if v['count']>0}
                !={(tx,rx,s) for tx in range(6) for rx in (1,3,4,6,8) for s in ('LOS','PARTIAL','BLOCKED')}
                or {v['scene'] for v in coverage['records']}!={'practical_low_suburban','practical_mid_urban','practical_low_urban'}):
            raise ValueError('Missing declared TX/sourceRX/state/scene training support')
    elif coverage['records']:raise ValueError('Clean-only D0 unexpectedly generated satellite views')
    clean_coverage=read(source/'clean_training_coverage.json')
    if clean_coverage['unique_samples']!=5580:raise ValueError('Incomplete legal L physical coverage')
    return dict(arm=row['arm'],row_id=row['row_id'],actual_source=str(source),reused=False,status='VERIFIED_FULL_ROW',
        **parsed,full_stdout_scanned=True,technical_errors=errors,
        stdout_diagnostics=dict(warning_lines=warnings,nonfinite_mention_count=len(mentions),nonfinite_examples=mentions[:20],lifecycle_lines=lifecycle,
            interpretation='All numeric step metrics and skip flags checked; inactive text diagnostic nan is recorded separately'),
        completion=done,prediction=pred,scoring=score,native_epochs=native,execution_epochs=execution,
        gradients=dict(records=gradients),parameter_updates=dict(records=updates),learning=learning,coverage=coverage,
        clean_coverage=clean_coverage,mechanism=mechanism,
        forward_counts=dict(main_training=main,source_panel=70560,native_V=52*27000,formal_prediction=1176000,
            initial_smoke=2,final_checkpoint_smoke=2,total_row=main+70560+52*27000+1176000+4))


def audit(plan):
    completion_path=Path(plan['base'])/'complete.json'
    if not completion_path.is_file():
        return dict(status='WAITING_FOR_MATRIX',rows=[],new_updates_parsed=0,completion=None,
            reads_IQ_or_checkpoints=False,reads_target_scores=False,remote_mutations=False)
    rows=[]
    for row in plan['rows']:
        result=audit_row(row)
        if result is not None:rows.append(result)
    completion=read(completion_path)
    if completion and (completion['status']!='ANALYZED' or completion['actual_model_count']!=3 or len(rows)!=3):
        raise ValueError('Three-new-row matrix completion differs')
    return dict(status='COMPLETE_NEW_LOG_AUDIT' if len(rows)==3 else 'PARTIAL_NEW_LOG_AUDIT',rows=rows,
        new_updates_parsed=sum(r['complete_steps'] for r in rows),completion=completion,
        reads_IQ_or_checkpoints=False,reads_target_scores=False,remote_mutations=False)
