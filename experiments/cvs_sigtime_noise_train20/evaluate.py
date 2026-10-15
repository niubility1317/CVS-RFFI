import gc,os,subprocess,sys,math
from pathlib import Path
from . import design as d
from .provenance import source_proof,own_model
from experiments.cvs_feature_sequence50 import predict as shared_predict

def predict(row):return shared_predict.predict(row,design=d,proof=source_proof,load=own_model)
def preflight(row):return shared_predict.preflight(row,design=d,proof=source_proof)

def source_health(c):
    from experiments.cvs_evidence_responsibility_train20.provenance import source_proof as baseline_proof
    source_proof(c);done=d.read(Path(c['output_root'])/'completion.json')
    reference=d.read(baseline_proof(d.parent_config(c['model_seed']))/'completion.json')
    now,old=done['final_source_metrics'],reference['final_source_metrics']
    if now['source_val_count']!=27000 or old['source_val_count']!=27000:raise ValueError('Full original five-RX source V required')
    lce=done['final_training_L_CE']
    if not math.isfinite(lce):raise ValueError('Finite final labeled training CE required')
    passed=(lce<=d.SOURCE_GATE['final_training_L_CE_max']
        and now['source_val_accuracy']>=old['source_val_accuracy']-d.SOURCE_GATE['first_seed_source_V_drop_max']
        and now['source_val_worst_rx']>=old['source_val_worst_rx']-d.SOURCE_GATE['first_seed_worst_source_RX_drop_max'])
    return dict(qualified=passed,rule=d.SOURCE_GATE,source_metrics=now,baseline_source_metrics=old,
        final_training_L_CE=lce,target_access=False)

def score(row):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh independent CPU truth-side scorer required')
    c,ids=preflight(row)
    import numpy as np
    from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
    out=d.BASE/row/'scores'
    if out.exists():raise FileExistsError('Preserve existing row scores')
    y,rx,days=truth_arrays(d.read(d.TRUTH),ids)
    with np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as data:decisions=data['NATIVE_HEAD'].copy()
    records,day_records=metric_rows(c,decisions,ids,y,rx,days);records+=day_records
    if len(records)!=18:raise ValueError('All eighteen same-row clean metric strata required')
    for record in records:record['readout']='NATIVE_HEAD'
    out.mkdir();np.savez_compressed(out/'correctness.npz',ids=ids,NATIVE_HEAD=decisions==y)
    d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
    d.write(d.BASE/row/'completion.json',dict(status='MODEL_PREDICTED_SCORED',config=c,count=168000,metric_records=18,
        truth_last=True,new_training_steps=4440,head_fit_updates=0))
    print('SCORED '+row,flush=True)

def aggregate(records):
    from statistics import mean,stdev
    overall=[row for row in records if row['dimension']=='overall']
    if len(overall)!=3 or {row['model_seed'] for row in overall}!=set(d.SEEDS):raise ValueError('Three actual fixed model seeds required')
    detail=[]
    for row in overall:
        group=[r for r in records if r['row_id']==row['row_id']]
        detail.append(dict(model_seed=row['model_seed'],accuracy=row['accuracy'],macro_f1=row['macro_f1'],
            errors=row['query_count']-sum(row['confusion'][k][k] for k in range(6)),
            worst_RX_accuracy=min(r['accuracy'] for r in group if r['dimension']=='receiver'),
            worst_day_accuracy=min(r['accuracy'] for r in group if r['dimension']=='day')))
    keys=('accuracy','macro_f1','worst_RX_accuracy','worst_day_accuracy')
    return dict(seed_count=3,seeds=sorted(d.SEEDS),per_seed=detail,
        **{key+'_mean':mean(r[key] for r in detail) for key in keys},
        accuracy_sample_SD=stdev(r['accuracy'] for r in detail))

def finish():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CPU process required for final truth-side aggregation')
    import numpy as np
    from statistics import mean
    records=[];baseline=[];pairs=[];mode_pairs=[]
    for seed in d.SEEDS:
        oldpath=Path(d.parent_config(seed)['output_root']).parent
        baseline+=d.read(oldpath/'scores/metrics.json')['records']
    for row in d.rows():
        c=d.config(row);preflight(c['row_id']);done=d.read(d.BASE/c['row_id']/'completion.json')
        if done['status']!='MODEL_PREDICTED_SCORED' or done['count']!=168000 or done['new_training_steps']!=4440:raise ValueError('Complete fixed three-seed trained matrix required')
        records+=d.read(d.BASE/c['row_id']/'scores/metrics.json')['records']
        oldpath=Path(d.parent_config(c['model_seed'])['output_root']).parent
        with np.load(oldpath/'scores/correctness.npz',allow_pickle=False) as before,np.load(d.BASE/c['row_id']/'scores/correctness.npz',allow_pickle=False) as after:
            if not np.array_equal(before['ids'],after['ids']):raise ValueError('Paired original baseline/new model physical IDs differ')
            a,b=before['NATIVE_HEAD'],after['NATIVE_HEAD'];repair=int((~a&b).sum());broken=int((a&~b).sum())
            pairs.append(dict(arm=c['arm'],model_seed=c['model_seed'],corrected_errors=repair,introduced_errors=broken,
                net_correct_change=repair-broken,accuracy_difference=(repair-broken)/168000))
    if len(records)!=len(d.rows())*18 or len(baseline)!=54:raise ValueError('Complete registered new and three-seed reused baseline metrics required')
    current={arm:aggregate([row for row in records if row['arm']==arm]) for arm in d.ARMS};original=aggregate(baseline)
    if 'CLEAN_SIG' in d.ARMS:
        for seed in d.SEEDS:
            with np.load(d.BASE/('NONE-s'+str(seed))/'scores/correctness.npz',allow_pickle=False) as before,np.load(d.BASE/('CLEAN_SIG-s'+str(seed))/'scores/correctness.npz',allow_pickle=False) as after:
                if not np.array_equal(before['ids'],after['ids']):raise ValueError('Matched CLEAN_SIG/NONE physical IDs differ')
                a,b=before['NATIVE_HEAD'],after['NATIVE_HEAD'];repair=int((~a&b).sum());broken=int((a&~b).sum())
                mode_pairs.append(dict(comparison='CLEAN_SIG minus NONE',model_seed=seed,corrected_errors=repair,introduced_errors=broken,
                    net_correct_change=repair-broken,accuracy_difference=(repair-broken)/168000))
    primary='CLEAN_SIG' if 'CLEAN_SIG' in d.ARMS else 'NONE'
    result=dict(status='ARTIFACTS_COMPLETE',arms=list(d.ARMS),source_fixed_primary=primary,records=records,reused_baseline_records=baseline,
        summary=current,baseline_summary=original,paired_errors_vs_baseline=pairs,paired_CLEAN_SIG_minus_NONE=mode_pairs,
        accuracy_difference_mean_vs_baseline={arm:mean(p['accuracy_difference'] for p in pairs if p['arm']==arm) for arm in d.ARMS},
        positive_seeds_vs_baseline={arm:sum(p['accuracy_difference']>0 for p in pairs if p['arm']==arm) for arm in d.ARMS},
        metric_differences_vs_baseline={arm:{key:value[key]-original[key] for key in ('accuracy_mean','macro_f1_mean','worst_RX_accuracy_mean','worst_day_accuracy_mean')} for arm,value in current.items()},
        new_backbone_epochs=len(d.rows())*20,new_training_steps=len(d.rows())*4440,reused_baseline_epochs=60,reused_baseline_training_steps=13320,
        new_prediction_decisions=len(d.rows())*168000,reused_baseline_decisions=504000,head_fit_updates=0,truth_last=True,
        goal84_numerical_check=current[primary]['accuracy_mean']>=.84,goal_completed=False,
        limitation='Default jointly trained head, actual deployed joint160; no newTX/satellite/Phase2 claim, no target feedback or independent-blind-test claim')
    d.write(d.BASE/'result.json',result);print('NOISE_TRAINING_MATRIX_COMPLETE',flush=True)

def run(request_path,runner_path):
    request=d.read(request_path)
    if request['rows']!=[d.config(row) for row in d.rows()] or request['arms']!=list(d.ARMS):raise ValueError('Fixed source-selected three-seed route/control matrix required')
    decision=d.source_selection(d.read(d.SOURCE_DECISION))
    if d.BASE.exists():raise FileExistsError('Reconcile existing run instead of resubmitting')
    d.BASE.mkdir(parents=True);d.write(d.BASE/'source_selection.json',decision)
    from .source import train
    import torch
    first=request['rows'][:len(d.ARMS)];later=request['rows'][len(d.ARMS):];health_rows=[]
    def predict_and_score(c):
        predict(c['row_id']);gc.collect();torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-u',str(runner_path),str(request_path),'--score',c['row_id']],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    for c in first:
        train(c);gc.collect();torch.cuda.empty_cache()
        health=source_health(c);d.write(Path(c['output_root'])/'formal_source_health.json',health)
        health_rows.append(dict(row_id=c['row_id'],**health))
    d.write(d.BASE/'first_seed_source_qualification.json',dict(qualified=all(row['qualified'] for row in health_rows),rows=health_rows,target_access=False))
    if not all(row['qualified'] for row in health_rows):
        d.write(d.BASE/'result.json',dict(status='SOURCE_FALSIFIED',rows_trained=len(first),new_training_steps=len(first)*4440,target_access=False,goal_completed=False));return
    for c in first:predict_and_score(c)
    for c in later:
        train(c);gc.collect();torch.cuda.empty_cache()
        d.write(Path(c['output_root'])/'formal_source_health.json',source_health(c))
        predict_and_score(c)
    subprocess.run([sys.executable,'-u',str(runner_path),str(request_path),'--finish'],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
