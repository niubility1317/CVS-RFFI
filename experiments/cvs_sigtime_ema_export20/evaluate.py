import gc,os,subprocess,sys
from pathlib import Path
from . import design as d
from .protocol import parent_source,frozen_proof,load_parent,own_model,freeze
from experiments.cvs_feature_sequence50 import predict as shared_predict

def source_compare(c):
    import torch
    from torch.nn import functional as F
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    d.validate(c);source=parent_source(c);out=d.BASE/c['row_id']/'source'
    if out.exists():raise FileExistsError('Preserve existing source comparison')
    out.mkdir(parents=True);old=d.parent_config(c['model_seed']);args=d.parent.make_args(old,'cuda:0')
    args.output_dir=str(out);args.run_id=d.RUN;args.candidate_id=c['row_id']
    expected=d.read(d.SOURCE);torch.set_num_threads(2)
    with numerical_context(d.FULL_FP32_POLICY):
        with installed(old) as native:ctx=native._build_ssdg_wisig_data(args,torch.device('cuda:0'))
        if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Original all-five-RX source roles only')
        models=dict(FINAL=load_parent(c,'cuda:0','model',False),EMA=load_parent(c,'cuda:0','ema_model',False))
        with torch.no_grad():
            z=models['EMA'].forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
            if z['tx_logits'].shape!=(2,6) or z['id_feat_joint'].shape!=(2,160) or not torch.isfinite(z['tx_logits']).all():raise ValueError('Own EMA real-checkpoint no-query smoke failed')
        print('SMOKE PASS own all-source final20 EMA; query_read=false',flush=True)
        sums={key:dict(count=0,errors=0,CE_sum=0.,RX={}) for key in models}
        with torch.no_grad(),fresh_parameter_caches(models['FINAL']),fresh_parameter_caches(models['EMA']):
            for x,y,domain,meta in ctx['val_loader']:
                x=x.to('cuda:0');y=y.to('cuda:0')
                for key,model in models.items():
                    logits=model.forward_identity_only(x)['tx_logits'].double()
                    if logits.shape!=(len(y),6) or not torch.isfinite(logits).all():raise FloatingPointError('Finite original six-class source logits required')
                    losses=F.cross_entropy(logits,y,reduction='none');correct=logits.argmax(1)==y;value=sums[key]
                    value['count']+=len(y);value['errors']+=int((~correct).sum());value['CE_sum']+=float(losses.sum())
                    for receiver in meta['rx_i'].unique().tolist():
                        mask=(meta['rx_i']==receiver).to('cuda:0');group=value['RX'].setdefault(str(receiver),[0,0])
                        group[0]+=int(correct[mask].sum());group[1]+=int(mask.sum())
        metrics={key:dict(count=value['count'],errors=value['errors'],accuracy=1-value['errors']/value['count'],mean_CE=value['CE_sum']/value['count'],
            receiver_accuracy={receiver:hit/count for receiver,(hit,count) in value['RX'].items()}) for key,value in sums.items()}
        if any(row['count']!=27000 or set(row['receiver_accuracy'])!=set(map(str,(1,3,4,6,8))) for row in metrics.values()):raise ValueError('Full source V shape differs')
        result=dict(status='SOURCE_COMPARE_COMPLETE',config=c,metrics=metrics,source_gate_if_first_seed=d.qualify(metrics['FINAL'],metrics['EMA']),
            source_unique_IQ_reads=27000,source_forward_packets=54002,new_optimizer_steps=0,head_fit_updates=0,target_access=False,U_truth_read=False,
            parent_checkpoint=str(source/'final_ssdg.pth'),state_key='ema_model',effective_source_roles=dict(L_s=6300,U_s=56700,V=27000))
        d.write(out/'source_compare.json',result)
        del models,ctx;gc.collect();torch.cuda.empty_cache()
        return result

def predict(row):return shared_predict.predict(row,design=d,proof=frozen_proof,load=own_model)
def preflight(row):return shared_predict.preflight(row,design=d,proof=frozen_proof)

def score(row):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh independent CPU truth-side scorer required')
    c,ids=preflight(row)
    import numpy as np
    from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
    out=d.BASE/row/'scores'
    if out.exists():raise FileExistsError('Preserve existing scores')
    y,rx,days=truth_arrays(d.read(d.TRUTH),ids)
    with np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as data:decisions=data['NATIVE_HEAD'].copy()
    records,day_records=metric_rows(c,decisions,ids,y,rx,days);records+=day_records
    if len(records)!=18:raise ValueError('All eighteen same-row clean metric strata required')
    for record in records:record['readout']='NATIVE_HEAD'
    out.mkdir();np.savez_compressed(out/'correctness.npz',ids=ids,NATIVE_HEAD=decisions==y)
    d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
    d.write(d.BASE/row/'completion.json',dict(status='OWN_EMA_PREDICTED_SCORED',config=c,count=168000,metric_records=18,truth_last=True,new_training_steps=0))
    print('SCORED '+row,flush=True)

def finish():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CPU process required for the final truth-side aggregation')
    import numpy as np
    from statistics import mean,stdev
    records=[];baseline=[];pairs=[]
    for row in d.rows():
        c=d.config(row);preflight(c['row_id']);done=d.read(d.BASE/c['row_id']/'completion.json')
        if done['status']!='OWN_EMA_PREDICTED_SCORED' or done['count']!=168000:raise ValueError('Complete fixed three-seed EMA matrix required')
        new=d.read(d.BASE/c['row_id']/'scores/metrics.json')['records'];oldpath=Path(d.parent_config(c['model_seed'])['output_root']).parent
        old=d.read(oldpath/'scores/metrics.json')['records'];records+=new;baseline+=old
        with np.load(oldpath/'scores/correctness.npz',allow_pickle=False) as before,np.load(d.BASE/c['row_id']/'scores/correctness.npz',allow_pickle=False) as after:
            if not np.array_equal(before['ids'],after['ids']):raise ValueError('Paired original student/own EMA physical IDs differ')
            a,b=before['NATIVE_HEAD'],after['NATIVE_HEAD'];repair=int((~a&b).sum());broken=int((a&~b).sum())
            pairs.append(dict(model_seed=c['model_seed'],corrected_errors=repair,introduced_errors=broken,net_correct_change=repair-broken,accuracy_difference=(repair-broken)/168000))
    if len(records)!=54 or len(baseline)!=54:raise ValueError('Complete3x18 own EMA and3x18 reused student metrics required')
    values=[row['accuracy'] for row in records if row['dimension']=='overall'];original=[row['accuracy'] for row in baseline if row['dimension']=='overall']
    if len(values)!=3 or len(original)!=3:raise ValueError('Three actual model seeds required')
    result=dict(status='ARTIFACTS_COMPLETE',records=records,reused_student_records=baseline,paired_errors=pairs,
        EMA_accuracy_mean=mean(values),EMA_accuracy_sample_SD=stdev(values),student_accuracy_mean=mean(original),
        accuracy_difference_mean=mean(p['accuracy_difference'] for p in pairs),positive_seeds=sum(p['accuracy_difference']>0 for p in pairs),
        new_prediction_decisions=504000,reused_student_decisions=504000,new_training_steps=0,truth_last=True,goal84_numerical_check=mean(values)>=.84,
        limitation='Performance export of an existing generic EMA; not a new training mechanism, unseenTX result or proof that the entire active goal is achieved')
    d.write(d.BASE/'result.json',result);print('EMA_EXPORT_MATRIX_COMPLETE',flush=True)

def run(request_path,runner_path):
    request=d.read(request_path)
    if request['rows']!=[d.config(row) for row in d.rows()]:raise ValueError('Fixed three own EMA exports required')
    if d.BASE.exists():raise FileExistsError('Reconcile existing run rather than resubmit')
    d.BASE.mkdir(parents=True);first=source_compare(request['rows'][0])
    qualified=first['source_gate_if_first_seed'];d.write(d.BASE/'source_qualification.json',dict(qualified=qualified,rule=d.GATE,target_access=False,first_seed=first))
    if not qualified:
        d.write(d.BASE/'result.json',dict(status='SOURCE_FALSIFIED',new_training_steps=0,target_access=False,goal84_completed=False));return
    for c in request['rows']:freeze(c)
    for index,c in enumerate(request['rows']):
        if index:source_compare(c)
        predict(c['row_id'])
        subprocess.run([sys.executable,'-u',str(runner_path),str(request_path),'--score',c['row_id']],check=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    subprocess.run([sys.executable,'-u',str(runner_path),str(request_path),'--finish'],check=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
