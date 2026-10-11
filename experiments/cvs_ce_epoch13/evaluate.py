"""Frozen-row clean prediction; separate truth-last scoring process."""
import argparse,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from . import design as d
from experiments.cvs_report_ssl import evaluate as shared

COUNT=168000
shared.d=d


def source_provenance(c):
    d.validate(c);source=Path(c['output_root'])
    done=d.read(source/'completion.json');init=d.read(source/'initialization.json')
    if any(done.get(k)!=v for k,v in dict(status='SOURCE_TRAINED',epoch=c['epochs'],logged_steps=c['updates'],optimizer_steps=c['updates'],target_access=False,target_evaluated=False,config=c,checkpoint_sources=[]).items()):raise ValueError('Source identity/budget differs')
    if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact']:raise ValueError('Scratch source provenance differs')
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
    return source


def checkpoint_provenance(c,ck):
    from experiments.cvs_feature_disentangle.source import clean
    from scripts.train_daot_rc4_baseline import resolved_config
    if any(ck.get(k)!=v for k,v in dict(epoch=c['epochs'],candidate_id=c['row_id'],run_id=c['run_id'],checkpoint_selection='final_only',a1_ema_successful_updates=c['updates']).items()):raise ValueError('Checkpoint identity/steps differ')
    a=ck['args']
    if a['baseline_ckpt'] or a['teacher_ckpt'] or not a['from_scratch'] or a['epochs']!=c['epochs'] or a['label_epochs']!=c['epochs'] or a['pseudo_epochs']!=0:raise ValueError('Actual scratch budget differs')
    if a['use_unlabeled'] or a['use_ema_teacher'] or a['use_concat_sat_channel_aug'] or a['label_smoothing']!=0 or a['phase1_lr_schedule']!='cosine':raise ValueError('Baseline mechanisms differ')
    if clean(resolved_config(SimpleNamespace(**a)))!=d.read(Path(c['output_root'])/'resolved_native_args.json'):raise ValueError('Checkpoint actual args mismatch')


def snapshot(row=None):
    configs=[d.config(r) for r in d.rows() if row is None or r['row_id']==row]
    if not configs:raise ValueError('Unregistered evaluation row')
    for c in configs:
        f=d.read(d.BASE/c['row_id']/'source_frozen.json')
        if any(f.get(k)!=v for k,v in dict(status='SOURCE_FROZEN',config=c,target_access=False,selection='final_only',epoch=c['epochs'],optimizer_steps=c['updates']).items()):raise ValueError('Row freeze differs')
        source_provenance(c)
    return configs


def predict(row):
    import numpy as np
    import torch
    from experiments.cvs_feature_disentangle.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    c=snapshot(row)[0];source=source_provenance(c);out=d.BASE/row/'prediction'
    if out.exists():raise FileExistsError('Existing prediction; reconcile, never overwrite')
    torch.set_num_threads(2);device=torch.device('cuda:0');started=time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY),installed(c,training=False) as native:
        ck=torch.load(source/'final_ssdg.pth',map_location=device,weights_only=False);checkpoint_provenance(c,ck)
        model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),device);model.load_state_dict(ck['model'],strict=True);model.eval();del ck
        with torch.no_grad():z=model(torch.zeros(2,2,256,device=device))
        if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Frozen scratch model forward failed')
        ids=shared.physical_ids();array=np.load(Path(d.CAPSULE)/'clean.npy',mmap_mode='r',allow_pickle=False)
        if array.shape!=(COUNT,2,256):raise ValueError('Clean IQ shape differs')
        out.mkdir(parents=True,exist_ok=False)
        d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(source/'final_ssdg.pth'),scratch_only=True,optimizer_steps=c['updates'],truth_read=False,query_fit=False))
        d.write(out/'resolved_config.json',dict(row_id=row,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,evaluation_commit=(d.ROOT/'release_commit.txt').read_text().strip(),views=['clean'],backend_flags=actual_flags(),hardware=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters())))
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();pred=[];logq=[]
        with torch.no_grad():
            for i in range(0,COUNT,256):
                logits=model(shared.iq_tensor(np.array(array[i:i+256],dtype=np.float32,copy=True),device))
                if not torch.isfinite(logits).all() or logits.shape!=(min(256,COUNT-i),6):raise ValueError('Invalid clean logits')
                pred.extend(logits.argmax(1).cpu().tolist());logq.extend(logits.log_softmax(1).cpu().tolist())
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        np.savez(out/'predictions.npz',ids=ids,clean=np.asarray(pred,dtype=np.int64),clean_log_probabilities=np.asarray(logq,dtype=np.float32))
        d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=COUNT,views=['clean'],truth_read=False,query_fit=False,inference_seconds=dict(clean=seconds),wall_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))
        print(json.dumps(dict(row_id=row,status='PREDICTIONS_COMPLETE',count=COUNT)),flush=True)


def prediction_preflight(configs):
    import numpy as np
    ids=shared.physical_ids()
    for c in configs:
        out=d.BASE/c['row_id']/'prediction';done=d.read(out/'complete.json');p=d.read(out/'provenance.json')
        if any(done.get(k)!=v for k,v in dict(status='PREDICTIONS_COMPLETE',count=COUNT,views=['clean'],truth_read=False,query_fit=False).items()):raise ValueError('Truth closed: incomplete prediction')
        if p['config']!=c or p['checkpoint']!=str(Path(c['output_root'])/'final_ssdg.pth') or p['optimizer_steps']!=c['updates'] or not p['scratch_only'] or p['truth_read'] or p['query_fit']:raise ValueError('Prediction provenance differs')
        with np.load(out/'predictions.npz',allow_pickle=False) as values:
            if set(values.files)!={'ids','clean','clean_log_probabilities'} or not np.array_equal(values['ids'],ids):raise ValueError('Prediction physical IDs differ')
            shared.validate_prediction(values['clean']);shared.validate_log_probabilities(values['clean_log_probabilities'],values['clean'])
    return ids


def score(row=None):
    import numpy as np
    from comparison_suite.score import csvwrite
    configs=snapshot(row);ids=prediction_preflight(configs);root=d.BASE/'row_scoring'/row if row else d.BASE
    if (root/'scoring_complete.json').exists():return
    if (root/'scores.json').exists():raise FileExistsError('Partial score: reconcile before proceeding')
    reference_configs=[dict(c,parent_checkpoint=c['reference_E200']['checkpoint'],parent_prediction=c['reference_E200']['prediction']) for c in configs]
    references=shared.parent_predictions(reference_configs,ids)
    y,rx,days=shared.truth_arrays(d.read(d.TRUTH),ids)  # independent scorer only
    results=[];day_results=[];parents=[];seen=set();resources=[]
    for c in configs:
        out=d.BASE/c['row_id']/'prediction';identity=dict(c,arm=c['budget_arm'])
        with np.load(out/'predictions.npz',allow_pickle=False) as p:
            rr,dd=shared.metric_rows(identity,p['clean'],ids,y,rx,days)
            for r in rr+dd:
                dim,k=r['dimension'],r['stratum'];mask=np.ones(len(y),bool) if dim=='overall' else rx==k if dim=='receiver' else days==k if dim=='day' else y==d.CLASSES.index(k)
                before=references[c['model_seed']][mask]==y[mask];after=p['clean'][mask]==y[mask]
                r.update(E200_fix_count=int((~before&after).sum()),E200_break_count=int((before&~after).sum()),NLL=float(-p['clean_log_probabilities'][np.arange(len(y))[mask],y[mask]].mean()))
            results.extend(rr);day_results.extend(dd)
        if c['model_seed'] not in seen:
            pc=dict(identity,arm='P0',row_id=f"P0-s{c['model_seed']}")
            parents.extend(shared.metric_rows(pc,references[c['model_seed']],ids,y,rx,days)[0]);seen.add(c['model_seed'])
        source=Path(c['output_root']);done=d.read(source/'completion.json')
        resources.append(dict(row_id=c['row_id'],epochs=c['epochs'],optimizer_steps=c['updates'],training_seconds=done['elapsed_seconds'],prediction=d.read(out/'complete.json'),reference_E200=c['reference_E200'],cost_scope='measured shared GPU wall cost; E200 historical wall cost separately referenced'))
    summary=shared.summarize(results+parents);pairs=shared.paired_results(results+parents)
    d.write(root/'scores.json',dict(status='SCORED',results=results,target_feedback_scope='explicit user exception for epoch-budget exploration only'))
    d.write(root/'day_scores.json',dict(results=day_results));d.write(root/'parent_scores.json',dict(results=parents,source='reused fixed E200 clean predictions'))
    d.write(root/'summary.json',dict(results=summary));d.write(root/'paired_results.json',dict(comparisons=pairs))
    d.write(root/'resources.json',dict(rows=resources))
    for filename,rows in [('scores',results),('day_scores',day_results),('summary',summary)]:csvwrite(root/(filename+'.csv'),[{k:v for k,v in r.items() if k!='confusion'} for r in rows])
    decisions=[dict(arm=p['treatment'],mean_accuracy_difference=p['mean_difference'],margin=.003,complete_three_seeds=p['complete_three_seeds'],within_margin=p['mean_difference']>=-.003 if p['complete_three_seeds'] else None) for p in pairs if p['control']=='P0']
    d.write(root/'compression_assessment.json',dict(criteria='clean three-seed mean no more than 0.3 percentage points below original E200; descriptive not a statistical noninferiority claim',decisions=decisions,benchmark_role='exposed development benchmark',source_V_used_for_epoch_choice=False,training_reads_target=False))
    lines=['# CE＋旧门控：压缩 cosine 日程的 clean 测试','', '每行从零训练，固定最终轮权重，prediction 固定后由独立 scorer 连接 truth。E200 复用原预测。','用户明确允许仅针对 epoch 效率探索使用测试成绩决定后续轮数；本比较不声称独立盲测。','', '|预算|种子|准确率 (%)|','|---|---|---:|']
    lines += [f"|{r['arm']}|{r['model_seed']}|{100*r['accuracy']:.4f}|" for r in results+parents if r['dimension']=='overall']
    (root/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(root/'scoring_complete.json',dict(status='SCORED_COMPLETE',models=len(configs),views=1,query_count_per_view=COUNT,prediction_count=len(configs)*COUNT,metric_records=len(results),day_metric_records=len(day_results),truth_last=True,independent_recount='VERIFIED'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['predict','score'],required=True);p.add_argument('--row');a=p.parse_args()
    if a.mode=='predict':predict(a.row)
    else:score(a.row)
