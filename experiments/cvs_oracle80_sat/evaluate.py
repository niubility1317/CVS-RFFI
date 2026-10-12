"""Seven views complete before a separate truth-side scorer can open labels."""
import os,time
from pathlib import Path
from types import SimpleNamespace
from . import design as d

def configure():
    from experiments.cvs_ce_epoch2040 import evaluate as e
    e.d=d;e.shared.d=d;e.checkpoint_provenance=checkpoint_provenance
    return e

def checkpoint_provenance(c,ck):
    from experiments.cvs_feature_disentangle.source import clean
    from scripts.train_daot_rc4_baseline import resolved_config
    expected=dict(epoch=30,candidate_id=c['row_id'],run_id=c['run_id'],checkpoint_selection='final_only',a1_ema_successful_updates=16890)
    if any(ck.get(k)!=v for k,v in expected.items()):raise ValueError('Checkpoint identity/budget differs')
    a=ck['args']
    if a['baseline_ckpt'] or a['teacher_ckpt'] or not a['from_scratch'] or a['epochs']!=30 or a['label_epochs']!=30 or a['pseudo_epochs']!=0 or a['use_unlabeled'] or a['use_ema_teacher'] or a['label_smoothing']!=0 or a['phase1_lr_schedule']!='cosine':raise ValueError('Scratch supervision changed')
    if any(a.get(k)!=v for k,v in d.SAT_ARGS.items()):raise ValueError('Satellite training recipe differs')
    if clean(resolved_config(SimpleNamespace(**a)))!=d.read(Path(c['output_root'])/'resolved_native_args.json'):raise ValueError('Checkpoint actual args differ')

def view_manifest():
    from experiments.cvs_feature_disentangle import evaluate as e
    return e.manifest(d)

def predict():
    import numpy as np,torch
    from .data import oracle_native
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    e=configure();c=e.snapshot(d.rows()[0]['row_id'])[0];source=e.source_provenance(c)
    manifest=view_manifest();ids=e.shared.physical_ids();out=d.BASE/c['row_id']/'prediction'
    if out.exists():raise FileExistsError('Existing prediction; reconcile')
    torch.set_num_threads(2);device=torch.device('cuda:0');start=time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY),oracle_native(c,training=False) as n:
        ck=torch.load(source/'final_ssdg.pth',map_location=device,weights_only=False);e.checkpoint_provenance(c,ck)
        if (ck['args']['labeled_ratio'],ck['args']['unlabeled_ratio'],ck['args']['source_val_ratio'])!=(.8,0.,.2):raise ValueError('Checkpoint split differs')
        model=n.build_baseline_model(SimpleNamespace(**ck['baseline_args']),device);model.load_state_dict(ck['model'],strict=True);model.eval()
        with torch.no_grad():z=model(torch.zeros(2,2,256,device=device))
        if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Frozen checkpoint smoke failed')
        out.mkdir(parents=True,exist_ok=False);predictions={'ids':ids};times={}
        d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(source/'final_ssdg.pth'),scratch_only=True,truth_read=False,query_fit=False))
        d.write(out/'resolved_config.json',dict(views=list(d.VIEWS),pid=os.getpid(),cwd=os.getcwd(),backend_flags=actual_flags(),commit=(d.ROOT/'release_commit.txt').read_text().strip()))
        with torch.no_grad():
            for view in d.VIEWS:
                array=np.load(manifest['clean_ref'] if view=='clean' else d.VIEWS_ROOT/(view+'.npy'),mmap_mode='r',allow_pickle=False)
                if array.shape!=(d.COUNT,2,256):raise ValueError('View dimensions differ')
                values=[];logp=[];torch.cuda.synchronize();t=time.perf_counter()
                for i in range(0,d.COUNT,256):
                    logits=model(e.shared.iq_tensor(np.array(array[i:i+256],dtype=np.float32,copy=True),device))
                    if logits.shape!=(min(256,d.COUNT-i),6) or not torch.isfinite(logits).all():raise ValueError('Invalid prediction logits')
                    values.extend(logits.argmax(1).cpu().tolist());logp.extend(logits.log_softmax(1).cpu().tolist())
                torch.cuda.synchronize();times[view]=time.perf_counter()-t
                predictions[view]=np.asarray(values,dtype=np.int64);predictions[view+'_log_probabilities']=np.asarray(logp,dtype=np.float32)
                print('PREDICTION VIEW COMPLETE '+view,flush=True)
        np.savez(out/'predictions.npz',**predictions)
        d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=d.COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,inference_seconds=times,wall_seconds=time.perf_counter()-start))

def prediction_preflight():
    import numpy as np
    e=configure();c=e.snapshot(d.rows()[0]['row_id'])[0];out=d.BASE/c['row_id']/'prediction';ids=e.shared.physical_ids()
    done=d.read(out/'complete.json');p=d.read(out/'provenance.json')
    if any(done.get(k)!=v for k,v in dict(status='PREDICTIONS_COMPLETE',count=d.COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False).items()):raise ValueError('Truth closed: incomplete predictions')
    if p['config']!=c or p['checkpoint']!=str(Path(c['output_root'])/'final_ssdg.pth') or not p['scratch_only'] or p['truth_read'] or p['query_fit']:raise ValueError('Prediction provenance differs')
    with np.load(out/'predictions.npz',allow_pickle=False) as values:
        if set(values.files)!={'ids',*d.VIEWS,*(v+'_log_probabilities' for v in d.VIEWS)} or not np.array_equal(values['ids'],ids):raise ValueError('Prediction IDs/keys differ')
        for view in d.VIEWS:
            e.shared.validate_prediction(values[view]);e.shared.validate_log_probabilities(values[view+'_log_probabilities'],values[view])
    return c,ids

def score():
    import numpy as np
    def metric(y,pred):
        cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,pred),1)
        denominator=cm.sum(0)+cm.sum(1)
        f1=np.divide(2*cm.diagonal(),denominator,out=np.zeros(6,dtype=float),where=denominator>0)
        return dict(count=len(y),accuracy=float((y==pred).mean()),Macro_F1=float(f1.mean()),confusion=cm.tolist())
    c,ids=prediction_preflight();out=d.BASE/'scores'
    if out.exists():raise FileExistsError('Existing scoring; reconcile')
    e=configure();y,rx,days=e.shared.truth_arrays(d.read(d.TRUTH),ids)
    records=[]
    with np.load(d.BASE/c['row_id']/'prediction/predictions.npz',allow_pickle=False) as predictions:
        for view in d.VIEWS:
            for dimension,values in [('overall',[None]),('receiver',sorted(set(rx))),('day',sorted(set(days))),('TX',range(6))]:
                for value in values:
                    mask=np.ones(d.COUNT,dtype=bool) if value is None else rx==value if dimension=='receiver' else days==value if dimension=='day' else y==value
                    result=metric(y[mask],predictions[view][mask]);result.update(view=view,dimension=dimension,stratum=d.CLASSES[value] if dimension=='TX' else value,model_seed=c['model_seed'],epochs=30,
                        NLL=float(-predictions[view+'_log_probabilities'][np.arange(d.COUNT)[mask],y[mask]].mean()))
                    records.append(result)
    d.write(out/'metrics.json',dict(records=records,truth_last=True))
    source=d.read(Path(c['output_root'])/'completion.json')
    d.write(out/'resources.json',dict(epochs=30,optimizer_updates=16890,training_seconds=source['elapsed_seconds'],prediction=d.read(d.BASE/c['row_id']/'prediction/complete.json')))
    overall=[r for r in records if r['dimension']=='overall']
    lines=['# 80% 全标签训练／20% 验证：E30＋LEO弱信道增强','',
        '全部七种视图预测固定后，由独立进程连接 truth 评分。单 seed2026092701；从零训练；clean CE＋0.68卫星CE；16890次更新；E1起启用三段LEO弱信道增强。',
        '该结果是特定划分、网络和 E30 预算下的全标签参考，不构成伪标签方法的数学上限。',
        '与默认7/63/30弱标注实验同时存在验证划分和训练曝光差异；不把全部差异归因于伪标签质量。',
        '使用已暴露基准，不声明新的独立盲测。','', '|视图|准确率 (%)|Macro-F1 (%)|','|---|---:|---:|']
    lines += [f"|{r['view']}|{100*r['accuracy']:.4f}|{100*r['Macro_F1']:.4f}|" for r in overall]
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'scoring_complete.json',dict(status='SCORED_COMPLETE',models=1,views=list(d.VIEWS),prediction_count=d.COUNT*len(d.VIEWS),truth_last=True,overall=overall))
    # Comparison consumes fixed old predictions only after all new views pass
    # prediction_preflight; no training/selection process reads these metrics.
    control=Path(c['control_prediction'])
    prior=d.read(control.parent/'complete.json')
    if prior['status']!='PREDICTIONS_COMPLETE' or prior['views']!=list(d.VIEWS) or prior['count']!=d.COUNT or prior['truth_read'] or prior['query_fit']:raise ValueError('Control predictions incomplete')
    comparisons=[]
    with np.load(control,allow_pickle=False) as before,np.load(d.BASE/c['row_id']/'prediction/predictions.npz',allow_pickle=False) as after:
        if not np.array_equal(before['ids'],ids):raise ValueError('Control physical IDs differ')
        for view in d.VIEWS:
            configure().shared.validate_prediction(before[view])
            old=before[view]==y;new=after[view]==y
            comparisons.append(dict(view=view,control_accuracy=float(old.mean()),augmented_accuracy=float(new.mean()),difference_pp=100*float(new.mean()-old.mean()),fix_count=int((~old&new).sum()),break_count=int((old&~new).sum()),model_seed=2026092701,control_run_id=c['control_run_id']))
    d.write(out/'control_comparison.json',dict(status='PAIRED_COMPLETE',records=comparisons,truth_last=True,interpretation='same physical roles/seed/E30/update count; augmented run has double input rows and changed batch statistics; single-seed descriptive comparison'))
    print('ORACLE SATELLITE SCORED COMPLETE',flush=True)
