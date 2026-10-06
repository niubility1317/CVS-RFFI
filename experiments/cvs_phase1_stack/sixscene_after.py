"""Queued evaluation only: original 136 frozen rows, clean plus six full views."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN='20261006-phase1-reference-stack-sixscene-manysig-m136-r02'
RELEASE='cvs_reference_stack_recovery_20261006_r02'
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
WORKER_ROOT=Path(PROJECT)/'releases'/RELEASE
WORKER_COMMIT=None  # Resolved from this same immutable release in original().
BASE=Path(PROJECT)/'runs'/RUN
VIEWS_ROOT=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
VIEWS=('clean',*SCENES)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def original():
    global WORKER_COMMIT
    WORKER_COMMIT=(Path(__file__).resolve().parents[2]/'release_commit.txt').read_text().strip()
    if (WORKER_ROOT/'release_commit.txt').read_text().strip()!=WORKER_COMMIT:raise ValueError('Worker release mismatch')
    sys.path[:0]=[str(WORKER_ROOT),str(WORKER_ROOT/'code')]
    from experiments.cvs_phase1_stack import dispatch as d
    if d.ROOT!=WORKER_ROOT:raise ValueError('Wrong scientific code import')
    return d


def parent_ready(d):
    p=Path(d.BASE)
    if (p/'failure.json').exists():raise RuntimeError('Parent technical failure; preserve all outputs, no automatic retry')
    if not (p/'completion.json').exists():return False
    done=d.read(p/'completion.json')
    if done!=dict(status='ANALYZED',rows=136,independent_recount='VERIFIED',target_feedback_forbidden=True):raise ValueError('Unexpected parent completion')
    if d.read(p/'all_sources_frozen.json')!=dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=[r['row_id'] for r in d.rows()],stages=list(d.STAGES),target_access=False):raise ValueError('All-source freeze missing')
    for stage in d.STAGES:d.validate_freeze(d.read(p/(stage+'_source_frozen.json')),stage)
    return True


def manifest(d):
    m=d.read(VIEWS_ROOT/'manifest.json')
    required=dict(status='VALIDATED_ONCE',count=168000,scenes=list(SCENES),classes=d.CLASSES,truth_read=False,channel='residual/post_sync/noeq',source_capsule=d.CAPSULE,clean_ref=d.CAPSULE+'/clean.npy',rows_share_observations=True)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Existing sixscene capsule does not match')
    return m


def predict(rid):
    d=original()
    if not parent_ready(d):raise RuntimeError('Parent not complete; query remains closed')
    row=next(r for r in d.rows() if r['row_id']==rid)
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.source import clean
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    c=d.read(Path(d.BASE)/'configs'/('source-'+rid+'.json'));d.validate(c)
    f=d.read(Path(d.BASE)/(row['stage']+'_source_frozen.json'))
    if next(r['config'] for r in f['rows'] if r['row_id']==rid)!=c:raise ValueError('Frozen config differs')
    source=Path(c['output_root']);done=d.read(source/'completion.json');init=d.read(source/'initialization.json')
    d.require_budget(done['logged_steps'],done['optimizer_steps'])
    if done['status']!='SOURCE_TRAINED' or done['config']!=c or done['epoch']!=200 or done['target_access'] or done['target_evaluated'] or not init['scratch_only'] or init['ancestors'] or init['checkpoint_sources'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH':raise ValueError('Checkpoint provenance mismatch')
    contract=d.read(source/'source_contract.json');expected=d.read(d.SOURCE)
    for key in ['role_ids','classes','source_rxs','source_days','ratios','split_seed','equalized','out_len','normalize']:
        if contract[key]!=expected[key]:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH '+key)
    torch.set_num_threads(2);device=torch.device('cuda:0')
    with numerical_context(d.FULL_FP32_POLICY),installed(c) as native:
        ck=torch.load(source/'final_ssdg.pth',map_location=device,weights_only=False)
        if ck['epoch']!=200 or ck['candidate_id']!=rid or ck['run_id']!=c['run_id'] or ck['checkpoint_selection']!='final_only' or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or not ck['args']['from_scratch']:raise ValueError('Checkpoint contents differ')
        from scripts.train_daot_rc4_baseline import resolved_config
        if clean(resolved_config(SimpleNamespace(**ck['args'])))!=d.read(source/'resolved_native_args.json'):raise ValueError('Checkpoint args differ')
        model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),device);model.load_state_dict(ck['model'],strict=True);model.eval()
        with torch.no_grad():z=model(torch.zeros(2,2,256,device=device))
        if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Checkpoint smoke failed')
        m=manifest(d)
        with np.load(VIEWS_ROOT/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
        with np.load(Path(d.CAPSULE)/'index.npz',allow_pickle=False) as ix:
            if len(ids)!=168000 or len(set(ids.tolist()))!=168000 or not np.array_equal(ids,ix['ids']):raise ValueError('Sixscene physical IDs differ')
        out=BASE/rid/'prediction';out.mkdir(parents=True,exist_ok=False)
        write(out/'provenance.json',dict(status='VERIFIED',checkpoint=str(source/'final_ssdg.pth'),scratch_sources=[],source_roles='EXACT_MATCH',own_E200=True,query_fit=False,truth_read=False))
        write(out/'resolved_config.json',dict(row_id=rid,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,worker_commit=WORKER_COMMIT,checkpoint_source_run=c['run_id'],evaluation_commit=WORKER_COMMIT,views=list(VIEWS),views_root=str(VIEWS_ROOT),backend_flags=actual_flags(),parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=0,hardware=torch.cuda.get_device_name(device),torch_version=torch.__version__,batch_size=256))
        predictions={};timings={};torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for view in VIEWS:
                array=np.load(m['clean_ref'] if view=='clean' else VIEWS_ROOT/(view+'.npy'),mmap_mode='r',allow_pickle=False)
                if array.shape!=(168000,2,256):raise ValueError('View shape mismatch')
                start=time.perf_counter();values=[]
                for i in range(0,len(ids),256):
                    a=np.ascontiguousarray(array[i:i+256],dtype=np.float32)
                    x=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
                    logits=model(x)
                    if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Invalid classifier output')
                    values.extend(logits.argmax(1).cpu().tolist())
                torch.cuda.synchronize();timings[view]=time.perf_counter()-start;predictions[view]=np.asarray(values,dtype=np.int64)
                print(json.dumps(dict(view=view,count=len(values),seconds=timings[view],truth_read=False)),flush=True)
        np.savez(out/'predictions.npz',ids=ids,**predictions)
        write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),truth_read=False,query_fit=False,inference_seconds=timings,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))


def prediction_preflight(d):
    import numpy as np
    manifest(d)
    with np.load(VIEWS_ROOT/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
    if len(ids)!=168000 or len(set(ids.tolist()))!=168000:raise ValueError('Invalid test population')
    for row in d.rows():
        out=BASE/row['row_id']/'prediction';done=d.read(out/'complete.json')
        if done['status']!='PREDICTIONS_COMPLETE' or done['count']!=len(ids) or done['views']!=list(VIEWS) or done['truth_read'] or done['query_fit']:raise ValueError('Truth remains closed: incomplete prediction')
        with np.load(out/'predictions.npz',allow_pickle=False) as p:
            if set(p.files)!=set(VIEWS)|{'ids'} or not np.array_equal(ids,p['ids']):raise ValueError('Prediction identity mismatch')
            for view in VIEWS:
                a=p[view]
                if a.shape!=(len(ids),) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:raise ValueError('Invalid prediction')
    return ids


def score():
    d=original()
    if not parent_ready(d):raise RuntimeError('Parent not frozen/completed')
    import numpy as np
    from comparison_suite.score import metrics,csvwrite
    ids=prediction_preflight(d)  # All 136x7 predictions fixed before opening truth.
    truth=d.read(d.TRUTH);y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids])
    if set(y)!=set(range(6)) or len(set(rx))!=7:raise ValueError('Truth population mismatch')
    results=[]
    for row in d.rows():
        with np.load(BASE/row['row_id']/'prediction/predictions.npz',allow_pickle=False) as p:
            for view in VIEWS:
                for dimension,strata in [('overall',['ALL']),('receiver',sorted(set(rx))),('transmitter',d.CLASSES)]:
                    for stratum in strata:
                        mask=np.ones(len(ids),bool) if dimension=='overall' else rx==stratum if dimension=='receiver' else y==d.CLASSES.index(stratum)
                        r=metrics(y[mask],p[view][mask],6)
                        # Independent bincount implementation checks every CM/metric.
                        cm=np.bincount(6*y[mask]+p[view][mask],minlength=36).reshape(6,6)
                        den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()
                        if r['confusion']!=cm.tolist() or r['query_count']!=int(cm.sum()) or abs(r['accuracy']-float(np.trace(cm)/cm.sum()))>1e-12 or abs(r['macro_f1']-f1)>1e-12:raise ValueError('Independent recount mismatch')
                        results.append(dict(**row,view=view,dimension=dimension,stratum=stratum,**r))
    write(BASE/'scores.json',dict(status='SCORED',results=results,target_feedback_forbidden=True))
    csvwrite(BASE/'scores.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in results])
    summary=[];paired=[];index={(r['row_id'],r['view'],r['dimension'],r['stratum']):r for r in results}
    for stage,arms in d.STAGES.items():
        control='bridge' if stage=='r2' else 'full' if stage=='r6' else 'base'
        for arm in arms:
            for view in VIEWS:
                group=[index[(stage+'-'+arm+'-s'+str(s),view,'overall','ALL')] for s in d.SEEDS]
                item=dict(stage=stage,arm=arm,view=view)
                for metric in ['accuracy','macro_f1']:
                    v=[r[metric] for r in group];item[metric+'_mean']=float(np.mean(v));item[metric+'_sd']=float(np.std(v,ddof=1))
                    delta=[100*(index[(stage+'-'+arm+'-s'+str(s),view,'overall','ALL')][metric]-index[(stage+'-'+control+'-s'+str(s),view,'overall','ALL')][metric]) for s in d.SEEDS]
                    paired.append(dict(stage=stage,arm=arm,control=control,view=view,metric=metric,seed_deltas_pp=delta,mean_pp=float(np.mean(delta)),sd_pp=float(np.std(delta,ddof=1))))
                worst=[min(index[(stage+'-'+arm+'-s'+str(s),view,'receiver',r)]['accuracy'] for r in sorted(set(rx))) for s in d.SEEDS]
                item.update(worst_rx_mean=float(np.mean(worst)),worst_rx_sd=float(np.std(worst,ddof=1)));summary.append(item)
    write(BASE/'summary.json',dict(results=summary,paired=paired));csvwrite(BASE/'summary.csv',summary)
    lines=['# reference_response Phase1 clean及六环境测试','','全部136个冻结模型；每视图完整168000个物理query，七视图共159936000次预测。全部预测固定后独立truth-last评分，第二实现复算通过。无目标结果回流。','', '| 阶段 | 组 | 视图 | Accuracy（%±SD） | Macro-F1（%±SD） | 最差RX均值（%） |','|---|---|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["stage"]} | {r["arm"]} | {r["view"]} | {100*r["accuracy_mean"]:.3f} ± {100*r["accuracy_sd"]:.3f} | {100*r["macro_f1_mean"]:.3f} ± {100*r["macro_f1_sd"]:.3f} | {100*r["worst_rx_mean"]:.3f} |')
    lines+=['','逐seed、RX、TX和混淆矩阵见scores.json/csv。配对差值见summary.json。四seed标准差仅描述模型随机性，六个信道观测固定。完整六环境与原三环境分层口径不同。Phase2/K/新增类/H均N/A。已暴露基准固定设计，不声明全新盲测。']
    (BASE/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(BASE/'scoring_complete.json',dict(status='SCORED_COMPLETE',models=136,views=7,query_count_per_view=168000,prediction_count=136*7*168000,metric_records=len(results),truth_last=True,independent_recount='VERIFIED'))


def dispatch():
    d=original();BASE.mkdir(parents=True,exist_ok=False);logs=Path(PROJECT)/'logs'/RUN;logs.mkdir(parents=True,exist_ok=False)
    from experiments.cvs_phase1_stack.capacity16 import proc
    write(BASE/'dispatcher.json',dict(**proc(os.getpid()),owner='codex/root/reference-stack-sixscene-20261006',evaluation_commit=WORKER_COMMIT,parent_run=d.RUN))
    try:
        manifest(d)
        while not parent_ready(d):
            write(BASE/'state.json',dict(status='WAITING_PARENT_COMPLETION',parent_run=d.RUN,updated_at=time.time(),target_read=False,gpu_reserved=False));time.sleep(60)
        sys.path.insert(0,str(WORKER_ROOT/'experiments/adv3b02_xuc/code'))
        from scripts.dispatch_xuc_full import available_gpu,occupancy
        pending=d.rows();active={};receipts=[];failures=[]
        while pending or active:
            for rid,job in list(active.items()):
                rc=job['process'].poll()
                if rc is not None:
                    if rc:failures.append(dict(row_id=rid,exit_code=rc))
                    del active[rid]
            while pending and len(active)<16 and not failures:
                gpu=available_gpu(active)
                if gpu is None:break
                cap=occupancy(active)[gpu]
                if len(cap['pids'])>=2 or cap['free_mb']<12000:break
                row=pending.pop(0);rid=row['row_id'];cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--mode','predict','--row',rid]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                path=logs/(rid+'.log')
                with path.open('x') as f:child=subprocess.Popen(cmd,cwd=WORKER_ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                active[rid]=dict(process=child,gpu=gpu);receipts.append(dict(row_id=rid,pid=child.pid,gpu=gpu,cwd=str(WORKER_ROOT),argv=cmd,log=str(path)))
                write(BASE/'launch.json',dict(rows=receipts))
            write(BASE/'state.json',dict(status='PREDICTING',active=list(active),pending=[r['row_id'] for r in pending],failures=failures,updated_at=time.time(),max_active=16,per_gpu_limit=2))
            if failures and not active:raise RuntimeError('Prediction failure; no retry '+repr(failures))
            if pending or active:time.sleep(10)
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--mode','score'],cwd=WORKER_ROOT,check=True)
        write(BASE/'completion.json',dict(status='ANALYZED',rows=136,views=list(VIEWS),independent_recount='VERIFIED',target_feedback_forbidden=True))
    except Exception as e:write(BASE/'failure.json',dict(status='FAILED',error=repr(e),no_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['dispatch','predict','score'],required=True);p.add_argument('--row');a=p.parse_args()
    if a.mode=='dispatch':dispatch()
    elif a.mode=='predict':predict(a.row)
    else:score()
