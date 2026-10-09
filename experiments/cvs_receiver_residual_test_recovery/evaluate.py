"""Frozen per-packet inference; independent truth-last scoring subprocess."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import importlib
from experiments.cvs_receiver_residual_test_recovery.contract import validate_source_contract
from experiments.cvs_receiver_residual_test_recovery.design import PROJECT, RUNS

d=None
TEST_RUN=None

def configure(run_id):
    global d,TEST_RUN
    selected=[r for r in RUNS if r["run_id"]==run_id]
    if len(selected)!=1:raise ValueError("Unregistered evaluation run")
    r=selected[0];release=Path(PROJECT)/"releases"/r["source_release"]
    if (release/"release_commit.txt").read_text().strip()!=r["source_commit"]:raise ValueError("Source release changed")
    sys.path[:0]=[str(release),str(release/"code")]
    d=importlib.import_module("experiments."+r["package"]+".design")
    if d.ROOT!=release or d.RUN!=r["parent_run_id"]:raise ValueError("Wrong source release imported")
    TEST_RUN=run_id
    return r

def prediction_root(c):
    return Path(d.PROJECT)/"runs"/TEST_RUN/d.row_id(c["arm"],c["model_seed"])/"prediction"


def source_provenance(c):
    source=Path(c['output_root'])
    done=d.read(source/'completion.json');initial=d.read(source/'initialization.json')
    resolved=d.read(source/'resolved_config.json');d.validate_config(resolved)
    if (done['status']!='SOURCE_TRAINED' or done['epoch']!=200 or done['steps']!=10000
        or done['target_access'] or done['target_evaluated'] or initial['status']!='SCRATCH'
        or not initial['scratch_only'] or initial['checkpoint'] is not None
        or initial['checkpoint_sources'] or initial['ancestors'] or initial['target_access']
        or initial['target_contact'] or initial['physical_roles']!='EXACT_MATCH'
        or initial['model_seed']!=c['model_seed'] or any(resolved[k]!=v for k,v in c.items())):
        raise ValueError('Own scratch checkpoint provenance invalid')
    contract=d.read(source/'source_contract.json');expected=d.read(d.SOURCE)
    validate_source_contract(contract,expected,d.CLASSES,c["dataset"])
    frozen=d.read(source/'source_selection.json')
    if (frozen['status']!='SOURCE_SELECTION_FROZEN' or frozen['epoch']!=200
        or frozen['arm']!=c['arm'] or frozen['variant']!='reference_response'
        or frozen['target_access'] or frozen['target_score_used']):
        raise ValueError('Source selection not frozen')
    return source,done,initial,resolved,contract


def snapshot():
    freeze=d.read(Path(d.PROJECT)/'runs'/TEST_RUN/'source_matrix_frozen.json')
    if (freeze['status']!='ALL_SOURCE_FROZEN' or freeze['run_id']!=TEST_RUN
        or freeze['rows']!=d.rows() or freeze['target_access'] is not False
        or freeze['selection']!='ALL_PREREGISTERED_FIXED_CONTROLS'):
        raise ValueError('All16 source rows must be frozen before any query')
    return freeze['rows']


def manifest():
    m=d.read(Path(d.VIEWS_ROOT)/'manifest.json')
    expected=dict(status='VALIDATED_ONCE',count=d.COUNT,scenes=list(d.SCENES),classes=d.CLASSES,
        truth_read=False,channel='residual/post_sync/noeq',source_capsule=d.CAPSULE,
        clean_ref=d.CAPSULE+'/clean.npy',rows_share_observations=True)
    if any(m.get(k)!=v for k,v in expected.items()): raise ValueError('Existing received views differ')
    return m


def physical_ids():
    import numpy as np
    with np.load(Path(d.VIEWS_ROOT)/'index.npz',allow_pickle=False) as p: ids=p['ids'].copy()
    with np.load(Path(d.CAPSULE)/'index.npz',allow_pickle=False) as p:
        if len(ids)!=d.COUNT or len(set(ids.tolist()))!=d.COUNT or not np.array_equal(ids,p['ids']):
            raise ValueError('Received-view physical IDs differ')
    return ids


def predict(rid):
    import numpy as np
    import torch
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    matches=[c for c in snapshot() if d.row_id(c['arm'],c['model_seed'])==rid]
    if len(matches)!=1: raise ValueError('Unregistered prediction row')
    c=matches[0];source,done,initial,resolved,contract=source_provenance(c)
    torch.set_num_threads(2);device=torch.device('cuda:0');started=time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY):
        ck=torch.load(source/'last.pt',map_location=device,weights_only=False)
        if (ck['config']!=resolved or ck['source_contract']!=contract or ck['initialization']!=initial
            or ck['epoch']!=200 or ck['method']!=d.METHOD or ck['arm']!=c['arm']
            or ck['variant']!='reference_response' or ck['selection']!='fixed_last_epoch'):
            raise ValueError('Checkpoint contents differ from provenance')
        model=d.build_model(c['arm']).to(device).eval()
        if model.contract()!=resolved['architecture_actual']: raise ValueError('Architecture differs')
        model.load_state_dict(ck['model'],strict=True);del ck
        if float(model.strength)!=1.: raise ValueError('Frozen residual strength not final')
        with torch.no_grad():
            logits=model(torch.zeros(2,2,256,device=device))
            if logits.shape!=(2,6) or not torch.isfinite(logits).all(): raise ValueError('Checkpoint smoke failed')
        m=manifest();ids=physical_ids()
        out=prediction_root(c);out.mkdir(parents=True,exist_ok=False)
        d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(source/'last.pt'),
            scratch_sources=[],source_roles='EXACT_MATCH',own_E200=True,optimizer_steps=10000,
            query_fit=False,truth_read=False))
        d.write(out/'resolved_config.json',dict(row_id=rid,run_id=TEST_RUN,pid=os.getpid(),cwd=os.getcwd(),
            python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),
            views=list(d.VIEWS),views_root=d.VIEWS_ROOT,backend_flags=actual_flags(),batch_size=256,
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=0,
            hardware=torch.cuda.get_device_name(device),torch_version=torch.__version__))
        predictions={};timings={};torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for view in d.VIEWS:
                a=np.load(m['clean_ref'] if view=='clean' else Path(d.VIEWS_ROOT)/(view+'.npy'),
                    mmap_mode='r',allow_pickle=False)
                if a.shape!=(d.COUNT,2,256): raise ValueError('Received-view shape differs')
                torch.cuda.synchronize();tic=time.perf_counter();values=[]
                for i in range(0,len(ids),256):
                    batch=np.ascontiguousarray(a[i:i+256],dtype=np.float32)
                    x=torch.frombuffer(bytearray(batch.tobytes()),dtype=torch.float32).reshape(batch.shape).to(device)
                    logits=model(x)
                    if logits.shape!=(len(x),6) or not torch.isfinite(logits).all(): raise ValueError('Nonfinite scores')
                    values.extend(logits.argmax(1).cpu().tolist())
                torch.cuda.synchronize();timings[view]=time.perf_counter()-tic
                predictions[view]=np.asarray(values,dtype=np.int64)
                print('PREDICT '+json.dumps(dict(row_id=rid,view=view,count=len(values),seconds=timings[view],truth_read=False)),flush=True)
        np.savez(out/'predictions.npz',ids=ids,**predictions)
        import resource
        d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=d.COUNT,views=list(d.VIEWS),
            truth_read=False,query_fit=False,inference_seconds=timings,wall_seconds=time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def validate_predictions():
    import numpy as np
    rows=snapshot();manifest();ids=physical_ids();base=Path(d.PROJECT)/'runs'/TEST_RUN
    # Every prediction/view is complete before the separate scorer opens truth.
    for c in rows:
        out=prediction_root(c);done=d.read(out/'complete.json')
        if (done['status']!='PREDICTIONS_COMPLETE' or done['count']!=d.COUNT or done['views']!=list(d.VIEWS)
            or done['truth_read'] or done['query_fit'] or d.read(out/'provenance.json')['config']!=c):
            raise ValueError('Truth remains closed: predictions incomplete')
        with np.load(out/'predictions.npz',allow_pickle=False) as p:
            if set(p.files)!=set(d.VIEWS)|{'ids'} or not np.array_equal(ids,p['ids']): raise ValueError('Prediction IDs/views differ')
            for v in d.VIEWS:
                a=p[v]
                if a.shape!=(d.COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6:
                    raise ValueError('Prediction array invalid')
    return rows,ids,base


def score():
    import numpy as np
    from comparison_suite.score import metrics
    rows,ids,base=validate_predictions()
    truth=d.read(d.TRUTH)
    y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids])
    if set(y)!=set(range(6)) or set(rx)!=set(map(str,[0,2,5,7,9,10,11])): raise ValueError('Truth coverage differs')
    results=[];resources=[]
    for c in rows:
        rid=d.row_id(c['arm'],c['model_seed']);row=dict(row_id=rid,arm=c['arm'],model_seed=c['model_seed'])
        source=Path(c['output_root']);out=prediction_root(c)
        resources.append(dict(**row,source=d.read(source/'resource_profile.json'),prediction=d.read(out/'complete.json')))
        with np.load(out/'predictions.npz',allow_pickle=False) as p:
            for view in d.VIEWS:
                for dim,strata in [('overall',['ALL']),('receiver',sorted(set(rx))),('transmitter',d.CLASSES)]:
                    for stratum in strata:
                        mask=np.ones(len(ids),bool) if dim=='overall' else rx==stratum if dim=='receiver' else y==d.CLASSES.index(stratum)
                        r=metrics(y[mask],p[view][mask],6)
                        cm=np.bincount(6*y[mask]+p[view][mask],minlength=36).reshape(6,6)
                        den=cm.sum(0)+cm.sum(1)
                        f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()
                        if (r['confusion']!=cm.tolist() or r['query_count']!=int(cm.sum())
                            or abs(r['accuracy']-float(np.trace(cm)/cm.sum()))>1e-12 or abs(r['macro_f1']-f1)>1e-12):
                            raise ValueError('Independent confusion/accuracy/F1 recount differs')
                        results.append(dict(**row,view=view,dimension=dim,stratum=stratum,**r))
    summary=[];paired=[]
    overall={(r['arm'],r['model_seed'],r['view']):r for r in results if r['dimension']=='overall'}
    for view in d.VIEWS:
        for arm in d.ARMS:
            rs=[overall[arm,s,view] for s in d.SEEDS]
            worst=[min(r['accuracy'] for r in results if r['row_id']==a['row_id'] and r['view']==view and r['dimension']=='receiver') for a in rs]
            item=dict(arm=arm,view=view,seed_count=4)
            for k in ('accuracy','macro_f1'):
                vals=[r[k] for r in rs];item[k+'_mean']=float(np.mean(vals));item[k+'_sd']=float(np.std(vals,ddof=1))
            item.update(worst_rx_mean=float(np.mean(worst)),worst_rx_sd=float(np.std(worst,ddof=1)));summary.append(item)
        for name,weights in [('displacement_vs_baseline',{'displacement':1,'baseline':-1}),
            ('contribution_vs_baseline',{'contribution':1,'baseline':-1}),
            ('combined_vs_baseline',{'combined':1,'baseline':-1}),
            ('interaction',{'combined':1,'displacement':-1,'contribution':-1,'baseline':1})]:
            vals=[100*sum(w*overall[a,s,view]['accuracy'] for a,w in weights.items()) for s in d.SEEDS]
            paired.append(dict(contrast=name,view=view,seed_differences_pp=vals,mean_pp=float(np.mean(vals)),sd_pp=float(np.std(vals,ddof=1))))
    d.write(base/'scores.json',dict(status='SCORED',results=results,target_feedback_forbidden=True))
    csvwrite(base/'scores.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in results])
    d.write(base/'summary.json',dict(results=summary));csvwrite(base/'summary.csv',summary)
    d.write(base/'paired_results.json',dict(comparisons=paired,interpretation='paired model seeds; four-seed descriptive evidence'))
    d.write(base/'resources.json',dict(rows=resources))
    lines=['# Receiver residual: fixed 2×2 comparison','',
        'All 16 scratch E200 rows were frozen before query access. Truth opened only after all 16×7 predictions.',
        'Exposed benchmark; six complete practical residual views share physical IDs. Phase2/K/new TX/H: N/A.',
        'No target feedback to tuning, ranking, model selection or selective rerun. All negative results retained.','',
        '|Arm|View|Accuracy mean ± SD|Worst RX mean|','|---|---|---:|---:|']
    lines += [f"|{r['arm']}|{r['view']}|{100*r['accuracy_mean']:.3f} ± {100*r['accuracy_sd']:.3f}%|{100*r['worst_rx_mean']:.3f}%|" for r in summary]
    (base/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(base/'scoring_complete.json',dict(status='SCORED_COMPLETE',rows=16,views=7,
        result_rows=len(results),independent_recount='VERIFIED',truth_last=True,target_feedback_forbidden=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["predict","score"],required=True);p.add_argument("--row");p.add_argument("--run",required=True)
    a=p.parse_args();configure(a.run);predict(a.row) if a.mode=="predict" else score()
