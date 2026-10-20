import argparse,json,os,subprocess,sys,time,traceback
from contextlib import ExitStack
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:sys.path.append(str(ROOT/'code'))
import numpy as np
import torch
from . import design as d
from .operators import counterfactuals,select_positions
from experiments.pamf_feature_refine.target import readers
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
from experiments.pamf_change_preserve.runner import load_restorer
from experiments.pamf_formal_restore.runner import source_residual_parent
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,code_commit
from experiments.pamf_restoration.frozen import classification_statistics
from experiments.pamf_feature_refine.practical_target import waveform_gap


def read(path):return json.loads(path.read_text())


def source_inputs(args):
    if args.parent_restoration.name!=d.PARENT_RUN or args.baseline_source_run.name!=d.BASELINE_RUN:
        raise ValueError('Registered originalR03 and fixed sourcebaseline required')
    provenance=source_residual_parent(args.parent_restoration)
    if provenance['parent_commit']!=d.PARENT_COMMIT:raise ValueError('Original legalR03 required')
    row=args.baseline_source_run/d.BASELINE_ARM
    summary=read(row/'summary.json');done=read(row/'prediction/complete.json')
    if (summary.get('source_only') is not True or summary.get('count')!=1800 or summary.get('target_reads')!=0 or
        done.get('role')!='source_V' or done.get('count')!=1800 or done.get('views')!=list(d.VIEWS) or
        any(done.get(k)!=0 for k in ('TX_label_reads','target_reads','state_updates'))):raise ValueError('Existing source-only R03 baseline predictions required')
    ids,source,index=source_reference(args.parent_cache/'validation',args.source_cache)
    positions=select_positions(source.rx[index])
    return ids,source,index,positions


@torch.no_grad()
def smoke(args):
    frozen=readers(args,'cpu');parent=load_restorer(args.parent_restoration/'residual_deterministic/final.pt').eval().requires_grad_(False)
    source=IQSource(args.source_cache,'L_s');positions=np.concatenate([np.flatnonzero(source.rx==rx)[:2] for rx in (1,3)])
    clean=np.asarray(source.iq[positions],np.float32);restored=np.asarray(parent(array_tensor(clean,'cpu')).tolist(),np.float32)
    for value in counterfactuals(clean,restored,clean).values():
        for model in frozen.values():
            logits=model(array_tensor(value,'cpu'))
            if logits.shape!=(4,6) or not torch.isfinite(logits).all():raise ValueError('Realcheckpoint source no-query smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',actual_R03_three_F=True,real_source_samples=4,
        original_default_heads=True,oracle_diagnostic_only=True,TX_label_reads=0,V_reads=0,target_reads=0,optimizer_updates=0))
    return frozen


@torch.no_grad()
def predict(args,frozen):
    ids,source,index,positions=source_inputs(args);selected=ids[positions];saved=args.output/'prediction';saved.mkdir()
    np.save(saved/'physical_ids.npy',selected,allow_pickle=False)
    before={seed:{k:v.clone() for k,v in model.state_dict().items()} for seed,model in frozen.items()}
    elapsed={};reference=args.parent_cache/'validation';base=args.baseline_source_run/d.BASELINE_ARM/'prediction'
    clean=np.load(reference/'clean.npy',mmap_mode='r',allow_pickle=False)
    if clean.shape!=(1800,2,256) or clean.dtype!=np.float32:raise ValueError('OriginalsourceVclean layout required')
    for view in d.VIEWS:
        started=time.perf_counter();observed=np.load(reference/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        parent=np.load(base/(view+'_parent_R03.npy'),mmap_mode='r',allow_pickle=False)
        if any(x.shape!=(1800,2,256) or x.dtype!=np.float32 for x in (observed,parent)):raise ValueError('Sourcebaseline layout differs')
        values=counterfactuals(np.asarray(observed[positions]),np.asarray(parent[positions]),np.asarray(clean[positions]))
        decisions={name:np.empty((d.COUNT,3),np.uint8) for name in d.READOUTS}
        for name,value in values.items():
            np.save(saved/(view+'_'+name+'_iq.npy'),value,allow_pickle=False)
            for offset in range(0,d.COUNT,d.BATCH):
                x=array_tensor(value[offset:offset+d.BATCH],'cpu')
                for column,model in enumerate(frozen.values()):
                    logits=model(x)
                    if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Finite original defaulthead logits required')
                    decisions[name][offset:offset+len(x),column]=np.asarray(logits.argmax(1).tolist(),np.uint8)
        with (saved/(view+'.npz')).open('xb') as stream:np.savez_compressed(stream,ids=selected,seeds=np.asarray(d.F_SEEDS),**decisions)
        elapsed[view]=time.perf_counter()-started
        print(json.dumps(dict(event='SOURCE_COUNTERFACTUAL_VIEW_COMPLETE',view=view,count=d.COUNT,readouts=len(d.READOUTS),seconds=elapsed[view])),flush=True)
    if any(not torch.equal(v,before[seed][k]) for seed,model in frozen.items() for k,v in model.state_dict().items()) or any(p.grad is not None for model in frozen.values() for p in model.parameters()):
        raise ValueError('Original frozenF state changed')
    write_json(saved/'complete.json',dict(status='SOURCE_PREDICTIONS_COMPLETE',role='source_V',views=list(d.VIEWS),count=d.COUNT,
        F_seeds=list(d.F_SEEDS),readouts=list(d.READOUTS),oracle_diagnostic_only=True,formal_target_predictions=False,
        runtime_supported=False,source_selection=False,TX_label_reads=0,target_reads=0,state_updates=0,per_view_seconds=elapsed))


def completed_predictions(output,ids):
    saved=output/'prediction';done=read(saved/'complete.json')
    expected=dict(status='SOURCE_PREDICTIONS_COMPLETE',role='source_V',views=list(d.VIEWS),count=d.COUNT,F_seeds=list(d.F_SEEDS),
        readouts=list(d.READOUTS),oracle_diagnostic_only=True,formal_target_predictions=False,runtime_supported=False,
        source_selection=False,TX_label_reads=0,target_reads=0,state_updates=0)
    if len(ids)!=d.COUNT or any(done.get(k)!=value for k,value in expected.items()):raise ValueError('All registered source oracle predictions required before TX labels')
    packets={}
    for view in d.VIEWS:
        with np.load(saved/(view+'.npz'),allow_pickle=False) as item:
            if set(item.files)!={'ids','seeds',*d.READOUTS} or not np.array_equal(item['ids'],ids) or item['seeds'].tolist()!=list(d.F_SEEDS):raise ValueError('Fixed source IDs/F seeds required')
            packets[view]={}
            for name in d.READOUTS:
                value=item[name]
                if value.shape!=(d.COUNT,3) or value.dtype.kind not in 'iu' or (value>5).any() or (value<0).any():raise ValueError('Complete legal source class predictions required')
                packets[view][name]=value.copy()
                iq=np.load(saved/(view+'_'+name+'_iq.npy'),mmap_mode='r',allow_pickle=False)
                if iq.shape!=(d.COUNT,2,256) or iq.dtype!=np.float32 or not np.isfinite(iq).all():raise ValueError('Complete sourcecounterfactualIQ required')
    return packets


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU source scorer required')
    ids,source,index,positions=source_inputs(args);selected=ids[positions]
    packets=completed_predictions(args.output,selected)
    # Labels are opened only by this independent child after all packets exist.
    labels=np.load(args.source_cache/'V/labels.npy',allow_pickle=False)[index[positions]]
    groups=[('overall','all',np.ones(d.COUNT,bool))]
    for dimension,values in (('TX',labels),('RX',source.rx[index[positions]]),('day',source.day[index[positions]])):
        groups.extend((dimension,str(key),values==key) for key in np.unique(values))
    clean=np.load(args.parent_cache/'validation/clean.npy',mmap_mode='r',allow_pickle=False)[positions]
    records=[];means={};waveform=[]
    for view in d.VIEWS:
        means[view]={}
        for name in d.READOUTS:
            value=np.load(args.output/'prediction'/(view+'_'+name+'_iq.npy'),mmap_mode='r',allow_pickle=False)
            waveform.append(dict(view=view,readout=name,metrics=waveform_gap(value,clean)))
            for column,seed in enumerate(d.F_SEEDS):
                for baseline in ('identity','parent_R03'):
                    for dimension,key,mask in groups:
                        records.append(dict(view=view,readout=name,F_seed=seed,baseline=baseline,dimension=dimension,value=key,
                            **classification_statistics(labels[mask],packets[view][name][mask,column],packets[view][baseline][mask,column])))
            rows=[r for r in records if r['view']==view and r['readout']==name and r['baseline']=='parent_R03' and r['dimension']=='overall']
            means[view][name]=dict(accuracy_mean=float(np.mean([r['accuracy'] for r in rows])),accuracy_std=float(np.std([r['accuracy'] for r in rows],ddof=1)),
                repaired_sum=sum(r['repaired'] for r in rows),damaged_sum=sum(r['damaged'] for r in rows),net_repaired_sum=sum(r['net_repaired'] for r in rows))
    write_json(args.output/'summary.json',dict(status='ANALYZED',source_only=True,role='source_V',count=d.COUNT,views=list(d.VIEWS),readouts=list(d.READOUTS),
        F_seeds=list(d.F_SEEDS),truth_last=True,independent_CPU_scorer=True,oracle_diagnostic_only=True,runtime_supported=False,
        formal_target_predictions=False,source_selection=False,target_reads=0,optimizer_updates=0,head_fit_updates=0,state_updates=0,
        source_TX_counts={str(key):int((labels==key).sum()) for key in np.unique(labels)},records=records,means=means,waveform=waveform,
        interpretation='Source-only diagnostic replacements use pairedclean;not runtimecorrection,not targetperformance or mathematicalupperbound'))


def run(args):
    if args.output.exists():raise FileExistsError('Immutable diagnostic output required')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned source diagnostic code/config commit required')
    source_inputs(args);args.output.mkdir(parents=True)
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,config=d.configuration()))
    try:
        import fcntl
        with ExitStack() as stack:
            for run in d.LOCK_RUNS:
                root=args.output.parent/run
                if not root.is_dir():raise ValueError('Existing own CPU coordination root required')
                handle=stack.enter_context((root/'score.lock').open('a'));fcntl.flock(handle,fcntl.LOCK_EX)
            print(json.dumps(dict(event='CPU_RESOURCE_GRANTED',locks=list(d.LOCK_RUNS),GPU_workers=0)),flush=True)
            frozen=smoke(args);predict(args,frozen);del frozen
            command=[sys.executable,'-u','-m','experiments.pamf_oracle_attribution.runner','score']
            for key in ('output','source_cache','parent_cache','parent_restoration','baseline_source_run'):command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_only=True,oracle_diagnostic_only=True,formal_target_predictions=False,target_reads=0,source_selection=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','score'))
    for name in ('output','source-cache','parent-cache','parent-restoration','baseline-source-run'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reader-root',type=Path);p.add_argument('--expected-commit');args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('This source-only diagnostic uses CPU only')
    torch.set_num_threads(2)
    if args.mode=='score':score(args)
    else:
        if args.reader_root is None or args.expected_commit is None:p.error('Original readers and codecommit required')
        run(args)


if __name__=='__main__':main()
