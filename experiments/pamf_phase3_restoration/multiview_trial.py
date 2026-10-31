"""Scratch single30 matched source-recipe control and larger prior; automatic Main tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .model import scratch_model
from .multiscale import scratch_multiscale
from .cache import tensor,load_F0,load_cache,close_cache,check_F0_state
from .source_pair import read,validate_source
from .objective import PairedObjective,feature_guide
from .train import learning_rate
from .source_inverse import align_source,R07
from .cost_probe import write,gpu_choice
from . import main_split_eval as main_eval

RUN='20261101-phase1-Restore-MSAT-M20-F0-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.multiview_trial'
ARMS=('TF_M20','MSAT_M20')
BASELINE_RUN='20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03'
STREAMS=('raw','DURA_v1','candidate')


def configuration(arm):
    if arm not in ARMS:raise ValueError('Only two fixed matched arms allowed')
    return dict(run_id=RUN,arm=arm,model_seed=2026092701,data_seed=2026110102,
        epochs=30,updates=7410,batch=256,training_stages=1,initialization='scratch',ancestor_checkpoints=[],
        architecture='TF196898' if arm=='TF_M20' else 'MSAT64-128-192-256/4attention8heads/full-rate-skips',
        source_roles={'L_s':6300,'U_s':56700},source_cache=R07,source_draws=list(range(1,21))+list(range(1,11)),
        source_recipe='R07/25MHz2.462GHz/post_sync/noEQ/reflect/refSNR30/15percentclean+sixuniform',
        objective='IQNMSE+.05STFT+.1actualF0fused160cos/perrecordIQgradientcap.5',
        optimizer='AdamW',lr=.001,minimum_lr=2e-5,warmup_updates=247,weight_decay=1e-4,gradient_clip=1.,
        checkpoint_selection='final30_only',TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,
        source_F0_knowledge='original3ff8/TriSIGTIME_SSL/E50-11100/defaulthead',default_DURA_changed=False)


def factory(arm):
    configuration(arm)
    return (scratch_model if arm=='TF_M20' else scratch_multiscale)(2026092701)


def save(path,G,arm,commit,epoch):
    if epoch not in (0,30):raise ValueError('Only scratch smoke or final30 states allowed')
    state={k:v.detach().cpu().clone() for k,v in G.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite learned G state')
    with path.open('xb') as f:torch.save(dict(config=configuration(arm),code_commit=commit,epoch=epoch,
        updates=0 if epoch==0 else 7410,state=state),f)


def load(path,arm,commit,device,epoch=30):
    if epoch not in (0,30):raise ValueError('Only E0 or fixed final30 allowed')
    p=torch.load(path,map_location='cpu',weights_only=False)
    if (set(p)!={'config','code_commit','epoch','updates','state'} or p['config']!=configuration(arm)
            or p['code_commit']!=commit or p['epoch']!=epoch or p['updates']!=(0 if epoch==0 else 7410)):
        raise ValueError('Exact scratch30 source-only G state required')
    G=factory(arm)
    if (set(p['state'])!=set(G.state_dict()) or any(v.dtype!=torch.float32 or not torch.isfinite(v).all()
            or v.shape!=G.state_dict()[k].shape for k,v in p['state'].items())):raise ValueError('Exact finite FP32 G state required')
    G.load_state_dict(p['state'],strict=True)
    return G.to(device).eval().requires_grad_(False)


def source(args):
    old=args.root/'runs'/R07/'cache';done=read(old/'complete.json')
    if (done.get('status'),done.get('train_count'),done.get('epochs'),done.get('TX_labels_read'),done.get('target_reads'))!=(
            'ACTION_SOURCE_CACHE_COMPLETE',63000,20,0,0):raise ValueError('Original complete R07 train cache required')
    ids=np.load(old/'train/physical_ids.npy',allow_pickle=False);roles=np.load(old/'train/roles.npy',allow_pickle=False)
    b=load_cache(args.cache,guided=False)
    try:align_source(ids,roles,b['ids'],b['roles'])
    finally:close_cache(b)
    return old,roles


def train(args,out):
    start=time.perf_counter();c=validate_source(read(args.inputs));old,roles=source(args)
    F,identity=load_F0(c,'cuda:0');main_eval.valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in F.state_dict().items()}
    clean=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if clean.shape!=(63000,2,256) or clean.dtype!=np.float32:raise ValueError('Original source clean FP32 required')
    try:y=tensor(clean,'cuda:0')
    finally:clean._mmap.close()
    if y.shape!=(63000,2,256) or not torch.isfinite(y).all():raise ValueError('Original clean source finite shape required')
    out.mkdir();write(out/'config.json',dict(configuration(args.arm),code_commit=args.commit,F0_identity=identity))
    G=factory(args.arm).to('cuda:0');objective=PairedObjective().to('cuda:0')
    # First neural work: actual source2 checkpoint/interface/input-gradient smoke.
    x=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:smoke_x=tensor(x[:2],'cuda:0')
    finally:x._mmap.close()
    before={k:v.detach().clone() for k,v in G.state_dict().items()}
    with torch.no_grad():target=F.features(y[:2]).detach()
    restored=G(smoke_x);torch.testing.assert_close(restored,smoke_x,atol=0,rtol=0)
    physical,_=objective(restored,y[:2]);guide,_=feature_guide(F,restored,y[:2],target)
    (physical+guide).mean().backward()
    if not G.output.weight.grad.abs().sum()>0:raise ValueError('Actual source objective G output gradient missing')
    G.zero_grad(set_to_none=True)
    if any(not torch.equal(v,G.state_dict()[k]) for k,v in before.items()):raise ValueError('Source smoke changed G')
    check_F0_state(F,frozen);save(out/'smoke_E0.pt',G,args.arm,args.commit,0)
    probe=load(out/'smoke_E0.pt',args.arm,args.commit,'cuda:0',epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(smoke_x),restored,atol=0,rtol=0)
    del probe
    write(out/'smoke.json',dict(status='PASS',original_source_rows=2,actual_F0_input_gradient=True,
        random_G_identity=True,strict_checkpoint_reload=True,optimizer_updates=0,TX_labels_read=0,target_inputs=0))
    print('MSAT_M20_REAL_SOURCE_SMOKE_PASS',args.arm,flush=True)
    cache_start=time.perf_counter()
    with torch.no_grad():features=torch.cat([F.features(y[i:i+256]).detach() for i in range(0,63000,256)])
    torch.cuda.synchronize();cache_seconds=time.perf_counter()-cache_start
    G.train();optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(2026110102);orders=np.stack([rng.permutation(63000) for _ in range(30)]).astype(np.int32)
    with (out/'data_schedule.npz').open('xb') as f:np.savez(f,orders=orders,draws=configuration(args.arm)['source_draws'])
    update=0;train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch,draw in enumerate(configuration(args.arm)['source_draws'],1):
            begin=time.perf_counter();folder=old/'epochs'/f'{draw:02d}';m=read(folder/'complete.json')
            if (m.get('count'),m.get('TX_labels_read'),m.get('target_reads'))!=(63000,0,0):raise ValueError('Complete source draw required')
            a=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
            if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source received FP32 required')
            try:x=tensor(a,'cuda:0')
            finally:a._mmap.close()
            if x.shape!=y.shape or not torch.isfinite(x).all():raise ValueError('Finite aligned source received required')
            sums=dict(loss=0.,iq_nmse=0.,spectral_error=0.,feature_error=0.,guide_weight=0.)
            for offset in range(0,63000,256):
                take=torch.tensor(orders[epoch-1,offset:offset+256].tolist(),device='cuda:0',dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                restored=G(x[take]);loss,terms=objective(restored,y[take]);guide,info=feature_guide(F,restored,y[take],features[take]);loss=loss+guide
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite source loss; preserve run')
                loss.mean().backward();torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();update+=1
                for key,value in [('loss',loss),*terms.items(),('feature_error',info['feature_error']),('guide_weight',info['weight'])]:
                    sums[key]+=float(value.detach().sum())
                if 'feature_standardized_error' in info:
                    sums.setdefault('feature_standardized_error',0.)
                    sums['feature_standardized_error']+=float(info['feature_standardized_error'].detach().sum())
            check_F0_state(F,frozen);torch.cuda.synchronize()
            row=dict(epoch=epoch,draw=draw,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,
                seconds=time.perf_counter()-begin,last_learning_rate=learning_rate(update),metrics={k:v/63000 for k,v in sums.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');print('MSAT_M20_EPOCH',args.arm,json.dumps(row),flush=True)
    if update!=7410:raise ValueError('Full fixed30 source budget required')
    save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=1890000,
        F0_clean_target_rows=63000,source_F0_feature_cache_seconds=cache_seconds,F0_identity=identity,
        G_parameters=sum(p.numel() for p in G.parameters()),training_seconds=time.perf_counter()-train_start,
        worker_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,
        checkpoint_selection='final30_only',model_state_F0_updates=0))


def predict(args,out):
    c=validate_source(read(args.inputs));F,identity=load_F0(c,'cuda:0');main_eval.valid_identity(identity)
    from experiments.cric_dura_comparison.evaluation import handle
    row=args.root/'runs'/RUN/'rows'/args.arm
    G=load(row/'training/final.pt',args.arm,args.commit,'cuda:0')
    models=[F,G];before=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
    old,_=source(args);a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:x=tensor(a[:2],'cuda:0')
    finally:a._mmap.close()
    out.mkdir()
    with torch.inference_mode():
        restored=G(x);logits=F.classify_features(F.features(restored))
        if not torch.isfinite(restored).all() or logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Actual frozen finalG/F0 source smoke failed')
    write(out/'smoke.json',dict(status='PASS',source_rows=2,TX_labels_read=0,target_inputs=0,model_updates=0))
    o=main_eval.official(args.root);manifest,ids,positions=handle(o['tests'][args.task]['views_root'],o['classes'])
    if len(ids)!=main_eval.COUNTS[args.task]:raise ValueError('Full original Main population required')
    base=args.root/'runs'/BASELINE_RUN/'prediction'/args.task
    base_done=read(base/'complete.json');main_eval.valid_identity(base_done['F0_identity'])
    if (base_done['status'],base_done['count'],base_done['model_state_updates'])!=('PREDICTIONS_COMPLETE',len(ids),0):raise ValueError('Exact frozen baseline predictions required')
    with np.load(base/'predictions.npz',allow_pickle=False) as f:
        if not np.array_equal(f['ids'],ids):raise ValueError('Baseline physical order differs')
        values={name+'__'+view:f[name+'__'+view].copy() for name in ('raw','DURA_v1') for view in main_eval.VIEWS}
    for view in main_eval.VIEWS:
        path=manifest['clean_ref'] if view=='clean' else str(Path(manifest['observations_root'])/(view+'.npy'))
        a=np.load(path,mmap_mode='r',allow_pickle=False);decisions=[]
        if a.ndim!=3 or a.shape[1:]!=(2,256) or a.dtype!=np.float32:raise ValueError('Exact original Main FP32 input required')
        try:
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    take=slice(start,start+128) if positions is None else positions[start:start+128]
                    restored=G(tensor(a[take],'cuda:0'));logits=F.classify_features(F.features(restored))
                    if not torch.isfinite(restored).all() or not torch.isfinite(logits).all():raise ValueError('Invalid frozen candidate IQ/logits')
                    decisions.extend(logits.argmax(1).cpu().tolist())
        finally:a._mmap.close()
        values['candidate__'+view]=np.asarray(decisions,np.int64)
        print('MSAT_M20_MAIN_VIEW_COMPLETE',args.arm,args.task,view,len(ids),flush=True)
    for m,snapshot in zip(models,before):check_F0_state(m,snapshot)
    with (out/'predictions.npz').open('xb') as f:np.savez(f,ids=ids,**values)
    write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',run_id=RUN,arm=args.arm,task=args.task,
        count=len(ids),code_commit=args.commit,F0_identity=identity,readouts=list(STREAMS),views=list(main_eval.VIEWS),
        baseline_reused_from=BASELINE_RUN,truth_read=False,query_fit=False,model_state_updates=0))


def score(args,out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU truth scorer required')
    row=args.root/'runs'/RUN/'rows'/args.arm;tests={}
    for task,n in main_eval.COUNTS.items():
        m=read(row/'prediction'/task/'complete.json');main_eval.valid_identity(m['F0_identity'])
        required=dict(status='PREDICTIONS_COMPLETE',run_id=RUN,arm=args.arm,task=task,count=n,code_commit=args.commit,
            readouts=list(STREAMS),views=list(main_eval.VIEWS),truth_read=False,query_fit=False,model_state_updates=0)
        if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Every full Main task/view must finish before truth')
        tests[task]=dict(count=n,views=list(main_eval.VIEWS),readouts=list(STREAMS))
    write(row/'prediction/complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=RUN,row_id=args.arm,
        tests=tests,views=list(main_eval.VIEWS),readouts=list(STREAMS),truth_read=False,query_fit=False,code_commit=args.commit))
    import fcntl
    with (args.root/'runs/20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01/score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        config=main_eval.official(args.root,'score');config.update(run_id=RUN,row_id=args.arm,readouts=list(STREAMS),
            prediction_output=str(row/'prediction'),output_root=str(out))
        from experiments.pamf_repl_view_shuffle.evaluation import score as original_score
        original_score(config)
    scores=read(out/'scores.json')
    write(row/'results.json',dict(status='ANALYZED',run_id=RUN,arm=args.arm,code_commit=args.commit,
        results=main_eval.summary(scores,streams=STREAMS),truth_last=True,default_DURA_changed=False,goal_complete=False))


def command(args,mode,arm,task=None):
    result=[sys.executable,'-u','-m',MODULE,mode,'--root',str(args.root),'--commit',args.commit,
        '--inputs',str(args.inputs),'--cache',str(args.cache),'--arm',arm]
    return result+(['--task',task] if task else [])


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir();(out/'rows').mkdir()
    for arm in ARMS:(out/'rows'/arm).mkdir();(out/'rows'/arm/'prediction').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    stages=['train','cross_rx','cross_both','score'];position={arm:0 for arm in ARMS};active={};failed=[]
    while any(p<4 for p in position.values()) or active:
        for arm,(p,gpu,stage) in list(active.items()):
            code=p.poll()
            if code is not None:
                write(out/('terminal-'+arm+'-'+stage+'.json'),dict(pid=p.pid,GPU=gpu,exit_code=code))
                if code:failed.append((arm,stage));position[arm]=4
                else:position[arm]+=1
                del active[arm]
        choices=[g for g in gpu_choice() if g['GPU'] not in [gpu for _,gpu,_ in active.values() if gpu is not None]]
        for arm,index in position.items():
            if arm in active or index==4:continue
            stage=stages[index]
            if stage=='score':
                if any(g is None for _,g,_ in active.values()):continue
                gpu=None;admission=None;cmd=command(args,'score',arm)
            else:
                if not choices or sum(g is not None for _,g,_ in active.values())>=2:continue
                admission=choices.pop(0);gpu=admission['GPU']
                cmd=command(args,'train' if stage=='train' else 'predict',arm,None if stage=='train' else stage)
            with (out/'logs'/(arm+'-'+stage+'.log')).open('xb') as log:
                p=subprocess.Popen(cmd,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[arm]=(p,gpu,stage)
            write(out/('submitted-'+arm+'-'+stage+'.json'),dict(pid=p.pid,command=cmd,CWD=os.getcwd(),admission=admission))
        if active or any(p<4 for p in position.values()):time.sleep(5)
    if failed:raise RuntimeError('Preserved failed rows: '+str(failed))
    for arm in ARMS:
        if read(out/'rows'/arm/'results.json')['status']!='ANALYZED':raise ValueError('Missing full row results')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',all_rows_trained30_and_main_seven_scored=True,
        default_DURA_changed=False,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','train','predict','score']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--task',choices=tuple(main_eval.COUNTS));a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git execution identity required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.mode=='run':controller(a);return
    if not a.arm:p.error('--arm required')
    row=a.root/'runs'/RUN/'rows'/a.arm
    if a.mode=='train':train(a,row/'training')
    elif a.mode=='predict':
        if not a.task:p.error('--task required')
        predict(a,row/'prediction'/a.task)
    else:score(a,row/'score')


if __name__=='__main__':main()
