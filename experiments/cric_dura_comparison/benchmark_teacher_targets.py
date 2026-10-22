"""Real source checkpoints, synthetic IQ, no-query acceleration falsification."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .io import write,tensor
from .g_lineage import STAGES,load,construct,trainable,objective
from .g_train import batches,learning_rate
from .teacher_targets import EpochTargets,SharedEpochTargets,produce_shared_epoch,frozen_prefix_predictor


def vector(parameters,grad=False):
    return torch.cat([(p.grad if grad and p.grad is not None else torch.zeros_like(p) if grad else p).detach().reshape(-1).cpu() for p in parameters])


def difference(a,b):
    delta=(a.double()-b.double())
    return dict(max_abs=float(delta.abs().max()),relative_L2=float(delta.norm()/a.double().norm().clamp_min(1e-12)))


def execute(initial,teacher,stage,clean,received,plan,cached,device,capture=False,shared=None):
    model=copy.deepcopy(initial).to(device).train();parameters=trainable(model,stage)
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    generator=torch.Generator(device=device).manual_seed(2026101919)
    cache=None
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    if shared is not None:
        cache=shared
    elif cached:
        if stage in ('action','r17'):
            prefix,predict=frozen_prefix_predictor(model,stage)
            cache=EpochTargets(prefix,len(clean),1,{'diagnostic':'synthetic-epoch1','stage':stage},predict=predict)
        else:
            predict=(lambda x:teacher.stage_outputs(x)[0][1:]) if stage=='r69' else None
            cache=EpochTargets(teacher,len(clean),1,{'diagnostic':'synthetic-epoch1','stage':stage},predict=predict)
    losses=[];first_grad=None
    for step,selected in enumerate(plan):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=learning_rate(step)
        finite=torch.ones((),dtype=torch.bool,device=device)
        total=torch.zeros((),dtype=torch.float64,device=device)
        micro=64 if stage in ('r38','r44','r46') else 256
        arrays_clean=clean[selected];arrays_received=received[selected]
        for offset in range(0,len(selected),micro):
            take=slice(offset,offset+micro)
            x=tensor(arrays_received[take],device);y=tensor(arrays_clean[take],device)
            desired=cache.get(selected[take],x,1) if cache is not None else None
            extra=dict(frozen_context=desired) if stage in ('action','r17') else dict(desired=desired)
            if stage=='action':
                kernel=torch.zeros(len(x),2,64,device=device);kernel[:,0,24]=1
                coefficients=torch.zeros(len(x),4,256,device=device);coefficients[:,0]=1
                extra['action_arrays']=(kernel,coefficients,torch.full((len(x),),1e-3,device=device))
            value=objective(model,teacher,stage,x,y,1,generator=generator,
                            skip_zero_spectral=cached,**extra)
            if cached:finite.logical_and_(torch.isfinite(value).all())
            elif not bool(torch.isfinite(value).all()):raise FloatingPointError('Online objective nonfinite')
            (value.sum()/len(selected)).backward()
            if cached:total+=value.detach().sum().double()
            else:total+=float(value.detach().sum())
        if cached and not bool(finite):raise FloatingPointError('Cached objective nonfinite')
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        if capture and step==0:first_grad=vector(parameters,True)
        optimizer.step()
        if capture:losses.append(float(total)/len(selected))
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    result=dict(seconds=seconds,updates=len(plan),exposures=sum(map(len,plan)),micro_batch=micro,
        extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-baseline,
        cache=cache.stats() if cache else None)
    if capture:result.update(weights=vector(parameters),first_grad=first_grad,losses=torch.tensor(losses))
    del model,optimizer,parameters,cache
    torch.cuda.empty_cache()
    return result


def run(fixture,output,device='cuda:0',steps=12,rounds=2,stages=('r38','r44','r46','r65','r69','r17')):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from .bootstrap import activate,numerical_policy
    root=Path(__file__).resolve().parents[2]
    sources={k:str(root/'local_artifacts/cric_dura_comparison_20261021/frozen_sources'/k) for k in ('F','AF','BASE','FOLD','G')}
    activate(dict(frozen_sources=sources));numerical_policy()
    torch.cuda.set_per_process_memory_fraction(.60)
    config=dict(status='SOURCE_ONLY_NO_QUERY_DIAGNOSTIC',device=torch.cuda.get_device_name(0),torch=torch.__version__,
        input='synthetic IQ, fixed630 pool; existing L-only source checkpoints',steps=steps,rounds=rounds,
        precision='FP32 TF32off',teacher_atol=1e-6,teacher_rtol=1e-5,
        gradient_max_abs=3e-5,gradient_relative_L2=1e-4,weight_max_abs=3e-5,weight_relative_L2=1e-4,
        target_reads=0,F_reads=0,TX_truth_reads=0,
        timing_scope='complete updates including target miss/build/gather/transfer and original optimizer; fixture generation/model load excluded; local shared desktop GPU, not N607 or end-to-end E20 speed')
    write(output/'config.json',config)
    fixture=Path(fixture)
    payload=torch.load(fixture/'r17.pt',map_location='cpu',weights_only=False)
    contract=payload['source_contract'];roles=payload['allowed_roles']
    if roles!='L':raise ValueError('Registered small L-only fixture required')
    rng=np.random.default_rng(2026102201)
    clean=rng.standard_normal((630,2,256)).astype(np.float32)
    clean/=np.sqrt(np.mean(clean**2,axis=(1,2),keepdims=True)+1e-8)
    received=(clean+.10*rng.standard_normal(clean.shape)).astype(np.float32)
    plan=list(batches(630,np.random.default_rng(2026102001)))[:steps]
    results=[]
    for stage in stages:
        parent_stage=STAGES[STAGES.index(stage)-1]
        parent,_=load(fixture/(parent_stage+'.pt'),contract,roles)
        torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
        initial=construct(stage,parent)
        teacher=parent.to(device).eval().requires_grad_(False) if stage not in ('action','r17') else None
        sample=tensor(received[:64],device)
        if teacher is not None:
            with torch.no_grad():expected=tuple(teacher.stage_outputs(sample)[0])
            cache=EpochTargets(teacher,630,1,{'diagnostic':'synthetic-epoch1'})
            got=cache.get(np.arange(64),sample)
            target_error=max(difference(a,b)['max_abs'] for a,b in zip(expected,got))
            for a,b in zip(expected,got):torch.testing.assert_close(a,b,atol=config['teacher_atol'],rtol=config['teacher_rtol'])
            del cache,expected,got
        else:target_error=None
        online=execute(initial,teacher,stage,clean,received,plan[:min(6,steps)],False,device,True)
        cached=execute(initial,teacher,stage,clean,received,plan[:min(6,steps)],True,device,True)
        gradient=difference(online['first_grad'],cached['first_grad']);weights=difference(online['weights'],cached['weights'])
        loss_error=difference(online['losses'],cached['losses'])
        entry=dict(stage=stage,target_max_abs=target_error,gradient=gradient,weights=weights,loss=loss_error,measurements=[])
        for name,metric in (('gradient',gradient),('weight',weights)):
            if metric['max_abs']>config[name+'_max_abs'] or metric['relative_L2']>config[name+'_relative_L2']:
                write(output/'failed_equivalence.json',entry);raise ValueError(stage+' '+name+' equivalence differs')
        torch.testing.assert_close(online['losses'],cached['losses'],atol=2e-6,rtol=2e-5)
        del online,cached
        for round_index in range(rounds):
            pair={}
            for use_cache in ((False,True) if round_index%2==0 else (True,False)):
                pair['cached' if use_cache else 'online']=execute(initial,teacher,stage,clean,received,plan,use_cache,device)
            pair['speedup']=pair['online']['seconds']/pair['cached']['seconds'];entry['measurements'].append(pair)
        entry['status']='EQUIVALENCE_AND_TIMING_COMPLETE';results.append(entry)
        write(output/(stage+'.json'),entry)
        print(json.dumps(dict(stage=stage,status=entry['status'],speedups=[v['speedup'] for v in entry['measurements']],gradient=gradient,weights=weights)),flush=True)
        del initial,teacher,parent;torch.cuda.empty_cache()
    write(output/'complete.json',dict(status='SOURCE_ONLY_BENCHMARK_COMPLETE',config=config,results=results))


def run_shared(fixture,output,device='cuda:0',steps=12):
    from .bootstrap import activate,numerical_policy
    root=Path(__file__).resolve().parents[2]
    activate(dict(frozen_sources={k:str(root/'local_artifacts/cric_dura_comparison_20261021/frozen_sources'/k) for k in ('F','AF','BASE','FOLD','G')}));numerical_policy()
    torch.cuda.set_per_process_memory_fraction(.60)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    payload=torch.load(Path(fixture)/'r65.pt',map_location='cpu',weights_only=False)
    teacher,_=load(Path(fixture)/'r65.pt',payload['source_contract'],payload['allowed_roles'],device)
    teacher.eval().requires_grad_(False)
    rng=np.random.default_rng(2026102201)
    clean=rng.standard_normal((630,2,256)).astype(np.float32);clean/=np.sqrt(np.mean(clean**2,axis=(1,2),keepdims=True)+1e-8)
    received=(clean+.1*rng.standard_normal(clean.shape)).astype(np.float32)
    class SyntheticView:
        ids=np.asarray(['synthetic-%04d'%i for i in range(630)])
        def __len__(self):return 630
        def ready(self):return True
        def batch(self,selected,epoch):return dict(received=received[selected])
    view=SyntheticView();identity=dict(diagnostic='synthetic-only-not-a-science-cache',teacher=str(Path(fixture)/'r65.pt'),epoch=1)
    produce_shared_epoch(output/'targets',identity,view,teacher,1,batch_size=64)
    creation=json.loads((output/'targets/epoch01/complete.json').read_text())
    trials=[]
    for seed in (2026092701,2026092702):
        torch.manual_seed(seed);initial=construct('r69',teacher)
        teacher.to(device).eval().requires_grad_(False)
        plan=list(batches(630,np.random.default_rng(2026102001+seed-2026092701)))[:steps]
        reader=SharedEpochTargets(output/'targets',identity,view.ids,1)
        a=execute(initial,teacher,'r69',clean,received,plan[:6],False,device,True)
        b=execute(initial,teacher,'r69',clean,received,plan[:6],True,device,True,shared=reader)
        for key in ('weights','first_grad','losses'):torch.testing.assert_close(a[key],b[key],atol=3e-5,rtol=1e-4)
        online=execute(initial,teacher,'r69',clean,received,plan,False,device)
        reader=SharedEpochTargets(output/'targets',identity,view.ids,1)
        cached=execute(initial,teacher,'r69',clean,received,plan,True,device,shared=reader)
        trials.append(dict(seed=seed,online=online,shared=cached))
        print(json.dumps(dict(event='SHARED_STUDENT_COMPLETE',seed=seed,online=online['seconds'],shared=cached['seconds'])),flush=True)
        del a,b,initial;torch.cuda.empty_cache()
    before=sum(v['online']['seconds'] for v in trials)
    after=creation['seconds']+sum(v['shared']['seconds'] for v in trials)
    write(output/'complete.json',dict(status='SHARED_TWO_STUDENT_DIAGNOSTIC_COMPLETE',input='synthetic630; fixed real R65 L-only teacher',
        producer=creation,students=trials,online_total_seconds=before,shared_total_including_producer=after,speedup=before/after,
        teacher_samples_computed_once=630,no_query=True,no_F=True,no_TX_truth=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=12);parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--stages',nargs='+',default=['r38','r44','r46','r65','r69','r17'])
    parser.add_argument('--shared',action='store_true')
    args=parser.parse_args()
    if args.shared:run_shared(args.fixture,args.output,steps=args.steps)
    else:run(args.fixture,args.output,steps=args.steps,rounds=args.rounds,stages=tuple(args.stages))
