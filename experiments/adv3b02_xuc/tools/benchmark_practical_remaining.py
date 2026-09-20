"""Paired synthetic ablations: no target truth, checkpoint or experiment launch."""
import argparse, cProfile, json, os, pstats, statistics, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from leo_practical import Config, SCENARIOS, apply_leo_practical_channel_batch as batch
from leo_practical.execution import FAST, REFERENCE
from cvsrffi.practical_parallel import parallel_batch, shutdown
from cvsrffi.practical_adapter import _output_tensor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(1)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    device=torch.device(a.device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def measure(fn):
        fn(); samples=[]
        for _ in range(a.repeats):
            sync();start=time.perf_counter();out=fn();sync();samples.append(time.perf_counter()-start)
        return dict(median_s=statistics.median(samples),samples_s=samples),out
    x=np.random.default_rng(41).normal(size=(a.batch_size,2,256))
    kw=dict(seed=392005,sample_ids=[f'synthetic{i}' for i in range(len(x))],session_ids=['rx']*len(x),
        realization_namespace='source_dynamic_E1_L')
    result=dict(scope='synthetic component and pipeline benchmark, not whole epoch',
        numpy=np.__version__,torch=torch.__version__,device=str(device),batch_size=len(x),cases=[])
    for route,eq,method in [('full',False,'zf'),('full',True,'zf'),('full',True,'mmse'),('residual',False,'zf')]:
        for scene in SCENARIOS:
            cfg=Config(fs_hz=25e6,fc_hz=2.462e9,scenario=scene,processing_route=route,
                equalization_enabled=eq,equalizer_method=method,zf_regularization=1e-6,equalizer_max_gain_db=20)
            baseline,old=measure(lambda:batch(x,cfg,**kw))
            row=dict(route=route,eq=eq,method=method,scene=scene,baseline=baseline,ablations={})
            for name,option in [(key,replace(REFERENCE,**{key:True})) for key in REFERENCE.__dataclass_fields__]+[('combined',FAST)]:
                timing,new=measure(lambda:batch(x,cfg,execution=option,**kw))
                assert np.array_equal(old[0],new[0]) and np.array_equal(old[2],new[2])
                for before,after in zip(old[1],new[1]):
                    if not option.light_metadata:assert before==after
                    else:
                        for k in after.keys()-{'metadata_level','geometry'}:assert before[k]==after[k]
                        for k in after['geometry']:assert before['geometry'][k]==after['geometry'][k]
                row['ablations'][name]=dict(**timing,speedup=baseline['median_s']/timing['median_s'],exact=True)
            timing,new=measure(lambda:parallel_batch(x,cfg,workers=2,execution=FAST,**kw))
            assert np.array_equal(old[0],new[0])
            row['ablations']['combined_parallel2']=dict(**timing,speedup=baseline['median_s']/timing['median_s'],exact=True)
            result['cases'].append(row)
            print(json.dumps(dict(progress=len(result['cases']),route=route,scene=scene,
                combined=row['ablations']['combined']['speedup'])),flush=True)
    # Profile optimized residual to identify the next real bottleneck.
    profile=cProfile.Profile();profile.enable();batch(x,cfg,execution=FAST,**kw);profile.disable()
    result['profile_top']=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_s=v[2],cumulative_s=v[3])
        for k,v in sorted(pstats.Stats(profile).stats.items(),key=lambda kv:kv[1][3],reverse=True)[:30]]
    from scripts.train_rc4_practical import native,build_args
    args=build_args(ROOT/'configs/rc4_practical_full_noeq_20260918.json','unused','unused','unused','unused','unused','synthetic_benchmark')
    torch.manual_seed(23)
    ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},args,input_len=256,num_domains=15)
    model=native.build_baseline_model(native._apply_model_cli_args(ma,args),device).eval()
    gpu=_output_tensor(x,device,torch.float32,True)
    domains=torch.arange(len(x),device=device)%15
    from cvsrffi.source_validation_reuse import SourceValidationReuse
    with torch.no_grad():
        ref=model(gpu,y_tx=None,return_aux=True,domain_labels=domains)['z_id']
        def twice():
            model(gpu,y_tx=None,return_aux=True)
            return model(gpu,y_tx=None,return_aux=True,domain_labels=domains)['z_id']
        def reuse():
            cache=SourceValidationReuse(model)
            out=model(gpu,y_tx=None,return_aux=True)
            cache.record(0,gpu,out['z_id']);z=cache.take(0,gpu)
            assert cache.hits==1
            return z
        legacy,_=measure(twice);new,z=measure(reuse);assert torch.equal(ref,z)
        result['validation_reuse']=dict(baseline=legacy,optimized=new,speedup=legacy['median_s']/new['median_s'],exact=True)
        # Real CPU channel producer and CVS GPU consumer, one bounded task ahead.
        from cvsrffi.bounded_prefetch import prefetch_map
        def produce(index):
            return batch(x,cfg,execution=FAST,**dict(kw,seed=392005+index))[0]
        def pipeline(prefetch):
            sequence=prefetch_map(produce,range(4)) if prefetch else map(produce,range(4))
            return [model(_output_tensor(y,device,torch.float32,True),y_tx=None,return_aux=True)['tx_logits'].cpu() for y in sequence]
        before,old=measure(lambda:pipeline(False));after,new=measure(lambda:pipeline(True))
        assert all(torch.equal(u,v) for u,v in zip(old,new))
        result['prefetch']=dict(baseline=before,optimized=after,speedup=before['median_s']/after['median_s'],exact=True,
            integration='fixed practical predictor; cross-training-batch prefetch remains disabled')
    shutdown()
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
