"""Synthetic execution benchmark only: no dataset, checkpoint, or experiment launch."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import cvsrffi.practical_adapter as adapter
from cvsrffi.practical_parallel import shutdown


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',required=True)
    p.add_argument('--model',action='store_true')
    p.add_argument('--scenes',default=','.join(adapter.PRACTICAL_ALL[1:]))
    a=p.parse_args()
    scenes=a.scenes.split(',')
    if not scenes or len(set(scenes))!=len(scenes) or not set(scenes)<=set(adapter.PRACTICAL_ALL[1:]):
        raise ValueError('Invalid synthetic benchmark scene list')
    destination=Path(a.output)
    if destination.exists():raise FileExistsError(destination)
    torch.set_num_threads(1)
    # Spawned workers inherit these limits before importing NumPy.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    device=torch.device(a.device)
    x=torch.randn(a.batch_size,2,256,generator=torch.Generator().manual_seed(41))
    gpu=x.to(device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def measure(fn):
        fn();times=[];out=None
        for _ in range(a.repeats):
            sync();start=time.perf_counter();out=fn();sync();times.append(time.perf_counter()-start)
        return dict(median_s=statistics.median(times),samples_s=times),out
    result=dict(scope='Synthetic execution only; no training dataset/checkpoint; not whole-epoch speedup',
        torch=torch.__version__,numpy=np.__version__,python=sys.version,
        platform=platform.platform(),device=str(device),batch_size=a.batch_size,repeats=a.repeats,
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,cases=[])
    adapter.set_evaluation_context([f'synthetic-{i}' for i in range(len(x))])
    with tempfile.TemporaryDirectory(prefix='cvs-practical-execution-') as cache:
        for route,eq,method in [('full',False,'zf'),('full',True,'zf'),('full',True,'mmse'),('residual',False,'zf')]:
            config=SimpleNamespace(practical_route=route,practical_equalization=eq,
                practical_equalizer_method=method,practical_fs_hz=25e6,practical_fc_hz=2.462e9,
                practical_receiver_seed=2027,practical_eval_cache_dir='',practical_channel_workers=0,
                practical_buffer_output=False)
            for scene in scenes:
                config.practical_eval_cache_dir=''
                config.practical_channel_workers=0
                config.practical_buffer_output=False
                gen=torch.Generator(device=device).manual_seed(7)
                def call(value=gpu):
                    y,_=adapter.apply_practical(value,scene,config,gen=gen)
                    return y.to(device)
                import leo_practical.channel as channel
                cached_kernel=channel.fractional_delay_kernel
                try:
                    channel.fractional_delay_kernel=channel._fractional_delay_kernel_reference
                    uncached_fir,original_iq=measure(call)
                finally:
                    channel.fractional_delay_kernel=cached_kernel
                base,reference=measure(call)
                assert torch.equal(original_iq,reference)
                assert adapter._last_meta['cache_event']=='bypass'
                config.practical_buffer_output=True
                buffered,y=measure(call);assert torch.equal(reference,y)
                config.practical_channel_workers=2
                parallel,y=measure(call);assert torch.equal(reference,y)
                assert adapter._last_meta['cache_event']=='bypass'
                config.practical_channel_workers=0
                config.practical_eval_cache_dir=cache
                sync();start=time.perf_counter();y=call();sync();cold=time.perf_counter()-start
                assert torch.equal(reference,y)
                cached_gpu,y=measure(call);assert torch.equal(reference,y)
                cached_cpu,y=measure(lambda:call(x));assert torch.equal(reference,y)
                assert adapter._last_meta['cache_event']=='hit'
                row=dict(route=route,equalization=eq,method=method,scene=scene,reference=base,
                    uncached_fir_reference=uncached_fir,
                    buffer=buffered,parallel2=parallel,fixed_cold_s=cold,
                    fixed_warm_existing_path=cached_gpu,fixed_warm_cpu_path=cached_cpu,exact_iq=True)
                result['cases'].append(row)
                print(json.dumps(dict(progress=len(result['cases']),route=route,scene=scene,
                    fir_cache_speedup=uncached_fir['median_s']/base['median_s'],
                    buffer_speedup=base['median_s']/buffered['median_s'],
                    parallel_speedup=base['median_s']/parallel['median_s'],
                    fixed_cpu_speedup=base['median_s']/cached_cpu['median_s'])),flush=True)
        config.practical_eval_cache_dir='';config.practical_buffer_output=False
        profile=cProfile.Profile();profile.enable();call();profile.disable()
        stats=pstats.Stats(profile)
        result['cpu_profile_top']=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],
            self_s=v[2],cumulative_s=v[3]) for k,v in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:25]]
    shutdown()
    if a.model:
        from scripts.train_rc4_practical import native,build_args
        from cvsrffi.a1_fast_runtime import GradientSnapshot
        args=build_args(ROOT/'configs/rc4_practical_full_noeq_20260918.json','unused','unused','unused','unused','unused','synthetic_benchmark')
        torch.manual_seed(23)
        ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},args,input_len=256,num_domains=15)
        model=native.build_baseline_model(native._apply_model_cli_args(ma,args),device)
        model.eval()
        with torch.no_grad():
            full,old=measure(lambda:model(gpu,y_tx=None,return_aux=True))
            identity,new=measure(lambda:model.forward_identity_only(gpu))
        exact=torch.equal(old['tx_logits'],new['tx_logits']) and torch.equal(old['z_id'],new['z_id'])
        assert exact
        model.train();out=model(gpu,y_tx=None,return_aux=True)
        (out['tx_logits'].square().mean()+out['dom_logits'].square().mean()).backward()
        def legacy():
            return (native._first_nonfinite_gradient(model),native._grad_norm(model),
                    native._grad_norm(model,lambda n:'backbone' in n),
                    native._grad_norm(model,lambda n:'aux' in n),
                    native._grad_norm(model,lambda n:'dom' in n or 'domain' in n))
        def snapshot():
            snap=GradientSnapshot.capture(model)
            return (snap.first_nonfinite(),snap.norm(),snap.norm(lambda n:'backbone' in n),
                    snap.norm(lambda n:'aux' in n),snap.norm(lambda n:'dom' in n or 'domain' in n))
        grad_old,g1=measure(legacy);grad_new,g2=measure(snapshot)
        assert all(l==r or (isinstance(l,float) and isinstance(r,float) and np.isnan(l) and np.isnan(r)) for l,r in zip(g1,g2))
        result['model']=dict(full_eval=full,identity_eval=identity,exact_identity=exact,
            gradient_legacy=grad_old,gradient_snapshot=grad_new,exact_gradient_observation=True,
            identity_speedup=full['median_s']/identity['median_s'],gradient_speedup=grad_old['median_s']/grad_new['median_s'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.writing')
    temporary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps(dict(complete=True,output=str(destination),cases=len(result['cases']),model=result.get('model'))),flush=True)


if __name__=='__main__':main()
