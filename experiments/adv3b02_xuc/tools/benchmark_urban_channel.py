"""Paired CPU timing and synthetic channel diagnostics; no IQ dataset or model."""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,collections,json,os,platform,statistics,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from leo_practical import Config,SCENARIOS,apply_leo_practical_channel_batch as batch
from leo_practical.execution import FAST,REFERENCE

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--diagnostic-samples',type=int,default=1024)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    if a.repeats<3 or a.batch_size<1 or a.diagnostic_samples<128:raise ValueError('invalid benchmark budget')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    x=np.random.default_rng(41).normal(size=(a.batch_size,2,256))
    x/=np.sqrt(np.mean(np.sum(x*x,axis=1),axis=1))[:,None,None]
    kw=dict(seed=392005,sample_ids=[f'synthetic{i}' for i in range(len(x))],session_ids=['synthetic_rx']*len(x),realization_namespace='source_dynamic_E91_L',receiver_seed=2027)
    options={'reference':REFERENCE,'previous_fast':replace(FAST,lazy_rng=False),'lazy_fast':FAST}
    result={'scope':'synthetic CPU component timing; not whole training and not RFF accuracy','host':platform.node(),'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'batch_size':len(x),'repeats':a.repeats,'cases':[],'physics':[]}
    for route,eq,method in [('full',False,'zf'),('full',True,'zf'),('full',True,'mmse'),('residual',False,'zf')]:
        for scene in SCENARIOS:
            cfg=Config(fs_hz=25e6,fc_hz=2.462e9,scenario=scene,processing_route=route,equalization_enabled=eq,equalizer_method=method,zf_regularization=1e-6,equalizer_max_gain_db=20)
            reference=batch(x,cfg,execution=REFERENCE,**kw)
            for opt in options.values():
                new=batch(x,cfg,execution=replace(opt,light_metadata=False),**kw)
                assert np.array_equal(reference[0],new[0]) and np.array_equal(reference[2],new[2]) and reference[1]==new[1]
            times={k:[] for k in options};names=list(options)
            # Rotate order to reduce systematic warm-up/order bias.
            for repeat in range(a.repeats):
                for name in names[repeat%3:]+names[:repeat%3]:
                    start=time.perf_counter();batch(x,cfg,execution=options[name],**kw);times[name].append(time.perf_counter()-start)
            med={k:statistics.median(v) for k,v in times.items()}
            result['cases'].append({'route':route,'equalization':eq,'method':method,'scene':scene,'samples_s':times,'median_s':med,'new_incremental_speedup':med['previous_fast']/med['lazy_fast'],'combined_vs_reference':med['reference']/med['lazy_fast'],'exact_iq_metadata_state':True})
            print(json.dumps({'case':len(result['cases']),'route':route,'scene':scene,'new_incremental_speedup':med['previous_fast']/med['lazy_fast']}),flush=True)
    # Same synthetic unit-power input and seeds across scenes. This isolates
    # generated-channel differences, not discriminability of real TX identities.
    n=a.diagnostic_samples;y=np.random.default_rng(812).normal(size=(n,2,256))
    y/=np.sqrt(np.mean(np.sum(y*y,axis=1),axis=1))[:,None,None]
    for scene in SCENARIOS:
        cfg=Config(fs_hz=25e6,fc_hz=2.462e9,scenario=scene,processing_route='residual',equalization_enabled=False)
        _,meta,_=batch(y,cfg,execution=replace(FAST,light_metadata=False),seed=392005,sample_ids=[f'physics{i}' for i in range(n)],session_ids=['synthetic_rx']*n,realization_namespace='source_physics_diagnostic',receiver_seed=2027)
        values={key:[float(m[key]) for m in meta] for key in ['quality_snr_db','signal_to_added_noise_db','output_frequency_hz','residual_kernel_energy']}
        stats={k:{'q05':float(np.quantile(v,.05)),'median':float(np.median(v)),'q95':float(np.quantile(v,.95)),'mean':float(np.mean(v))} for k,v in values.items()}
        stats['signal_to_added_noise_db']['fraction_below_0']=float(np.mean(np.array(values['signal_to_added_noise_db'])<0))
        stats['signal_to_added_noise_db']['fraction_below_10']=float(np.mean(np.array(values['signal_to_added_noise_db'])<10))
        result['physics'].append({'scene':scene,'count':n,'config':asdict(cfg),'state_counts':dict(collections.Counter(m['state_end'] for m in meta)),'lock_counts':dict(collections.Counter(m['receiver_lock_status'] for m in meta)),'statistics':stats,'true_total_snr_known':False})
        print('physics',scene,stats['signal_to_added_noise_db']['median'],flush=True)
    result['status']='VERIFIED';out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
