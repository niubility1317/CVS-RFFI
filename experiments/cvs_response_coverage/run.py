import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import json
import subprocess
import sys
import time
from . import design as d
from .core import audit,source,decision


def main():
    c=d.config();d.validate(c)
    if d.BASE.exists():raise FileExistsError('Preserve existing local analysis; reconcile before resubmission')
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments/cvs_response_coverage'],cwd=d.ROOT,text=True)
    if dirty.strip():raise ValueError('Commit actual source-analysis code/config first')
    registration=d.read(d.ROOT/'experiments/cvs_response_coverage/registration.json')
    if len(registration['rows'])!=1 or registration['rows'][0]['config']!=c:raise ValueError('Registered analysis differs')
    d.BASE.mkdir();started=time.perf_counter();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
    d.write(d.BASE/'resolved_config.json',dict(config=c,commit=commit,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
        original_source_commit='64de4b84bad10a796893882c74f2539a3de0c0ce',checkpoint_consumed=False,raw_IQ_consumed=False))
    try:
        prior=d.read(d.TRAIN_REGISTRATION)
        if any(r['config']['training_mechanism']['response_radius']!=c['radius'] for r in prior['rows']):raise ValueError('Current response radius differs')
        predicted,actual,old_ratio=source(d.read(d.EVIDENCE))
        print('SOURCE INPUT PASS fixed L-derived primary; no IQ, model or target',flush=True)
        primary=audit(predicted,actual,c['primary_grid'],c['radius']);control=audit(predicted,actual,c['resolution_control'],c['radius'])
        value=dict(status='SOURCE_CAPACITY_ANALYZED',config=c,prior_primary_relative_MSE=old_ratio,primary=primary,resolution_control=control,
            verdict=decision(primary,control,c),model_forwards=0,optimizer_updates=0,head_fit_updates=0,target_access=False,
            interpretation='Ideal magnitude-response capacity only; no identity gain, finite-IQ fidelity or training superiority established',
            source_limit='RX-associated contrast mixes hardware, propagation, acquisition and interactions; not an identified receiver transfer function',
            proxy_limit='Flat within-bin infinite-signal power proxy excludes finite-window/boundary effects, input spectral weights, log floor and phase effects',
            no_training_parameters_selected=True,automatic_training=False)
        d.write(d.BASE/'results.json',value)
        d.write(d.BASE/'complete.json',dict(status='ANALYZED',verdict=value['verdict'],elapsed_seconds=time.perf_counter()-started,
            checkpoint_consumed=False,raw_IQ_consumed=False,target_access=False,model_forwards=0,optimizer_updates=0))
        print(json.dumps(dict(status='ANALYZED',verdict=value['verdict'],coverage=primary['weighted_direction_coverage'],resolution_coverage=control['weighted_direction_coverage'])),flush=True)
    except Exception as exc:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=str(exc),automatic_retry=False));raise


if __name__=='__main__':main()
