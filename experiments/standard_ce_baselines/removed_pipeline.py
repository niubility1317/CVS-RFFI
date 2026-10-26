"""Fixed E200 pure-CE SIG-TIME removal; isolated train/predict/truth-side score."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261026-phase1-pamf-nosigtime-purece-e200-manysig-m8-r01'
SEEDS=tuple(range(2026092701,2026092709))


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def worker(seed,gpu,run):
    rid=f'sig_time_removed-s{seed}';out=run/rid;out.mkdir(exist_ok=False)
    base=read(ROOT/'experiments/standard_ce_baselines/configs/residual_fusion-s2026092701.json')
    c=dict(base,variant='sig_time_removed',model_seed=seed,output_root=str(out/'source'))
    config=out/'source_config.json';write(config,c)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def execute(module,cfg,log):
        with (out/log).open('x') as handle:
            process=subprocess.Popen([sys.executable,'-u','-m',module,'--config',str(cfg)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            write(out/'process.json',dict(pid=process.pid,gpu=gpu,module=module,cwd=str(ROOT),seed=seed))
            if process.wait()!=0:raise RuntimeError(f'{rid}: {module} failed; preserve log')
    execute('experiments.standard_ce_baselines.source',config,'source.log')
    done=read(out/'source/completion.json')
    if done['epoch']!=200 or done['steps']!=10000:raise ValueError('Fixed budget mismatch')
    write(out/'source/freeze.json',dict(status='FROZEN',epoch=200,variant='sig_time_removed',model_seed=seed,
        checkpoint=str(out/'source/last.pt'),target_access=False,target_score_used=False))
    old=read(ROOT/'experiments/standard_ce_baselines/configs/sigtime_20261014/launch_spec.json')
    p=dict(row_id=rid,variant='sig_time_removed',model_seed=seed,source_output=str(out/'source'),
        source_contract=c['source_contract'],capsule=old['capsule'],output_root=str(out/'prediction'))
    pc=out/'prediction_config.json';write(pc,p)
    execute('experiments.standard_ce_baselines.predict',pc,'prediction.log')
    sc=out/'score_config.json';write(sc,dict(p,prediction_output=str(out/'prediction'),truth=old['truth'],output_root=str(out/'scores')))
    execute('experiments.standard_ce_baselines.score',sc,'score.log')
    return dict(row_id=rid,model_seed=seed,gpu=gpu,status='SCORED',checkpoint=str(out/'source/last.pt'))


def smoke():
    import torch
    from experiments.standard_ce_baselines.model import build
    cp=PROJECT/'runs/20261014-phase1-sigtime-purece-e200-clean-manysig-m8-r01/sig_time-s2026092701/source/last.pt'
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    if (payload['variant']!='sig_time' or payload['epoch']!=200 or
            not payload['initialization']['scratch_only'] or payload['initialization']['ancestors']):
        raise ValueError('Real checkpoint provenance mismatch')
    model=build('sig_time',2026092701);model.load_state_dict(payload['model'],strict=True);model.eval()
    with torch.no_grad():result=model(torch.zeros(2,2,256))
    if result.shape!=(2,6) or not torch.isfinite(result).all():raise ValueError('Real checkpoint no-query smoke failed')
    print('REAL_CHECKPOINT_NO_QUERY_SMOKE_PASS; scratch ablation weights do not inherit checkpoint',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);a=p.parse_args()
    gpus=[int(x) for x in a.gpus.split(',')]
    if not gpus or len(set(gpus))!=len(gpus) or any(g not in range(8) for g in gpus):raise ValueError('GPU list')
    run=PROJECT/'runs'/RUN;run.mkdir(exist_ok=False)
    smoke()
    state=dict(status='RUNNING',run_id=RUN,pid=os.getpid(),cwd=str(ROOT),command=sys.argv,gpus=gpus,
        commit=(ROOT/'release_commit.txt').read_text().strip(),rows={},failures=[])
    write(run/'pipeline_state.json',state)
    # One sequential queue per assigned GPU, including the final waiting seed.
    queues={gpu:list(SEEDS[i::len(gpus)]) for i,gpu in enumerate(gpus)}
    def queue(gpu,seeds):
        completed=[]
        for seed in seeds:
            try:row=worker(seed,gpu,run)
            except Exception as error:return completed,dict(seed=seed,gpu=gpu,error=str(error))
            completed.append(row)
        return completed,None
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(queue,gpu,seeds) for gpu,seeds in queues.items()]
        for future in as_completed(futures):
            completed,error=future.result()
            state['rows'].update({r['row_id']:r for r in completed})
            if error:state['failures'].append(error)
            write(run/'pipeline_state.json',state)
    state['status']='ARTIFACTS_COMPLETE' if len(state['rows'])==8 and not state['failures'] else 'PARTIAL'
    write(run/'pipeline_state.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
