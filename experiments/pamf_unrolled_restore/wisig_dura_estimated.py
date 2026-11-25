"""Train matched estimated-supervision and IQ-only DURA on WiSig Main source pairs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import perf_counter, sleep

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_capacity import DuraCapacity
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main import publish as transport
from . import wisig_dura_stress as evaluation
from . import wisig_dura_stress_fit as training
from . import wisig_ten_channel_audit as calibration
from . import wisig_ten_channel_estimator as estimator
from .wisig_lora_source_fit import source_iq
from .wisig_paired_estimator import voltages


RUN = '20261125-phase1-wisig-dura-estimated-main-m2-r01'
ARMS = ('estimated','blind')
METHODS = ('raw','R69',*(f'trained_{arm}' for arm in ARMS))
BANK_PER_EPOCH = 2048
BANK_WORKERS = 4
ESTIMATE_EVERY_EPOCH_CYCLES = 4
MIN_FREE_MIB = 8192


def estimate_pair(job):
    clean,received,options = job
    action,var,_ = estimator.choose_and_apply(voltages(clean),voltages(received),options)
    return np.stack((action.real,action.imag)).astype(np.float32),np.float32(var)


def bank(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Prepare source pseudo targets on CPU')
    root = Path(rt['run_root'])/'estimates'
    root.mkdir()
    cache = Path(rt['source_root'])
    options = read(rt['iq_estimator'])
    clean,fit_ids,_ = source_iq(cache)
    views = training.source_views(cache,root,'default')
    rng = np.random.default_rng(training.SEED+46)
    with ProcessPoolExecutor(max_workers=BANK_WORKERS,mp_context=multiprocessing.get_context('spawn')) as tasks:
        for epoch,view in enumerate(views):
            fit = fit_ids[view[0][fit_ids]>=.15]
            selected = np.sort(rng.choice(fit,BANK_PER_EPOCH,replace=False))
            observed,target,identity = training.examples(clean,view,selected)
            if identity.any():
                raise ValueError('The estimated targets require a distorted source IQ pair')
            input_rows = ((x,y,options) for x,y in zip(target,observed))
            values = list(tasks.map(estimate_pair,input_rows,chunksize=24))
            action,variance = (np.stack(rows) for rows in zip(*values))
            if not np.isfinite(action).all() or not np.isfinite(variance).all() or (variance<=0).any():
                raise ValueError('Nonfinite or invalid source pseudo supervision')
            np.save(root/f'{epoch+1:02}-ids.npy',selected,allow_pickle=False)
            np.savez(root/f'{epoch+1:02}-targets.npz',action=action,variance=variance)
            print('IQ_ESTIMATE_BANK',epoch+1,len(selected),flush=True)
    write(root/'complete.json',dict(status='IQ_ESTIMATES_FROZEN',epochs=len(views),per_epoch=BANK_PER_EPOCH,
                                     simulator_parameter_reads=0,TX_reads=0,F_reads=0,target_reads=0,
                                     options={key:options[key] for key in ('complexity','shrink','blend')}))


def bank_data(root,views,with_estimates,fit_ids):
    folder = root/'estimates'
    marker = read(folder/'complete.json')
    if (marker['status'],marker['epochs'],marker['per_epoch'])!=('IQ_ESTIMATES_FROZEN',len(views),BANK_PER_EPOCH):
        raise ValueError('Expected the frozen source pseudo-target bank')
    output = []
    for epoch,view in enumerate(views):
        ids = np.load(folder/f'{epoch+1:02}-ids.npy',allow_pickle=False)
        action = variance = None
        if with_estimates:
            with np.load(folder/f'{epoch+1:02}-targets.npz',allow_pickle=False) as archive:
                action,variance = archive['action'],archive['variance']
        if (ids.shape!=(BANK_PER_EPOCH,) or len(np.unique(ids))!=len(ids) or
                not np.isin(ids,fit_ids).all() or (view[0][ids]<.15).any()):
            raise ValueError('Estimated target rows differ from source U fit IDs')
        output.append((ids,action,variance))
    return output


def fit(rt,arm,*,smoke=False):
    if arm not in ARMS:
        raise ValueError('Unknown DURA supervision arm')
    root,cache = Path(rt['run_root']),Path(rt['source_root'])
    clean,fit_ids,calibration_ids = source_iq(cache)
    views = training.source_views(cache,root,'default')
    targets = bank_data(root,views,arm=='estimated',fit_ids)
    torch.set_num_threads(2)
    torch.manual_seed(training.SEED)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda:0'
    model = DuraCapacity('expanded').to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,training.STEPS,eta_min=3e-5)
    sampler = np.random.default_rng(training.SEED+2)
    start = perf_counter()
    for step in range(1,4 if smoke else training.STEPS+1):
        epoch = (step-1)%len(views)
        pseudo = ((step-1)//len(views))%ESTIMATE_EVERY_EPOCH_CYCLES==0
        estimates = None
        if pseudo:
            ids,action,variance = targets[epoch]
            positions = sampler.choice(len(ids),training.BATCH,replace=False)
            selected = ids[positions]
            if arm=='estimated':
                estimates = tensor(action[positions],device),tensor(variance[positions],device)
        else:
            selected = sampler.choice(fit_ids,training.BATCH,replace=False)
        rows = training.examples(clean,views[epoch],selected)
        logging = step==1 or step%100==0 or smoke
        components = {} if logging else None
        loss = training.fit_loss(model,*(tensor(item,device) for item in rows),estimates=estimates,trace=components)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        lr.step()
        if logging:
            print('WISIG_DURA_FIT',arm,step,float(loss.detach()),components,perf_counter()-start,flush=True)
    if smoke:
        print('SOURCE_IQ_SMOKE_PASS',arm,'target_reads=0',flush=True)
        return
    record = dict(status='SOURCE_FROZEN',arm=arm,steps=training.STEPS,seed=training.SEED,batch=training.BATCH,
                  threshold=training.threshold(clean,calibration_ids,device),fit_count=len(fit_ids),
                  calibration_count=len(calibration_ids),params=sum(p.numel() for p in model.parameters()),
                  seconds=perf_counter()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,
                  simulator_parameter_reads=0,TX_reads=0,F_reads=0,target_reads=0,
                  estimated_supervision=arm=='estimated',per_view_bank=BANK_PER_EPOCH)
    with (root/f'{arm}.pt').open('xb') as destination:
        torch.save(dict(weights=model.state_dict(),metadata=record),destination)
    write(root/f'{arm}-complete.json',record)


def frozen_fit(root,arm):
    record = read(root/f'{arm}-complete.json')
    checkpoint = torch.load(root/f'{arm}.pt',map_location='cuda:0',weights_only=True)
    if ((record,record['status'],record['arm'],record['steps']) !=
            (checkpoint['metadata'],'SOURCE_FROZEN',arm,training.STEPS) or
            record['estimated_supervision']!=(arm=='estimated') or
            any(record[key] for key in ('simulator_parameter_reads','TX_reads','F_reads','target_reads'))):
        raise ValueError('Unexpected WiSig DURA source checkpoint')
    model = DuraCapacity('expanded').cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['weights'],strict=True)
    return model,record['threshold']


def gpu_selection(minimum=MIN_FREE_MIB):
    command = ['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu',
               '--format=csv,noheader,nounits']
    samples = []
    for _ in range(5):
        rows = subprocess.check_output(command,text=True).splitlines()
        samples.append([tuple(map(int,row.split(','))) for row in rows])
        sleep(1)
    medians = np.median(samples,axis=0)
    eligible = [row for row in medians if row[2]>=minimum]
    ordered = sorted(eligible,key=lambda row:(.7*row[3]+30*row[1]/(row[1]+row[2]),row[1]))
    return [int(row[0]) for row in ordered],medians.tolist()


def launch(rt,operation,name,gpu=None):
    root = Path(rt['run_root'])
    context = root/'source_runtime.json' if operation in ('bank','fit','fit-smoke') else Path(rt['release_root'])/'runtime.json'
    cmd = [rt['python'],'-u','-m',rt['runner'],operation,str(context),name]
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (root/'logs'/f'{operation}-{name}.log').open('x',encoding='utf-8') as log:
        worker = subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT)
    print('WISIG_DURA_WORKER',operation,name,worker.pid,gpu,flush=True)
    return worker


def complete(worker,stage):
    if worker.wait():
        raise RuntimeError(f'{stage} failed; inspect this run log')


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root/'logs').mkdir()
    state = root/'state.json'
    try:
        write(root/'source_runtime.json',{key:rt[key] for key in ('run_root','source_root','iq_estimator')})
        write(state,dict(status='ESTIMATING_SOURCE_SUPERVISION'))
        complete(launch(rt,'bank','iq'),'IQ pseudo target bank')
        candidates,load = gpu_selection()
        if len(candidates)<2:
            raise RuntimeError(f'Fewer than two cards have {MIN_FREE_MIB} MiB free')
        gpus = dict(zip(ARMS,candidates[:2]))
        write(root/'gpu_selection.json',dict(assigned=gpus,samples=load,minimum_free_mib=MIN_FREE_MIB))
        write(state,dict(status='SOURCE_SMOKE',gpus=gpus),exclusive=False)
        for arm,gpu in gpus.items():
            complete(launch(rt,'fit-smoke',arm,gpu),f'{arm} source smoke')
        write(state,dict(status='SOURCE_TRAINING',gpus=gpus),exclusive=False)
        workers = {arm:launch(rt,'fit',arm,gpu) for arm,gpu in gpus.items()}
        for arm,worker in workers.items():
            complete(worker,f'{arm} source training')
        write(state,dict(status='FROZEN_CHECKPOINT_SMOKE'),exclusive=False)
        complete(launch(rt,'smoke','readout',gpus['estimated']),'frozen no-query smoke')
        # Prediction uses the shared scheduler's live memory check on every device.
        allowed,_ = gpu_selection(minimum=0)
        write(state,dict(status='TARGET_PREDICTING'),exclusive=False)
        schedule(dict(rt,allowed_gpus=allowed,max_workers=2,max_compute_processes=None),
                 [('predict',case) for case in evaluation.CASES],launch)
        write(state,dict(status='PREDICTIONS_FROZEN'),exclusive=False)
        complete(launch(rt,'score','cpu'),'independent truth-last scoring')
        write(state,dict(status='ANALYZED'),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def publish(repo,output,context,ssh_config):
    ctx = read(context)
    rt = {key:ctx[key] for key in ('project','python','frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project/'runs'
    paths = dict(source_root=str(runs/'20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'/'cache'),
                 iq_estimator=str(runs/calibration.RUN/'configuration.json'),
                 parent_root=str(runs/'20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'),
                 baseline_root=str(runs/evaluation.BASELINE),
                 recognizer_run=str(runs/evaluation.F_RUN),
                 baseline_replay=str(runs/'20261124-phase1-wisig-preamble-dura-main-m4-r01'),
                 strong_view_root=str(runs/evaluation.RUN/'strong_target_iq'))
    rt.update(run_id=RUN,runner=__spec__.name,code_commit=transport.release_commit(repo),
              run_root=str(runs/RUN),release_root=str(project/'releases'/f'{RUN}-release'),
              external_releases=list(paths.values()),**paths)
    extras = ('tools/wisig_paired_main_uncertainty.py','tools/r17_activation_factorization.py')
    transport.release(repo,output,rt,ssh_config,archive_extras=extras)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    release = sub.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        release.add_argument('--'+field,type=Path,required=True)
    commands = dict(dispatch=lambda rt,_:dispatch(rt),bank=lambda rt,_:bank(rt),fit=fit,
                    smoke=lambda rt,_:evaluation.smoke(rt,arms=ARMS,load=frozen_fit),
                    predict=lambda rt,name:evaluation.predict(rt,name,methods=METHODS,load=frozen_fit),
                    **{'fit-smoke':lambda rt,name:fit(rt,name,smoke=True)})
    for field in (*commands,'score'):
        subcommand = sub.add_parser(field)
        subcommand.add_argument('runtime',type=Path)
        subcommand.add_argument('name',nargs='?')
    args = parser.parse_args()
    if args.action=='publish':
        publish(args.repo,args.output,args.context,args.ssh_config)
    elif args.action=='score':
        from .wisig_dura_stress_score import score

        score(read(args.runtime),methods=METHODS)
    else:
        commands[args.action](read(args.runtime),args.name)


if __name__=='__main__':
    main()
