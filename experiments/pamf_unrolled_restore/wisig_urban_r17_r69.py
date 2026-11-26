"""Continue Adaptive R17 and its direct R69 student with matched urban source IQ."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path, PurePosixPath
import sys
from threading import Thread
from time import perf_counter, sleep

import numpy as np
import torch

from experiments.cric_dura_comparison import g_lineage, g_train
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_repl_view_shuffle.runner import activate
from experiments.pamf_r44_pruned_refine.models import spectral_error
from experiments.pamf_stage2_inference.engine import InferenceEngine
from . import wisig_dura_estimated as workers
from . import wisig_dura_stress as evaluation
from . import wisig_dura_stress_data as data
from . import wisig_original_dura as replay
from . import wisig_r17_direct_distill as direct
from . import wisig_r69_low_urban as previous


RUN = '20261126-phase1-wisig-adaptive-r17-r69-urban-m1-r01'
PHASES = ('r17_base', 'r69_base', 'r17_strong', 'r69_strong')
ARMS = ('r17_start', *PHASES)
METHODS = ('raw', 'R69', *(f'trained_{arm}' for arm in ARMS))
SOURCE_KEYS = (*direct.SOURCE_KEYS, 'direct_root', 'previous_root')
SOURCE_SEED = 2026112621
RECEIVER_SEED = 2026112622
SHUFFLE_SEED = 2026112623
COUNT = 63000
SCHEMA = 'adaptive_r17_r69_urban_v1'
GROUPS = ('mid_urban', 'low_urban', 'clean', 'replay')


def phase_epochs(phase):
    return range(1, 11) if phase.endswith('_base') else range(11, 31)


def epoch_batches(epoch):
    return g_train.batches(COUNT, np.random.default_rng(SHUFFLE_SEED + epoch))


def split_batch(indices):
    n = len(indices)
    return np.split(indices, (n//4, 5*n//8, 7*n//8))


def configs(epoch):
    code = str(previous.CODE)
    if code not in sys.path:
        sys.path.insert(0, code)
    from leo_practical import Config

    cfg = tuple(Config(fs_hz=25e6, fc_hz=2462e6, scenario=f'practical_{name}', processing_route='full',
                       mode='post_sync', equalization_enabled=False, boundary='reflect') for name in GROUPS[:2])
    return tuple(map(data.harder, cfg)) if epoch > 10 else cfg


def source_coordinates(source):
    ids, rx, day = (replay.targets.load(source, key) for key in ('physical_ids', 'rx', 'day'))
    if len(ids) != COUNT:
        raise ValueError('Expected original 63,000 source packet IDs')
    return ids, np.asarray([f'rx{r}:day{d}:eq1' for r, d in zip(rx, day, strict=True)])


def namespace(epoch):
    return f'wisig/adaptive-r17-r69-urban/source-only/{epoch:02d}'


def prepare(rt):
    source = Path(rt['action_iq'])/'train'
    ids, sessions = source_coordinates(source)
    cache = Path(rt['run_root'])/'urban_iq'
    with data.pool(source/'clean.npy', ids, previous.CODE) as pool:
        for epoch in range(1,31):
            rows = [split_batch(indices) for indices in epoch_batches(epoch)]
            part = cache/f'{epoch:02d}.npy'
            observed = np.lib.format.open_memmap(part, mode='w+', dtype=np.float32, shape=(COUNT,2,256))
            for group, cfg in enumerate(configs(epoch)):
                selected = np.concatenate([row[group] for row in rows])
                jobs = data.batches(selected,cfg,sessions,SOURCE_SEED,RECEIVER_SEED,namespace(epoch),False)
                for indices, iq in pool.map(data.sample,jobs):
                    observed[indices] = iq
            observed.flush()
            del observed
            (cache/f'{epoch:02d}.ready').touch(exist_ok=False)
            print('URBAN_IQ_GROUP', epoch, 'strong' if epoch>10 else 'default', 39375, flush=True)
    write(cache/'complete.json',dict(status='SOURCE_IQ_COMPLETE',default_groups=10,strong_groups=20,
          simulator_supervision_reads=0,target_reads=0))


def starting_r69(rt, contract, r17_path):
    parent = Path(rt['previous_root'])
    path = parent/'student/final.pt'
    expected = str(Path(rt['direct_root'])/'student/final.pt')
    marker = read(parent/'student/complete.json')
    direct_marker = read(Path(rt['direct_root'])/'student/complete.json')
    if (parent.name != previous.RUN or Path(rt['direct_root']).name != direct.RUN
            or marker['contract'] != contract or marker['ancestors'] != [expected]
            or direct_marker['contract'] != contract or direct_marker['teacher'] != str(r17_path)
            or (direct_marker['status'],direct_marker['epoch'],direct_marker['updates']) != ('SOURCE_FROZEN',20,4940)):
        raise ValueError('Unexpected direct-distilled R69 ancestry')
    engine, _ = previous.frozen_fit(parent,previous.ARM)
    return engine.pipeline.model, path


def load_phase(root, phase, contract):
    folder = Path(root)/phase
    path = folder/'final.pt'
    marker = read(folder/'complete.json')
    item = torch.load(path,map_location='cpu',weights_only=True)
    stage = phase.split('_')[0]
    if (marker['status'] != 'SOURCE_FROZEN' or item['schema'] != SCHEMA
            or (item['phase'],item['stage'],item['last_epoch'],item['updates'],item['contract']) != (
                phase,stage,phase_epochs(phase).stop-1,247*len(phase_epochs(phase)),contract)
            or item['ancestry'] != marker['ancestry'] or any(item[key] != 0 for key in
                ('TX_labels_read','F_reads','target_reads','simulator_supervision_reads'))):
        raise ValueError('Unexpected frozen urban continuation checkpoint')
    model = g_lineage.skeleton(stage)
    model.load_state_dict(item['weights'],strict=True)
    return model.cuda().eval().requires_grad_(False), path


def training_models(rt,phase):
    replay.setup_torch()
    torch.manual_seed(replay.SEED)
    original, r17_path, contract = direct.teacher_from(rt)
    root = Path(rt['run_root'])
    r17 = phase.startswith('r17')
    if r17:
        anchor, init_path = original.cuda(), r17_path
        teacher, teacher_path = None, None
    else:
        anchor, init_path = starting_r69(rt,contract,r17_path)
        teacher_phase = phase.replace('r69','r17')
        teacher, teacher_path = load_phase(root,teacher_phase,contract)
    if phase.endswith('_strong'):
        init, init_path = load_phase(root,phase.replace('_strong','_base'),contract)
    else:
        init = anchor
    model = deepcopy(init)
    model.prior.requires_grad_(True)
    ancestry = dict(initialization=str(init_path),anchor=str(r17_path if r17 else Path(rt['previous_root'])/'student/final.pt'),
                    teacher=None if r17 else str(teacher_path),r07=str(Path(rt['teacher_root'])/'G/action/final.pt'))
    return model, anchor, teacher, contract, ancestry


def waveform_nmse(restored, target, reference):
    power = reference.square().mean((1,2)).clamp_min(1e-4)
    return (restored-target).square().mean((1,2))/power


def group_objective(model,anchor,teacher,name,y,x,weight):
    if name == 'clean':
        return waveform_nmse(model(y),x,x)
    if name == 'replay':
        with torch.no_grad():
            target = anchor(y)
        return waveform_nmse(model(y),target,x)
    stages = model.stage_outputs(y)[0]
    restored = stages[-1]
    losses = waveform_nmse(restored,x,x) + .05*spectral_error(restored,x)
    if teacher is not None:
        with torch.no_grad():
            desired = teacher.stage_outputs(y)[0][1:]
        distill = [waveform_nmse(learned,wanted,x) for learned,wanted in zip(stages,desired,strict=True)]
        losses = losses + weight*torch.stack(distill).mean(0)
    return losses


def smoke(rt):
    model, _, _, contract, ancestry = training_models(rt,'r17_base')
    view = replay.OriginalSourceView(dict(rt,action_supervision='operator'))
    source = Path(rt['action_iq'])/'train'
    ids, sessions = source_coordinates(source)
    x = np.array(view.clean[:8])
    y = data.channel(x,configs(1)[0],ids[:8],sessions[:8],SOURCE_SEED,RECEIVER_SEED,namespace(1),False)
    x, y = tensor(x,'cuda:0'), tensor(y,'cuda:0')
    loss = group_objective(model,None,None,'mid_urban',y,x,0.).mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.prior.parameters(),1.,error_if_nonfinite=True)
    if not torch.isfinite(loss) or not float(norm)>0 or any(p.grad is not None for p in model.frozen.parameters()):
        raise FloatingPointError('Invalid continuation gradients')
    anchor, student_path = starting_r69(rt,contract,Path(ancestry['initialization']))
    student = deepcopy(anchor)
    student.prior.requires_grad_(True)
    student_loss = group_objective(student,anchor,model.eval().requires_grad_(False),'mid_urban',y,x,.5).mean()
    student_loss.backward()
    student_norm = torch.nn.utils.clip_grad_norm_(student.prior.parameters(),1.,error_if_nonfinite=True)
    if (not torch.isfinite(student_loss) or not float(student_norm)>0
            or any(p.grad is not None for p in student.frozen.parameters())):
        raise FloatingPointError('Invalid direct student continuation gradients')
    write(Path(rt['run_root'])/'source_smoke.json',dict(status='PASS',source_packets=len(x),ancestry=ancestry,
          r17_loss=float(loss.detach()),r17_gradient_norm=float(norm),r69_start=str(student_path),
          r69_loss=float(student_loss.detach()),r69_gradient_norm=float(student_norm),
          simulator_supervision_reads=0,target_reads=0))


def learning_rate(step, total, strong):
    start, peak, end = (1e-5,3e-5,3e-6) if strong else (2e-5,5e-5,5e-6)
    if step < 100:
        return start + (peak-start)*step/100
    return end + (peak-end)*(1+math.cos(math.pi*(step-100)/(total-101)))/2


def wait_iq(cache,epoch):
    while not (cache/f'{epoch:02d}.ready').exists():
        if (cache/'FAILED').exists():
            raise RuntimeError('Urban source IQ generator exited before this group was closed')
        sleep(2.)
    iq = np.load(cache/f'{epoch:02d}.npy',mmap_mode='r',allow_pickle=False)
    if iq.shape != (COUNT,2,256) or iq.dtype != np.float32:
        raise ValueError('Wrong urban source IQ layout')
    return iq


def fit(rt,phase):
    model, anchor, teacher, contract, ancestry = training_models(rt,phase)
    root = Path(rt['run_root'])
    view = replay.OriginalSourceView(dict(rt,action_supervision='operator'))
    if not view.ready() or int(view.is_L.sum()) != 6300:
        raise ValueError('Original WiSig Main L/U training contract required')
    folder = root/phase
    folder.mkdir()
    epochs = phase_epochs(phase)
    stage = phase.split('_')[0]
    total = 247*len(epochs)
    groups = [{'params':list(model.prior.parameters())}]
    if stage == 'r69':
        groups.append({'params':list(model.frozen.estimator.parameters())})
    optimizer = torch.optim.AdamW(groups,lr=2e-5,weight_decay=1e-4)
    started, steps = perf_counter(), 0
    with (folder/'training.jsonl').open('x',encoding='utf-8') as log:
        for offset, epoch in enumerate(epochs):
            iq = wait_iq(root/'urban_iq',epoch)
            historical = replay.targets.received(Path(rt['action_iq']), 1+(epoch-1)%20)
            model.prior.train()
            if stage == 'r69':
                model.frozen.estimator.requires_grad_(offset>=2)
            totals = np.zeros(4)
            counts = np.zeros(4,dtype=np.int64)
            weight = .5 - .4*offset/(len(epochs)-1)
            for selected in epoch_batches(epoch):
                rate = learning_rate(steps,total,epoch>10)
                for index, group in enumerate(optimizer.param_groups):
                    group['lr'] = rate * (.1 if index else 1.)
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for key, indices, array in zip(GROUPS,split_batch(selected),(iq,iq,view.clean,historical),strict=True):
                    x, y = tensor(view.clean[indices],'cuda:0'), tensor(array[indices],'cuda:0')
                    group_loss = group_objective(model,anchor,teacher,key,y,x,weight).sum()
                    (group_loss/len(selected)).backward()
                    losses.append(group_loss.detach())
                amounts = torch.stack(losses)
                if not torch.isfinite(amounts).all():
                    raise FloatingPointError('Nonfinite urban continuation loss')
                gradient = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),1.,error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                totals += amounts.cpu().tolist()
                counts += [len(part) for part in split_batch(selected)]
                if steps == 1:
                    first = dict(phase=phase,updates=steps,loss=float(amounts.sum())/len(selected),
                                 gradient_norm=float(gradient),ancestry=ancestry)
                    write(folder/'first_update.json',first)
                    print('URBAN_FIRST_UPDATE',first,flush=True)
            if counts.tolist() != [15750,23625,15750,7875]:
                raise ValueError('Unexpected fixed source exposure mix')
            row = dict(phase=phase,epoch=epoch,updates=steps,counts=counts.tolist(),
                       losses=(totals/counts).tolist(),seconds=perf_counter()-started)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            log.flush()
            write(root/'progress.json',row,exclusive=False)
            print('URBAN_EPOCH',row,flush=True)
    if steps != total:
        raise ValueError('Unexpected continuation update count')
    frozen, ref = (model.frozen,anchor.frozen) if stage=='r17' else (model.frozen.coarse,anchor.frozen.coarse)
    if any(not torch.equal(a,b) for a,b in zip(frozen.state_dict().values(),ref.state_dict().values(),strict=True)):
        raise ValueError('Frozen receive context changed')
    proof = dict(phase=phase,stage=stage,last_epoch=epochs.stop-1,updates=steps,ancestry=ancestry,
                 contract=contract,TX_labels_read=0,F_reads=0,target_reads=0,simulator_supervision_reads=0)
    torch.save(dict(proof,schema=SCHEMA,weights={k:v.cpu() for k,v in model.state_dict().items()}),folder/'final.pt')
    write(folder/'complete.json',dict(proof,status='SOURCE_FROZEN',seconds=perf_counter()-started))


def readout_loader(rt):
    model, _, contract = direct.teacher_from(rt)

    def load(root,arm):
        restorer = model.cuda() if arm == 'r17_start' else load_phase(root,arm,contract)[0]
        r17 = arm.startswith('r17')
        return InferenceEngine(restorer, reuse=True, graph=not r17, native=r17), None

    return load


def evaluation_smoke(rt):
    activate(rt)
    replay.setup_torch()
    root = Path(rt['run_root'])
    _, heads = evaluation.readers(rt)
    loader = readout_loader(rt)
    iq = tensor(replay.targets.received(Path(rt['action_iq']),1)[:4],'cuda:0')
    for arm in ARMS:
        model, _ = loader(root,arm)
        decisions = evaluation.classify(heads,model(iq)[0])
        if len(decisions)!=len(evaluation.SEEDS) or any(len(row)!=len(iq) for row in decisions):
            raise ValueError('Invalid frozen PAMF predictions')
    write(root/'evaluation_smoke.json',dict(status='PASS',targets_read=0,arms=list(ARMS)))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root/'logs').mkdir()
    cache = root/'urban_iq'
    cache.mkdir()
    state = root/'state.json'
    try:
        write(root/'source_runtime.json',{k:rt[k] for k in SOURCE_KEYS})
        write(state,dict(status='SOURCE_SMOKE_AND_IQ_PREPARATION'))
        prep = evaluation.launch(rt,'source-iq','cpu')

        def monitor_prep():
            if prep.wait():
                (cache/'FAILED').touch()

        Thread(target=monitor_prep,daemon=True).start()
        allowed, _ = workers.gpu_selection(minimum=0)
        single = dict(rt,allowed_gpus=allowed,max_workers=1,max_compute_processes=None)
        schedule(single,[('fit-smoke','r17')],evaluation.launch)
        for phase in PHASES:
            write(state,dict(status='SOURCE_TRAINING',phase=phase),exclusive=False)
            schedule(single,[('fit',phase)],evaluation.launch)
        workers.complete(prep,'source IQ augmentation')
        write(state,dict(status='FROZEN_CHECKPOINT_SMOKE'),exclusive=False)
        schedule(single,[('smoke','readout')],evaluation.launch)
        write(state,dict(status='TARGET_PREDICTING'),exclusive=False)
        schedule(dict(single,max_workers=2),[('predict',case) for case in previous.CASES],evaluation.launch)
        write(state,dict(status='PREDICTIONS_FROZEN'),exclusive=False)
        workers.complete(evaluation.launch(rt,'score','cpu'),'independent Full-only scoring')
        write(state,dict(status='ANALYZED'),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def publish(repo,output,context,ssh_config):
    settings = read(context)
    rt = {k:settings[k] for k in ('project','python','frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project/'runs'
    paths = dict(action_iq=str(runs/replay.action_design.RUN/'cache'),teacher_root=str(runs/replay.OPERATOR_RUN),
                 direct_root=str(runs/direct.RUN),previous_root=str(runs/previous.RUN),
                 baseline_root=str(runs/evaluation.BASELINE),recognizer_run=str(runs/evaluation.F_RUN),
                 strong_view_root=str(runs/evaluation.RUN/'strong_target_iq'),
                 decision_replay=str(runs/replay.baseline.RUN/'predictions'))
    external = [*paths.values(),str(runs/previous.RUN/'student/final.pt'),str(runs/replay.OPERATOR_RUN/'G/r17/final.pt'),
                *(str(runs/run/'predictions') for run in (previous.RUN,direct.RUN,replay.OPERATOR_RUN))]
    rt.update(run_id=RUN,runner=__spec__.name,code_commit=release_commit(repo),run_root=str(runs/RUN),
              release_root=str(project/'releases'/f'{RUN}-release'),external_releases=external,**paths)
    release(repo,output,rt,ssh_config,archive_extras=(
        'tools/wisig_paired_main_uncertainty.py','tools/r17_activation_factorization.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    jobs = parser.add_subparsers(dest='action',required=True)
    pub = jobs.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    commands = {'dispatch':dispatch,'source-iq':prepare,'fit-smoke':smoke,'smoke':evaluation_smoke}
    for key in (*commands,'fit','predict','score'):
        worker = jobs.add_parser(key)
        worker.add_argument('runtime',type=Path)
        worker.add_argument('name',nargs='?')
    args = parser.parse_args()
    if args.action=='publish':
        return publish(args.repo,args.output,args.context,args.ssh_config)
    rt = read(args.runtime)
    if args.action=='fit':
        if args.name not in PHASES:
            raise ValueError('Unknown continuation phase')
        return fit(rt,args.name)
    if args.action=='predict':
        if args.name not in previous.CASES:
            raise ValueError('Only Full test shards are registered')
        replay.setup_torch()
        return evaluation.predict(rt,args.name,methods=METHODS,load=readout_loader(rt),force_practical=True,
                                  restore=replay.restore,clean_passthrough=False)
    if args.action=='score':
        from .wisig_dura_stress_score import score

        references = (
            ('adaptive_direct',Path(rt['direct_root'])/'predictions','trained_direct_distill'),
            ('previous_low_urban',Path(rt['previous_root'])/'predictions','trained_low_urban'),
            ('adaptive_multistage',Path(rt['teacher_root'])/'predictions','trained_adaptive_operator'),
        )
        return score(rt,methods=METHODS,force_practical=True,clean_passthrough=False,routes=previous.ROUTES,
                     references=references)
    return commands[args.action](rt)


if __name__=='__main__':
    main()
