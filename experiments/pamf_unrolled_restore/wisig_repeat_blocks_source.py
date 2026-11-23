"""Matched repeated-preamble restoration and polar readouts on frozen WiSig source V."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.runner import activate

from . import wisig_joint_preamble_source as tri
from . import wisig_uncertain_source as core
from .wisig_repeat_blocks import RepeatBlockRecovery, generic_observation, observable, polar_readouts
from .wisig_tri_learned_replay import BASELINE_COMMIT, parent
from .wisig_uncertain_prior import as_complex, as_iq, fit_loss


RUN = '20261123-phase1-wisig-repeat-block-source-m3-r01'
ARMS = ('visible','masked')
READOUTS = ('coarse','joint','amplitude','phase','gated')
MODES = ('direct','r69',*(f'{arm}_{mode}' for arm in ARMS for mode in READOUTS))
STEPS, BATCH, DETAIL_WEIGHT = 1600, 48, .20


def proof(arm):
    return dict(status='SOURCE_FROZEN', arm=arm, seed=core.SEED, steps=STEPS, batch=BATCH,
                weak_weight=DETAIL_WEIGHT, F_reads=0, V_reads=0, TX_reads=0, target_reads=0, trace_reads=0)


def fit(rt, arm, *, smoke):
    if set(rt) != {'run_root','source_root'}:
        raise ValueError('Fit must receive only IQ and output roots')
    device = 'cuda:0'
    torch.manual_seed(core.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, fit_rows, calibration = core.source_iq(Path(rt['source_root']))
    model = RepeatBlockRecovery(masked=(arm=='masked')).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    rng = np.random.default_rng(core.SEED+2)
    generator = torch.Generator(device=device).manual_seed(core.SEED+1)
    start = perf_counter()
    for step in range(1,(1 if smoke else STEPS)+1):
        rows = rng.choice(fit_rows,BATCH,replace=False)
        x = as_complex(tensor(data[rows],device))
        opt.zero_grad(set_to_none=True)
        loss, items = fit_loss(model,x,generator,DETAIL_WEIGHT,perturbation=generic_observation)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite WiSig spectral-block source loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if step == 1 and (model.prior[-1].weight.grad.norm()<=0 or not any(
                p.grad is not None and p.grad.norm()>0 for p in model.detail.parameters())):
            raise ValueError('Both restoration stages must receive gradients')
        opt.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('WISIG_REPEAT_BLOCK_FIT',step,*items.tolist(),norm.item(),flush=True)
    torch.cuda.synchronize()
    result = dict(proof(arm), seconds=perf_counter()-start, parameters=sum(p.numel() for p in model.parameters()))
    root = Path(rt['run_root'])
    if smoke:
        write(root/f'smoke-{arm}.json',dict(result,status='PASS',steps=1))
        return
    with torch.no_grad():
        features = []
        for chunk in np.array_split(calibration,len(calibration)//64):
            _, _, var, transition = observable(as_complex(tensor(data[chunk],device)))
            features.append(torch.stack((var,transition)))
        noise, tail = torch.cat(features,dim=1)
        gate = [float(1.05*noise.max()),float(.95*tail.min())]
    result.update(gate=gate, calibration_rows=len(calibration),fit_pool=len(fit_rows))
    with (root/f'{arm}.pt').open('xb') as stream:
        torch.save(dict(state=model.state_dict(),evidence=result),stream)
    write(root/f'{arm}-complete.json',result)


def load_models(rt):
    root = Path(rt['run_root'])
    arms, thresholds = [], []
    for arm in ARMS:
        checkpoint = torch.load(root/f'{arm}.pt',map_location='cuda:0',weights_only=True)
        evidence = read(root/f'{arm}-complete.json')
        if evidence != checkpoint['evidence'] or any(evidence[k]!=v for k,v in proof(arm).items()):
            raise ValueError('Only the matching fixed source-U checkpoint is allowed')
        model = RepeatBlockRecovery(masked=(arm=='masked')).cuda().eval().requires_grad_(False)
        model.load_state_dict(checkpoint['state'],strict=True)
        arms.append(model)
        thresholds.append(evidence['gate'])
    if thresholds[0]!=thresholds[1] or not all(np.isfinite(thresholds[0])) or min(thresholds[0])<0:
        raise ValueError('Both arms require the same U-only guard')
    return arms, thresholds[0]


def triggered(y, gate):
    _, _, noise, tail = observable(y)
    return (noise>gate[0]) | (tail<gate[1])


def gated(model, y, gate):
    active = triggered(y, gate)
    if not active.any():
        return y, active
    output = y.clone()
    output[active] = model(y[active])[1]
    return output, active


def proposals(y, models, gate):
    active = triggered(y,gate)
    outputs = []
    for model in models:
        coarse, joint, _ = model(y)
        amplitude, phase = polar_readouts(y,joint)
        outputs.extend((coarse,joint,amplitude,phase,torch.where(active[:,None],joint,y)))
    return outputs, active


@torch.no_grad()
def smoke(rt):
    data, _, rows = core.source_iq(Path(rt['source_root']))
    y = as_complex(tensor(data[rows[:4]],'cuda:0'))
    arms, gate = load_models(rt)
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    reference = core.baseline.original_r69(Path(rt['parent_root']))
    raw = as_iq(y)
    restored, = reference(raw)
    outputs,_ = proposals(y,arms,gate)
    for wave in (raw,restored,*(as_iq(output) for output in outputs)):
        z = reader.classify_features(reader.features(wave))
        if z.shape!=(len(raw),6) or not torch.isfinite(z).all():
            raise ValueError('Real frozen WiSig classifier and recovery smoke failed')
    x = y[:1]
    batch = as_complex(tensor(data[rows[:64]],x.device))
    disturbed = generic_observation(batch[:,None],torch.Generator(device=x.device).manual_seed(2026112332))[:,0]
    choices = triggered(disturbed,gate).nonzero().flatten()
    if len(choices):
        disturbed = disturbed[choices[:1]]
    else:
        disturbed = disturbed[:1]
    timings = {}
    for name, iq in (('clean',x),('generic_disturbed',disturbed)):
        durations = []
        for i in range(28):
            torch.cuda.synchronize()
            started = perf_counter()
            _, active = gated(arms[1],iq,gate)
            torch.cuda.synchronize()
            if i>=8:
                durations.append((perf_counter()-started)*1000)
        timings[name] = dict(active=bool(active.item()), median_ms=float(np.median(durations)))
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',count=4,target_reads=0,V_reads=0,b1=timings))


@torch.no_grad()
def evaluate(rt):
    old = parent(rt,tri.RUN,BASELINE_COMMIT)/'predictions'
    reference = read(old/'complete.json')
    cache,_,_,_,ids,_,_ = core.baseline.handles(rt)
    if (reference['status'],reference['modes'],reference['count'],reference['target_reads'],reference['source_truth_reads']) != (
            'FROZEN',list(tri.MODES),len(ids),0,0):
        raise ValueError('Expected the same reader frozen decisions and source physical panel')
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    models, gate = load_models(rt)
    root = Path(rt['run_root'])/'predictions'
    activity, seconds = {}, {}
    for route, directory in core.baseline.ROUTES.items():
        for view in core.baseline.VIEWS:
            begin = perf_counter()
            key = f'{route}_{view}'
            with np.load(old/f'{key}.npz',allow_pickle=False) as record:
                grid = record['decision']
                if not np.array_equal(record['ids'],ids) or grid.shape!=(len(tri.MODES),len(ids)) or grid.dtype!=np.uint8:
                    raise ValueError('Ancestor prediction rows differ')
                paired = grid[[tri.MODES.index('direct'),tri.MODES.index('r69')]]
            iq = np.load(cache/directory/f'{view}.npy',mmap_mode='r',allow_pickle=False)
            if iq.shape!=(len(ids),2,256) or iq.dtype!=np.float32:
                raise ValueError('Unexpected source IQ matrix')
            decisions, count = [],0
            for chunk in range(0,len(iq),BATCH):
                y = as_complex(tensor(iq[chunk:chunk+BATCH],'cuda:0'))
                values, active = proposals(y,models,gate)
                logits = torch.stack([reader.classify_features(reader.features(as_iq(x))) for x in values])
                if logits.shape!=(len(MODES)-2,len(y),6) or not torch.isfinite(logits).all():
                    raise ValueError('Unexpected real frozen reader decisions')
                decisions.append(logits.argmax(-1).tolist())
                count += int(active.sum())
            matrix = np.concatenate((paired,np.concatenate(decisions,axis=1))).astype(np.uint8)
            root.mkdir(exist_ok=True)
            with (root/f'{key}.npz').open('xb') as stream:
                np.savez(stream,ids=ids,decision=matrix)
            activity[key] = count
            seconds[key] = perf_counter()-begin
            print('WISIG_REPEAT_BLOCK_FROZEN',key,len(ids),count,flush=True)
    write(root/'complete.json',dict(status='FROZEN',modes=list(MODES),views=list(core.baseline.VIEWS),
          count=len(ids),target_reads=0,source_truth_reads=0,gate_activity=activity, evaluation_seconds=seconds))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root/'state.json'
    worker = core.baseline.start
    try:
        for action in ('fit_smoke','fit'):
            slots = [core.baseline.select_gpu()]
            slots.append(core.baseline.select_gpu(exclude=slots))
            write(status,dict(status=action.upper(),gpus=dict(zip(ARMS,slots,strict=True))),exclusive=False)
            jobs = [core.train_worker(rt,action,name,gpu) for name,gpu in zip(ARMS,slots,strict=True)]
            if any([job.wait() for job in jobs]):
                raise RuntimeError(f'WiSig {action} failed')
        for action in ('smoke','evaluate','score'):
            gpu = None if action=='score' else core.baseline.select_gpu()
            write(status,dict(status=action.upper(),gpu=gpu),exclusive=False)
            if worker(rt,action,'repeat',gpu,module=rt['runner'],extra_keys=('recognizer_run',)).wait():
                raise RuntimeError(f'WiSig {action} failed')
        write(status,dict(status='SOURCE_SCORED'),exclusive=False)
    except Exception as exc:
        write(status,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    config = read(context)
    rt = {key:config[key] for key in ('project','python','frozen_sources')}
    p = PurePosixPath(rt['project'])
    runs = p/'runs'
    rt.update(run_id=RUN,runner=__spec__.name,code_commit=release_commit(repo),
              run_root=str(runs/RUN),release_root=str(p/'releases'/f'{RUN}-release'),
              source_root=str(runs/core.baseline.CACHE_RUN/'cache'),parent_root=str(runs/core.baseline.R17_RUN),
              recognizer_run=str(runs/tri.tri_plan.RUN))
    release(repo,output,rt,ssh_config,archive_extras=('tools/r17_activation_factorization.py',))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','fit_smoke','fit','smoke','evaluate','score'):
        item = actions.add_parser(action)
        item.add_argument('runtime',type=Path)
        if action!='dispatch':
            item.add_argument('name',choices=ARMS if action in ('fit','fit_smoke') else ('repeat',))
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action in ('fit','fit_smoke'):
        fit(rt,args['name'],smoke=action=='fit_smoke')
    elif action=='score':
        comparisons = (('masked_coarse','visible_coarse'),('masked_joint','visible_joint'),
                       ('masked_amplitude','masked_joint'),('masked_phase','masked_joint'),
                       ('masked_gated','r69'),('masked_gated','direct'))
        core.baseline.score(rt,MODES,comparisons=comparisons)
    else:
        activate(rt)
        if action=='smoke':
            smoke(rt)
        else:
            evaluate(rt)


if __name__=='__main__':
    main()
