"""Joint learned timing and recovery with an independent WiSig source scorer."""

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
from .wisig_anchored_window import AnchoredWindow, generic_crop, objective
from .wisig_tri_learned_replay import BASELINE_COMMIT, parent
from .wisig_uncertain_prior import as_complex, as_iq


RUN = '20261124-phase1-wisig-anchored-window-source-m4-r01'
ARMS = ('repeat', 'preamble')
READOUTS = ('timed', 'coarse', 'joint', 'gated')
MODES = ('direct', 'r69', *(f'{arm}_{name}' for arm in ARMS for name in READOUTS))
COMPARISONS = (*(('preamble_'+name,'repeat_'+name) for name in READOUTS),
               ('preamble_joint','preamble_timed'),('preamble_gated','direct'),
               *((name,'r69') for name in MODES[2:]))
SEED, STEPS, BATCH, HELD = core.SEED, 40_000, 24, 512
CHECKS = (1800, 6000, 12000, 20000, 30000)
ROUTES = core.baseline.ROUTES


def default_model(arm):
    return AnchoredWindow(arm=='preamble')


def proof(arm):
    return dict(status='SOURCE_FROZEN', arm=arm, steps=STEPS, seed=SEED, batch=BATCH,
                target_reads=0, F_reads=0, V_reads=0, TX_truth_reads=0, trace_reads=0)


def guard_scores(analysis):
    noise,logits = analysis[3:5]
    margin = logits.max(-1).values-logits[:,logits.shape[-1]//2]
    return margin,noise


def fit(rt, arm, *, smoke, build=default_model, calculate=objective, gate_features=None):
    if set(rt) != {'run_root','source_root'}:
        raise ValueError('Only the source IQ and output roots are open to fitting')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, ids, calibration = core.source_iq(Path(rt['source_root']))
    training, held = ids[HELD:], ids[:HELD]
    device = 'cuda:0'
    torch.manual_seed(SEED)
    model = build(arm).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=3e-5)
    rng = np.random.default_rng(SEED+2)
    generator = torch.Generator(device=device).manual_seed(SEED+1)

    @torch.no_grad()
    def held_costs():
        fixed = torch.Generator(device=device).manual_seed(SEED+5)
        per_batch = [calculate(model,as_complex(tensor(source[chunk],device)),fixed,identity=False)[1]*len(chunk)
                     for chunk in np.array_split(held,HELD//BATCH)]
        return (torch.stack(per_batch).sum(0)/HELD).tolist()

    history = {}
    started = perf_counter()
    for step in range(1,(1 if smoke else STEPS)+1):
        batch = as_complex(tensor(source[rng.choice(training,BATCH,replace=False)],device))
        loss, values = calculate(model,batch,generator,identity=True)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite WiSig source objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if step==1 and any(net[-1].weight.grad.norm()<=0 for net in (model.timing,model.coarse,model.refine)):
            raise ValueError('Timing, coarse and fine must receive source IQ loss gradients')
        opt.step()
        schedule.step()
        if step==1 or step%50==0:
            print('WISIG_ANCHORED_FIT',step,*values.detach().tolist(),float(norm),flush=True)
        if not smoke and step in CHECKS:
            model.eval()
            history[str(step)] = held_costs()
            print('WISIG_ANCHORED_HELD',step,*history[str(step)],flush=True)
            model.train()
    torch.cuda.synchronize()
    seconds = perf_counter()-started
    root = Path(rt['run_root'])
    if smoke:
        write(root/f'smoke-{arm}.json',dict(status='PASS', source_U_count=BATCH, target_reads=0))
        return
    with torch.no_grad():
        model.eval()
        metrics = []
        for chunk in np.array_split(calibration,len(calibration)//BATCH):
            analysis = model.analyze(as_complex(tensor(source[chunk],device)))
            values = guard_scores(analysis) if gate_features is None else gate_features(model,analysis)
            metrics.append(torch.stack(values))
        gate = (1.02*torch.cat(metrics,dim=1).amax(1)+1e-6).tolist()
        evidence = dict(proof(arm),gate=gate,optimizer_seconds=seconds,fit_count=len(training),
                        calibration_count=len(calibration),held_count=HELD,held=held_costs(),held_trajectory=history,
                        parameters=sum(p.numel() for p in model.parameters()),peak_cuda=torch.cuda.max_memory_allocated())
    with (root/f'{arm}.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(),evidence=evidence),stream)
    write(root/f'{arm}-complete.json',evidence)


def load_models(rt, *, arms=ARMS, build=default_model):
    root = Path(rt['run_root'])
    for arm in arms:
        checkpoint = torch.load(root/f'{arm}.pt',map_location='cuda:0',weights_only=True)
        metadata = read(root/f'{arm}-complete.json')
        if checkpoint['evidence'] != metadata or any(metadata[k]!=v for k,v in proof(arm).items()):
            raise ValueError('Expected matching frozen source-only WiSig candidate')
        gate = metadata['gate']
        if len(gate)!=2 or not np.isfinite(gate).all() or min(gate)<=0:
            raise ValueError('Invalid source-only frozen guard')
        model = build(arm).cuda().eval().requires_grad_(False)
        model.load_state_dict(checkpoint['weights'],strict=True)
        yield model,gate


def proposals(model, gate, y):
    stages,shift,margin,noise = model(y)
    active = (margin>gate[0]) | (noise>gate[1])
    return (*stages,torch.where(active[:,None],stages[-1],y)),active,shift


def guarded(model, gate, y, *, gate_features=None):
    analysis = model.analyze(y)
    margin,noise = guard_scores(analysis) if gate_features is None else gate_features(model,analysis)
    active = (margin>gate[0]) | (noise>gate[1])
    if not active.any():
        return y, active
    stages,_,_,_ = model.restore(tuple(field[active] for field in analysis))
    result = y.clone()
    result[active] = stages[-1]
    return result,active


def classify(reader,y):
    logits = reader.classify_features(reader.features(as_iq(y)))
    if logits.shape!=(len(y),6) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite decisions from the registered WiSig reader')
    return logits


@torch.no_grad()
def smoke(rt, *, models_from=load_models, variants=proposals, disturb=generic_crop, gate_features=None):
    source,_, calibration = core.source_iq(Path(rt['source_root']))
    y = as_complex(tensor(source[calibration[:16]],'cuda:0'))
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    models = tuple(models_from(rt))
    restored, = core.baseline.original_r69(Path(rt['parent_root']))(as_iq(y[:4]))
    classify(reader, y[:4])
    classify(reader, as_complex(restored))
    for model, gate in models:
        for variant in variants(model,gate,y[:4])[0]:
            classify(reader,variant)
    model, gate = models[-1]
    disturbed = disturb(y,torch.Generator(device=y.device).manual_seed(SEED+12),model)
    if gate_features is None:
        index = model(disturbed)[2].argmax().view(1)
    else:
        scores = gate_features(model,model.analyze(disturbed))
        index = torch.maximum(scores[0]/gate[0],scores[1]/gate[1]).argmax().view(1)
    timings = {}
    for label, sample in (('source_clean',y[:1]),('generic_disturbed',disturbed[index])):
        elapsed = []
        for i in range(28):
            torch.cuda.synchronize()
            started = perf_counter()
            _, active = guarded(model,gate,sample,gate_features=gate_features)
            torch.cuda.synchronize()
            if i>=8:
                elapsed.append(1000*(perf_counter()-started))
        timings[label] = dict(active=bool(active.item()),median_ms=float(np.median(elapsed)))
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',count=4,target_reads=0,V_reads=0,b1=timings))


@torch.no_grad()
def evaluate(rt, route, *, arms=ARMS, models_from=load_models, variants=proposals, auxiliary=None):
    old = parent(rt,tri.RUN,BASELINE_COMMIT)/'predictions'
    ref = read(old/'complete.json')
    cache,_,_,_,ids,_,_ = core.baseline.handles(rt)
    if (ref['status'],ref['modes'],ref['count'],ref['target_reads'],ref['source_truth_reads']) != (
            'FROZEN',list(tri.MODES),len(ids),0,0):
        raise ValueError('Frozen same-reader comparator is required')
    models = tuple(models_from(rt))
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    root = Path(rt['run_root'])/'predictions'
    measures = {}
    for view in core.baseline.VIEWS:
        key = f'{route}_{view}'
        with np.load(old/f'{key}.npz',allow_pickle=False) as record:
            grid = record['decision']
            if not np.array_equal(record['ids'],ids) or grid.shape!=(len(tri.MODES),len(ids)) or grid.dtype!=np.uint8:
                raise ValueError('Frozen comparator physical rows differ')
            paired = grid[[tri.MODES.index('direct'),tri.MODES.index('r69')]]
        iq = np.load(cache/ROUTES[route]/f'{view}.npy',mmap_mode='r',allow_pickle=False)
        if iq.shape!=(len(ids),2,256) or iq.dtype!=np.float32:
            raise ValueError('Unexpected WiSig source prediction view')
        decisions,activity,shifts,estimates = [],[0,0],[[],[]],[]
        for start in range(0,len(iq),48):
            y = as_complex(tensor(iq[start:start+48],'cuda:0'))
            if auxiliary is not None:
                estimates.append(auxiliary(models[-1][0],y))
            options = []
            for i,(model,gate) in enumerate(models):
                values,active,shift = variants(model,gate,y)
                options.extend(classify(reader,value).argmax(-1) for value in values)
                activity[i] += int(active.sum())
                shifts[i].append(shift.cpu())
            decisions.append(torch.stack(options).cpu())
        candidate = np.asarray(torch.cat(decisions,dim=1).tolist(),dtype=np.uint8)
        matrix = np.concatenate((paired,candidate))
        with (root/f'{key}.npz').open('xb') as stream:
            np.savez(stream,ids=ids,decision=matrix)
        offsets = {arm:torch.cat(delta).abs() for arm,delta in zip(arms,shifts,strict=True)}
        measures[key] = dict(count=len(ids),active=dict(zip(arms,activity,strict=True)),
                             moves={arm:int((delta!=0).sum()) for arm,delta in offsets.items()},
                             mean_boundary_filled={arm:delta.float().mean().item() for arm,delta in offsets.items()})
        if estimates:
            names = tuple(estimates[0])
            measures[key]['receiver_estimates'] = {
                name:torch.cat([item[name] for item in estimates]).quantile(
                    torch.tensor([.1,.5,.9],device='cuda:0')).tolist() for name in names
            }
        print('WISIG_ANCHORED_FROZEN',key,len(ids),*activity,flush=True)
    write(root/f'{route}-complete.json',measures)


def dispatch(rt, *, arms=ARMS, modes=MODES):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root/'state.json'

    def state(name, **details):
        write(status,dict(status=name,**details),exclusive=False)

    def start(action,name,gpu=None):
        return core.baseline.start(rt,action,name,gpu,module=rt['runner'],extra_keys=('recognizer_run',))

    def pair(action,names,launch):
        first = core.baseline.select_gpu()
        gpus = (first,core.baseline.select_gpu(exclude=(first,)))
        state(action.upper(),gpus=dict(zip(names,gpus,strict=True)))
        jobs = [launch(action,name,gpu) for name,gpu in zip(names,gpus,strict=True)]
        if any([job.wait() for job in jobs]):
            raise RuntimeError(f'WiSig {action} failed')

    try:
        for action in ('fit_smoke','fit'):
            pair(action,arms,lambda kind,name,gpu:core.train_worker(rt,kind,name,gpu))
        state('SMOKE')
        if start('smoke','source',core.baseline.select_gpu()).wait():
            raise RuntimeError('Real checkpoint smoke failed')
        predictions = root/'predictions'
        predictions.mkdir()
        pair('evaluate',ROUTES,start)
        measures = {key:item for route in ROUTES for key,item in read(predictions/f'{route}-complete.json').items()}
        expected = {f'{route}_{view}' for route in ROUTES for view in core.baseline.VIEWS}
        if set(measures)!=expected or any(x['count']!=1800 for x in measures.values()):
            raise ValueError('Each route must finish every frozen source prediction before scoring')
        write(predictions/'complete.json',dict(status='FROZEN',count=1800,modes=list(modes),
              views=list(core.baseline.VIEWS),source_truth_reads=0,target_reads=0,summaries=measures))
        state('SCORE')
        if start('score','source').wait():
            raise RuntimeError('Independent WiSig score failed')
        state('SOURCE_SCORED')
    except Exception as exc:
        state('FAILED',error=str(exc))
        raise


def publish(repo, output, context, ssh_config, *, run=RUN, runner=None):
    cfg = read(context)
    rt = {key:cfg[key] for key in ('project','python','frozen_sources')}
    p = PurePosixPath(rt['project'])
    runs = p/'runs'
    rt.update(run_id=run,runner=runner or __spec__.name,code_commit=release_commit(repo),
              run_root=str(runs/run),release_root=str(p/'releases'/f'{run}-release'),
              source_root=str(runs/core.baseline.CACHE_RUN/'cache'),parent_root=str(runs/core.baseline.R17_RUN),
              recognizer_run=str(runs/tri.tri_plan.RUN))
    release(repo,output,rt,ssh_config,archive_extras=('tools/r17_activation_factorization.py',))


def main(*, run=RUN, runner=None, arms=ARMS, modes=MODES, comparisons=COMPARISONS, fit_arm=fit,
         run_smoke=smoke, run_evaluation=evaluate):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','fit_smoke','fit','smoke','evaluate','score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime',type=Path)
        if action!='dispatch':
            worker.add_argument('name',choices=arms if action in ('fit_smoke','fit') else ROUTES if action=='evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args, run=run, runner=runner)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt, arms=arms, modes=modes)
    elif action in ('fit_smoke','fit'):
        fit_arm(rt,args['name'],smoke=action=='fit_smoke')
    elif action=='score':
        core.baseline.score(rt,modes,comparisons=comparisons)
    else:
        activate(rt)
        if action=='smoke':
            run_smoke(rt)
        else:
            run_evaluation(rt,args['name'])


if __name__=='__main__':
    main()
