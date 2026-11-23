"""Source-only action support and clean stress test for the frozen rank-four LoRa restorer."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .action_support import observed_support
from .channel_blind import corruption_score
from .channel_blind_source import panel, receive_only, setup
from .chirp_spectrum_source import freeze, read_frozen, score_frozen
from .complex_vector_source import load_reader, publish, score
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import coordinates, source_control
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_guard import generic_noise
from .weak_guard_source import RANK_RUN, parent_model


RUN = '20261124-phase1-lora-rank-split-action-source-m2-r01'
SCHEMA = 'lora_split_action_calibration_v1'
ISOLATION = dict(F_reads=0, V_reads=0, TX_reads=0, target_reads=0, practical_reads=0, trace_reads=0)
MODES = ('direct', 'reference', 'rank_old', 'open_quarter', 'open_half', 'split_near', 'split_full')
PAIRS = (*((name, 'rank_old') for name in MODES[3:]), ('split_near', 'open_half'), ('split_full', 'open_half'))


def limits(root, device, original, *, schema=SCHEMA, count=2):
    bound = read(root / 'support-calibration.json')
    if (bound['schema'] != schema or bound['status'] != 'SOURCE_FROZEN' or bound['parent'] != RANK_RUN
            or bound['old_threshold'] != original or any(bound[k] != v for k, v in ISOLATION.items())):
        raise ValueError('Unexpected source-only split action calibration')
    threshold = torch.as_tensor(bound['thresholds'], dtype=torch.float32, device=device)
    if threshold.shape != (count,) or not torch.isfinite(threshold).all():
        raise ValueError('Unexpected split action threshold')
    return threshold


def evidence(model, observed, quality):
    return observed_support(model.clean_detail, observed, quality, model.bands, model.selectors)


def output_delta(model, y, rotation, quality, rms, dose):
    refined = model(y, quality, rotation)[-1]
    return real_iq((torch.lerp(y, refined, dose) - y) * rotation * rms)


@torch.no_grad()
def recover(iq, tracker, model, original, thresholds):
    y, rotation, q, rms = coordinates(tracker, iq)
    old = corruption_score(y) > original
    near, wide = old.clone(), old.clone()
    quarter, half = iq.clone(), iq.clone()
    if old.any():
        half[old] += output_delta(model, y[old], rotation[old], q[old], rms[old], .5)
        quarter[old] = half[old]
    inactive = ~old
    if inactive.any():
        frame = tuple(t[inactive] for t in (y, rotation, q, rms))
        checks = evidence(model, frame[0], frame[2]) > thresholds
        near[inactive], wide[inactive] = checks.unbind(-1)
        refined = model(frame[0], frame[2], frame[1])[-1]
        for out, dose in ((half, .5), (quarter, .25)):
            update = torch.lerp(frame[0], refined, dose) - frame[0]
            out[inactive] += real_iq(update * frame[1] * frame[3])

    def edit(accepted):
        return torch.where(accepted[:, None, None, None], half, iq)

    return (edit(old), quarter, half, edit(near), edit(wide)), (old, near, wide)


def render(iq, _unused, baseline, tracker, models, *, original, thresholds, counts=None):
    ref, _ = baseline(iq)
    outputs, masks = recover(iq, tracker, models[0], original, thresholds)
    if counts is not None:
        counts.append(torch.stack(masks, dim=1).cpu().numpy())
    return torch.stack((iq, ref, *outputs)), int(masks[1].sum())


@torch.no_grad()
def source_calibrate(rt, *, smoke=False, measure=evidence, renderer=render, modes=MODES,
                     schema=SCHEMA, smoke_edit_index=4):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, held), contract = setup(rt, device, smoke=smoke)
    model, original = parent_model(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    start_time = perf_counter()

    def values(iq):
        rows, old = [], []
        for portion in iq.split(48):
            y, _, q, _ = coordinates(tracker, portion)
            rows.append(measure(model, y, q))
            old.append(corruption_score(y) > original)
        return torch.cat(rows), torch.cat(old)

    boundary = values(cal)[0].amax(0) + 1e-5
    rng = torch.Generator(device=device).manual_seed(2026113229)
    if smoke:
        binding = source_control(rt, 'full', device)
        if binding[2] != contract:
            raise ValueError('Inconsistent frozen source recognizer and rank recovery')
        disturbed = real_iq(generic_noise(complex_iq(cal[:3]), rng, amplitude=.06))
        cases = torch.cat((cal[:3], disturbed))
        output, _ = renderer(cases, None, binding[3], tracker, (model,), original=original, thresholds=boundary)
        logits = load_reader(root, contract, device)(output.flatten(0,1))[0]
        if logits.shape != (len(cases)*len(modes),10) or not torch.isfinite(logits).all():
            raise ValueError('Real checkpoint action support smoke failed')
        if not torch.equal(output[0], cases) or not output[smoke_edit_index, 3:].ne(cases[3:]).any():
            raise ValueError('Real checkpoint smoke did not exercise the candidate edit')
        write(root/'checkpoint_smoke.json',dict(status='PASS',modes=len(modes),source_packets=len(cases),target_reads=0))
        print('LORA_SPLIT_ACTION_CHECKPOINT_PASS',flush=True)
        return

    data = (('clean', held), *((str(d),real_iq(generic_noise(complex_iq(held),rng,amplitude=d)))
                               for d in (.015,.03,.06,.12)))
    diagnostic = {}
    for name, iq in data:
        tallies, old = values(iq)
        support = tallies > boundary
        diagnostic[name] = dict(old=int(old.sum()), split=support.sum(0).tolist(),
                                added=(support & ~old[:, None]).sum(0).tolist())
    write(root/'support-calibration.json',dict(schema=schema,status='SOURCE_FROZEN',parent=RANK_RUN,
          n_cal=len(cal), n_held=len(held), old_threshold=original, thresholds=boundary.tolist(),
          calibration_seconds=perf_counter()-start_time, generic_held=diagnostic, **ISOLATION))
    print('LORA_SPLIT_ACTION_FROZEN',boundary.tolist(),diagnostic,flush=True)


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    model, original = parent_model(root, binding[2], device)
    thresholds = limits(root, device, original)
    counts = []
    renderer = partial(render, original=original, thresholds=thresholds, counts=counts)
    score(rt, route, panel=panel, modes=MODES, render=renderer, binding=binding, frozen_models=(model,),
          require_clean_bypass=False,replay_fn=receive_only,finalize=freeze,reader_chunk=144)
    selected = np.concatenate(counts, axis=0).reshape(len(VIEWS), -1, 3)
    write(root/f'{route}-support-counts.json',dict(masks=['old','split_near_total','split_full_total'],
          by_view={name:values.sum(0).tolist() for name, values in zip(VIEWS,selected,strict=True)}))


def reserve_indices(rx):
    reserve = np.ones(len(rx),dtype=bool)
    reserve[panel(rx)] = False
    return np.flatnonzero(reserve)


@torch.no_grad()
def evaluate_reserve(rt, *, modes=MODES, renderer=render, load_limits=limits):
    def bind(root, contract, device, control, tracker):
        model, original = parent_model(root,contract,device)
        thresholds = load_limits(root,device,original)
        return partial(renderer, _unused=None, baseline=control, tracker=tracker, models=(model,),
                       original=original, thresholds=thresholds)
    predict_clean_reserve(rt, modes=modes, bind=bind)


@torch.no_grad()
def predict_clean_reserve(rt, *, modes, bind):
    root, device = Path(rt['run_root']), 'cuda:0'
    source, cache, contract, control, tracker = source_control(rt,'full',device)
    _, rx, ids, _ = source_role(source,contract,'V')
    index = reserve_indices(rx)
    recorded = np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    if recorded.shape[1:] != (len(rx),6,2,256) or len(index) != 11100:
        raise ValueError('Clean reserve must be the 11100 packets outside the fixed 900 source panel')
    recover = bind(root,contract,device,control,tracker)
    reader=load_reader(root,contract,device)
    parts=[]
    for ids_chunk in np.array_split(index,1+len(index)//48):
        iq=tensor(recorded[0,ids_chunk],device)
        rows,_=recover(iq)
        base=reader(iq)[0]
        if not torch.isfinite(base).all():
            raise ValueError('Nonfinite clean reserve direct logits')
        direct_top1=base.argmax(1)
        view=[direct_top1]
        for row in rows[1:]:
            changed=(row!=iq).flatten(1).any(1)
            prediction=direct_top1.clone()
            if changed.any():
                logits=reader(row[changed])[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite clean reserve logits')
                prediction[changed]=logits.argmax(1)
            view.append(prediction)
        parts.append(torch.stack(view).cpu().numpy())
        if len(parts)%40==0:
            print('LORA_CLEAN_RESERVE_PREDICTED',sum(x.shape[1] for x in parts),flush=True)
    with (root/'clean-reserve-predictions.npz').open('xb') as out:
        np.savez(out,ids=ids[index],rows=np.concatenate(parts,axis=1),modes=np.asarray(modes))
    print('LORA_CLEAN_RESERVE_FROZEN',len(index),flush=True)


def independent_score(rt, *, modes=MODES, comparisons=PAIRS):
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    with np.load(root/'clean-reserve-predictions.npz',allow_pickle=False) as stored:
        ids, pred = stored['ids'],stored['rows']
        if (tuple(stored['modes']) != modes or ids.shape != (11100,) or pred.shape != (len(modes),11100)
                or pred.dtype.kind not in 'iu' or (pred<0).any() or (pred>=10).any()):
            raise ValueError('Incomplete clean reserve predictions')
    main = {r:read_frozen(root,r,modes) for r in ROUTES}
    fixed = np.asarray(main['full'][0])
    main_clean=main['full'][1]['clean']
    if list(fixed)!=main['residual'][0] or not np.array_equal(main_clean,main['residual'][1]['clean']):
        raise ValueError('Full and residual clean predictions must match on the same samples')
    clean_ids=np.concatenate((fixed,ids))
    official=np.asarray(source_contract(source)['role_ids']['V'])
    if len(set(clean_ids)) != len(official) or set(clean_ids)!=set(official):
        raise ValueError('Main and reserve clean predictions must form the complete source V set')
    lookup={key:i for i,key in enumerate(official)}
    positions=np.array([lookup[k] for k in clean_ids])
    score_frozen(rt,modes=modes,comparisons=comparisons)
    full_clean=np.concatenate((main_clean,pred),axis=1)
    with np.load(source/'V_metadata.npz',allow_pickle=False) as truth:
        if not np.array_equal(truth['ids'][positions],clean_ids):
            raise ValueError('Source clean truth and fixed prediction identifiers differ')
        matches=full_clean==truth['y'][positions]
        rx=np.asarray(SOURCE)[truth['rx_i'][positions]]
    def count(mask):
        correct=matches[:,mask]
        base=correct[0]
        return dict(n=int(mask.sum()),correct=correct.sum(1).tolist(),
                    repairs=(correct&~base).sum(1).tolist(),harms=(~correct&base).sum(1).tolist(),
                    switched=(full_clean[:,mask]!=full_clean[:1,mask]).sum(1).tolist())
    write(root/'clean-complete-scores.json',dict(status='SOURCE_SCORED',modes=list(modes), target_reads=0,
        truth_after_predictions=True, summary=count(np.ones(len(clean_ids),dtype=bool)),
        per_rx={r:count(rx==r) for r in SOURCE},scope='source V clean complete main and independent reserve'))
    print('LORA_CLEAN_COMPLETE_SCORED',len(clean_ids),flush=True)


@torch.no_grad()
def timing(rt):
    root,device=Path(rt['run_root']),'cuda:0'
    (_,cal,_),contract=setup(rt,device,smoke=True)
    model,original=parent_model(root,contract,device)
    threshold=limits(root,device,original)
    tracker=PacketRestorer('consensus').to(device).eval()
    rng=torch.Generator(device=device).manual_seed(2026113259)
    disturbed=real_iq(generic_noise(complex_iq(cal[:1]),rng,amplitude=.12))
    results={}
    for name,iq in (('clean',cal[:1]),('generic',disturbed)):
        periods=[]
        for _ in range(32):
            torch.cuda.synchronize()
            start_time=perf_counter()
            y,p,q,r=coordinates(tracker,iq)
            active=bool((corruption_score(y)>original).item())
            if not active:
                active=bool((evidence(model,y,q)[0,0]>threshold[0]).item())
            if active:
                iq+output_delta(model,y,p,q,r,.5)
            torch.cuda.synchronize()
            periods.append((perf_counter()-start_time)*1000)
        results[name] = dict(median_ms=float(np.median(periods[8:])),active=int(active))
    write(root/'b1.json',dict(scope='split_near single readout, before frozen classifier',timing=results,
                             gpu=torch.cuda.get_device_name(),n_measurements=24,
                             parameters=sum(p.numel() for p in model.parameters())))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    launch=partial(start,module=rt['runner'])
    try:
        for stage, rows in (('smoke',('source',)),('fit',('source',)),('timing',('source',)),
                            ('evaluate',ROUTES),('evaluate',('reserve',))):
            schedule(rt,[(stage,name) for name in rows],launch)
        if launch(rt,'score','source').wait():
            raise RuntimeError('Independent rank split action scoring failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as error:
        write(root/'state.json',dict(status='FAILED',error=str(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    entry=commands.add_parser('publish')
    for option in ('repo','output','context','ssh-config','parent-submit'):
        entry.add_argument('--'+option,type=Path,required=True)
    for stage in ('dispatch','smoke','fit','timing','evaluate','score'):
        entry=commands.add_parser(stage)
        entry.add_argument('runtime',type=Path)
        if stage!='dispatch':
            entry.add_argument('row',choices=(*ROUTES,'reserve') if stage=='evaluate' else ('source',))
    opts=vars(parser.parse_args())
    action=opts.pop('action')
    if action=='publish':
        publish(**opts,run=RUN,runner=__spec__.name)
        return
    rt=read(opts['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        independent_score(rt)
    else:
        activate(rt)
        if action=='evaluate':
            if opts['row']=='reserve':
                evaluate_reserve(rt)
            else:
                evaluate(rt,opts['row'])
        elif action=='timing':
            timing(rt)
        else:
            source_calibrate(rt,smoke=action=='smoke')


if __name__=='__main__':
    main()
