"""Freeze a U-only waveform library and compare its LoRa engineering readouts."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import ISOLATION, SEED, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .empirical_chirp_prior import EmpiricalChirpPrior
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import ROUTES, freeze, publish as launch_release, score_frozen
from .refine_runner import select_gpu, start
from .runner import activate
from .spectral_block_recovery import varied_observation


RUN = '20261124-phase1-lora-empirical-chirp-source-m2-r01'
MODES = ('direct','reference','all_half','cross_half','agreement_half','agreement_full')
PROOF = dict(status='SOURCE_FROZEN', templates=5000, calibration_rows=2500, parameters=0, steps=0,
             seed=SEED,**ISOLATION)


@torch.no_grad()
def setup_bank(rt):
    device = 'cuda:0'
    started = perf_counter()
    (source,calibration,_), contract = setup(rt,device)
    tracker = PacketRestorer('consensus').to(device).eval()
    basis = torch.cat([coordinates(tracker,iq)[0] for iq in source.split(64)])
    library = torch.fft.fft(basis,norm='ortho')
    values = torch.cat([corruption_score(coordinates(tracker,iq)[0]) for iq in calibration.split(64)])
    if not torch.isfinite(library).all() or not torch.isfinite(values).all():
        raise ValueError('Nonfinite source template or calibration')
    record = dict(PROOF,threshold=float(1.05*values.max()),seconds=perf_counter()-started,
                  library_bytes=library.numel()*library.element_size())
    root = Path(rt['run_root'])
    with (root/'library.pt').open('xb') as stream:
        torch.save(dict(library=library,contract=contract,record=record),stream)
    write(root/'library.json',record)


def load(root, contract, device):
    payload = torch.load(root/'library.pt',map_location=device,weights_only=True)
    record = read(root/'library.json')
    bank = payload['library']
    if (payload['contract']!=contract or payload['record']!=record or any(record[k]!=v for k,v in PROOF.items())
            or bank.shape!=(PROOF['templates'],6,256) or bank.dtype!=torch.complex64
            or not np.isfinite(record['threshold']) or record['threshold']<=0):
        raise ValueError('Expected the fixed legal source-U library and gate')
    return EmpiricalChirpPrior(bank).eval(), record['threshold']


def recover(iq, tracker, bank, threshold):
    y, rotation, _, rms = coordinates(tracker,iq)
    active = corruption_score(y)>threshold
    outputs = [iq.clone() for _ in MODES[2:]]
    if active.any():
        z, frame = y[active], rotation[active]*rms[active]
        for slot, restored in zip(outputs,bank(z),strict=True):
            slot[active] = iq[active]+real_iq((restored-z)*frame)
    return outputs, int(active.sum())


def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    values, active = recover(iq,tracker,models[0],threshold)
    return torch.stack((iq,reference,*values)), active


@torch.no_grad()
def smoke(rt):
    root = Path(rt['run_root'])
    device = 'cuda:0'
    (_,source,_), contract = setup(rt,device,smoke=True)
    bank, threshold = load(root,contract,device)
    reader = load_reader(root,contract,device)
    _,_,expected,baseline,tracker = source_control(rt,'full',device)
    if contract!=expected:
        raise ValueError('Expected same source contracts for the baselines')
    x = real_iq(varied_observation(complex_iq(source),torch.Generator(device=device).manual_seed(2026112403)))
    quality = corruption_score(coordinates(tracker,x)[0])
    x = x[quality.argmax().view(1)]
    waves, active = render(torch.cat((source[:1],x)),None,baseline,tracker,(bank,),threshold=threshold)
    logits = reader(waves.flatten(0,1))[0]
    if logits.shape!=(2*len(MODES),10) or not torch.isfinite(logits).all() or not active:
        raise ValueError('Real frozen source-only smoke failed')
    measures = {}
    for name, sample in (('source_clean',source[:1]),('generic_disturbed',x)):
        times = []
        for i in range(28):
            torch.cuda.synchronize()
            started = perf_counter()
            _, active = recover(sample,tracker,bank,threshold)
            torch.cuda.synchronize()
            if i>=8:
                times.append(1000*(perf_counter()-started))
        measures[name] = dict(active=bool(active),median_ms=float(np.median(times)))
    write(root/'smoke.json',dict(status='PASS',target_reads=0,V_reads=0,b1_all_readouts=measures,
                                 gpu=torch.cuda.get_device_name()))


def evaluate(rt, route):
    binding = source_control(rt,route,'cuda:0')
    bank, threshold = load(Path(rt['run_root']),binding[2],'cuda:0')
    score(rt,route,panel=panel,modes=MODES,render=partial(render,threshold=threshold),binding=binding,
          frozen_models=(bank,),require_clean_bypass=False,replay_fn=receive_only,finalize=freeze)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root/'state.json'
    launch = partial(start,rt,module=rt['runner'])
    try:
        for action in ('setup','smoke'):
            gpu = select_gpu()
            write(status,dict(status=action.upper(),gpu=gpu),exclusive=False)
            if launch(action,'source',gpu).wait():
                raise RuntimeError(f'LoRa empirical prior {action} failed')
        gpus = [select_gpu()]
        gpus.append(select_gpu(exclude=gpus))
        write(status,dict(status='EVALUATE',gpus=dict(zip(ROUTES,gpus,strict=True))),exclusive=False)
        jobs = [launch('evaluate',route,gpu) for route,gpu in zip(ROUTES,gpus,strict=True)]
        if any([job.wait() for job in jobs]):
            raise RuntimeError('LoRa empirical prior source predictions failed')
        write(status,dict(status='SCORE'),exclusive=False)
        if launch('score','source').wait():
            raise RuntimeError('Independent empirical prior source scoring failed')
        write(status,dict(status='SOURCE_SCORED'),exclusive=False)
    except Exception as exc:
        write(status,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','setup','smoke','evaluate','score'):
        sub = actions.add_parser(action)
        sub.add_argument('runtime',type=Path)
        if action!='dispatch':
            sub.add_argument('name',choices=ROUTES if action=='evaluate' else ('source',))
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action=='publish':
        launch_release(**args,run=RUN,runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        comparisons = (('cross_half','all_half'),('agreement_half','cross_half'),('agreement_half','direct'))
        score_frozen(rt,MODES,comparisons=comparisons)
    else:
        activate(rt)
        if action=='setup':
            setup_bank(rt)
        elif action=='smoke':
            smoke(rt)
        else:
            evaluate(rt,args['name'])


if __name__=='__main__':
    main()
