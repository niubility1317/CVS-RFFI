"""Train matched LoRa empirical-prior arms using disjoint source U roles."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind_source import FitRecipe, fit, setup
from .data import SOURCE
from .dura_repair import PacketRestorer
from .empirical_tangent import EmpiricalTangent
from .full_packet_views import source_contract
from .joint_unroll_source import coordinates
from .local_phase import generic_phase_observation
from .local_wiener_source import ROUTES, score_frozen
from .natural_detail import natural_objective
from .refine_runner import select_gpu, start
from .runner import activate
from .spectral_block_source import evaluate, paired_tasks, publish, restore, smoke


RUN = '20261124-phase1-lora-empirical-tangent-source-m4-r01'
ARMS = ('matched','protected')
READOUTS = ('coarse','full','half')
MODES = ('direct','reference',*(f'{arm}_{mode}' for arm in ARMS for mode in READOUTS))
COMPARISONS = (*(('protected_'+mode,'matched_'+mode) for mode in READOUTS),
               ('protected_half','direct'),('protected_full','protected_coarse'),
               *((name,'reference') for name in MODES[2:]))
BANK_PER_RX, FIT_PER_RX = 1000,3400
BANK_RECORD = dict(templates=BANK_PER_RX*len(SOURCE),disjoint_fit=FIT_PER_RX*len(SOURCE),
                   F_reads=0,V_reads=0,TX_reads=0,target_reads=0,practical_reads=0)


@torch.no_grad()
def bank(rt):
    begun = perf_counter()
    device = 'cuda:0'
    root = Path(rt['run_root'])
    joint = FIT_PER_RX+BANK_PER_RX
    (iq, _, _), contract = setup(rt,device,fit_per_rx=joint)
    chosen = iq.unflatten(0,(len(SOURCE),joint))[:,:BANK_PER_RX].flatten(0,1)
    tracker = PacketRestorer('consensus').to(device).eval()
    library = torch.cat([torch.fft.fft(coordinates(tracker,rows)[0],norm='ortho') for rows in chosen.split(64)])
    if not torch.isfinite(library).all():
        raise ValueError('Nonfinite receive-only LoRa template')
    with (root/'bank.pt').open('xb') as f:
        torch.save(dict(library=library,contract=contract),f)
    write(root/'bank.json',dict(status='SOURCE_BANK_READY',**BANK_RECORD,
          storage_bytes=library.numel()*library.element_size(),wall_seconds=perf_counter()-begun))


def recipes(rt):
    root = Path(rt['run_root'])
    config = read(root/'bank.json')
    payload = torch.load(root/'bank.pt',map_location='cuda:0',weights_only=True)
    library = payload['library']
    if (payload['contract']!=source_contract(Path(rt['source_root'])) or config['status']!='SOURCE_BANK_READY'
            or any(config[k]!=v for k,v in BANK_RECORD.items())
            or library.shape!=(BANK_RECORD['templates'],6,256) or library.dtype!=torch.complex64):
        raise ValueError('Expected the fixed source-U bank and its matching waveform contract')
    return {arm:FitRecipe(arm,40_000,partial(EmpiricalTangent,library,protect=arm=='protected'),
                          generic_phase_observation,restore,coarse_first=True,batch=24,fit_per_rx=FIT_PER_RX,
                          validation_steps=(1800,6000,12000,20000,30000),bank_per_rx=BANK_PER_RX) for arm in ARMS}


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    launch = partial(start,rt,module=rt['runner'])
    try:
        for name in ('bank','fit_smoke','fit','smoke','evaluate','score'):
            if name in ('fit_smoke','fit','evaluate'):
                paired_tasks(rt,name,ROUTES if name=='evaluate' else ARMS)
            else:
                gpu = select_gpu() if name!='score' else None
                write(state,dict(status=name.upper(),gpu=gpu),exclusive=False)
                if launch(name,'source',gpu).wait():
                    raise RuntimeError(f'LoRa empirical tangent {name} failed')
        write(state,dict(status='SOURCE_SCORED'),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    pub = commands.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','bank','fit_smoke','fit','smoke','evaluate','score'):
        command = commands.add_parser(action)
        command.add_argument('runtime',type=Path)
        if action!='dispatch':
            command.add_argument('name',choices=ARMS if action in ('fit','fit_smoke') else ROUTES if action=='evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args,run=RUN,runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        score_frozen(rt,MODES,comparisons=COMPARISONS)
    else:
        activate(rt)
        if action=='bank':
            bank(rt)
        else:
            bound = recipes(rt)
            if action in ('fit','fit_smoke'):
                fit(rt,smoke=action=='fit_smoke',recipe=bound[args['name']],receiver_batches=True,
                    calculate=partial(natural_objective,preserve=.50))
            elif action=='smoke':
                smoke(rt,bound,MODES)
            else:
                evaluate(rt,args['name'],bound,MODES)


if __name__=='__main__':
    main()
