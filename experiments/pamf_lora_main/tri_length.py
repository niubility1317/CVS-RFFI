"""Native-rate LoRa apertures for the fixed physical packet leaves."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import SOURCE, TARGET, VIEWS
from .full_packet_data import COUNTS, LIMIT
from .full_packet_views import native, source_role, target_contract
from .packet_physics import aligned_view, save_views
from .physical import ChirpResponse
from .repair_features import AlignedTrace


RUN = '20261125-phase1-lora-pamf-tri-iq-aperture-m4-r01'
SIZES = (192,384,512)
ROWS = {'control': {}, **{f'iq{n}': {'samples':n} for n in SIZES}}


def adapt(model, samples):
    if samples == 256:
        return model
    if samples not in SIZES:
        raise ValueError('Unsupported registered IQ size')
    raw = model.raw.encoder
    core = raw.core.id_backbone
    device = model.head.weight.device
    core.input_len = samples
    core.sinc.input_len = samples
    # Keep learned passbands and the physical time support of the Sinc kernels
    # in the original reader's calibrated 256-kHz coordinate system.
    sinc = core.sinc
    ratio = samples/256
    sinc.sample_rate *= ratio
    sinc.kernel_size = round(sinc.kernel_size*ratio)|1
    n = torch.arange(sinc.kernel_size,device=device,dtype=torch.float32)-(sinc.kernel_size-1)/2
    sinc.t_ = (n/sinc.sample_rate).view(1,-1)
    sinc.window_ = torch.hamming_window(sinc.kernel_size,periodic=False,alpha=.54,beta=.46,
                                        device=device).view(1,-1)
    raw.response = ChirpResponse(fft_factor=4,samples=samples).to(device)
    model.trace = AlignedTrace(samples=samples).to(device)
    for view in raw.sequence_pool.views:
        view.variable_iq_grid = True
    return model


def indices(y, rx):
    rng = np.random.default_rng(20261125)
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero((y==c)&(rx==r)),24,replace=False)
        for c in range(10) for r in range(5)
    ]))


def screen_inputs(data):
    with np.load(data/'V_metadata.npz',allow_pickle=False) as meta:
        selection = indices(meta['y'],meta['rx_i'])
        return selection, meta['y'][selection],meta['rx_i'][selection]


def practical_u(item, *, samples):
    z, rx, pid, slot = item
    return aligned_view(z, VIEWS[1+int(slot)%6], rx, pid, 'source-U', samples=samples)


def source_cache(run, source, contract, samples):
    path = run/'native_cache'/f'{samples}'
    path.mkdir(parents=True,exist_ok=False)
    pindex, _, _ = screen_inputs(source)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for role in ('L_s','U_s','V'):
            packets,rx,ids,base = source_role(source,contract,role)
            scenes = VIEWS[:1] if role=='U_s' else VIEWS
            namespace = 'source-V' if role=='V' else 'source'
            if role=='V':
                packets,rx,ids = packets[pindex],rx[pindex],ids[pindex]
            save_views(pool,path/f'{role}.npy',packets,rx,ids,namespace,scenes,samples=samples)
            if role=='U_s':
                output = np.lib.format.open_memmap(path/'U_s_practical.npy', mode='w+',dtype=np.float32,
                          shape=(COUNTS[role],6,2,samples))
                for i,z in enumerate(pool.map(partial(practical_u,samples=samples),
                                               zip(packets,rx,ids,base,strict=True),chunksize=24)):
                    output[i]=z
                output.flush()
                print('VARIABLE_U_PRACTICAL_READY',samples,len(output),flush=True)
    write(path/'complete.json',dict(status='SOURCE_READY',samples=samples,source_contract=contract['protocol'],
                                   U_truth_reads=0,target_reads=0))
    return path


def training_arrays(source, views, contract, samples, device):
    expected = dict(status='SOURCE_READY',samples=samples,source_contract=contract['protocol'],
                    U_truth_reads=0,target_reads=0)
    if read(views/'complete.json')!=expected:
        raise ValueError('Native length source cache incomplete')
    with np.load(source/'L_s_metadata.npz',allow_pickle=False) as label:
        if label['ids'].tolist()!=contract['role_ids']['L_s']:
            raise ValueError('L source physical order differs')
        yn,rn=label['y'],label['rx_i']
    with np.load(source/'U_s_metadata.npz',allow_pickle=False) as unlabeled:
        if 'y' in unlabeled.files or unlabeled['ids'].tolist()!=contract['role_ids']['U_s']:
            raise ValueError('U source must remain TX-blind with original physical order')
        ur=tensor(unlabeled['rx_i'],device)
    _,vy,vr=screen_inputs(source)
    mapped = {}
    for role,n,t in (('L_s',2800,7),('U_s',25200,1),('V',1200,7),('U_s_practical',25200,None)):
        values=np.load(views/f'{role}.npy',mmap_mode='r',allow_pickle=False)
        shape=(n,6,2,samples)
        if t is not None:
            shape=(t,*shape)
        if values.shape!=shape or values.dtype!=np.float32:
            raise ValueError(f'Variable IQ shape differs for {role}')
        mapped[role]=values
    return ((tensor(mapped['L_s'],device),yn,rn),
            (tensor(mapped['V'],device),tensor(vy,device),tensor(vr,device)),
            (tensor(mapped['U_s'][0],device),ur),tensor(mapped['U_s_practical'],device))


def target_cache(run, source, samples):
    target_contract(source)
    path=run/'native_target'/f'{samples}'
    path.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for rx in TARGET:
            raw = native(source,rx,LIMIT)
            save_views(pool,path/f'{rx}.npy',raw,[rx]*LIMIT,range(LIMIT),'target',VIEWS,samples=samples)
    write(path/'complete.json',dict(status='TARGET_READY',samples=samples,target_truth_read=False))
    return path
