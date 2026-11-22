"""Recover 1 MHz preambles for the registered full Phase 1 LoRa split."""
import argparse
from pathlib import Path
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import ROLES, SOURCE, TARGET, packet_indices, recording
from .packet_data import CHIRPS, PACKET_LEN

PROTOCOL='lora_full_packets_v1'
COUNTS={'L_s':2800,'U_s':25200,'V':12000}
LIMIT=2000


def iq_rows(h5, count):
    """Read interleaved-in-file I and Q columns once per physical packet."""
    x=h5['data']
    if x.shape!=(count,2*PACKET_LEN):
        raise ValueError('Full LoRa recording physical shape differs')
    result=np.empty((count,PACKET_LEN),dtype=np.complex64)
    for i in range(0,count,512):
        sl=slice(i,min(i+512,count))
        block=x[sl]
        result[sl].real=block[:,:PACKET_LEN]
        result[sl].imag=block[:,PACKET_LEN:]
        if not np.isfinite(result[sl]).all():
            raise ValueError('Nonfinite captured LoRa IQ')
    return result


def original_source(previous):
    prior=read(previous/'contract.json')
    if (prior['protocol']!='lora_rx5_main_v1' or prior['receiver_names']!=list(SOURCE)
            or prior['source_rxs']!=list(range(5)) or prior['counts']!=COUNTS):
        raise ValueError('The original registered full LoRa source contract is required')
    roles={}
    for role in ROLES:
        with np.load(previous/f'{role}_metadata.npz',allow_pickle=False) as m:
            if ('y' in m.files)!=(role!='U_s') or m['ids'].tolist()!=prior['role_ids'][role]:
                raise ValueError('Physical source roles or TX label permissions differ')
            roles[role]={k:m[k] for k in m.files}
    seen=np.concatenate([roles[r]['base_index'] for r in ROLES])
    if not np.array_equal(np.sort(seen),np.arange(40000)):
        raise ValueError('The physical source index must appear in exactly one role')
    return roles,prior['role_ids']


def prepare_source(archive_dir,previous,out,scratch):
    roles,ids=original_source(previous)
    maps={r:np.lib.format.open_memmap(out/f'{r}_packets.npy',mode='w+',dtype=np.complex64,
              shape=(COUNTS[r],PACKET_LEN)) for r in ROLES}
    with zipfile.ZipFile(archive_dir/'multiple_receiver_train.zip') as z:
        for rx,name in enumerate(SOURCE):
            with recording(z,f'{name}_train.h5',scratch) as p:
                capture=iq_rows(p,8000)
            order=packet_indices(8000,rx)
            for role,m in roles.items():
                choose=m['rx_i']==rx
                base=m['base_index'][choose]-8000*rx
                if not len(base) or np.any(base<0) or np.any(base>=8000):
                    raise ValueError('Physical source receiver binding differs')
                maps[role][choose]=capture[order[base]]
            print('FULL_NATIVE_SOURCE',name,len(capture),flush=True)
    for role in ROLES:
        maps[role].flush()
        np.savez(out/f'{role}_metadata.npz',**roles[role])
    write(out/'contract.json',dict(protocol=PROTOCOL,source_only=True,chirps=list(CHIRPS),
          source_receivers=list(SOURCE),counts=COUNTS,role_ids=ids,input_fs=1_000_000))


def prepare_target(archive_dir,out,scratch):
    with zipfile.ZipFile(archive_dir/'multiple_receiver_test.zip') as z:
        for rx,name in enumerate(TARGET):
            with recording(z,f'{name}_test.h5',scratch) as p:
                capture=iq_rows(p,LIMIT)
            np.save(out/f'{name}_packets.npy',capture[packet_indices(LIMIT,rx)])
            print('FULL_NATIVE_TARGET_BLIND',name,len(capture),flush=True)
    write(out/'contract.json',dict(protocol=PROTOCOL,receivers=list(TARGET),per_receiver=LIMIT,
          TX_truth_read=False,input_fs=1_000_000))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('role',choices=('source','target'))
    for name in ('archives','output','scratch'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous',type=Path)
    a=p.parse_args()
    if a.role=='source' and a.previous is None:
        p.error('source conversion needs the previously registered metadata')
    a.output.mkdir(parents=True,exist_ok=False)
    a.scratch.mkdir(parents=True,exist_ok=True)
    if a.role=='source':
        prepare_source(a.archives,a.previous,a.output,a.scratch)
    else:
        prepare_target(a.archives,a.output,a.scratch)
