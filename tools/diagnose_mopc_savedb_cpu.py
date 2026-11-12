"""Compare original and explicitly clipped MoPC on one saved B/support, CPU only."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase2_practical.io import activate,read,require
from experiments.pamf_phase2_practical.external_execution import seed_for
from experiments.pamf_phase2_practical.group1 import source_scope
from experiments.pamf_phase2_practical.group1_source import load_source
from experiments.pamf_phase2_practical.group2_b_state import load_b_state
from experiments.pamf_phase2_practical.group2_c_backbone_worker import support_only
from experiments.pamf_phase2_practical.group2_cross_fit import register_c
from experiments.pamf_phase2_practical.clean_source_smoke import check


BASE=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUNTIME=BASE/'releases/20261109-phase2-bsource-savedb-mopc-m4-r15-release/runtime.json'
JOB=BASE/'runs/20261109-phase2-bsource-savedb-mopc-m4-r15/jobs/00002.json'


def main():
    torch.set_num_threads(1);rt=read(RUNTIME);job=read(JOB);activate(rt)
    require(job['key']=='protonet_cda-s2026092702-c0195' and [10,20] in job['points'],'Wrong original diagnostic')
    sr=source_scope(rt);sr['run_root']=rt['source_run_root']
    bundle=load_source(sr,job['profile'],job['seed'],'cpu');check(rt,bundle)
    cap=Path(rt['data_root'])/job['capsule'];manifest=read(cap/'manifest.json')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE','Invalid capsule')
    specs={(v['k'],v['new_count']):v for v in (read(p) for p in (cap/'splits').glob('*.json'))}
    old_id=specs[10,0]['split_id'];new_id=specs[10,20]['split_id']
    _,_,old_positions,_=support_only(cap,old_id)
    x,y,positions,_=support_only(cap,new_id);mask=y<6
    require(np.array_equal(old_positions,positions[mask]),'B/C support identities differ')
    b_seed=seed_for(int(job['seed']),job['capsule'],10,'B')
    c_seed=seed_for(int(job['seed']),job['capsule'],10,20,'C')
    b_path=Path(job['load_B_for_mopc'])/'B-K10.pt'
    new_x=torch.tensor(x[~mask].tolist(),dtype=torch.float32)
    new_y=torch.tensor(y[~mask].tolist(),dtype=torch.long)
    results=[]
    for cap_value in (None,1.0):
        anchor=load_b_state(b_path,job,10,old_id,b_seed,'cpu')['anchor'];start=time.perf_counter()
        try:
            state=register_c({'mopc_hr':anchor},new_x,new_y,seed=c_seed,methods=('mopc_hr',),
                             mopc_max_grad_norm=cap_value)['mopc_hr']
            item=dict(clip=cap_value,status='COMPLETE',updates=len(state.loss_trace),
                      max_loss=max(v['loss'] for v in state.loss_trace),
                      clipped_updates=state.protocol_metadata.get('gradient_clipped_updates',0))
        except FloatingPointError as exc:item=dict(clip=cap_value,status='NONFINITE',error=str(exc))
        item['seconds']=time.perf_counter()-start;results.append(item)
    print(json.dumps(dict(status='CPU_SUPPORT_DIAGNOSTIC_COMPLETE',task=job['key'],k=10,n=20,
        saved_b=str(b_path),device='cpu',source_only_smoke='PASS',results=results,query_rows=0,truth_read=False,
        identity_predictions=0,official_retry=False,claim='CPU and GPU kernels differ; clipping changes optimization'),allow_nan=False))


if __name__=='__main__':main()
