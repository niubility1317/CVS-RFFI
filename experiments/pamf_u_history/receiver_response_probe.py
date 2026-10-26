"""CPU source-L/U mixed-response diagnostic;no F/G weights or target access."""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write
from .receiver_response import Response,decompose,PairMoments,agreement,RXS,DAYS

RUN='20261026-diagnostic-PAMF-RXResponse-LU-source-r01'
RELEASE='20261026-diagnostic-PAMF-RXResponse-LU-source-release-r01'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
RECIPE=dict(L_records=6300,U_records=56700,U_H_views=170100,source_rxs=list(RXS),source_days=list(DAYS),
    window=[80,160],period=20,delay_count=320,delay_step=1/16,relative_ridge=1e-6,soft_radius=4.,smooth_complex_degrees=[1,2],
    fit_role='L_s_only',crossed_folds=18,held_TX=1,held_day=1,median_r2_min=.25,positive_folds_min=15,RX_total_variance_fraction_min=.01,
    U_cache_replica=1,U_TX_labels='absent',model_weights=0,G_weights=0,optimizer_updates=0,V_reads=0,target_reads=0)


def validate_context(c):
    if (set(c)!={'project','python','frozen_sources','source_contract','source_cache','U_channel_cache'} or c['frozen_sources']
        or c['U_channel_cache']!=c['project']+'/runs/20261026-diagnostic-PAMF-URealizations-cache-m3-r01'):
        raise ValueError('Only original source handles and existing H-only cache;no F/G/target dependencies')
    return c


def quality_summary(values):
    a=np.concatenate(values)
    return {k:dict(mean=float(a[:,j].mean()),p10=float(np.quantile(a[:,j],.1)),median=float(np.median(a[:,j]))) for j,k in enumerate(('template_quality','repeat_coherence','coherent_fraction','normalized_relative_CFO'))}


def run(rt):
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    if rt['run_id']!=RUN or not rt.get('cpu_only'):raise ValueError('Fixed CPU-only response diagnostic required')
    c=validate_context(read(rt['context']));contract=read(c['source_contract'])
    if contract['counts']!=dict(L_s=6300,U_s=56700,V=27000) or contract['source_rxs']!=list(RXS) or contract['source_days']!=list(DAYS):raise ValueError('Existing originalF0 source roles required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=True);start=time.monotonic();torch.set_num_threads(2)
    write(root/'process.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),started_at=time.time()))
    write(root/'resolved_config.json',dict(rt,recipe=RECIPE))
    L=SourceDataset(c['source_cache'],'L_s',contract);U=SourceDataset(c['source_cache'],'U_s',contract)
    if len(L)!=6300 or len(U)!=56700 or 'y' in U.metadata or L.role!='L_s' or U.role!='U_s' or set(L.metadata['eq_i'])!={1} or set(U.metadata['eq_i'])!={1}:raise ValueError('Exact equalized1 L and TX-hidden U only')
    p=Path(c['U_channel_cache']);done=read(p/'complete.json');header=read(p/'fresh1/complete.json')
    if (done['status']!='SOURCE_U_REALIZATION_CACHE_COMPLETE' or done['U_truth_read'] or done['V_reads'] or done['target_reads']
        or header['source_contract']!=contract or header['U_seen']!=56700 or header['namespace']!='PAMF/realization_diversity/U/fresh1' or header['scenes']!=list(SCENES)):
        raise ValueError('Completed same-contract sourceH-only cache required')
    with np.load(p/'fresh1/physical_index.npz',allow_pickle=False) as a:
        if not np.array_equal(a['physical'],np.stack([U.metadata[k] for k in KEYS],1)) or list(a['ids'])!=contract['role_ids']['U_s']:raise ValueError('U/H physical alignment differs')
    H=np.load(p/'fresh1/channel_U.npy',mmap_mode='r',allow_pickle=False)
    if H.shape!=(3,56700,2,256) or H.dtype!=np.float32:raise ValueError('Fixed existing H array required')
    response=Response();coeff=[];lquality=[]
    for begin in range(0,len(L),256):
        z,q=response(np.asarray(L.x[begin:begin+256].tolist(),np.float64));coeff.append(z);lquality.append(q)
        if begin==0:
            if z.shape!=(256,4) or not np.isfinite(z).all():raise ValueError('Actual L response smoke failed')
            write(root/'smoke.json',dict(status='PASS',actual_source_L=256,learned_parameters=0,F_G_weight_reads=0,optimizer_updates=0,target_reads=0))
        write(root/'progress.json',dict(stage='L',seen=min(begin+256,len(L)),total=6300),exclusive=False)
    coefficients=np.concatenate(coeff);result,private=decompose(coefficients,L.metadata['y'],L.metadata['rx_i'],L.metadata['day_i'])
    pairs=PairMoments(private['RX_top2'],private['TX_top2']);uq={v:[] for v in ('raw',*SCENES)};rxsum={r:np.zeros(4) for r in RXS};rxcount={r:0 for r in RXS}
    for begin in range(0,len(U),256):
        end=min(begin+256,len(U));rx=U.metadata['rx_i'][begin:end]
        raw,q=response(np.asarray(U.x[begin:end].tolist(),np.float64));uq['raw'].append(q)
        for r in np.unique(rx):rxsum[int(r)]+=raw[rx==r].sum(0);rxcount[int(r)]+=int((rx==r).sum())
        for j,scene in enumerate(SCENES):
            z,q=response(H[j,begin:end]);uq[scene].append(q);pairs.update(raw,z,rx,scene)
        with (root/'steps.jsonl').open('a',encoding='utf-8') as f:
            import json
            f.write(json.dumps(dict(U_seen=end,H_seen=3*end,seconds=time.monotonic()-start))+'\n')
        write(root/'progress.json',dict(stage='U_raw_3H',seen=end,total=56700,H_seen=3*end),exclusive=False)
        if begin%4096==0:print('RX_RESPONSE_U',end,'of',len(U),flush=True)
    if sum(rxcount.values())!=56700 or any(n!=11340 for n in rxcount.values()):raise ValueError('Complete fiveRX hiddenU required')
    umean=np.stack([rxsum[r]/rxcount[r] for r in RXS]);umean-=umean.mean(0)
    pair_results=pairs.result()
    if len(pair_results)!=15 or any(row['count']!=11340 for row in pair_results):raise ValueError('Complete same-packet U H coverage required')
    np.savez(root/'source_response_statistics.npz',**private)
    write(root/'result.json',dict(status='RX_RESPONSE_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],recipe=RECIPE,
        L_seen=6300,U_seen=56700,H_seen=170100,L_quality=quality_summary(lquality),U_quality={v:quality_summary(q) for v,q in uq.items()},
        L_common_structure=result,U_raw_mean_pattern_vs_L=agreement(private['RX_effect'],umean),U_same_packet_changes=pair_results,
        F_G_weight_reads=0,optimizer_updates=0,U_truth_reads=0,V_reads=0,target_reads=0,seconds=time.monotonic()-start,
        scope='source-response prerequisite only;not hardware identification,identity preservation,unseenRX estimation or F performance'))
    print('RX_RESPONSE_COMPLETE',result['common_component_supported'],flush=True)


def dispatch(rt):
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.practical_full_2x2.cache import atomic
    # Release already has its exclusive controller.log;no GPU or other-owner actions.
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    job=dict(id='response',kind='response',config=rt['runtime_file'],execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.receiver_response_probe')
    child,item=launch(rt,job,None)
    state=dict(status='RUNNING',controller_pid=os.getpid(),worker=item,code_commit=rt['code_commit'])
    atomic(root/'pipeline_state.json',state)
    code=child.wait();good=code==0 and (root/'result.json').exists()
    state.update(status='RX_RESPONSE_DIAGNOSTIC_COMPLETE' if good else 'FAILED',exit_code=code)
    root.mkdir(parents=True,exist_ok=True);atomic(root/'pipeline_state.json',state)
    if not good:raise RuntimeError('Source response failed;preserve exclusive root')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else run(c)
