"""NumPy recount of saved tensors; no model loading or metric-based reselection."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np

RUN='20261003-diagnostic-cvs-mirror-mechanism-public-m8-r01'
GROUPS=('noise','tone','periodic','dc','zero')
DELAYS=(2,8,16)


def packet_metrics(a,b):
    if a.shape != (30,2,31,4,4) or b.shape != a.shape:
        raise ValueError('Expected all thirty public signals and all mirror pairs')
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite Q')
    base=np.linalg.norm(a.reshape(30,-1),axis=1)
    changed=np.linalg.norm(b.reshape(30,-1),axis=1)
    absolute=np.linalg.norm((b-a).reshape(30,-1),axis=1)
    pair_abs=np.sqrt(np.square(b-a).sum(axis=(1,3,4)))
    pair_base=np.sqrt(np.square(a).sum(axis=(1,3,4)))
    return dict(base=base,changed=changed,absolute=absolute,relative=absolute/np.maximum(base,1e-12),
                pair_absolute=pair_abs,pair_base=pair_base)


def rows_from_arrays(arrays,variant,seed):
    packets=[];pairs=[]
    for dtype in ('fp32','fp64'):
        a=arrays[dtype+'_base_q']
        bound=1/np.sqrt(2) if variant=='mirror_subspace' else 1.
        for delay in DELAYS:
            b=arrays[f'{dtype}_d{delay}_q'];metrics=packet_metrics(a,b)
            if max(metrics['pair_base'].max(),np.sqrt(np.square(b.astype(float)).sum(axis=(1,3,4))).max())>bound+1e-5:
                raise ValueError('Theoretical pair Q bound violated')
            for i in range(30):
                embedding=np.linalg.norm(arrays[f'embedding_d{delay}'][i].astype(float)-arrays['embedding_base'][i])
                packets.append(dict(variant=variant,model_seed=seed,arithmetic=dtype,delay=delay,
                    packet=i,signal_group=GROUPS[i//6],base_norm=float(metrics['base'][i]),
                    changed_norm=float(metrics['changed'][i]),absolute_change=float(metrics['absolute'][i]),
                    relative_change=float(metrics['relative'][i]),fp32_unit_embedding_distance=float(embedding)))
                for frequency in range(31):
                    row=dict(variant=variant,model_seed=seed,arithmetic=dtype,delay=delay,packet=i,
                        signal_group=GROUPS[i//6],positive_bin=frequency+1,
                        absolute_change=float(metrics['pair_absolute'][i,frequency]),base_norm=float(metrics['pair_base'][i,frequency]))
                    for label,prefix in (('base',dtype+'_base_'),('changed',f'{dtype}_d{delay}_')):
                        singular=arrays[prefix+'singular'][i,frequency]
                        row.update({label+'_singular_max':float(singular[0]),label+'_singular_min':float(singular[1]),
                            label+'_singular_ratio':float(singular[1]/max(singular[0],1e-300)),
                            label+'_energy_floor':bool(arrays[prefix+'energy_floor'][i,frequency]),
                            label+'_determinant_floor':bool(arrays[prefix+'determinant_floor'][i,frequency]),
                            label+'_determinant':float(arrays[prefix+'determinant'][i,frequency])})
                    pairs.append(row)
    return packets,pairs


def write_csv(path,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(input_path,report):
    targets=['public_packets.csv','public_frequency_pairs.csv','public_summary.csv','cpu_gpu_comparison.csv','analysis_validation.json']
    if any((report/'evidence'/name).exists() for name in targets):
        raise FileExistsError('Preserve existing diagnostic analysis; no output replacement')
    done=json.loads((input_path/'completion.json').read_text(encoding='utf-8'))
    expected={(v,s) for v in ('mirror_energy','mirror_subspace') for s in range(2026092701,2026092705)}
    if (done['status']!='VERIFIED' or done['target_access'] or done['training'] or
            len(done['rows'])!=8 or {(r['variant'],r['model_seed']) for r in done['rows']}!=expected):
        raise ValueError('Incomplete frozen diagnostic matrix')
    packets=[];pairs=[];comparison=[]
    for row in done['rows']:
        with np.load(input_path/row['row_id']/'public_arrays.npz',allow_pickle=False) as arrays:
            p,q=rows_from_arrays(arrays,row['variant'],row['model_seed'])
        packets.extend(p);pairs.extend(q)
        for delay in DELAYS:
            cpu=[r for r in p if r['arithmetic']=='fp32' and r['delay']==delay and r['packet']<18]
            gpu=next(r for r in row['original_gpu_fir'] if r['name']=='delay'+str(delay))
            comparison.append(dict(row_id=row['row_id'],delay=delay,cpu_relative_mean=statistics.mean(r['relative_change'] for r in cpu),
                original_gpu_relative_mean=gpu['relation_relative_distance'],
                cpu_embedding_mean=statistics.mean(r['fp32_unit_embedding_distance'] for r in cpu),
                original_gpu_embedding_mean=gpu['mean_unit_embedding_distance']))
    summary=[]
    for variant in ('mirror_energy','mirror_subspace'):
        for dtype in ('fp32','fp64'):
            for delay in DELAYS:
                for group in GROUPS:
                    rows=[r for r in packets if (r['variant'],r['arithmetic'],r['delay'],r['signal_group'])==(variant,dtype,delay,group)]
                    if len(rows)!=24:raise ValueError('Incomplete four-seed/six-public-signal group')
                    summary.append(dict(variant=variant,arithmetic=dtype,delay=delay,signal_group=group,
                        **{k+'_mean':statistics.mean(r[k] for r in rows) for k in ('base_norm','changed_norm','absolute_change','relative_change','fp32_unit_embedding_distance')}))
    report.mkdir(parents=True,exist_ok=True);e=report/'evidence';e.mkdir(exist_ok=True)
    for name,rows in [('public_packets',packets),('public_frequency_pairs',pairs),('public_summary',summary),('cpu_gpu_comparison',comparison)]:write_csv(e/(name+'.csv'),rows)
    proof=dict(status='VERIFIED',models=8,public_signals_per_model=30,arithmetic_modes=['fp32','fp64'],
        delays=list(DELAYS),packet_records=len(packets),frequency_pair_records=len(pairs),
        summary_records=len(summary),target_access=False,source_IQ_access=False,optimizer_steps=0,
        all_tensors_recounted=True,artifact_root=str(input_path),new_model_selection=False,
        limitations=['Synthetic diagnostic only; not device accuracy or source identity information',
                    'CPU Torch2.10 vs original GPU Torch2.1 results are explicitly separate',
                    'FP64 is posthoc arithmetic for diagnostics; original training and checkpoint stay FP32'])
    with (e/'analysis_validation.json').open('x',encoding='utf-8') as f:
        f.write(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(proof));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();analyze(a.input,a.report)
