"""Aggregate complete frozen-checkpoint deployment measurements, no model/data access."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np

def dump_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--hardware',type=Path);a=p.parse_args()
    raw=json.loads(a.input.read_text(encoding='utf-8'));data=raw['results']
    if raw['status']!='VERIFIED' or raw['models']!=104 or len(data)!=208 or raw['dataset_read'] or raw['truth_read']:raise ValueError('Incomplete or wrong resource benchmark')
    variants=list(dict.fromkeys(r['variant'] for r in data))
    if len(variants)!=13 or 'native' in variants:raise ValueError('Wrong models')
    rows=[];trials=[];summaries=[]
    for v in variants:
        for batch in (1,128):
            group=sorted([r for r in data if r['variant']==v and r['batch_size']==batch],key=lambda r:r['model_seed'])
            if len(group)!=8 or {r['model_seed'] for r in group}!=set(range(2026092701,2026092709)):raise ValueError('Missing seed')
            pooled=[]
            for r in group:
                values=r['latency_trials_ms']
                if len(values)!=(300 if batch==1 else 100) or min(values)<=0:raise ValueError('Invalid trials')
                if abs(statistics.mean(values)-r['mean_ms'])>1e-10:raise ValueError('Recorded mean mismatch')
                for q in (50,95,99):
                    if abs(float(np.percentile(values,q))-r[f'p{q}_ms'])>1e-10:raise ValueError('Recorded percentile mismatch')
                if r['hardware']!='NVIDIA GeForce RTX 3090' or r['precision']!='FP32' or r['tf32'] or r['truth_read'] or r['dataset_read']:raise ValueError('Different conditions')
                if r['gpu_peak_reserved_bytes']<r['gpu_peak_allocated_bytes'] or r['cpu_inference_peak_rss_bytes']<r['cpu_baseline_rss_bytes'] or r['model_storage_bytes']<r['resident_state_bytes']:raise ValueError('Invalid memory/storage')
                row={k:r[k] for k in ('variant','model_seed','batch_size','trials','warmups','mean_ms','sd_ms','p50_ms','p95_ms','p99_ms','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','cpu_baseline_rss_bytes','cpu_inference_peak_rss_bytes','model_storage_bytes','resident_state_bytes','total_parameters')}
                rows.append(row);pooled.extend(values)
                trials.extend(dict(variant=v,model_seed=r['model_seed'],batch_size=batch,trial=i,end_to_end_ms=x) for i,x in enumerate(values,1))
            storage={r['model_storage_bytes'] for r in group};state={r['resident_state_bytes'] for r in group}
            if len(storage)!=1 or len(state)!=1:raise ValueError('Storage differs across fixed architecture seeds')
            summaries.append(dict(variant=v,batch_size=batch,actual_seed_count=8,request_count=len(pooled),
                end_to_end_mean_ms=statistics.mean(pooled),seed_mean_sd_ms=statistics.stdev(r['mean_ms'] for r in group),
                p50_ms=float(np.percentile(pooled,50)),p95_ms=float(np.percentile(pooled,95)),p99_ms=float(np.percentile(pooled,99)),
                gpu_peak_allocated_mib=max(r['gpu_peak_allocated_bytes'] for r in group)/2**20,
                gpu_peak_reserved_mib=max(r['gpu_peak_reserved_bytes'] for r in group)/2**20,
                cpu_peak_rss_mib=max(r['cpu_inference_peak_rss_bytes'] for r in group)/2**20,
                cpu_baseline_rss_mean_mib=statistics.mean(r['cpu_baseline_rss_bytes'] for r in group)/2**20,
                model_storage_mib=next(iter(storage))/2**20,resident_state_mib=next(iter(state))/2**20,
                end_to_end_throughput_samples_s=batch*1000/statistics.mean(pooled)))
    notes=dict(run_id=raw['run_id'],hardware=raw['hardware'],precision='FP32 / TF32 disabled',cuda_version=data[0]['cuda_version'],torch_version=data[0]['torch_version'],gpu=0,
        timing_scope=data[0]['timing_scope'],storage_scope=data[0]['storage_scope'],input_kind=data[0]['input_kind'],input_seed=data[0]['input_seed'],
        memory_scope='Max across eight checkpoints of inference-window PyTorch allocated/reserved peaks after warmup/cache clearing. CPU total-process RSS sampled every1ms in separate50-call memory phase; may miss narrower spikes; baseline includes framework/runtime.',
        percentiles='Pool equal-count trials across eight fixed model seeds:2400 requests B1,800 B128 per model. Quantiles use linear interpolation, not mean of per-seed quantiles.',
        uncertainty='SD of eight per-checkpoint mean latencies is distinct from within-checkpoint per-request SD and P95/P99.',
        boundaries='Synthetic normalized IQ, real frozen E200 checkpoints, no real dataset/truth or accuracy reevaluation. No RF acquisition/front-end synchronization/network/disk/queue. Desktop RTX3090 software chain, not satellite hardware evidence.')
    if a.hardware:notes.update(json.loads(a.hardware.read_text(encoding='utf-8')))
    a.output.mkdir(parents=True,exist_ok=False)
    dump_csv(a.output/'deployment_measurements.csv',rows);dump_csv(a.output/'deployment_trials.csv',trials);dump_csv(a.output/'deployment_summary.csv',summaries)
    (a.output/'deployment_input.json').write_text(json.dumps(dict(rows=rows,trials=trials,summary=summaries,notes=notes),ensure_ascii=False),encoding='utf-8')
    (a.output/'measurement_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',models=104,profiles=len(rows),requests=len(trials),summary_rows=len(summaries))))

if __name__=='__main__':main()
