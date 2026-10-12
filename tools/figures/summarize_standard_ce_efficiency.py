"""Aggregate recorded resource measurements; repair fused sequence MAC undercounts.

No checkpoint, IQ, truth, inference or training is accessed. Sequence costs are
derived from the registered module weight shapes and fixed 64-token input.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.standard_ce_baselines.model import build

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def sequence_macs(variant):
    model=build(variant);length=64
    parameters=sum(p.numel() for p in model.parameters())
    if variant=='bilstm3':
        recurrent=sum(p.numel() for name,p in model.encoder.named_parameters() if name.startswith('weight_'))
        result=length*recurrent+model.classifier.weight.numel()
    else:
        result=length*model.embedding.weight.numel()+model.classifier.weight.numel()
        for layer in model.encoder.layers:
            result+=length*(layer.self_attn.in_proj_weight.numel()+layer.self_attn.out_proj.weight.numel()+
                            layer.linear1.weight.numel()+layer.linear2.weight.numel())
            result+=2*length*length*layer.self_attn.embed_dim
    return parameters,result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=read_csv(args.source/'resource_profiles.csv');accuracy=read_csv(args.source/'network_summary.csv')
    if len(rows)!=104 or len(accuracy)!=13 or any(r['variant']=='native' for r in rows):raise ValueError('Wrong matrix')
    sequence={v:sequence_macs(v) for v in ('bilstm3','transformer6')}
    output=[]
    for row in rows:
        v=row['variant'];n=int(row['total_parameters']);raw=int(row['conv_linear_macs_per_sample'])
        if row['hardware']!='NVIDIA GeForce RTX 3090':raise ValueError('Different hardware')
        macs=sequence[v][1] if v in sequence else raw
        if v in sequence and n!=sequence[v][0]:raise ValueError('Sequence architecture mismatch')
        if macs<raw:raise ValueError('Invalid fused-op correction')
        b1=float(row['inference_batch1_ms']);b128=float(row['inference_batch128_ms'])
        output.append(dict(variant=v,network=row['network'],model_seed=int(row['model_seed']),
            parameters=n,trainable_parameters=int(row['trainable_parameters']),gradient_used_parameters=int(row['gradient_used_parameters']),
            fp32_weights_mib=4*n/2**20,resident_state_mib=int(row['resident_state_bytes'])/2**20,
            recorded_partial_macs=raw,matrix_macs=macs,matrix_flops=2*macs,fft_calls=int(row['fft_calls_per_sample']),
            inference_b1_ms=b1,inference_b128_ms=b128,throughput_b128_samples_s=128000/b128,
            per_sample_b128_us=1000*b128/128,
            source_training_peak_mib=int(row['training_peak_cuda_allocated_bytes'])/2**20,
            clean_inference_peak_mib=int(row['full_clean_peak_cuda_allocated_bytes'])/2**20,
            e200_training_minutes=float(row['training_elapsed_seconds'])/60,
            training_b128_ms=float(row['training_batch128_ms']),
            clean_pipeline_seconds=float(row['full_clean_inference_seconds']),
            clean_pipeline_samples_s=168000/float(row['full_clean_inference_seconds'])))
    summary=[]
    fields=[k for k in output[0] if k not in ('variant','network','model_seed')]
    for a in accuracy:
        group=[r for r in output if r['variant']==a['variant']]
        if {r['model_seed'] for r in group}!=set(range(2026092701,2026092709)):raise ValueError('Incomplete seeds')
        item=dict(variant=a['variant'],network=a['network'],actual_seed_count=8,
            accuracy_pct=float(a['accuracy_pct']),accuracy_sd_pp=float(a['accuracy_sd_pp']),
            macro_f1_pct=float(a['macro_f1_pct']),macro_f1_sd_pp=float(a['macro_f1_sd_pp']))
        for field in fields:
            item[field+'_mean']=statistics.mean(r[field] for r in group)
            item[field+'_sd']=statistics.stdev(r[field] for r in group)
        summary.append(item)
    notes=dict(input_shape=[2,256],precision='FP32',gpu='RTX 3090',seed_count=8,
        timing='Recorded post-training synthetic measurements: 5 warmups, 20 timed forwards; timings are means per call. Shared-GPU historical runs, not exclusive latency trials.',
        throughput='128000 / B128 milliseconds per seed, then mean and sample SD across eight seeds.',
        macs='Convolution and matrix MACs. Recorded counts retained separately; analytic recurrent and full attention matrices repair fused LSTM/Transformer omissions.',
        flops='2 x matrix/convolution MACs by convention; excludes FFT, normalization, nonlinear activation, pooling, bias additions, physical/elementwise operations. Not total-network FLOPs.',
        fft='CVS invokes two FFT operations per sample; calls are separately shown, not zero-cost operations.',
        memory='Recorded CUDA allocated peaks; not reserved/process GPU memory. Clean prediction uses B128. Resident FP32 model state includes buffers; FP32 weights are parameter count x 4 bytes.',
        pipeline='Full clean timing includes input transfer, prediction readback and prediction-file serialization, unlike synthetic forward latency.',
        uncertainty='Sample SD across model seeds/runs (ddof=1); not a confidence interval.',
        source_runs=['20261010-phase1-standard-ce-baselines-manysig-m56-r01',
                     '20261012-phase1-standard-ce-seed-extension-manysig-m52-r01'])
    args.output.mkdir(parents=True);write_csv(args.output/'efficiency_per_seed.csv',output)
    write_csv(args.output/'efficiency_summary.csv',summary)
    (args.output/'efficiency_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.output/'efficiency_input.json').write_text(json.dumps(dict(rows=output,summary=summary,notes=notes),ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',models=104,networks=13,fused_sequence_macs=sequence)))

if __name__=='__main__':main()
