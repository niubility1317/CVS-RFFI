"""Recompute completed source-only timing samples; no model, IQ or truth reads."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def percentile(values, q):
    values=sorted(values);position=(len(values)-1)*q
    lower=math.floor(position);upper=math.ceil(position)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def analyze(source, output, paper=None):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    complete=read(source/'complete.json');resolved=read(source/'resolved.json')
    profiles=read(source/'operator_profiles.json');summary=read(source/'summary.json')
    assert complete['status']=='TIMING_COMPLETE' and complete['source_IQ_only'] and complete['exclusive_GPU']
    cases=complete['cases'];batches=complete['batches'];rounds=complete['rounds']
    assert len(cases)==20 and len(set(cases))==20 and batches==[1,8,32,64,128] and rounds==5
    assert resolved['source_IQ_only'] and not resolved['restored_IQ_cache']
    assert resolved['precision']=='FP32' and resolved['TF32'] is False and resolved['cases']==cases
    rows=[json.loads(line) for line in (source/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    expected={(case,batch,r) for case in cases for batch in batches for r in range(rounds)}
    keyed={(r['case'],r['batch'],r['round']):r for r in rows}
    assert len(rows)==len(keyed)==len(expected)==500 and set(keyed)==expected
    reported={(r['case'],r['batch'],r['round']):r for r in summary['results']}
    assert set(reported)==expected and summary['exclusive_GPU']
    grouped=defaultdict(list)
    for key,row in keyed.items():
        repeat=1000 if row['batch']==1 else 100
        assert row['exclusive_GPU'] and row['repeats']==repeat and row['warmup']==(100 if repeat==1000 else 20)
        assert len(row['wall_ms'])==len(row['device_ms'])==repeat and len(row['H2D_D2H_ms'])==20
        assert all(math.isfinite(v) and v>=0 for field in ('wall_ms','device_ms','H2D_D2H_ms') for v in row[field])
        assert row['incremental_peak_bytes']==max(0,row['peak_allocated_bytes']-row['resident_baseline_bytes'])
        median=statistics.median(row['wall_ms']);p95=percentile(row['wall_ms'],.95)
        assert math.isclose(median,reported[key]['median_ms'],rel_tol=1e-10,abs_tol=1e-10)
        assert math.isclose(p95,reported[key]['P95_ms'],rel_tol=1e-10,abs_tol=1e-10)
        assert math.isclose(row['batch']*1000/median,reported[key]['throughput_IQ_s'],rel_tol=1e-10)
        grouped[key[:2]].append(row)
    aggregate=[]
    for case in cases:
        for batch in batches:
            group=sorted(grouped[case,batch],key=lambda r:r['round'])
            pool=lambda field:[v for r in group for v in r[field]]
            wall=pool('wall_ms');device=pool('device_ms');transfer=pool('H2D_D2H_ms')
            medians=[statistics.median(r['wall_ms']) for r in group];median=statistics.median(wall)
            aggregate.append(dict(case=case,batch=batch,rounds=5,samples=len(wall),median_ms=median,P95_ms=percentile(wall,.95),
                device_median_ms=statistics.median(device),H2D_D2H_median_ms=statistics.median(transfer),H2D_D2H_P95_ms=percentile(transfer,.95),
                round_median_mean_ms=statistics.mean(medians),round_median_SD_ms=statistics.stdev(medians),
                round_median_min_ms=min(medians),round_median_max_ms=max(medians),throughput_IQ_s=batch*1000/median,
                cold_first_call_ms=group[0]['cold_first_call_ms'],resident_baseline_min_bytes=min(r['resident_baseline_bytes'] for r in group),
                resident_baseline_max_bytes=max(r['resident_baseline_bytes'] for r in group),incremental_peak_max_bytes=max(r['incremental_peak_bytes'] for r in group)))
    with (output/'latency_summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    by_key={(r['case'],r['batch']):r for r in aggregate}
    gpu=by_key['T4_graph_reuse_GF',1];native=by_key['T0_native_GF',1]
    verification=dict(status='VERIFIED',case_rounds=len(rows),case_batch_cells=len(aggregate),wall_calls=sum(len(r['wall_ms']) for r in rows),
        transfer_calls=sum(len(r['H2D_D2H_ms']) for r in rows),round_summary_recomputed=True,source_IQ_only=True,
        exclusive_GPU=True,default_GF_B1=gpu,same_weight_native_GF_B1=native,same_weight_GPU_speedup=native['median_ms']/gpu['median_ms'],
        F_registered_parameters=resolved['F_parameters'],G_registered_parameters=resolved['G_parameters'],
        scheduler_reservation_bytes=resolved['scheduler_reservation_bytes'],operator_profiles=profiles,
        limitations=['Fixed Fseed01/Gseed01, five timing rounds, not five model seeds.',
        'Pooled median/P95 over all recorded calls; between-round sample SD computed from five round medians.',
        'Resident baseline includes every co-resident benchmark model, Graph pools and 13GiB scheduling reservation; not per-model VRAM.',
        'Transfer boundary returns CPU classes for logits and CPU restored IQ for G-only. GPU wall excludes transfer and includes Python dispatch/events/sync.',
        'Cold first call is first call per case/batch after model loading and operator profiles; not a fresh process; model load includes F/G only.',
        'Profiler is eager and counts G43 FFT/IFFT calls; optimized residual-reuse G has27 from implementation, not this eager count.',
        'Samples verify finite outputs and per-round exclusivity; they do not independently establish route-wise argmax equivalence or new accuracy.'])
    (output/'latency_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    lines=['# 最终 T1＋DURA 正式时延结果','',
        'run：`20261021-phase1-T1-DURA-latency-r02`；完整替代旧统一时延导入失败行，原失败证据保留。',
        '',f"完整核验 20 种调用 × 5 个 batch × 5 轮＝500 个测量单元，{verification['wall_calls']} 次 GPU wall 调用和 {verification['transfer_calls']} 次传输调用；独占 RTX3090、Py3.10/Torch2.1、FP32/TF32off、CPU2。只使用固定源 L IQ 和冻结 seed01 模型。",'',
        '## B1 完整结果','',
        '| 调用 | GPU median / P95 (ms) | H2D/D2H median (ms) | 五轮 median 的 SD (ms) |',
        '|---|---:|---:|---:|']
    for case in cases:
        r=by_key[case,1];lines.append(f"| {case} | {r['median_ms']:.6f} / {r['P95_ms']:.6f} | {r['H2D_D2H_median_ms']:.6f} | {r['round_median_SD_ms']:.6f} |")
    lines+=['','## 默认在线 G＋F 的 batch 结果','','| batch | median (ms/batch) | P95 (ms/batch) | IQ/s |','|---:|---:|---:|---:|']
    for batch in batches:
        r=by_key['T4_graph_reuse_GF',batch];lines.append(f"| {batch} | {r['median_ms']:.6f} | {r['P95_ms']:.6f} | {r['throughput_IQ_s']:.2f} |")
    lines+=['',f"B1 同权重 native → 默认 Graph＋reuse 的 GPU wall median 为 {native['median_ms']:.6f} → {gpu['median_ms']:.6f} ms，配对执行路线加速 {verification['same_weight_GPU_speedup']:.3f} 倍；未改权重或精度。该数据属于最终 T1，不改标历史 F50 的 4.531972 ms。",'',
        '## 计算与口径','',f"F / G 的注册参数数为 {resolved['F_parameters']} / {resolved['G_parameters']}，合计 {resolved['F_parameters']+resolved['G_parameters']}；F / G 的 eager Conv/Linear MAC 分别为 {profiles['T1_F']['conv_linear_MACs']} / {profiles['DURA_G']['conv_linear_MACs']}。MAC 不包含 FFT 复数运算、CG 归约、功能式 cosine head、归一化和逐元素运算，不能称为总 FLOPs。",'',
        'G eager profile 实测 43 次 FFT/IFFT；默认 reuse 路径的实现预算为 27 次，两种执行路径明确区分。显存基线包含全部共驻留模型、Graph 池和 13 GiB 调度预约；CSV 单列基线与增量峰值，不能当作单个部署模型显存。', '',
        '冷调用在加载与 eager profile 后逐 case/batch 测量，不是新进程冷启动。五轮只量化计时变化；完整系统三种子科学结果与机制消融另按原矩阵闭合。有限源输入上的时延测量不能证明完整数据上的各路线预测一致性，也不能替代冻结科学测试。']
    (output/'latency_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if paper:
        table=Path(paper)/'tables/final_t1_latency.tex'
        text=[r'\begin{table}[t]',r'\centering\scriptsize',r'\caption{Final T1/DURA B1 timing on an exclusive RTX3090. Pooled statistics over five rounds (5000 GPU calls/case); transfer columns use 100 calls/case. All times are ms.}',
            r'\label{tab:final-t1-latency}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Call & GPU med. & P95 & Transfer med. & Round SD\\',r'\midrule']
        names={'F_eager':'F eager','F_graph':'F Graph','T0_native_GF':'G+F native','T4_graph_reuse_G':'G default','T4_graph_reuse_GF':'G+F default',
            'riei_ce':'RIEI CE','riei_full':'RIEI Full','drift_ce':'DRIFT CE','drift_full':'DRIFT Full','ibrff_ce':'IB-RFF CE','ibrff_full':'IB-RFF Full','sdg_ce':'SDG CE','sdg_full':'SDG Full'}
        for case,label in names.items():
            r=by_key[case,1];text.append(f"{label} & {r['median_ms']:.3f} & {r['P95_ms']:.3f} & {r['H2D_D2H_median_ms']:.3f} & {r['round_median_SD_ms']:.3f}\\\\")
        text+=[r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{table}[t]',r'\centering\footnotesize',
            r'\caption{Same-weight execution ablation at B1 (ms). P95 is for the measured online G+F call.}',
            r'\label{tab:final-t1-routes}',r'\begin{tabular}{lrrr}',r'\toprule',r'Route & G median & G+F median & G+F P95\\',r'\midrule']
        for route in ('T0_native','T1_tensor','T2_reuse','T3_graph','T4_graph_reuse'):
            g=by_key[route+'_G',1];gf=by_key[route+'_GF',1]
            text.append(f"{route.split('_')[0]} & {g['median_ms']:.3f} & {gf['median_ms']:.3f} & {gf['P95_ms']:.3f}\\\\")
        text+=[r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{table}[t]',r'\centering\footnotesize',
            r'\caption{Default online G+F batch timing. Batch latency is not single-packet waiting latency; the measured curve is nonmonotonic.}',
            r'\label{tab:final-t1-batches}',r'\begin{tabular}{rrrr}',r'\toprule',r'Batch & Median (ms) & P95 (ms) & IQ/s\\',r'\midrule']
        for batch in batches:
            r=by_key['T4_graph_reuse_GF',batch]
            text.append(f"{batch} & {r['median_ms']:.3f} & {r['P95_ms']:.3f} & {r['throughput_IQ_s']:.1f}\\\\")
        text+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
        table.write_text('\n'.join(text)+'\n',encoding='utf-8')
    return verification


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--paper')
    a=p.parse_args();result=analyze(a.input,a.output,a.paper)
    print(json.dumps({key:result[key] for key in ('status','case_rounds','wall_calls','transfer_calls','same_weight_GPU_speedup','default_GF_B1')}))
