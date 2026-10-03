"""Recompute the actual 40-row/320-record matrix and all frozen controls."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
from experiments.cvs_channel_response_clean.prepare import RUN, SOURCE_RUN, OLD_RUN
from experiments.cvs_channel_response_clean.collect import CLASSES, validate_terminal
from experiments.cvs_channel_response_identity.model import VARIANTS, response_contract

BASELINES=('native','cvcnn','real_cnn','resnet1d','residual_fusion','energy_equivariant',
           'coupled_lag4','adaptive_volterra_lag4','neural_residual_shallow','channel_dual')
SEEDS=tuple(range(2026092701,2026092705))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)
def close(actual,expected):
    return isinstance(expected,(int,float)) and math.isfinite(expected) and abs(actual-expected)<=1e-12


def validate_results(data,selection,previous):
    candidate=selection.get('selected_variant')
    if (candidate not in VARIANTS or selection.get('status')!='SOURCE_SELECTION_FROZEN' or
            selection.get('scope')!='channel_response_source' or selection.get('new_candidate_selected') is not True or
            selection.get('target_access') is not False or selection.get('target_score_used') is not False):
        raise ValueError('Selected new source-only channel candidate required')
    if data.get('classes')!=CLASSES:raise ValueError('Frozen class map differs')
    marker=data['marker'];records=data['results']['results'];summary=data['summary']['summary'];pairs=data['summary']['paired']
    expected_marker=dict(status='SCORED_COMPLETE',rows=44,models=11,seeds=4,records=352,query_count=168000,view='clean',target_feedback_forbidden=True)
    if any(marker.get(k)!=v for k,v in expected_marker.items()) or len(records)!=352:
        raise ValueError('Incomplete registered forty-row clean matrix')
    methods=(*BASELINES,candidate)
    receivers=sorted({r['receiver'] for r in records if r['receiver']!='ALL'})
    lookup={(r['method'],r['receiver'],r['model_seed']):r for r in records}
    expected={(m,rx,s) for m in methods for rx in ('ALL',*receivers) for s in SEEDS}
    if len(receivers)!=7 or len(lookup)!=352 or set(lookup)!=expected:
        raise ValueError('Missing/duplicate method,receiver,seed')
    for row in records:
        if row.get('view')!='clean' or row.get('row_id')!=row['method']+'-s'+str(row['model_seed']):
            raise ValueError('Score view or row identity differs')
        cm=np.asarray(row['confusion']);count=168000 if row['receiver']=='ALL' else 24000
        per_class=28000 if row['receiver']=='ALL' else 4000
        if (cm.shape!=(6,6) or not np.issubdtype(cm.dtype,np.integer) or np.any(cm<0) or
                int(cm.sum())!=count or row['query_count']!=count or np.any(cm.sum(1)!=per_class)):
            raise ValueError('Confusion schema, fixed count, or TX denominator mismatch')
        true=np.diag(cm);den=cm.sum(0)+cm.sum(1)
        actual=dict(accuracy=float(true.sum()/count),macro_accuracy=float((true/cm.sum(1)).mean()),
                    macro_f1=float(np.divide(2*true,den,out=np.zeros(6),where=den>0).mean()))
        if any(not close(v,row.get(k)) for k,v in actual.items()):raise ValueError('Independent confusion metrics differ')
    for method in methods:
        for seed in SEEDS:
            total=np.asarray(lookup[method,'ALL',seed]['confusion'])
            partition=sum((np.asarray(lookup[method,rx,seed]['confusion']) for rx in receivers),np.zeros((6,6),dtype=np.int64))
            if not np.array_equal(total,partition):raise ValueError('RX partition does not sum to overall confusion')
    old=previous['results']['results'];old_lookup={(r['method'],r['receiver'],r['model_seed']):r for r in old}
    old_expected={(m,rx,s) for m in BASELINES for rx in ('ALL',*receivers) for s in SEEDS}
    if len(old)!=320 or set(old_lookup)!=old_expected or any(lookup[k]!=r for k,r in old_lookup.items()):
        raise ValueError('Frozen original 320 control metrics changed or incomplete')
    summary_expected={(m,rx) for m in methods for rx in ('ALL',*receivers)}
    if len(summary)!=88 or {(r['method'],r['receiver']) for r in summary}!=summary_expected:
        raise ValueError('Incomplete/duplicate summary')
    for row in summary:
        if row.get('view')!='clean' or row.get('model_seeds')!=list(SEEDS) or row.get('query_count_per_seed')!=(168000 if row['receiver']=='ALL' else 24000):
            raise ValueError('Summary seed/view/count contract differs')
        for key in ('accuracy','macro_accuracy','macro_f1'):
            values=[lookup[row['method'],row['receiver'],seed][key] for seed in SEEDS]
            if not close(st.mean(values),row.get(key+'_mean')) or not close(st.stdev(values),row.get(key+'_seed_sd')):
                raise ValueError('Independent four-seed summary differs')
    pair_expected={(m,rx) for m in BASELINES for rx in ('ALL',*receivers)}
    if len(pairs)!=80 or {(r['baseline'],r['receiver']) for r in pairs}!=pair_expected:
        raise ValueError('Incomplete/duplicate paired comparisons')
    for row in pairs:
        values=[100*(lookup[candidate,row['receiver'],seed]['accuracy']-lookup[row['baseline'],row['receiver'],seed]['accuracy']) for seed in SEEDS]
        if (row.get('view')!='clean' or row['candidate']!=candidate or row['model_seeds']!=list(SEEDS) or
                len(row['accuracy_delta_pp_by_seed'])!=4 or any(not close(v,p) for v,p in zip(values,row['accuracy_delta_pp_by_seed']))):
            raise ValueError('Paired seed mapping differs')
        if (not close(st.mean(values),row.get('accuracy_delta_pp_mean')) or not close(st.stdev(values),row.get('accuracy_delta_pp_seed_sd')) or
                row['positive_seeds']!=sum(v>0 for v in values)):
            raise ValueError('Independent paired mean/SD/sign differs')
    return lookup,receivers


def analyze(root):
    folder=root/'automation_reports/CV-SincNet'/RUN;e=folder/'evidence'
    data=read(e/'final_readback.json');selection=read(e/'performance_selection.json')
    validate_terminal(data,selection)
    previous=read(root/'automation_reports/CV-SincNet'/OLD_RUN/'evidence/final_readback.json')
    lookup,receivers=validate_results(data,selection,previous);candidate=selection['selected_variant']
    records=data['results']['results'];summary=data['summary']['summary'];pairs=data['summary']['paired']
    methods=(*BASELINES,candidate);by={(r['method'],r['receiver']):r for r in summary}
    gains={r['baseline']:r for r in pairs if r['receiver']=='ALL'};tx=[]
    for method in methods:
        for i,name in enumerate(CLASSES):
            values=[lookup[method,'ALL',seed]['confusion'][i][i]/28000 for seed in SEEDS]
            tx.append(dict(method=method,transmitter=name,query_count_per_seed=28000,accuracy_mean=st.mean(values),accuracy_seed_sd=st.stdev(values)))
    source=read(root/'automation_reports/CV-SincNet'/SOURCE_RUN/'evidence/source_research_complete.json')
    models=[r for r in source['rows'] if r['resolved']['variant']==candidate]
    if len(models)!=4 or {r['resolved']['model_seed'] for r in models}!=set(SEEDS):raise ValueError('Selected source resource seed matrix differs')
    new={r['resolved']['model_seed']:r for r in data['rows'] if r['resolved']['variant']==candidate}
    architecture=response_contract(candidate);parameters=architecture['base_trainable_parameters']+architecture['new_trainable_parameters']
    resources=[]
    for row in models:
        profile=row['profile'];seed=row['resolved']['model_seed'];prediction=new[seed]['completion']
        if profile['total_parameters']!=parameters or profile['trainable_parameters']!=parameters:
            raise ValueError('Actual source parameter count differs')
        resources.append(dict(model_seed=seed,parameters=parameters,trainable_parameters=profile['trainable_parameters'],
            resident_state_bytes=profile['resident_state_bytes'],conv_linear_MACs=profile['conv_linear_macs_per_sample'],
            mac_scope=profile['mac_scope'],hardware=profile['hardware'],torch_version=profile['torch_version'],
            inference_batch1_ms=profile['inference_batch1_ms'],training_batch128_ms=profile['training_batch128_ms'],
            training_peak_bytes=max(r['peak_cuda_allocated_bytes'] for r in row['epochs']),
            full_clean_prediction_seconds=prediction['prediction_seconds'],prediction_peak_bytes=prediction['peak_cuda_allocated_bytes'],
            CPU_peak_bytes=None,onboard_transmission_bytes=None))
    csvwrite(e/'per_transmitter_summary.csv',tx);csvwrite(e/'clean_summary.csv',summary)
    csvwrite(e/'clean_paired.csv',pairs);csvwrite(e/'clean_scored_results.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in records])
    write(e/'clean_scored_results.json',data['results'])
    write(e/'resource_summary.json',dict(rows=resources,response_contract=architecture,
        scope='Measured same-run hardware; concurrency affects timing; CPU peak and onboard transmission unmeasured'))
    candidate_all=by[candidate,'ALL'];control=gains['channel_dual']
    verdict=dict(status='VERIFIED',candidate=candidate,accuracy_mean=candidate_all['accuracy_mean'],accuracy_seed_sd=candidate_all['accuracy_seed_sd'],
        paired_vs_channel_dual=control,all352_confusions_recomputed=True,old320_metrics_exactly_unchanged=True,
        all88_summaries_recomputed=True,all80_pairs_recomputed=True,all_RX_sums_match=True,target_feedback=False,
        response_compensation_active=architecture['response_compensation_active'],response_order_active=architecture['response_order_active'],
        scientific_verdict='POSITIVE_MEAN_ON_FIXED_CLEAN_BENCHMARK' if control['accuracy_delta_pp_mean']>0 else 'NO_MEAN_IMPROVEMENT_ON_FIXED_CLEAN_BENCHMARK',
        scope='Four model seeds on historically exposed clean benchmark; not broad statistical or deployment proof')
    write(e/'analysis_validation.json',verdict);write(e/'iteration_verdict.json',verdict)
    pct=lambda v:f'{100*v:.4f}'
    text='# CVS 显式补偿响应：独立 clean 结果\n\n'
    text+=f"源规则选中 `{candidate}`。四 seed 准确率为 {pct(candidate_all['accuracy_mean'])}% ± {pct(candidate_all['accuracy_seed_sd'])}%，较双路控制 {control['accuracy_delta_pp_mean']:+.4f} 个百分点。结论：`{verdict['scientific_verdict']}`。\n\n"
    text+=f"本候选实际启用补偿分支：`{architecture['response_compensation_active']}`；运算次序差分支：`{architecture['response_order_active']}`。归因限于该实际结构，未选候选不访问 query。\n\n"
    text+='四 seed 使用同一物理划分，覆盖模型与加载器随机性，不能替代独立数据域重复。本轮均值和配对差值仅描述该固定基准。下表保留所有负差值，不据此调整候选、超参数或选择性重跑。\n\n'
    text+='|模型|准确率/% ± seed SD|Macro-F1/% ± seed SD|\n|---|---:|---:|\n'
    for method in methods:
        r=by[method,'ALL'];text+=f"|{method}|{pct(r['accuracy_mean'])} ± {pct(r['accuracy_seed_sd'])}|{pct(r['macro_f1_mean'])} ± {pct(r['macro_f1_seed_sd'])}|\n"
    text+='\n|对照|配对均值 ± SD/百分点|正差值 seed|四 seed 差分/百分点|\n|---|---:|---:|---|\n'
    for method in BASELINES:
        r=gains[method];text+=f"|{method}|{r['accuracy_delta_pp_mean']:+.4f} ± {r['accuracy_delta_pp_seed_sd']:.4f}|{r['positive_seeds']}/4|"+', '.join(f'{v:+.4f}' for v in r['accuracy_delta_pp_by_seed'])+'|\n'
    text+='\n|RX（每 seed 24,000 包）|双路控制/%|新候选/% ± SD|差值/百分点|\n|---|---:|---:|---:|\n'
    for rx in receivers:
        r=by[candidate,rx];c=by['channel_dual',rx]
        text+=f"|{rx}|{pct(c['accuracy_mean'])}|{pct(r['accuracy_mean'])} ± {pct(r['accuracy_seed_sd'])}|{100*(r['accuracy_mean']-c['accuracy_mean']):+.4f}|\n"
    text+='\n|TX（每 seed 28,000 包）|双路控制/%|新候选/% ± SD|\n|---|---:|---:|\n'
    txby={(r['method'],r['transmitter']):r for r in tx}
    for name in CLASSES:
        r=txby[candidate,name];c=txby['channel_dual',name]
        text+=f"|{name}|{pct(c['accuracy_mean'])}|{pct(r['accuracy_mean'])} ± {pct(r['accuracy_seed_sd'])}|\n"
    text+='\n保持原物理数据、单一 CE、E200×50、batch 128 和完整 FP32；从零训练，源域固定规则先选择后冻结。4 个新预测与 40 个旧冻结预测统一为 44 行，逐包面对全部 6 类。全部预测固定后独立 truth-last 评分；352 个混淆矩阵、88 组汇总、80 组配对和 RX 分解复算通过。原 320 条控制指标逐字段完全一致。\n\n'
    text+=f"实际可训练参数 {parameters}；常驻模型状态 {resources[0]['resident_state_bytes']} bytes。batch 1 推理均值 {st.mean(r['inference_batch1_ms'] for r in resources):.4f} ms，batch 128 训练 {st.mean(r['training_batch128_ms'] for r in resources):.4f} ms；完整 168,000 query 预测均值 {st.mean(r['full_clean_prediction_seconds'] for r in resources):.3f} s。逐 seed 硬件、训练/预测峰值显存和 MAC 口径见资源表；并发会影响耗时。CPU 峰值和新增星载传输未测量，记 N/A。\n\n"
    text+='该固定 clean 基准历史已暴露，不称首次盲测。无 LEO、support 适应、新增类、unknown 或在轨结果；D92 三阶段与 K×新增类表为 N/A。补偿器是单一 CE 学习的受限判别式变换，不等于真实信道逆；运算次序差不等于纯 TX 硬件信息。跨接收机 clean 结果不能单独证明信道扰动鲁棒性、唯一 TX 硬件恢复或任意 RX/信道不变性。\n\n'
    text+='[逐行评分](evidence/clean_scored_results.json) · [全部 RX 汇总](evidence/clean_summary.csv) · [全部 TX 汇总](evidence/per_transmitter_summary.csv) · [配对差值](evidence/clean_paired.csv) · [资源](evidence/resource_summary.json) · [独立复算](evidence/analysis_validation.json)\n'
    (folder/'report.md').write_text(text,encoding='utf-8');print(json.dumps(verdict,ensure_ascii=False))
    return verdict


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
