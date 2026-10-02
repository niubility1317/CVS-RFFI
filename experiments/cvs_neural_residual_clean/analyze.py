"""Independent same-row clean metric validation for the neural family."""
import argparse,json,statistics as st
from pathlib import Path
import numpy as np
from experiments.cvs_moment_residual_clean.analyze import (
    BASELINES, SEEDS, validate_results as _validate_results,read,write,csvwrite,
)
from experiments.cvs_neural_residual_identity.model import VARIANTS
from experiments.cvs_neural_residual_clean.prepare import RUN,SOURCE_RUN,OLD_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

def validate_results(data, selection, previous):
    if selection.get('selected_variant') not in VARIANTS:
        raise ValueError('Unregistered neural-residual candidate')
    return _validate_results(data, selection, previous)

def analyze(root):
    folder=root/'automation_reports/CV-SincNet'/RUN;e=folder/'evidence'
    data=read(e/'final_readback.json');selection=read(e/'performance_selection.json')
    previous=read(root/'automation_reports/CV-SincNet'/OLD_RUN/'evidence/final_readback.json')
    lookup,receivers=validate_results(data,selection,previous);candidate=selection['selected_variant']
    new=[r for r in data['rows'] if r['resolved']['variant']==candidate]
    if len(new)!=4 or any(r['resolved']['backend_flags']!=FULL_FP32_POLICY or r['resolved']['numerical_policy']!=FULL_FP32_POLICY for r in new):raise ValueError('Actual prediction FP32 mismatch')
    if data['dispatcher_process'] or any(r['process'] for r in new):raise ValueError('Clean processes not independently terminal')
    records=data['results']['results'];summary=data['summary']['summary'];pairs=data['summary']['paired']
    methods=(*BASELINES,candidate);all_summary={r['method']:r for r in summary if r['receiver']=='ALL'}
    by={(r['method'],r['receiver']):r for r in summary};gains={r['baseline']:r for r in pairs if r['receiver']=='ALL'}
    tx=[]
    for method in methods:
        for i,name in enumerate(data['classes']):
            values=[]
            for seed in SEEDS:
                cm=np.asarray(lookup[method,'ALL',seed]['confusion']);assert cm[i].sum()==28000
                values.append(float(cm[i,i]/28000))
            tx.append(dict(method=method,transmitter=name,query_count_per_seed=28000,accuracy_mean=st.mean(values),accuracy_seed_sd=st.stdev(values)))
    csvwrite(e/'per_transmitter_summary.csv',tx);csvwrite(e/'clean_summary.csv',summary)
    csvwrite(e/'clean_scored_results.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in records]);write(e/'clean_scored_results.json',data['results'])
    source=read(root/'automation_reports/CV-SincNet'/SOURCE_RUN/'evidence/source_research_complete.json')
    models=[r for r in source['rows'] if r['resolved']['variant']==candidate]
    assert len(models)==4
    resources=[]
    for r in models:
        p=r['profile'];s=r['resolved']['model_seed'];prediction=next(v['completion'] for v in new if v['resolved']['model_seed']==s)
        assert p['total_parameters']==(220987 if candidate==VARIANTS[0] else 239419)
        resources.append(dict(model_seed=s,parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],resident_state_bytes=p['resident_state_bytes'],
            conv_linear_MACs=p['conv_linear_macs_per_sample'],mac_scope=p['mac_scope'],hardware=p['hardware'],torch_version=p['torch_version'],
            inference_batch1_ms=p['inference_batch1_ms'],training_batch128_ms=p['training_batch128_ms'],training_peak_bytes=max(x['peak_cuda_allocated_bytes'] for x in r['epochs']),
            full_clean_prediction_seconds=prediction['prediction_seconds'],prediction_peak_bytes=prediction['peak_cuda_allocated_bytes'],CPU_peak_bytes=None,onboard_transmission_bytes=None))
    write(e/'resource_summary.json',dict(rows=resources,scope='Measured same-run hardware; not an exclusive benchmark; onboard/CPU peak N/A'))
    verdict=dict(status='VERIFIED',candidate=candidate,accuracy_mean=all_summary[candidate]['accuracy_mean'],accuracy_seed_sd=all_summary[candidate]['accuracy_seed_sd'],
        paired_vs_adaptive=gains['adaptive_volterra_lag4'],all288_confusions_recomputed=True,old256_metrics_exactly_unchanged=True,all_mean_sd_and_pairs_recomputed=True,all_RX_sums_match=True,
        target_feedback=False,scientific_verdict='PERFORMANCE_IMPROVEMENT' if gains['adaptive_volterra_lag4']['accuracy_delta_pp_mean']>0 else 'NO_MEAN_PERFORMANCE_IMPROVEMENT',
        scope='Fixed historically exposed clean benchmark; source-selected pureCE neural architecture; four model seeds; no arbitraryRX or uniqueTX hardware separation claim')
    write(e/'analysis_validation.json',verdict);write(e/'iteration_verdict.json',verdict)
    def fmt(v):return f'{100*v:.4f}'
    text='# CVS可学习复卷积残差：独立clean结果\n\n'
    text+=f"源规则选中`{candidate}`；四seed准确率{fmt(verdict['accuracy_mean'])}%±{fmt(verdict['accuracy_seed_sd'])}%，较当前adaptive控制{gains['adaptive_volterra_lag4']['accuracy_delta_pp_mean']:+.4f}个百分点。结论：`{verdict['scientific_verdict']}`。\n\n"
    text+='|模型|准确率/%±seed SD|Macro-F1/%±seed SD|\n|---|---:|---:|\n'
    for method in methods:
        r=all_summary[method];text+=f"|{method}|{fmt(r['accuracy_mean'])}±{fmt(r['accuracy_seed_sd'])}|{fmt(r['macro_f1_mean'])}±{fmt(r['macro_f1_seed_sd'])}|\n"
    text+='\n|对照|配对均值±SD/百分点|正提升seed|四seed差分/百分点|\n|---|---:|---:|---|\n'
    for method in BASELINES:
        r=gains[method];text+=f"|{method}|{r['accuracy_delta_pp_mean']:+.4f}±{r['accuracy_delta_pp_seed_sd']:.4f}|{r['positive_seeds']}/4|"+', '.join(f'{v:+.4f}' for v in r['accuracy_delta_pp_by_seed'])+'|\n'
    text+='\n|RX（每seed24000包）|adaptive/%|新候选/%±seed SD|差值/百分点|\n|---|---:|---:|---:|\n'
    for rx in receivers:
        r=by[candidate,rx];c=by['adaptive_volterra_lag4',rx]
        text+=f"|{rx}|{fmt(c['accuracy_mean'])}|{fmt(r['accuracy_mean'])}±{fmt(r['accuracy_seed_sd'])}|{100*(r['accuracy_mean']-c['accuracy_mean']):+.4f}|\n"
    text+='\n|TX（每seed28000包）|adaptive/%|新候选/%±seed SD|\n|---|---:|---:|\n'
    txby={(r['method'],r['transmitter']):r for r in tx}
    for name in data['classes']:
        r=txby[candidate,name];c=txby['adaptive_volterra_lag4',name]
        text+=f"|{name}|{fmt(c['accuracy_mean'])}|{fmt(r['accuracy_mean'])}±{fmt(r['accuracy_seed_sd'])}|\n"
    text+='\n仅修改网络结构；保持原物理数据、单一CE、E200/AdamW/cosine和无增强。选模在新query前冻结；4份新预测和32份旧冻结预测组成36行，均逐样本面对全部6类。预测固定后独立truth-last评分，全部288个混淆矩阵、72组汇总、64组配对和RX求和复算通过；旧256条控制指标逐字段一致。所有负结果保留，不反馈结构/参数/重排或选择性重跑。\n\n'
    text+=f"实际参数{resources[0]['parameters']}，模型状态{resources[0]['resident_state_bytes']}bytes；batch1推理均值{st.mean(r['inference_batch1_ms'] for r in resources):.4f}ms，batch128训练{st.mean(r['training_batch128_ms'] for r in resources):.4f}ms；完整168000query预测均值{st.mean(r['full_clean_prediction_seconds'] for r in resources):.3f}s。逐seed硬件、显存峰值和状态见资源表；并发会影响耗时。MAC包含实际卷积与矩阵乘法，未覆盖FFT、归一化、门控、池化或逐元素运算。CPU峰值和星载传输未测量，记N/A。\n\n"
    text+='本次是固定历史已暴露clean基准，不能称首次盲测；四seed只覆盖模型初始化变化，不能保证各RX/TX均提高或任意域泛化。无LEO、支持集适应、新增类或在轨测试；D92三阶段/K×新增类为N/A。\n\n[逐row评分](evidence/clean_scored_results.json) · [评分CSV](evidence/clean_scored_results.csv) · [TX汇总](evidence/per_transmitter_summary.csv) · [资源](evidence/resource_summary.json) · [独立复算](evidence/analysis_validation.json) · [源报告](../'+SOURCE_RUN+'/report.md)。\n'
    (folder/'report.md').write_text(text,encoding='utf-8');print(json.dumps(verdict,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
