"""Recompute every fixed clean result; scores never change models or selection."""
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np
from experiments.cvs_moment_residual_clean.prepare import RUN,SOURCE_RUN,OLD_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

BASELINES=('native','cvcnn','real_cnn','resnet1d','residual_fusion','energy_equivariant','coupled_lag4','adaptive_volterra_lag4')
SEEDS=tuple(range(2026092701,2026092705))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def pct(value):return f'{100*value:.4f}%'
def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({key for row in rows for key in row}));writer.writeheader();writer.writerows(rows)

def validate_results(data,selection,previous):
    candidate=selection['selected_variant'];records=data['results']['results']
    summary=data['summary']['summary'];pairs=data['summary']['paired']
    marker=data['marker'];methods=(*BASELINES,candidate)
    if not selection['new_candidate_selected'] or selection['target_access'] or selection['target_score_used']:
        raise ValueError('Selected new source-only candidate required')
    if marker['status']!='SCORED_COMPLETE' or marker['rows']!=36 or marker['query_count']!=168000 or len(records)!=288:
        raise ValueError('Incomplete registered clean matrix')
    lookup={(row['method'],row['receiver'],row['model_seed']):row for row in records}
    receivers=sorted({row['receiver'] for row in records if row['receiver']!='ALL'})
    if len(receivers)!=7 or len(lookup)!=288 or set(lookup)!={(method,rx,seed) for method in methods for rx in ('ALL',*receivers) for seed in SEEDS}:
        raise ValueError('Missing/duplicate method,receiver,seed')
    for row in records:
        if row.get('view')!='clean':raise ValueError('Non-clean result outside registered scope')
        cm=np.asarray(row['confusion']);number=int(cm.sum())
        if cm.shape!=(6,6) or not np.issubdtype(cm.dtype,np.integer) or np.any(cm<0) or number!=row['query_count'] or number!=(168000 if row['receiver']=='ALL' else 24000):
            raise ValueError('Confusion schema or fixed count mismatch')
        true=np.diag(cm);denominator=cm.sum(0)+cm.sum(1)
        f1=np.divide(2*true,denominator,out=np.zeros(6,dtype=float),where=denominator>0).mean()
        macro=np.divide(true,cm.sum(1),out=np.zeros(6,dtype=float),where=cm.sum(1)>0).mean()
        if any(abs(actual-row[key])>1e-12 for key,actual in [('accuracy',true.sum()/number),('macro_f1',f1),('macro_accuracy',macro)]):
            raise ValueError('Independent confusion metrics differ')
    for method in methods:
        for seed in SEEDS:
            total=np.asarray(lookup[(method,'ALL',seed)]['confusion'])
            if not np.array_equal(total,sum((np.asarray(lookup[(method,rx,seed)]['confusion']) for rx in receivers),np.zeros((6,6),dtype=np.int64))):
                raise ValueError('RX partition does not sum to overall confusion')
    old=previous['results']['results']
    if len(old)!=256 or any(lookup[(row['method'],row['receiver'],row['model_seed'])]!=row for row in old):
        raise ValueError('Frozen original control metrics changed')
    if len(summary)!=72 or len({(row['method'],row['receiver']) for row in summary})!=72:
        raise ValueError('Incomplete/duplicate summary')
    for row in summary:
        values=[lookup[(row['method'],row['receiver'],seed)] for seed in SEEDS]
        for key in ('accuracy','macro_accuracy','macro_f1'):
            measured=[value[key] for value in values]
            if abs(statistics.mean(measured)-row[key+'_mean'])>1e-12 or abs(statistics.stdev(measured)-row[key+'_seed_sd'])>1e-12:
                raise ValueError('Independent four-seed summary differs')
    if len(pairs)!=64 or {(row['baseline'],row['receiver']) for row in pairs}!={(method,rx) for method in BASELINES for rx in ('ALL',*receivers)}:
        raise ValueError('Incomplete/duplicate paired comparisons')
    for row in pairs:
        values=[100*(lookup[(candidate,row['receiver'],seed)]['accuracy']-lookup[(row['baseline'],row['receiver'],seed)]['accuracy']) for seed in SEEDS]
        if row['candidate']!=candidate or row['model_seeds']!=list(SEEDS) or not np.allclose(values,row['accuracy_delta_pp_by_seed'],atol=1e-12,rtol=0):
            raise ValueError('Paired seed mapping differs')
        if abs(statistics.mean(values)-row['accuracy_delta_pp_mean'])>1e-12 or abs(statistics.stdev(values)-row['accuracy_delta_pp_seed_sd'])>1e-12 or sum(value>0 for value in values)!=row['positive_seeds']:
            raise ValueError('Independent paired mean/SD/sign differs')
    return lookup,receivers

def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;evidence=report/'evidence'
    data=read(evidence/'final_readback.json');selection=read(evidence/'performance_selection.json');candidate=selection['selected_variant']
    new_rows=[row for row in data['rows'] if row['resolved']['variant']==candidate]
    if len(new_rows)!=4 or any(row['resolved'].get('backend_flags')!=FULL_FP32_POLICY or row['resolved'].get('numerical_policy')!=FULL_FP32_POLICY for row in new_rows):
        raise ValueError('New prediction/source FP32 contract differs')
    previous=read(root/'automation_reports/CV-SincNet'/OLD_RUN/'evidence/final_readback.json')
    lookup,receivers=validate_results(data,selection,previous)
    records=data['results']['results'];summary=data['summary']['summary'];pairs=data['summary']['paired'];methods=(*BASELINES,candidate)
    tx=[]
    if data['classes']!=['14-10','14-7','20-15','20-19','6-15','8-20']:raise ValueError('Frozen classes differ')
    for method in methods:
        for i,name in enumerate(data['classes']):
            values=[]
            for seed in SEEDS:
                cm=np.asarray(lookup[(method,'ALL',seed)]['confusion'])
                if cm[i].sum()!=28000:raise ValueError('Fixed TX denominator differs')
                values.append(float(cm[i,i]/28000))
            tx.append(dict(method=method,transmitter=name,query_count=28000,accuracy_mean=statistics.mean(values),accuracy_seed_sd=statistics.stdev(values)))
    csvwrite(evidence/'per_transmitter_summary.csv',tx)
    csvwrite(evidence/'clean_scored_results.csv',[{key:value for key,value in row.items() if key!='confusion'} for row in records])
    csvwrite(evidence/'clean_summary.csv',summary)
    all_summary={row['method']:row for row in summary if row['receiver']=='ALL'}
    gains={row['baseline']:row for row in pairs if row['receiver']=='ALL'};new=all_summary[candidate]
    improved=gains['adaptive_volterra_lag4']['accuracy_delta_pp_mean']>0
    verdict=dict(status='VERIFIED',candidate=candidate,accuracy_mean=new['accuracy_mean'],accuracy_sd=new['accuracy_seed_sd'],
        paired_vs_adaptive=gains['adaptive_volterra_lag4'],paired_vs_coupled=gains['coupled_lag4'],
        paired_vs_energy=gains['energy_equivariant'],paired_vs_residual=gains['residual_fusion'],paired_vs_native=gains['native'],
        all288_confusions_recomputed=True,old256_metrics_exactly_unchanged=True,all_mean_sd_and_pairs_recomputed=True,
        all_RX_sums_match=True,scientific_verdict='PERFORMANCE_IMPROVEMENT' if improved else 'NO_MEAN_PERFORMANCE_IMPROVEMENT',
        genuine_RFF_physics_aware_goal_achieved=False,target_feedback=False,
        physical_scope='CE-trained packet-moment residual with two new global parameters;no uniqueTX hardware/causalTX-RX/wholeCFO-RX-LTI invariance claim')
    write(evidence/'analysis_validation.json',verdict);write(evidence/'iteration_verdict.json',verdict)
    source=read(root/'automation_reports/CV-SincNet'/SOURCE_RUN/'evidence/source_research_complete.json')
    source_rows=[row for row in source['rows'] if row['resolved']['variant']==candidate];profile=source_rows[0]['profile']
    predictions=[row['completion'] for row in new_rows]
    if len(source_rows)!=4 or profile['total_parameters']!=202557:raise ValueError('Actual source resource matrix differs')
    resource=dict(candidate=candidate,parameters=profile['total_parameters'],trainable_parameters=profile['trainable_parameters'],
        gradient_parameters=profile['gradient_used_parameters'],resident_model_state_bytes=profile['resident_state_bytes'],
        conv_linear_macs=profile['conv_linear_macs_per_sample'],fft_calls_per_sample=profile['fft_calls_per_sample'],
        inference_batch1_ms_mean=statistics.mean(row['profile']['inference_batch1_ms'] for row in source_rows),
        train_batch128_ms_mean=statistics.mean(row['profile']['training_batch128_ms'] for row in source_rows),
        actual_training_peak_bytes=max(rec['peak_cuda_allocated_bytes'] for row in source_rows for rec in row['epochs']),
        full_clean_prediction_seconds_mean=statistics.mean(row['prediction_seconds'] for row in predictions),
        actual_prediction_peak_bytes=max(row['peak_cuda_allocated_bytes'] for row in predictions),
        hardware=profile['hardware'],torch_version=profile['torch_version'],mac_scope=profile['mac_scope'],moment_contract=profile['moment_contract'],
        frozen_coefficients=[dict(model_seed=row['resolved']['model_seed'],alpha=row['physical_diagnostics']['moment_input']['coefficients'],
            beta=row['physical_diagnostics']['moment_input']['projection_coefficients']) for row in source_rows],
        new_onboard_transmission_bytes=None,CPU_peak_bytes=None)
    write(evidence/'resource_summary.json',resource)
    text='# CVS可学习包内矩残差：独立clean测试报告\n\n'
    text+=f"选中 `{candidate}` 的四seed准确率为 **{pct(new['accuracy_mean'])} ± {100*new['accuracy_seed_sd']:.4f}%**，Macro-F1为{pct(new['macro_f1_mean'])} ± {100*new['macro_f1_seed_sd']:.4f}%。较当前adaptive控制{gains['adaptive_volterra_lag4']['accuracy_delta_pp_mean']:+.4f}个百分点，较原residual_fusion {gains['residual_fusion']['accuracy_delta_pp_mean']:+.4f}个百分点。\n\n"
    text+=('本轮在固定clean基准上较当前同核心控制取得平均性能提升。' if improved else '本轮未进一步提高平均性能，全部负差值保留。')+'四seed均值不保证每个seed、接收机或发射机改善；参数轻量是次要条件。测试及公共链仍不足以证明唯一TX硬件恢复或TX/RX因果分离，总体RFF目标尚未证明完成。\n\n'
    text+='|模型|准确率均值±样本SD|Macro-F1均值±样本SD|\n|---|---:|---:|\n'
    for method in methods:
        row=all_summary[method];text+=f"|{method}|{pct(row['accuracy_mean'])}±{100*row['accuracy_seed_sd']:.4f}%|{pct(row['macro_f1_mean'])}±{100*row['macro_f1_seed_sd']:.4f}%|\n"
    text+='\n|对照|配对均值±SD（百分点）|正提升seed|四seed差分（百分点）|\n|---|---:|---:|---|\n'
    for method in BASELINES:
        row=gains[method];text+=f"|{method}|{row['accuracy_delta_pp_mean']:+.4f}±{row['accuracy_delta_pp_seed_sd']:.4f}|{row['positive_seeds']}/4|"+', '.join(f'{value:+.4f}' for value in row['accuracy_delta_pp_by_seed'])+'|\n'
    text+='\n|seed|原CVS|原residual_fusion|当前adaptive控制|新候选|\n|---|---:|---:|---:|---:|\n'
    for seed in SEEDS:text+=f'|{seed}|'+ '|'.join(pct(lookup[(method,'ALL',seed)]['accuracy']) for method in ('native','residual_fusion','adaptive_volterra_lag4',candidate))+'|\n'
    by={(row['method'],row['receiver']):row for row in summary}
    text+='\n|RX（每seed24000包）|原残差CVS|当前adaptive|新候选均值±SD|较当前adaptive／百分点|\n|---|---:|---:|---:|---:|\n'
    for rx in receivers:
        row=by[(candidate,rx)];control=by[('adaptive_volterra_lag4',rx)]
        text+=f"|{rx}|{pct(by[('residual_fusion',rx)]['accuracy_mean'])}|{pct(control['accuracy_mean'])}|{pct(row['accuracy_mean'])}±{100*row['accuracy_seed_sd']:.4f}%|{100*(row['accuracy_mean']-control['accuracy_mean']):+.4f}|\n"
    txby={(row['method'],row['transmitter']):row for row in tx}
    text+='\n|TX（每seed28000包）|原残差CVS|当前adaptive|新候选均值±SD|\n|---|---:|---:|---:|\n'
    for name in data['classes']:
        row=txby[(candidate,name)];text+=f"|{name}|{pct(txby[('residual_fusion',name)]['accuracy_mean'])}|{pct(txby[('adaptive_volterra_lag4',name)]['accuracy_mean'])}|{pct(row['accuracy_mean'])}±{100*row['accuracy_seed_sd']:.4f}%|\n"
    text+='\n|seed|alpha3|alpha5|beta3|beta5|\n|---|---:|---:|---:|---:|\n'
    for row in resource['frozen_coefficients']:text+=f"|{row['model_seed']}|"+'|'.join(f'{value:.8g}' for value in row['alpha']+row['beta'])+'|\n'
    text+='\n四系数只由源CE学习；query推理只读。新增beta调节同包矩残差，保留原received IQ与原相位记忆；这些系数不是PA或晶振参数。源选模在新query前冻结：8新scratch E200×50＋4当前adaptive源记录，四seed mean(0.5V+0.5最差源RX)最高优先，完全并列后才比成本。未选候选不访问query。数学、通信、物理及RFF设计、全源日志和公共链见[源报告](../'+SOURCE_RUN+'/report.md)。\n\n'
    text+=f"实际参数{profile['total_parameters']}，较当前控制增加2；模型常驻状态{profile['resident_state_bytes']} bytes。Conv/Linear MAC {profile['conv_linear_macs_per_sample']} /包，未计包内矩/逐元素计算，不能据此认定总计算量不变。RTX3090/Torch2.1/完整FP32；batch1推理{resource['inference_batch1_ms_mean']:.4f} ms，batch128训练{resource['train_batch128_ms_mean']:.4f} ms；实际训练峰值{resource['actual_training_peak_bytes']} bytes，168000包预测均值{resource['full_clean_prediction_seconds_mean']:.3f} s，预测峰值{resource['actual_prediction_peak_bytes']} bytes。受并发影响，不作独占硬件基准结论；CPU峰值、星载/新增传输/SFT为N/A。\n\n"
    text+='4份新预测＋32份旧冻结预测统一36行；全部逐包面对6类argmax，无truth/role、query拟合、配额或全局重排。全部预测固定、物理ID对齐后独立scorer连接truth。288个混淆矩阵、72组均值/样本SD、64组配对差分及RX求和独立复算，256条旧控制逐字段保持一致。新源/预测完整FP32；旧32控制保留原历史精度。\n\n'
    text+='仅原划分、四模型seed、CE唯一、无增强、身份骨干、clean闭集工程基准。历史已暴露，不称首次盲测；无LEO、support、适应、新增类、unknown或实测在轨结论。包内矩非流式因果运算；输入相位协变不推出整网CFO/RX/LTI不变。D92三阶段/K×新增类为N/A。目标结果不回流结构、矩定义、系数、epoch、seed或选择性重跑。\n\n'
    text+='[完整288条评分](evidence/clean_scored_results.json) · [评分CSV](evidence/clean_scored_results.csv) · [全部54条TX汇总](evidence/per_transmitter_summary.csv) · [分析复算](evidence/analysis_validation.json) · [资源](evidence/resource_summary.json) · [终态读回](evidence/final_readback.json)。状态ANALYZED，Git交付另行独立核实。\n'
    (report/'report.md').write_text(text,encoding='utf-8');print(json.dumps(verdict,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);analyze(parser.parse_args().root)
