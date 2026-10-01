"""Recompute complete frozen metrics and write the research report; no fitting."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from experiments.cvs_rf_operator_clean.prepare import RUN,SOURCE_RUN,OLD_RUN,PREVIOUS_PHYSICAL_RUN


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)
def pct(v):return f'{100*v:.4f}%'
def avg_sd(values):return statistics.mean(values),statistics.stdev(values)


def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence';e.mkdir(exist_ok=True)
    d=read(e/'final_readback.json');source=read(root/'automation_reports/CV-SincNet'/SOURCE_RUN/'evidence/source_research_complete.json')
    previous=read(root/'automation_reports/CV-SincNet'/OLD_RUN/'evidence/final_readback.json')
    selection=read(e/'performance_selection.json');candidate=selection['selected_variant']
    records=d['results']['results'];summary=d['summary']['summary'];paired=d['summary']['paired']
    if d['marker']['status']!='SCORED_COMPLETE' or d['marker']['rows']!=24 or len(records)!=192 or d['marker']['query_count']!=168000:
        raise ValueError('Full registered clean matrix not complete')
    lookup={(r['method'],r['receiver'],r['model_seed']):r for r in records}
    for r in records:
        cm=np.asarray(r['confusion'],dtype=np.int64);tp=np.diag(cm);n=int(cm.sum())
        f1=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros(6,dtype=float),where=(cm.sum(0)+cm.sum(1))>0).mean()
        if cm.shape!=(6,6) or n!=r['query_count'] or abs(tp.sum()/n-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:
            raise ValueError('Confusion-derived metrics mismatch')
    old=previous['results']['results']
    if len(old)!=160 or any(lookup[(r['method'],r['receiver'],r['model_seed'])]!=r for r in old):raise ValueError('Frozen original160 metrics changed')
    for s in summary:
        rows=[r for r in records if r['method']==s['method'] and r['receiver']==s['receiver']]
        if len(rows)!=4:raise ValueError('Incomplete seed group')
        for key in ('accuracy','macro_f1','macro_accuracy'):
            m,sd=avg_sd([r[key] for r in rows])
            if abs(m-s[key+'_mean'])>1e-12 or abs(sd-s[key+'_seed_sd'])>1e-12:raise ValueError('Summary statistic mismatch')
    seeds=sorted({r['model_seed'] for r in records});rx=sorted({r['receiver'] for r in records if r['receiver']!='ALL'})
    all_summary={s['method']:s for s in summary if s['receiver']=='ALL'}
    gains={p['baseline']:p for p in paired if p['receiver']=='ALL'}
    for p in paired:
        delta=[100*(lookup[(candidate,p['receiver'],s)]['accuracy']-lookup[(p['baseline'],p['receiver'],s)]['accuracy']) for s in seeds]
        if not np.allclose(delta,p['accuracy_delta_pp_by_seed'],atol=1e-12,rtol=0):raise ValueError('Paired seed mismatch')
    # These source references only supply frozen resource measurements to analysis.
    first=read(root/'automation_reports/CV-SincNet/20261001-phase1-cvs-clean-architecture-manysig-m32-r01/evidence/source_complete_readback.json')
    second=read(root/'automation_reports/CV-SincNet/20261001-phase1-cvs-residual-identity-manysig-m8-r01/evidence/source_research_complete.json')
    source_rows=first['rows']+second['rows']+source['rows'];resources=[]
    for method in [*('native','cvcnn','real_cnn','resnet1d','residual_fusion'),candidate]:
        rows=[r for r in source_rows if r['resolved']['variant']==method]
        if len(rows)!=4:raise ValueError('Resource seed group incomplete')
        entry=dict(method=method,hardware=rows[0]['profile']['hardware'],torch_version=rows[0]['profile']['torch_version'])
        for field in ('total_parameters','trainable_parameters','gradient_used_parameters','resident_state_bytes','conv_linear_macs_per_sample','fft_calls_per_sample','inference_batch1_ms','inference_batch128_ms','training_batch128_ms','benchmark_peak_cuda_allocated_bytes'):
            values=[r['profile'].get(field) for r in rows]
            entry[field]=statistics.mean(values) if all(v is not None for v in values) else None
        peaks=[]
        for row in rows:
            if row.get('completion',{}).get('training_peak_cuda_allocated_bytes') is not None:peaks.append(row['completion']['training_peak_cuda_allocated_bytes'])
            elif row.get('epochs'):peaks.append(max(ep['peak_cuda_allocated_bytes'] for ep in row['epochs']))
            elif row.get('epoch'):peaks.append(row['epoch'].get('peak_cuda_allocated_bytes'))
        entry['actual_training_peak_max_bytes']=max(peaks) if len(peaks)==4 and all(v is not None for v in peaks) else None
        pred=[r['completion'] for r in d['rows'] if r['resolved']['variant']==method]
        entry['full_test_prediction_seconds_mean']=statistics.mean(p['prediction_seconds'] for p in pred)
        entry['actual_prediction_peak_max_bytes']=max(p['peak_cuda_allocated_bytes'] for p in pred)
        resources.append(entry)
    res={r['method']:r for r in resources};new=res[candidate]
    reduction={m:{f:100*(1-new[f]/res[m][f]) for f in ('total_parameters','gradient_used_parameters','resident_state_bytes','conv_linear_macs_per_sample')} for m in ('native','residual_fusion')}
    tx=[];classes=d['classes']
    for method in all_summary:
        for index,name in enumerate(classes):
            vals=[];counts=[]
            for seed in seeds:
                cm=np.asarray(lookup[(method,'ALL',seed)]['confusion']);counts.append(int(cm[index].sum()));vals.append(float(cm[index,index]/cm[index].sum()))
            mean,sd=avg_sd(vals);tx.append(dict(method=method,transmitter=name,query_count_per_seed=counts[0],accuracy_mean=mean,accuracy_seed_sd=sd))
    write(e/'resource_summary.json',resources);csvwrite(e/'resource_summary.csv',resources);csvwrite(e/'per_transmitter_summary.csv',tx)
    write(e/'analysis_validation.json',dict(status='VERIFIED',all192_confusions_recomputed=True,all_seed_mean_sample_sd_recomputed=True,old160_metrics_exactly_unchanged=True,
        selected_variant=candidate,paired=gains,resource_reduction_percent=reduction,goal_performance_vs_native=gains['native']['accuracy_delta_pp_mean']>0,
        goal_performance_vs_previous=gains['residual_fusion']['accuracy_delta_pp_mean']>0,target_feedback=False))
    phys=read(root/'automation_reports/CV-SincNet'/PREVIOUS_PHYSICAL_RUN/'evidence/final_readback.json')
    current_capsules={r['resolved'].get('p1_capsule') for r in d['rows'] if r['resolved']['variant']==candidate}
    previous_capsules={r['resolved'].get('p1_capsule') for r in phys['rows'] if r['resolved']['variant']=='coherence_phase'}
    if phys['marker']['status']!='SCORED_COMPLETE' or phys['marker']['query_count']!=168000 or phys['classes']!=classes or current_capsules!=previous_capsules or len(current_capsules)!=1 or None in current_capsules:raise ValueError('Previous physical fixed comparison contract mismatch')
    prev={(r['receiver'],r['model_seed']):r for r in phys['results']['results'] if r['method']=='coherence_phase'}
    supplemental=[]
    for receiver in ['ALL',*rx]:
        values=[]
        for seed in seeds:
            a=lookup[(candidate,receiver,seed)];b=prev[(receiver,seed)]
            if a['query_count']!=b['query_count']:raise ValueError('Previous physical query count mismatch')
            values.append(100*(a['accuracy']-b['accuracy']))
        mean,sd=avg_sd(values);supplemental.append(dict(receiver=receiver,candidate=candidate,baseline='coherence_phase',accuracy_delta_pp_by_seed=values,accuracy_delta_pp_mean=mean,accuracy_delta_pp_seed_sd=sd,positive_seeds=sum(v>0 for v in values)))
    write(e/'paired_previous_physical.json',dict(previous_run=PREVIOUS_PHYSICAL_RUN,scope='report_only_frozen_results_same_capsule_classes_counts_model_seeds',target_feedback=False,rows=supplemental))
    oldtext=(report/'report.md').read_text(encoding='utf-8')
    s=f'# CVS性能优先继续优化：{candidate} clean测试报告\n\n'
    c=all_summary[candidate];s+=f"本轮已完成8行源训练和24行clean确认，状态ANALYZED。选中`{candidate}`的准确率为**{pct(c['accuracy_mean'])}±{100*c['accuracy_seed_sd']:.4f}%**，较原CVS配对{gains['native']['accuracy_delta_pp_mean']:+.4f}个百分点，较上一轮残差CVS{gains['residual_fusion']['accuracy_delta_pp_mean']:+.4f}个百分点。性能是主要目标，参数轻量是次要条件；全部负结果如实保留。\n\n"
    if gains['residual_fusion']['accuracy_delta_pp_mean']>0:
        s+='当前固定测试集上，四个训练 seed 的平均准确率取得进一步提升。四个 seed 的证据仍有限，平均值提升不代表每个 seed、RX 或 TX 都改善，也不构成对其他数据或星地场景的泛化保证。\n\n'
    else:
        s+='按性能主要目标，本轮未取得比上一轮更高的平均准确率。参数或状态减少仅作为资源结果记录，不能替代识别性能提升，也不能据本次测试反馈重选或调参。\n\n'
    s+='## 总体与配对结果\n\n|模型|accuracy均值±seed标准差|Macro-F1均值±标准差|参数量|\n|---|---:|---:|---:|\n'
    for method in ('native','cvcnn','real_cnn','resnet1d','residual_fusion',candidate):
        a=all_summary[method];s+=f"|{method}|{pct(a['accuracy_mean'])}±{100*a['accuracy_seed_sd']:.4f}%|{pct(a['macro_f1_mean'])}±{100*a['macro_f1_seed_sd']:.4f}%|{int(res[method]['total_parameters'])}|\n"
    s+='\n|控制组|配对提升均值±标准差（百分点）|正提升seed数|4个seed逐行提升（百分点）|\n|---|---:|---:|---|\n'
    for m in ('native','residual_fusion','cvcnn','real_cnn','resnet1d'):
        p=gains[m];values=', '.join(f'{x:+.4f}' for x in p['accuracy_delta_pp_by_seed']);s+=f"|{m}|{p['accuracy_delta_pp_mean']:+.4f}±{p['accuracy_delta_pp_seed_sd']:.4f}|{p['positive_seeds']}/4|{values}|\n"
    s+='\n|model seed|原CVS|上轮残差CVS|本轮选中CVS|\n|---|---:|---:|---:|\n'
    for seed in seeds:s+=f"|{seed}|{pct(lookup[('native','ALL',seed)]['accuracy'])}|{pct(lookup[('residual_fusion','ALL',seed)]['accuracy'])}|{pct(lookup[(candidate,'ALL',seed)]['accuracy'])}|\n"
    p=supplemental[0]
    s+=f"\n上一次已完成的 coherence_phase 作为报告对照，配对均值变化{p['accuracy_delta_pp_mean']:+.4f}±{p['accuracy_delta_pp_seed_sd']:.4f}个百分点，{p['positive_seeds']}/4 seed提升。复用既有评分，不新增推理/选模；同capsule/类别/每RX样本数/seed已核实。本轮替换PA路径并使用原残差框架，此前coherence_phase含相位补充路径。这是整体版本对照，不能仅归因某一物理性质。完整[配对结果](evidence/paired_previous_physical.json)。\n"
    s+='\n## 各接收机与发射机\n\n|测试RX|原CVS|上轮残差CVS|本轮CVS均值±标准差|较上轮（百分点）|\n|---|---:|---:|---:|---:|\n'
    by={(x['method'],x['receiver']):x for x in summary}
    for receiver in rx:
        a=by[(candidate,receiver)];oldrx=by[('residual_fusion',receiver)];s+=f"|{receiver}|{pct(by[('native',receiver)]['accuracy_mean'])}|{pct(oldrx['accuracy_mean'])}|{pct(a['accuracy_mean'])}±{100*a['accuracy_seed_sd']:.4f}%|{100*(a['accuracy_mean']-oldrx['accuracy_mean']):+.4f}|\n"
    s+='\n|TX（每seed28000query）|原CVS|上轮残差CVS|本轮CVS均值±标准差|\n|---|---:|---:|---:|\n'
    txby={(x['method'],x['transmitter']):x for x in tx}
    for name in classes:
        a=txby[(candidate,name)];s+=f"|{name}|{pct(txby[('native',name)]['accuracy_mean'])}|{pct(txby[('residual_fusion',name)]['accuracy_mean'])}|{pct(a['accuracy_mean'])}±{100*a['accuracy_seed_sd']:.4f}%|\n"
    s+='\n完整192条同row评分及混淆矩阵见[评分JSON](evidence/clean_scored_results.json)，所有模型TX表见[CSV](evidence/per_transmitter_summary.csv)。每RX每seed24000query，全部168000query。样本数不是独立训练seed数，标准差按4个模型seed计算；不把同包重复预测当作更多独立测试样本。\n'
    s+='\n## 完整源训练、选择与机制\n\n两候选各4seed，全从零E200×50，固定L6300/U56700unused/V27000及split392005。普通身份CE唯一损失；域骨干/信道与数据增强/teacher/EMA/继承/extra loss关闭。完整1600epoch、80000step及CSV/全部stdout解析，实际物理角色与原契约相同；startup Torch2.1/NumPy2兼容警告单独记录，显式copy路径可正常训练与预测。\n\n|候选|源V均值|最差源RX均值|性能分数|\n|---|---:|---:|---:|\n'
    for method,st in selection['source_summaries'].items():s+=f"|{method}|{pct(st['source_accuracy'])}|{pct(st['worst_rx_accuracy'])}|{pct(st['score'])}|\n"
    s+='\n本轮登记开始即遵守用户“性能优先，轻量次要”。源选择以0.5源V＋0.5最差源RX最高分优先，源性能完全并列后才比较成本；没有0.2个百分点成本优先容差。保持E200权重和原source_selection，确认记录引用同一冻结结果。实际8行源产物重算与冻结一致，未选候选不测试；test成绩不回流研发/选择/重训。\n\n'
    s+='本轮以residual_fusion框架为共同基线，只替换PA路径为因果复数MP/GMP直接及共轭算子。奇数阶1/3/5，延迟0..3，GMP再加包络滞后1/2；径向保护保持I/Q共同缩放，32个观测通道在进入实数CNN前构造成全局相位不变量。共有系数和其他模块初始化配对一致，GMP仅增加512个交叉记忆系数。合成检查验证受约束映射，不能由此推导真实TX/RX解耦或识别收益。输入为已均衡、CFO加回的received IQ，滤波系数不是已识别的TX硬件参数。整体网络普通时间/频率路径不保证相位/CFO/接收机不变。详见[前瞻物理假设与证据边界](../../../docs/CVS_RF_OPERATOR_HYPOTHESIS_20261002.md)。\n'
    s+='\n## 实测资源成本\n\n|指标|原CVS|上轮残差CVS|本轮CVS|\n|---|---:|---:|---:|\n'
    for field,label in [('total_parameters','总参数'),('gradient_used_parameters','实际CE梯度参数'),('resident_state_bytes','常驻模型状态字节'),('conv_linear_macs_per_sample','Conv/Linear MAC/包'),('fft_calls_per_sample','FFT次数/包'),('inference_batch1_ms','合成batch1推理ms'),('inference_batch128_ms','合成batch128推理ms'),('training_batch128_ms','合成batch128训练步ms'),('actual_training_peak_max_bytes','实际训练进程峰值allocated显存bytes'),('actual_prediction_peak_max_bytes','实际预测进程峰值allocated显存bytes'),('full_test_prediction_seconds_mean','168000query预测秒数（4seed均值）')]:
        vals=[res[m][field] for m in ('native','residual_fusion',candidate)];s+=f"|{label}|"+'|'.join('N/A' if v is None else f'{v:.3f}' if isinstance(v,float) and not v.is_integer() else str(int(v)) for v in vals)+'|\n'
    def change(value):return f'减少{value:.4f}%' if value>=0 else f'增加{-value:.4f}%'
    s+=f"\n总参数较原CVS{change(reduction['native']['total_parameters'])}，较上轮{change(reduction['residual_fusion']['total_parameters'])}；Conv/Linear MAC分别{change(reduction['native']['conv_linear_macs_per_sample'])}与{change(reduction['residual_fusion']['conv_linear_macs_per_sample'])}，不包含FFT/归一化/逐元素运算。资源是次要条件，参数变化不等于同比例加速。\n"
    s+='\nRTX3090、Torch2.1/FP32，同一profile定义：推理warmup5/重复20、训练warmup5/重复10，副本不回到训练状态；profile峰值含原模型和副本，实际训练峰值取E200结束profile之前，预测峰值独立。源profile计时不等价真实星载耗时；历史profile可能GPU共用，不能据小幅时延差断言加速。闭集Phase1无SFT，训练参数表不能当作星载适应成本。常驻状态只计模型state_dict，未计runtime/optimizer/传输协议；新增部署传输与星载常驻状态N/A，本轮本地checkpoint权重大小由状态字节代理。完整[资源JSON](evidence/resource_summary.json)、[CSV](evidence/resource_summary.csv)。\n'
    s+='\n## 协议、验证及结论边界\n\n新4预测逐包面对6个注册类，无truth/role/目标统计拟合/类别配额/全局重排，只读clean.npy/index.npz/manifest。checkpoint实际scratch来源和完整角色契约在query前核对。原20预测/config/provenance只读复用，重新评分的160条原指标逐字段完全相同；新增192条全部混淆矩阵、4seed均值/样本标准差与配对提升独立重算一致。聚焦物理算子及clean协议检查通过，源与确认各有一次独立P0/P1审查通过。现有VALIDATED_ONCE数据不变，不重复验证或添加来源链。\n\n'
    s+='结论只适用于当前同划分、4模型seed、纯CE、无增强、clean闭集研究。目标测试历史已公开，不能称为新盲测；不同RX或TX的下降不能隐藏。D92适应前/旧类support适应后/新类注册后及K×新增类表均N/A；不做LEO/真实卫星/未知类/部署适应/新类结论。若本轮未超过上一轮，明确按性能主要目标未取得进一步准确率提升，保留其参数结果而不作为性能晋级，不按本次test成绩重排或选择性重训。\n\n'
    s+='[分析复算证据](evidence/analysis_validation.json) · [源性能冻结](evidence/performance_selection.json) · [源完整审计](evidence/source_completion_validation.json) · [独立最终读回](evidence/final_readback.json)。源训练release提交见source_research_complete.json实际resolved.commit；测试release提交、实际PID与输出详见读回。Git交付状态最后由远端OID独立核实。\n'
    (report/'report.md').write_text(s,encoding='utf-8')
    write(e/'source_curve_summary.json',dict(rows=[dict(row_id=r['row_id'],epochs=len(r['epochs']),final_ce=r['epochs'][-1]['clean_ce'],best_source_V=max(x['source_val_accuracy'] for x in r['epochs']),fixed_E200_V=r['epochs'][-1]['source_val_accuracy'],train_seconds=r['completion']['elapsed_seconds'],full_log_audit=r['log_audit']) for r in source['rows']]))
    print(json.dumps(dict(status='VERIFIED',candidate=candidate,accuracy_mean=c['accuracy_mean'],accuracy_sd=c['accuracy_seed_sd'],gains=gains,parameters=new['total_parameters'],report=str(report/'report.md')),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
