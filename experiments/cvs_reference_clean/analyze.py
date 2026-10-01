"""Independently recompute all frozen clean metrics and report paired outcomes."""
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np
from experiments.cvs_reference_clean.prepare import RUN,SOURCE_RUN,OLD_RUN


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def pct(x):return f'{100*x:.4f}%'
def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence';d=read(e/'final_readback.json');selection=read(e/'performance_selection.json')
    candidate=selection['selected_variant'];records=d['results']['results'];summary=d['summary']['summary'];pairs=d['summary']['paired']
    if not selection['new_candidate_selected'] or d['marker']['status']!='SCORED_COMPLETE' or d['marker']['rows']!=24 or d['marker']['query_count']!=168000 or len(records)!=192:
        raise ValueError('Registered selected-only clean matrix incomplete')
    lookup={(r['method'],r['receiver'],r['model_seed']):r for r in records}
    if len(lookup)!=192:raise ValueError('Duplicate scored rows')
    for r in records:
        cm=np.asarray(r['confusion']);n=int(cm.sum())
        if cm.shape!=(6,6) or np.any(cm<0) or n!=r['query_count']:raise ValueError('Confusion schema/count mismatch')
        tp=np.diag(cm);f1=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros(6,dtype=float),where=(cm.sum(0)+cm.sum(1))>0).mean()
        if abs(tp.sum()/n-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Confusion metrics mismatch')
    previous=read(root/'automation_reports/CV-SincNet'/OLD_RUN/'evidence/final_readback.json')
    old=previous['results']['results']
    if len(old)!=160 or any(lookup[(r['method'],r['receiver'],r['model_seed'])]!=r for r in old):raise ValueError('Original control metrics changed')
    for x in summary:
        rows=[r for r in records if r['method']==x['method'] and r['receiver']==x['receiver']]
        if len(rows)!=4:raise ValueError('Missing seed group')
        for key in ['accuracy','macro_accuracy','macro_f1']:
            values=[r[key] for r in rows]
            if abs(statistics.mean(values)-x[key+'_mean'])>1e-12 or abs(statistics.stdev(values)-x[key+'_seed_sd'])>1e-12:raise ValueError('Summary differs from independent mean/SD')
    seeds=sorted({r['model_seed'] for r in records});receivers=sorted({r['receiver'] for r in records if r['receiver']!='ALL'})
    for p in pairs:
        values=[100*(lookup[(candidate,p['receiver'],s)]['accuracy']-lookup[(p['baseline'],p['receiver'],s)]['accuracy']) for s in seeds]
        if not np.allclose(values,p['accuracy_delta_pp_by_seed'],atol=1e-12,rtol=0) or abs(statistics.mean(values)-p['accuracy_delta_pp_mean'])>1e-12 or abs(statistics.stdev(values)-p['accuracy_delta_pp_seed_sd'])>1e-12:raise ValueError('Paired gains differ')
    tx=[]
    for method in ['native','cvcnn','real_cnn','resnet1d','residual_fusion',candidate]:
        for i,name in enumerate(d['classes']):
            values=[]
            for seed in seeds:
                cm=np.asarray(lookup[(method,'ALL',seed)]['confusion'])
                if cm[i].sum()!=28000:raise ValueError('TX query denominator changed')
                values.append(float(cm[i,i]/cm[i].sum()))
            tx.append(dict(method=method,transmitter=name,query_count=28000,accuracy_mean=statistics.mean(values),accuracy_seed_sd=statistics.stdev(values)))
    csvwrite(e/'per_transmitter_summary.csv',tx);csvwrite(e/'clean_scored_results.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in records]);csvwrite(e/'clean_summary.csv',summary)
    all_summary={r['method']:r for r in summary if r['receiver']=='ALL'};gain={r['baseline']:r for r in pairs if r['receiver']=='ALL'}
    new=all_summary[candidate];improved=gain['residual_fusion']['accuracy_delta_pp_mean']>0
    verdict=dict(status='VERIFIED',candidate=candidate,accuracy_mean=new['accuracy_mean'],accuracy_sd=new['accuracy_seed_sd'],paired_vs_residual=gain['residual_fusion'],paired_vs_native=gain['native'],all192_confusions_recomputed=True,old160_metrics_exactly_unchanged=True,all_mean_sd_and_pairs_recomputed=True,scientific_verdict='PERFORMANCE_IMPROVEMENT' if improved else 'PERFORMANCE_REGRESSION',genuine_RFF_physics_aware_goal_achieved=False,physical_scope='Known-excitation relative response branch with raw identity core and explicit TX/RX confounds; no identified TX parameters',target_feedback=False)
    write(e/'analysis_validation.json',verdict);write(e/'iteration_verdict.json',verdict)
    s='# CVS 已知激励相对响应：独立 clean 测试报告\n\n'
    s+=f"选中 `{candidate}` 的四seed准确率为 **{pct(new['accuracy_mean'])} ± {100*new['accuracy_seed_sd']:.4f}%**，Macro-F1 为 {pct(new['macro_f1_mean'])} ± {100*new['macro_f1_seed_sd']:.4f}%。较原残差 CVS 配对 {gain['residual_fusion']['accuracy_delta_pp_mean']:+.4f} 个百分点，较原 CVS {gain['native']['accuracy_delta_pp_mean']:+.4f} 个百分点。\n\n"
    s+=('本轮在固定clean闭集基准上取得平均性能提升。' if improved else '本轮没有进一步提高平均性能，保留全部负结果。')+'参数轻量是次要条件；四seed证据不保证每个seed、RX或TX改善。参考分支的受限flat增益/相位归一化与三阶响应证据仍不足以声称整网TX/RX因果分离，真正RFF physics-aware目标尚未完成。\n\n'
    s+='## 同矩阵结果\n\n|模型|准确率均值±seed标准差|Macro-F1均值±seed标准差|\n|---|---:|---:|\n'
    for method in ['native','cvcnn','real_cnn','resnet1d','residual_fusion',candidate]:
        a=all_summary[method];s+=f"|{method}|{pct(a['accuracy_mean'])}±{100*a['accuracy_seed_sd']:.4f}%|{pct(a['macro_f1_mean'])}±{100*a['macro_f1_seed_sd']:.4f}%|\n"
    s+='\n|对照|配对均值±SD（百分点）|正提升seed|四seed逐行差分（百分点）|\n|---|---:|---:|---|\n'
    for method in ['residual_fusion','native','cvcnn','real_cnn','resnet1d']:
        a=gain[method];s+=f"|{method}|{a['accuracy_delta_pp_mean']:+.4f}±{a['accuracy_delta_pp_seed_sd']:.4f}|{a['positive_seeds']}/4|"+', '.join(f'{x:+.4f}' for x in a['accuracy_delta_pp_by_seed'])+'|\n'
    s+='\n|seed|原CVS|原残差CVS|本轮CVS|\n|---|---:|---:|---:|\n'
    for seed in seeds:s+=f"|{seed}|"+'|'.join(pct(lookup[(method,'ALL',seed)]['accuracy']) for method in ['native','residual_fusion',candidate])+'|\n'
    s+='\n## RX/TX 分层\n\n|RX（每seed24000包）|原CVS|原残差CVS|本轮CVS均值±SD|较残差（百分点）|\n|---|---:|---:|---:|---:|\n'
    by={(x['method'],x['receiver']):x for x in summary}
    for rx in receivers:
        a=by[(candidate,rx)];b=by[('residual_fusion',rx)];s+=f"|{rx}|{pct(by[('native',rx)]['accuracy_mean'])}|{pct(b['accuracy_mean'])}|{pct(a['accuracy_mean'])}±{100*a['accuracy_seed_sd']:.4f}%|{100*(a['accuracy_mean']-b['accuracy_mean']):+.4f}|\n"
    s+='\n|TX（每seed28000包）|原CVS|原残差CVS|本轮CVS均值±SD|\n|---|---:|---:|---:|\n';txby={(x['method'],x['transmitter']):x for x in tx}
    for name in d['classes']:
        a=txby[(candidate,name)];s+=f"|{name}|{pct(txby[('native',name)]['accuracy_mean'])}|{pct(txby[('residual_fusion',name)]['accuracy_mean'])}|{pct(a['accuracy_mean'])}±{100*a['accuracy_seed_sd']:.4f}%|\n"
    source=read(root/'automation_reports/CV-SincNet'/SOURCE_RUN/'evidence/source_research_complete.json')
    rr=[r for r in source['rows'] if r['resolved']['variant']==candidate];p=rr[0]['profile']
    preds=[r['completion'] for r in d['rows'] if r['resolved']['variant']==candidate]
    resource=dict(candidate=candidate,parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],gradient_parameters=p['gradient_used_parameters'],resident_model_state_bytes=p['resident_state_bytes'],conv_linear_macs=p['conv_linear_macs_per_sample'],inference_batch1_ms_mean=statistics.mean(r['profile']['inference_batch1_ms'] for r in rr),train_batch128_ms_mean=statistics.mean(r['profile']['training_batch128_ms'] for r in rr),actual_training_peak_bytes=max(x['peak_cuda_allocated_bytes'] for r in rr for x in r['epochs']),full_clean_prediction_seconds_mean=statistics.mean(x['prediction_seconds'] for x in preds),actual_prediction_peak_bytes=max(x['peak_cuda_allocated_bytes'] for x in preds),hardware=p['hardware'],torch_version=p['torch_version'],mac_scope=p['mac_scope'],public_reference_buffer_bytes=p['public_reference_buffer_bytes'],matched_correlator_complex_macs=p['matched_correlator_complex_macs_per_sample'],matched_correlator_real_mac_equivalent=p['matched_correlator_real_multiply_accumulate_equivalent'],new_onboard_transmission_bytes=None)
    write(e/'resource_summary.json',resource)
    s+='\n## 固定协议、物理证据与开销\n\n源选择已在任何新query前冻结：4个新scratch E200×50记录，加4个已有残差源指标，最高四seed0.5×V+0.5×最差源RX优先，完全并列才比成本。实际源来源/全部物理角色/表示/预算重算一致；选择不读取目标成绩，不查询未选候选。完整曲线、源分层、数学属性、三阶相对响应闭式、固定TX→信道→RX合成交换与相同波形混淆反例见[完整源报告](../'+SOURCE_RUN+'/report.md)。\n\n'
    s+=f"所有学习参数 {p['total_parameters']} 个，比残差控制多12800个；参考数学前端0参数，新投影和门控可训练。checkpoint模型状态 {p['resident_state_bytes']} bytes，另有固定公共参考buffer {p['public_reference_buffer_bytes']} bytes；实数Conv/Linear MAC {p['conv_linear_macs_per_sample']} /包；另计6400 complex correlator MAC（等价25600实数MAC）；不计FFT及归一化等逐元素运算，实际耗时包含它们。RTX3090/Torch2.1/FP32，合成batch1推理均值 {resource['inference_batch1_ms_mean']:.4f} ms，batch128训练步均值 {resource['train_batch128_ms_mean']:.4f} ms。实际训练峰值 {resource['actual_training_peak_bytes']} bytes；168000包预测均值 {resource['full_clean_prediction_seconds_mean']:.3f} s、预测峰值 {resource['actual_prediction_peak_bytes']} bytes。上述状态统计不包含runtime/optimizer/临时分配；公开常量单列，不能据checkpoint字节推断总显存；星载耗时、额外传输bytes、SFT成本为N/A，不据少参数宣称省算力。\n\n"
    s+='四份新预测和二十份旧冻结预测统一24行，逐包面对六类argmax；只读clean输入，无truth/role、query拟合、真实类别配额或全局重排。所有预测完整、物理ID对齐后独立scorer才连接truth。192份混淆矩阵、均值/样本SD与配对差分独立复算；160条旧对照逐字段保持一致。\n\n'
    s+='仅原划分、四model seed、CE唯一、无增强、身份骨干、clean闭集代理基准；历史已暴露，不能称新盲测。不声明LEO、未知类、新类注册或部署适应。D92三阶段与K×新增类指标为N/A。目标结果不回流结构、超参数、选模、重排或选择性重跑。\n\n'
    s+='[全192条评分与混淆矩阵](evidence/clean_scored_results.json) · [评分CSV](evidence/clean_scored_results.csv) · [全模型TX表](evidence/per_transmitter_summary.csv) · [分析复算](evidence/analysis_validation.json) · [资源JSON](evidence/resource_summary.json) · [最终独立读回](evidence/final_readback.json)。状态ANALYZED；实际commit/PID/输出见独立读回，Git最终交付另行读回确认。\n'
    (report/'report.md').write_text(s,encoding='utf-8');print(json.dumps(verdict,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
