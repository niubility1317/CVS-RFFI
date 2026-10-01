"""Complete frozen source evidence; descriptive diagnostics never feed selection."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from experiments.cvs_gauge_identity.prepare import RUN,PREDECESSOR,CONFIRM_RUN
from experiments.cvs_gauge_identity.dispatch import select_source_candidate


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def pct(v):return f'{100*v:.4f}%'


def analyze(root):
    out=root/'automation_reports/CV-SincNet'/RUN;e=out/'evidence'
    d=read(e/'source_research_complete.json');selection=d['source_selection']
    prior=read(root/'automation_reports/CV-SincNet'/PREDECESSOR/'evidence/source_research_complete.json')
    controls=[r for r in prior['rows'] if r['resolved']['variant']=='residual_fusion']
    if len(controls)!=4 or len(d['rows'])!=8:raise ValueError('Incomplete source matrix')
    for row in controls:
        actual=next(r for r in d['source_controls'] if r['seed']==row['resolved']['model_seed'])
        m=row['completion']['final_source_metrics']
        if (m['source_val_accuracy']!=actual['accuracy'] or m['source_val_worst_rx']!=actual['worst_rx'] or len(row['epochs'])!=200):
            raise ValueError('Historical source curve differs from current control evidence')
    records=list(d['source_controls'])
    for row in d['rows']:
        c=row['resolved'];m=row['completion']['final_source_metrics'];p=row['profile']
        records.append(dict(variant=c['variant'],seed=c['model_seed'],accuracy=m['source_val_accuracy'],worst_rx=m['source_val_worst_rx'],parameters=p['total_parameters'],macs=p['conv_linear_macs_per_sample']))
    if select_source_candidate(records)!=selection:raise ValueError('Independent source ranking differs')
    allrows=controls+d['rows'];curves=[];last=[];rx=[];resources=[];physics=[];groups=[]
    for row in allrows:
        c=row['resolved'];variant=c['variant'];seed=c['model_seed'];epochs=row['epochs']
        if [x['epoch'] for x in epochs]!=list(range(1,201)):raise ValueError('Incomplete curve')
        if any(x['optimizer_steps']!=50 or x['source_sample_exposure']!=6300 for x in epochs):raise ValueError('Budget changed')
        for x in epochs:
            curves.append(dict(variant=variant,seed=seed,epoch=x['epoch'],source_val_accuracy=x['source_val_accuracy'],source_val_worst_rx=x['source_val_worst_rx'],source_val_ce=x['source_val_ce'],train_ce=x['clean_ce'],gradient_norm=x['gradient_norm'],learning_rate=x['learning_rate'],elapsed_seconds=x['elapsed_seconds']))
        best=max(epochs,key=lambda x:x['source_val_accuracy']);last.append(dict(variant=variant,seed=seed,fixed_E200_V=epochs[-1]['source_val_accuracy'],fixed_E200_worst_RX=epochs[-1]['source_val_worst_rx'],final_CE=epochs[-1]['clean_ce'],best_V=best['source_val_accuracy'],best_V_epoch=best['epoch'],best_epoch_used_for_selection=False,training_seconds=row['completion']['elapsed_seconds']))
        for receiver,accuracy in epochs[-1]['source_val_rx_accuracy'].items():rx.append(dict(variant=variant,seed=seed,receiver=receiver,accuracy=accuracy))
        p=row['profile'];resources.append(dict(variant=variant,seed=seed,hardware=p['hardware'],torch_version=p['torch_version'],parameters=p['total_parameters'],gradient_parameters=p['gradient_used_parameters'],resident_state_bytes=p['resident_state_bytes'],conv_linear_macs=p['conv_linear_macs_per_sample'],batch1_inference_ms=p['inference_batch1_ms'],batch128_inference_ms=p['inference_batch128_ms'],batch128_train_ms=p['training_batch128_ms'],actual_training_peak_bytes=max(x['peak_cuda_allocated_bytes'] for x in epochs),profile_clone_peak_bytes=p['benchmark_peak_cuda_allocated_bytes']))
        if variant!='residual_fusion':
            diag=row['source_diagnostics'];phys=row['physical_diagnostics']
            if diag['count']!=27000 or len(diag['groups'])!=90 or diag['target_access'] or diag['used_for_training'] or diag['used_for_selection']:raise ValueError('Source diagnostic mismatch')
            physics.append(dict(row_id=row['row_id'],variant=variant,seed=seed,phase_logit_error=phys['phase_logit_max_abs_error'],phase_tolerance_pass=phys['phase_logit_max_abs_error']<1e-3,affine_logit_error=phys['affine_logit_max_abs_error'],affine_invariance_claimed=phys['affine_invariance_claimed'],am_am_distance=phys['unit_embedding_distances']['handset_am_am'],am_pm_distance=phys['unit_embedding_distances']['handset_am_pm'],rx_iq_distance=phys['unit_embedding_distances']['rx_iq_counterexample'],multipath_distance=phys['unit_embedding_distances']['multipath_counterexample'],between_tx_centroid_distance=diag['mean_between_tx_centroid_squared_distance'],within_tx_rx_centroid_distance=diag['mean_within_tx_rx_centroid_squared_distance'],last_batch_fallback_fraction_mean=statistics.mean(x['gauge_diagnostics']['fallback_fraction'] for x in epochs)))
            groups += [dict(variant=variant,seed=seed,**{k:v for k,v in g.items() if k!='unit_embedding_mean'}) for g in diag['groups']]
    for name,rows in [('source_curves',curves),('source_fixed_last_and_best',last),('source_rx_by_seed',rx),('source_resource_by_seed',resources),('source_physics_by_seed',physics),('source_tx_rx_day',groups)]:csvwrite(e/(name+'.csv'),rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for variant in selection['candidate_universe']:
        rows=[r for r in allrows if r['resolved']['variant']==variant]
        for ax,key,label in zip(axes,['source_val_accuracy','source_val_worst_rx','clean_ce'],['Source V accuracy (%)','Worst source RX accuracy (%)','Training CE']):
            a=np.asarray([[x[key] for x in r['epochs']] for r in rows]);factor=100 if key!='clean_ce' else 1
            mean=a.mean(0)*factor;sd=a.std(0,ddof=1)*factor;epochs=np.arange(1,201)
            ax.plot(epochs,mean,label=variant);ax.fill_between(epochs,mean-sd,mean+sd,alpha=.13);ax.set_xlabel('Epoch');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[2].set_yscale('log');fig.savefig(e/'source_curves.png',dpi=200);fig.savefig(e/'source_curves.pdf');plt.close(fig)
    paired=[]
    for variant in selection['candidate_universe'][1:]:
        delta=[]
        for row in [r for r in last if r['variant']==variant]:
            control=next(x for x in last if x['variant']=='residual_fusion' and x['seed']==row['seed'])
            delta.append(dict(variant=variant,seed=row['seed'],V_delta_pp=100*(row['fixed_E200_V']-control['fixed_E200_V']),worst_RX_delta_pp=100*(row['fixed_E200_worst_RX']-control['fixed_E200_worst_RX'])))
        paired+=delta
    csvwrite(e/'source_paired_control.csv',paired)
    write(e/'source_analysis_validation.json',dict(status='VERIFIED',new_rows=8,control_rows=4,new_epochs=1600,new_steps=80000,curve_rows=len(curves),source_ranking_recomputed=True,target_access=False,target_scores_used=False,new_candidate_selected=selection['new_candidate_selected'],selected_variant=selection['selected_variant'],genuine_RFF_physics_aware_goal_achieved=False))
    s='# 完整 IQ 相位规范化 CVS：完整源实验报告\n\n'
    s+='八个新模型均完成 E200×50，共解析1600轮、80000步、CSV及全部文本日志。八个新模型为独立 scratch，没有增强、域骨干、继承或附加损失；L6300、V27000、U56700不用，物理角色一致。四个历史残差源控制的完整200轮曲线与当前只读最终指标匹配。\n\n'
    s+='固定源规则选择 `'+selection['selected_variant']+'`。'
    if selection['new_candidate_selected']:
        s+='新相位候选胜出，状态 SOURCE_TRAINING_COMPLETE_AWAITING_CLEAN；按原预登记默认继续四个选中模型的clean测试。尚不能根据源结果声称识别性能提升。\n\n'
    else:
        s+='已有残差控制胜出。两个相位候选不访问新query，测试为N/A；保留历史残差测试，不将源拒绝写成测试收益。真正RFF physics-aware与性能目标仍未完成。\n\n'
    s+='|候选|四seed源V均值|最差源RX均值|性能分数|参数|\n|---|---:|---:|---:|---:|\n'
    for v,p in selection['source_summaries'].items():s+=f"|{v}|{pct(p['source_accuracy'])}|{pct(p['worst_rx_accuracy'])}|{pct(p['score'])}|{p['parameters']}|\n"
    s+='\n性能分数为四seed的0.5×V+0.5×最差源RX均值。完全并列才比V、最差RX、MAC、参数与固定残差/峰值/相干顺序；没有成本优先容差。来源与物理角色/表示/预算已核查，只读源记录，未加载控制权重；目标成绩不参与排名。\n\n'
    s+='|模型|seed|固定E200 V|固定E200最差源RX|末轮训练CE|最佳V（仅诊断）|最佳轮（未选模）|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in last:s+=f"|{r['variant']}|{r['seed']}|{pct(r['fixed_E200_V'])}|{pct(r['fixed_E200_worst_RX'])}|{r['final_CE']:.6f}|{pct(r['best_V'])}|{r['best_V_epoch']}|\n"
    s+='\n![完整源曲线：均值及四seed样本标准差](evidence/source_curves.png)\n\n曲线包括所有200轮，不只末轮。阴影按四个模型seed计算，源RX都是登记的源接收机，不称为独立未知RX测试。完整[2400条曲线CSV](evidence/source_curves.csv)、[源RX逐seed](evidence/source_rx_by_seed.csv)、[配对源差分](evidence/source_paired_control.csv)。\n\n'
    s+='## 冻结物理诊断\n\n全网公共相位不变性在实数运算下成立，完整256点IQ只乘单个复标量。相干平均只确定参考，没有用平均替换IQ或消除整包CFO。所有候选都保留频偏、AM/AM幅度形状和相对AM/PM响应。接收机IQ和多径仍能改变特征，这是明确反例；不声称TX参数唯一辨识。\n\n'
    s+='|新源行|公共相位logit误差|CFO斜率logit差（非不变性声明）|AM/AM距离|AM/PM距离|RX IQ距离|多径距离|末batch回退比例均值|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in physics:s+='|'+r['row_id']+'|'+'|'.join(f'{r[k]:.6g}' for k in ['phase_logit_error','affine_logit_error','am_am_distance','am_pm_distance','rx_iq_distance','multipath_distance','last_batch_fallback_fraction_mean'])+'|\n'
    s+='\n公共相位误差对照0.001作为诊断，任何失败原样保留，不改变训练、排名或测试精度。近峰选点存在不连续边界；FP32/TF32不能保证逐位不变。合成波形是手设周期20信号，不是标准L-STF系数，也不是训练增强或身份测试。源V几何覆盖每行27000包、90个TX/RX/day组；[完整分层](evidence/source_tx_rx_day.csv)及[冻结物理CSV](evidence/source_physics_by_seed.csv)都是解释证据，不回流选择。\n\n'
    s+='## 实际资源与边界\n\n|模型|参数|常驻模型字节|MAC/包|batch1推理ms均值|batch128训练ms均值|实际训练峰值bytes最大|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for v in selection['candidate_universe']:
        rr=[r for r in resources if r['variant']==v]
        s+=f"|{v}|{rr[0]['parameters']}|{rr[0]['resident_state_bytes']}|{rr[0]['conv_linear_macs']}|{statistics.mean(r['batch1_inference_ms'] for r in rr):.4f}|{statistics.mean(r['batch128_train_ms'] for r in rr):.4f}|{max(r['actual_training_peak_bytes'] for r in rr)}|\n"
    s+='\n相位前端0个学习参数，实际所有164225个参数接收CE梯度。MAC仅计Conv/Linear，不含FFT、相位参考、归一化与逐元素运算；这些开销包含实际计时和峰值。RTX3090/Torch2.1/FP32，profile用一次性副本，训练峰值来自profile前日志；历史硬件并发口径保留，不据微小计时差声称加速。常驻字节只计模型，星载runtime/优化器/新增传输字节N/A。完整[资源逐seed](evidence/source_resource_by_seed.csv)。\n\n'
    s+='当前clean是历史已暴露的闭集代理基准。仅CE、无增强、只身份、clean-only授权不扩大到LEO、support微调、新类或校准硬件测量。D92三阶段/K×新增类指标N/A。源指标与可证明物理性质都不能替代独立测试优势，更不能证明TX/RX因果分离。\n\n'
    s+='[前瞻假设](../../../docs/CVS_GAUGE_IDENTITY_HYPOTHESIS_20261002.md) · [完整源审计](evidence/source_completion_validation.json) · [源排名](evidence/source_selection.json) · [独立复算](evidence/source_analysis_validation.json) · [发布读回](evidence/source_launch_readback.json)。\n'
    (out/'report.md').write_text(s,encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',selected=selection['selected_variant'],new_candidate_selected=selection['new_candidate_selected'],source_summaries=selection['source_summaries']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
