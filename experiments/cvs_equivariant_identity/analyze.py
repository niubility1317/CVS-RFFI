"""Complete source evidence; physical explanations never rank candidates."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.cvs_equivariant_identity.prepare import RUN
from experiments.cvs_equivariant_identity.collect import validate_completed

CONTROL_RUN='20261001-phase1-cvs-residual-identity-manysig-m8-r01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def pct(x):return f'{100*x:.4f}%'


def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence'
    data=read(e/'source_research_complete.json');selection=validate_completed(data)
    previous=read(root/'automation_reports/CV-SincNet'/CONTROL_RUN/'evidence/source_research_complete.json')
    controls=[r for r in previous['rows'] if r['resolved']['variant']=='residual_fusion']
    if len(controls)!=4:raise ValueError('Missing four original source curves')
    for r in controls:
        actual=next(x for x in data['source_controls'] if x['seed']==r['resolved']['model_seed'])
        m=r['completion']['final_source_metrics']
        if actual['accuracy']!=m['source_val_accuracy'] or actual['worst_rx']!=m['source_val_worst_rx']:
            raise ValueError('Original source curves differ from live verified controls')
    allrows=controls+data['rows'];curves=[];last=[];rx=[];cost=[];gates=[];geometry=[];cascade=[];isolated=[];phases=[]
    for r in allrows:
        cfg=r['resolved'];v=cfg['variant'];seed=cfg['model_seed'];epochs=r['epochs']
        if [x['epoch'] for x in epochs]!=list(range(1,201)) or any(x['optimizer_steps']!=50 or x['source_sample_exposure']!=6300 for x in epochs):
            raise ValueError('Complete source curve/budget mismatch')
        for x in epochs:
            curves.append(dict(variant=v,seed=seed,epoch=x['epoch'],source_val_accuracy=x['source_val_accuracy'],source_val_worst_rx=x['source_val_worst_rx'],source_val_ce=x['source_val_ce'],train_ce=x['clean_ce'],gradient_norm=x['gradient_norm'],learning_rate=x['learning_rate'],elapsed_seconds=x['elapsed_seconds']))
            if v=='equivariant_memory':gates.append(dict(seed=seed,epoch=x['epoch'],**x['equivariant_diagnostics']))
        f=epochs[-1];best=max(epochs,key=lambda x:x['source_val_accuracy'])
        last.append(dict(variant=v,seed=seed,fixed_E200_V=f['source_val_accuracy'],fixed_E200_worst_RX=f['source_val_worst_rx'],final_CE=f['clean_ce'],best_V=best['source_val_accuracy'],best_V_epoch=best['epoch'],best_epoch_used_for_selection=False,elapsed_with_diagnostics_profile_seconds=r['completion']['elapsed_seconds']))
        rx.extend(dict(variant=v,seed=seed,receiver=k,accuracy=a) for k,a in f['source_val_rx_accuracy'].items())
        p=r['profile'];cost.append(dict(variant=v,seed=seed,hardware=p['hardware'],torch_version=p['torch_version'],parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],gradient_parameters=p['gradient_used_parameters'],resident_state_bytes=p['resident_state_bytes'],conv_linear_macs=p['conv_linear_macs_per_sample'],fft_calls=p['fft_calls_per_sample'],batch1_inference_ms=p['inference_batch1_ms'],batch128_inference_ms=p['inference_batch128_ms'],batch128_train_ms=p['training_batch128_ms'],actual_training_peak_bytes=max(x['peak_cuda_allocated_bytes'] for x in epochs),profile_clone_peak_bytes=p['benchmark_peak_cuda_allocated_bytes']))
        if v=='equivariant_memory':
            diag=r['source_diagnostics'];phys=r['physical_diagnostics']
            geometry.extend(dict(seed=seed,**{k:a for k,a in g.items() if k!='unit_embedding_mean'}) for g in diag['groups'])
            cascade.extend(dict(seed=seed,**{k:a for k,a in g.items() if k!='source_classifier_logits'}) for g in phys['records'])
            isolated.extend(dict(seed=seed,**g) for g in phys['isolated_tx_changes'])
            phases.extend(dict(seed=seed,**g) for g in phys['phase_audit'])
    paired=[]
    for r in last:
        if r['variant']=='equivariant_memory':
            c=next(x for x in last if x['variant']=='residual_fusion' and x['seed']==r['seed'])
            paired.append(dict(seed=r['seed'],V_delta_pp=100*(r['fixed_E200_V']-c['fixed_E200_V']),worst_RX_delta_pp=100*(r['fixed_E200_worst_RX']-c['fixed_E200_worst_RX'])))
    rx_summary=[]
    for name in ['identity','flat_phase_gain','relative_cfo','lti_channel','rx_iq_image','rx_cubic']:
        rows=[r for r in cascade if r['tx']=='ideal' and r['rx']==name]
        rx_summary.append(dict(tx='ideal',rx=name,seeds=len(rows),whole_unit_embedding_distance_mean=statistics.mean(r['unit_embedding_distance_same_tx_identity_rx'] for r in rows)))
    tables=dict(source_curves=curves,source_fixed_last_and_best=last,source_rx_by_seed=rx,source_resource_by_seed=cost,source_equivariant_by_epoch=gates,source_tx_rx_day=geometry,source_physics_cascade=cascade,source_physics_isolated_tx=isolated,source_whole_phase_audit=phases,source_paired_control=paired,source_physics_rx_summary=rx_summary)
    for name,rows in tables.items():csvwrite(e/(name+'.csv'),rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for v in selection['candidate_universe']:
        rows=[r for r in allrows if r['resolved']['variant']==v]
        for ax,key,label in zip(axes,['source_val_accuracy','source_val_worst_rx','clean_ce'],['Source V accuracy (%)','Worst source RX accuracy (%)','Training CE']):
            a=np.asarray([[x[key] for x in r['epochs']] for r in rows]);factor=1 if key=='clean_ce' else 100
            mean=a.mean(0)*factor;sd=a.std(0,ddof=1)*factor
            ax.plot(range(1,201),mean,label=v);ax.fill_between(range(1,201),mean-sd,mean+sd,alpha=.13);ax.set_xlabel('Epoch');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[2].set_yscale('log');fig.savefig(e/'source_curves.png',dpi=200);fig.savefig(e/'source_curves.pdf');plt.close(fig)
    phase_error=max(r['whole_logit_max_abs_error'] for r in phases);phase_distance=max(r['whole_unit_embedding_max_distance'] for r in phases)
    validation=dict(status='VERIFIED',new_rows=4,control_rows=4,new_epochs=800,new_steps=40000,curve_rows=len(curves),source_ranking_recomputed=True,selected_variant=selection['selected_variant'],new_candidate_selected=selection['new_candidate_selected'],target_access=False,target_scores_used=False,genuine_RFF_physics_aware_goal_achieved=False,whole_phase_max_logit_error=phase_error,whole_phase_max_unit_embedding_distance=phase_distance,physical_scope='Whole constant-phase constraint only; causal memory filters with full-packet RMS/readout; relative phase retained; no universal RX/CFO/channel invariance or unique TX parameter recovery')
    write(e/'source_analysis_validation.json',validation)
    text='# CVS 全路径复相位等变记忆网络：完整源实验报告\n\n'
    text+='四个新scratch模型完成E200×50。完整核对800轮、40000步、详细文本、epoch JSONL/CSV，实际为普通CE权重1、无增强、无域骨干、无继承。原物理L6300/V27000、U56700unused，每轮50步覆盖6300样本。原残差四份完整源曲线与实时核实的终值一致。\n\n'
    text+='固定源规则选择 `'+selection['selected_variant']+'`。'+('新候选胜出，下一步默认执行冻结后4seed clean测试；源成绩不能证明测试提升。' if selection['new_candidate_selected'] else '原残差基线胜出，新网络不访问query，测试N/A；默认收尾独立核实历史测试，不重跑历史。')+'\n\n'
    text+='|候选|四seed源V均值|最差源RX均值|固定性能分数|参数|\n|---|---:|---:|---:|---:|\n'
    for v,a in selection['source_summaries'].items():text+=f"|{v}|{pct(a['source_accuracy'])}|{pct(a['worst_rx_accuracy'])}|{pct(a['score'])}|{a['parameters']}|\n"
    text+='\n分数为四seed的0.5V＋0.5最差源RX均值，最高优先，完全并列后才比V、最差RX、MAC、参数及固定顺序。公共相位数值误差与合成机制不参与排名或停止。\n\n'
    text+='|模型|seed|E200源V|E200最差源RX|末轮CE|最佳源V（仅诊断）|最佳轮（未选）|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in last:text+=f"|{r['variant']}|{r['seed']}|{pct(r['fixed_E200_V'])}|{pct(r['fixed_E200_worst_RX'])}|{r['final_CE']:.6f}|{pct(r['best_V'])}|{r['best_V_epoch']}|\n"
    text+='\n![完整源曲线](evidence/source_curves.png)\n\n四seed样本SD阴影覆盖完整200轮；[全部1600曲线记录](evidence/source_curves.csv)、[源RX逐seed](evidence/source_rx_by_seed.csv)、[同seed源差分](evidence/source_paired_control.csv)。源RX仍为源验证，不称作未知RX测试。\n\n'
    text+='## 数学约束与物理证据\n\nSinc使用相同实系数作用于I/Q；时间与received记忆路径的无bias复卷积、逐包复通道RMS、实径向门控均保持公共常相位等变。功率、lag1/2复相关和位置功率形成不变读出，原rawFFT频谱也不变，全部身份输入无未约束IQ旁路。相对相位/CFO仍可进入复相关，未将波形全部取模。行为基底及左padding滤波因果，但RMS/读出使用整包，整体不是在线因果模型。\n\n'
    text+=f'冻结后每seed对固定30个TX/RX组合测试3个公共相位，四seed最大logit误差为{phase_error:.8g}，最大单位嵌入距离为{phase_distance:.8g}。这是90个扰动/seed的实测数值证据，不是身份准确率或任意接收链解耦证据。数值容差1e-3只报告，不作为选模门槛。\n\n'
    text+='|seed|公共相位rad|全网logit最大绝对误差|单位嵌入最大距离|\n|---|---:|---:|---:|\n'
    for r in phases:text+=f"|{r['seed']}|{r['theta_radians']}|{r['whole_logit_max_abs_error']:.8g}|{r['whole_unit_embedding_max_distance']:.8g}|\n"
    text+='\n|seed|源V的TX中心间平方距离|同TX各RX中心偏移平方距离|\n|---|---:|---:|\n'
    for r in data['rows']:
        d=r['source_diagnostics'];text+=f"|{r['resolved']['model_seed']}|{d['mean_between_tx_centroid_squared_distance']:.6g}|{d['mean_within_tx_rx_centroid_squared_distance']:.6g}|\n"
    text+='\n源V几何覆盖27000包/90个TX×RX×day组；它检验表征关联，不证明硬件因果分离。实际时间/记忆路径权重、径向门控和末batch梯度见[全部800轮](evidence/source_equivariant_by_epoch.csv)，不把非零梯度当作性能归因。\n\n'
    text+='受控链沿用100Msps公共稳态L-STF→TX cubic/image/memory→周期FIR→RX gain/image/cubic→12tone投影→25Msps/相对CFO/RMS。5TX×6RX只用于冻结机制解释，没有增强、正式数据、目标、噪声/瞬态或真实WiSig完整均衡器复现。received多项式也可能表示RX/均衡器影响，不是已辨识TX PA系数。\n\n'
    text+='|固定理想TX的RX/信道变化|全网单位嵌入距离均值（四seed）|\n|---|---:|\n'
    for r in rx_summary:text+=f"|{r['rx']}|{r['whole_unit_embedding_distance_mean']:.6g}|\n"
    text+='\n[全部120组合](evidence/source_physics_cascade.csv)、[孤立TX变化](evidence/source_physics_isolated_tx.csv)保留TX敏感性与RX扰动。TX/RX镜像及三阶相同received波形的反例仍成立，完整证据保留误差。全网公共相位不变、RX干扰稳定性及真实身份性能分别评价；不声称CFO/LTI/RX普遍不变或TX参数唯一恢复。\n\n'
    text+='## 实际资源\n\n|模型|参数|checkpoint模型bytes|实数Conv/Linear MAC/包|batch1推理ms均值|batch128训练ms均值|实际训练峰值bytes最大|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for v in selection['candidate_universe']:
        rr=[r for r in cost if r['variant']==v];p=rr[0];text+=f"|{v}|{p['parameters']}|{p['resident_state_bytes']}|{p['conv_linear_macs']}|{statistics.mean(r['batch1_inference_ms'] for r in rr):.4f}|{statistics.mean(r['batch128_train_ms'] for r in rr):.4f}|{max(r['actual_training_peak_bytes'] for r in rr)}|\n"
    text+='\n新模型202553参数，比原164225增加38328（约23.34%）；成本次要，不据更多或更少参数宣称成功。复卷积实数矩阵实际执行已由Conv计入MAC；FFT/RMS/门控/读出逐元素运算不计入MAC，但实际计时和显存包含它们。RTX3090/Torch2.1/FP32，一次性副本profile；并发条件可能影响细小计时差。checkpoint状态不包括optimizer/临时分配；星载耗时、额外传输bytes和适应训练N/A。[逐seed资源](evidence/source_resource_by_seed.csv)。\n\n'
    text+='尚无本轮新clean成绩。历史已暴露六类clean为代理基准，不扩展LEO/SFT/新类；D92三阶段及K×新增类N/A。任何新测试结果不反馈结构、参数、选模或重跑；负结果全部保留。真正RFF physics aware与实际识别优势仍需独立证据，目标未宣称完成。\n\n'
    text+='[前瞻结构与边界](../../../docs/CVS_EQUIVARIANT_IDENTITY_HYPOTHESIS_20261002.md) · [完整源审计](evidence/source_completion_validation.json) · [固定源选择](evidence/source_selection.json) · [独立分析](evidence/source_analysis_validation.json)。\n'
    (report/'report.md').write_text(text,encoding='utf-8');print(json.dumps(validation,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
