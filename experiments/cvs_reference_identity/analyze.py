"""Full source/physics report; explanatory diagnostics never rank candidates."""
import argparse
import csv
import json
from pathlib import Path
import statistics

RUN='20261002-phase1-cvs-reference-identity-manysig-m4-r01'
PREDECESSOR='20261001-phase1-cvs-residual-identity-manysig-m8-r01'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader();writer.writerows(rows)
def pct(value):return f'{100*value:.4f}%'


def analyze(root):
    from experiments.cvs_reference_identity.collect import validate_completed
    report=root/'automation_reports/CV-SincNet'/RUN; evidence=report/'evidence'
    data=read(evidence/'source_research_complete.json');selection=validate_completed(data)
    prior=read(root/'automation_reports/CV-SincNet'/PREDECESSOR/'evidence/source_research_complete.json')
    controls=[r for r in prior['rows'] if r['resolved']['variant']=='residual_fusion']
    if len(controls)!=4:raise ValueError('Missing original source curves')
    for row in controls:
        actual=next(r for r in data['source_controls'] if r['seed']==row['resolved']['model_seed'])
        metrics=row['completion']['final_source_metrics']
        if actual['accuracy']!=metrics['source_val_accuracy'] or actual['worst_rx']!=metrics['source_val_worst_rx']:
            raise ValueError('Historical source curve differs from live source control')
    allrows=controls+data['rows'];curves=[];last=[];receivers=[];resources=[];response=[];geometry=[];cascade=[];isolated=[]
    for row in allrows:
        cfg=row['resolved'];variant=cfg['variant'];seed=cfg['model_seed'];epochs=row['epochs']
        if [x['epoch'] for x in epochs]!=list(range(1,201)) or any(x['optimizer_steps']!=50 or x['source_sample_exposure']!=6300 for x in epochs):
            raise ValueError('Incomplete or mismatched source curve')
        for x in epochs:
            curves.append(dict(variant=variant,seed=seed,epoch=x['epoch'],source_val_accuracy=x['source_val_accuracy'],
                source_val_worst_rx=x['source_val_worst_rx'],source_val_ce=x['source_val_ce'],train_ce=x['clean_ce'],
                gradient_norm=x['gradient_norm'],learning_rate=x['learning_rate'],elapsed_seconds=x['elapsed_seconds']))
            if variant=='reference_response':response.append(dict(seed=seed,epoch=x['epoch'],**x['response_diagnostics']))
        final=epochs[-1];best=max(epochs,key=lambda x:x['source_val_accuracy'])
        last.append(dict(variant=variant,seed=seed,fixed_E200_V=final['source_val_accuracy'],fixed_E200_worst_RX=final['source_val_worst_rx'],
            final_CE=final['clean_ce'],best_V=best['source_val_accuracy'],best_V_epoch=best['epoch'],best_epoch_used_for_selection=False,
            elapsed_with_final_diagnostics_and_profile_seconds=row['completion']['elapsed_seconds']))
        receivers += [dict(variant=variant,seed=seed,receiver=k,accuracy=v) for k,v in final['source_val_rx_accuracy'].items()]
        p=row['profile']
        resources.append(dict(variant=variant,seed=seed,hardware=p['hardware'],torch_version=p['torch_version'],parameters=p['total_parameters'],
            gradient_parameters=p['gradient_used_parameters'],resident_state_bytes=p['resident_state_bytes'],
            public_reference_buffer_bytes=p.get('public_reference_buffer_bytes',0),real_conv_linear_macs=p['conv_linear_macs_per_sample'],
            separate_complex_correlator_macs=p.get('matched_correlator_complex_macs_per_sample',0),
            complex_correlator_real_mac_equivalent=p.get('matched_correlator_real_multiply_accumulate_equivalent',0),
            batch1_inference_ms=p['inference_batch1_ms'],batch128_inference_ms=p['inference_batch128_ms'],batch128_train_ms=p['training_batch128_ms'],
            actual_training_peak_bytes=max(x['peak_cuda_allocated_bytes'] for x in epochs),profile_clone_peak_bytes=p['benchmark_peak_cuda_allocated_bytes']))
        if variant=='reference_response':
            diag=row['source_diagnostics'];phys=row['physical_diagnostics']
            geometry += [dict(variant=variant,seed=seed,**{k:v for k,v in group.items() if k!='unit_embedding_mean'}) for group in diag['groups']]
            cascade += [dict(seed=seed,**{k:v for k,v in rec.items() if k!='source_classifier_logits'}) for rec in phys['records']]
            isolated += [dict(seed=seed,**r) for r in phys['isolated_tx_changes']]
    paired=[]
    for row in last:
        if row['variant']!='reference_response':continue
        control=next(c for c in last if c['variant']=='residual_fusion' and c['seed']==row['seed'])
        paired.append(dict(seed=row['seed'],V_delta_pp=100*(row['fixed_E200_V']-control['fixed_E200_V']),
            worst_RX_delta_pp=100*(row['fixed_E200_worst_RX']-control['fixed_E200_worst_RX'])))
    rx_summary=[]
    for rx in ['identity','flat_phase_gain','relative_cfo','lti_channel','rx_iq_image','rx_cubic']:
        rows=[r for r in cascade if r['tx']=='ideal' and r['rx']==rx]
        rx_summary.append(dict(tx='ideal',rx=rx,seeds=len(rows),
            whole_unit_embedding_distance_mean=statistics.mean(r['unit_embedding_distance_same_tx_identity_rx'] for r in rows),
            relative_transfer_distance_mean=statistics.mean(r['relative_transfer_distance_same_tx_identity_rx'] for r in rows)))
    tables=dict(source_curves=curves,source_fixed_last_and_best=last,source_rx_by_seed=receivers,source_resource_by_seed=resources,
        source_response_by_epoch=response,source_tx_rx_day=geometry,source_physics_cascade=cascade,source_physics_isolated_tx=isolated,
        source_paired_control=paired,source_physics_rx_summary=rx_summary)
    for name,rows in tables.items():csvwrite(evidence/(name+'.csv'),rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for variant in selection['candidate_universe']:
        rows=[r for r in allrows if r['resolved']['variant']==variant]
        for ax,key,label in zip(axes,['source_val_accuracy','source_val_worst_rx','clean_ce'],['Source V accuracy (%)','Worst source RX accuracy (%)','Training CE']):
            values=np.asarray([[x[key] for x in r['epochs']] for r in rows]);factor=1 if key=='clean_ce' else 100
            mean=values.mean(0)*factor;sd=values.std(0,ddof=1)*factor
            ax.plot(range(1,201),mean,label=variant);ax.fill_between(range(1,201),mean-sd,mean+sd,alpha=.13)
            ax.set_xlabel('Epoch');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[2].set_yscale('log')
    fig.savefig(evidence/'source_curves.png',dpi=200);fig.savefig(evidence/'source_curves.pdf');plt.close(fig)
    validation=dict(status='VERIFIED',new_rows=4,control_rows=4,new_epochs=800,new_steps=40000,curve_rows=1600,
        source_ranking_recomputed=True,selected_variant=selection['selected_variant'],new_candidate_selected=selection['new_candidate_selected'],
        target_access=False,target_scores_used=False,genuine_RFF_physics_aware_goal_achieved=False,
        interpretation='Known-excitation response with controlled sensitivity evidence; no identified TX parameters or universal whole-network RX/channel invariance; '+
        ('new clean result pending source-selected completion' if selection['new_candidate_selected'] else 'unselected reference clean N/A; retain independently verified original baseline test'))
    write(evidence/'source_analysis_validation.json',validation)
    text='# CVS 已知激励相对响应：完整源实验报告\n\n'
    text+='四个新模型完成 E200×50，完整解析 800 轮、40000 步，将逐步 JSONL、逐轮 JSONL/CSV 与详细文本日志逐项核对。相同物理 L6300/V27000，U56700 不用；scratch、普通身份 CE、无增强、无域骨干、无继承或附加损失。四份原残差源控制的完整曲线与实时核实的最终源指标一致。\n\n'
    text+='固定性能优先源规则选择 `'+selection['selected_variant']+'`。'
    text+=('新候选胜出，等待默认执行冻结四 seed 的 clean 测试；源结果尚不能证明测试提升。\n\n' if selection['new_candidate_selected'] else '原残差基线胜出，新候选不访问 query，测试为 N/A；默认收尾保留并独立核实原历史测试，不把源拒绝写成测试收益。\n\n')
    text+='|候选|四 seed 源 V 均值|最差源 RX 均值|性能分数|参数|\n|---|---:|---:|---:|---:|\n'
    for variant,item in selection['source_summaries'].items():text+=f"|{variant}|{pct(item['source_accuracy'])}|{pct(item['worst_rx_accuracy'])}|{pct(item['score'])}|{item['parameters']}|\n"
    text+='\n分数为四 seed 的 0.5×V＋0.5×最差源 RX 均值。性能完全并列后才比较 V、最差 RX、实数 Conv/Linear MAC、参数及原残差/新响应固定顺序。没有成本容差带，没有目标成绩参与排名，没有加载源控制权重。\n\n'
    text+='|模型|seed|固定 E200 V|固定 E200 最差源 RX|末轮训练 CE|最佳 V（仅诊断）|最佳轮（未选模）|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in last:text+=f"|{row['variant']}|{row['seed']}|{pct(row['fixed_E200_V'])}|{pct(row['fixed_E200_worst_RX'])}|{row['final_CE']:.6f}|{pct(row['best_V'])}|{row['best_V_epoch']}|\n"
    text+='\n![完整源曲线](evidence/source_curves.png)\n\n阴影为四 seed 的样本标准差，含所有 200 轮；源 RX 分层仍是源验证，不称作未知 RX 测试。完整 [1600 条曲线](evidence/source_curves.csv)、[源 RX 逐 seed](evidence/source_rx_by_seed.csv)、[配对源差分](evidence/source_paired_control.csv)。\n\n'
    text+='## 物理机制与实际学习\n\n已知公共 L-STF 的 12 个激励频点形成相对复响应，附带匹配质量、周期相干度、显式相对 CFO 和功率比例。它是相对接收响应，可能包含 TX、信道、RX 和均衡器影响。仅参考分支在非退化输入上近似消去一个 flat 复增益；原始身份路径保留，不宣称全网公共相位或任意信道不变。\n\n'
    text+='|seed|E200 门控绝对值均值|末 batch 门控梯度范数|末 batch 投影梯度范数|源 V TX 中心间平方距离|同 TX 各 RX 中心偏移平方距离|\n|---|---:|---:|---:|---:|---:|\n'
    for row in data['rows']:
        diag=row['source_diagnostics'];r=row['epochs'][-1]['response_diagnostics'];seed=row['resolved']['model_seed']
        text+=f"|{seed}|{r['response_gain_abs_mean']:.6g}|{r['response_gain_gradient_norm']:.6g}|{r['response_projection_gradient_norm']:.6g}|{diag['mean_between_tx_centroid_squared_distance']:.6g}|{diag['mean_within_tx_rx_centroid_squared_distance']:.6g}|\n"
    text+='\n门控和梯度来自每轮最后一个源 batch，不是完整源 V 的分支贡献消融；非零门控只能证明分支参与学习，不能单独证明其改善身份判别。源 V 几何覆盖每行 27000 包及 90 个 TX×RX×day 组，不把组间关联解释成硬件因果分离。[全部 800 轮响应与梯度](evidence/source_response_by_epoch.csv)、[360 个源分层单元](evidence/source_tx_rx_day.csv)。\n\n'
    text+='冻结受控链为 100 Msps 公共稳态激励→TX 三阶/镜像/记忆→周期 FIR 信道→RX 增益/镜像/三阶→12 tone 投影→25 Msps 采样→相对 CFO→RMS。每个模型固定 30 个组合；没有训练增强、目标访问、噪声/瞬态或真实 WiSig MMSE 均衡复现。合成配置不是六个真实源身份，不给合成身份准确率。\n\n'
    text+='|seed|孤立 TX 变化|整网单位嵌入距离|相对响应距离|\n|---|---|---:|---:|\n'
    for rec in isolated:text+=f"|{rec['seed']}|{rec['tx']}|{rec['unit_embedding_distance_to_ideal_tx']:.6g}|{rec['relative_transfer_distance_to_ideal_tx']:.6g}|\n"
    text+='\n上述 TX 变化固定 identity RX，以理想 TX 为参照；[全部 120 个 TX/RX 组合](evidence/source_physics_cascade.csv)同时保留各 RX/信道扰动下同 TX 的变化。unity 信道下 TX 镜像与 RX 镜像、TX 三阶与 RX 三阶能产生相同观测，逐行最大波形误差保存在完整证据中。未知频响还可吸收部分带内失真；不能从该结构宣称 TX PA/IQ 参数唯一辨识、校准晶振频偏或任意接收机解耦。\n\n'
    text+='|固定理想 TX 的 RX/信道变化|相对复响应距离均值|整网单位嵌入距离均值（四 seed）|\n|---|---:|---:|\n'
    for item in rx_summary:text+=f"|{item['rx']}|{item['relative_transfer_distance_mean']:.6g}|{item['whole_unit_embedding_distance_mean']:.6g}|\n"
    text+='\nflat 复增益/相位经整包 RMS 后消去幅度尺度，参考分支的相对复响应变化接近数值误差，整网嵌入仍明显改变。相对 CFO 的形状校正也不等于整网 CFO 不变，模型显式保留 CFO。以上是本轮可定位的局限：局部物理坐标没有约束原始身份路径；不能把它包装成已经实现整体物理感知。该受控反例不证明源指标小幅下降的唯一原因，也不改变候选排名。下一轮应先研究全部身份路径的干扰作用及合法源域身份信息保留，再前瞻登记新的结构；不依据历史目标成绩调参。[RX 受控汇总](evidence/source_physics_rx_summary.csv)。\n\n'
    text+='## 实际资源\n\n|模型|参数|checkpoint 模型字节|额外公共参考 buffer 字节|实数 Conv/Linear MAC/包|独立 complex correlator MAC/包|batch1 推理 ms 均值|batch128 训练 ms 均值|训练峰值 bytes 最大|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for variant in selection['candidate_universe']:
        rr=[r for r in resources if r['variant']==variant];p=rr[0]
        text+=f"|{variant}|{p['parameters']}|{p['resident_state_bytes']}|{p['public_reference_buffer_bytes']}|{p['real_conv_linear_macs']}|{p['separate_complex_correlator_macs']}|{statistics.mean(r['batch1_inference_ms'] for r in rr):.4f}|{statistics.mean(r['batch128_train_ms'] for r in rr):.4f}|{max(r['actual_training_peak_bytes'] for r in rr)}|\n"
    text+='\n新增 12800 参数（7.79%）；固定前端无学习参数，但有公共 buffer 和 6400 complex MAC（等价 25600 实数 MAC）的相关库开销。Conv/Linear 数量不含 FFT/归一化/逐元素运算，实际计时及峰值包含所有执行。公共 buffer 不进入 checkpoint，常驻量另计。RTX3090/Torch2.1/FP32，用一次性副本 profile；历史并发条件可能影响细小计时差，不能据此声称加速。星载 runtime、优化器常驻及新增传输字节 N/A。[资源逐 seed](evidence/source_resource_by_seed.csv)。\n\n'
    text+='当前没有新的 clean 测试结论；历史已暴露的六类 clean 是代理基准。协议不扩展到 LEO、SFT、新类或硬件标定；D92 三阶段/K×新增类指标 N/A。物理构造、机制证据与独立识别性能分别评价，真正 RFF physics aware 和性能研发目标仍未宣称完成。\n\n'
    text+='[前瞻设计与闭式](../../../docs/CVS_REFERENCE_RESPONSE_HYPOTHESIS_20261002.md) · [完整源审计](evidence/source_completion_validation.json) · [源排名](evidence/source_selection.json) · [独立分析](evidence/source_analysis_validation.json)。\n'
    (report/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
