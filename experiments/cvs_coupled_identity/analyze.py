"""Full source curves/physics/geometry/resources; no target score inputs."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.cvs_coupled_identity.prepare import RUN
from experiments.cvs_coupled_identity.collect import validate_completed
from experiments.cvs_coupled_identity.model import VARIANTS
CONTROL_RUN='20261002-phase1-cvs-energy-identity-manysig-m8-r01'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def table(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence'
    data=read(e/'source_research_complete.json');selection=validate_completed(data)
    controls=[r for r in read(root/'automation_reports/CV-SincNet'/CONTROL_RUN/'evidence/source_research_complete.json')['rows'] if r['resolved']['variant']=='energy_equivariant']
    if len(controls)!=4:raise ValueError('Missing complete original source curves')
    for r in controls:
        actual=next(a for a in data['source_controls'] if a['seed']==r['resolved']['model_seed'])
        if any(actual[k]!=r['completion']['final_source_metrics'][v] for k,v in [('accuracy','source_val_accuracy'),('worst_rx','source_val_worst_rx')]):
            raise ValueError('Historical source curves differ from independently verified fixed source controls')
    curves=[];final=[];rx=[];resources=[];sync=[];cells=[];geometry=[];cascade=[];tx=[];phases=[];paired=[];conditioner=[];normalizations=[];public_normalizations=[];inputs=[];public_inputs=[]
    for r in controls+data['rows']:
        c=r['resolved'];v=c['variant'];seed=c['model_seed'];epochs=r['epochs'];p=r['profile']
        if [a['epoch'] for a in epochs]!=list(range(1,201)):raise ValueError('Incomplete source curve')
        for a in epochs:
            curves.append(dict(variant=v,seed=seed,epoch=a['epoch'],V=a['source_val_accuracy'],worst_RX=a['source_val_worst_rx'],V_CE=a['source_val_ce'],train_CE=a['clean_ce'],gradient_norm=a['gradient_norm'],learning_rate=a['learning_rate'],elapsed_seconds=a['elapsed_seconds']))
            if v in VARIANTS:
                d=a['coupled_diagnostics'];flat={k:value for k,value in d.items() if not isinstance(value,dict)}
                flat.update({'core_'+k:value for k,value in d.get('core',{}).items()})
                sync.append(dict(variant=v,seed=seed,epoch=a['epoch'],**flat))
                conditioner.append(dict(variant=v,seed=seed,epoch=a['epoch'],gradient_used_parameters=a['alignment_gradient_used_parameters'],gradient_norm=a['alignment_gradient_norm'],alignment_strength=a['alignment_strength'],whole_gradient_norm=a['gradient_norm']))
                normalizations.extend(dict(variant=v,seed=seed,epoch=a['epoch'],**record) for record in d['normalization']['records'])
                inputs.extend(dict(variant=v,seed=seed,epoch=a['epoch'],**record) for record in d['coupled_input']['records'])
        f=epochs[-1];best=max(epochs,key=lambda a:a['source_val_accuracy'])
        final.append(dict(variant=v,seed=seed,V=f['source_val_accuracy'],worst_RX=f['source_val_worst_rx'],CE=f['clean_ce'],best_V=best['source_val_accuracy'],best_epoch=best['epoch'],best_epoch_selected=False))
        rx.extend(dict(variant=v,seed=seed,receiver=k,accuracy=val) for k,val in f['source_val_rx_accuracy'].items())
        resources.append(dict(variant=v,seed=seed,parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],gradient_used_parameters=p['gradient_used_parameters'],resident_state_bytes=p['resident_state_bytes'],conv_linear_macs=p['conv_linear_macs_per_sample'],fft_calls=p['fft_calls_per_sample'],inference_batch1_ms=p['inference_batch1_ms'],inference_batch128_ms=p['inference_batch128_ms'],training_batch128_ms=p['training_batch128_ms'],source_peak_bytes=max(a['peak_cuda_allocated_bytes'] for a in epochs),profile_clone_peak_bytes=p['benchmark_peak_cuda_allocated_bytes'],hardware=p['hardware'],torch_version=p['torch_version'],elapsed_seconds=r['completion']['elapsed_seconds']))
        if v in VARIANTS:
            d=r['source_diagnostics'];ph=r['physical_diagnostics']
            public_normalizations.extend(dict(variant=v,seed=seed,**record) for record in ph['normalization']['records'])
            public_inputs.extend(dict(variant=v,seed=seed,**record) for record in ph['coupled_input']['records'])
            geometry.append(dict(variant=v,seed=seed,between_TX=d['mean_between_tx_centroid_squared_distance'],within_TX_RX=d['mean_within_tx_rx_centroid_squared_distance']))
            cells.extend(dict(variant=v,seed=seed,**{k:val for k,val in a.items() if not isinstance(val,list)}) for a in d['groups'])
            cascade.extend(dict(variant=v,seed=seed,**{k:val for k,val in a.items() if not isinstance(val,list)}) for a in ph['records'])
            tx.extend(dict(variant=v,seed=seed,**a) for a in ph['isolated_tx_changes'])
            phases.extend(dict(variant=v,seed=seed,**a) for a in ph['phase_audit'])
            base=next(a for a in data['source_controls'] if a['seed']==seed)
            paired.append(dict(variant=v,seed=seed,V_delta_pp=100*(f['source_val_accuracy']-base['accuracy']),worst_RX_delta_pp=100*(f['source_val_worst_rx']-base['worst_rx'])))
    if len(curves)!=2400 or len(sync)!=1600 or len(conditioner)!=1600 or len(cells)!=720 or len(phases)!=24 or len(normalizations)!=9600 or len(public_normalizations)!=48:raise ValueError('Incomplete complete source report dimensions')
    table(e/'source_energy_normalization_all9600.csv',normalizations);table(e/'source_public_normalization_all48.csv',public_normalizations)
    if len(inputs)!=1600 or len(public_inputs)!=8:raise ValueError('Incomplete actual coupled input evidence')
    table(e/'source_coupled_input_all1600.csv',inputs);table(e/'source_public_coupled_input_all8.csv',public_inputs)
    for name,rows in dict(source_curves=curves,source_fixed_last_and_best=final,source_RX=rx,source_resources=resources,source_energy_by_epoch=sync,source_alignment_optimization=conditioner,source_all720_cells=cells,source_geometry=geometry,source_physics_cascade=cascade,source_isolated_TX=tx,source_affine_phase_audit=phases,source_paired_control=paired).items():table(e/(name+'.csv'),rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for v in selection['candidate_universe']:
        rows=[r for r in controls+data['rows'] if r['resolved']['variant']==v]
        for ax,k,label,factor in zip(axes,('source_val_accuracy','source_val_worst_rx','clean_ce'),('Source V accuracy (%)','Worst source RX accuracy (%)','Training CE'),(100,100,1)):
            a=np.asarray([[x[k] for x in r['epochs']] for r in rows])*factor;mean=a.mean(0);sd=a.std(0,ddof=1)
            ax.plot(range(1,201),mean,label=v);ax.fill_between(range(1,201),mean-sd,mean+sd,alpha=.12);ax.set_xlabel('Epoch');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);axes[2].set_yscale('log');fig.savefig(e/'source_curves.png',dpi=180);fig.savefig(e/'source_curves.pdf');plt.close(fig)
    validation=dict(status='VERIFIED',new_rows=8,new_epochs=1600,new_steps=80000,control_rows=4,curve_records=2400,
        source_selection_recomputed=True,selected_variant=selection['selected_variant'],new_candidate_selected=selection['new_candidate_selected'],
        actual_fullFP32_training_validation_physics_profile=True,all720_source_cells_complete=True,target_access=False,target_scores_used=False,goal_achieved=False,
        whole_phase_max_logit_error=max(a['whole_logit_max_abs_error'] for a in phases if a['received_cfo_hz']==0.),
        whole_affine_max_logit_error=max(a['whole_logit_max_abs_error'] for a in phases if a['received_cfo_hz']!=0.),
        coordinate_reconstruction_max_error=max(r['physical_diagnostics']['coordinate_reconstruction_max_error'] for r in data['rows']),
        source_normalization_measurements=9600,public_normalization_measurements=48,
        measured_coupled_input_records=len(inputs),public_coupled_input_records=len(public_inputs),
        scope='Wholeconstantphase property;fixed causal envelope-memory input;raw IQ retained;not uniqueTX or arbitraryRX/LTI/frequency invariance')
    write(e/'source_analysis_validation.json',validation)
    text='# CVS 因果包络耦合：完整源实验报告\n\n'
    text+='8 个新 scratch 模型完成 E200×50；完整核对 80000 步、1600 轮、详细文本与紧凑 epoch JSONL/CSV。实际为普通 CE、无增强、无域骨干、无权重继承，原 L6300/V27000、U56700 unused；全部新模型采用固定完整 FP32。当前四份能量保留控制完整源曲线与实时核实终值一致。\n\n'
    text+='固定性能优先源规则选择 `'+selection['selected_variant']+'`。'+('新候选胜出，下一步冻结后默认4行clean测试；源成绩还不能证明测试提升。' if selection['new_candidate_selected'] else '当前源赢家保留，未选包络耦合候选不读query；默认保留已核实历史clean，不重跑历史。')+'\n\n'
    text+='| 候选 | 四seed源V（%） | 最差源RX（%） | 固定性能分数（%） | 参数 |\n|---|---:|---:|---:|---:|\n'
    for v,a in selection['source_summaries'].items():text+=f"| {v} | {100*a['source_accuracy']:.4f} | {100*a['worst_rx_accuracy']:.4f} | {100*a['score']:.4f} | {a['parameters']} |\n"
    text+='\n四 seed mean(0.5V+0.5最差源RX)最高优先；完全并列后才比较V、最差RX和成本。物理误差不参与选模，最佳轮只作诊断，所有选模使用E200。\n\n| 模型 | seed | E200 V（%） | E200 最差RX（%） | 末轮CE | 最佳V（%） | 最佳轮（未选） |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for a in final:text+=f"| {a['variant']} | {a['seed']} | {100*a['V']:.4f} | {100*a['worst_RX']:.4f} | {a['CE']:.6f} | {100*a['best_V']:.4f} | {a['best_epoch']} |\n"
    text+='\n![完整源曲线](evidence/source_curves.png)\n\n四seed样本SD阴影覆盖全部200轮。[2400条曲线](evidence/source_curves.csv)、[1600轮同步测量](evidence/source_energy_by_epoch.csv)、[全部720个源单元](evidence/source_all720_cells.csv)、[同seed源差分](evidence/source_paired_control.csv)。源RX是源验证，不称未知RX测试。\n\n'
    text+='## 因果包络输入的实际执行\n\n输入仍12项：`z[n-m]((P[n-m]+P[n-m-lag])/8)^q`，m0至3/q0至2。lag1或4使用因果补零，三阶项含滞后功率，五阶项含两时刻功率乘积。保留原四段位置功率读出、网络深宽和202553参数，同seed初始state与energy控制一致。新增的是lift线性空间中的交互，不证明旧全网无法近似它们。共同常相位不改变功率，新基函数保持charge1；不宣称任意received CFO或RX不变。\n\n'
    text+=f"全部1600轮源末batch和8个冻结公共输入均测量实际behavior.0.conv输入，共{len(inputs)}+{len(public_inputs)}条；三阶/五阶相对变化的分母是原aligned项L2范数clamp至1e-12。它是输入变化，不是恢复TX器件参数，也不参与候选排序。\n\n|模型|seed|实际lag|复数项数|公式最大误差|三阶相对变化均值|五阶相对变化均值|\n|---|---:|---:|---:|---:|---:|---:|\n"
    for a in inputs:
        if a['epoch']==200:text+=f"|{a['variant']}|{a['seed']}|{a['actual_envelope_lag']}|{a['complex_terms']}|{a['input_formula_max_abs_error']:.6g}|{a['degree3_relative_input_change_mean']:.6g}|{a['degree5_relative_input_change_mean']:.6g}|\n"
    text+='\n[全部1600条源输入测量](evidence/source_coupled_input_all1600.csv) · [8条公共输入测量](evidence/source_public_coupled_input_all8.csv)。40ns/160ns只是25MHz下的设计时间间隔，未测量真实器件记忆；received包络同时包含RX、channel和noise影响，不等于完整GMP、DPD或PA辨识。固定功率平均可能平滑有用瞬态，负结果保留。\n\n'
    text+='## 整体物理性质与边界\n\n六个复数时间/行为block使用跨通道/时间共享的包内RMS分母，归一化本身保留相对通道能量；学习尺度与径向门可以改变比例。两个候选均raw received输入、固定alpha0、不校正CFO。重复相关仅作received相对频率诊断，lag20/window80:160，主值±625kHz、歧义1.25MHz。不宣称整网频偏/RX/LTI不变、唯一TX参数或TX/RX因果分离。\n\n'
    text+=f"冻结公共链每模型5TX×6RX，检查公共相位及±80kHz仿射相位。全8权重最大公共相位logit误差{validation['whole_phase_max_logit_error']:.8g}，最大仿射logit误差{validation['whole_affine_max_logit_error']:.8g}；坐标逐点重建最大误差{validation['coordinate_reconstruction_max_error']:.8g}。常相位容差1e-3只报告，不是选模或停机门槛；附加频偏响应不称不变性误差。\n\n| 模型 | seed | 相位rad | 附加相对CFO Hz | 整网logit响应 | 单位嵌入距离 | 主值跨界数 | 有效相关数 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for a in phases:text+=f"| {a['variant']} | {a['seed']} | {a['theta_radians']} | {a['received_cfo_hz']} | {a['whole_logit_max_abs_error']:.8g} | {a['whole_unit_embedding_max_distance']:.8g} | {a['estimated_cfo_principal_branch_crossing_count']} | {a['valid_correlation_count']} |\n"
    text+='\n源V同步统计是27000包/90单元全部覆盖，估计均为received相对CFO；没有真频偏标签，不能把偏差称为晶振测量误差。\n\n| 模型 | seed | 相对CFO范围 Hz | 相干度均值 | fallback比例 | TX中心间平方距离 | 同TX跨RX中心偏移 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for a in geometry:
        rr=[c for c in cells if c['variant']==a['variant'] and c['seed']==a['seed']]
        text+=f"| {a['variant']} | {a['seed']} | {min(c['relative_cfo_hz_min'] for c in rr):.3f} 至 {max(c['relative_cfo_hz_max'] for c in rr):.3f} | {statistics.mean(c['coherence_mean'] for c in rr):.6f} | {statistics.mean(c['fallback_fraction'] for c in rr):.6g} | {a['between_TX']:.6g} | {a['within_TX_RX']:.6g} |\n"
    text+='\n## 实际共享归一化与固定校正\n\n[9600条源末batch测量](evidence/source_energy_normalization_all9600.csv)覆盖1600轮×6block，[48条冻结公共测量](evidence/source_public_normalization_all48.csv)覆盖8模型×6block。测量基于实际conv输出及共享分母，只比较归一化前后的相对通道能量；学习尺度和径向门不在该比值守恒声明内。测量误差仅报告，不参与选模。\n\n[1600轮固定校正执行](evidence/source_alignment_optimization.csv)与完整80000步核对；两版本均无学习校正参数，相关梯度N/A。\n\n| 模型 | seed | 最终alpha | 200轮校正梯度 | 残余频率公式最大误差Hz | 有效公式包数 |\n|---|---:|---:|---:|---:|---:|\n'
    for a in geometry:
        cc=[c for c in conditioner if c['variant']==a['variant'] and c['seed']==a['seed']];rr=[c for c in cells if c['variant']==a['variant'] and c['seed']==a['seed']]
        grads=[c['gradient_norm'] for c in cc if c['gradient_norm'] is not None];errors=[c['residual_formula_max_error_hz'] for c in rr if c['residual_formula_max_error_hz'] is not None]
        text+=f"| {a['variant']} | {a['seed']} | {cc[-1]['alignment_strength']:.8g} | {statistics.mean(grads) if grads else 'N/A'} | {max(errors) if errors else 'N/A'} | {sum(c['residual_formula_eligible_count'] for c in rr)} |\n"
    text+='\n[部分波形协变与整网响应](evidence/source_affine_phase_audit.csv)逐行列出有效包数、主值跨界、波形误差与logit响应，两种误差含义不同。物理干预采用冻结公共合成波形，不是训练增强，也不涉及真实query。\n'
    text+='\nTX/RX 镜像及三阶相同波形反例保留。LTI/RX影响不承诺消除，公共合成链不是精确WiSig均衡器、真实硬件采集或完整瞬态/噪声仿真。源嵌入几何只说明关联，不证明TX/RX因果分离。[全部240个公共组合](evidence/source_physics_cascade.csv)、[32个孤立TX变化](evidence/source_isolated_TX.csv)。\n\n## 实测成本\n\n| 模型 | 参数 | 常驻模型bytes | Conv/Linear MAC/包 | batch1推理ms均值 | batch128训练ms均值 | 源训练峰值bytes最大 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for v in selection['candidate_universe']:
        rr=[a for a in resources if a['variant']==v];a=rr[0]
        text+=f"| {v} | {a['parameters']} | {a['resident_state_bytes']} | {a['conv_linear_macs']} | {statistics.mean(c['inference_batch1_ms'] for c in rr):.4f} | {statistics.mean(c['training_batch128_ms'] for c in rr):.4f} | {max(c['source_peak_bytes'] for c in rr)} |\n"
    text+='\nRTX3090/Torch2.1/FP32；两个新候选和四份energy控制均采用显式完整FP32。三候选具有相同核心、深宽、原读出和202553参数。新候选只在行为输入引入固定滞后包络平均，两个候选均raw/alpha0；与energy控制差异是输入lift。MAC相同不表示总运算量相同。不增加学习参数；相对原residual_fusion164225参数的比较不是等参数单因子对照。MAC仅计Conv/Linear/矩阵乘，FFT/归一化/包络逐元素乘积/相关/atan2等不计入MAC，实测延时与显存包括全部执行。副本profile不更新正式模型。并发条件可能影响小幅计时差；星载/额外传输/SFT成本N/A。[逐seed完整资源](evidence/source_resources.csv)。\n\n'
    text+=('新候选选中，默认 clean 收尾待完成，不能宣称实验已全部完成。' if selection['new_candidate_selected'] else '当前源赢家选中，按预登记规则保留其已核实历史 clean；本轮两个未选新候选的测试记 N/A，没有新增 query。')+'不宣称识别优势或总体目标完成。历史clean代理口径、无LEO/SFT/新增类；D92三阶段N/A。目标成绩不反馈结构、超参数、重排或选择性重跑，负结果保留。\n\n[前瞻设计](../../../docs/CVS_COUPLED_ENVELOPE_HYPOTHESIS_20261002.md) · [全量日志审计](evidence/source_completion_validation.json) · [源选择](evidence/source_selection.json) · [独立分析](evidence/source_analysis_validation.json)。\n'
    (report/'report.md').write_text(text,encoding='utf-8');print(json.dumps(validation))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
