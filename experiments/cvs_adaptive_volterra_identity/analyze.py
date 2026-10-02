"""Full source curves/physics/geometry/resources; no target score inputs."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.cvs_adaptive_volterra_identity.prepare import RUN
from experiments.cvs_adaptive_volterra_identity.collect import validate_completed
from experiments.cvs_adaptive_volterra_identity.model import VARIANTS
CONTROL_RUN='20261002-phase1-cvs-coupled-identity-manysig-m8-r01'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def table(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence'
    data=read(e/'source_research_complete.json');selection=validate_completed(data)
    controls=[r for r in read(root/'automation_reports/CV-SincNet'/CONTROL_RUN/'evidence/source_research_complete.json')['rows'] if r['resolved']['variant']=='coupled_lag4']
    if len(controls)!=4:raise ValueError('Missing complete original source curves')
    for r in controls:
        actual=next(a for a in data['source_controls'] if a['seed']==r['resolved']['model_seed'])
        if any(actual[k]!=r['completion']['final_source_metrics'][v] for k,v in [('accuracy','source_val_accuracy'),('worst_rx','source_val_worst_rx')]):
            raise ValueError('Historical source curves differ from independently verified fixed source controls')
    gates=[];curves=[];final=[];rx=[];resources=[];sync=[];cells=[];geometry=[];cascade=[];tx=[];phases=[];paired=[];conditioner=[];normalizations=[];public_normalizations=[];inputs=[];public_inputs=[]
    for r in controls+data['rows']:
        c=r['resolved'];v=c['variant'];seed=c['model_seed'];epochs=r['epochs'];p=r['profile']
        if [a['epoch'] for a in epochs]!=list(range(1,201)):raise ValueError('Incomplete source curve')
        for a in epochs:
            curves.append(dict(variant=v,seed=seed,epoch=a['epoch'],V=a['source_val_accuracy'],worst_RX=a['source_val_worst_rx'],V_CE=a['source_val_ce'],train_CE=a['clean_ce'],gradient_norm=a['gradient_norm'],learning_rate=a['learning_rate'],elapsed_seconds=a['elapsed_seconds']))
            if v in VARIANTS:
                d=a['adaptive_diagnostics'];flat={k:value for k,value in d.items() if not isinstance(value,dict)}
                flat.update({'core_'+k:value for k,value in d.get('core',{}).items()})
                sync.append(dict(variant=v,seed=seed,epoch=a['epoch'],**flat))
                conditioner.append(dict(variant=v,seed=seed,epoch=a['epoch'],gradient_used_parameters=a['alignment_gradient_used_parameters'],gradient_norm=a['alignment_gradient_norm'],alignment_strength=a['alignment_strength'],whole_gradient_norm=a['gradient_norm']))
                normalizations.extend(dict(variant=v,seed=seed,epoch=a['epoch'],**record) for record in d['normalization']['records'])
                gates.append(dict(variant=v,seed=seed,epoch=a['epoch'],**{k:value for k,value in a.items() if k.startswith('mix_') or k=='mixture_gradient_used_parameters'}))
                inputs.extend(dict(variant=v,seed=seed,epoch=a['epoch'],mix_raw3=a['mix_raw3'],mix_raw5=a['mix_raw5'],mix_coefficient3=a['mix_coefficient3'],mix_coefficient5=a['mix_coefficient5'],**record) for record in d['adaptive_input']['records'])
        f=epochs[-1];best=max(epochs,key=lambda a:a['source_val_accuracy'])
        final.append(dict(variant=v,seed=seed,V=f['source_val_accuracy'],worst_RX=f['source_val_worst_rx'],CE=f['clean_ce'],best_V=best['source_val_accuracy'],best_epoch=best['epoch'],best_epoch_selected=False))
        rx.extend(dict(variant=v,seed=seed,receiver=k,accuracy=val) for k,val in f['source_val_rx_accuracy'].items())
        resources.append(dict(variant=v,seed=seed,parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],gradient_used_parameters=p['gradient_used_parameters'],resident_state_bytes=p['resident_state_bytes'],conv_linear_macs=p['conv_linear_macs_per_sample'],fft_calls=p['fft_calls_per_sample'],inference_batch1_ms=p['inference_batch1_ms'],inference_batch128_ms=p['inference_batch128_ms'],training_batch128_ms=p['training_batch128_ms'],source_peak_bytes=max(a['peak_cuda_allocated_bytes'] for a in epochs),profile_clone_peak_bytes=p['benchmark_peak_cuda_allocated_bytes'],hardware=p['hardware'],torch_version=p['torch_version'],elapsed_seconds=r['completion']['elapsed_seconds']))
        if v in VARIANTS:
            d=r['source_diagnostics'];ph=r['physical_diagnostics']
            public_normalizations.extend(dict(variant=v,seed=seed,**record) for record in ph['normalization']['records'])
            public_inputs.extend(dict(variant=v,seed=seed,mix_coefficient3=ph['adaptive_input']['coefficients'][0],mix_coefficient5=ph['adaptive_input']['coefficients'][1],**record) for record in ph['adaptive_input']['records'])
            geometry.append(dict(variant=v,seed=seed,between_TX=d['mean_between_tx_centroid_squared_distance'],within_TX_RX=d['mean_within_tx_rx_centroid_squared_distance']))
            cells.extend(dict(variant=v,seed=seed,**{k:val for k,val in a.items() if not isinstance(val,list)}) for a in d['groups'])
            cascade.extend(dict(variant=v,seed=seed,**{k:val for k,val in a.items() if not isinstance(val,list)}) for a in ph['records'])
            tx.extend(dict(variant=v,seed=seed,**a) for a in ph['isolated_tx_changes'])
            phases.extend(dict(variant=v,seed=seed,**a) for a in ph['phase_audit'])
            base=next(a for a in data['source_controls'] if a['seed']==seed)
            paired.append(dict(variant=v,seed=seed,V_delta_pp=100*(f['source_val_accuracy']-base['accuracy']),worst_RX_delta_pp=100*(f['source_val_worst_rx']-base['worst_rx'])))
    if len(curves)!=2400 or len(sync)!=1600 or len(conditioner)!=1600 or len(cells)!=720 or len(phases)!=24 or len(normalizations)!=9600 or len(public_normalizations)!=48:raise ValueError('Incomplete complete source report dimensions')
    table(e/'source_energy_normalization_all9600.csv',normalizations);table(e/'source_public_normalization_all48.csv',public_normalizations)
    if len(inputs)!=1600 or len(public_inputs)!=8 or len(gates)!=1600:raise ValueError('Incomplete actual Volterra input evidence')
    table(e/'source_gates_all1600.csv',gates);table(e/'source_adaptive_input_all1600.csv',inputs);table(e/'source_public_adaptive_input_all8.csv',public_inputs)
    for name,rows in dict(source_curves=curves,source_fixed_last_and_best=final,source_RX=rx,source_resources=resources,source_adaptive_by_epoch=sync,source_alignment_optimization=conditioner,source_all720_cells=cells,source_geometry=geometry,source_physics_cascade=cascade,source_isolated_TX=tx,source_affine_phase_audit=phases,source_paired_control=paired).items():table(e/(name+'.csv'),rows)
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
        measured_adaptive_input_records=len(inputs),public_adaptive_input_records=len(public_inputs),
        scope='Wholeconstantphase property;CE-trained two-gate complex phase-memory residual input;raw IQ retained;not uniqueTX or arbitraryRX/LTI/frequency invariance')
    write(e/'source_analysis_validation.json',validation)
    text='# CVS 可学习相位记忆残差：完整源实验报告\n\n'
    text+='8个新scratch模型完成E200×50；完整核对80000步、1600轮、逐步/epoch JSONL、详细stdout和紧凑JSONL/CSV。实际仅CE、无增强、无域骨干；原L6300/V27000、U56700不使用。两个全局门参数从零开始，总202555参数（控制202553，新增2），全部参与CE梯度；FP32/TF32关闭。控制仅源元数据，无权重继承。\n\n'
    text+='固定源规则选择 `'+selection['selected_variant']+'`。'+('新候选胜出，冻结后默认4份新clean预测；本报告仍没有新测试结论。' if selection['new_candidate_selected'] else '当前控制保留，未选新候选不访问query；复用原控制测试，两个未选模型测试N/A。')+'总体目标尚未证明。\n\n'
    text+='|候选|四seed源V（%）|最差源RX（%）|固定分数（%）|参数|\n|---|---:|---:|---:|---:|\n'
    for v,a in selection['source_summaries'].items():text+=f"|{v}|{100*a['source_accuracy']:.4f}|{100*a['worst_rx_accuracy']:.4f}|{100*a['score']:.4f}|{a['parameters']}|\n"
    text+='\n按四seed mean(0.5V+0.5最差RX)最高选择；完全并列才比V、最差RX、MAC、参数和固定顺序。所有模型仅E200权重参与，最好轮次只是诊断，不读目标成绩。\n\n|模型|seed|E200 V（%）|最差RX（%）|末轮CE|最好V（%，未选）|最好轮|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for a in final:text+=f"|{a['variant']}|{a['seed']}|{100*a['V']:.4f}|{100*a['worst_RX']:.4f}|{a['CE']:.6g}|{100*a['best_V']:.4f}|{a['best_epoch']}|\n"
    text+='\n![全部200轮源曲线](evidence/source_curves.png)\n\n2400条曲线含四控制及八新模型，阴影是四seed样本SD。全部720个新模型TX/RX/day单元完整覆盖V；源RX分层不是未知RX测试。[曲线](evidence/source_curves.csv) · [同seed控制差分](evidence/source_paired_control.csv) · [全部源单元](evidence/source_all720_cells.csv)。\n\n'
    text+='## 实际学习系数与卷积输入\n\n在t=n-m，p=(P[t]+P[t-4])/8、C=z[t-l]²conj(z[t-2l])/4；保留z，三阶为zp+tanh(raw3)(C-zp)，五阶为zp²+tanh(raw5)(Cp-zp²)。m0至3、l固定1或4；两个raw精确零初始化。零门时保持原控制输入和共享scratch输出；实测训练系数不称硬件PA参数。\n\n|模型|seed|末轮a3|末轮a5|门梯度范数均值|三阶相对输入变化|五阶相对输入变化|公式误差|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for a in gates:
        if a['epoch']!=200:continue
        record=next(r for r in inputs if r['variant']==a['variant'] and r['seed']==a['seed'] and r['epoch']==200)
        text+=f"|{a['variant']}|{a['seed']}|{a['mix_coefficient3']:.8g}|{a['mix_coefficient5']:.8g}|{a['mix_gradient_norm']:.8g}|{record['degree3_relative_input_change_mean']:.8g}|{record['degree5_relative_input_change_mean']:.8g}|{record['input_formula_max_abs_error']:.8g}|\n"
    text+='\n完整80000步审计两门梯度、前后raw/实际tanh系数及连续性；1600轮记录门状态/梯度均值，输入诊断来自当轮实际末batch28包；公共冻结输入每模型30包。相对变化分母是控制项L2范数clamp至1e-12，不是识别收益，也不参与选模。[1600轮门测量](evidence/source_gates_all1600.csv) · [1600条源输入](evidence/source_adaptive_input_all1600.csv) · [8条公共输入](evidence/source_public_adaptive_input_all8.csv)。\n\n'
    text+='## 数学、通信与RFF边界\n\n复三阶延迟平衡l+l-2l=0使输入lift对仿射相位协变；整网仅保留常相位性质，不声称CFO/RX/LTI不变。25MHz的lag1/4为40/160ns，三阶新增历史2l为80/320ns；这些是设计尺度，未测真实器件记忆。两个系数是全局模型参数，不按包/RX/目标拟合。raw received输入、固定alpha0；重复相关只测received相对频率，不恢复TX晶振。零门保留控制函数不证明整网动力学等距或更快收敛。\n\n'
    text+=f"冻结公共链5TX×6RX保留RX镜像/三阶同波形反例。八模型最大常相位logit误差{validation['whole_phase_max_logit_error']:.8g}，附加±80kHz最大logit响应{validation['whole_affine_max_logit_error']:.8g}；后者不是CFO不变性误差。逐点坐标重建最大误差{validation['coordinate_reconstruction_max_error']:.8g}。相位容差只报告，不选模或停机。\n\n"
    text+='源几何、同步和公共TX/RX干预是received表征证据；不同硬件链可能产生相同接收波形，不能认定唯一TX器件恢复、完整Volterra、DPD、因果TX/RX分离或真实在轨泛化。公共链不是精确WiSig均衡器，公共设置无实际源身份标签，合成身份准确率N/A。全部9600条实际源block归一化及48条公共测量覆盖共享分母；学习尺度/径向门仍可改变相对通道能量。[归一化](evidence/source_energy_normalization_all9600.csv) · [公共链](evidence/source_physics_cascade.csv) · [相位测量](evidence/source_affine_phase_audit.csv)。\n\n'
    text+='## 实际成本与未完成项\n\n|模型|参数|模型常驻bytes|Conv/Linear MAC/包|batch1推理ms均值|batch128训练ms均值|源峰值bytes最大|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for v in selection['candidate_universe']:
        rr=[a for a in resources if a['variant']==v];a=rr[0]
        text+=f"|{v}|{a['parameters']}|{a['resident_state_bytes']}|{a['conv_linear_macs']}|{statistics.mean(c['inference_batch1_ms'] for c in rr):.4f}|{statistics.mean(c['training_batch128_ms'] for c in rr):.4f}|{max(c['source_peak_bytes'] for c in rr)}|\n"
    text+='\nRTX3090/Torch2.1/完整FP32。新模型多2个学习参数；Conv/Linear形状相同不等于总运算量相同，复乘/tanh/残差/RMS/FFT不计入该MAC计数。实测时延与显存包含执行且受并发影响，不作独占成本结论；profile使用副本，不更新正式模型。星载、传输、support/SFT/新增类和D92三阶段N/A。\n\n'
    text+=('默认clean确认仍待完成；不得把源分数当作测试提升。' if selection['new_candidate_selected'] else '未选新候选clean N/A，按固定规则保留已完成控制测试；需核实原测试元数据完成本轮收尾。')+' 所有负差值保留，目标成绩不反馈结构/超参数/选模/重排/选择性重跑。总体性能/RFF目标仍未达到。\n\n[前瞻设计](../../../docs/CVS_ADAPTIVE_VOLTERRA_PHASE_MEMORY_20261002.md) · [预登记](experiment.json) · [全量日志审计](evidence/source_completion_validation.json) · [固定源选择](evidence/source_selection.json) · [独立分析](evidence/source_analysis_validation.json)。\n'
    (report/'report.md').write_text(text,encoding='utf-8');print(json.dumps(validation))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
