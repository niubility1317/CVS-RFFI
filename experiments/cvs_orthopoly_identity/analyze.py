"""Full fixed source analysis; target predictions and scores are never inputs."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.cvs_orthopoly_identity.prepare import RUN
from experiments.cvs_orthopoly_identity.dispatch import CONTROL,CONTROL_RUN
from experiments.cvs_orthopoly_identity.collect import validate_completed,audit_input
from experiments.cvs_orthopoly_identity.model import VARIANTS

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def table(p,rows):
    if not rows:raise ValueError('Empty source evidence table')
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

def input_rows(lift,count,variant,seed,epoch=None):
    audit_input(lift,count)
    rec=lift['records'][0];rows=[]
    for basis in rec['records']:
        energies=basis['order_energy_mean']
        rows.append(dict(variant=variant,seed=seed,epoch=epoch,envelope_lag=rec['actual_envelope_lag'],
            input_formula_max_abs_error=rec['input_formula_max_abs_error'],input_abs_max=rec['input_abs_max'],
            **{k:v for k,v in basis.items() if k!='order_energy_mean'},
            order1_energy=energies[0],order3_energy=energies[1],order5_energy=energies[2]))
    return rows

def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence'
    data=read(e/'source_research_complete.json');selection=validate_completed(data)
    controls=[r for r in read(root/'automation_reports/CV-SincNet'/CONTROL_RUN/'evidence/source_research_complete.json')['rows'] if r['resolved']['variant']==CONTROL]
    if len(controls)!=4:raise ValueError('Missing four complete original source curves')
    for r in controls:
        actual=next(a for a in data['source_controls'] if a['seed']==r['resolved']['model_seed'])
        if any(actual[k]!=r['completion']['final_source_metrics'][v] for k,v in [('accuracy','source_val_accuracy'),('worst_rx','source_val_worst_rx')]):raise ValueError('Source control curves do not match independently read final metadata')
    tables={k:[] for k in ('source_curves','source_final','source_RX','source_resources','source_all720_cells','source_geometry','source_paired_control','source_input_all6400','public_input_all32','source_normalization_all9600','public_normalization_all48','source_phase_all24','public_cascade_all240')}
    for row in controls+data['rows']:
        c=row['resolved'];v=c['variant'];seed=c['model_seed'];ep=row['epochs'];p=row['profile']
        if [a['epoch'] for a in ep]!=list(range(1,201)):raise ValueError('Incomplete 200epoch source curve')
        for a in ep:
            tables['source_curves'].append(dict(variant=v,seed=seed,epoch=a['epoch'],V=a['source_val_accuracy'],worst_RX=a['source_val_worst_rx'],V_CE=a['source_val_ce'],train_CE=a['clean_ce'],gradient_norm=a['gradient_norm'],learning_rate=a['learning_rate'],elapsed_seconds=a['elapsed_seconds']))
            if v in VARIANTS:
                d=a['orthopoly_diagnostics']
                tables['source_input_all6400'].extend(input_rows(d['orthopoly_input'],28,v,seed,a['epoch']))
                tables['source_normalization_all9600'].extend(dict(variant=v,seed=seed,epoch=a['epoch'],**r) for r in d['normalization']['records'])
        f=ep[-1];best=max(ep,key=lambda a:a['source_val_accuracy'])
        tables['source_final'].append(dict(variant=v,seed=seed,V=f['source_val_accuracy'],worst_RX=f['source_val_worst_rx'],CE=f['clean_ce'],best_V=best['source_val_accuracy'],best_epoch=best['epoch'],best_epoch_selected=False))
        tables['source_RX'].extend(dict(variant=v,seed=seed,receiver=k,accuracy=a) for k,a in f['source_val_rx_accuracy'].items())
        tables['source_resources'].append(dict(variant=v,seed=seed,parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],gradient_used_parameters=p['gradient_used_parameters'],resident_state_bytes=p['resident_state_bytes'],conv_linear_macs=p['conv_linear_macs_per_sample'],fft_calls=p['fft_calls_per_sample'],inference_batch1_ms=p['inference_batch1_ms'],inference_batch128_ms=p['inference_batch128_ms'],training_batch128_ms=p['training_batch128_ms'],source_peak_bytes=max(a['peak_cuda_allocated_bytes'] for a in ep),profile_clone_peak_bytes=p['benchmark_peak_cuda_allocated_bytes'],hardware=p['hardware'],torch_version=p['torch_version'],elapsed_seconds=row['completion']['elapsed_seconds']))
        if v in VARIANTS:
            d=row['source_diagnostics'];ph=row['physical_diagnostics']
            tables['source_all720_cells'].extend(dict(variant=v,seed=seed,**{k:a for k,a in r.items() if not isinstance(a,list)}) for r in d['groups'])
            tables['source_geometry'].append(dict(variant=v,seed=seed,between_TX=d['mean_between_tx_centroid_squared_distance'],within_TX_RX=d['mean_within_tx_rx_centroid_squared_distance']))
            tables['public_input_all32'].extend(input_rows(ph['orthopoly_input'],30,v,seed))
            tables['public_normalization_all48'].extend(dict(variant=v,seed=seed,**r) for r in ph['normalization']['records'])
            tables['source_phase_all24'].extend(dict(variant=v,seed=seed,**r) for r in ph['phase_audit'])
            tables['public_cascade_all240'].extend(dict(variant=v,seed=seed,**{k:a for k,a in r.items() if not isinstance(a,list)}) for r in ph['records'])
            base=next(a for a in data['source_controls'] if a['seed']==seed)
            tables['source_paired_control'].append(dict(variant=v,seed=seed,V_delta_pp=100*(f['source_val_accuracy']-base['accuracy']),worst_RX_delta_pp=100*(f['source_val_worst_rx']-base['worst_rx'])))
    counts=dict(source_curves=2400,source_final=12,source_RX=60,source_resources=12,source_all720_cells=720,source_geometry=8,source_paired_control=8,source_input_all6400=6400,public_input_all32=32,source_normalization_all9600=9600,public_normalization_all48=48,source_phase_all24=24,public_cascade_all240=240)
    if any(len(tables[k])!=n for k,n in counts.items()):raise ValueError('Incomplete full source evidence dimensions')
    for name,rows in tables.items():table(e/(name+'.csv'),rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for v in selection['candidate_universe']:
        rr=[r for r in controls+data['rows'] if r['resolved']['variant']==v]
        for ax,k,label,factor in zip(axes,('source_val_accuracy','source_val_worst_rx','clean_ce'),('Source V accuracy (%)','Worst source RX accuracy (%)','Training CE'),(100,100,1)):
            a=np.asarray([[x[k] for x in r['epochs']] for r in rr])*factor;mean=a.mean(0);sd=a.std(0,ddof=1)
            ax.plot(range(1,201),mean,label=v);ax.fill_between(range(1,201),mean-sd,mean+sd,alpha=.12);ax.set_xlabel('Epoch');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);axes[2].set_yscale('log');fig.savefig(e/'source_curves.png',dpi=180);fig.savefig(e/'source_curves.pdf');plt.close(fig)
    inputs=tables['source_input_all6400'];phases=tables['source_phase_all24'];eligible=[r for r in inputs if r['eligible_packets']]
    validation=dict(status='VERIFIED',new_rows=8,new_epochs=1600,new_steps=80000,control_rows=4,table_counts=counts,source_selection_recomputed=True,
        selected_variant=selection['selected_variant'],new_candidate_selected=selection['new_candidate_selected'],target_access=False,target_scores_used=False,goal_achieved=False,
        raw_IQ_max_error=max(r['raw_order1_max_abs_error'] for r in inputs),original_terms_reconstruction_max_error=max(r['reconstruction_max_abs_error'] for r in inputs),
        orthogonal_gram_offdiag_max=max(r['orthogonal_normalized_gram_offdiag_max'] for r in eligible) if eligible else None,
        whole_constant_phase_max_logit_error=max(a['whole_logit_max_abs_error'] for a in phases if a['received_cfo_hz']==0.),
        whole_affine_phase_max_logit_response=max(a['whole_logit_max_abs_error'] for a in phases if a['received_cfo_hz']!=0.),
        numerical_measurements_scope='Last28 source packets per epoch/four same-lag order blocks; not fullV input Gram; full27000V90cells geometry per model',
        scope='Input conditioning and raw IQ/reconstruction; no TX/RX hardware separation or whole-CFO/RX/LTI invariance claim')
    write(e/'source_analysis_validation.json',validation)
    text='# CVS 包内正交包络输入：完整源实验报告\n\n'
    text+='8个新scratch模型全部完成E200×50；完整审计80000步、1600轮与详细文本/紧凑JSONL/CSV。仅CE、无增强、身份骨干、原L6300/V27000/U56700unused、完整FP32、无权重继承。四个当前adaptive源控制只读元数据与完整源曲线终值一致。\n\n'
    text+='固定源规则选择 `'+selection['selected_variant']+'`。'+('新候选胜出，冻结后默认4份新clean＋32份旧控制，36行独立truth-last评分；测试尚未完成。' if selection['new_candidate_selected'] else '当前源控制保留；未选新候选测试N/A、零新query，按预登记复用其既有已核实clean。')+'本报告不证明测试性能提升或总体目标完成。\n\n'
    text+='|候选|四seed源V（%）|最差源RX（%）|固定性能分数（%）|参数|\n|---|---:|---:|---:|---:|\n'
    for v,a in selection['source_summaries'].items():text+=f"|{v}|{100*a['source_accuracy']:.4f}|{100*a['worst_rx_accuracy']:.4f}|{100*a['score']:.4f}|{a['parameters']}|\n"
    text+='\n最高四seed mean(0.5V+0.5最差源RX)，完全并列后才比较V、最差RX与成本。全部选模固定E200；最佳轮仅诊断。\n\n|模型|seed|E200 V（%）|最差RX（%）|CE|最佳V（%）|最佳轮（未选）|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for a in tables['source_final']:text+=f"|{a['variant']}|{a['seed']}|{100*a['V']:.4f}|{100*a['worst_RX']:.4f}|{a['CE']:.6f}|{100*a['best_V']:.4f}|{a['best_epoch']}|\n"
    text+='\n![完整2400条源曲线](evidence/source_curves.png)\n\n'
    text+='## 实际输入与物理边界\n\n原IQ retained；三阶/五阶按同包加权矩去除相关项。6400个实际源末batch同延迟阶次记录覆盖1600epoch×4delay，公共32个记录覆盖8模型×4delay。Gram只在记录的非退化包上计算；不能说覆盖全部源V，也不证明跨delay或类别正交。\n\n'
    text+=f"原IQ最大误差{validation['raw_IQ_max_error']:.8g}，原三阶/五阶重建最大误差{validation['original_terms_reconstruction_max_error']:.8g}，符合记录条件的正交Gram最大残余{validation['orthogonal_gram_offdiag_max']}。误差只报告，不参与排名或停机。\n\n|模型|seed|delay|原相关均值|正交残余最大值|重建误差|有效包数|三阶能量|五阶能量|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for a in inputs:
        if a['epoch']==200:text+=f"|{a['variant']}|{a['seed']}|{a['delay']}|{a['original_normalized_gram_offdiag_mean']}|{a['orthogonal_normalized_gram_offdiag_max']}|{a['reconstruction_max_abs_error']:.6g}|{a['eligible_packets']}|{a['order3_energy']:.6g}|{a['order5_energy']:.6g}|\n"
    text+=f"\n全模型公共常相位最大logit误差{validation['whole_constant_phase_max_logit_error']:.8g}，±80kHz仿射相位最大logit响应{validation['whole_affine_phase_max_logit_response']:.8g}。后者是频偏响应，不是证明不变性的误差。原TX/RX同波形反例保留；输入正交不代表唯一TX硬件系数、TX/RX解耦、RX/LTI不变或真实在轨验证。全包矩不是流式因果运算，没有持久拟合状态。\n\n"
    text+='## 实际成本\n\n|模型|参数|常驻模型bytes|Conv/Linear MAC/包|batch1推理ms均值|batch128训练ms均值|源峰值bytes最大|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for v in selection['candidate_universe']:
        rr=[a for a in tables['source_resources'] if a['variant']==v];a=rr[0]
        text+=f"|{v}|{a['parameters']}|{a['resident_state_bytes']}|{a['conv_linear_macs']}|{statistics.mean(r['inference_batch1_ms'] for r in rr):.4f}|{statistics.mean(r['training_batch128_ms'] for r in rr):.4f}|{max(r['source_peak_bytes'] for r in rr)}|\n"
    text+='\nRTX3090/Torch2.1/完整FP32；统计口径见逐seed资源。MAC不包含包内矩/除法/逐元素运算，不等于总计算相同；实际延时及显存包含全执行，但并发非独占基准。无SFT/星载/新类，未测量CPU峰值与新增传输项为N/A。相对原residual_fusion不是等参数单因子比较。\n\n'
    text+=' · '.join(f"[{name}](evidence/{name}.csv)" for name in tables)
    text+='\n\n[前瞻结构推导](../../../docs/CVS_PACKET_ORTHOGONAL_ENVELOPE_20261002.md) · [全部日志审计](evidence/source_completion_validation.json) · [源冻结](evidence/source_selection.json) · [完整分析核对](evidence/source_analysis_validation.json)。只测clean；目标结果不反馈结构、lag、尺度、超参数、候选、seed或选择性重跑，保留全部负结果。\n'
    (report/'report.md').write_text(text,encoding='utf-8');print(json.dumps(validation))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
