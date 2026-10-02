"""Complete source/physics/resource report for the eight curvature candidates."""
import argparse,json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cvs_phase_curvature_identity.collect import validate_completed
from experiments.cvs_phase_curvature_identity.prepare import RUN
from experiments.cvs_phase_curvature_identity.dispatch import CONTROL,CONTROL_RUN
from experiments.cvs_phase_curvature_identity.analyze_history import analyze as history,csvwrite,dump

def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence'
    data=json.loads((e/'source_research_complete.json').read_text(encoding='utf-8'));selection=validate_completed(data)
    old=json.loads((root/'automation_reports/CV-SincNet'/CONTROL_RUN/'evidence/source_research_complete.json').read_text(encoding='utf-8'))
    control=[r for r in old['rows'] if r['resolved']['variant']==CONTROL]
    tables={k:[] for k in ('source_curves','source_final','source_curvature_all9600','source_gates_all1600','source_all720_cells','source_resources','source_geometry','public_cascade_all240','public_phase_all24')}
    for r in [*control,*data['rows']]:
        v=r['resolved']['variant'];seed=r['resolved']['model_seed'];profile=r['profile'];last=r['epochs'][-1]
        for epoch in r['epochs']:
            tables['source_curves'].append(dict(variant=v,seed=seed,**{k:epoch[k] for k in ('epoch','clean_ce','source_val_ce','source_val_accuracy','source_val_worst_rx','gradient_norm','learning_rate','elapsed_seconds')}))
            if v==CONTROL:continue
            tables['source_gates_all1600'].append(dict(variant=v,seed=seed,epoch=epoch['epoch'],**{k:value for k,value in epoch.items() if k.startswith(('memory_','mix_'))}))
            for measurement in epoch['curvature_diagnostics']['feature_curvature']['records']:
                tables['source_curvature_all9600'].append(dict(variant=v,seed=seed,epoch=epoch['epoch'],**{k:value for k,value in measurement.items() if not isinstance(value,list)}))
        tables['source_final'].append(dict(variant=v,seed=seed,accuracy=last['source_val_accuracy'],worst_rx=last['source_val_worst_rx'],score=.5*(last['source_val_accuracy']+last['source_val_worst_rx']),train_ce=last['clean_ce'],val_ce=last['source_val_ce']))
        tables['source_resources'].append(dict(variant=v,seed=seed,**{k:profile[k] for k in ('total_parameters','trainable_parameters','resident_state_bytes','conv_linear_macs_per_sample','inference_batch1_ms','training_batch128_ms','hardware','torch_version')},actual_training_peak_bytes=max(q['peak_cuda_allocated_bytes'] for q in r['epochs'])))
        if v==CONTROL:continue
        diag=r['source_diagnostics'];phys=r['physical_diagnostics']
        for cell in diag['groups']:tables['source_all720_cells'].append(dict(variant=v,seed=seed,**{k:value for k,value in cell.items() if not isinstance(value,list)}))
        tables['source_geometry'].append(dict(variant=v,seed=seed,**{k:diag[k] for k in ('mean_between_tx_centroid_squared_distance','mean_within_tx_rx_centroid_squared_distance')}))
        for cell in phys['records']:tables['public_cascade_all240'].append(dict(variant=v,seed=seed,**{k:value for k,value in cell.items() if not isinstance(value,list)}))
        for cell in phys['phase_audit']:tables['public_phase_all24'].append(dict(variant=v,seed=seed,**cell))
    for key,records in tables.items():csvwrite(e/(key+'.csv'),records)
    assert {k:len(tables[k]) for k in ('source_curves','source_final','source_curvature_all9600','source_gates_all1600','source_all720_cells','public_cascade_all240','public_phase_all24')}==dict(source_curves=2400,source_final=12,source_curvature_all9600=9600,source_gates_all1600=1600,source_all720_cells=720,public_cascade_all240=240,public_phase_all24=24)
    summary=history(root,e/'history_analysis',include_current=True)
    validation=dict(status='VERIFIED',source_selection=selection,full_source_epochs=1600,full_source_steps=80000,table_counts={k:len(v) for k,v in tables.items()},target_access=False,mechanism_attribution='Pending separate frozen source V ablations')
    dump(e/'source_analysis_validation.json',validation)
    variants=list(selection['source_summaries']);fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    colors=['#334155','#0284c7','#d97706']
    for v,color in zip(variants,colors):
        records=[r for r in tables['source_curves'] if r['variant']==v]
        label=v.replace('adaptive_volterra_lag4','adaptive control').replace('phase_curvature_','curvature ')
        for ax,key,scale,title in [(axes[0,0],'source_val_accuracy',100,'Source V accuracy (%)'),(axes[0,1],'source_val_worst_rx',100,'Worst seen-source RX accuracy (%)'),(axes[1,0],'clean_ce',1,'Training CE (log scale)'),(axes[1,1],'source_val_ce',1,'Source V CE (log scale)')]:
            means=[st.mean(r[key] for r in records if r['epoch']==i)*scale for i in range(1,201)]
            ax.plot(range(1,201),means,color=color,label=label,lw=1.7);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.18)
            if scale==1:ax.set_yscale('log')
    axes[0,0].legend(fontsize=8);fig.savefig(e/'source_curves.png',dpi=180);fig.savefig(e/'source_curves.pdf');plt.close(fig)
    text='# CVS六层相位曲率残差：完整源实验结果\n\n'
    text+='8行均完成E200，每行10000步。独立重算固定12条源记录和源选择，完整80000步日志审计通过；未根据目标成绩改变排名。\n\n|方法|源V准确率|最差源RX|固定源分数|参数|\n|---|---:|---:|---:|---:|\n'
    for v,s in selection['source_summaries'].items():text+=f"|{v}|{100*s['source_accuracy']:.4f}%|{100*s['worst_rx_accuracy']:.4f}%|{100*s['score']:.4f}%|{s['parameters']}|\n"
    text+='\n源规则选中`'+selection['selected_variant']+'`。'+('条件clean测试尚待完成，只允许胜出候选4行新预测加32冻结对照。' if selection['new_candidate_selected'] else '保留原adaptive控制，两个未选候选的clean为N/A，复用原控制已经完成的clean结果。')+'以上是源验证指标，不是测试成绩。\n\n'
    text+='![完整源曲线](evidence/source_curves.png)\n\n'
    text+='[逐seed结果](evidence/source_final.csv)、[完整2400轮含控制曲线](evidence/source_curves.csv)、[9600个实际曲率输出](evidence/source_curvature_all9600.csv)、[1600轮八系数日志](evidence/source_gates_all1600.csv)、[全部720个TX×RX×day单元](evidence/source_all720_cells.csv)、[资源](evidence/source_resources.csv)、[公共级联240条](evidence/public_cascade_all240.csv)、[独立复算](evidence/source_analysis_validation.json)。全部负结果保留。\n\n'
    text+='训练继续使用原物理划分、scratch、CE唯一、无增强、身份骨干、完整FP32。曲率系数作用于接收特征，不能解释为TX硬件参数；局部仿射相位协变不等于整网CFO不变。每轮末28个训练样本的输出修正遥测不代表完整V分布；实际识别贡献由独立冻结V消融另行测量。\n\n'
    text+='参数比原控制增加6；Conv/Linear MAC未包含新增逐元素计算，耗时/显存按实测报告。跨seed只覆盖四次初始化随机性。源V来自同一组已见RX，最差源RX并非留一RX泛化验证。完整历史源分析见[evidence/history_analysis/source_history_analysis.json](evidence/history_analysis/source_history_analysis.json)，含48模型、9600轮，不读取目标评分；已有400000步历史原始日志审计引用保存证据，本轮另审计80000步。\n'
    (report/'report.md').write_text(text,encoding='utf-8');print(json.dumps(validation,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
