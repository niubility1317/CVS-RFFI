"""Source-only full-curve report; preserves fixed E200 selection."""
import argparse,csv,json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.cvs_response_fusion_identity.prepare import RUN
from experiments.cvs_response_fusion_identity.collect import validate_completed
from experiments.cvs_response_fusion_identity.dispatch import CONTROL_RUN,CONTROL,CHANNEL_CONTROL_RUN,CHANNEL_CONTROL,CANDIDATES,RESPONSE_CONTROL_RUN

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

def analyze(root):
    folder=root/'automation_reports/CV-SincNet'/RUN;e=folder/'evidence';data=read(e/'source_research_complete.json')
    selection=validate_completed(data)
    old=read(root/'automation_reports/CV-SincNet'/CONTROL_RUN/'evidence/source_research_complete.json')
    controls=[r for r in old['rows'] if r['resolved']['variant']==CONTROL]
    prior=read(root/'automation_reports/CV-SincNet'/CHANNEL_CONTROL_RUN/'evidence/source_research_complete.json')
    controls += [r for r in prior['rows'] if r['resolved']['variant']==CHANNEL_CONTROL]
    response_prior=read(root/'automation_reports/CV-SincNet'/RESPONSE_CONTROL_RUN/'evidence/source_research_complete.json')
    controls+=response_prior['rows']
    assert len(controls)==20
    curves=[];final=[];resources=[];neural=[];fusion=[];cells=[]
    for row in controls+data['rows']:
        variant=row['resolved']['variant'];seed=row['resolved']['model_seed'];base=dict(variant=variant,model_seed=seed)
        assert [x['epoch'] for x in row['epochs']]==list(range(1,201))
        for epoch in row['epochs']:
            curves.append(dict(base,epoch=epoch['epoch'],train_CE=epoch['clean_ce'],V_CE=epoch['source_val_ce'],V=epoch['source_val_accuracy'],worst_RX=epoch['source_val_worst_rx'],score=.5*(epoch['source_val_accuracy']+epoch['source_val_worst_rx']),gradient_norm=epoch['gradient_norm'],response_gradient_norm=epoch.get('response_gradient_norm')))
            if 'response_diagnostics' in epoch:
                neural.append(dict(base,epoch=epoch['epoch'],**epoch['response_diagnostics']['channel_response']))
            if 'response_fusion' in epoch.get('response_diagnostics',{}):
                fusion.append(dict(base,epoch=epoch['epoch'],**epoch['response_diagnostics']['response_fusion']))
        last=curves[-1];final.append(last)
        p=row['profile'];resources.append(dict(base,parameters=p['total_parameters'],trainable_parameters=p['trainable_parameters'],resident_bytes=p['resident_state_bytes'],conv_linear_MACs=p['conv_linear_macs_per_sample'],inference_batch1_ms=p['inference_batch1_ms'],training_batch128_ms=p['training_batch128_ms'],peak_training_bytes=max(x['peak_cuda_allocated_bytes'] for x in row['epochs']),hardware=p['hardware'],torch_version=p['torch_version']))
        for r in row['source_diagnostics']['groups']:cells.append(dict(base,**{k:v for k,v in r.items() if not isinstance(v,(list,dict))}))
    for name,rows in [('source_curves',curves),('source_final',final),('source_resources',resources),('response_outputs',neural),('fusion_outputs',fusion),('source_cells',cells)]:table(e/(name+'.csv'),rows)
    summary=[]
    for variant in CANDIDATES:
        rows=[r for r in final if r['variant']==variant];profiles=[r for r in resources if r['variant']==variant]
        differences=[100*(r['score']-next(c['score'] for c in final if c['variant']==CHANNEL_CONTROL and c['model_seed']==r['model_seed'])) for r in rows]
        summary.append(dict(variant=variant,V=st.mean(r['V'] for r in rows),worst_RX=st.mean(r['worst_RX'] for r in rows),score=st.mean(r['score'] for r in rows),paired_score_pp=st.mean(differences),paired_score_SD=st.stdev(differences),positive_seeds=sum(d>0 for d in differences),parameters=profiles[0]['parameters'],inference_ms=st.mean(r['inference_batch1_ms'] for r in profiles),training_ms=st.mean(r['training_batch128_ms'] for r in profiles)))
    table(e/'source_summary.csv',summary)
    figure,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for variant,color in zip(CANDIDATES,['#334155','#0284c7','#db7a13','#15803d','#7c3aed','#dc2626','#0891b2']):
        for ax,key,label in zip(axes.flat,['V','worst_RX','train_CE','V_CE'],['Source V accuracy (%)','Worst seen-source RX (%)','Training CE','Source V CE']):
            values=[st.mean(r[key] for r in curves if r['variant']==variant and r['epoch']==epoch) for epoch in range(1,201)]
            ax.plot(range(1,201),[x*100 if key in ('V','worst_RX') else x for x in values],label=variant,color=color,linewidth=1.5)
            ax.set(title=label,xlabel='Epoch');ax.grid(alpha=.15)
            if key.endswith('CE'):ax.set_yscale('log')
    axes[0,0].legend(fontsize=8)
    for ext in ('png','pdf'):figure.savefig(e/('source_curves.'+ext),dpi=180)
    plt.close(figure)
    text='# CVS约束响应融合：完整源训练结果\n\n8个scratch模型完成200轮×50步。源选择由完整28记录独立重算，训练策略和单一CE保持不变。以下是源验证指标，不是测试结果。\n\n'
    text+='|结构|源V/%|最差源RX/%|固定源分数/%|相对双路控制/百分点±配对SD|正提升seed|参数|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{100*r['V']:.4f}|{100*r['worst_RX']:.4f}|{100*r['score']:.4f}|{r['paired_score_pp']:+.4f}±{r['paired_score_SD']:.4f}|{r['positive_seeds']}/4|{r['parameters']}|\n"
    text+=f"\n固定规则选中`{selection['selected_variant']}`。"+('后续只对该候选4模型执行预登记clean测试。' if selection['new_candidate_selected'] else '保留既有控制，未选候选不访问query，复用已有控制clean完成证据。')+'\n\n![完整5600轮含控制曲线](evidence/source_curves.png)\n\n'
    text+='|结构|batch1推理/ms|batch128训练/ms|\n|---|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{r['inference_ms']:.4f}|{r['training_ms']:.4f}|\n"
    text+='\n资源在实际RTX3090/FP32环境测量，并行运行可能影响时间。MAC计数包含实际aten卷积（包括功能式复卷积）及矩阵乘法，未计6×6线性求解、FFT、归一化、门控、池化、动态FIR的unfold及逐元素乘加；不是总FLOPs。实际训练与推理耗时包含全部算子。峰值内存与常驻状态保留逐seed值。新模块同样只由CE更新；零出口的首步内部零梯度是初始化性质，不能与训练后失活混同。\n\n'
    text+='源V使用已见源RX；四seed不是独立数据集。新增容量是否提高独立clean识别必须由冻结测试证明。没有追加损失、增强、重加权、采样策略、teacher或目标适应。D92的适应三阶段/K×新增类为N/A。\n\n[逐seed](evidence/source_final.csv) · [完整曲线](evidence/source_curves.csv) · [实际新增分支输出](evidence/response_outputs.csv) · [全部TX/RX/day单元](evidence/source_cells.csv) · [资源](evidence/source_resources.csv) · [80000步审计](evidence/source_completion_validation.json) · [原设计](../../../docs/CVS_RESPONSE_FUSION_DESIGN_20261003.md)。\n'
    (folder/'report.md').write_text(text,encoding='utf-8')
    write(e/'source_analysis_validation.json',dict(status='VERIFIED',models=8,source_epochs=1600,control_epochs=4000,selection=selection,summary=summary,response_output_records=len(neural),target_results_read=False))
    print(json.dumps(dict(status='VERIFIED',summary=summary,selection=selection),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
