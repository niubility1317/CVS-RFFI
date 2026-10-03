"""Recompute geometry from saved source features and verify prior paired predictions."""
import argparse,csv,json,statistics as st
from pathlib import Path
from experiments.cvs_response_geometry.publish import inspect,ssh,RUN,PROJECT

REMOTE=r'''
from pathlib import Path
import json,numpy as np
exec(STATISTICS,globals())
root=Path(ROOT);state=json.loads((root/'pipeline_state.json').read_text())
if state['status']!='COMPLETE' or len(state['rows'])!=12:raise ValueError('Incomplete geometry matrix')
rows=[]
for rid,row in state['rows'].items():
    out=Path(row['output_root']);done=json.loads((out/'completion.json').read_text());resolved=json.loads((out/'resolved_config.json').read_text())
    if done['status']!='SOURCE_GEOMETRY_COMPLETE' or done['target_access'] is not False or done['optimizer_updates']!=0 or done['source_selection_changed'] is not False or done['model_unchanged'] is not True:raise ValueError('Invalid frozen source geometry')
    with np.load(out/'source_geometry.npz',allow_pickle=False) as value:q={k:value[k] for k in value.files}
    contract=json.loads((out/'source_loader/source_contract.json').read_text())
    if set(q['ids'].tolist())!=set(contract['role_ids']['V']):raise ValueError('Physical V mismatch')
    for name in ('all_on','auxiliary_off'):
        prior=root.parent/'20261003-diagnostic-cvs-response-attribution-source-manysig-m12-r01'/rid/(name+'_source_predictions.npz')
        with np.load(prior,allow_pickle=False) as old:
            if any(not np.array_equal(q[k],old[k]) for k in ('ids','truth','receiver')) or not np.array_equal(q[name].argmax(1),old['predictions']):raise ValueError('Prior paired source prediction differs')
    result=analyze_arrays(q)
    for name in ('all_on','auxiliary_off'):
        stored=next(r for r in done['rows'] if r['condition']==name)
        if result['classification'][name]['accuracy']!=stored['accuracy'] or {rx:r['accuracy'] for rx,r in result['classification'][name]['receiver'].items()}!=stored['rx_accuracy']:raise ValueError('Independent source prediction recount differs')
    if result['classification']['all_on']['accuracy']!=resolved['source_record']['accuracy']:raise ValueError('E200 source reproduction differs')
    rows.append(dict(row_id=rid,variant=done['variant'],seed=done['model_seed'],analysis=result))
if {(r['variant'],r['seed']) for r in rows}!={(v,s) for v in ('response_mean','response_attention','response_order_attention') for s in range(2026092701,2026092705)}:raise ValueError('Incomplete matrix')
print(json.dumps(dict(status='VERIFIED',models=12,source_packets_per_model=27000,decisions=648000,previous_predictions_exact=True,rows=rows,target_access=False,model_fitting=False),allow_nan=False))
'''


def collect(root,output):
    live=inspect(output)
    if not live['pipeline'] or live['pipeline']['status']!='COMPLETE':return False
    if live['dispatcher_process'] or len(live['rows'])!=12 or any(r['process'] or r['state']['status']!='COMPLETE' for r in live['rows']):raise ValueError('Not independently terminal')
    script=REMOTE.replace('STATISTICS',repr(Path(__file__).with_name('statistics.py').read_text(encoding='utf-8'))).replace('ROOT',repr(PROJECT+'/runs/'+RUN))
    wrapper='import subprocess\nsubprocess.run(["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-c",'+repr(script)+'],check=True)'
    data=json.loads(ssh(wrapper));e=root/'automation_reports/CV-SincNet'/RUN/'evidence';e.mkdir(parents=True,exist_ok=True)
    for name,value in [('final_readback.json',live),('geometry_analysis.json',data)]:
        (e/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    flat=[]
    for row in data['rows']:
        a=row['analysis'];r=dict(variant=row['variant'],seed=row['seed'])
        r.update({name:value['mean'] for name,value in a['geometry'].items()})
        for name in ('all_on','auxiliary_off'):
            r[name+'_accuracy']=a['classification'][name]['accuracy'];r[name+'_ce']=a['classification'][name]['ce']
            for part in ('both_correct','both_wrong','helped','hurt'):
                p=a['classification'][name]['partitions'][part];r[name+'_'+part+'_count']=p['count'];r[name+'_'+part+'_ce']=p['ce']['mean'];r[name+'_'+part+'_margin']=p['margin']['mean']
        for name,v in a['nested_scatter'].items():
            for factor,fraction in v['fractions'].items():r[name+'_'+factor+'_fraction']=fraction
        flat.append(r)
    with (e/'per_seed_geometry.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    summary=[]
    for variant in ('response_mean','response_attention','response_order_attention'):
        values=[r for r in flat if r['variant']==variant]
        summary.append(dict(variant=variant,**{k:st.mean(r[k] for r in values) if all(r[k] is not None for r in values) else None for k in flat[0] if k not in ('variant','seed')}))
    (e/'geometry_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text='# CVS补偿响应：完整源V判别几何\n\n状态VERIFIED。12个冻结模型，各27000条源V；主干与响应特征、补偿系数及分类输出均已保留。独立从特征与分类头复算logit，从保存的logit复算CE/间隔；648000个分类决定与上一份源诊断逐样本完全一致。未拟合模型或使用目标数据。\n\n'
    text+='|结构|响应/主干余弦均值|平行响应能量比例|单位特征变化|响应位于分类差异子空间的能量比例|全开CE|关闭CE|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{r['response_base_cosine']:.6f}|{r['parallel_response_energy_fraction']:.6f}|{r['unit_embedding_movement']:.6f}|{r['response_head_span_energy_fraction']:.6f}|{r['all_on_ce']:.6f}|{r['auxiliary_off_ce']:.6f}|\n"
    text+='\n分类差异子空间由归一化类别权重减去类别均值后定义，排除所有logit共同平移。几何比例是逐样本计算再求均值；它不是互信息、可恢复身份信息或新架构收益保证。\n\n|结构|表示|TX均值差异占比|同TX不同RX/day均值差异占比|cell内占比|\n|---|---|---:|---:|---:|\n'
    for r in summary:
        for name in ('base_unit','response_unit','joint_unit','compensation_coefficients'):
            values=[r[name+'_'+k+'_fraction'] for k in ('between_tx','within_tx_between_rx_day','within_cell')]
            text+='|'+r['variant']+'|'+name+'|'+'|'.join('N/A' if v is None else f'{v:.6f}' for v in values)+'|\n'
    text+='\n嵌套散度按TX→TX/RX/day→cell内分解，是给定源数据上的描述量。相同TX在不同RX/day的均值差异不能直接解释为纯信道因素，cell内也可能包含噪声、调制及信道变化。方向统计不能推出可辨识的TX/RX解耦。\n\n[逐seed、正确性分区与CE](evidence/per_seed_geometry.csv) · [完整RX/day单元、预测间隔与独立核对](evidence/geometry_analysis.json) · [四seed汇总](evidence/geometry_summary.json)。每包原始特征保留在N607，不作为新模型训练输入。\n'
    (e.parent/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',models=12,summary=summary),ensure_ascii=False));return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)
