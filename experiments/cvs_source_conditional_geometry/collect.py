import argparse,csv,json,math,statistics
from pathlib import Path
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh
SCRIPT=r'''
import json
from pathlib import Path
base=Path(BASE)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
rows=[]
for row in ROWS:
 p=base/row['row_id'];rows.append(dict(row=row,resolved=read(p/'resolved_config.json'),completion=read(p/'completion.json'),statistics=read(p/'statistics.json'),log=(base/'logs'/(row['row_id']+'.log')).read_text() if (base/'logs'/(row['row_id']+'.log')).exists() else None))
print(json.dumps(dict(completion=read(base/'completion.json'),failure=read(base/'failure.json'),rows=rows)))
'''

def verify(v):
    if v['failure'] or not v['completion'] or v['completion']['status']!='SOURCE_DIAGNOSTIC_COMPLETE':raise ValueError('Source diagnostic not complete')
    if len(v['rows'])!=3 or set(v['completion']['rows'])!={r['row_id'] for r in d.rows()}:raise ValueError('Incomplete matrix')
    for item in v['rows']:
        c=d.config(item['row']);resolved=item['resolved'];done=item['completion'];stats=item['statistics']
        if resolved['config']!=c or done['config']!=c or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or done['identity_updates'] or done['target_access']:raise ValueError('Metadata differs')
        budget=done['budget'];assert budget['parameters_buffers_gradients_RNG_modes_unchanged'] and budget['model_forwards']==50 and budget['L_occurrences']==6300 and budget['first_order_derivative_calls']==100
        assert len(budget['forward_batch_sizes'])==50 and sum(budget['forward_batch_sizes'])==6300
        assert stats['cell_count']==90 and stats['source_L_count']==6300 and len(stats['records'])==18 and stats['target_access'] is False
        expected={(stat,level,day) for stat in c['statistics'] for level in c['levels'] for day in c['days']}
        assert {(r['statistic'],r['layer'],r['day']) for r in stats['records']}==expected
        for r in stats['records']:
            assert r['source_packets']==2100 and r['coordinates']>0 and len(r['cell_metrics'])==30
            assert {(x['TX'],x['RX']) for x in r['cell_metrics']}=={(y,rx) for y in c['classes'] for rx in c['receivers']}
            assert all(x['n']==70 for x in r['cell_metrics'])
            assert math.isclose(sum(r['components'].values()),r['total_centered_energy'],rel_tol=1e-10,abs_tol=1e-12)
            assert len(r['RX_shuffled_controls'])==32 and [x['index'] for x in r['RX_shuffled_controls']]==list(range(32))
            assert all(x['seed']==c['diagnostic_seed']+r['day']*100003+x['index']*1597 for x in r['RX_shuffled_controls'])
        assert 'Traceback' not in item['log'] and 'SOURCE_DIAGNOSTIC_COMPLETE ' in item['log']
    return dict(v,status='VERIFIED',source_only=True,target_access=False,models=3,formal_model_forwards=150,formal_L_occurrences=18900,identity_updates=0)

def report(v,out):
    table=[]
    for item in v['rows']:
        records=item['statistics']['records'];seed=item['row']['model_seed']
        for stat in ('scaled_margin','scaled_ce','normalized_margin'):
            for level in ('time_down','t2.pool'):
                selected=[r for r in records if r['statistic']==stat and r['layer']==level];total=sum(r['total_centered_energy'] for r in selected)
                actual=statistics.mean(r['conditional_interaction_energy'] for r in selected)
                controls=[statistics.mean(r['RX_shuffled_controls'][i]['interaction_energy'] for r in selected) for i in range(32)]
                table.append(dict(seed=seed,statistic=stat,layer=level,interaction_fraction=sum(r['components']['interaction'] for r in selected)/total if total>0 else None,
                    identity_fraction=sum(r['components']['identity'] for r in selected)/total if total>0 else None,
                    receiver_fraction=sum(r['components']['receiver'] for r in selected)/total if total>0 else None,
                    within_fraction=sum(r['components']['within'] for r in selected)/total if total>0 else None,
                    interaction_energy=actual,shuffle_mean=statistics.mean(controls),shuffle_max=max(controls),shuffle_exceedances=sum(x>=actual for x in controls),
                    mixed_variance_penalty=statistics.mean(r['mixed_diagonal_variance_penalty'] for r in selected)))
    d.write(out/'summary.json',table)
    with (out/'summary.csv').open('w',encoding='utf-8',newline='') as stream:w=csv.DictWriter(stream,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# 原 E20 完整源域条件敏感度诊断','', '**VERIFIED：三个固定原始 E20 均完成。** 每模型完整6300条L、50前向，合计18900条L、150前向、300次一阶求导；另有发布预检24条L、3前向。身份／辅助头更新均为0。未访问目标、U或V前向。','',
        '|seed|统计|层|TX主效应占比|RX主效应占比|交互占比|单元内占比|交互能量|置乱均值／最大值|置乱≥实际次数／32|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"|{r['seed']}|{r['statistic']}|{r['layer']}|{r['identity_fraction']:.4%}|{r['receiver_fraction']:.4%}|{r['interaction_fraction']:.4%}|{r['within_fraction']:.4%}|{r['interaction_energy']:.7g}|{r['shuffle_mean']:.7g}／{r['shuffle_max']:.7g}|{r['shuffle_exceedances']}/32|")
    lines+=['','占比为三个day的对应能量之和除以总中心化能量之和。各day均2100样本、30单元，每单元70条。不是源/目标准确率，不把不同阶损失的数值大小当作等强度训练。','',
        '置乱仅在同TX/day内重新均衡划分RX，32组固定置乱跨统计／层保持相同，不额外执行模型。用于有限源池参照；未验证时间相关性等交换性假设，不作为总体显著性检验。','',
        'scaled_margin/scaled_ce使用完整L的每层固定RMS作为离线单位换算，normalized_margin使用原逐样本RMS。尺度不回填任何正在运行的训练；原E20已学习过这五个RX，本诊断不能充当未知RX验证。','',
        '这些结果仅检验是否存在当前混合方差统计未明确逐身份约束的接收机关联。关联不等于有害因素，去除关联也不保证识别收益。84%目标仍需固定训练后的完整三种子clean测试证明。','',
        '|seed|诊断秒数|峰值GPU内存 MiB|源L准确率（已见训练样本）|','|---|---:|---:|---:|']
    for item in v['rows']:
        z=item['completion'];lines.append(f"|{item['row']['model_seed']}|{z['seconds']:.3f}|{z['peak_cuda_memory_bytes']/1024**2:.2f}|{z['budget']['source_L_accuracy']:.4%}|")
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    payload=SCRIPT.replace('BASE',repr(d.BASE.as_posix())).replace('ROWS',repr(d.rows()));v=verify(json.loads(ssh(payload)));a.output.mkdir(parents=True,exist_ok=True)
    d.write(a.output/'complete_source_evidence.json',v);report(v,a.output);print(json.dumps(dict(status=v['status'],models=3,records=54,source_only=True,target_access=False)))
