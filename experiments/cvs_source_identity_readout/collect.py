import argparse,json,statistics,csv
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
 p=base/row['row_id'];rows.append(dict(row=row,resolved=read(p/'resolved_config.json'),completion=read(p/'completion.json'),statistics={policy:read(p/(policy+'_statistics.json')) for policy in POLICIES},log=(base/'logs'/(row['row_id']+'.log')).read_text() if (base/'logs'/(row['row_id']+'.log')).exists() else None))
print(json.dumps(dict(completion=read(base/'completion.json'),failure=read(base/'failure.json'),rows=rows)))
'''

def verify(v):
    if v['failure'] or not v['completion'] or v['completion']['status']!='SOURCE_DIAGNOSTIC_COMPLETE':raise ValueError('Source readout matrix not complete')
    assert len(v['rows'])==3 and set(v['completion']['rows'])=={r['row_id'] for r in d.rows()}
    for item in v['rows']:
        c=d.config(item['row']);resolved=item['resolved'];done=item['completion']
        assert resolved['config']==done['config']==c and done['status']=='SOURCE_DIAGNOSTIC_COMPLETE' and not done['target_access'] and done['identity_updates']==done['head_updates']==done['U_forward']==done['V_forward']==0
        assert done['budget']['model_forwards']==100 and done['budget']['L_occurrences']==12600 and done['budget']['first_order_derivative_calls']==0 and done['budget']['parameters_buffers_gradients_RNG_modes_unchanged']
        assert 'SOURCE_DIAGNOSTIC_COMPLETE ' in item['log'] and not any(x in item['log'] for x in ('Traceback','RuntimeError','FloatingPointError'))
        for policy,stats in item['statistics'].items():
            assert stats['input_policy']==policy and stats['query_coverage_per_method']==6300 and stats['all_six_TX_uniform_rule'] and not stats['target_access']
            assert len(stats['geometry'])==3 and len(stats['crossfit'])==45 and len(stats['records'])==193
            for r in stats['crossfit']:assert r['rank']<=5 and r['denominator_min_eigenvalue']>0 and r['generalized_eigen_relative_residual']<1e-7 and r['fit_query_disjoint'] and r['held_RX_excluded_from_statistics_and_prototypes']
            for r in stats['geometry']:assert r['identity_mean_rank']<=5 and r['relative_closure_error']<1e-8
            for r in stats['records']:
                n=sum(sum(x) for x in r['confusion']);correct=sum(r['confusion'][k][k] for k in range(6))
                assert n==r['count'] and abs(correct/n-r['accuracy'])<1e-12
            assert all(r['count']==6300 for r in stats['records'] if r['scope'] in ('overall_crossfit','seen_L_in_sample'))
    return dict(v,status='VERIFIED',source_only=True,target_access=False)

def report(v,out):
    flat=[];geometry=[]
    for item in v['rows']:
        seed=item['row']['model_seed']
        for policy,stats in item['statistics'].items():
            flat.extend(dict(seed=seed,input_policy=policy,**r) for r in stats['records'] if r['scope'] in ('overall_crossfit','seen_L_in_sample'))
            geometry.extend(dict(seed=seed,input_policy=policy,**r) for r in stats['geometry'])
    d.write(out/'readout_results.json',dict(records=flat,geometry=geometry,source_only=True,target_access=False,unknown_RX_validation=False))
    with (out/'readout_results.csv').open('w',encoding='utf-8',newline='') as stream:
        rows=[{k:v for k,v in r.items() if k!='confusion'} for r in flat];w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 冻结 E20 实际表示的身份可读出性','', '**VERIFIED：三个固定 E20 源域诊断完成。** 身份／辅助头更新为零，不读取 U、V 前向或目标。每模型两种输入策略各6300条L／50前向，共300模型前向／37800样本模型读出；发布预检另24L／3前向。','',
        'historical_explicit_CFO 为历史原模型函数；no_explicit_CFO 屏蔽参考向量第27坐标，其他28个坐标、参数、缓冲、RNG保持不变，内部频偏补偿仍执行。后者是原权重上的功能诊断，不是重新训练的无CFO基线，也不代表原始IQ不含频偏。','',
        '每个读出使用L内3折×5RX；查询fold/RX不参加方向、中心和原型拟合，每PID每方法恰好读出一次。冻结骨干已经见过全部L和五源RX，本结果仅是统计读出迁移，不能称未知RX验证。','',
        '|输入策略|层／读出|2701|2702|2703|三种子均值|','|---|---|---:|---:|---:|---:|']
    for policy in d.config(d.rows()[0])['input_policies']:
        keys=list(dict.fromkeys((r['layer'],r['readout']) for r in flat if r['input_policy']==policy))
        for level,name in keys:
            values=[next(r['accuracy'] for r in flat if r['seed']==s and r['input_policy']==policy and r['layer']==level and r['readout']==name) for s in d.SEEDS]
            lines.append('|'+policy+'|'+level+'/'+name+'|'+'|'.join(f'{100*x:.4f}%' for x in values)+f'|{100*statistics.mean(values):.4f}%|')
    lines+=['','native_frozen_head 是已见L上的原身份头读出，不能与crossfit读出当作相同训练路径对照。raw/stable原型均只由fit L建立，所有TX统一竞争。RX_separate对四个RX各自的原型余弦分数作logmeanexp；merged平均RX/day单元后建立共同原型。固定tau=.1，不由成绩选择。','',
        '|输入策略／seed|层|TX占比|RX占比|交互占比|within占比|全L TX秩|闭合相对误差|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in geometry:
        f=r['energy_fractions'];fmt=lambda x:'N/A' if x is None else f'{x:.4%}'
        lines.append('|'+r['input_policy']+'/'+str(r['seed'])+'|'+r['layer']+'|'+'|'.join(fmt(f[k]) for k in ('TX','RX','interaction','within'))+f"|{r['identity_mean_rank']}|{r['relative_closure_error']:.3g}|")
    lines+=['','这里统计实际表示，不是虚拟作用敏感度。TX/RX均值按RX/day单元等权，within包含同TX/RX的day变化；全局6TX均值的判别秩≤5。分母使用有限总体PSD外积、固定alpha=beta=1、shrinkage=.1和相对eps1e-6。全L方向只作几何描述，绝不回用于crossfit查询评分。所有45个拟合中的特征值、分母最小特征值、方向分量能量、fit/query数量和残差保留原JSON。交互可能承载身份响应，不把存在交互等同污染，不据该比值删掉全部交互。','',
        '所有三种子、两输入、三层、四读出方法与全部fold/RX混淆矩阵、NLL、margin保留在complete_source_evidence.json。源诊断不会证明84%目标达成，也不据目标特定TX加权。后续训练需要同代码路径CE/SSL控制、原身份头与关系任务分别报告，并由固定后独立clean测试验证。']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();policies=d.config(d.rows()[0])['input_policies']
    payload=SCRIPT.replace('BASE',repr(d.BASE.as_posix())).replace('ROWS',repr(d.rows())).replace('POLICIES',repr(policies));v=verify(json.loads(ssh(payload)));a.output.mkdir(parents=True,exist_ok=True)
    d.write(a.output/'complete_source_evidence.json',v);report(v,a.output);print(json.dumps(dict(status=v['status'],models=3,source_only=True,target_access=False)))
