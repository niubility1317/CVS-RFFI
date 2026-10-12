import argparse,csv,io,json,math,statistics
from pathlib import Path
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh

SCRIPT=r'''
import json
from pathlib import Path
c=CONFIG;base=Path(c['base'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
rows=[]
for row in c['rows']:
 p=base/row['row_id'];records=p/'records.jsonl';csv=p/'records.csv';log=base/'logs'/(row['row_id']+'.log')
 rows.append(dict(row=row,completion=read(p/'completion.json'),resolved=read(p/'resolved_config.json'),provenance=read(p/'provenance.json'),records=[json.loads(l) for l in records.read_text().splitlines() if l.strip()] if records.is_file() else [],csv=csv.read_text() if csv.is_file() else None,log=log.read_text() if log.is_file() else None))
print(json.dumps(dict(completion=read(base/'completion.json'),failure=read(base/'failure.json'),rows=rows)))
'''


def validate(v):
    if v['failure']:raise ValueError('Source diagnostic failed; preserve evidence')
    done=v['completion'];expected={r['row_id'] for r in d.rows()}
    if not done or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or set(done['rows'])!=expected:raise ValueError('Incomplete source matrix')
    if len(v['rows'])!=3 or {r['row']['row_id'] for r in v['rows']}!=expected:raise ValueError('Wrong workers')
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite evidence')
        if isinstance(x,dict):
            for val in x.values():finite(val)
        if isinstance(x,list):
            for val in x:finite(val)
    for row in v['rows']:
        c=d.config(row['row']);z=row['completion'];records=row['records']
        assert z and z['config']==c and z['status']=='SOURCE_DIAGNOSTIC_COMPLETE'
        assert z['models']==6 and z['batches']==len(records)==72 and z['temporary_optimizer_steps']==576
        assert z['persistent_identity_updates']==z['identity_updates']==z['new_checkpoints']==0
        assert not z['target_access'] and not z['U_truth'] and not z['V_forward'] and z['source_originals_unchanged']
        assert row['resolved']['config']==c and not row['resolved']['target_loader_constructed']
        assert {p['key'] for p in row['provenance']['models']}=={x['key'] for x in c['models']}
        assert {(r['model_key'],r['batch']) for r in records}=={(x['key'],i) for x in c['models'] for i in range(12)}
        compact=list(csv.DictReader(io.StringIO(row['csv'])));assert len(compact)==72*8*4
        assert row['log'] and 'Traceback' not in row['log']
        for record in records:
            assert record['original_parameters_buffers_grads_modes_unchanged'] and record['persistent_identity_updates']==0 and not record['target_access']
            assert len(record['main_L_ids'])==len(set(record['main_L_ids']))==128
            assert len(record['reference_L_ids'])==len(set(record['reference_L_ids']))==128
            assert not set(record['main_L_ids'])&set(record['reference_L_ids'])
            assert len(record['U_ids'])==len(set(record['U_ids']))==128
            assert len(record['cases'])==8 and {(x['mode'],x['objective']) for x in record['cases']}=={(m,o) for m in d.MODES for o in d.OBJECTIVES}
            assert record['model_forwards']==len(record['measured_forward_batch_sizes'])==12
            assert record['L_occurrences']==1144 and record['U_occurrences']==256
            assert sum(record['measured_forward_batch_sizes'])==1400
            for case in record['cases']:
                assert set(case['groups'])=={'shallow','middle','rest','all'}
                assert case['optimizer']['stored_moments_used']==(case['mode']=='saved_final')
                if case['objective']=='IDENTITY':
                    assert all(x['displacement_change_vs_identity']['norm_b']==0 for x in case['groups'].values())
        assert z['model_forwards']==864 and z['L_occurrences']==82368 and z['U_occurrences']==18432
        finite(records)
    return dict(v,status='VERIFIED',models=18,batches=216,temporary_optimizer_steps=1728,persistent_identity_updates=0,target_access=False)


def report(v,out):
    buckets={}
    for row in v['rows']:
        for record in row['records']:
            for case in record['cases']:
                for group,metrics in case['groups'].items():
                    key=(record['model_key'],record['model_seed'],case['mode'],case['objective'],group)
                    buckets.setdefault(key,[]).append((record,metrics))
    summary=[]
    for (model,seed,mode,objective,group),values in buckets.items():
        mean=lambda xs:statistics.mean([x for x in xs if x is not None]) if any(x is not None for x in xs) else None
        summary.append(dict(model=model,seed=seed,mode=mode,objective=objective,group=group,n=len(values),
            mean_cap=mean([r['cap']['scalar'] for r,_ in values]),
            gradient_cosine=mean([m['gradient_vs_identity']['cosine'] for _,m in values]),
            displacement_cosine=mean([m['displacement_vs_identity']['cosine'] for _,m in values]),
            displacement_change_ratio=mean([m['displacement_change_vs_identity']['norm_ratio_b_over_a'] for _,m in values]),
            relative_parameter_displacement=mean([m['displacement_vs_parameters']['norm_ratio_b_over_a'] for _,m in values]),
            source_reference_directional_derivative=mean([m['source_reference_directional_derivative'] for _,m in values])))
    with (out/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    lines=['# 浅中层 Fishr 与 MixStyle 的 AdamW 源诊断','',
        '完整 18 个固定 final20 模型、216 批、1,728 次临时 AdamW 调用已核对。持久身份更新为 0，无目标数据访问。下表为 FULL 相对 IDENTITY 的全模型位移，每行汇总固定三种子各 12 批。分层、逐种子、四目标的全部结果见 summary.csv。','',
        '|模型|优化器模式|平均梯度余弦|平均位移余弦|平均位移差/身份位移|平均相对参数位移|','|---|---|---:|---:|---:|---:|']
    for model in sorted({s['model'] for s in summary}):
        for mode in d.MODES:
            rows=[s for s in summary if s['model']==model and s['mode']==mode and s['objective']=='FULL' and s['group']=='all']
            fmt=lambda key:'N/A' if any(s[key] is None for s in rows) else f'{statistics.mean(s[key] for s in rows):.6g}'
            lines.append('|'+model+'|'+mode+'|'+'|'.join(fmt(k) for k in ('gradient_cosine','displacement_cosine','displacement_change_ratio','relative_parameter_displacement'))+'|')
    lines+=['','保存状态与重置状态的学习率也不同，跨模式差不能仅归因于动量。模式内部的 FULL 与 IDENTITY 才隔离 Fishr 项的影响。浅层组为 time_fuse，中层组为 t2；其余参数另报，不能把组名等同于完整网络深度分区。',
        '独立 L 参照与当前主 L 不重叠，但原训练已见过，并可能与辅助 A/B 重叠。梯度和位移的内积仅为一阶方向导数，没有执行更新后前向，不能称为实测误差下降或未知 RX 泛化。',
        '保存状态用于 final20 权重上的局部反事实；诊断位置是完全启用的目标，并非恢复历史训练步骤。负结果和零范数均保留。未生成新候选，不据此读取或挑选目标测试。',
        '正式诊断共 2,592 次模型前向、247,104 次 L 样本读取、55,296 次 U 样本读取。发布前真实源预检额外 30 次前向、1,180 次 L、256 次 U、8 次临时 AdamW；成本与正式诊断分列。']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    v=validate(json.loads(ssh(SCRIPT.replace('CONFIG',repr(dict(base=d.BASE.as_posix(),rows=d.rows()))))))
    d.write(a.output/'complete_source_evidence.json',v);report(v,a.output)
    print(json.dumps({k:v[k] for k in ('status','models','batches','temporary_optimizer_steps','persistent_identity_updates','target_access')}))
