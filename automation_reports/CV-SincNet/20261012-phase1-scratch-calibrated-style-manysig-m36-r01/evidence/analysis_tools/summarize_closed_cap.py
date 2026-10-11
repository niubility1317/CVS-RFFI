"""Read every closed source cap record; no predictions, scores or model updates."""
import argparse
from collections import Counter
import json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from experiments.cvs_e20_boundary_style.archive_evidence import read_expanded

def summarize(records):
    count=len(records)
    if not count:return None
    groups=('shallow','middle','rest')
    scalar=[r['scalar'] for r in records]
    binding=Counter();violations=0
    for r in records:
        bound={g:.2*r['groups'][g]['norm_id']/(r['groups'][g]['raw_R']+1e-12) for g in groups if r['groups'][g]['raw_R']>0}
        expected=min([1.]+list(bound.values()))
        assert math.isclose(r['scalar'],expected,rel_tol=1e-10,abs_tol=1e-12)
        actual={g for g,v in bound.items() if v==expected and expected<1}
        assert actual==set(r['active_bound_groups'])
        binding.update(actual)
        for g in groups:
            v=r['groups'][g]
            violations+=int(v['capped_R']>v['norm_bound']+1e-7*max(1.,v['norm_bound']))
    retention={}
    for g in groups:
        raw=math.fsum(r['groups'][g]['raw_R'] for r in records)
        capped=math.fsum(r['groups'][g]['capped_R'] for r in records)
        ratios=[r['groups'][g]['capped_R']/r['groups'][g]['norm_id'] for r in records if r['groups'][g]['norm_id']>0]
        retention[g]=dict(sum_raw_norm=raw,sum_capped_norm=capped,norm_retention=capped/raw if raw else None,mean_capped_to_identity_ratio=statistics.mean(ratios) if ratios else None,max_capped_to_identity_ratio=max(ratios) if ratios else None)
    return dict(steps=count,clipped=sum(s<1 for s in scalar),scalar_below_point1=sum(s<.1 for s in scalar),scalar_below_point01=sum(s<.01 for s in scalar),zero_scalar=sum(s==0 for s in scalar),scalar_mean=statistics.mean(scalar),scalar_min=min(scalar),binding_group_counts={g:binding[g] for g in groups},bound_violations=violations,groups=retention,extra_gradient_seconds=math.fsum(r['seconds'] for r in records))

def build(source):
    assert source['source_only'] and not source['errors']
    rows=[]
    for row in source['rows']:
        assert row['steps']==4440 and len(row['epochs'])==20
        records=row['joint_cap_records']
        if not records:continue
        assert len(records)==999 and [r['step'] for r in records]==list(range(448,4441,4))
        full=summarize(records)
        assert full['bound_violations']==0
        epochs={str(e):summarize([r for r in records if r['epoch']==e]) for e in range(3,21)}
        assert sum(v['steps'] for v in epochs.values())==999
        rows.append(dict(row_id=row['row_id'],seed=row['seed'],full=full,epochs=epochs))
    return dict(status='VERIFIED',source_only=True,target_access=False,source_matrix_status=source['status'],closed_source_rows=len(source['rows']),closed_cap_rows=len(rows),cap_steps=sum(r['full']['steps'] for r in rows),rows=rows)

def report(v):
    lines=['# 完整训练中的联合限幅触发情况','',f"完整源训练{v['closed_source_rows']}行，其中限幅方案{v['closed_cap_rows']}行，共{v['cap_steps']}个辅助步骤。只读取闭合源日志；当前整批状态{v['source_matrix_status']}。",'','|方案／种子|限幅次数/999|s<0.1|s<0.01|s均值|最小值|零梯度缩放次数|','|---|---:|---:|---:|---:|---:|---:|']
    for row in v['rows']:
        s=row['full'];lines.append(f"|{row['row_id']}|{s['clipped']}/{s['steps']}|{s['scalar_below_point1']}|{s['scalar_below_point01']}|{s['scalar_mean']:.6f}|{s['scalar_min']:.6f}|{s['zero_scalar']}|")
    lines+=['','|方案／种子|浅层触发|中层触发|其余参数触发|浅层范数保留率|中层范数保留率|其余参数范数保留率|额外求导耗时（秒）|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in v['rows']:
        s=row['full'];b=s['binding_group_counts'];g=s['groups']
        pct=lambda name:'N/A' if g[name]['norm_retention'] is None else f"{g[name]['norm_retention']:.2%}"
        lines.append(f"|{row['row_id']}|{b['shallow']}|{b['middle']}|{b['rest']}|{pct('shallow')}|{pct('middle')}|{pct('rest')}|{s['extra_gradient_seconds']:.2f}|")
    lines+=['','范数保留率＝所有辅助步骤的限幅后梯度范数之和／限幅前范数之和，不能解释为最终权重位移保留率。触发次数按达到最小缩放界的参数组统计，并列时每组分别计数。无触发的步骤不归到任何组。每轮的完整统计保留于同名JSON。','','全参数三组的实际范数上限逐条通过。统计项仍经主身份骨干反向传播；仅凭缩放比例不能判定信息是否有效或辅助学习是否过弱。梯度方向、AdamW状态、与身份错误的关系仍影响更新，当前表不测量这些因果关系。','','CAP和DELAYCAP在各自固定方案下比较；现有完成种子不齐，不进行均值排名。DELAY无运行时限幅，不混入本表。额外求导耗时为实际共享GPU环境下记录，不能单独推算独占GPU成本。','','本分析不读取目标测试结果，不修改训练，不新增训练；继续完成固定36行矩阵。']
    if v['source_matrix_status']=='COMPLETE':
        lines=[line.replace('现有完成种子不齐，不进行均值排名。','本批全部三种子完成；此表保留逐种子机制统计，不作目标性能排名。').replace('继续完成固定36行矩阵。','固定36行矩阵源日志已全部核验完成。') for line in lines]
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v=build(read_expanded(a.source))
    (a.output/'evidence/closed_cap_utilization.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.output/'closed_cap_utilization.md').write_text(report(v),encoding='utf-8')
    print(json.dumps({k:x for k,x in v.items() if k!='rows'}))
