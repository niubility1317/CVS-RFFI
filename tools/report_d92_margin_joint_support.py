"""Render verified Margin support results only; never fit or select a method."""
import argparse
import json
import math
import itertools
from pathlib import Path

PATHS=('R0','R_MARGIN_seq')
METRICS=('A_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy','C_h',
    'adaptation_gain_B_minus_A','support_adaptation_B_minus_B0','total_old_accuracy_drop',
    'C_abs_new_old_gap','C_old_columns_accuracy','new_competition_loss')


def require(value,message):
    if not value:raise ValueError(message)


def values(rows,**filters):
    chosen=[r for r in rows if r['metric'] in METRICS and all(r.get(k)==v for k,v in filters.items())]
    result={r['metric']:r for r in chosen}
    require(len(chosen)==len(result)==len(METRICS),'Missing/duplicate report metrics: '+repr(filters))
    for row in chosen:
        require(type(row['measured_parent_count']) is int and row['measured_parent_count']>=0,'Invalid measured parent count')
        require(row['mean'] is None or type(row['mean']) in (int,float) and math.isfinite(row['mean']),'Invalid mean')
        require((row['mean'] is None)==(row['measured_parent_count']==0),'Mean/count availability mismatch')
    return result


def percent(value):return 'N/A' if value is None else f'{100*value:.4f}'


def assemble(summary,execution,*,source):
    require(summary['status']=='COMPLETE_MARGIN_JOINT_PROBE_VERIFIED'
        and summary['summary_schema']=='d92_margin_joint_support_summary_v1'
        and summary['scope']=='SUPPORT_ONLY_MARGIN_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION'
        and summary['schema']==summary['algorithm']['schema']=='d92_margin_joint_local_ridge_v1'
        and summary['method']=='D92-MarginJointLocalRidge-v1','Unverified Margin summary')
    require(summary['query_rows_used']==summary['source_rows_used']==0 and summary['old_class_count']==6,'Input/registry mismatch')
    require(summary['algorithm']['proximal_coefficient']==0. and summary['algorithm']['coordinate_ball_radius']==.5,
        'Margin CE-only/hard-ball contract mismatch')
    limits=summary['qp_resources']
    require(isinstance(limits,dict) and set(limits)=={'max_transitions','max_factor_buffer_bytes'}
        and all(type(v) is int and v>0 for v in limits.values()),'Explicit QP resource contract missing')
    for peak in ('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes'):
        require(type(summary['coverage'].get(peak)) is int and summary['coverage'][peak]>=0,
            'Missing/invalid measured QP peak: '+peak)
    require(summary['coverage']['margin_qp_peak_factor_buffer_bytes']<=limits['max_factor_buffer_bytes'],
        'Measured QP factor-buffer peak exceeded declared resource scope')
    require(summary['actual_A'] is summary['adaptation_gain_B_minus_A'] is None,'Missing ground A cannot be fabricated')
    require(execution['status']=='VERIFIED' and execution['runtime_commit']==summary['release_commit'],'Analysis/runtime binding mismatch')
    coverage=summary['coverage'];parents=coverage['episodes']
    require(type(parents) is int and parents>0 and parents%20==0,'Incomplete declared K/new matrix')
    per_cell=parents//20
    require(coverage['k1_episodes']==5*per_cell and coverage['oof_episodes']==15*per_cell
        and coverage['sequence_paths']==225*per_cell,'Physical path coverage mismatch')
    stats=summary['statistics'];overall={};matrix=[]
    for path in PATHS:
        cell=values(stats['overall'],diagnostic='oof',population='new_present',path=path)
        require(cell['B_old_accuracy']['measured_parent_count']==12*per_cell,'Incomplete overall measured population')
        overall[path]={m:cell[m]['mean'] for m in METRICS}
        require(overall[path]['A_old_accuracy'] is overall[path]['adaptation_gain_B_minus_A'] is None,'Overall A fabricated')
    delta={m:None if overall['R0'][m] is None or overall['R_MARGIN_seq'][m] is None else
        overall['R_MARGIN_seq'][m]-overall['R0'][m] for m in METRICS}
    for diagnostic in ('oof','proxy'):
        for path in PATHS:
            for k in (1,5,10,20):
                for new in (0,2,5,10,20):
                    cell=values(stats['by_k_new_count'],diagnostic=diagnostic,path=path,k=k,new_count=new)
                    base=values(stats['by_k_new_count'],diagnostic=diagnostic,path='R0',k=k,new_count=new)
                    require(cell['B_old_accuracy']['measured_parent_count']==(0 if k==1 else per_cell),'Missing matrix parents')
                    require(cell['A_old_accuracy']['mean'] is cell['adaptation_gain_B_minus_A']['mean'] is None,'Cell A fabricated')
                    if k==1:require(all(v['mean'] is None for v in cell.values()),'True K1 held evidence fabricated')
                    if new==0:require(all(cell[m]['mean'] is None for m in ('C_new_accuracy','C_h','C_abs_new_old_gap')),'Absent new classes fabricated')
                    matrix.append(dict(diagnostic=diagnostic,path=path,k=k,old_count=6,new_count=new,registered_count=6+new,
                        measured_parents=cell['B_old_accuracy']['measured_parent_count'],
                        means={m:cell[m]['mean'] for m in METRICS},
                        deltas={m:None if cell[m]['mean'] is None or base[m]['mean'] is None else cell[m]['mean']-base[m]['mean'] for m in METRICS}))
    strata={}
    for table,dims in (('by_receiver_scene',('cohort','receiver','scenario')),('by_model_cohort',('model_seed','cohort'))):
        # Preserve every declared stratum/cell; do not select a best subgroup.
        rows=stats[table];stratum_keys={tuple(r[k] for k in dims) for r in rows}
        require(bool(stratum_keys),'Missing report strata')
        groups=[(d,p,k,n)+key for d,p,k,n,key in itertools.product(('oof','proxy'),PATHS,(1,5,10,20),(0,2,5,10,20),stratum_keys)]
        selected=[]
        for group in sorted(groups,key=repr):
            filters=dict(zip(('diagnostic','path','k','new_count')+dims,group));cell=values(rows,**filters)
            selected.append(dict(filters,means={m:cell[m]['mean'] for m in METRICS},
                measured_parents=cell['B_old_accuracy']['measured_parent_count']))
        for cell in matrix:
            require(sum(r['measured_parents'] for r in selected if all(r[k]==cell[k] for k in ('diagnostic','path','k','new_count')))
                ==cell['measured_parents'],'Incomplete measured report strata: '+table)
        strata[table]=selected
    lines=['# MarginJoint 完整 support 报告','',
        f'run：`{summary["run_id"]}`；已核验 {parents} 个 parent、{coverage["sequence_paths"]} 条物理路径。',
        '本报告只描述合法 support 的持出诊断，不代表 query 准确率或新增独立数据验证。',
        'A 与 B−A 为 N/A：实际地面头尚未接入。R0/B0 是 support 分类器，不能代替 A。',
        'B/C 按同一 row、同一旧 support 与物理旧 held 样本配对；C 对全部注册类统一竞争并继承当次实际 B。',
        'H、注册下降和绝对新旧差先逐 parent 计算，再等 parent 汇总；proxy 先对全部 anchor 求 parent 内均值。',
        'true K1 只有全 support 解析头，无独立 held；其 OOF/proxy 指标单列 N/A。proxy 的 train K1 不等于真实 K1 独立证据。',
        '理想方向为适应提升至少 10 个百分点、注册旧类下降至多 1 个百分点、新旧绝对差至多 3 个百分点；均为 soft 目标，不触发晋级、选模或重跑。',
        '', '| 路径 | A旧 | B旧 | C旧 | C新 | H | B−A | B−B0 | 注册下降 | 平均绝对差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for path in PATHS:lines.append('| '+' | '.join([path]+[percent(overall[path][m]) for m in METRICS[:9]])+' |')
    lines+=['','准确率/H 用 %，差值用百分点。整体仅含 K5/10/20、新增2/5/10/20 的可测 parent。',
        '注册下降变小需要同时检查 B 与 C，不能把 B 下降导致的差值缩小解释为保旧改善。',
        '', '| 相对 R0 的变化 | '+' | '.join(METRICS[1:9])+' |','|---|'+'---:|'*8,
        '| R_MARGIN_seq | '+' | '.join(percent(delta[m]) for m in METRICS[1:9])+' |','']
    def table(title,rows,dimension_keys):
        lines.extend([title,'','| '+' | '.join(dimension_keys)+' | A旧 | B旧 | C旧 | C新 | H | B−A | B−B0 | 注册下降 | 绝对差 |',
            '|'+'---|'*len(dimension_keys)+'---:|'*9])
        for row in rows:lines.append('| '+' | '.join([str(row[k]) for k in dimension_keys]+[percent(row['means'][m]) for m in METRICS[:9]])+' |')
        lines.append('')
    table('完整 K×新增类数矩阵',matrix,('diagnostic','path','k','old_count','new_count','registered_count','measured_parents'))
    for name,rows in strata.items():table(name,rows,tuple(k for k in rows[0] if k!='means'))
    lines += ['模型与资源口径','',
        '目标仅为 pooled class-RMS CE。分类梯度为归档的 g_Z，不包含 +Z 近端梯度；坐标受 0.5 硬球约束，Armijo 使用投影后的实际 delta。',
        'C 使用原始 Gaussian 核和带自由截距的凸 margin QP；A 与实际工作集因子的尝试/完成、完整约束扫描、谱诊断、RHS 和缓存分别按实际执行记录，不能套用固定因子数。',
        'KKT 伴随只在工作集独立且严格互补的可微范围核对；非光滑或技术失败保留实际状态，不静默退回无约束头。完整 K/L 两端梯度传入 adapter。',
        '全部物理旧 support 对全部注册类的间隔约束只作用于训练 support。正原间隔可保持其严格正确；负原间隔不保证每对赢家不变，也不保证未来 query 保旧。远处自由截距 b 仍影响完整分数。',
        '两个 QP 峰值跨阶段/行取 MAX，其余实际执行计数取 SUM；factor guard计live scratch输入与Cholesky输出，不统一覆盖返回状态冻结时的布局/bytes瞬时复制、QP构建临时数组、BLAS workspace、归档或整个进程RSS。',
        '以下保留所有实测资源字段；n²×RHS、数组字节和参数计数为数值代理，不是实测 FLOPs、设备峰值或完整部署包。未测项为 N/A；嵌套累计计时不可相加当墙钟。',
        '', '| 资源字段 | 原值 |','|---|---|']
    lines += ['| qp_resources | '+json.dumps(limits,ensure_ascii=False,allow_nan=False)+' |']
    for key,value in summary['resources'].items():lines.append('| '+key+' | '+('N/A' if value is None else json.dumps(value,ensure_ascii=False,allow_nan=False))+' |')
    lines+=['','| 实际执行计数 | 数值 |','|---|---:|']
    for key,value in coverage.items():lines.append('| '+key+' | '+str(value)+' |')
    lines+=['',f'独立分析来源：`{source}`；训练 commit：`{summary["release_commit"]}`；分析 commit：`{execution["commit"]}`。','']
    interpretation=dict(status='COMPLETE_MARGIN_SUPPORT_REPORT_RENDERED_DECISION_PENDING',run_id=summary['run_id'],
        scope=summary['scope'],declared_main_path='R_MARGIN_seq',automatic_promotion=False,actual_A=None,
        overall=overall,delta_vs_R0=delta,matrix=matrix,strata=strata,resources=summary['resources'],source=source,
        runtime_commit=execution['runtime_commit'],analysis_commit=execution['commit'],qp_resources=limits)
    return '\n'.join(lines),interpretation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('summary','execution','report','interpretation'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();require(not args.report.exists() and not args.interpretation.exists(),'Exclusive report outputs required')
    report,value=assemble(json.loads(args.summary.read_text(encoding='utf-8')),json.loads(args.execution.read_text(encoding='utf-8')),source=str(args.summary))
    for path in (args.report,args.interpretation):path.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x',encoding='utf-8') as stream:stream.write(report)
    with args.interpretation.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)


if __name__=='__main__':main()
