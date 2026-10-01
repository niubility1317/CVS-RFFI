"""Render complete frozen three-stage query scores; no truth/model/history access."""
import argparse
from copy import deepcopy
import itertools
import math
from pathlib import Path

from score_d92_margin_joint_benchmark import (METHOD,SCHEMA,STATUS,SCOPE,KS,NEWS,METRICS,
    require,read,write,_statistics,aggregate_training_resources)

REPORT_SCHEMA='d92_margin_joint_query_report_v1'
REPORT_STATUS='MARGIN_JOINT_QUERY_REPORT_COMPLETE'
DELTAS=frozenset(('adaptation_gain_B_minus_A','total_old_accuracy_drop','C_abs_new_old_gap'))


def _close(a,b,message):
    require((a is None and b is None) or type(a) in (int,float) and type(b) in (int,float)
        and not isinstance(a,bool) and not isinstance(b,bool) and math.isfinite(a) and math.isfinite(b)
        and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),message)


def validate_score(score,current_metadata=None,*,allow_baseline=False):
    require(score['schema']==SCHEMA and score['status']==STATUS and score['scope']==SCOPE
        and (allow_baseline or score['method']==METHOD),'Only a complete fixed three-stage query score is accepted')
    require(score['prediction_validation_complete_before_truth'] is True and score['prediction_streams_independently_reread'] is True
        and score['selection_feedback_forbidden'] is True
        and all(score[k] is False for k in ('query_fit_access','source_fit_access','training_performed','model_called')),
        'Score did not preserve the truth-last boundary')
    metadata=score['current_metadata']
    if current_metadata is not None:require(metadata==current_metadata,'Explicit current metadata mismatch')
    spec=metadata['spec'];declared={r['row_id']:r for r in spec['rows']}
    require(score['run_id']==spec['run_id'] and score['group_id']==spec['group_id'] and score['row_count']==len(declared)
        and score['algorithm']==spec['benchmark']['config']['algorithm'] and score['qp_resources']==spec['benchmark']['config']['qp_resources'],
        'Score/spec identity or explicit QP resource mismatch')
    complete=metadata['supervisor_complete'];startup=metadata['supervisor_startup']
    require(complete['status']=='MARGIN_QUERY_BENCHMARK_PREDICTIONS_COMPLETE'
        and complete['all_predictions_fixed'] is True and complete['row_count']==complete['completed_row_count']==len(declared)
        and complete['resolved_spec']==startup['resolved_spec']==spec
        and complete['runtime_commit']==startup['runtime_commit']==score['release_commit']
        and complete['code_commit']==startup['code_commit']==score['code_commit']==spec['code']['commit'],
        'Current runtime completion binding mismatch')
    parents=score['parents'];require(len(parents)==score['parent_count'] and bool(parents),'Missing full physical parent scores')
    cells={};groups={};seen=set()
    for p in parents:
        rid=p['row_id'];require(rid in declared,'Score escaped declared model row')
        row=declared[rid];co=spec['benchmark']['cohorts'][row['cohort']]
        require(p['run_id']==score['run_id'] and p['release_commit']==score['release_commit']
            and p['model_seed']==row['expected_model_seed'] and p['checkpoint_sha256']==row['expected_checkpoint_sha256']
            and p['capsule_id']==co['expected_capsule_id'] and p['cohort']==row['cohort'],'Physical source pairing mismatch')
        cell=(rid,p['receiver'],p['scenario'],p['support_seed'],p['k'],p['new_count'])
        require(cell not in seen and p['k'] in KS and p['new_count'] in NEWS,'Duplicate or unexpected parent cell');seen.add(cell)
        require(p['old_class_count']==6 and p['registered_class_count']==6+p['new_count']
            and p['old_query_count']==len(p['old_query_ids'])>0 and p['new_query_count']==len(p['new_query_ids'])
            and p['query_count']==p['old_query_count']+p['new_query_count']
            and len(set(p['old_query_ids']+p['new_query_ids']))==p['query_count'],'Invalid old/new physical query coverage')
        require((p['new_query_count']==0)==(p['new_count']==0),'new0 query boundary mismatch')
        groups.setdefault(rid,set()).add((p['receiver'],p['scenario'],p['support_seed']))
        cells.setdefault(rid,set()).add(cell[1:])
        m=p['metrics'];require(set(m)==set(METRICS),'Missing or unexpected paired metric')
        for k in ('A_old_accuracy','B_old_accuracy','C_old_accuracy'):
            require(type(m[k]) in (int,float) and not isinstance(m[k],bool) and math.isfinite(m[k]) and 0<=m[k]<=1,
                'All query K levels require measured A/B/C old accuracy')
        a,b,old,new=(m[k] for k in ('A_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy'))
        if p['new_count']:
            require(type(new) in (int,float) and not isinstance(new,bool) and math.isfinite(new) and 0<=new<=1,'Missing new query accuracy')
        else:require(new is m['C_h'] is m['C_abs_new_old_gap'] is None,'new0 new/H/gap must remain N/A')
        _close(m['adaptation_gain_B_minus_A'],b-a,'B−A must be paired within parent')
        _close(m['total_old_accuracy_drop'],b-old,'B−C old must be paired within parent')
        _close(m['C_h'],None if new is None else (0. if old+new==0 else 2*old*new/(old+new)),'H must be computed within parent')
        _close(m['C_abs_new_old_gap'],None if new is None else abs(old-new),'Absolute gap must be computed within parent')
    require(set(cells)==set(declared),'Missing declared model row')
    for rid,pairs in groups.items():
        require(cells[rid]=={(rx,scene,seed,k,n) for rx,scene,seed in pairs for k in KS for n in NEWS},'Incomplete K/new matrix')
    require(score['statistics']==_statistics(parents),'Reported tables differ from complete parent-first statistics')
    resources=aggregate_training_resources(metadata['rows'])
    require(all(score['resources'].get(k)==v for k,v in resources.items()),'Actual SUM/MAX resource aggregate mismatch')
    return score


def _cells(rows,dimensions,population='all'):
    result={}
    for row in rows:
        if row['population']!=population:continue
        key=tuple(row[d] for d in dimensions)
        group=result.setdefault(key,{})
        require(row['metric'] not in group,'Duplicate table metric');group[row['metric']]=row
    cells=[]
    for key,group in sorted(result.items(),key=repr):
        require(set(group)==set(METRICS),'Incomplete table metric set')
        cells.append(dict(zip(dimensions,key),parent_count=group['A_old_accuracy']['parent_count'],
            measured_parent_counts={m:group[m]['measured_parent_count'] for m in METRICS},values={m:group[m]['mean'] for m in METRICS}))
    return cells


def _format(value):return 'N/A' if value is None else f'{100*value:.4f}'


def _baseline_pairing(current,baseline):
    validate_score(baseline,allow_baseline=True)
    def key(p):return tuple(p[k] for k in ('model_seed','checkpoint_sha256','capsule_id','receiver','scenario','support_seed','k','new_count'))
    lookup={key(p):p for p in baseline['parents']}
    require(len(lookup)==len(baseline['parents']) and set(lookup)=={key(p) for p in current['parents']},'Explicit baseline has unmatched or duplicate physical parents')
    rows=[]
    for p in current['parents']:
        old=lookup[key(p)]
        require(sorted(p['old_query_ids'])==sorted(old['old_query_ids']) and sorted(p['new_query_ids'])==sorted(old['new_query_ids'])
            and p['old_classes']==old['old_classes'] and p['registered_classes']==old['registered_classes'],
            'Explicit baseline uses different physical queries or registered classes')
        rows.append(dict(**{k:p[k] for k in ('row_id','model_seed','receiver','scenario','support_seed','k','new_count')},
            baseline_method=baseline['method'],current_metrics=deepcopy(p['metrics']),baseline_metrics=deepcopy(old['metrics']),
            differences={m:None if p['metrics'][m] is None or old['metrics'][m] is None else p['metrics'][m]-old['metrics'][m] for m in METRICS}))
    return rows


def _legacy_baseline_pairing(current,baseline_scores,baseline_metadata):
    """Bind all legacy identities before reading any comparison metric."""
    scores=baseline_scores if isinstance(baseline_scores,list) else [baseline_scores]
    metadata=baseline_metadata if isinstance(baseline_metadata,list) else [baseline_metadata]
    require(len(scores)==len(metadata) and all(isinstance(v,dict) for v in metadata),'Explicit equal-length legacy baseline metadata required')
    current_rows={r['row_id']:r for r in current['current_metadata']['spec']['rows']}
    lookup={}
    keys=('model_seed','receiver','scenario','k','new_count','support_seed')
    for scored,meta in zip(scores,metadata):
        require(set(meta)=={'method','spec','startup','complete'},'Explicit legacy method/spec/startup/complete required')
        spec,startup,done,method=(meta[k] for k in ('spec','startup','complete','method'))
        require(scored['status']=='SCORED' and scored['selection_feedback_forbidden'] is True
            and done['status']=='SCORED' and done['selection_feedback_forbidden'] is True
            and startup['spec']==spec and startup['commit']==done['commit']
            and done['model_rows']==len(spec['rows']) and done['records']==len(scored['results']),
            'Legacy baseline did not complete under its actual runtime/spec')
        require(method==spec['confirmation']['candidate_method'] or method=='D92','Explicit baseline method differs from registered candidate')
        data=spec['data'];capsule=data['capsule_id']
        require(spec['confirmation']['reuse_validated_capsule_id']==capsule,'Legacy VALIDATED_ONCE capsule binding mismatch')
        model_rows={r['seeds']['model']:r for r in spec['rows']}
        require(len(model_rows)==len(spec['rows']),'Legacy model row mapping is ambiguous')
        expected=set(itertools.product(model_rows,data['target_receivers'],data['scenarios'],data['k'],data['new_class_counts'],data['support_seeds']))
        selected=[r for r in scored['results'] if r['method']==method]
        actual={tuple(r[k] for k in keys) for r in selected}
        require(len(selected)==len(actual)==len(expected) and actual==expected,'Legacy baseline method matrix is incomplete')
        for record in selected:
            source=model_rows[record['model_seed']]
            require(record['row_id']==source['row_id'] and record['class_count']==len(record['classes'])==6+record['new_count']
                and type(record['query_count']) is int and record['query_count']>0
                and len(set(record['classes']))==len(record['classes']),'Legacy record row/class/count identity mismatch')
            checkpoint=source['expected_checkpoint_sha256']
            require(record.get('checkpoint_sha256',checkpoint)==checkpoint and record.get('capsule_id',capsule)==capsule,
                'Legacy optional checkpoint/capsule identity mismatch')
            key=(capsule,checkpoint)+tuple(record[k] for k in keys)
            require(key not in lookup,'Duplicate legacy baseline physical parent')
            lookup[key]=(record,source,meta)
    paired=[]
    for p in current['parents']:
        key=(p['capsule_id'],p['checkpoint_sha256'])+tuple(p[k] for k in keys)
        require(key in lookup,'Current physical parent missing from explicit legacy baseline')
        old,source,meta=lookup[key]
        require(old['split_id']==p['split_id'] and set(old['classes'])==set(p['registered_classes'])
            and set(old['classes'][:6])==set(p['old_classes']) and old['query_count']==p['query_count'],
            'Legacy baseline split/class/query-count pairing mismatch')
        require(source['reuse_row_root']==current_rows[p['row_id']]['row_root'],'Legacy/current frozen source row mismatch')
        if 'query_ids' in old:
            require(set(old['query_ids'])==set(p['old_query_ids']+p['new_query_ids']),'Legacy explicit physical query IDs changed')
        paired.append((p,old,meta))
    # Identity/full-coverage checks above precede comparison values. No A/B or
    # forgetting metric is fabricated from a legacy registration-only score.
    result=[]
    for p,old,meta in paired:
        values={new:old[prior] for new,prior in (('C_old_accuracy','old_accuracy'),('C_new_accuracy','new_accuracy'),('C_h','harmonic_mean'))}
        for name,value in values.items():
            missing=p['new_count']==0 and name!='C_old_accuracy'
            require(value is None if missing else type(value) in (int,float) and not isinstance(value,bool)
                and math.isfinite(value) and 0<=value<=1,'Invalid legacy C metric or new0 boundary')
        cold,cnew=values['C_old_accuracy'],values['C_new_accuracy']
        _close(values['C_h'],None if cnew is None else (0. if cold+cnew==0 else 2*cold*cnew/(cold+cnew)),
            'Legacy cell H is inconsistent with its own C metrics')
        result.append(dict(**{k:p[k] for k in ('row_id','model_seed','receiver','scenario','support_seed','k','new_count')},
            baseline_method=meta['method'],baseline_run_id=meta['spec']['run_id'],baseline_runtime_commit=meta['complete']['commit'],
            baseline_metrics=dict.fromkeys(METRICS)|values,
            differences={m:None if m not in values or values[m] is None else p['metrics'][m]-values[m] for m in METRICS},
            unavailable_reason='LEGACY_REGISTRATION_ONLY_SCORE_HAS_NO_MATCHED_A_B_OR_FORGETTING',
            physical_binding='SAME_VALIDATED_ONCE_CAPSULE_AND_SPLIT; LEGACY_QUERY_IDS_OPTIONAL'))
    return result


def report_benchmark(*,score,current_metadata=None,baseline_score=None,baseline_metadata=None):
    """Use this complete artifact only; explicit optional baseline is never searched."""
    validate_score(score,current_metadata)
    if baseline_score is None:comparisons=None
    elif isinstance(baseline_score,list) or baseline_score.get('status')=='SCORED':
        comparisons=_legacy_baseline_pairing(score,baseline_score,baseline_metadata)
    else:comparisons=_baseline_pairing(score,baseline_score)
    dimensions=dict(overall=(),by_k_new_count=('k','new_count'),by_receiver_scene=('receiver','scenario','k','new_count'),
        by_model_row=('model_seed','row_id','k','new_count'),by_support_seed=('support_seed','k','new_count'),
        by_model_receiver_scene_support_seed=('model_seed','row_id','receiver','scenario','support_seed','k','new_count'))
    tables={name:_cells(score['statistics'][name],dims) for name,dims in dimensions.items()}
    lines=['# D92 Margin 联合方法：冻结 query benchmark 三阶段报告','',
        f'完整 {score["row_count"]} 个模型行、{score["parent_count"]} 个物理 parent；旧类数 6，K 为 1、5、10、20，新增类数为 0、2、5、10、20。',
        'A 为适应前的原地面冻结六旧类头；B 只用旧 support 适应；C 继承同一 split 当次实际 B，注册新旧类后对全部注册列统一竞争。',
        'A/B/C 旧类准确率来自同一物理旧 query。所有行的三流预测及完成元数据先固定并独立读回，之后才由独立 scorer 连接 truth。',
        '本报告使用重复的冻结 benchmark 数据，不称为新的独立确认；结果不反馈参数、checkpoint、方法选择或选择性重跑。',
        'Query benchmark 的 K1 有独立 query，正常报告；新增 0 的新类准确率、H 和新旧绝对差记 N/A。',
        '准确率与 H 使用百分比；B−A、B−C旧及绝对差使用百分点。先逐 parent 计算 H、差值和绝对差，再等 parent 平均，不以均值重算这些量。',
        'B−A≥10 个百分点、B−C旧≤1 个百分点、绝对差≤3 个百分点为理想方向，不是晋级或重跑门槛。','']
    headings=('A旧（%）','B旧（%）','C旧（%）','C新（%）','H（%）','B−A（pp）','B−C旧（pp）','绝对差（pp）')
    for name,dims in dimensions.items():
        lines += ['## '+name,'','| '+' | '.join(list(dims)+['parent','H可测parent','旧类数','总注册类数']+list(headings))+' |',
            '|'+ '---|'*(len(dims)+4)+'---:|'*len(headings)]
        for row in tables[name]:
            registered=str(6+row['new_count']) if 'new_count' in row else '6 至 26'
            lines.append('| '+' | '.join([str(row[d]) for d in dims]+[str(row['parent_count']),str(row['measured_parent_counts']['C_h']),'6',registered]+[_format(row['values'][m]) for m in METRICS])+' |')
        lines.append('')
    lines += ['## 资源和范围','',
        '下列资源来自本次 predictor 已完成 marker，保留其测量口径。prediction 墙钟含拟合、单样本推理和归档，不能改称纯训练时间；计数不等于实际 FLOP。',
        '未提供的训练参数数、纯训练时间、设备型号、峰值 GPU、最小部署常驻量、增量星地传输和能耗均为 N/A。磁盘包字节不等于传输字节。',
        '| 行 | device 声明 | 当次 predictor 资源 |','|---|---|---|']
    import json
    for row in score['current_metadata']['rows']:
        lines.append('| '+row['binding']['row_id']+' | '+str(row.get('predictor_device') or 'N/A')+' | '+json.dumps(row['resources'],ensure_ascii=False,allow_nan=False)+' |')
    lines += ['',f'运行：`{score["run_id"]}`；实际 release commit：`{score["release_commit"]}`；preparation code commit：`{score["code_commit"]}`。','']
    if comparisons is not None:
        lines += ['## 显式已完成 baseline','',
            '仅比较 root/用户显式提供、已经完成且物理 query 与 source 身份逐 parent 匹配的 score；没有自动搜索历史结果，也不把 baseline 的 R0/B 冒充 A。',
            'Legacy SCORED/results 只提供 C旧/C新/H；A、B 与遗忘为 N/A。旧记录缺物理 ID 数组时沿用同一 VALIDATED_ONCE capsule+split_id 绑定，实际 baseline 完成及预测根由 root 独立核实。',
            '| 模型 | 接收机 | 场景 | support seed | K | 新增 | baseline | ΔC旧（pp） | ΔC新（pp） | ΔH（pp） |',
            '|---|---|---|---|---|---|---|---:|---:|---:|']
        for row in comparisons:
            lines.append('| '+' | '.join([str(row[k]) for k in ('model_seed','receiver','scenario','support_seed','k','new_count','baseline_method')]
                +[_format(row['differences'][m]) for m in ('C_old_accuracy','C_new_accuracy','C_h')])+' |')
        lines.append('')
    result=dict(schema=REPORT_SCHEMA,status=REPORT_STATUS,method=METHOD,scope=SCOPE,run_id=score['run_id'],group_id=score['group_id'],
        release_commit=score['release_commit'],code_commit=score['code_commit'],row_count=score['row_count'],parent_count=score['parent_count'],
        qp_resources=score['qp_resources'],tables=tables,resources=deepcopy(score['resources']),
        row_resources=deepcopy(score['current_metadata']['rows']),baseline_comparisons=comparisons,
        data_reuse_statement=score['data_reuse_statement'],automatic_promotion=False,goal_complete=False,
        selection_feedback_forbidden=True,performance_improvement_claimed=False)
    return '\n'.join(lines),result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('score','report','interpretation'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('baseline-score','baseline-spec','baseline-startup','baseline-complete'):
        p.add_argument('--'+name,type=Path,action='append')
    p.add_argument('--baseline-method');a=p.parse_args()
    require(not a.report.exists() and not a.interpretation.exists(),'Exclusive report outputs required')
    baselines=None;metadata=None
    if a.baseline_score:
        baselines=[read(v) for v in a.baseline_score]
        if any(v.get('status')=='SCORED' for v in baselines):
            require(a.baseline_method and a.baseline_spec and a.baseline_startup and a.baseline_complete
                and len(a.baseline_score)==len(a.baseline_spec)==len(a.baseline_startup)==len(a.baseline_complete),
                'Legacy baseline requires explicit matching spec/startup/complete/method')
            metadata=[dict(method=a.baseline_method,spec=read(s),startup=read(st),complete=read(c))
                for s,st,c in zip(a.baseline_spec,a.baseline_startup,a.baseline_complete)]
        else:
            require(len(baselines)==1,'Use one complete same-schema baseline');baselines=baselines[0]
    value=read(a.score);text,result=report_benchmark(score=value,baseline_score=baselines,baseline_metadata=metadata)
    for path in (a.report,a.interpretation):path.parent.mkdir(parents=True,exist_ok=True)
    with a.report.open('x',encoding='utf-8') as stream:stream.write(text)
    write(a.interpretation,result)
    require(read(a.interpretation)==result and a.report.read_text(encoding='utf-8')==text,'Report independent readback failed')


if __name__=='__main__':main()
