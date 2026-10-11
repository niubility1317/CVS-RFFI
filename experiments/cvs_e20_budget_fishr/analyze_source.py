"""Full closed source-log analysis; prediction, scoring and truth are never opened."""
import argparse
import csv
import gzip
import io
import json
import math
import re
import statistics
import time
from pathlib import Path


class Moments:
    def __init__(self):
        self.values=[]
    def add(self,value):
        if value is None:return
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Nonfinite/non-numeric observation')
        self.values.append(float(value))
    def result(self):
        values=self.values;n=len(values)
        if not n:return dict(observations=0,mean=None,min=None,max=None,first=None,last=None)
        ordered=sorted(values)
        def quantile(p):
            position=(n-1)*p;lo=int(position);hi=math.ceil(position)
            return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
        return dict(observations=n,mean=math.fsum(values)/n,min=ordered[0],max=ordered[-1],
                    first=values[0],last=values[-1],positive=sum(v>0 for v in values),
                    zero=sum(v==0 for v in values),q10=quantile(.1),median=quantile(.5),q90=quantile(.9))


def _add(mapping,key,value):
    mapping.setdefault(key,Moments()).add(value)


def _finish(mapping):
    return {key:value.result() for key,value in mapping.items()}


def _finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(_finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(_finite(v) for v in value)
    return True


def _read_json(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    if not _finite(value):raise ValueError('Nonfinite JSON: '+str(path))
    return value


def _read_jsonl(path):
    rows=[]
    with path.open(encoding='utf-8') as stream:
        for index,line in enumerate(stream,1):
            if not line.strip():raise ValueError('Blank structured record '+str(path)+':'+str(index))
            item=json.loads(line)
            if not _finite(item):raise ValueError('Nonfinite record '+str(path)+':'+str(index))
            rows.append(item)
    return rows


def _expected_counts(c):
    n=c['updates'];active=n-c['semi']['warmup_updates'];mc=c['multilevel']
    aux=n//mc['interval']-c['semi']['warmup_updates']//mc['interval'] if mc['enabled'] else 0
    return dict(identity_updates=n,L_packets=n*c['batch_size'],U_packets=active*c['batch_size'] if c['unlabeled'] else 0,
                weak_forwards=active if c['unlabeled'] else 0,physical_L=0,physical_U=0,feature_U=0,margin_U=0,
                EMA_updates=0,satellite_calls=0,auxiliary_steps=aux,auxiliary_L=aux*mc['batch_size'],
                auxiliary_forwards=aux*(2 if mc['mixstyle_l'] else 1),
                fishr_steps=aux if mc['fishr_mode']!='none' else 0,mixstyle_steps=aux if mc['mixstyle_l'] else 0,
                mixstyle_U_steps=active if mc['mixstyle_u'] else 0)


def _source_stdout(path):
    """The worker reuses stdout for target evaluation: STOP at source-end marker.

    This reads every source-phase line, but deliberately never reads the later
    prediction/scoring segment of the shared log, even if it already exists.
    """
    result=dict(path=str(path),source_lines=0,source_utf8_bytes=0,closed_source_segment=False,
                errors=[],warnings=[],nonfinite_markers=[],resolved=[],epochs=[],warmup=[],source_trained=None,
                target_segment_read=False)
    if not path.is_file():return result
    with path.open(encoding='utf-8',errors='strict') as stream:
        for line in stream:
            result['source_lines']+=1;result['source_utf8_bytes']+=len(line.encode('utf-8'))
            if re.search(r'Traceback|CUDA out of memory|FloatingPointError|RuntimeError|ValueError|\bKilled\b',line):
                result['errors'].append(dict(line=result['source_lines'],text=line.rstrip()))
            if re.search(r'\b(?:Warning|warning|UserWarning|RuntimeWarning)\b',line):
                result['warnings'].append(dict(line=result['source_lines'],text=line.rstrip()))
            if re.search(r'(?<![A-Za-z_])(?:NaN|nan|Infinity|inf)(?![A-Za-z_])',line):
                result['nonfinite_markers'].append(dict(line=result['source_lines'],text=line.rstrip()))
            for prefix,key in [('RESOLVED ','resolved'),('EPOCH ','epochs'),('HEAD_WARMUP ','warmup')]:
                if line.startswith(prefix):result[key].append(json.loads(line[len(prefix):]))
            if line.startswith('SOURCE_TRAINED '):
                result['source_trained']=json.loads(line[len('SOURCE_TRAINED '):])
                result['closed_source_segment']=True
                break
    return result


def _style_add(target,info,packets):
    target.setdefault('calls',0);target['calls']+=1
    target.setdefault('enabled_calls',0);target['enabled_calls']+=int(info['enabled'])
    target.setdefault('packet_layer_occurrences',0);target['packet_layer_occurrences']+=packets
    for key in ('eligible_packets','changed_packets'):
        target.setdefault(key,0);target[key]+=int(info.get(key,0))
    target.setdefault('stats',{})
    for key in ('changed_ratio','movement_mean','movement_max','delta_ratio'):
        if key in info:_add(target['stats'],key,info[key])
    if not 0<=info.get('changed_packets',0)<=info.get('eligible_packets',0)<=packets:
        raise ValueError('Invalid actual style coverage')
    donors=info.get('donors',[])
    if donors and (len(donors)!=packets or any(not 0<=j<packets for j in donors)):
        raise ValueError('Invalid logged donor occurrence index')


def _style_finish(value):
    result={k:v for k,v in value.items() if k!='stats'}
    result['metrics']=_finish(value.get('stats',{}))
    count=result.get('packet_layer_occurrences',0)
    result['eligible_fraction']=result.get('eligible_packets',0)/count if count else None
    result['changed_fraction']=result.get('changed_packets',0)/count if count else None
    result['conditional_changed_fraction']=result.get('changed_packets',0)/result['eligible_packets'] if result.get('eligible_packets') else None
    return result


def _check_warmup(source,c,stdout):
    summary=_read_json(source/'head_warmup.json');records=_read_jsonl(source/'head_warmup.jsonl')
    expected_steps=200 if c['multilevel']['enabled'] else 0
    if len(records)!=expected_steps or summary['optimizer_steps']!=expected_steps:
        raise ValueError('Incomplete head warmup')
    if not summary['base_state_unchanged'] or summary['backbone_updates']!=0 or summary['auxiliary_inheritance']:
        raise ValueError('Warmup changed backbone or inherited auxiliary state')
    if summary['source_L_cache_packets']!=(6300 if expected_steps else 0) or summary['head_packet_occurrences']!=expected_steps*60:
        raise ValueError('Warmup packet budget differs')
    if summary['cache_forward_calls']!=(50 if expected_steps else 0):raise ValueError('Warmup cache coverage differs')
    stats={}
    for i,row in enumerate(records,1):
        if row['step']!=i or row['source_L_packets']!=60 or row['backbone_updates'] or row['lr']!=.003 or row['weight_decay']!=.0001:
            raise ValueError('Warmup step/optimizer protocol differs')
        for key,value in row.items():
            if isinstance(value,(int,float)):_add(stats,key,value)
    printed=[records[i-1] for i in (1,50,100,150,200)] if expected_steps else []
    if stdout['warmup']!=printed:raise ValueError('Warmup stdout/structured record mismatch')
    return dict(summary=summary,records=records,statistics=_finish(stats),
                backbone_unchanged_evidence='warmup runtime exact state_dict comparison; retained summary true; no extra checkpoint load')


def analyze_row(base,registered,c):
    rid=registered['row_id'];source=base/rid/'source'
    base_result=dict(row_id=rid,arm=registered['arm'],seed=registered['model_seed'])
    if not (source/'completion.json').is_file():
        return dict(base_result,status='PARTIAL',reason='source completion absent; live gzip not opened')
    stdout=_source_stdout(base/'logs'/(rid+'.log'))
    if not stdout['closed_source_segment']:
        return dict(base_result,status='PARTIAL',reason='source stdout not closed by SOURCE_TRAINED; gzip not opened',
                    source_stdout_lines_seen=stdout['source_lines'])
    done=_read_json(source/'completion.json');resolved=_read_json(source/'resolved_config.json')
    initialization=_read_json(source/'initialization.json');frozen=_read_json(base/rid/'source_frozen.json')
    if done['config']!=c or frozen['config']!=c or any(resolved.get(k)!=v for k,v in c.items()):
        raise ValueError('Actual configuration differs from fixed preregistration')
    if done['status']!='SOURCE_TRAINED' or done['target_access'] or done['target_evaluated'] or done['epoch']!=20 or done['total_epochs']!=40:
        raise ValueError('Source completion or budget contract differs')
    if frozen['status']!='SOURCE_FROZEN' or frozen['target_access'] or frozen['target_scores_consumed']:
        raise ValueError('Source freeze differs')
    if initialization!=done['initialization'] or initialization['scratch_only'] or not initialization['reference_E20_loaded']:
        raise ValueError('Missing E20 provenance')
    if initialization['source_roles']!='EXACT_MATCH' or initialization['target_contact'] or initialization['auxiliary_inheritance']:
        raise ValueError('Invalid inherited source provenance')
    if initialization['parent_checkpoint']!=c['parent_checkpoint'] or initialization['checkpoint_sources']!=[c['parent_checkpoint']]:
        raise ValueError('Wrong fixed parent')
    if stdout['resolved']!=[resolved] or stdout['source_trained']!=done:
        raise ValueError('Full source stdout differs from committed artifacts')
    warmup=_check_warmup(source,c,stdout)
    if resolved['head_warmup']!=warmup['summary'] or done['resources']['head_warmup']!=warmup['summary']:
        raise ValueError('Warmup cost/summary mismatch')
    fixed=_expected_counts(c);counts={k:0 for k in fixed};counts['mixed_U_packets']=0
    per_epoch={};metric_stats={};detail_stats={};first={};gradients={};active_gradients={};gradient_ratios={}
    audit_records=[];layers={};l_style={};u_style={};budgets={};combined={};budget_snapshots=[];epochs_running=[]
    count=0;step_file=source/'step_metrics.jsonl.gz'
    stable_before=(step_file.stat().st_size,step_file.stat().st_mtime_ns)
    # Completion and full source-end markers prove the gzip writer has closed.
    # Reading to EOF also checks the gzip footer/CRC; truncated files are errors.
    with gzip.open(step_file,'rt',encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line);count+=1
            if item['step']!=count or item['epoch']!=(count-1)//222+1 or not _finite(item):
                raise ValueError('Nonconsecutive/nonfinite full source step '+str(count))
            ep=item['epoch'];metrics=item['metrics'];detail=item['details']
            aggregate=per_epoch.setdefault(ep,dict(steps=0,sums={},metric_counts={},style_U={},budget={}))
            aggregate['steps']+=1
            for key,value in metrics.items():
                if isinstance(value,(int,float)):
                    _add(metric_stats,key,value);aggregate['sums'][key]=aggregate['sums'].get(key,0.)+value
                    aggregate['metric_counts'][key]=aggregate['metric_counts'].get(key,0)+1
            counts['identity_updates']+=1;counts['L_packets']+=c['batch_size']
            for key in counts:
                if key in ('identity_updates','L_packets','EMA_updates','satellite_calls'):continue
                counts[key]+=detail[key]
                if detail[key] and key not in first:first[key]=count
            for key,value in detail.items():
                if isinstance(value,(int,float)):_add(detail_stats,key,value)
            active=count>c['semi']['warmup_updates']
            if detail['U_packets']!=(c['batch_size'] if active and c['unlabeled'] else 0):raise ValueError('U execution schedule differs')
            expected_aux=int(active and c['multilevel']['enabled'] and count%c['multilevel']['interval']==0)
            if detail['auxiliary_steps']!=expected_aux:raise ValueError('Auxiliary execution schedule differs')
            audit=item.get('gradient_audit')
            if bool(audit)!=(count%444==0):raise ValueError('Missing/unexpected scheduled gradient audit')
            if audit:
                audit_records.append(dict(step=count,epoch=ep,audit=audit))
                for loss,groups in audit['weighted_dense_gradient_norms'].items():
                    for group,value in groups.items():
                        _add(gradients,loss+'/'+group,value)
                        if active:_add(active_gradients,loss+'/'+group,value)
                        reference=audit['weighted_dense_gradient_norms']['ce'].get(group)
                        if reference is not None and reference>0:
                            _add(gradient_ratios,loss+'/'+group,value/reference)
            multi=detail.get('multilevel')
            if multi:
                if multi['auxiliary_packets']!=60 or multi['distinct_PID']!=60 or set(multi['source_RX'])!={1,3,4,6,8}:
                    raise ValueError('Auxiliary source support differs')
                for level,info in multi['levels'].items():
                    stats=layers.setdefault(level,{})
                    for key,value in info.items():
                        if isinstance(value,(int,float)):_add(stats,key,value)
                    if info.get('style') is not None:_style_add(l_style.setdefault(level,{}),info['style'],60)
            style=detail.get('U_style')
            if style:
                if style['used_U_truth'] or style['role']!='U_strong':raise ValueError('Wrong U style role')
                changed=0
                for level,info in style['levels'].items():
                    _style_add(u_style.setdefault(level,{}),info,c['batch_size'])
                    _style_add(aggregate['style_U'].setdefault(level,{}),info,c['batch_size'])
                    changed+=info['changed_packets']
                if changed!=detail['mixed_U_packets']:raise ValueError('U packet-layer count mismatch')
            budget=detail.get('fishr_budget')
            if budget and budget['levels']:
                budget_snapshots.append(dict(step=count,epoch=ep,**budget))
                for level,info in budget['levels'].items():
                    record=budgets.setdefault(level,dict(calls=0,cap_count=0,skip_reasons={},stats={}))
                    record['calls']+=1;record['cap_count']+=int(info['cap_active'])
                    if info['reason']:record['skip_reasons'][info['reason']]=record['skip_reasons'].get(info['reason'],0)+1
                    for key,value in info.items():
                        if isinstance(value,(int,float)):
                            _add(record['stats'],key,value)
                            _add(aggregate['budget'].setdefault(level,{}),key,value)
                    if not info['coefficient_detached']:raise ValueError('Fishr coefficient not detached')
                for level,info in budget['combined_group_gradient_ratios'].items():
                    for key,value in info.items():_add(combined,level+'/'+key,value)
            if count%222==0:epochs_running.append(dict(epoch=ep,counts=dict(counts)))
    if stable_before!=(step_file.stat().st_size,step_file.stat().st_mtime_ns):raise ValueError('Supposedly closed source log changed during scan')
    if count!=4440 or done['optimizer_steps']!=4440 or done['logged_steps']!=4440 or len(per_epoch)!=20:
        raise ValueError('Incomplete 4440-step/20-epoch source record')
    if any(aggregate['steps']!=222 for aggregate in per_epoch.values()):raise ValueError('Incomplete epoch')
    if any(counts[k]!=value for k,value in fixed.items()) or counts!=done['mechanism_execution']:
        raise ValueError('Actual mechanism counts differ from fixed schedule/completion')
    if done['distinct_U_visited']!=(56700 if c['unlabeled'] else 0):raise ValueError('Incomplete U coverage')
    epochs=_read_jsonl(source/'epoch_metrics.jsonl')
    if [row['epoch'] for row in epochs]!=list(range(1,21)) or stdout['epochs']!=epochs:
        raise ValueError('Epoch JSONL/stdout sequence differs')
    for ep,cumulative in zip(epochs,epochs_running):
        aggregate=per_epoch[ep['epoch']]
        if ep['optimizer_steps']!=ep['epoch']*222 or ep['total_epoch']!=20+ep['epoch'] or ep['counts']!=cumulative['counts']:
            raise ValueError('Epoch budget/cumulative mechanism count mismatch')
        for key,value in aggregate['sums'].items():
            if key not in ep or not math.isclose(ep[key],value/aggregate['metric_counts'][key],rel_tol=1e-9,abs_tol=1e-10):
                raise ValueError('Full step/epoch mean mismatch: '+key)
        expected_v=ep['epoch'] in c['validation']['epochs']
        if bool(ep['validation'])!=expected_v:raise ValueError('Source V schedule mismatch')
        if expected_v and (ep['validation']['V_count']!=27000 or set(ep['validation']['V_RX_accuracy'])!={'1','3','4','6','8'}):
            raise ValueError('Source V support differs')
    with (source/'epoch_metrics.csv').open(encoding='utf-8',newline='') as stream:csv_rows=list(csv.DictReader(stream))
    if len(csv_rows)!=20:raise ValueError('Incomplete compact CSV')
    for row,ep in zip(csv_rows,epochs):
        if int(row['epoch'])!=ep['epoch']:raise ValueError('CSV epoch differs')
        for key,value in row.items():
            expected=ep[key]
            if isinstance(expected,(int,float)) and not math.isclose(float(value),expected,rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('CSV/JSON epoch metric mismatch: '+key)
    for aggregate in per_epoch.values():
        aggregate['style_U']={k:_style_finish(v) for k,v in aggregate['style_U'].items()}
        aggregate['budget']={k:_finish(v) for k,v in aggregate['budget'].items()}
    budget_summary={level:dict(calls=value['calls'],cap_count=value['cap_count'],cap_fraction=value['cap_count']/value['calls'],
                     skip_reasons=value['skip_reasons'],statistics=_finish(value['stats'])) for level,value in budgets.items()}
    return dict(base_result,status='COMPLETE',steps=count,source_only=True,epochs=epochs,
                source_V=[dict(epoch=e['epoch'],**e['validation']) for e in epochs if e['validation']],
                parameter_movement=[dict(epoch=e['epoch'],groups=e['weight_movement']) for e in epochs],
                step_metrics=_finish(metric_stats),detail_summary=_finish(detail_stats),counts=counts,
                fixed_expected_counts=fixed,first_execution=first,per_epoch=per_epoch,
                gradient_audits=audit_records,gradient_summary=_finish(gradients),
                active_gradient_summary=_finish(active_gradients),gradient_ratio_summary=_finish(gradient_ratios),
                layer_summary={level:_finish(value) for level,value in layers.items()},
                L_style={level:_style_finish(value) for level,value in l_style.items()},
                U_style={level:_style_finish(value) for level,value in u_style.items()},
                fishr_budget=budget_summary,combined_fishr_gradient_ratios=_finish(combined),
                fishr_budget_records=budget_snapshots,warmup=warmup,initialization=initialization,
                resolved_execution=resolved['execution'],execution_commit=resolved['commit'],resources=done['resources'],
                stdout_errors=stdout['errors'],
                source_stdout=dict(path=stdout['path'],lines=stdout['source_lines'],utf8_bytes=stdout['source_utf8_bytes'],
                    errors=stdout['errors'],warnings=stdout['warnings'],nonfinite_markers=stdout['nonfinite_markers'],
                    closed_source_segment=True,target_segment_read=False),
                files=dict(step_log=str(step_file),epochs=str(source/'epoch_metrics.jsonl'),csv=str(source/'epoch_metrics.csv'),
                    warmup=str(source/'head_warmup.jsonl'),completion=str(source/'completion.json')))


def analyze(base,configs):
    base=Path(base);rows=[];pending=[];errors=[]
    for c in configs:
        row={key:c[key] for key in ('row_id','arm','model_seed')}
        try:
            result=analyze_row(base,row,c)
            if result['status']=='COMPLETE':rows.append(result)
            else:pending.append(result)
        except (ValueError,KeyError,TypeError,OSError,EOFError) as exc:
            errors.append(dict(row,reason=repr(exc),status='INVALID_SOURCE_EVIDENCE'))
    return dict(status='FAILED' if errors else 'PARTIAL' if pending else 'COMPLETE',run_id=base.name,
                read_at=time.time(),expected_rows=len(configs),rows=rows,pending=pending,errors=errors,
                total_steps=sum(r['steps'] for r in rows),total_head_warmup_steps=sum(r['warmup']['summary']['optimizer_steps'] for r in rows),
                source_only=True,full_closed_source_step_epoch_CSV_warmup_and_stdout_read=True,
                stdout_boundary='All source lines through SOURCE_TRAINED; later prediction/scoring segment never read',
                non_closed_gzip_read=False,target_prediction_or_scores_read=False)


# LOCAL CONTROL: remote payload stops above this line; stdlib source reader only.
def collect():
    from . import design as d
    from experiments.cvs_phase1_stack.publish import ssh
    own=Path(__file__).read_text(encoding='utf-8').split('# LOCAL CONTROL:')[0]
    configs=[d.config(row) for row in d.rows()]
    payload=own+'\nprint(json.dumps(analyze('+repr(d.BASE.as_posix())+','+repr(configs)+'),allow_nan=False))\n'
    compile(payload,'source-only-analysis','exec')
    return json.loads(ssh(payload))


def report(value):
    def fmt(number,digits=4):return 'N/A' if number is None else f'{number:.{digits}f}'
    def stat(row,level,key):return row.get('fishr_budget',{}).get(level,{}).get('statistics',{}).get(key,{}).get('mean')
    lines=['# E20＋20 全量源域训练分析','',
        f"状态：{value['status']}。完整核对 {len(value['rows'])}/{value['expected_rows']} 行、{value['total_steps']} 次主更新、{value['total_head_warmup_steps']} 次辅助头预热更新。",
        '每个完成行读取全部 4,440 条逐步记录、20 轮 JSONL/CSV、完整暖头记录，以及共享 stdout 中从开头至 SOURCE_TRAINED 的全部源训练内容。后续预测和评分段不读取。未完成行的活动 gzip 不打开。','',
        '|方案／种子|主 CE：首轮→末轮|末轮 U 接受数／128|末轮源 V／最差 RX|辅助步数|主训练秒／暖头秒|',
        '|---|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        first,last=row['epochs'][0],row['epochs'][-1];v=last['validation'];cost=row['resources']
        lines.append(f"|{row['row_id']}|{first['ce']:.4f} → {last['ce']:.4f}|{last['accepted_U']:.2f}|{v['V_accuracy']:.4%}／{v['V_worst_RX']:.4%}|{row['counts']['auxiliary_steps']}|{cost['identity_training_and_source_diagnostics_seconds']:.1f}／{row['warmup']['summary']['seconds']:.1f}|")
    lines+=['','## 三种子源域汇总','','下表只使用固定末轮源 V，不作候选排序或选模。标准差为三个模型种子的样本标准差。','',
        '|方案|完成种子数|源 V 均值 ± 标准差|最差 RX 均值|相对 CE 配对变化（百分点）|',
        '|---|---:|---:|---:|---:|']
    reference={r['seed']:r['source_V'][-1]['V_accuracy'] for r in value['rows'] if r['arm']=='CE'}
    for arm in dict.fromkeys(r['arm'] for r in value['rows']):
        group=[r for r in value['rows'] if r['arm']==arm]
        accuracy=[r['source_V'][-1]['V_accuracy'] for r in group]
        paired=[r['source_V'][-1]['V_accuracy']-reference[r['seed']] for r in group if r['seed'] in reference]
        deviation=statistics.stdev(accuracy) if len(accuracy)>1 else None
        lines.append(f"|{arm}|{len(group)}/3|{statistics.mean(accuracy):.4%} ± {fmt(deviation*100 if deviation is not None else None)}%|{statistics.mean(r['source_V'][-1]['V_worst_RX'] for r in group):.4%}|{fmt(statistics.mean(paired)*100 if paired else None)}|")
    lines+=['','## Fishr 实际力度','',
        '下表使用所有实际 Fishr 调用，而不是只用每 444 步一次的梯度审计。每层系数停止梯度；两层最终取均值，因此每层均值前预算比例和最终直接贡献比例不同。中层项还会回传浅层，合并后的浅层比例需单独读取。','',
        '|方案／种子|层|调用数|系数均值|cap 比例|请求比例|均值前达到比例|均值后直接比例|合并后对主 CE 比例|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        for level,summary in row['fishr_budget'].items():
            combined=row['combined_fishr_gradient_ratios'].get(level+'/ratio_to_main_ce',{}).get('mean')
            lines.append(f"|{row['row_id']}|{level}|{summary['calls']}|{fmt(stat(row,level,'coefficient'),2)}|{summary['cap_fraction']:.2%}|{fmt(stat(row,level,'requested_ratio_pre_level_mean'))}|{fmt(stat(row,level,'achieved_ratio_pre_level_mean'))}|{fmt(stat(row,level,'achieved_direct_contribution_ratio'))}|{fmt(combined)}|")
    lines+=['','## MixStyle 实际覆盖','',
        '覆盖分母为该层实际调用中的样本出现次数；两层之和是 packet-layer 次数，不是不同物理样本数。未开启 Bernoulli 调用也计入调用分母；强度等稀疏指标使用各自实际观测次数。','',
        '|方案／种子|角色／层|调用数|合法 donor 比例|实际改变比例|相对特征变化均值|',
        '|---|---|---:|---:|---:|---:|']
    for row in value['rows']:
        for role in ('L_style','U_style'):
            for level,summary in row[role].items():
                delta=summary['metrics'].get('delta_ratio',{}).get('mean')
                lines.append(f"|{row['row_id']}|{role}/{level}|{summary['calls']}|{summary['eligible_fraction']:.2%}|{summary['changed_fraction']:.2%}|{fmt(delta)}|")
    lines+=['','## 证据与限制','',
        '- JSON 保留全部 20 轮的损失、学习率、源 V 各 RX／日期、参数位移；保留所有 10 次完整分组梯度审计及其独立观测数。',
        '- 每次 Fishr 的系数、cap、请求／实测比例及合并梯度比例全部保留。源 V 只作健康诊断，不选轮次、方案或重跑。',
        '- 暖头的骨干不变结论来自训练时逐位 state_dict 比较及日志。分析器不重新加载 checkpoint，也不访问目标预测或评分。',
        '- 主训练成本为追加 20 轮；父训练 20 轮单独复用，累计 40 轮。暖头缓存和 200 次更新成本另列。','',
        '未完成行：'+(', '.join(r['row_id'] for r in value['pending']) or '无。'),
        '证据异常：'+('; '.join(r['row_id']+': '+r['reason'] for r in value['errors']) or '无。')]
    stdout_errors=[r['row_id'] for r in value['rows'] if r['source_stdout']['errors']]
    lines+=['源训练 stdout 异常标记：'+(', '.join(stdout_errors) or '无。')]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    from . import design as d
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=collect()
    d.write(args.output/'evidence/source_full_analysis.json',value)
    (args.output/'source_trajectory.md').write_text(report(value),encoding='utf-8')
    print(json.dumps(dict(status=value['status'],complete_rows=len(value['rows']),expected_rows=value['expected_rows'],
                         steps=value['total_steps'],pending=value['pending'],errors=value['errors'],source_only=True)))
