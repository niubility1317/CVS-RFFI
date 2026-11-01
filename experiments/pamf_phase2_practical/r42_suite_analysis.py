"""Streaming, score-only readout of the frozen R42 completion suite."""
import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from .io import read, write, require
from .milkr_r42_suite import tasks, task_runtime
from .r42_suite_worker import desired_rows, row_key

METRICS=('old_macro_accuracy','new_macro_accuracy','H_macro','macro_f1','class_floor','accuracy','macro_accuracy')


@dataclass
class Moments:
    n:int=0
    mean:float=0.
    m2:float=0.

    def add(self,value):
        if value is None:return
        require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),'Nonfinite or nonnumeric score')
        self.n+=1;delta=value-self.mean;self.mean+=delta/self.n;self.m2+=delta*(value-self.mean)

    def result(self):
        return dict(n=self.n,mean=self.mean if self.n else None,sample_SD=math.sqrt(max(self.m2,0)/(self.n-1)) if self.n>1 else None)


@dataclass
class SeedMoments:
    units:Moments=field(default_factory=Moments)
    seeds:dict=field(default_factory=dict)

    def add(self,seed,value):
        self.units.add(value);self.seeds.setdefault(str(seed),Moments()).add(value)

    def result(self):
        means=Moments();seeds={s:m.result() for s,m in sorted(self.seeds.items())}
        for value in seeds.values():means.add(value['mean'])
        return dict(unit_equal_weight=self.units.result(),seed_means=seeds,seed_mean_statistics=means.result())


def _scope(job):
    _,pool,episode,rx=job['capsule'].split('/')
    scope='MIX' if episode.startswith('MIX') else episode[:2]
    return scope,pool,episode,rx


def _strata(job):
    scope,pool,episode,rx=_scope(job)
    return [('all',scope),('pool',scope+'/'+pool),('receiver',scope+'/'+rx),('episode',episode),('scene',scope)]


def _validate_rows(payload,job,expected,natural=None):
    require(isinstance(payload.get('rows'),list),'Score rows missing')
    if natural is None:require(payload['model_seed']==job['seed'] and payload.get('truth_last') is True,'Wrong score seed/truth-last state')
    found={}
    for row in payload['rows']:
        key=row_key(row);require(key in expected and key not in found,'Unexpected or duplicate scored row')
        require(row.get('split_id') and isinstance(row.get('registered_classes'),list),'Missing scored split/classes')
        require(len(row['registered_classes'])==6+row['new_count'],'Wrong registered class count')
        require(isinstance(row.get('metrics'),dict),'Missing row metrics')
        metrics=row['metrics'];require(all(k in metrics for k in METRICS),'Missing metric fields; do not fill missing scores')
        require(isinstance(metrics.get('count'),int) and metrics['count']>0,'Missing query count')
        if row['stage']=='C':require(all(metrics[k] is not None for k in METRICS[:5]),'Incomplete C metrics')
        if natural is not None:
            source=natural[key]
            require(all(row[k]==source[k] for k in ('split_id','registered_classes')),'Common-query split/classes mismatch')
            require(metrics['count']<=source['metrics']['count'],'Common query larger than natural query')
        else:
            _,pool,episode,rx=_scope(job)
            for keyname,value in (('physical_pool',pool),('episode',episode),('receiver',rx),('model_seed',job['seed'])):
                require(row.get(keyname)==value,'Wrong scored unit identity: '+keyname)
            if row.get('query_count') is not None:require(metrics['count']==row['query_count'],'Natural query count mismatch')
        found[key]=row
    require(set(found)==set(expected),'Incomplete score row matrix')
    return found


class Aggregates:
    def __init__(self):self.rows={};self.classes={};self.pairs={}

    def add_row(self,dataset,job,row):
        arm,stage,k,n,path=row_key(row);seed=job['seed'];metrics=row['metrics']
        for stratum,value in _strata(job):
            self._metrics((dataset,stratum,value,arm,stage,k,n,path),seed,metrics)
        for family in ('by_day','by_scene'):
            for label,partial in row.get(family,{}).items():
                self._metrics((dataset,family,_scope(job)[0]+'/'+str(label),arm,stage,k,n,path),seed,partial)
        recalls=metrics.get('class_recall');counts=metrics.get('class_counts')
        if recalls is not None or counts is not None:
            require(isinstance(recalls,list) and isinstance(counts,list) and len(recalls)==len(counts)==len(row['registered_classes']),'Incomplete class metrics')
            for class_name,recall,count in zip(row['registered_classes'],recalls,counts):
                require(isinstance(count,int) and count>=0,'Invalid class count')
                if count>0:
                    for stratum,value in _strata(job):
                        key=(dataset,stratum,value,arm,stage,k,n,path,class_name)
                        self.classes.setdefault(key,SeedMoments()).add(seed,recall)

    def _metrics(self,key,seed,metrics):
        bucket=self.rows.setdefault(key,{m:SeedMoments() for m in METRICS})
        for metric in METRICS:
            require(metric in metrics,'Incomplete day/scene metric')
            bucket[metric].add(seed,metrics[metric])

    def add_pairs(self,dataset,job,rows):
        for key,row in rows.items():
            arm,stage,k,n,path=key
            if arm!='R42' or stage not in ('B','C'):continue
            sp=int(path[1]);qp=int(path[-1])
            comparisons=[]
            if qp in (0,1):comparisons.append(('matched_original','LocalRidge',f'S{sp}Q{qp}',True))
            if qp==2:
                comparisons.extend((('fusion_vs_original_raw','LocalRidge',f'S{sp}Q0',False),('fusion_vs_original_G','LocalRidge',f'S{sp}Q1',False)))
            if qp in (1,2):comparisons.append(('G_only_vs_R42_raw' if qp==1 else 'fusion_vs_R42_raw','R42',f'S{sp}Q0',False))
            for comparison,other_arm,other_path,matched in comparisons:
                other=rows[(other_arm,stage,k,n,other_path)]
                require(row['split_id']==other['split_id'] and row['registered_classes']==other['registered_classes'],'Unpaired comparison split')
                require(row['metrics']['count']==other['metrics']['count'],'Unpaired comparison query counts')
                primary='H_macro' if stage=='C' else 'old_macro_accuracy'
                for stratum,value in _strata(job):
                    pairkey=(dataset,stratum,value,stage,k,n,path,comparison,other_arm,other_path,matched)
                    bucket=self.pairs.setdefault(pairkey,dict(metrics={m:SeedMoments() for m in METRICS},units=0,wins=0,at_least_5pp=0,nonnegative=0,primary=primary))
                    a=row['metrics'][primary];b=other['metrics'][primary]
                    require(a is not None and b is not None,'Missing primary paired metric')
                    delta=100*(a-b);bucket['units']+=1;bucket['wins']+=delta>1e-10;bucket['nonnegative']+=delta>=-1e-10;bucket['at_least_5pp']+=delta>=5-1e-10
                    for metric in METRICS:
                        a=row['metrics'][metric];b=other['metrics'][metric]
                        bucket['metrics'][metric].add(job['seed'],100*(a-b) if a is not None and b is not None else None)

    def result(self):
        rows=[];classes=[];pairs=[]
        for key,bucket in sorted(self.rows.items()):
            dataset,stratum,value,arm,stage,k,n,path=key
            rows.append(dict(dataset=dataset,stratum=stratum,value=value,arm=arm,stage=stage,k=k,new_count=n,input_path=path,metrics={m:s.result() for m,s in bucket.items()}))
        for key,stats in sorted(self.classes.items()):
            dataset,stratum,value,arm,stage,k,n,path,class_name=key
            classes.append(dict(dataset=dataset,stratum=stratum,value=value,arm=arm,stage=stage,k=k,new_count=n,input_path=path,class_name=class_name,recall=stats.result()))
        for key,bucket in sorted(self.pairs.items()):
            dataset,stratum,value,stage,k,n,path,comparison,arm,other_path,matched=key
            metrics={m:s.result() for m,s in bucket['metrics'].items()};mean=metrics[bucket['primary']]['unit_equal_weight']['mean']
            pairs.append(dict(dataset=dataset,stratum=stratum,value=value,stage=stage,k=k,new_count=n,input_path=path,comparison=comparison,
                comparator_arm=arm,comparator_path=other_path,matched_input=matched,units=bucket['units'],primary_metric=bucket['primary'],
                unit_wins=bucket['wins'],unit_nonnegative=bucket['nonnegative'],unit_at_least_5pp=bucket['at_least_5pp'],
                mean_at_least_5pp=mean>=5-1e-10 if mean is not None else None,delta_pp=metrics))
        return rows,classes,pairs


def _resources(folder,buckets):
    for path in sorted(folder.glob('*-resources.json')):
        value=read(path);require(value.get('status')=='COMPLETE','Incomplete actual resource measurement')
        key=(value.get('arm','shared'),value.get('stage','unknown'),value.get('support_path'),value.get('query_path'))
        bucket=buckets.setdefault(key,dict(count=0,sums={},peaks={},null_fields=set()))
        bucket['count']+=1
        for name in ('wall_seconds','CPU_user_seconds','CPU_system_seconds','G_seconds','F_feature_seconds','F_orbit_feature_seconds','F_native_feature_seconds','source_encoder_seconds','head_fit_seconds','head_seconds','head_state_array_bytes','native_optimizer_steps'):
            if name in value:
                if value[name] is None:bucket['null_fields'].add(name)
                else:
                    require(isinstance(value[name],(int,float)) and math.isfinite(value[name]),'Invalid measured resource')
                    bucket['sums'][name]=bucket['sums'].get(name,0)+value[name]
        for name in ('CPU_RSS_sampled_peak','CUDA_allocated_peak','CUDA_allocated_delta','CUDA_reserved'):
            if value.get(name) is not None:bucket['peaks'][name]=max(bucket['peaks'].get(name,0),value[name])


def _source_cost(rt):
    result=[]
    for seed in rt['model_seeds']:
        folder=Path(rt['run_root'])/'sources'/f's{seed}';path=folder/'source_selection.json'
        if not path.is_file():result.append(dict(model_seed=seed,status='MISSING_METADATA',source_new_optimizer_updates=None));continue
        p=read(path);training=p.get('inherited_source_metric_training',[]);total=0;deployed=0;valid=bool(training);final_seen=False
        for row in training:
            for orbit in ('C4','C8'):
                steps=row.get(orbit,{}).get('steps')
                if not isinstance(steps,int):valid=False;continue
                total+=steps
                if row.get('held_RX','absent') is None:deployed+=steps;final_seen=True
        result.append(dict(model_seed=seed,status=p.get('status'),source_new_optimizer_updates=p.get('source_new_optimizer_updates'),
            inherited_source_training_optimizer_updates=total if valid else None,inherited_deployed_encoder_optimizer_updates=deployed if valid and final_seen else None,
            parent_source_checkpoint=p.get('parent_source_checkpoint'),parent_source_run_id=p.get('parent_source_run_id'),
            source_correlation_weighting=p.get('source_correlation_weighting')))
    new=[r['source_new_optimizer_updates'] for r in result]
    return dict(per_seed=result,new_optimizer_updates=sum(new) if all(isinstance(v,int) for v in new) else None,
                inherited_cost_scope='original source folds plus final fit; no inherited source cost attributed as new zero-cost training')


def _csv(path,rows,fields):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:row.get(k) for k in fields})


def _flat(record,stats):
    return dict(record,n=stats['unit_equal_weight']['n'],mean=stats['unit_equal_weight']['mean'],unit_sample_SD=stats['unit_equal_weight']['sample_SD'],
                seed_count=stats['seed_mean_statistics']['n'],seed_mean=stats['seed_mean_statistics']['mean'],seed_sample_SD=stats['seed_mean_statistics']['sample_SD'])


def analyze(rt,output):
    matrix=tasks(rt)
    if rt.get('mix_only') is True:
        matrix=[task for task in matrix if '/MIX' in task['capsule']]
        require(len(matrix)==672,'Wrong MIX-only matrix')
    require(len({t['key'] for t in matrix})==len(matrix) and len({(t['seed'],t['capsule']) for t in matrix})==len(matrix),'Duplicate task units')
    root=Path(rt['run_root']);missing=[];complete=0
    for task in matrix:
        score=Path(task_runtime(rt,task)['run_root'])/'scores'/task['capsule'].split('/')[-1]
        if not (score/'complete.json').is_file():missing.append(task['key']);continue
        done=read(score/'complete.json')
        require(done['status']=='SCORED_COMPLETE' and done.get('truth_last') is True,'Invalid independent score completion')
        paths=['metrics.json']+(['common_query_metrics.json'] if '/MIX' in task['capsule'] else [])
        if not all((score/p).is_file() for p in paths):missing.append(task['key']);continue
        complete+=1
    if missing:return dict(status='NOT_COMPLETE',expected_tasks=len(matrix),completed_tasks=complete,missing_tasks=len(missing),missing_task_sample=missing[:20],artifacts_created=False)
    out=Path(output);require(not out.exists(),'Analysis output already exists; use a new output directory')
    aggregate=Aggregates();resources={};row_count=0;common_rows=0;mix_units=0
    for task in matrix:
        folder=Path(task_runtime(rt,task)['run_root']);rx=task['capsule'].split('/')[-1];score=folder/'scores'/rx
        expected={row_key(r) for r in desired_rows(task['points'])};p=read(score/'metrics.json');done=read(score/'complete.json')
        require(done['rows']==len(expected),'Wrong score completion row count')
        natural=_validate_rows(p,task,expected);row_count+=len(natural)
        # Metadata only: no prediction arrays, received IQ, or truth-side files.
        index=read(folder/'predictions'/rx/'predictions.json')
        require(index['capsule']==task['capsule'] and index['capsule_id']==p['capsule_id'] and index['model_seed']==task['seed'],'Score/prediction unit mismatch')
        require(len(index['rows'])==len(expected) and {row_key(r) for r in index['rows']}==expected,'Prediction row matrix mismatch')
        for row in natural.values():aggregate.add_row('natural',task,row)
        aggregate.add_pairs('natural',task,natural)
        if '/MIX' in task['capsule']:
            mix_units+=1;common=read(score/'common_query_metrics.json')
            require(common.get('truth_last') is True and common.get('target_feedback') is False,'Wrong common-query score state')
            values=_validate_rows(common,task,expected,natural);counts=common['common_query_counts']
            require(set(counts)=={str(n) for _,_,_,n,_ in expected},'Common query N coverage differs')
            for row in values.values():
                require(row['metrics']['count']==counts[str(row['new_count'])],'Common query counts differ across K')
                aggregate.add_row('common_query',task,row)
            aggregate.add_pairs('common_query',task,values);common_rows+=len(values)
        _resources(folder/'predictions'/rx,resources)
    rows,classes,pairs=aggregate.result()
    measured=[]
    for key,bucket in sorted(resources.items(),key=lambda x:str(x[0])):
        arm,stage,sp,qp=key
        measured.append(dict(arm=arm,stage=stage,support_path=sp,query_path=qp,measurements=bucket['count'],actual_sums=bucket['sums'],observed_peaks=bucket['peaks'],null_fields=sorted(bucket['null_fields'])))
    main=lambda r:r['stratum']=='all' and r['value']=='MIX' and r['stage']=='C' and r['k']==5 and r['new_count']==20
    summary=dict(status='ANALYSIS_COMPLETE',scope='MIX_ONLY' if rt.get('mix_only') is True else 'GROUP1_FULL',run_id=rt.get('run_id'),completed_tasks=len(matrix),natural_score_rows=row_count,common_query_score_rows=common_rows,
        MIX_units_per_KN=mix_units,source_cost=_source_cost(rt),row_aggregates=rows,paired_comparisons=pairs,class_recall=classes,
        main_K5N20=dict(row_aggregates=[r for r in rows if main(r)],paired_comparisons=[r for r in pairs if main(r)]),
        resources=dict(actual_measurements_only=True,reused_rows_missing_costs_not_zero=True,shared_GPU_not_isolated=True,MACs=None,FLOPs=None,
            measurement_scope='Only actual existing Measure records; sums describe newly measured execution, not complete inherited deployment cost. State-byte sums are across fits, not simultaneous memory peaks.',groups=measured),
        conventions=dict(metric_units='proportions; paired differences in percentage points',unit_weight='equal capsule/seed units within each stratum',
            seed_SD='sample SD of the available per-seed unit means; four original seeds in the full balanced matrix',
            H_aggregation='mean of per-unit macro H; not H recomputed from the averaged old/new components',threshold_tolerance_pp=1e-10,
            A0='shared frozen original default head, represented once under the LocalRidge arm',
            natural_common_separate=True,not_blind_candidate_selection=True,no_truth_or_query_arrays_read=True,
            goal_scope='K5N20 main result and all MIX K/N reported; no unmeasured or missing result is inferred',
            fusion_comparisons='Fusion vs either original input is cross-input; matched_original only compares identical S/Q inputs'))
    out.mkdir(parents=True,exist_ok=False);write(out/'summary.json',summary)
    def flat_rows():
        for row in rows:
            metadata={k:v for k,v in row.items() if k!='metrics'}
            for m,s in row['metrics'].items():yield _flat(dict(metadata,metric=m),s)
    fields=['dataset','stratum','value','arm','stage','k','new_count','input_path','metric','n','mean','unit_sample_SD','seed_count','seed_mean','seed_sample_SD']
    _csv(out/'summary.csv',flat_rows(),fields)
    def flat_pairs():
        for row in pairs:
            metadata={k:v for k,v in row.items() if k!='delta_pp'}
            for m,s in row['delta_pp'].items():yield _flat(dict(metadata,metric=m),s)
    _csv(out/'paired.csv',flat_pairs(),[k for k in fields if k!='arm']+['comparison','comparator_arm','comparator_path','matched_input','units','primary_metric','unit_wins','unit_nonnegative','unit_at_least_5pp','mean_at_least_5pp'])
    _csv(out/'class_recall.csv',(_flat({k:v for k,v in r.items() if k!='recall'},r['recall']) for r in classes),[k for k in fields if k!='metric']+['class_name'])
    _report(out,summary)
    return dict(status='ANALYSIS_COMPLETE',completed_tasks=len(matrix),natural_score_rows=row_count,output=str(out))


def _fmt(value,scale=100):return '—' if value is None else f'{scale*value:.3f}'


def _report(out,summary):
    lines=['# 冻结 R42 完整评分汇总','',f"已核对 {summary['completed_tasks']} 个任务、{summary['natural_score_rows']} 条自然 query 评分；每个 MIX K/N 包含 {summary['MIX_units_per_KN']} 个等权任务单元。自然 query 与共同 query 结果分别保存，不混合平均。",'',
           '## 主工作点 K5/N20（C 阶段）','', '| query 口径 | 方法 | 路径 | 旧类宏准确率（%） | 新类宏准确率（%） | 宏 H（%） | 四 seed H 均值的样本 SD（百分点） |','|---|---|---|---:|---:|---:|---:|']
    if summary.get('scope') == 'MIX_ONLY':
        lines[0] = '# 冻结 R42 MIX 全 K/N 评分汇总（单场景暂停）'
    for r in summary['main_K5N20']['row_aggregates']:
        m=r['metrics'];h=m['H_macro']
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['input_path']} | {_fmt(m['old_macro_accuracy']['unit_equal_weight']['mean'])} | {_fmt(m['new_macro_accuracy']['unit_equal_weight']['mean'])} | {_fmt(h['unit_equal_weight']['mean'])} | {_fmt(h['seed_mean_statistics']['sample_SD'])} |")
    lines.extend(['','## 主工作点配对差','', '| query 口径 | 对比 | 路径 | ΔH（百分点） | 同输入 | 均值 ≥5pp | 单元胜出／总数 | 单元 ≥5pp |','|---|---|---|---:|---|---|---:|---:|'])
    for r in summary['main_K5N20']['paired_comparisons']:
        d=r['delta_pp']['H_macro']['unit_equal_weight']['mean']
        lines.append(f"| {r['dataset']} | {r['comparison']} | {r['input_path']} vs {r['comparator_path']} | {_fmt(d,1)} | {'是' if r['matched_input'] else '否'} | {'是' if r['mean_at_least_5pp'] else '否'} | {r['unit_wins']}/{r['units']} | {r['unit_at_least_5pp']} |")
    lines.extend(['','融合与原 raw／G-only 的比较均属于输入路径改变，不能作为同输入配对收益。全部 K/N、每 pool、RX、MIX／纯景、逐类和可用 day／scene 细分见 `summary.json`、`summary.csv`、`paired.csv`、`class_recall.csv`。缺项与不适用指标保留为空，不填零。','',
        f"本轮源 NN 新优化更新：{summary['source_cost']['new_optimizer_updates'] if summary['source_cost']['new_optimizer_updates'] is not None else '元数据不足，未知'}。继承源训练的更新和部署成本逐 seed 单列。",'',
        '资源只汇总现有真实 Measure 记录，F、G、源编码器与分类头分别列出；共享 GPU 测量不称为孤立性能，复用成本缺项不当作零。MACs／FLOPs 未实测，保持未知。','',
        '本报告只读取固定评分及运行元数据，没有读取 query IQ、预测数组或 truth 文件。结果属于已曝光的开发评估，不构成盲选证据。'])
    with (out/'report.md').open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=analyze(read(args.runtime),args.output);print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result['status']=='ANALYSIS_COMPLETE' else 2)
