"""Native Group1 raw-input comparisons; reads fixed score metadata only."""
import argparse
import json
from pathlib import Path
from .io import read, write, require
from .milkr_r42_suite import tasks, task_runtime
from .r42_suite_worker import desired_rows, row_key
from .r42_suite_analysis import METRICS, SeedMoments, _csv, _flat, _source_cost

DA=('protonet_cda','mrior_sda','dadda_sda','twostage','feature_separation_hu2024','poster_ft','radionet_ada_knn')
CIL=('orthogonal','lora_triplet_knn','mopc_hr')
PROFILES=DA+CIL
IGNORED={'issl':'USER_STOPPED','issl_author_behavior':'USER_STOPPED','tweak':'N/A_native_tracks_unbound','csil':'N/A_author_MATLAB_Fisher_unbound'}


def native_rows(profile,episode):
    require(profile in PROFILES,'Not a required native external profile')
    ks=(1,5,10,20) if episode=='MIX01' else (5,)
    rows=[('A0',k,0) for k in ks]
    if profile in DA:rows.extend(('B',k,0) for k in ks)
    else:rows.extend(('C',k,n) for k in ks for n in ((2,5,10,20) if episode=='MIX01' else (20,)))
    return set(rows)


def _native_key(row):return row['stage'],int(row['k']),int(row['new_count'])


def _locations(task):
    _,pool,episode,rx=task['capsule'].split('/')
    return pool,episode,rx


def _scopes(task,k,n):
    pool,episode,rx=_locations(task);scopes=[]
    if episode.startswith('MIX') and (k,n) in ((5,0),(5,20)):scopes.append('main_MIX8_K5N20')
    if episode=='MIX01':scopes.append('curves_MIX01')
    if not episode.startswith('MIX'):scopes.append('diagnostic_'+episode[:2])
    return [(scope,'all',scope) for scope in scopes]+[(scope,'pool',pool) for scope in scopes]+[(scope,'receiver',rx) for scope in scopes]+[(scope,'episode',episode) for scope in scopes]


def _jobs_plan(rt,manifest,matrix):
    jobs=manifest['jobs'] if isinstance(manifest,dict) else manifest
    require(isinstance(jobs,list),'External jobs must be a jobs list or a jobs object')
    units={(t['seed'],t['capsule']):t for t in matrix};plans={p:dict(source={},predict={},score={}) for p in PROFILES};seen={};missing=[]
    for item in jobs:
        profile=item.get('profile')
        if profile not in PROFILES:continue  # ISSL/Tweak/CSIL and PAMF are not required external rows.
        kind=item['kind'];require(kind in ('source','predict','score'),'Unknown external job kind')
        identity=(profile,kind,str(item['output']))
        signature=(item['seed'],item.get('capsule'),item.get('prediction'),item.get('completion'))
        if identity in seen:
            require(seen[identity]==signature,'Conflicting duplicate external job pointer');continue
        seen[identity]=signature;plan=plans[profile];seed=item['seed']
        require(seed in rt['model_seeds'],'External source/F seed outside original four')
        if kind=='source':
            require(seed not in plan['source'],'Ambiguous external source pointer')
            plan['source'][seed]=item
        elif kind=='predict':
            key=(seed,item['capsule']);require(key in units,'External capsule/seed outside suite matrix')
            plan['predict'].setdefault(key,[]).append(item)
        else:
            pointer=(seed,str(item['prediction']));require(pointer not in plan['score'],'Ambiguous score pointer for one prediction directory')
            plan['score'][pointer]=item
    pilot=(rt['model_seeds'][0],'capsules/P01/MIX01/1-1')
    for profile,plan in plans.items():
        for seed in rt['model_seeds']:
            if seed not in plan['source']:missing.append(profile+f':source:{seed}')
        for key in units:
            parts=plan['predict'].get(key,[])
            if not parts:missing.append(profile+':predict:'+units[key]['key']);continue
            expected_fragments=2 if key==pilot else 1
            require(len(parts)==expected_fragments,'Wrong native pilot/full fragment coverage: '+profile+str(key))
            if expected_fragments==2:require({p.get('scope') for p in parts}=={'pilot_main','curve_extra'},'Pilot and curve_extra must form one disjoint unit')
            for part in parts:
                if (part['seed'],str(part['output'])) not in plan['score']:missing.append(profile+':score:'+part['id'])
        prediction_count=sum(map(len,plan['predict'].values()))
        expected_count=len(matrix)+(pilot in units)
        if prediction_count!=expected_count and not any(s.startswith(profile+':predict:') for s in missing):
            raise ValueError('Wrong external prediction directory count: '+profile)
        referenced={(p['seed'],str(p['output'])) for parts in plan['predict'].values() for p in parts}
        require(set(plan['score']).issubset(referenced),'Orphan external score job')
    return plans,missing


def _ready(path,status):
    path=Path(path)
    return path.is_file() and read(path).get('status')==status


def _preflight(rt,matrix,plans,missing):
    missing=list(missing)
    for task in matrix:
        score=Path(task_runtime(rt,task)['run_root'])/'scores'/_locations(task)[2]
        if not _ready(score/'complete.json','SCORED_COMPLETE') or not (score/'metrics.json').is_file():missing.append('R42suite:'+task['key'])
    for profile,plan in plans.items():
        for seed,item in plan['source'].items():
            if not _ready(item['completion'],'GROUP1_SOURCE_READY'):missing.append(profile+f':source_marker:{seed}')
        for key,parts in plan['predict'].items():
            for part in parts:
                directory=Path(part['output']);score=plan['score'].get((part['seed'],str(directory)))
                if not _ready(part['completion'],'PREDICTIONS_COMPLETE') or not (directory/'predictions.json').is_file():missing.append(profile+':prediction_marker:'+part['id'])
                if score is not None and (not _ready(score['completion'],'SCORED_COMPLETE') or not (Path(score['output'])/'metrics.json').is_file()):missing.append(profile+':score_marker:'+score['id'])
    return missing


def _suite_rows(rt,task):
    folder=Path(task_runtime(rt,task)['run_root']);score=folder/'scores'/_locations(task)[2];p=read(score/'metrics.json')
    require(p['model_seed']==task['seed'] and p.get('truth_last') is True,'R42 score seed/truth-last mismatch')
    rows={}
    for r in p['rows']:
        key=row_key(r);require(key not in rows,'Duplicate R42 row');rows[key]=r
        pool,episode,rx=_locations(task)
        require(all(r.get(k)==v for k,v in (('physical_pool',pool),('episode',episode),('receiver',rx),('model_seed',task['seed']))),'Wrong R42 scored unit metadata')
        require(all(m in r['metrics'] for m in METRICS),'Missing R42 metrics')
    expected={row_key(r) for r in desired_rows(task['points'])};require(set(rows)==expected,'Incomplete R42 suite rows')
    index=read(folder/'predictions'/_locations(task)[2]/'predictions.json')
    require(index['capsule']==task['capsule'] and index['model_seed']==task['seed'] and index['capsule_id']==p['capsule_id'],'R42 scored unit binding differs')
    return p['capsule_id'],rows


def _external_rows(plan,profile,task,capsule_id):
    result={};pool,episode,rx=_locations(task)
    for part in plan['predict'][task['seed'],task['capsule']]:
        pred=Path(part['output']);scorejob=plan['score'][task['seed'],str(pred)];score=Path(scorejob['output']);index=read(pred/'predictions.json');payload=read(score/'metrics.json')
        require(index['capsule']==task['capsule'] and index['model_seed']==payload['model_seed']==task['seed'],'External capsule/seed mismatch')
        require(index['capsule_id']==payload['capsule_id']==capsule_id,'External/suite capsule_id mismatch')
        require(index.get('profile',profile)==profile and payload.get('truth_last') is True,'Wrong external profile/truth-last')
        expected={(_native_key(r),r['prediction']):r for r in index['rows']}
        require(len(expected)==len(index['rows']),'Duplicate external prediction row')
        require(read(scorejob['completion'])['rows']==len(expected),'External score completion count differs')
        scored=set()
        for row in payload['rows']:
            key=_native_key(row);binding=(key,row['prediction'])
            require(binding in expected and binding not in scored and key not in result,'Duplicate/unbound external scored row')
            original=expected[binding];scored.add(binding)
            require(row['arm']==profile and row['input_path']=='P0' and row['stage'] in ('A0','B','C'),'External row is not native raw P0')
            require(all(row[k]==original[k] for k in ('split_id','registered_classes','arm','input_path')),'External score/prediction row differs')
            require(row['physical_pool']==pool and row['episode']==episode and row['receiver']==rx and row['model_seed']==task['seed'],'Wrong external scored unit metadata')
            require(all(m in row['metrics'] for m in METRICS),'Missing native metrics; do not fill zero')
            result[key]=(row,pred,score/'metrics.json')
        require(scored==set(expected),'Missing external scored prediction rows')
    require(set(result)==native_rows(profile,episode),'Incomplete or nonnative external stage/work-point matrix')
    return result


def _budget(profile,stage,k,n,row,pred):
    old=6*k if stage=='B' else 0;new=k*n if stage=='C' else 0
    resource=row.get('resources') or {}
    native=resource.get('native_resources') or {}
    for field,expected in (('actual_target_old_labels',old),('actual_target_new_labels',new)):
        for data in (resource,native):
            if data.get(field) is not None:require(data[field]==expected,'Native actual label budget differs: '+profile+field)
    fit=pred/(f'B-K{k}-fit.json' if stage=='B' else f'C-K{k}-N{n}-fit.json' if stage=='C' else f'A0-K{k}.json')
    measured=read(fit) if fit.is_file() else None
    if measured is not None:
        require(measured.get('status')=='COMPLETE','Referenced native resource is not complete')
        for field,expected in (('actual_target_old_labels',old),('actual_target_new_labels',new)):
            if measured.get(field) is not None:require(measured[field]==expected,'Measured actual label budget differs')
    steps=None
    for data in (measured or {},resource,native):
        for field in ('optimizer_steps','native_optimizer_steps'):
            if data.get(field) is not None:steps=data[field];break
        if steps is not None:break
    return dict(actual_old_labels=old,actual_new_labels=new,label_budget_evidence='native stage contract, checked against any recorded actual budget',
                optimizer_steps=steps,resource_pointer=str(fit) if measured is not None else None,resource_metadata_present=measured is not None,
                original_prediction_dir=str(pred),native_input='P0_raw_only',native_DURA_used=False)


def _source_refs(rt,plans):
    result=[]
    for profile,plan in plans.items():
        for seed,item in sorted(plan['source'].items()):
            marker=read(item['completion'])
            require(marker.get('profile')==profile and marker.get('seed',marker.get('model_seed'))==seed,'External source identity differs')
            resource=Path(item['output'])/marker.get('resources','A_resources.json')
            measured=read(resource) if resource.is_file() else None
            result.append(dict(profile=profile,seed=seed,completion=item['completion'],source_output=item['output'],
                resource_pointer=str(resource) if measured is not None else None,optimizer_steps=measured.get('optimizer_steps') if measured else None,
                source_resource_status=measured.get('status') if measured else 'MISSING_METADATA'))
    return result


def analyze(rt,manifest,output):
    matrix=tasks(rt);require(len({(t['seed'],t['capsule']) for t in matrix})==len(matrix),'Duplicate suite task unit')
    plans,missing=_jobs_plan(rt,manifest,matrix);missing=_preflight(rt,matrix,plans,missing)
    if missing:return dict(status='NOT_COMPLETE',missing_count=len(missing),missing_sample=missing[:25],artifacts_created=False)
    out=Path(output);require(not out.exists(),'Output already exists; choose a new directory')
    aggregates={};pairs={};units=[];budgets={};profile_units={p:0 for p in PROFILES};native_count=0
    for task in matrix:
        capsule_id,suite=_suite_rows(rt,task)
        for profile,plan in plans.items():
            external=_external_rows(plan,profile,task,capsule_id);profile_units[profile]+=1
            for (stage,k,n),(row,pred,metrics_file) in external.items():
                native_count+=1;budget=_budget(profile,stage,k,n,row,pred);metric=row['metrics'];primary='H_macro' if stage=='C' else 'old_macro_accuracy'
                require(metric[primary] is not None,'Missing native primary score')
                if stage=='C':require(all(metric[m] is not None for m in METRICS[:5]),'Missing native C components')
                for scope,stratum,value in _scopes(task,k,n):
                    key=(profile,scope,stratum,value,stage,k,n)
                    bucket=aggregates.setdefault(key,{m:SeedMoments() for m in METRICS})
                    for m in METRICS:bucket[m].add(task['seed'],metric[m])
                    budgets.setdefault(key,dict(old_labels=budget['actual_old_labels'],new_labels=budget['actual_new_labels'],optimizer=SeedMoments(),resource_records=0))
                    budgets[key]['optimizer'].add(task['seed'],budget['optimizer_steps']);budgets[key]['resource_records']+=budget['resource_metadata_present']
                if stage=='A0':continue
                for method in ('R42','LocalRidge'):
                    reference=suite[(method,stage,k,n,'S0Q0')]
                    require(row['split_id']==reference['split_id'] and row['registered_classes']==reference['registered_classes'],'External/suite split/classes mismatch')
                    require(metric['count']==reference['metrics']['count'],'External/suite query counts differ')
                    delta={m:100*(reference['metrics'][m]-metric[m]) if reference['metrics'][m] is not None and metric[m] is not None else None for m in METRICS}
                    pair_budget=dict(comparator_actual_old_labels=6*k,comparator_actual_new_labels=n*k if stage=='C' else 0,comparator_optimizer_steps=0,
                        label_equivalent=stage=='B',source_training_equivalent=False)
                    for scope,stratum,value in _scopes(task,k,n):
                        key=(profile,scope,stratum,value,stage,k,n,method);bucket=pairs.setdefault(key,dict(metrics={m:SeedMoments() for m in METRICS},units=0,wins=0,plus5=0))
                        bucket['units']+=1;bucket['wins']+=delta[primary]>1e-10;bucket['plus5']+=delta[primary]>=5-1e-10
                        for m in METRICS:bucket['metrics'][m].add(task['seed'],delta[m])
                    units.append(dict(profile=profile,category='DA' if profile in DA else 'CIL',capsule=task['capsule'],model_seed=task['seed'],
                        stage=stage,k=k,new_count=n,comparator=method,comparator_input='S0Q0',external_input='P0',matched_raw_input=True,
                        primary_metric=primary,delta_pp=delta,unit_win=delta[primary]>1e-10,unit_at_least_5pp=delta[primary]>=5-1e-10,
                        split_id=row['split_id'],metrics_file=str(metrics_file),**budget,**pair_budget))
    rows=[];paired=[]
    for key,stats in sorted(aggregates.items()):
        profile,scope,stratum,value,stage,k,n=key;budget=budgets[key]
        rows.append(dict(profile=profile,category='DA' if profile in DA else 'CIL',scope=scope,stratum=stratum,value=value,stage=stage,k=k,new_count=n,input_path='P0',
            actual_old_labels=budget['old_labels'],actual_new_labels=budget['new_labels'],actual_optimizer_steps=budget['optimizer'].result(),
            resource_record_count=budget['resource_records'],metrics={m:s.result() for m,s in stats.items()}))
    for key,bucket in sorted(pairs.items()):
        profile,scope,stratum,value,stage,k,n,method=key;primary='H_macro' if stage=='C' else 'old_macro_accuracy';stats={m:s.result() for m,s in bucket['metrics'].items()}
        paired.append(dict(profile=profile,scope=scope,stratum=stratum,value=value,stage=stage,k=k,new_count=n,comparator=method,
            matched_raw_input=True,label_equivalent=stage=='B',units=bucket['units'],unit_wins=bucket['wins'],unit_at_least_5pp=bucket['plus5'],
            primary_metric=primary,mean_at_least_5pp=stats[primary]['unit_equal_weight']['mean']>=5-1e-10,delta_pp=stats))
    summary=dict(status='ANALYSIS_COMPLETE',run_id=rt.get('run_id'),profiles=list(PROFILES),profile_units=profile_units,native_score_rows=native_count,
        native_summary=rows,paired_summary=paired,source_resources=_source_refs(rt,plans),R42_source_cost=_source_cost(rt),
        unavailable=dict(issl='USER_STOPPED excluded',tweak='N/A',csil='N/A',DA_C='N/A',CIL_B='N/A'),
        conventions=dict(metric_units='proportions; deltas in percentage points; per-unit equal weighting',seed_SD='sample SD of four per-seed means',
            native_stage='seven DA only B; three CIL only C; all have source A0',CIL_label_budget='0 old + N*K new versus R42/LocalRidge 6*K old + N*K new; not label equivalent',
            DA_label_budget='both use 6*K old target support; source recipes and source metadata permissions differ',
            input='external P0 raw compared only to S0Q0 raw; DURA and fusion not pooled or claimed matched',
            scopes='main: all eight MIX K5/N20; curves: MIX01 only; diagnostics: five episodes each LS/MU/LU at K5/N20',
            absent_metrics='None retained; N/A stages absent, never zero',global_pipeline_status_ignored=True,
            resources='existing native Measure pointers and steps only; missing metadata remains unknown; shared GPU is not isolated',MACs=None,FLOPs=None,
            score_only=True,no_truth_or_query_arrays_read=True))
    out.mkdir(parents=True,exist_ok=False);write(out/'summary.json',summary)
    fields=['profile','category','scope','stratum','value','stage','k','new_count','input_path','metric','n','mean','unit_sample_SD','seed_count','seed_mean','seed_sample_SD','actual_old_labels','actual_new_labels','resource_record_count']
    def flat_native():
        for r in rows:
            meta={k:v for k,v in r.items() if k not in ('metrics','actual_optimizer_steps')}
            for m,s in r['metrics'].items():yield _flat(dict(meta,metric=m),s)
    _csv(out/'native_summary.csv',flat_native(),fields)
    def flat_pairs():
        for r in paired:
            meta={k:v for k,v in r.items() if k!='delta_pp'}
            for m,s in r['delta_pp'].items():yield _flat(dict(meta,metric=m),s)
    _csv(out/'paired_summary.csv',flat_pairs(),['profile','scope','stratum','value','stage','k','new_count','comparator','matched_raw_input','label_equivalent','metric','n','mean','unit_sample_SD','seed_count','seed_mean','seed_sample_SD','units','unit_wins','unit_at_least_5pp','mean_at_least_5pp'])
    def flat_units():
        for r in units:
            meta={k:v for k,v in r.items() if k!='delta_pp'}
            yield dict(meta,**{m+'_delta_pp':v for m,v in r['delta_pp'].items()})
    unitfields=['profile','category','capsule','model_seed','stage','k','new_count','comparator','comparator_input','external_input','matched_raw_input','label_equivalent','source_training_equivalent','primary_metric','unit_win','unit_at_least_5pp','actual_old_labels','actual_new_labels','comparator_actual_old_labels','comparator_actual_new_labels','optimizer_steps','comparator_optimizer_steps','resource_pointer','resource_metadata_present','original_prediction_dir','metrics_file','split_id']+[m+'_delta_pp' for m in METRICS]
    _csv(out/'paired_units.csv',flat_units(),unitfields)
    _report(out,summary)
    return dict(status='ANALYSIS_COMPLETE',profiles=len(PROFILES),matched_units_per_profile=len(matrix),native_score_rows=native_count,output=str(out))


def _report(out,summary):
    lines=['# 原生外部方法与冻结 R42 的 raw 主对比','',
        '只纳入九个既有原生 profile 和 MoPC-HR。ISSL 按用户停止指令排除，Tweak／CSIL 保留 N/A；没有把旧全局队列的 FAILED／BLOCKED 视为其他方法失败。','',
        '## 八组 MIX 主工作点 K5/N20','',
        '| 外部方法 | 原生阶段 | 原生主指标（%） | 四 seed 均值样本 SD（百分点） | R42 raw 配对差（百分点） | LocalRidge raw 配对差（百分点） | 外部旧／新标签 | R42 旧／新标签 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['native_summary']:
        if r['scope']!='main_MIX8_K5N20' or r['stratum']!='all' or r['stage']=='A0':continue
        metric='H_macro' if r['stage']=='C' else 'old_macro_accuracy';s=r['metrics'][metric];differences={}
        for method in ('R42','LocalRidge'):
            p=next(p for p in summary['paired_summary'] if all(p[k]==r[k] for k in ('profile','scope','stratum','value','stage','k','new_count')) and p['comparator']==method)
            differences[method]=p['delta_pp'][metric]['unit_equal_weight']['mean']
        sd=s['seed_mean_statistics']['sample_SD'];sd='—' if sd is None else f'{100*sd:.3f}'
        lines.append(f"| {r['profile']} | {r['stage']} | {100*s['unit_equal_weight']['mean']:.3f} | {sd} | {differences['R42']:.3f} | {differences['LocalRidge']:.3f} | {r['actual_old_labels']}/{r['actual_new_labels']} | {6*r['k']}/{r['new_count']*r['k']} |")
    lines.extend(['',
        'B 阶段使用旧类宏准确率，C 阶段使用宏 H，并另列旧／新类宏准确率、宏 F1 和逐类下限。原生 CIL 使用 0 个旧类目标标签与 N×K 个新类标签；R42／LocalRidge 使用 6×K 个旧类标签与 N×K 个新类标签，不能称为标签等价。DA 的目标旧类预算相同，也不意味着源训练条件等价。','',
        '外部 P0 与 R42／LocalRidge 的 S0Q0 均为原始 received 输入。没有把 DURA-only、融合或恢复 support 路径计入外部同输入收益。原生 DA 的 C、原生 CIL 的 B 均为 N/A，没有填零。','',
        '主 MIX、仅 MIX01 的 K/N 曲线、LS／MU／LU 五组诊断分别汇总；每 pool、RX、episode 与逐单元配对差见 CSV。资源只引用既有真实 JSON 和实际优化步，缺失记录保留未知，不重测，不推测 MACs／FLOPs，共享 GPU 不称孤立测量。','',
        '文件：`summary.json`、`native_summary.csv`、`paired_summary.csv`、`paired_units.csv`。本聚合器只读取固定评分及运行元数据，不读取 truth、预测数组或 query IQ。'])
    with (out/'report.md').open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--external-jobs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=analyze(read(a.runtime),read(a.external_jobs),a.output);print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result['status']=='ANALYSIS_COMPLETE' else 2)
