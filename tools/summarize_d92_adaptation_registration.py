"""Analysis-only A/B/C adaptation and registration report for complete matrices.

No fit, prediction, truth access, run-state mutation or experiment decision.
Both completion/source/physical-identity barriers precede score-file access.
"""
import argparse
from collections import defaultdict
import csv
from itertools import product
import json
import math
from pathlib import Path, PurePosixPath

KEY = ('model_seed','receiver','scenario','k','new_count','support_seed')
METHODS = {'D92-BranchLocalRidge-v1':'branch_local_ridge',
           'D92-BranchLocalMargin-v1':'branch_local_margin'}
IDENTITY_SCHEMA = 'd92_adaptation_registration_identity_v1'
MEASURES = ('A_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy',
            'adaptation_gain_pp','registration_old_drop_pp','old_new_gap_pp','C_harmonic_mean')
CLAIM = ('Descriptive analysis of previously scored fixed target matrices, not independent generalization. '
         'B-A includes enabling target support and replacing the frozen DG classifier with the method head; '
         'it does not isolate parameter updates. B and C are independently fitted from their declared support; '
         'state inheritance is neither assumed nor required. The 10/1/3 pp ideals do not authorize, promote, '
         'reject, select, tune or rerun an experiment.')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def matrix(spec):
    conf, data = spec['confirmation'], spec['data']
    method = conf['candidate_method']
    check(method in METHODS and conf['candidate_folder'] == METHODS[method], 'Unsupported candidate method/folder')
    check(conf['candidate_predictor'] == 'evaluate_d92_'+METHODS[method]+'.py'
          and conf['candidate_mode'] == 'd92_'+METHODS[method]+'_registration', 'Candidate inference identity mismatch')
    result = dict(model_seed=[r['seeds']['model'] for r in spec['rows']], receiver=data['target_receivers'],
        scenario=data['scenarios'], k=data['k'], new_count=data['new_class_counts'], support_seed=data['support_seeds'])
    for key, values in result.items():
        check(isinstance(values,list) and values and len(values)==len(set(values)), 'Invalid or duplicate matrix axis: '+key)
        check(all(isinstance(v,str) and v for v in values) if key in ('receiver','scenario')
              else all(type(v) is int and v >= (0 if key=='new_count' else 1) for v in values), 'Invalid axis value: '+key)
    check(sorted(result['model_seed'])==list(range(2026092701,2026092705))
          and len(result['scenario'])==3 and len(result['support_seed'])==5
          and result['k']==[1,5,10,20] and result['new_count']==[0,2,5,10,20]
          and len(result['receiver']) in (1,3), 'Complete frozen matrix required for descriptive report')
    count=math.prod(len(v) for v in result.values()); dg=4*len(result['receiver'])*3
    for name,value in (('expected_split_count',count//4),('splits_per_model',count//4),
                       ('model_rows',4),('predictions_total',2*count+dg)):
        if name in conf: check(conf[name]==value, 'Registered matrix count mismatch: '+name)
    old=conf['old_classes']
    check(len(old)==len(set(old))==6 and all(isinstance(v,str) and v for v in old), 'Six unique old physical classes required')
    check(data['capsule_id']==conf['reuse_validated_capsule_id'], 'Capsule binding mismatch')
    root=PurePosixPath(spec['execution']['remote_run_root'])
    check(root.is_absolute() and '..' not in root.parts and root.name==spec['run_id'], 'Registered run root mismatch')
    return result


def bind_specs(specs):
    check(len(specs)==2, 'Both complete rx3/rx1 cohorts required')
    matrices=[matrix(s) for s in specs]
    first,second=matrices
    check(sorted(len(m['receiver']) for m in matrices)==[1,3]
          and len(set(first['receiver']+second['receiver']))==4, 'Distinct complete receiver cohorts required')
    check(all(first[k]==second[k] for k in KEY if k!='receiver'), 'Cohort matrix mismatch')
    check(specs[0]['confirmation']['candidate_method']==specs[1]['confirmation']['candidate_method'], 'Cohorts use different methods')
    check(specs[0]['confirmation']['old_classes']==specs[1]['confirmation']['old_classes'], 'Cohort old-class contract mismatch')
    check(len({s['run_id'] for s in specs})==2 and len({s['data']['capsule_id'] for s in specs})==2,
          'Cohort run/capsule identities must be distinct')
    source_maps=[]
    for spec in specs:
        rows=spec['rows'];check(len({r['row_id'] for r in rows})==4, 'Duplicate model row identity')
        sources={}
        for row in rows:
            sha=row['expected_checkpoint_sha256']
            check(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha), 'Invalid checkpoint identity')
            for key in ('row_id','source_root','reuse_row_root'):
                check(isinstance(row[key],str) and bool(row[key]), 'Missing model source binding: '+key)
            sources[row['seeds']['model']]=(row['row_id'],row['source_root'],sha)
        source_maps.append(sources)
    check(source_maps[0]==source_maps[1], 'Cohort model/source/checkpoint mismatch')
    return matrices


def bind_identities(specs, identities):
    """Metadata-only validation; caller has not opened score files yet."""
    from d92_adaptation_registration_pairing import validate_pairs
    matrices=bind_specs(specs)
    check(len(identities)==2, 'Two identity metadata inputs required')
    bound=[]
    for spec,identity,axes in zip(specs,identities,matrices):
        check(identity['schema']==IDENTITY_SCHEMA and identity['run_id']==spec['run_id']
              and identity['capsule_id']==spec['data']['capsule_id']
              and identity['candidate_method']==spec['confirmation']['candidate_method'], 'Identity input run/capsule/method mismatch')
        records=identity['models']; models={r['model_seed']:r for r in records}
        check(len(records)==len(models)==4 and set(models)==set(axes['model_seed']), 'Identity model coverage mismatch')
        expected=set(product(axes['receiver'],axes['scenario'],axes['k'],axes['support_seed']))
        paired={}
        for row in spec['rows']:
            seed=row['seeds']['model']; record=models[seed]
            for field,source in (('row_id','row_id'),('source_root','source_root'),
                                 ('reuse_row_root','reuse_row_root'),('checkpoint_sha256','expected_checkpoint_sha256')):
                check(record[field]==row[source], 'Identity model source mismatch: '+field)
            for item in record['splits']+record['frozen_dg']:
                check(item['capsule_id']==identity['capsule_id'], 'Identity split capsule mismatch')
            pairs=validate_pairs(splits=record['splits'],frozen_dg=record['frozen_dg'],
                old_classes=spec['confirmation']['old_classes'],expected_new_counts=axes['new_count'],
                expected_episode_keys=expected,model_seed=seed,checkpoint_sha256=record['checkpoint_sha256'])
            for pair in pairs:
                key=(seed,pair['receiver'],pair['scenario'],pair['k'],pair['new_count'],pair['support_seed'])
                check(key not in paired, 'Duplicate physical A/B/C identity')
                paired[key]=pair
        check(set(paired)==set(product(*(axes[k] for k in KEY))), 'Incomplete physical A/B/C matrix')
        bound.append(paired)
    return matrices,bound


def score_records(spec, data, axes, pairs):
    """Bind all record identities before any metric aggregation."""
    check(data['status']=='SCORED' and data['selection_feedback_forbidden'] is True, 'Scores are not complete/frozen')
    method=spec['confirmation']['candidate_method']; expected=set(pairs)
    expected_dg=set(product(axes['model_seed'],axes['receiver'],axes['scenario']))
    rows=data['results']; candidate={}; dg={}; baseline={}
    models={r['seeds']['model']:r for r in spec['rows']}
    for row in rows:
        check(row['model_seed'] in models, 'Unregistered score model')
        source=models[row['model_seed']]
        check(row['row_id']==source['row_id'], 'Score row/model mismatch')
        for key,value in (('checkpoint_sha256',source['expected_checkpoint_sha256']),('capsule_id',spec['data']['capsule_id'])):
            if key in row: check(row[key]==value, 'Score source binding mismatch: '+key)
        if row['method']=='frozen_dg':
            key=tuple(row[k] for k in ('model_seed','receiver','scenario'))
            check(key in expected_dg and key not in dg and row['new_count']==0, 'Duplicate/unbound frozen DG record')
            dg[key]=row
        else:
            check(row['method'] in (method,'D92'), 'Unexpected score method')
            key=tuple(row[k] for k in KEY); target=candidate if row['method']==method else baseline
            check(key in expected and key not in target, 'Duplicate/unbound score cell')
            target[key]=row
    check(set(candidate)==set(baseline)==expected and set(dg)==expected_dg
          and len(rows)==2*len(expected)+len(expected_dg), 'Incomplete score matrix')
    for key,pair in pairs.items():
        seed,rx,scene,k,new,support=key
        bkey=(seed,rx,scene,k,0,support)
        for prefix,row in (('a',dg[(seed,rx,scene)]),('b',candidate[bkey]),('c',candidate[key])):
            check(row['split_id']==pair[prefix+'_split_id'] and row['classes']==pair[prefix+'_classes']
                  and row['class_count']==len(row['classes']) and row['query_count']==pair[prefix+'_query_count'],
                  'Physical '+prefix.upper()+' score identity mismatch')
        original=baseline[key]
        check(all(original[field]==candidate[key][field] for field in ('row_id','split_id','classes','class_count','query_count')),
              'Frozen original D92/candidate split identity mismatch')
    return candidate,dg


def accuracy(value, name):
    check(type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1, 'Invalid accuracy: '+name)
    return value


def ideal_description(values):
    gain,drop,gap=(values[k] for k in ('adaptation_gain_pp','registration_old_drop_pp','old_new_gap_pp'))
    distance=dict(adaptation_shortfall_pp=max(0.,10.-gain),registration_drop_excess_pp=max(0.,drop-1.),
                  old_new_gap_excess_pp=None if gap is None else max(0.,gap-3.))
    status=dict(adaptation_at_least_10pp=gain>=10.-1e-12,registration_drop_at_most_1pp=drop<=1.+1e-12,
                old_new_gap_at_most_3pp=None if gap is None else gap<=3.+1e-12,
                all_three_met=None if gap is None else gain>=10.-1e-12 and drop<=1.+1e-12 and gap<=3.+1e-12)
    return dict(descriptive_ideal_distance_pp=distance,descriptive_ideal_status=status)


def cell_measures(a,b,c,new_count):
    av,bv,cv=(accuracy(r['old_accuracy'],name) for r,name in ((a,'Aold'),(b,'Bold'),(c,'Cold')))
    check(a['new_accuracy'] is a['harmonic_mean'] is b['new_accuracy'] is b['harmonic_mean'] is None,
          'A/B old-only metrics must have null new/H')
    if new_count:
        nv=accuracy(c['new_accuracy'],'Cnew'); h=accuracy(c['harmonic_mean'],'C H')
        expected=2*cv*nv/(cv+nv) if cv+nv else 0.
        check(math.isclose(h,expected,rel_tol=1e-10,abs_tol=1e-12), 'C harmonic mean disagrees with old/new accuracy')
        gap=100*abs(nv-cv)
    else:
        check(c['new_accuracy'] is c['harmonic_mean'] is None, 'Old-only C new/H must be null')
        nv=h=gap=None
    values=dict(A_old_accuracy=av,B_old_accuracy=bv,C_old_accuracy=cv,C_new_accuracy=nv,
        adaptation_gain_pp=100*(bv-av),registration_old_drop_pp=100*(bv-cv),old_new_gap_pp=gap,C_harmonic_mean=h)
    return dict(values,**ideal_description(values))


def aggregate(cells, dimensions=(), joint=False):
    groups=defaultdict(list)
    for row in cells:
        if not joint or row['new_count']>0: groups[tuple(row[k] for k in dimensions)].append(row)
    result=[]
    for key,rows in sorted(groups.items()):
        means={};counts={}
        for name in MEASURES:
            values=[r[name] for r in rows if r[name] is not None]
            means[name]=math.fsum(values)/len(values) if values else None; counts[name]=len(values)
        result.append(dict(zip(dimensions,key),cells=len(rows),defined_cells=counts,**means,**ideal_description(means)))
    return result


def tables(cells):
    return dict(per_k_new=aggregate(cells,('k','new_count')),per_k=aggregate(cells,('k',),True),
        per_k_all=aggregate(cells,('k',)),per_model_k=aggregate(cells,('model_seed','k'),True),
        per_receiver_scene_k=aggregate(cells,('receiver','scenario','k'),True),
        overall_joint=aggregate(cells,joint=True),overall_all=aggregate(cells))


def _analyze_bound(specs,scored,matrices,bound):
    check(len(scored)==2, 'Two complete score inputs required')
    indexed=[score_records(s,d,m,p) for s,d,m,p in zip(specs,scored,matrices,bound)]
    cohorts=[]; all_cells=[]
    for spec,axes,pairs,(candidate,dg) in zip(specs,matrices,bound,indexed):
        cells=[]
        for key in sorted(pairs):
            seed,rx,scene,k,new,support=key
            values=cell_measures(dg[(seed,rx,scene)],candidate[(seed,rx,scene,k,0,support)],candidate[key],new)
            pair=pairs[key]
            cells.append(dict(zip(KEY,key),**values,a_split_id=pair['a_split_id'],b_split_id=pair['b_split_id'],c_split_id=pair['c_split_id']))
        cohorts.append(dict(run_id=spec['run_id'],label='rx3' if len(axes['receiver'])==3 else 'rx1',
            matrix=axes,cells=len(cells),tables=tables(cells)))
        all_cells.extend(cells)
    check(len(all_cells)==4800, 'Full 4800-cell analysis required')
    return dict(status='VERIFIED_DESCRIPTIVE_ADAPTATION_REGISTRATION_ANALYSIS',analysis_only=True,
        method=specs[0]['confirmation']['candidate_method'],claim_scope=CLAIM,
        automatic_promotion=False,selected_candidate=None,selection_feedback_forbidden=True,
        ideals=dict(adaptation_gain_min_pp=10.,registration_old_drop_max_pp=1.,old_new_gap_max_pp=3.),
        aggregation='Equal physical paired-cell means. Combined rx3/rx1 contributes 3:1 by cells. '
            'Gap is mean absolute per-cell gap; H is mean per-cell H. per_k/overall_joint exclude new_count=0; '
            'per_k_all/overall_all include it, with null new/gap/H excluded and defined counts reported. '
            'Ideal distances/status evaluate these displayed means, not a gate or a significance claim.',
        physical_pairing_status='VERIFIED',cohorts=cohorts,combined=dict(cells=4800,tables=tables(all_cells)),cells=all_cells)


def analyze(specs,scored,identities):
    matrices,bound=bind_identities(specs,identities)
    return _analyze_bound(specs,scored,matrices,bound)


def prepare(results,identities):
    check(len(results)==len(identities)==2, 'Two results and two identity metadata paths required')
    folders=[Path(p) for p in results]
    check(len({p.resolve() for p in folders})==2, 'Result inputs must be distinct')
    complete=[read(p/'complete.json') for p in folders]
    check(all(v['status']=='SCORED' for v in complete), 'Both cohorts must be SCORED before score access')
    startups=[read(p/'startup.json') for p in folders];specs=[s['spec'] for s in startups]
    matrices=bind_specs(specs)
    for startup,terminal,spec,axes in zip(startups,complete,specs,matrices):
        count=math.prod(len(v) for v in axes.values()); dg=4*len(axes['receiver'])*3
        check(startup['commit']==terminal['commit'] and terminal['records']==2*count+dg, 'Completion commit/count binding mismatch')
        if 'run_id' in terminal: check(terminal['run_id']==spec['run_id'], 'Completion run identity mismatch')
    metadata=[read(p) for p in identities]
    matrices,bound=bind_identities(specs,metadata)
    scored=[read(p/'scores.json') for p in folders]
    result=_analyze_bound(specs,scored,matrices,bound)
    result.update(input_results=[str(p) for p in folders],input_identities=[str(p) for p in identities])
    return result


def write(result,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for label,group in [('combined_rx4',result['combined'])]+[(c['label'],c) for c in result['cohorts']]:
        folder=out/label;folder.mkdir()
        for name,rows in group['tables'].items():
            with (folder/(name+'.csv')).open('x',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerows({k:'N/A' if v is None else json.dumps(v,sort_keys=True) if isinstance(v,dict) else v
                                 for k,v in row.items()} for row in rows)
    lines=['# 适应与注册 A/B/C 描述性报告','',result['method'],'',CLAIM,'',
        'A 为无 target-support 的 frozen_dg；B 为当前同方法 new_count=0；C 为当前同方法当前注册规模。'
        'B-A 包含分类头替换，不能解释成纯参数更新。B、C 可从各自 support 独立拟合，不要求继承状态。','',
        '10 pp 适应增益、≤1 pp 注册旧类下降、≤3 pp 新旧差仅是理想目标。'
        'descriptive_ideal_status 不改变原 run，不决定启动、晋级、淘汰、调参或重跑。','',
        result['aggregation'],'','| K | 新增类数 | Aold % | Bold % | Cold % | Cnew % | B-A pp | B-C pp | 绝对新旧差 pp | H % |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['combined']['tables']['per_k_new']:
        values=[row['k'],row['new_count']]+['N/A' if row[k] is None else f'{row[k]*(100 if k.endswith("accuracy") or k=="C_harmonic_mean" else 1):.3f}' for k in MEASURES]
        lines.append('| '+' | '.join(map(str,values))+' |')
    lines+=['','新增 0 类时 C=B，新类、绝对新旧差和 H 为 N/A。差值先按物理配对单元计算，再等权平均；'
        '不能把两个 cohort 均值等权平均，不能用平均准确率重新计算 H 或绝对新旧差。',
        '完整逐单元证据、理想距离和 descriptive_ideal_status 在 analysis.json；每 cohort 与总表的 K、K×新增类数等 CSV 同时保存。'
        '复用的模型、support 与 query 不构成独立样本，这份报告不作显著性或独立泛化声明。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',nargs=2,required=True)
    parser.add_argument('--identities',nargs=2,required=True,help='Metadata-only rx3/rx1 identities in matching order')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.output).exists(): raise FileExistsError(args.output)
    result=prepare(args.results,args.identities);write(result,args.output)
    print(json.dumps(dict(status=result['status'],analysis_only=True,automatic_promotion=False)))


if __name__=='__main__':main()
