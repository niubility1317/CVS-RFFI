"""Combine completed frozen scores, preserving seed pairing and population counts."""
from collections import defaultdict
import csv
from pathlib import Path
import statistics
from experiments.cric_dura_comparison.io import read,write
from .plan import VIEWS
from experiments.pamf_no_reference_completion.analyze import normalize_counts


def combine(records):
    if not records: raise ValueError('Empty population')
    size=len(records[0]['confusion']);cm=[[0]*size for _ in range(size)]
    for rec in records:
        if len(rec['confusion'])!=size or any(len(line)!=size for line in rec['confusion']):raise ValueError('Incompatible confusion dimensions')
        for i in range(size):
            for j in range(size):cm[i][j]+=int(rec['confusion'][i][j])
    total=sum(map(sum,cm))
    if total!=sum(r['count'] for r in records) or not total:raise ValueError('Population counts differ')
    f=[]
    for i in range(size):
        den=sum(cm[i])+sum(row[i] for row in cm)
        f.append(2*cm[i][i]/den if den else 0.)
    return dict(count=total,accuracy=sum(cm[i][i] for i in range(size))/total,macro_f1=statistics.mean(f))


def run(rt):
    root=Path(rt['run_root']); own=list(read(root/'sources.json').values());ext=read(root/'external_sources.json')
    if len(own)!=226 or len(ext)!=528:raise ValueError('Complete registered sources required')
    # A fixed six-receiver identity set is derived from the existing m1
    # scored metadata, never by metric value, to report m5 on the same targets.
    anchor=next(v for v in ext if v['row']['row_id']=='m1-natural-riei-A-s2026092701')
    a=read(Path(anchor['scores'])/'scores.json')
    common=sorted({r['stratum'] for r in a['results'] if r['test']=='cross_rx' and r['dimension']=='receiver'})
    if len(common)!=6:raise ValueError('Six fixed target receivers required')
    all_records=[]
    for entry in own+ext:
        r=entry['row']; external='family' in r and r.get('kind')=='external'
        done=read(Path(entry['scores'])/'complete.json');score=read(Path(entry['scores'])/'scores.json')
        if done['status']!='SCORED_COMPLETE' or not score.get('truth_last') or not score.get('target_feedback_forbidden'):
            raise ValueError('Independent completed scores required')
        seed=r['model_seed']
        if score.get('model_seed',seed)!=seed:raise ValueError('Score model seed differs')
        setting=r['protocol_id']; family=r['family'] if external else 'PAMF'
        method={'A':'sup_noaug','B':'pl_noaug','C':'sup','D':'pl'}[r['cell']] if external else r['method']
        structure='full' if external else r['structure']; group='external' if external else r['family']
        views=list(VIEWS) if external else r['views']
        routes=['raw', 'DURA-'+r['G_roles']] if external else r['readouts']
        records=normalize_counts(score['results'])
        for view in views:
            for route in routes:
                selected=[x for x in records if x['test'] in ('cross_rx','cross_both') and x['readout']==route and
                          x['view']==view and x['dimension']=='overall' and x['stratum']=='ALL']
                by={x['test']:x for x in selected}
                if len(selected)!=2 or set(by)!= {'cross_rx','cross_both'}:raise ValueError('Scored population or route missing')
                groups={**by,'pooled':combine(list(by.values()))}
                for test,value in groups.items():
                    for metric in ('accuracy','macro_f1'):
                        all_records.append(dict(row_id=r['row_id'],setting=setting,family=family,method=method,structure=structure,
                            group=group,model_seed=seed,test=test,readout=route,view=view,metric=metric,
                            count=value['count'],percent=100.*value[metric]))
                if setting=='F0':
                    for test in ('cross_rx','cross_both'):
                        xs=[x for x in records if x['test']==test and x['readout']==route and x['view']==view and
                            x['dimension']=='receiver' and x['stratum'] in common]
                        if len(xs)!=6 or {x['stratum'] for x in xs}!=set(common):raise ValueError('Common receiver result missing')
                        groups['common_'+test]=combine(xs)
                    groups['common_pooled']=combine(
                        [x for x in records if x['test'] in ('cross_rx','cross_both') and x['readout']==route and x['view']==view and
                         x['dimension']=='receiver' and x['stratum'] in common])
                    for test in ('common_cross_rx','common_cross_both','common_pooled'):
                        value=groups[test]
                        for metric in ('accuracy','macro_f1'):
                            all_records.append(dict(row_id=r['row_id'],setting='m5-common',family=family,method=method,structure=structure,
                                group=group,model_seed=seed,test=test.removeprefix('common_'),readout=route,view=view,metric=metric,
                                count=value['count'],percent=100.*value[metric]))
    history=Path(rt['release_root'])/'historical_architecture.csv'
    with history.open(encoding='utf-8-sig',newline='') as f: old_architecture=list(csv.DictReader(f))
    if len(old_architecture)!=112 or len({(r['variant'],r['model_seed']) for r in old_architecture})!=112:
        raise ValueError('Existing 14-network eight-seed clean panel required')
    for rec in old_architecture:
        for metric in ('accuracy','macro_f1'):
            all_records.append(dict(row_id=rec['run_id']+'/'+rec['variant']+'-s'+rec['model_seed'],setting='F0',
                family=rec['variant'],method='ce',structure='historic',group='architecture_history',model_seed=int(rec['model_seed']),
                test='pooled',readout='raw',view='clean',metric=metric,count=168000,percent=100.*float(rec[metric])))
    grouped=defaultdict(list)
    for rec in all_records:
        if rec['view']=='clean':continue
        key=tuple((k,v) for k,v in rec.items() if k not in ('view','percent'))
        grouped[key].append(rec['percent'])
    for key,values in grouped.items():
        if len(values)!=6:raise ValueError('Incomplete practical scenes')
        all_records.append(dict(key,view='six_practical_mean',percent=statistics.mean(values)))
    groups=defaultdict(list)
    for rec in all_records:
        key=tuple((k,v) for k,v in rec.items() if k not in ('row_id','model_seed','percent'))
        groups[key].append((rec['model_seed'],rec['percent']))
    summary=[]
    for key,values in groups.items():
        if len(values)!=len({s for s,_ in values}):raise ValueError('Duplicate model seed')
        for scope, subset in [('all',values),('matched_three',[(s,v) for s,v in values if s in (2026092701,2026092702,2026092703)])]:
            if not subset:continue
            vals=[v for s,v in subset]
            summary.append(dict(key,seed_scope=scope,seed_n=len(vals),mean=statistics.mean(vals),sd=statistics.stdev(vals) if len(vals)>1 else 0.))
    index={(r['setting'],r['family'],r['group'],r['method'],r['structure'],r['model_seed'],r['test'],r['readout'],r['view'],r['metric']):r for r in all_records}
    paired=[]
    for rec in all_records:
        if rec['family']=='PAMF' or rec['group']!='external':continue
        k=(rec['setting'],'PAMF','factorial',rec['method'],'full',rec['model_seed'],rec['test'],rec['readout'],rec['view'],rec['metric'])
        if k not in index:continue
        target=index[k]
        paired.append(dict(setting=rec['setting'],method=rec['method'],comparison=rec['family'],model_seed=rec['model_seed'],
                           test=rec['test'],readout=rec['readout'],view=rec['view'],metric=rec['metric'],
                           count=rec['count'],pamf_minus_comparison_pp=target['percent']-rec['percent']))
    structural=[]
    for rec in all_records:
        if rec['group'] not in ('branch','structure_ce') or (rec['group']=='branch' and rec['structure']=='full') or rec['setting']!='F0':continue
        k=('F0','PAMF','branch','ce','full',rec['model_seed'],rec['test'],rec['readout'],rec['view'],rec['metric'])
        if k not in index:raise ValueError('Matched structural baseline missing')
        structural.append(dict(structure=rec['structure'],model_seed=rec['model_seed'],test=rec['test'],view=rec['view'],
                               metric=rec['metric'],count=rec['count'],variant_minus_full_pp=rec['percent']-index[k]['percent']))
    architecture=[]
    for rec in all_records:
        if rec['group']!='architecture_history':continue
        k=('F0','PAMF','architecture','ce','full',rec['model_seed'],'pooled','raw','clean',rec['metric'])
        if k not in index:raise ValueError('Matched E200 architecture seed missing')
        if index[k]['count']!=168000:raise ValueError('E200 paired sample population differs')
        architecture.append(dict(comparison=rec['family'],model_seed=rec['model_seed'],metric=rec['metric'],
                                  count=168000,pamf_minus_comparison_pp=index[k]['percent']-rec['percent']))
    paired_groups=defaultdict(list)
    for rec in paired:
        key=tuple((k,v) for k,v in rec.items() if k not in ('model_seed','pamf_minus_comparison_pp'))
        paired_groups[key].append(rec['pamf_minus_comparison_pp'])
    paired_summary=[dict(key,seed_n=len(vals),mean=statistics.mean(vals),sd=statistics.stdev(vals) if len(vals)>1 else 0.)
                    for key,vals in paired_groups.items()]
    out=Path(rt.get('analysis_root',root/'analysis'));out.mkdir(parents=True,exist_ok=False)
    for name,data in (('per_seed.csv',all_records),('summary.csv',summary),('paired_external.csv',paired),
                      ('paired_external_summary.csv',paired_summary),('paired_structural.csv',structural),('paired_architecture.csv',architecture)):
        with (out/name).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# PAMF 新时域、PA 与频谱统计结构的 Phase1 全条件结果','',
        '模型和方法按固定数据合同与种子配对，主比较保留监督、普通 PL 和完整半监督的实际训练配方差异。'
        '直接识别与恢复后识别分列；不同源接收机数量曲线固定同一组六个目标接收机。'
        '跨接收机和跨日跨接收机的综合 Macro-F1 根据合并混淆矩阵计算。历史 E200 网络只保留已有 clean 综合评分。','',
        '## 标准划分：直接识别、六种信道场景平均','',
        '| 方法 | 训练条件 | 种子数 | 综合准确率（%） |','|---|---|---:|---:|']
    names={'riei':'RIEI','drift':'DRIFT','ibrff':'IB-RFF','sdg':'SDG','PAMF':'PAMF'}
    for item in summary:
        if (item['setting']=='F0' and item['structure']=='full' and item['group'] in ('factorial','external')
                and item['test']=='pooled' and item['readout']=='raw' and item['view']=='six_practical_mean'
                and item['metric']=='accuracy' and item['seed_scope']=='matched_three'):
            lines.append(f"| {names.get(item['family'],item['family'])} | {item['method']} | {item['seed_n']} | {item['mean']:.2f} ± {item['sd']:.2f} |")
    lines += ['', '完整逐种子、逐场景和同种子配对结果见同目录 CSV。',
              '历史数据已用于方法开发；不将该对照宣称为独立盲选。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write(out/'complete.json',dict(status='ANALYZED',pamf_rows=226,external_rows=528,common_receivers=common,
          target_feedback_forbidden=True,truth_last=True,pooled_macro_f1_from_confusion=True))
