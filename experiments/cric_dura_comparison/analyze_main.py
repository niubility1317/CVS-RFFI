"""Pure-stdlib reproduction of all153 fixed T1/main-control score rows."""
import argparse
from collections import defaultdict
import csv,json,math,statistics
from pathlib import Path

SEEDS=(2026092701,2026092702,2026092703)
PROTOCOLS=('F0','F1','F2','ManyRx','chrono','label01','label05',
           'm1-natural','m2-natural','m3-natural','m4-natural','m6-natural',
           'm1-fixed','m2-fixed','m3-fixed','m4-fixed','m6-fixed')
REGIMES=('native_cric','matched_sat_supervised','matched_sat_plain_pl')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(row,classes):
    cm=row['confusion']
    if len(cm)!=classes or any(len(a)!=classes for a in cm) or any(type(n) is not int or n<0 for a in cm for n in a):
        raise ValueError('Invalid registered-class confusion matrix')
    n=sum(map(sum,cm));correct=sum(cm[i][i] for i in range(classes))
    if not n or n!=row['count']:raise ValueError('CM count mismatch')
    f=[]
    for i in range(classes):
        den=sum(cm[i])+sum(a[i] for a in cm);f.append(2*cm[i][i]/den if den else 0.)
    result=dict(accuracy=correct/n,macro_f1=sum(f)/classes)
    if any(not math.isfinite(row[k]) or abs(row[k]-v)>1e-10 for k,v in result.items()):
        raise ValueError('Stored score differs from CM')
    return result


def moments(values):
    if len(values)!=3:raise ValueError('Exactly three actual F seeds required')
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values),
                minimum=min(values),maximum=max(values),positive_seeds=sum(v>0 for v in values))


def write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate(folder):
    folder=Path(folder);manifest=read(folder/'manifest.json')
    expected={(p,r,s) for p in PROTOCOLS for r in REGIMES for s in SEEDS}
    if len(manifest['rows'])!=153:raise ValueError('Incomplete153 matrix')
    values={};counts={};seen=set();nets=[];cost=[];cm_count=0;stage_count=0
    for item in manifest['rows']:
        rec=read(folder/item['member']);score=rec['score'];done=rec['complete'];cfg=rec['source_config']
        p=item['protocol_id'];seed=item['model_seed'];regime=cfg.get('regime')
        if not regime:
            if p!='F0' or '-native_cric-' not in item['row_id'] or len(rec['source_metadata'])!=3:
                raise ValueError('Only the3 original fixed T1 rows may lack new source config')
            regime='native_cric'
        identity=(p,regime,seed)
        if identity in seen or identity not in expected:raise ValueError('Unexpected/duplicate scientific row')
        seen.add(identity)
        classes=10 if p=='ManyRx' else 6
        if cfg and cfg['num_classes']!=classes:raise ValueError('Registered class set differs')
        if (score['run_id'],score['row_id'],score['model_seed'])!=(item['score_run_id'],item['row_id'],seed):
            raise ValueError('Saved score belongs to another source row')
        if done['status']!='SCORED_COMPLETE' or score['status']!='SCORED' or score.get('truth_last') is not True or score.get('target_feedback_forbidden') is not True:
            raise ValueError('Completed independent truth-last score required')
        readers=('raw','DURA-L','DURA-LU') if regime=='matched_sat_supervised' else ('raw','DURA-LU')
        if set(done['readouts'])!=set(readers) or done['views']!=list(VIEWS):raise ValueError('Registered readout/view matrix differs')
        raw={}
        local={}
        for row in score['results']:
            k=(row['readout'],row['test'],row['view'],row['dimension'],str(row['stratum']))
            if k in local or k[0] not in readers or k[2] not in VIEWS:raise ValueError('Unexpected/duplicate score stratum')
            local[k]=metrics(row,classes);cm_count+=1
            values[(*identity,*k)]=local[k];counts[(*identity,*k)]=row['count']
            if k[0]=='raw':raw[k[1:]]=row
        if set(k[1:] for k in local if k[0]=='raw')!={(t,v,d,s) for t,_,d,s in raw for v in VIEWS}:
            raise ValueError('Missing within-row task/view/stratum cells')
        for reader in readers:
            if {k[1:] for k in local if k[0]==reader}!=set(raw):raise ValueError('Readout stratum matrix incomplete')
        pair_map={(r['readout'],r['test'],r['view'],r['dimension'],str(r['stratum'])):r for r in score['paired_raw']}
        if len(pair_map)!=len(score['paired_raw']):raise ValueError('Duplicate net correction stratum')
        for k in local:
            reader,test,view,dim,stratum=k
            if reader=='raw':continue
            pr=pair_map[k];rr=raw[k[1:]];n=counts[(*identity,*k)]
            selected=next(r for r in score['results'] if (r['readout'],r['test'],r['view'],r['dimension'],str(r['stratum']))==k)
            net=sum(selected['confusion'][i][i]-rr['confusion'][i][i] for i in range(classes))
            raw_correct=sum(rr['confusion'][i][i] for i in range(classes))
            if (pr['count']!=n or n!=rr['count'] or pr['net']!=net or pr['repairs']-pr['damage']!=net
                    or min(pr['repairs'],pr['damage'])<0 or pr['repairs']>n-raw_correct or pr['damage']>raw_correct):
                raise ValueError('Paired corrections differ from CMs')
            nets.append(dict(protocol=p,regime=regime,seed=seed,readout=reader,test=test,view=view,dimension=dim,
                             stratum=stratum,count=n,repairs=pr['repairs'],damage=pr['damage'],net=net))
        metadata=rec['source_metadata'];evidence=metadata.get('complete.json')
        if evidence:
            segments=evidence.get('segments',[])
            if evidence.get('updates')!=11100 or evidence.get('epoch')!=50 or len(segments)!=3 or evidence.get('target_access') is not False:
                raise ValueError('Source T1 trajectory incomplete')
            if [s['updates'] for s in segments]!=[4440,2220,4440] or [s['epoch'] for s in segments]!=[20,10,20]:
                raise ValueError('Source segment budgets differ')
            seconds=sum(s['seconds'] for s in segments);stage_count+=len(segments)
            if abs(seconds-evidence['seconds'])>1e-8:raise ValueError('F cumulative elapsed differs from segments')
            pl=[v for k,v in metadata.items() if k.startswith('segment') and k.endswith('/progress.json')]
            pl_seen=sum(v.get('PL_seen',0) for v in pl);pl_accepted=sum(v.get('PL_accepted',0) for v in pl)
        else:
            segments=list(metadata.values());seconds=sum(s['elapsed_seconds'] for s in segments)
            if sum(s['optimizer_steps'] for s in segments)!=11100 or any(s['target_access'] for s in segments):
                raise ValueError('Original fixed T1 ancestry budget or permissions differ')
            stage_count+=len(segments);pl_seen=pl_accepted=None
        cost.append(dict(protocol=p,regime=regime,seed=seed,new_F=p!='F0' or regime!='native_cric',
                         epochs=50,updates=11100,shared_worker_seconds=seconds,
                         ordinary_PL_seen=pl_seen if regime=='matched_sat_plain_pl' else None,
                         ordinary_PL_accepted=pl_accepted if regime=='matched_sat_plain_pl' else None,
                         cost_scope='F cumulative elapsed under shared load; G ancestry and IQ cache separate'))
    if seen!=expected:raise ValueError('Missing protocol/regime/seed')
    # Each contract has the same physical target cells across all3 methods/seeds.
    for p in PROTOCOLS:
        base={(k[4],k[5],k[6],k[7]):n for k,n in counts.items() if k[:3]==(p,REGIMES[0],SEEDS[0]) and k[3]=='raw'}
        for regime in REGIMES:
            for seed in SEEDS:
                current={(k[4],k[5],k[6],k[7]):n for k,n in counts.items() if k[:3]==(p,regime,seed) and k[3]=='raw'}
                if current!=base:raise ValueError('Method/seed physical target strata differ')
        if p.startswith('m') and p[1].isdigit() and base[('main','clean','overall','ALL')]!=144000:
            raise ValueError('Source-count curve must retain the six-target144000 contract')
    # Six views average within each seed before any across-seed statistic.
    clean=[k for k in values if k[5]=='clean']
    for k in clean:
        rows=[values[(*k[:5],view,*k[6:])] for view in VIEWS[1:]]
        values[(*k[:5],'six_practical_mean',*k[6:])]={name:statistics.mean(r[name] for r in rows) for name in ('accuracy','macro_f1')}
    per=[];groups=defaultdict(dict)
    for k,metric in sorted(values.items()):
        p,regime,seed,reader,test,view,dim,stratum=k
        for name,v in metric.items():
            per.append(dict(protocol=p,regime=regime,seed=seed,readout=reader,test=test,view=view,
                            dimension=dim,stratum=stratum,metric=name,percent=100*v))
            groups[(p,regime,reader,test,view,dim,stratum,name)][seed]=100*v
    fields=('protocol','regime','readout','test','view','dimension','stratum','metric')
    summary=[dict(zip(fields,k),**moments([v[s] for s in SEEDS])) for k,v in sorted(groups.items())]
    paired=[]
    for p in PROTOCOLS:
        for reader in ('raw','DURA-LU'):
            available={k[4:] for k in values if k[:4]==(p,REGIMES[0],SEEDS[0],reader)}
            for condition in sorted(available):
                for name in ('accuracy','macro_f1'):
                    for label,a,b in (('native_minus_supervised',REGIMES[0],REGIMES[1]),
                                      ('native_minus_ordinaryPL',REGIMES[0],REGIMES[2]),
                                      ('ordinaryPL_minus_supervised',REGIMES[2],REGIMES[1])):
                        v=[100*(values[(p,a,s,reader,*condition)][name]-values[(p,b,s,reader,*condition)][name]) for s in SEEDS]
                        paired.append(dict(protocol=p,effect=label,readout=reader,**dict(zip(('test','view','dimension','stratum'),condition)),metric=name,**moments(v)))
        for regime in REGIMES:
            available={k[4:] for k in values if k[:4]==(p,regime,SEEDS[0],'raw')}
            for reader in (('DURA-L','DURA-LU') if regime==REGIMES[1] else ('DURA-LU',)):
                for condition in sorted(available):
                    for name in ('accuracy','macro_f1'):
                        v=[100*(values[(p,regime,s,reader,*condition)][name]-values[(p,regime,s,'raw',*condition)][name]) for s in SEEDS]
                        paired.append(dict(protocol=p,effect=regime+'_G_minus_raw',readout=reader,**dict(zip(('test','view','dimension','stratum'),condition)),metric=name,**moments(v)))
    return dict(per_seed=per,summary=summary,paired=paired,net_corrections=nets,F_cost=cost),dict(
        status='ANALYZED_MAIN153',logical_models=153,protocols=list(PROTOCOLS),regimes=list(REGIMES),F_seeds=list(SEEDS),
        serialized_CM_recomputed=cm_count,net_strata_recomputed=len(nets),source_F_stage_metadata=stage_count,
        new_F_models=150,reused_original_F_models=3,all_updates=11100,source_count_m5_excluded=True,
        classes_by_protocol={p:10 if p=='ManyRx' else 6 for p in PROTOCOLS},
        target_truth_reads=0,score_feedback_used=False,conditional_on_fixed_G_per_contract=True,
        all_F_shared_worker_seconds=sum(r['shared_worker_seconds'] for r in cost),
        new_F_shared_worker_seconds=sum(r['shared_worker_seconds'] for r in cost if r['new_F']))


def run(folder, output):
    data,verification=aggregate(folder);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,rows in data.items():write_csv(out/('main_'+name+'.csv'),rows)
    for name in ('summary','paired'):
        write_csv(out/('main_'+name+'_overall.csv'),[r for r in data[name] if r['dimension']=='overall'])
    lookup={(r['protocol'],r['regime'],r['readout'],r['test'],r['view'],r['metric']):r for r in data['summary'] if r['dimension']=='overall'}
    report=['# 最终T1主比较：17契约×3方法配方×3 Fseed完整153模型读出','',
        '全部冻结预测后独立truth-last评分；旧F0输入绑定坏行由6条修复控制替代。三种子变化条件于同契约固定G，不是三个独立完整系统。',
        'native_cric为固定原生T1；匹配监督/普通PL保留已登记星地拼接0.68/PL0.95。它们比较完整方法配方，不能把差值直接解释为某一个组件的因果贡献。',
        'F0/F1/F2、ManyRx、日期与标签比例契约分列。ManyRx使用其原登记10类，其余契约6类，不混合总体平均。源数量曲线仅m1/2/3/4/6的同六目标RX、每景144000包，不插入F0的五源/七目标168000结果。','',
        '|契约|读出|原生T1 clean %|原生T1六景 %|监督控制六景 %|普通PL六景 %|',
        '|---|---|---:|---:|---:|---:|']
    def fmt(p,r,reader,view):
        row=lookup[p,r,reader,'main',view,'accuracy'];return '%.4f ± %.4f'%(row['mean'],row['sd'])
    for p in PROTOCOLS:
        for reader in ('raw','DURA-LU'):
            report.append('|%s|%s|%s|%s|%s|%s|'%(p,reader,fmt(p,REGIMES[0],reader,'clean'),fmt(p,REGIMES[0],reader,'six_practical_mean'),fmt(p,REGIMES[1],reader,'six_practical_mean'),fmt(p,REGIMES[2],reader,'six_practical_mean')))
    report+=['','所有CSV均保留每景/物理RX/TX/day和chrono附任务的Accuracy、Macro-F1、配对差及净纠错。六景先在seed内平均，再报告三Fseed样本SD，SD不作为显著性检验。',
             '监督F+G-L为完整L-only，监督F+G-LU明确通过G用U。普通PL F+G-L的51补充及其四格公平性仍在独立队列，不能从本面板填造。',
             'F成本累计按完整三段及历史复用分别计；elapsed是共享worker墙钟，不是独占GPU时间，G谱系、共享IQ缓存与部署时延另列。其他结构/细机制/外部DURA/专用时延仍在运行，本面板闭合不等于整体交付。']
    (out/'main_results.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    if (Path(folder)/'crossed_inputs.json').is_file():
        verification['m3_crossed']=crossed(folder,out)
    (out/'main_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification))


def crossed(folder, output):
    folder=Path(folder);out=Path(output);manifest=read(folder/'manifest.json')
    bases={};all_rows={};expected={(f,g) for f in SEEDS for g in SEEDS};cm_count=0
    for item in manifest['rows']:
        if item['protocol_id']!='m3-natural' or item['regime']!='native_cric':continue
        rec=read(folder/item['member']);seed=item['model_seed'];bases[seed]=rec
        all_rows[(seed,SEEDS[0])]=(rec['score'],'DURA-LU')
    if set(bases)!=set(SEEDS):raise ValueError('Three original m3 F readouts required')
    extras=read(folder/'crossed_inputs.json')['rows']
    if len(extras)!=6:raise ValueError('Six predeclared crossed additions required')
    for cell in extras:
        f,g=cell['F_seed'],cell['G_seed'];s=cell['score'];done=cell['complete']
        if (cell['protocol_id']!='m3-natural' or g not in SEEDS[1:] or f not in SEEDS
                or (f,g) in all_rows or s['model_seed']!=f or s['status']!='SCORED'
                or s.get('truth_last') is not True or s.get('target_feedback_forbidden') is not True
                or done['status']!='SCORED_COMPLETE' or done['views']!=list(VIEWS)
                or cell['source_output']!=bases[f]['prediction_config']['source_output']):
            raise ValueError('Crossed row/source identity or freezing differs')
        readers=set(r['readout'] for r in s['results'])-{'raw'}
        if readers!={'DURA-Gseed%d'%g}:raise ValueError('Wrong G seed readout')
        raw=lambda obj:{(r['test'],r['view'],r['dimension'],str(r['stratum'])):(r['count'],r['confusion'])
                        for r in obj['results'] if r['readout']=='raw'}
        if raw(s)!=raw(bases[f]['score']):raise ValueError('Same-F crossed raw matrix changed')
        all_rows[(f,g)]=(s,next(iter(readers)))
    if set(all_rows)!=expected:raise ValueError('Missing3x3 cells')
    values={};per=[]
    for (f,g),(score,reader) in all_rows.items():
        raw={};selected={};pairs={}
        for row in score['results']:
            k=(row['test'],row['view'],row['dimension'],str(row['stratum']))
            metrics(row,6)
            if (f,g) not in {(seed,SEEDS[0]) for seed in SEEDS}:cm_count+=1
            if row['readout']=='raw':raw[k]=row
            if row['readout']==reader:selected[k]=row
        if set(selected)!=set(raw):raise ValueError('Crossed view/strata incomplete')
        pairs={(r['test'],r['view'],r['dimension'],str(r['stratum'])):r for r in score['paired_raw'] if r['readout']==reader}
        for k,row in selected.items():
            net=sum(row['confusion'][i][i]-raw[k]['confusion'][i][i] for i in range(6));p=pairs[k]
            if p['net']!=net or p['repairs']-p['damage']!=net or p['count']!=row['count']:
                raise ValueError('Crossed net corrections differ')
            values[(f,g,*k)]=metrics(row,6)
    for k in [k for k in values if k[3]=='clean']:
        v=[values[(*k[:3],view,*k[4:])] for view in VIEWS[1:]]
        values[(*k[:3],'six_practical_mean',*k[4:])]={name:statistics.mean(r[name] for r in v) for name in ('accuracy','macro_f1')}
    conditions={k[2:] for k in values};summary=[]
    for k,v in sorted(values.items()):
        f,g,test,view,dim,stratum=k
        for metric,value in v.items():per.append(dict(F_seed=f,G_seed=g,test=test,view=view,dimension=dim,stratum=stratum,metric=metric,percent=100*value))
    for condition in sorted(conditions):
        for metric in ('accuracy','macro_f1'):
            for g in SEEDS:
                v=[100*values[(f,g,*condition)][metric] for f in SEEDS]
                summary.append(dict(summary_type='fixed_G_over3F',G_seed=g,**dict(zip(('test','view','dimension','stratum'),condition)),metric=metric,**moments(v)))
            v=[100*values[(s,s,*condition)][metric] for s in SEEDS]
            summary.append(dict(summary_type='three_diagonal_independent_systems',G_seed='diagonal',
                **dict(zip(('test','view','dimension','stratum'),condition)),metric=metric,**moments(v)))
    write_csv(out/'main_crossed_per_cell.csv',per);write_csv(out/'main_crossed_summary.csv',summary)
    write_csv(out/'main_crossed_overall.csv',[r for r in summary if r['dimension']=='overall'])
    return dict(contract='m3-natural',F_seeds=list(SEEDS),G_seeds=list(SEEDS),crossed_cells=9,
                new_score_rows=6,reused_main_rows=3,additional_serialized_CM_recomputed=cm_count,
                same_F_raw_identical=True,diagonal_system_repeats=3,full_grid_not9_independent_repeats=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)
