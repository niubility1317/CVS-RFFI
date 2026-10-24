"""Independent CM replay and complete three-seed summaries after all predictions/scores."""
from collections import defaultdict
import csv
import gzip
from itertools import combinations,product
import json
from pathlib import Path
import statistics
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.analyze_main import metrics
from .plan import SEEDS,RUN

FIELDS=('protocol','group','method','test','readout','view','dimension','stratum','metric')

def csvfile(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(groups):
    for key,values in sorted(groups.items()):
        if set(values)!=set(SEEDS):raise ValueError(('Incomplete actual three-seed cell',key))
        a=[values[s] for s in SEEDS]
        yield dict(zip(FIELDS,key),mean=statistics.mean(a),sd=statistics.stdev(a),minimum=min(a),maximum=max(a))

def aggregate(records,out):
    groups=defaultdict(dict);gains=defaultdict(dict);overall={};CM=netcount=0;logical=0
    with gzip.open(out/'per_seed.csv.gz','xt',encoding='utf-8',newline='') as f,gzip.open(out/'net_corrections.csv.gz','xt',encoding='utf-8',newline='') as nf:
        writer=csv.DictWriter(f,fieldnames=['row_id',*FIELDS,'seed','percent','packets_per_view']);writer.writeheader()
        nw=csv.DictWriter(nf,fieldnames=['row_id','protocol','method','seed','test','readout','view','dimension','stratum','count','repairs','damage','net']);nw.writeheader()
        for rec in records:
            row,score=rec['row'],rec['score'];logical+=1;seed=row['model_seed'];classes=row['classes']
            if score['model_seed']!=seed or not score['truth_last'] or not score['target_feedback_forbidden']:raise ValueError('Score identity/independence differs')
            values={};counts={};correct={}
            for result in score['results']:
                k=tuple(result[n] for n in ('test','readout','view','dimension','stratum'))
                if k in counts:raise ValueError('Duplicate serialized CM')
                mm=metrics(result,classes);CM+=1;counts[k]=result['count'];correct[k]=sum(result['confusion'][i][i] for i in range(classes))
                for metric,v in mm.items():values[(*k,metric)]=100*v
            for p in score['paired_raw']:
                k=tuple(p[n] for n in ('test','readout','view','dimension','stratum'));raw=(k[0],'raw',*k[2:])
                if p['net']!=p['repairs']-p['damage'] or p['net']!=correct[k]-correct[raw] or p['count']!=counts[k]:raise ValueError('Net correction differs from CMs')
                nw.writerow(dict(row_id=row['row_id'],protocol=row['protocol_id'],method=row['method'],seed=seed,**p));netcount+=1
            bases={(t,rd,dim,s,metric) for t,rd,v,dim,s,metric in values}
            for t,rd,dim,s,metric in bases:
                a=[values[t,rd,v,dim,s,metric] for v in ('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')]
                values[t,rd,'six_practical_mean',dim,s,metric]=statistics.mean(a)
                counts[t,rd,'six_practical_mean',dim,s]=counts[t,rd,'clean',dim,s]
            for k,v in values.items():
                t,rd,view,dim,s,metric=k;key=(row['protocol_id'],row['group'],row['method'],t,rd,view,dim,s,metric)
                if seed in groups[key]:raise ValueError('Duplicated F seed in a scientific cell')
                groups[key][seed]=v
                writer.writerow(dict(row_id=row['row_id'],**dict(zip(FIELDS,key)),seed=seed,percent=v,packets_per_view=counts[k[:-1]]))
                if rd!='raw':gains[key][seed]=v-values[t,'raw',view,dim,s,metric]
                if dim=='overall':overall[(row['protocol_id'],row['method'],seed,t,rd,view,metric)]=v
    def save(path,iterator):
        with gzip.open(path,'xt',encoding='utf-8',newline='') as f:
            w=None
            for r in iterator:
                if w is None:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader()
                w.writerow(r)
    save(out/'summary.csv.gz',stats(groups));save(out/'restoration_gain.csv.gz',stats(gains))
    csvfile(out/'summary_overall.csv',[r for r in stats(groups) if r['dimension']=='overall'])
    csvfile(out/'restoration_gain_overall.csv',[r for r in stats(gains) if r['dimension']=='overall'])
    effects=[]
    for t,rd,view,metric in product(('cross_date','cross_rx','cross_both'),('raw','DURA-L','DURA-LU'),
        ('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','six_practical_mean'),('accuracy','macro_f1')):
        cells={bits:{s:overall['F0',('REPL' if bits=='111' else 'supervised' if bits=='110' else 'ABC'+bits),s,t,rd,view,metric] for s in SEEDS}
               for bits in (''.join(x) for x in product('01',repeat=3))}
        for n in range(1,4):
            for chosen in combinations(range(3),n):
                a=[sum(__import__('math').prod(1 if bits[i]=='1' else -1 for i in chosen)*v[s] for bits,v in cells.items())/(2**(3-n)) for s in SEEDS]
                effects.append(dict(effect=''.join('ABC'[i] for i in chosen),test=t,readout=rd,view=view,metric=metric,mean=statistics.mean(a),sd=statistics.stdev(a),positive_seeds=sum(v>0 for v in a)))
        for bit,name in [('011','full_minus_no_L_CRIC'),('101','full_minus_no_L_consistency'),('110','full_minus_no_U')]:
            a=[cells['111'][s]-cells[bit][s] for s in SEEDS];effects.append(dict(effect=name,test=t,readout=rd,view=view,metric=metric,mean=statistics.mean(a),sd=statistics.stdev(a),positive_seeds=sum(v>0 for v in a)))
    csvfile(out/'core_effects.csv',effects)
    crossed=[];diagonal=defaultdict(dict)
    for fs,gs,t,view,metric in product(SEEDS,SEEDS,('cross_date','cross_rx','cross_both'),
        ('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','six_practical_mean'),('accuracy','macro_f1')):
        method='REPL' if gs==SEEDS[0] else 'REPL_Gseed'+str(gs)
        key=('m3-natural',method,fs,t,'DURA-LU',view,metric)
        if key not in overall:continue
        value=overall[key]
        crossed.append(dict(F_seed=fs,G_seed=gs,test=t,view=view,metric=metric,percent=value,diagonal=fs==gs))
        if fs==gs:diagonal[t,view,metric][fs]=value
    if crossed:
        csvfile(out/'m3_crossed_cells.csv',crossed)
        diag=[]
        for (t,v,m),values in sorted(diagonal.items()):
            if set(values)!=set(SEEDS):raise ValueError('Incomplete three diagonal F/G system seeds')
            a=list(values.values());diag.append(dict(test=t,view=v,metric=m,mean=statistics.mean(a),sd=statistics.stdev(a),independent_system_seeds=3))
        csvfile(out/'m3_crossed_diagonal.csv',diag)
    return dict(status='VERIFIED_SERIALIZED_INPUTS',logical_rows=logical,serialized_CM_recomputed=CM,net_corrections_recomputed=netcount,
        three_actual_F_seeds=list(SEEDS),six_scene_within_seed_first=True,fixed_G_variation_not_three_full_systems=True,
        posthoc_historical_clean_based_method_selection_disclosed=True,no_target_feedback_used=True)

def run(rt,jobs):
    root=Path(rt['run_root']);out=root/'analysis';out.mkdir(exist_ok=False)
    scorejobs=[j for j in jobs if j['kind']=='score']
    if len(scorejobs)!=228 or any(read(j['completion'])['status']!='SCORED_COMPLETE' for j in scorejobs):raise ValueError('Complete score matrix required')
    def records():
        with gzip.open(out/'score_inputs.jsonl.gz','xt',encoding='utf-8') as f:
            for job in scorejobs:
                c=read(job['config']);score=read(Path(c['output_root'])/'scores.json')
                row={k:c[k] for k in ('row_id','protocol_id','group','method','model_seed')};row['classes']=len(c['classes'])
                rec=dict(row=row,score=score);f.write(json.dumps(rec)+'\n');yield rec
    verification=aggregate(records(),out)
    sources=read(root/'sources.json');stages={};logical=[];epochs=[]
    for rid,rec in sources.items():
        done=read(Path(rec['source_output'])/'complete.json')
        if done['status']!='SOURCE_FROZEN' or done['updates']!=11100 or done['target_access']:raise ValueError('Incomplete F model')
        logical.append(dict(row_id=rid,reused_F=rec['reused'],cumulative_worker_seconds=sum(s['seconds'] for s in done['segments'])))
        for s in done['segments']:
            folder=s['folder']
            if folder in stages:continue
            new=s['config']['run_id']==RUN
            stages[folder]=dict(folder=folder.replace(rt['project'],'PROJECT'),new_training=new,epochs=s['epoch'],updates=s['updates'],shared_worker_seconds=s['seconds'])
            path=Path(folder)/'metrics_epoch.jsonl'
            if path.exists():
                for line in path.read_text().splitlines():epochs.append(dict(folder=folder.replace(rt['project'],'PROJECT'),new_training=new,epoch=json.loads(line)))
    csvfile(out/'F_logical_cost.csv',logical);csvfile(out/'F_unique_stage_cost.csv',list(stages.values()))
    with gzip.open(out/'F_epoch_inputs.jsonl.gz','xt',encoding='utf-8') as f:
        for e in epochs:f.write(json.dumps(e)+'\n')
    actual=sum(s['epochs'] for s in stages.values() if s['new_training'])
    if actual!=8670:raise ValueError(('Unique training budget differs',actual))
    verification.update(status='VERIFIED_ALL_228_ROWS',new_unique_model_epochs=actual,new_unique_updates=actual*222,new_unique_shared_worker_hours=sum(s['shared_worker_seconds'] for s in stages.values() if s['new_training'])/3600,
        new_G_training=0,new_restored_IQ_caches=0,historical_G_costs='retain previously verified ancestry; not zero historical cost',timing_case_rounds=1175)
    write(out/'verification.json',verification)
    text=['# PAMF-REPL：完整实验统计','', '201 个 PAMF 逻辑模型读出、21 个已有 G 组合读出和 6 个三源 F/G 交叉补充读出均已完成独立评分。',
        '本轮不重新训练 DURA；复用既有完整 G 谱系和 22 个 IQ 缓存。三 F 种子通常条件于固定 G，三源交叉的三对角才是三个完整 F/G 训练重复，九格不是九次独立重复。',
        'REPL 曾按用户指定的已公开 F0 clean 成绩选择，明确记录这一选择背景，不称独立盲选。各项负结果、缺项和零伪标签接受均保留。','',
        'summary_overall.csv 与完整压缩表给出三类泛化、raw/G-L/G-LU、七景、各 RX/TX/day 的 Accuracy/Macro-F1。core_effects.csv 给出监督 CRIC、监督一致性与完整 REPL 无标签模块的三组主效应、交互和完整条件移除差。',
        '原生 EMA、相邻/强视图门控已不属于当前 REPL，不伪造其消融；内部分别检验风格、梯度保护、融合对比、外推。结构在完整 REPL 与严格监督 CE 两层分别报告；严格 CE 的曝光不同，不与 000 多视图 CE 骨架混称。','',
        f"新增唯一训练 {actual} 模型轮 / {actual*222} 更新；共享 worker elapsed {verification['new_unique_shared_worker_hours']:.6f} 小时，不是独占 GPU 小时或墙钟。复用历史阶段单列并按文件夹去重。",'',
        '四个独占计时面板共 1175 case-round：REPL 九结构450、严格监督九结构450、监督/普通PL控制100、七种已有G与REPL的组合175。没有重测纯G。F为eager，GF为既定图执行；五round不是五个seed，共驻留不称单模型显存。','']
    (out/'results.md').write_text('\n'.join(text),encoding='utf-8')
    write(root/'analysis_complete.json',dict(status='ANALYZED',verification=verification,output_root=str(out)))
