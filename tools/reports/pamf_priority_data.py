"""Prepare the user's five priority comparisons from frozen, independently scored evidence."""
import collections,csv,gzip,json,math,shutil,statistics,sys,zipfile
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_pl_comparison import PIDS,SHORT,SCENES,SEEDS
OLD=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r20'
OUT=ROOT/'exports/reports/20261026-PAMF-priority-comparisons-r22'
P=ROOT/'local_artifacts/pamf_priority_report_20261026_r21'
FROZEN=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/frozen_statistics'
ARCHIVE=ROOT/'exports/phase1_external_dg/20261024-practical-residual-full2x2-complete-r14.zip'
LABELS=['RIEI','DRIFT','IB-RFF','SDG','PAMF','PAMF+DURA'];METHODS=('riei','drift','ibrff','sdg')
VIEWS=(*SCENES,'six_practical_mean')
VIEW=dict(zip(VIEWS,['CLEAN','高郊区','中郊区','低郊区','高城市','中城市','低城市','综合星地']))
REG={'supervised':'监督','semi_supervised':'普通PL半监督'}
COLORS=['#718FA8','#6B9C97','#9A88AE','#82A36E','#C4364E','#E29525']

def rows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def write(p,data):
    data=list(data);p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open if p.suffix=='.gz' else open)(p,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def extract():
    """Read retained aggregate counts; never open individual predictions, IQ, or target truth."""
    if OUT.exists():raise FileExistsError(OUT)
    (OUT/'tables').mkdir(parents=True);(OUT/'figures').mkdir()
    for n in ('complete_summary.csv','per_seed_overall.csv.gz','data_contracts.csv'):
        q=OUT/'frozen_inputs'/n;q.parent.mkdir(exist_ok=True);shutil.copy2(OLD/'tables'/n,q)
    pamf_receivers=[];natural=None
    with gzip.open(FROZEN/'analysis/score_inputs.jsonl.gz','rt',encoding='utf-8-sig') as f:
        for line in f:
            x=json.loads(line);row=x['row'];p=row['protocol_id'];m=row['method']
            if p=='m1-natural' and m=='ordinary_PL' and natural is None:
                natural={a['stratum']:a['count'] for a in x['score']['results'] if a['test']=='cross_rx' and a['view']=='clean' and a['readout']=='raw' and a['dimension']=='receiver'}
            if p!='F0' or m!='ordinary_PL':continue
            for a in x['score']['results']:
                if a['test']=='cross_rx' and a['dimension']=='receiver' and a['readout'] in ('raw','DURA-LU') and a['view'] in SCENES:
                    cm=a['confusion'];correct=sum(cm[i][i] for i in range(len(cm)));assert sum(map(sum,cm))==a['count']
                    pamf_receivers.append(dict(method='ordinary_PL',seed=row['model_seed'],readout=a['readout'],view=a['view'],receiver=a['stratum'],count=a['count'],correct=correct,source='PAMF frozen score confusion counts'))
    assert natural and len(natural)==6 and set(natural.values())=={18000}
    counts=[r for r in pamf_receivers if r['receiver'] in natural];historical=[];external=[];ext_natural=set()
    with zipfile.ZipFile(ARCHIVE) as z:
        with gzip.open(z.open('results/per_seed.csv.gz'),'rt',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r['protocol']=='m1-natural' and r['family']=='riei' and r['cell']=='D' and r['test']=='cross_rx' and r['dimension']=='receiver' and r['readout']=='raw' and r['view']=='clean':ext_natural.add(r['stratum'])
                if r['protocol']!='F0' or r['test'] not in ('cross_rx','cross_both'):continue
                if r['family']=='pamf' and r['cell']=='A' and r['dimension']=='overall' and r['view']=='clean' and r['readout']=='raw':historical.append(r)
                if r['family'] in METHODS and r['cell']=='D' and r['test']=='cross_rx' and r['dimension']=='receiver' and r['view'] in SCENES and r['readout']=='raw' and r['metric']=='accuracy' and r['stratum'] in natural:
                    n=natural[r['stratum']];correct=round(float(r['percent'])*n/100)
                    assert abs(correct/n*100-float(r['percent']))<1e-9
                    external.append(dict(method=r['family'],seed=int(r['seed']),readout='raw',view=r['view'],receiver=r['stratum'],count=n,correct=correct,source='External frozen receiver accuracy; equalized 18000 packets/RX'))
    assert ext_natural==set(natural) and len(counts)==252 and len(external)==504
    counts.extend(external);write(OUT/'frozen_inputs/m5_common_receiver_counts.csv',counts)
    write(OUT/'frozen_inputs/PAMF_existing_no_channel_supervised.csv',historical)
    dump(OUT/'frozen_inputs/source_scope.json',dict(model_seeds=SEEDS,common_receiver_names=sorted(natural),common_target_indices=[0,2,5,7,9,10],packets_per_receiver_per_scene=18000,common_packets_per_scene=108000,F0_original_targets=7,F0_original_packets=126000,m5_origin='Same-seed F0 main classifiers; six common receivers aggregated from frozen receiver statistics',individual_truth_read=False,IQ_read=False,checkpoint_read=False,new_training=0,new_prediction=0,new_scoring=0))

class Data:
    def __init__(self):
        fields=('protocol','method','cell','test','readout','view','metric','seed')
        self.source={tuple(r[k] for k in fields):float(r['percent']) for r in rows(OUT/'frozen_inputs/per_seed_overall.csv.gz')}
        self.histor={(r['test'],r['metric'],int(r['seed'])):float(r['percent']) for r in rows(OUT/'frozen_inputs/PAMF_existing_no_channel_supervised.csv')}
        grouped=collections.defaultdict(list)
        for r in rows(OUT/'frozen_inputs/m5_common_receiver_counts.csv'):grouped[r['method'],r['readout'],r['view'],int(r['seed'])].append(r)
        self.m5={}
        for k,rs in grouped.items():
            assert len(rs)==6 and sum(int(r['count']) for r in rs)==108000
            self.m5[k]=100*sum(int(r['correct']) for r in rs)/108000
        self.records={}
    def get(self,p,label,reg='semi_supervised',view='six_practical_mean',aug=True,common6=False,test='cross_rx'):
        i=LABELS.index(label);m=METHODS[i] if i<4 else 'supervised' if reg=='supervised' else 'ordinary_PL'
        cell=('C' if reg=='supervised' else 'D') if aug else ('A' if reg=='supervised' else 'B')
        rd='raw' if i<5 else 'DURA-L' if reg=='supervised' else 'DURA-LU'
        scope='six_common_target_receivers' if common6 else 'registered_target_population'
        if not aug and i>=4:
            if reg!='supervised' or i!=4:return None
            a=[self.histor[test,'accuracy',s] for s in SEEDS];scope='existing_matched_supervised_no_channel_recipe'
        elif common6 and p=='F0':
            assert reg=='semi_supervised' and aug and test=='cross_rx'
            a=[statistics.mean(self.m5[m,rd,v,s] for v in SCENES[1:]) if view=='six_practical_mean' else self.m5[m,rd,view,s] for s in SEEDS]
        else:
            if i>=4:cell='native'
            a=[self.source[p,m,cell,test,rd,view,'accuracy',str(s)] for s in SEEDS]
        key='|'.join(map(str,(p,label,reg,view,aug,scope,test)))
        self.records[key]=dict(id=key,protocol=p,method=label,learning_regime=reg,channel_augmentation=aug,readout=rd,test=test,view=view,scope=scope,mean=statistics.mean(a),sd=statistics.stdev(a),seed_values=';'.join(map(str,a)),seed_n=3)
        return self.records[key]

def build():
    d=Data();spec=[]
    def chart(title,categories,series,kind='bar',note=''):
        return dict(title=title,categories=categories,series=series,type=kind,note=note,unit='Accuracy (%)')
    def series(label,records):return dict(name=label,ids=[r['id'] if r else None for r in records])
    def sheet(name,slug,title,purpose,design,charts,pids,notes='',reg='semi_supervised'):
        s=dict(name=name,slug=slug,title=title,question=purpose,design=design,charts=charts,protocols=pids,regime=reg,notes=notes,priority=True);spec.append(s);return s
    off=[series(REG[g],[d.get('F0',l,g,'clean',False) for l in LABELS[:5]]) for g in REG]
    sheet('主1_无增强监督与PL','main_no_channel','无星地增强：监督与普通 PL 的 CLEAN 对比','比较关闭附加星地信道增强时的直接识别','四种外部方法采用匹配无增强监督/PL；PAMF监督使用既有无增强匹配参照，当前普通PL缺项。',[chart('无增强 · 标准划分 · 目标跨接收机 CLEAN',LABELS[:5],off,note='PAMF普通PL缺项；监督PAMF保留既有无增强参照配方，不把两种增强图的差当成纯增强效应。')],['F0'],notes='PAMF半监督缺少匹配无增强结果；留空并标缺，不补零，不以历史复杂半监督替代。',reg='paired_learning')
    ch=[]
    for v in ('clean','six_practical_mean'):
        ch.append(chart('有增强 · '+VIEW[v],LABELS,[series(REG[g],[d.get('F0',l,g,v) for l in LABELS]) for g in REG]))
    sheet('主2_增强监督与PL','main_channel_on','有星地增强：监督与普通 PL 对比','同时报告CLEAN与综合星地，保留单PAMF与PAMF+DURA','每种方法按监督/PL成对；只有PAMF+DURA接入恢复器。监督恢复器仅用有标签源IQ，半监督恢复器可用全部允许源训练IQ。',ch,['F0'],reg='paired_learning')
    for name,slug,title,pids,design in [
        ('主3_接收机划分','receiver_splits','接收机划分：半监督＋增强',list(PIDS[:3]),'三套5源/7目标划分，分别训练匹配模型与恢复器，三模型seed统计。'),
        ('主4_标注比例','label_fraction','标注比例：半监督＋增强',['label01','label05','F0'],'1%/5%/10%源训练标注比例，其余源训练IQ仅通过普通PL使用；源验证数量保持。'),
        ('主5_源数量双柱','source_count_paired','源接收机数量：CLEAN与综合星地',['m1-natural','m2-natural','m3-natural','m4-natural','F0','m6-natural'],'1至6源均以共同6个目标接收机比较；5源使用同种子主实验模型的冻结逐接收机统计，重新汇总共同6目标。')]:
        charts=[]
        for p in pids:
            common=slug=='source_count_paired';ct=SHORT[p] if not common else str(5 if p=='F0' else int(p[1]))+'个源接收机'
            charts.append(chart(ct,LABELS,[series(VIEW[v],[d.get(p,l,view=v,common6=common) for l in LABELS]) for v in ('clean','six_practical_mean')]))
        sheet(name,slug,title,'在每个数据设置内配对比较两类测试条件',design,charts,pids,notes='综合星地为六种practical的种子内均值；CLEAN独立。' if slug!='source_count_paired' else '5源不是新训模型；主实验原7目标成绩保留在主对比与补充表，本图仅汇总共同6目标。')
    scene_series=[series(l,[d.get('F0',l,view=v) for v in SCENES]) for l in LABELS]
    sheet('主6_七场景','seven_scenes','主对比实验：CLEAN与六种星地环境','逐场景展示完整主比较，不以综合均值替代逐景','均为普通PL半监督＋训练信道增强；四种外部方法和单PAMF直接识别，仅PAMF+DURA先恢复。',[chart('标准划分 · 目标跨接收机 · 七场景',[VIEW[v] for v in SCENES],scene_series)],['F0'])
    pids=['m1-natural','m2-natural','m3-natural','m4-natural','F0','m6-natural']
    for v,tail in [('clean','CLEAN'),('six_practical_mean','综合星地')]:
        sheet('主7_源数量_'+tail,'source_count_'+('clean' if v=='clean' else 'practical'),'1至6源接收机：'+tail,'观察源接收机数量和自然训练量同时增长时的结果','折线横轴完整1/2/3/4/5/6；全部使用普通PL＋增强；共同6目标、每景108000包。',[chart('源接收机数量 · '+tail,[str(n) for n in range(1,7)],[series(l,[d.get(p,l,view=v,common6=True) for p in pids]) for l in LABELS],kind='line')],pids,notes='5源沿用主实验的同种子模型，仅对冻结逐接收机计数汇总共同6目标；不重训、不重复预测。样本预算随源数变化，不能只解释为接收机多样性因果效应。')
    before=[];after=[]
    for l in LABELS[:5]:
        before.append(d.get('F0',l));after.append(d.get('F0','PAMF+DURA') if l=='PAMF' else None)
        if l!='PAMF':
            m=METHODS[LABELS.index(l)];a=[d.source['F0',m,'D','cross_rx','DURA-LU','six_practical_mean','accuracy',str(s)] for s in SEEDS]
            k='DURA|'+l;d.records[k]=dict(id=k,protocol='F0',method=l,learning_regime='semi_supervised',channel_augmentation=True,readout='DURA-LU',test='cross_rx',view='six_practical_mean',scope='registered_target_population',mean=statistics.mean(a),sd=statistics.stdev(a),seed_values=';'.join(map(str,a)),seed_n=3);after[-1]=d.records[k]
    sheet('主8_DURA泛用性','DURA_generality','DURA泛用性：五种识别方法均纳入','同一识别器接入恢复器前后综合星地性能变化','全部普通PL＋增强，标准划分、目标跨接收机，固定同一模型和输入；恢复器可用全部允许源训练IQ。',[chart('综合星地 · 未接DURA／接入DURA',LABELS[:5],[series('未接入DURA',before),series('接入DURA',after)])],['F0'],notes='PAMF也纳入泛用性。表中提升按种子内相减计算，不能用两个总体标准差相减。')
    master=list(d.records.values());write(OUT/'tables/priority_summary.csv',master)
    gains=[]
    for l,a,b in zip(LABELS[:5],before,after):
        dif=[y-x for x,y in zip(map(float,a['seed_values'].split(';')),map(float,b['seed_values'].split(';')))];gains.append(dict(method=l,before_mean=a['mean'],after_mean=b['mean'],delta_pp=statistics.mean(dif),delta_sd=statistics.stdev(dif),positive_seeds=sum(v>0 for v in dif),seed_values=';'.join(map(str,dif))))
    write(OUT/'tables/DURA_generality_priority.csv',gains)
    contracts={r['protocol']:r for r in rows(OLD/'tables/data_contracts.csv')}
    oldpayload=json.loads((ROOT/'local_artifacts/pamf_pl_comparison_report_20261025_r20/weekly_workbook_data.json').read_text(encoding='utf-8'))
    rename={s['name']:'补_'+s['name'] for s in oldpayload['sheets']}
    def remap(x):
        if isinstance(x,str):
            for a,b in rename.items():x=x.replace('@num:'+a+':','@num:'+b+':').replace('@copy:'+a+':','@copy:'+b+':')
            return x.replace('DURA+PAMF','PAMF+DURA')
        if isinstance(x,list):return [remap(a) for a in x]
        if isinstance(x,dict):return {k:remap(v) for k,v in x.items()}
        return x
    supplements=[]
    for s in oldpayload['sheets']:
        x=remap(s);x['name']=rename[s['name']];x['priority']=False
        x['explain'][0][1]='补充保留：原r20学习条件分组；主图按本次优先实验重新组织。'
        if s['name']=='实验总览':x['explain'][1][1]='原r20矩阵594模型覆盖；本版另引用既有无增强监督PAMF标准划分3模型。'
        supplements.append(x)
    for reg,tag in REG.items():
        cs=[]
        for l in LABELS[:4]:
            m=METHODS[LABELS.index(l)];cell='C' if reg=='supervised' else 'D';rd='DURA-L' if reg=='supervised' else 'DURA-LU';after=[]
            for p in PIDS:
                a=[d.source[p,m,cell,'cross_rx',rd,'six_practical_mean','accuracy',str(seed)] for seed in SEEDS];k='legacy_DURA|'+reg+'|'+p+'|'+l
                d.records[k]=dict(id=k,protocol=p,method=l,learning_regime=reg,channel_augmentation=True,readout=rd,test='cross_rx',view='six_practical_mean',scope='registered_target_population',mean=statistics.mean(a),sd=statistics.stdev(a),seed_values=';'.join(map(str,a)),seed_n=3);after.append(d.records[k])
            cs.append(chart(l,[SHORT[p] for p in PIDS],[series('未接入DURA',[d.get(p,l,reg) for p in PIDS]),series('接入DURA',after)]))
        supplements.append(dict(name='补_DURA'+tag+'11设置',slug=reg+'_DURA_all_settings',title=tag+'：外部DURA泛用性11设置',question='保留上一版完整11设置的四个方法面板',design='补充资料；本轮主DURA图另含PAMF。这里仍是上一版四方法11设置的完整逐方法对照。',charts=cs,protocols=list(PIDS),regime=reg,notes='完整保留旧面板，优先级低于本轮主图。',priority=False))
    for s in spec+[s for s in supplements if 'charts' in s]:
        s['explain']=[['实验目的',s['question']],['设计与固定项',s['design']],['测试与统计','目标跨接收机测试；三种子先种子内汇总，误差为模型种子样本标准差。综合星地仅平均六种practical，不含CLEAN。'],['模型与恢复','PAMF+DURA表示先DURA恢复、后同一PAMF识别；不增加识别器训练次数。'],['数据日期','训练/源验证为2021年3月8、15、23日；源与目标接收机互斥。'],['来源与缺项','已冻结独立评分结果；本次新增训练、预测、目标真值访问均为0。无增强PAMF普通PL缺项按用户要求留空。']]
        part=[]
        for p in s['protocols']:
            r=contracts[p];common=s['slug'].startswith('source_count');count=108000 if common and p=='F0' else int(r['cross_rx']);target='0, 2, 5, 7, 9, 10' if common and p=='F0' else r['target_Rx'].replace(';',', ')
            part.append([str(5 if p=='F0' else int(p[1]))+'源（主模型）' if common else r['display'],int(r['classes']),r['source_Rx'].replace(';',', '),target,int(r['L_s']),int(r['U_s']),int(r['V']),count])
        s['partitionHeaders']=['数据设置','类别数','源接收机索引','展示目标接收机索引','有标签训练包','无标签训练池包','源验证包','展示测试包/景'];s['partitionData']=part
    master=list(d.records.values());write(OUT/'tables/priority_summary.csv',master)
    payload=dict(sheets=spec+supplements,records=master,labels=LABELS,colors=COLORS,output=str(OUT/'PAMF重点实验与补充数据.xlsx'),supplement_source=str(OLD/'PAMF实验周报数据-监督与半监督分组.xlsx'),missing='PAMF ordinary PL without channel augmentation',source_date='2026-10-26')
    dump(P/'weekly_workbook_data.json',payload);dump(OUT/'chart_plan.json',dict(priority=[dict(name=s['name'],slug=s['slug'],panels=len(s['charts'])) for s in spec],supplement_sheets=len(supplements),missing_current_PAMF_PL_no_channel=True,m5_same_main_models_common_six_targets=True))
    print(json.dumps(dict(priority_sheets=len(spec),priority_panels=sum(len(s['charts']) for s in spec),supplement_sheets=len(supplements),records=len(master)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    {'extract':extract,'build':build}[sys.argv[1]]()
