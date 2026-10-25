"""One explicit reporting model: learning regime, training augmentation, inference route."""
import csv,gzip,json,math,statistics,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_pl_comparison import OUT,PIDS,SHORT,TASKS,TN,SCENES,SN,SEEDS,rows,write
REGIMES=('supervised','semi_supervised')
RN={'supervised':'监督','semi_supervised':'普通PL半监督'}
EN={'supervised':'Supervised','semi_supervised':'Semi-supervised (ordinary PL)'}
METHODS=('riei','drift','ibrff','sdg')
NAMES=dict(zip(METHODS,('RIEI','DRIFT','IB-RFF','SDG')))
LABELS=['RIEI','DRIFT','IB-RFF','SDG','PAMF','DURA+PAMF']
VIEW=dict(SN,clean='CLEAN',six_practical_mean='六种practical平均')
TEST={'cross_date':'源接收机跨日诊断','cross_rx':'目标跨接收机','cross_both':'目标日期与接收机双跨'}
METRIC={'accuracy':'准确率','macro_f1':'Macro-F1'}
def mode(r):return 'supervised' if r['method']=='supervised' or r['cell'] in ('A','C') else 'semi_supervised'
def settings(regime):return ('C','supervised','DURA-L') if regime=='supervised' else ('D','ordinary_PL','DURA-LU')
def panels(regime):
    cell,own,rd=settings(regime)
    return [(m,cell,'raw',NAMES[m]) for m in METHODS]+[(own,'native','raw','PAMF'),(own,'native',rd,'DURA+PAMF')]
class Results:
    def __init__(self):
        self.source=list(rows(OUT/'tables/complete_summary.csv'))
        self.index={tuple(r[k] for k in ('protocol','method','cell','test','readout','view','metric')):r for r in self.source}
        self.seed={tuple(r[k] for k in ('protocol','method','cell','test','readout','view','metric'))+(int(r['seed']),):float(r['percent']) for r in rows(OUT/'tables/per_seed_overall.csv.gz')}
    def get(self,p,m,cell,t='cross_rx',rd='raw',view='six_practical_mean',metric='accuracy'):
        return self.index[p,m,cell,t,rd,view,metric]
    def main(self,regime,p,t='cross_rx',view='six_practical_mean',metric='accuracy'):
        return [dict(self.get(p,m,cell,t,rd,view,metric),display_method=label,learning_regime=regime,learning_display=RN[regime]) for m,cell,rd,label in panels(regime)]
    def gain(self,regime,p,m,t='cross_rx',view='six_practical_mean',metric='accuracy'):
        cell,own,rd=settings(regime);cell='native' if m==own else cell
        before=self.get(p,m,cell,t,'raw',view,metric);after=self.get(p,m,cell,t,rd,view,metric)
        a=[self.seed[p,m,cell,t,rd,view,metric,s]-self.seed[p,m,cell,t,'raw',view,metric,s] for s in SEEDS]
        return dict(protocol=p,method=m,display_method=NAMES.get(m,'PAMF'),learning_regime=regime,learning_display=RN[regime],cell=cell,test=t,view=view,metric=metric,readout_before='raw',readout_after=rd,before_mean=before['mean'],before_sd=before['sd'],after_mean=after['mean'],after_sd=after['sd'],mean=statistics.mean(a),sd=statistics.stdev(a),positive_seeds=sum(v>0 for v in a),seed_values=';'.join(map(str,a)))
def refresh():
    r=Results();main=[];external=[];own=[]
    for regime in REGIMES:
        for p in PIDS:
            for t in TASKS:
                for view in (*SCENES,'six_practical_mean'):
                    for metric in METRIC:
                        main.extend(r.main(regime,p,t,view,metric))
                        external.extend(r.gain(regime,p,m,t,view,metric) for m in METHODS)
                        own.append(r.gain(regime,p,settings(regime)[1],t,view,metric))
    write(OUT/'tables/main_comparison.csv',main)
    write(OUT/'tables/DURA_generality.csv',external)
    write(OUT/'tables/PAMF_restoration_by_regime.csv',own)
    grouped=list(rows(OUT/'tables/experiment_groups.csv'))
    for g in grouped:
        reg='supervised' if g['method']=='supervised' or g['cell'] in ('A','C') else 'semi_supervised'
        is_own=g['method'] in ('supervised','ordinary_PL')
        g.update(learning_regime=reg,learning_display=RN[reg],main_comparison=str(g['cell'] in ('C','D','native')),section=reg+('_PAMF_main' if is_own else '_external_channel_on' if g['cell'] in ('C','D') else '_external_no_channel_target_CLEAN_only'),DURA_readout_section='PAMF_system_and_restoration_control' if is_own else 'generality_only',DURA_main_source_IQ='labeled_source_only' if reg=='supervised' else 'all_permitted_source_train_IQ')
    write(OUT/'tables/experiment_groups.csv',grouped)
    v=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    v.update(logical_main_models=330,main_display_system_rows=396,main_models_per_learning_regime=165,main_display_rows_per_learning_regime=198,main_regimes=RN,reporting_policy='Supervised and ordinary-PL semi-supervised comparisons separated first; training augmentation separated second; inference routes explicit; no-channel target CLEAN only')
    (OUT/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='REGIMES_SEPARATED',main_rows=len(main),external_DURA_rows=len(external),PAMF_restoration_rows=len(own),unique_main_classifiers=330,logical_all_classifiers=len(grouped)),ensure_ascii=False),flush=True)
if __name__=='__main__':refresh()
