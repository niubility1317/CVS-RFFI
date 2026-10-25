"""Vertical publication bar charts; each performance figure has one learning regime."""
import csv,json,math,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_reporting_model import *
P=ROOT/'local_artifacts/pamf_pl_comparison_report_20261025_r20'
def main():
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    data=json.loads((P/'weekly_workbook_data.json').read_text(encoding='utf-8'));by={s['name']:s for s in data['sheets']};result=Results()
    english=dict(zip([SHORT[p] for p in PIDS],['Standard split','RX split 1','RX split 2','10-class ManyRx','1% labels','5% labels','1 source','2 sources','3 sources','4 sources','6 sources']))
    english.update({'方法':'Method','监督':'Supervised','普通PL半监督':'Semi-supervised (ordinary PL)','CLEAN':'Clean','六景平均':'S6 mean','六景准确率':'S6 accuracy','跨接收机CLEAN':'RX shift: clean','双跨CLEAN':'Joint shift: clean','直接识别':'Direct','未接入DURA':'Without DURA','接入DURA':'With DURA','源接收机跨日诊断':'Date shift\n(source RX)','目标跨接收机':'Target RX shift','目标日期与接收机双跨':'Target joint shift','高郊区':'High S','中郊区':'Mid S','低郊区':'Low S','高城市':'High U','中城市':'Mid U','低城市':'Low U','PAMF常规执行':'PAMF eager','DURA+PAMF优化图':'DURA+PAMF graph','外部无增强':'External: no channel','外部有增强':'External: channel','每组独立模型数':'Models per learning regime','独立模型数':'Classifier models'})
    gray=['#333333','#BBBBBB','#666666','#EEEEEE','#888888','#FFFFFF'];hatches=['','///','\\\\','xx','..','++'];manifest=[]
    def resolve(v):
        if not isinstance(v,str) or not v.startswith('@'):return v
        _,name,row,c=v.split(':');return resolve(by[name]['data'][int(row)-5][ord(c)-65])
    def sd(v):
        if not isinstance(v,str) or not v.startswith('@num:'):return None
        _,name,row,c=v.split(':');sp=by[name];r=sp['data'][int(row)-5]
        if c=='I' and sp['copy']:return r[9]
        if name.startswith('DURA泛用性') and c in ('G','H'):
            p=next(p for p in PIDS if SHORT[p]==r[0]);m=next(m for m in METHODS if NAMES[m]==r[1]);t=next(t for t in TASKS if TEST[t]==r[3]);view=next(k for k,v in VIEW.items() if v==r[4]);metric=next(k for k,v in METRIC.items() if v==r[5]);cell,_,rd=settings(sp['regime']);return float(result.get(p,m,cell,t,'raw' if c=='G' else rd,view,metric)['sd'])
        return None
    def save(fig,slug,caption,records,wide,regime):
        fig.tight_layout(pad=.75)
        for ext in ('pdf','svg','png'):fig.savefig(OUT/'figures'/(slug+'.'+ext),dpi=300)
        plt.close(fig);folder=OUT/'tables/figure_sources';folder.mkdir(exist_ok=True)
        with (folder/(slug+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        manifest.append(dict(name=slug,caption=caption,wide=wide,learning_regime=regime,chart_type='vertical_column'))
    for sp in data['sheets']:
        ch=sp['chart'];slug=ch['slug'];rs=[[resolve(v) for v in r] for r in ch['data']];cats=[english.get(str(r[0]),str(r[0])) for r in rs];labels=[english.get(h,h) for h in ch['headers'][1:]];n=len(labels);wide=n>=5
        fig,ax=plt.subplots(figsize=(7.15 if wide else 3.6,2.9 if wide else 2.65));pos=np.arange(len(rs));width=.78/n;records=[]
        for j,label in enumerate(labels):
            a=[float(r[j+1]) for r in rs];err=[sd(r[j+1]) for r in ch['data']];has=all(x is not None for x in err)
            ax.bar(pos+(j-(n-1)/2)*width,a,yerr=err if has else None,width=width*.92,label=label,color=gray[j%6],edgecolor='black',linewidth=.35,hatch=hatches[j%6],error_kw={'lw':.55,'capsize':1})
            records.extend(dict(experiment=sp['name'],learning_regime=sp['regime'] or 'coverage_only',category=str(r[0]),series=ch['headers'][j+1],mean=v,sd='' if e is None else e,unit=ch['unit']) for r,v,e in zip(rs,a,err))
        ax.set(xticks=pos,xticklabels=cats,ylabel=ch['unit']);ax.set_ylim(0,100 if '%' in ch['unit'] else math.ceil(max(float(r[j]) for r in rs for j in range(1,len(r)))*1.18/5)*5)
        if any(len(c)>13 for c in cats):ax.tick_params(axis='x',rotation=20)
        if n>1:ax.legend(frameon=False,ncol=3 if n>3 else n,loc='lower center',bbox_to_anchor=(.5,1.01),handlelength=1.8,columnspacing=.9)
        if sp['regime']:
            title=EN[sp['regime']]+('; no channel augmentation' if '无增强' in sp['name'] else '')
            ax.set_title(title,loc='left',fontsize=8,pad=30 if n>3 else 19 if n>1 else 8)
        if ch.get('delta'):
            for i,vals in enumerate(rs):
                top=max(float(vals[1])+(sd(ch['data'][i][1]) or 0),float(vals[2])+(sd(ch['data'][i][2]) or 0))
                ax.text(i,top+2,f'{float(vals[2])-float(vals[1]):+.2f} pp',ha='center',va='bottom',fontsize=6.6)
        if 'labels' in slug:ax.set_xlabel('Labeled fraction of source training pool')
        elif 'sources' in slug:ax.set_xlabel('Number of source receivers (tested levels)')
        elif 'timing' in slug:ax.set_xlabel('Batch size')
        caption=ch['title']+'。'+('误差线为三个模型种子的样本标准差。' if any(sd(r[1]) is not None for r in ch['data']) else '覆盖统计或5轮计时不冒称模型种子标准差。')
        save(fig,slug,caption,records,wide,sp['regime'])
    for reg in REGIMES:
        rr=[result.gain(reg,p,m) for p in PIDS for m in METHODS];fig,axes=plt.subplots(2,2,figsize=(7.15,5.25));x=np.arange(11);w=.36
        for ax,m in zip(axes.flat,METHODS):
            a=[next(r for r in rr if r['protocol']==p and r['method']==m) for p in PIDS]
            for j,(mean,sdcol,label) in enumerate([('before_mean','before_sd','Without DURA'),('after_mean','after_sd','With DURA')]):
                ax.bar(x+(j-.5)*w,[float(r[mean]) for r in a],yerr=[float(r[sdcol]) for r in a],width=w*.94,label=label,color=gray[j],edgecolor='black',linewidth=.3,hatch=hatches[j],error_kw={'lw':.4,'capsize':.7})
            ax.set(xticks=x,xticklabels=[english[SHORT[p]] for p in PIDS],ylabel='S6 accuracy (%)',ylim=(0,100));ax.tick_params(axis='x',rotation=55,labelsize=6);ax.set_title(NAMES[m]+' - '+EN[reg],fontsize=7,pad=5);ax.legend(frameon=False,fontsize=6,ncol=2,loc='upper left')
        save(fig,f'{reg}_dura_all_settings',RN[reg]+'组：四种外部方法分面，每个设置为未接DURA/接入DURA双柱；保持同一识别器与输入，误差线为三个模型种子的样本标准差。',rr,True,reg)
    keep={r['name'] for r in manifest}
    for file in (OUT/'figures').iterdir():
        if file.suffix in ('.pdf','.svg','.png') and file.stem not in keep:file.unlink()
    for file in (OUT/'tables/figure_sources').glob('*.csv'):
        if file.stem not in keep:file.unlink()
    (OUT/'figures/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='VERTICAL_FIGURES_WRITTEN',figures=len(manifest),single_regime_per_performance_figure=True),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
