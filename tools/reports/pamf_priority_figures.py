"""Publication figures for the five user-prioritized comparisons; no training or scoring."""
import csv,json,math,shutil,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_priority_data import OUT,OLD,P,LABELS,COLORS,SHORT,write,dump
EN={'监督':'Supervised','普通PL半监督':'Semi-supervised (PL)','未接入DURA':'Without DURA','接入DURA':'With DURA','CLEAN':'Clean','综合星地':'Satellite mean','高郊区':'High suburban','中郊区':'Mid suburban','低郊区':'Low suburban','高城市':'High urban','中城市':'Mid urban','低城市':'Low urban','标准接收机划分':'Main split','接收机划分一':'RX split 1','接收机划分二':'RX split 2','1%标注':'1% labels','5%标注':'5% labels','大规模十类数据':'10-class ManyRx'}
def eng(s):
    if s.startswith('双跨 · '):return 'Unseen date + RX: '+eng(s.split(' · ',1)[1])
    if s in EN:return EN[s]
    if s.endswith('个源接收机'):return s.replace('个源接收机',' source RX')
    if s.startswith('有增强 · '):return 'With augmentation: '+EN.get(s.split(' · ',1)[1],s.split(' · ',1)[1])
    if s.startswith('无增强'):return 'No channel augmentation: target clean'
    if s.startswith('综合星地 · '):return 'Satellite mean: restoration transfer'
    if s.startswith('源接收机数量 · '):return 'Source receiver count: '+EN.get(s.split(' · ')[1],s)
    if s.startswith('标准划分 · '):return 'Main split: clean and six satellite environments'
    return s
def main():
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgb
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    payload=json.loads((P/'weekly_workbook_data.json').read_text(encoding='utf-8'));rec={r['id']:r for r in payload['records']}
    joint_only='--joint-only' in sys.argv
    manifest=json.loads((OUT/'figures/manifest.json').read_text(encoding='utf-8')) if joint_only else []
    specs=[]
    for src in payload['sheets']:
        if not src.get('priority'):continue
        if joint_only:
            sp=dict(src,slug=src['jointSlug'],charts=src['jointCharts'],title=src['title']+'：跨日且跨接收机',test='cross_both')
        else:sp=dict(src,test='cross_rx')
        specs.append(sp)
    if '--lines-only' in sys.argv:specs=[s for s in specs if s['charts'][0]['type']=='line']
    names={s['slug'] for s in specs};manifest=[r for r in manifest if r['name'] not in names]
    def tint(c,ratio=.62):return tuple(1-(1-x)*(1-ratio) for x in to_rgb(c))
    for sp in specs:
        n=len(sp['charts']);nc=2 if n>1 else 1;nr=math.ceil(n/nc);fig,axes=plt.subplots(nr,nc,figsize=(8.6,3.5*nr if n>1 else 3.9),squeeze=False)
        used=[]
        for panel,(ch,ax) in enumerate(zip(sp['charts'],axes.flat)):
            cats=ch['categories'];x=np.arange(len(cats));series=ch['series'];legend=[]
            if ch['type']=='line':
                for j,ss in enumerate(series):
                    rr=[rec[k] if k else None for k in ss['ids']];ys=[r['mean'] if r else np.nan for r in rr];sd=[r['sd'] if r else 0 for r in rr];used.extend(r for r in rr if r)
                    ax.errorbar(range(1,7),ys,yerr=sd,color=COLORS[j],label=ss['name'],linestyle='--' if j==5 else '-',lw=2.5 if j>=4 else 1.35,marker=['o','s','^','v','D','P'][j],markersize=5 if j>=4 else 4,capsize=2,alpha=1 if j>=4 else .86,zorder=4 if j>=4 else 2)
                ax.set(xticks=range(1,7),xlim=(.7,6.3),xlabel='Number of source receivers')
                ax.axvspan(4.9,5.1,color='#E8CA8F',alpha=.23,zorder=0)
                ax.text(5,98,'5: main models, common six targets',ha='right',va='top',fontsize=7,color='#6D542D')
                ax.legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.02),frameon=False)
            else:
                width=.34 if len(series)==2 else .115;offsets=(np.arange(len(series))-(len(series)-1)/2)*width
                for j,ss in enumerate(series):
                    rr=[rec[k] if k else None for k in ss['ids']];used.extend(r for r in rr if r);ys=[r['mean'] if r else np.nan for r in rr];sd=[r['sd'] if r else 0 for r in rr]
                    multi_method=len(series)>2
                    colors=[COLORS[j]]*len(cats) if multi_method else [COLORS[LABELS.index(c)] if c in LABELS else '#718FA8' for c in cats]
                    fills=colors if multi_method or j==1 else [tint(c) for c in colors]
                    ax.bar(x+offsets[j],ys,width=width*.92,yerr=sd,color=fills,edgecolor=colors,linewidth=.65,hatch=None if multi_method or j==1 else '///',error_kw={'elinewidth':.75,'capsize':2},zorder=3)
                    for i,(c,r) in enumerate(zip(cats,rr)):
                        if r is None:ax.text(x[i]+offsets[j],6,'PL missing',rotation=90,color='#A32F45',ha='center',va='bottom',fontsize=8)
                        elif c in ('PAMF','PAMF+DURA') and len(cats)<=6 and not sp['slug'].startswith('DURA_generality'):ax.text(x[i]+offsets[j],r['mean']+r['sd']+2,f"{r['mean']:.1f}",ha='center',va='bottom',fontsize=7,color=colors[i],fontweight='bold')
                    legend.append(Patch(facecolor=COLORS[j],edgecolor=COLORS[j],label=ss['name']) if multi_method else Patch(facecolor='#696969' if j else '#E0E0E0',edgecolor='#777777',hatch=None if j else '///',label=eng(ss['name'])))
                if sp['slug'].startswith('DURA_generality'):
                    for i in range(len(cats)):
                        a,b=(rec[s['ids'][i]] for s in series);ax.text(i,max(a['mean']+a['sd'],b['mean']+b['sd'])+4,f"{b['mean']-a['mean']:+.2f} pp",ha='center',fontsize=8,color='#333333')
                ax.set(xticks=x,xticklabels=[eng(c) for c in cats]);ax.tick_params(axis='x',rotation=18 if len(cats)<=6 else 24)
                for tick,c in zip(ax.get_xticklabels(),cats):
                    if c in ('PAMF','PAMF+DURA'):tick.set_color(COLORS[LABELS.index(c)]);tick.set_fontweight('bold')
                ax.legend(handles=legend,ncol=3 if multi_method else 2,loc='lower left',bbox_to_anchor=(0,1.02),frameon=False)
            ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.grid(axis='y',color='#DBE0E5',lw=.6,zorder=0);ax.set_axisbelow(True)
            if n>1:ax.set_title(f'({chr(97+panel)}) '+eng(ch['title']),loc='left',pad=39,fontsize=10,fontweight='bold')
            elif ch['type']!='line' or sp['test']=='cross_both':ax.set_title(eng(ch['title']),loc='left',pad=40,fontsize=10,fontweight='bold')
        for ax in list(axes.flat)[n:]:ax.axis('off')
        fig.tight_layout(pad=1.15,h_pad=3.0,w_pad=2.4)
        for suffix in ('pdf','svg','png'):fig.savefig(OUT/'figures'/(sp['slug']+'.'+suffix),dpi=300,bbox_inches='tight')
        plt.close(fig)
        unique={r['id']:r for r in used};write(OUT/'tables/figure_sources'/(sp['slug']+'.csv'),unique.values())
        manifest.append(dict(name=sp['slug'],caption=sp['title'],wide=True,panels=n,chart_type='line' if sp['charts'][0]['type']=='line' else 'paired_column' if len(sp['charts'][0]['series'])==2 else 'grouped_column',PAMF_included=True,missing_PL_PAMF=sp['slug'].startswith('main_no_channel'),test=sp['test']))
    dump(OUT/'figures/manifest.json',manifest)
    # Old publication files remain available as supplementary material.
    dest=OUT/'figures/supplement_r20';dest.mkdir(exist_ok=True)
    for src in (OLD/'figures').iterdir():
        if src.is_file() and src.suffix in ('.pdf','.svg','.png','.json'):shutil.copy2(src,dest/src.name)
    print(json.dumps(dict(priority_figures=len(manifest),priority_panels=sum(x['panels'] for x in manifest),previous_standalone_figures_retained=26)),flush=True)
if __name__=='__main__':main()
