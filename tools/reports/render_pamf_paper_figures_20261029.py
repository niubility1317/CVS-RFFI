"""Reformat frozen paper statistics; never train, predict, or score samples."""
import argparse,csv,json,statistics,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.4,'axes.labelsize':7.5,'legend.fontsize':6.4,'xtick.labelsize':6.8,'ytick.labelsize':6.8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'RIEI':'#7295ae','DRIFT':'#5e9e99','IB-RFF':'#9984af','SDG':'#80a36b','PAMF':'#c83050','PAMF+DURA':'#e59b23'}
    manifest=[]
    def save(fig,n,sources,rows):
        fig.savefig(out/f'fig{n:02}.pdf',bbox_inches='tight',pad_inches=.025);fig.savefig(out/f'fig{n:02}.png',dpi=220,bbox_inches='tight',pad_inches=.025);plt.close(fig)
        keys=list(dict.fromkeys(k for r in rows for k in r))
        if keys:
            with (out/'data'/f'fig{n:02}.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        manifest.append({'paper_figure':n,'sources':sources,'data':f'data/fig{n:02}.csv' if keys else None})
    fig,ax=plt.subplots(figsize=(7.05,2.65));ax.set(xlim=(0,10),ylim=(0,4.1));ax.axis('off')
    def box(x,y,w,h,t,c='#edf2f5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',edgecolor='#607080',facecolor=c,lw=.7));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7.3)
    def arrow(x,y,u,v):ax.annotate('',(u,v),(x,y),arrowprops={'arrowstyle':'->','lw':.85,'color':'#475764'})
    box(.1,2.6,2.1,1,'Source receivers\nLabeled IQ + unlabeled IQ\nTX labels hidden in U')
    box(2.8,2.6,3.7,1,'PAMF representation + joint cosine head\nSinc / spectrum / PA / reference\nThree local SIG-TIME projections','#f6e7ec');arrow(2.2,3.1,2.75,3.1)
    box(7.05,2.6,2.8,1,'Source-only Tri-SIG-TIME SSL\nCross-RX pairing and perturbation\nLabeled-gradient anchoring');arrow(6.5,3.1,7,3.1)
    box(.1,.4,2.1,1,'Unseen target receiver\nOne received IQ window\nNo target updating')
    box(2.8,.4,2.4,1,'Optional frozen DURA\nCoarse estimate + 2 stages\nNo TX labels in G training','#fbf0d9');arrow(2.2,.9,2.75,.9)
    box(5.8,.4,2.3,1,'Frozen PAMF\nSame local permutation\nOriginal joint head','#f6e7ec');arrow(5.2,.9,5.75,.9)
    box(8.65,.4,1.2,1,'Known TX\nidentity');arrow(8.1,.9,8.6,.9)
    ax.annotate('',(6.9,1.45),(6.9,2.45),arrowprops={'arrowstyle':'->','linestyle':'--','lw':.8});ax.text(6.98,1.95,'freeze',fontsize=6.6)
    ax.plot([2.2,2.45,2.45,6.9,6.9],[.9,.9,1.85,1.85,1.45],ls='--',lw=.7,color='#607080');ax.text(3.35,1.94,'Direct path bypasses DURA',fontsize=6.6)
    save(fig,1,['Current method definition; DURA v1 mechanism'],[])
    arch=read(a.results/'15_architecture_E200_summary.csv'); names=[r['network'] for r in sorted([r for r in arch if r['test']=='cross_both'],key=lambda r:-float(r['accuracy_mean_pct']))]
    fig,axs=plt.subplots(1,2,figsize=(7.05,4.0),sharey=True);used=[]
    for ax,test,title in zip(axs,['cross_rx','cross_both'],['(a) Cross-receiver, clean','(b) Receiver + date holdout, clean']):
        for i,name in enumerate(names):
            rs=[r for r in arch if r['network']==name and r['test']==test]
            if not rs:continue
            r=rs[0];mu=float(r['accuracy_mean_pct']);sd=float(r['accuracy_sd_pp']);c=colors['PAMF'] if 'PAMF' in name else '#6b8798';ax.errorbar(mu,i,xerr=sd,fmt='o',ms=3,c=c,capsize=2,lw=.8);ax.text(min(95,mu+sd+1),i,f'{mu:.2f} ± {sd:.2f}',va='center',fontsize=5.7);used.append(r)
        ax.set(xlim=(0,105),xlabel='Accuracy (%)',title=title);ax.grid(axis='x',alpha=.2)
    axs[0].set_yticks(range(len(names)),names);axs[0].invert_yaxis();fig.tight_layout(w_pad=1.6);save(fig,2,['Selected r3 pp. 1–2; table 15, E200 CE, 8 seeds'],used)
    def linepanel(ax,rows,title,ylim):
        cats=list(dict.fromkeys(r['category'] for r in rows)); series=list(dict.fromkeys(r['series'] for r in rows))
        if all(re.search(r'\d+',c) for c in cats):cats.sort(key=lambda c:int(re.search(r'\d+',c).group()))
        for s in series:
            rs=[next(r for r in rows if r['series']==s and r['category']==c) for c in cats];x=np.arange(len(cats));ax.errorbar(x,[float(r['estimate']) for r in rs],yerr=[float(r.get('sd') or 0) for r in rs],c=colors.get(s,'#617888'),marker='o',ms=3,lw=1.0,capsize=1.6,label=s)
        ax.set_xticks(range(len(cats)),[c.replace('Practical ','').replace('suburban','sub.').replace('urban','urb.') for c in cats],rotation=25,ha='right');ax.set(title=title,ylabel='Accuracy (%)',ylim=ylim);ax.grid(axis='y',alpha=.2)
    # Factorial comparison, six-scene panel. Bars keep an explicit zero baseline.
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.6));used=[]
    for ax,stem,title in zip(axs,['Fig022_learning_augmented_cross_rx','Fig031_learning_augmented_cross_both'],['(a) Cross-receiver','(b) Receiver + date holdout']):
        rows=[r for r in read(a.catalog/'source_data'/f'{stem}_plot.csv') if r['panel']=='b' and r['category']!='PAMF+DURA'];cats=list(dict.fromkeys(r['category'] for r in rows));series=list(dict.fromkeys(r['series'] for r in rows));used.extend(rows)
        for j,s in enumerate(series):
            rs=[next((r for r in rows if r['category']==c and r['series']==s),None) for c in cats];x=np.arange(len(cats))+(j-(len(series)-1)/2)*.23
            ax.bar(x,[float(r['estimate']) if r else np.nan for r in rs],.21,yerr=[float(r.get('sd') or 0) if r else 0 for r in rs],label=s,color=['#d2dce2','#91a8b5','#c83050'][j],capsize=1.3,error_kw={'lw':.6})
        ax.set_xticks(range(len(cats)),cats,rotation=20,ha='right');ax.set(title=title,ylabel='Six-scene accuracy (%)',ylim=(0,100));ax.grid(axis='y',alpha=.2)
    axs[0].legend(loc='upper left',frameon=False,fontsize=5.5);fig.tight_layout();save(fig,3,['Selected r3 pp. 25–26, six-scene panels'],used)
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.8));used=[]
    for ax,stem,title in zip(axs,['Fig025_seven_scenes_cross_rx','Fig034_seven_scenes_cross_both'],['(a) Cross-receiver','(b) Receiver + date holdout']):
        rows=read(a.catalog/'source_data'/f'{stem}_plot.csv');used.extend(rows);linepanel(ax,rows,title,(25,100))
    axs[0].legend(ncol=3,fontsize=5.7,frameon=False,loc='upper center');fig.tight_layout();save(fig,4,['Selected r3 pp. 31–32'],used)
    fig,axs=plt.subplots(2,2,figsize=(7.05,4.0));used=[]
    for ax,stem,title in zip(axs.flat,['Fig023_receiver_robustness_cross_rx','Fig032_receiver_robustness_cross_both','Fig024_label_robustness_cross_rx','Fig033_label_robustness_cross_both'],['(a) Receiver splits, cross-RX','(b) Receiver splits, double holdout','(c) Label fraction, cross-RX','(d) Label fraction, double holdout']):
        rows=[r for r in read(a.catalog/'source_data'/f'{stem}_plot.csv') if r['panel']=='b'];used.extend(rows);linepanel(ax,rows,title,(15,90) if 'label_' in stem else (30,90))
    axs[0,0].legend(ncol=3,fontsize=5.5,frameon=False);fig.tight_layout();save(fig,5,['Selected r3 pp. 27–30, six-scene panels'],used)
    fig,axs=plt.subplots(2,2,figsize=(7.05,3.95));used=[]
    for ax,stem,title,ylim in zip(axs.flat,['Fig026_source_count_clean_cross_rx','Fig035_source_count_clean_cross_both','Fig027_source_count_practical_cross_rx','Fig036_source_count_practical_cross_both'],['(a) Clean, cross-RX','(b) Clean, double holdout','(c) Six scenes, cross-RX','(d) Six scenes, double holdout'],[(25,100),(25,100),(15,90),(15,90)]):
        rows=read(a.catalog/'source_data'/f'{stem}_plot.csv');used.extend(rows);linepanel(ax,rows,title,ylim)
    axs[0,0].legend(ncol=3,fontsize=5.5,frameon=False);fig.tight_layout();save(fig,6,['Selected r3 pp. 33–36; same data as pp. 37–38'],used)
    rows=read(a.catalog/'source_data'/'Fig029B_Fig038_DURA_gains_plot.csv');fig,axs=plt.subplots(1,2,figsize=(7.05,2.8))
    for ax,test,title in zip(axs,['cross_rx','cross_both'],['(a) Cross-receiver','(b) Receiver + date holdout']):
        sub=[r for r in rows if r['test']==test];cats=list(dict.fromkeys(r['category'] for r in sub))
        for i,c in enumerate(cats):
            v=[next(r for r in sub if r['category']==c and r['series']==s) for s in ['Without DURA','With DURA']];mu=[float(r['estimate']) for r in v];ax.bar([i-.16,i+.16],mu,width=.3,color=['#b5c6cf',colors.get(c,'#c83050')],yerr=[float(r['sd']) for r in v],capsize=1.5,error_kw={'lw':.7})
            for x,y in zip([i-.16,i+.16],mu):ax.text(x,y+.65,f'{y:.2f}',ha='center',fontsize=5.3)
            ax.annotate('',(i+.16,mu[1]+3),(i-.16,mu[0]+3),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=-.65','lw':.8,'color':'#486674'});ax.text(i,max(mu)+8,f'+{mu[1]-mu[0]:.2f}',ha='center',fontsize=6)
        ax.set_xticks(range(len(cats)),cats);ax.set(ylim=(0,100),title=title,ylabel='Six-scene accuracy (%)');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,7,['Selected r3 pp. 39–42; use gain-annotated alternative once'],rows)
    variants=['w/o Time coordinate','w/o SIG-TIME','w/o PA branch','w/o Reference response','Phase-randomized reference']
    seeds=read(a.results/'all_overall_per_seed.csv');data=[r for r in seeds if ((r['collection']=='current' and r['method']=='PAMF') or (r['collection']=='historical_repl' and r['method']=='PAMF-CE')) and r['data_setting']=='Main split' and r['scene']=='six_practical_mean' and r['test'] in ['cross_rx','cross_both'] and r['readout']=='raw' and r['dimension']=='overall']
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.9),sharey=True);used=[]
    for ax,method,title in zip(axs,['PAMF-CE','PAMF'],['(a) CE structural control','(b) Complete SSL structural control']):
        for k,test in enumerate(['cross_rx','cross_both']):
            base={r['model_seed']:r for r in data if r['method']==method and r['test']==test and r['variant']=='Full three-view'}
            for i,v in enumerate(variants):
                sub=[r for r in data if r['method']==method and r['test']==test and r['variant']==v];ds=[float(r['accuracy_pct'])-float(base[r['model_seed']]['accuracy_pct']) for r in sub];assert len(ds)==3,(method,test,v,len(ds))
                mu=statistics.mean(ds);sd=statistics.stdev(ds);ax.errorbar(mu,i+(k-.5)*.19,xerr=sd,fmt=['o','s'][k],ms=3,c=['#bc3855','#42818a'][k],capsize=2,lw=.8,label=['Cross-RX','Double holdout'][k] if i==0 else None)
                used.append(dict(method=method,test=test,variant=v,delta_mean_pp=mu,paired_sd_pp=sd,seed_n=3,paired_seed_deltas=json.dumps(dict(zip([r['model_seed'] for r in sub],ds)))))
        ax.axvline(0,c='#7b858e',ls='--',lw=.8);ax.set(xlim=(-10,3),title=title,xlabel='Variant minus full structure (pp)');ax.grid(axis='x',alpha=.2)
    labels=['No time coordinate','No SIG-TIME','No PA branch','No reference','Reference phase shuffle'];axs[0].set_yticks(range(len(labels)),labels);axs[0].invert_yaxis();axs[0].legend(frameon=False);fig.tight_layout();save(fig,8,['Selected r3 pp. 43–50; full paired source-seed records for both holdouts'],used)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figure_count':len(manifest),'output':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
