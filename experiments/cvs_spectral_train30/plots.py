"""Render closed identity scores without opening IQ or models."""
import argparse,json
from pathlib import Path
def render(data,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 if data['status']!='ANALYZED':raise ValueError('Closed rows only')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);s=data['summary'];views=list(s['B11']);labels=['Clean','Sub high','Sub mid','Sub low','Urban high','Urban mid','Urban low'];colors={'B00':'#777777','B10':'#0072B2','B01':'#E69F00','B11':'#009E73','FULL_AS':'#AA4499'};fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');xx=np.arange(7)
 ax=axes[0,0]
 for name,color in colors.items():ax.plot(xx,[100*s[name][v]['accuracy'] for v in views],color=color,marker='o',lw=1.6,ls='--' if name=='FULL_AS' else '-',label=name+(' (higher As share)' if name=='FULL_AS' else ''))
 ax.axhline(60,color='.5',ls=':',lw=.8);ax.set_xticks(xx,labels,rotation=25,ha='right');ax.set(title='A  Fixed default-head identity accuracy',ylabel='Accuracy (%)',ylim=(25,100));ax.legend(frameon=False,fontsize=8)
 ax=axes[0,1];names=['B10-B00','B11-B00','B11-B10','B11-B01'];vals=[data['paired'][k]['clean']['delta_pp'] for k in names];ax.bar(names,vals,color=[colors['B10'],colors['B11'],colors['B11'],colors['B11']]);ax.axhline(0,color='.3',lw=.8)
 for i,v in enumerate(vals):ax.annotate(f'{v:+.3f}',(i,v),xytext=(0,4 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
 ax.set(title='B  Clean differences, exact matched factor controls',ylabel='Difference (pp)');ax.margins(y=.3)
 ax=axes[1,0];vals=[data['interaction_pp'][v] for v in views];ax.bar(xx,vals,color='#56B4E9');ax.axhline(0,color='.3',lw=.8);ax.set_xticks(xx,labels,rotation=25,ha='right');ax.set(title='C  Interaction: B11 - B10 - B01 + B00',ylabel='Interaction (pp)')
 for i,v in enumerate(vals):ax.annotate(f'{v:+.3f}',(i,v),xytext=(0,4 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
 ax.margins(y=.3);ax=axes[1,1]
 for j,name in enumerate(colors):
  pos=np.arange(3)+(j-2)*.15;vals=[data['suburban_gaps_pp'][name][v] for v in views[1:4]];ax.bar(pos,vals,width=.14,color=colors[name],label=name)
 ax.axhline(2,color='.3',ls='--',lw=1,label='Exploratory 2 pp gap');ax.set_xticks(range(3),labels[1:4]);ax.set(title='D  Suburban gap to the same model clean',ylabel='Clean minus suburban (pp)');ax.legend(frameon=False,fontsize=8)
 for ax in axes.flat:ax.grid(axis='y',color='.9',lw=.6);ax.set_axisbelow(True)
 fig.suptitle('First unchanged bounded FIR training | full source | E20 + 10 | Ag only E21-30 | seed 2026092701',fontsize=13);fig.savefig(out/'fixed_FIR_E30.png',dpi=180);fig.savefig(out/'fixed_FIR_E30.pdf');plt.close(fig)
 return dict(status='PLOTS_WRITTEN',PNG=1,PDF=1,IQ=0,CP=0,NN=0,updates=0)
def main():
 p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('out',type=Path);a=p.parse_args();print(json.dumps(render(json.loads(a.result.read_text()),a.out)))
if __name__=='__main__':main()
