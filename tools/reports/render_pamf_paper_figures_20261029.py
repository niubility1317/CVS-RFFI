"""Author only the PAMF framework diagram; experiments use selected vector artwork."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--catalog',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--only-figure',type=int,choices=[1],default=1);a=ap.parse_args()
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
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()
