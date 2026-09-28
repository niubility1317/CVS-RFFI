"""Render full paired outcomes without changing the frozen method or its selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.summary.read_text(encoding='utf-8'));out=a.summary.parent
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    keys=[('old_accuracy','Old accuracy'),('new_accuracy','New accuracy'),('harmonic_mean','Harmonic mean H')]
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharex=True)
    for ax,(metric,title) in zip(axes,keys):
        for method,color,marker in [('D92','#637285','o'),('D92-SCV-v1','#be3d35','s')]:
            rows=[r for r in data['tables']['per_k'] if r['method']==method]
            ax.plot(range(4),[100*r[metric] for r in rows],marker=marker,color=color,label=method,lw=2)
        ax.set_title(title);ax.set_xticks(range(4),[1,5,10,20]);ax.set_xlabel('Support shots K');ax.set_ylabel('%');ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False);fig.suptitle('Fixed Phase1: paired confirmation did not meet the comprehensive goal',fontsize=13)
    fig.text(.5,.015,'4 model seeds; 3 RX; 3 scenarios; 5 support draws. Equal mean across new-class counts. H averaged per episode.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(out/'old_new_h.png',dpi=180);fig.savefig(out/'old_new_h.svg');plt.close(fig)
    table=data['tables']['per_k_new'];index={(r['k'],r['new_count'],r['method']):r for r in table}
    fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True)
    arrays=[]
    for metric,_ in keys:
        arrays.append(np.array([[100*(index[k,n,'D92-SCV-v1'][metric]-index[k,n,'D92'][metric]) for n in [2,5,10,20]] for k in [1,5,10,20]]))
    bound=max(float(np.abs(v).max()) for v in arrays)
    for ax,values,(_,title) in zip(axes,arrays,keys):
        im=ax.imshow(values,cmap='RdBu',vmin=-bound,vmax=bound)
        ax.set_title('Delta '+title);ax.set_xticks(range(4),[2,5,10,20]);ax.set_yticks(range(4),[1,5,10,20]);ax.set_xlabel('New classes');ax.set_ylabel('K')
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(v)>.65*bound else 'black',fontsize=9)
    fig.suptitle('D92-SCV-v1 minus D92 (percentage points); all registered class-count cells',fontsize=12)
    fig.colorbar(im,ax=axes,shrink=.7,label='percentage points',fraction=.025,pad=.025)
    fig.savefig(out/'delta_all_k_new.png',dpi=180,bbox_inches='tight');fig.savefig(out/'delta_all_k_new.svg',bbox_inches='tight');plt.close(fig)
    print(out/'old_new_h.png');print(out/'delta_all_k_new.png')


if __name__=='__main__':main()
