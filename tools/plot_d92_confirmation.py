"""Render full paired outcomes without changing the frozen method or its selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def normalize_svg(path):
    from xml.etree import ElementTree
    text=path.read_text(encoding='utf-8')
    clean='\n'.join(line.rstrip() for line in text.splitlines())+'\n'
    before=ElementTree.fromstring(text);after=ElementTree.fromstring(clean)
    assert [(e.tag,dict((k,' '.join(v.split())) for k,v in e.attrib.items())) for e in before.iter()]==[(e.tag,dict((k,' '.join(v.split())) for k,v in e.attrib.items())) for e in after.iter()]
    path.write_text(clean,encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--normalize-only',action='store_true');a=p.parse_args()
    data=json.loads(a.summary.read_text(encoding='utf-8'));out=a.summary.parent
    if a.normalize_only:
        for name in ['old_new_h.svg','delta_all_k_new.svg']:normalize_svg(out/name)
        print('SVG_XML_GEOMETRY_UNCHANGED');return
    methods=data.get('methods',['D92','D92-SCV-v1'])
    matrix=data.get('matrix',dict(model_seed=list(range(4)),receiver=list(range(3)),scenario=list(range(3)),support_seed=list(range(5)),k=[1,5,10,20],new_count=[0,2,5,10,20]))
    shots=matrix['k'];new_counts=[n for n in matrix['new_count'] if n>0]
    if any((out/name).exists() for name in ['old_new_h.png','old_new_h.svg','delta_all_k_new.png','delta_all_k_new.svg']):
        raise FileExistsError('Existing figure; preserve previous artifacts')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    keys=[('old_accuracy','Old accuracy'),('new_accuracy','New accuracy'),('harmonic_mean','Harmonic mean H')]
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharex=True)
    for ax,(metric,title) in zip(axes,keys):
        for method,color,marker in [(methods[0],'#637285','o'),(methods[1],'#be3d35','s')]:
            rows={r['k']:r for r in data['tables']['per_k'] if r['method']==method}
            ax.plot(range(len(shots)),[100*rows[k][metric] for k in shots],marker=marker,color=color,label=method,lw=2)
        ax.set_title(title);ax.set_xticks(range(len(shots)),shots);ax.set_xlabel('Support shots K');ax.set_ylabel('%');ax.grid(axis='y',alpha=.2)
    verdict='passed' if data['preregistered_guard_pass'] else 'not met'
    axes[0].legend(frameon=False);fig.suptitle('Fixed Phase1: preregistered paired criteria '+verdict,fontsize=13)
    fig.text(.5,.015,f"{len(matrix['model_seed'])} model seeds; {len(matrix['receiver'])} RX; {len(matrix['scenario'])} scenarios; {len(matrix['support_seed'])} support draws. Equal cell means; H averaged per episode.",ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(out/'old_new_h.png',dpi=180);fig.savefig(out/'old_new_h.svg');plt.close(fig)
    table=data['tables']['per_k_new'];index={(r['k'],r['new_count'],r['method']):r for r in table}
    fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True)
    arrays=[]
    for metric,_ in keys:
        arrays.append(np.array([[100*(index[k,n,methods[1]][metric]-index[k,n,methods[0]][metric]) for n in new_counts] for k in shots]))
    bound=max(1e-6,max(float(np.abs(v).max()) for v in arrays))
    for ax,values,(_,title) in zip(axes,arrays,keys):
        im=ax.imshow(values,cmap='RdBu',vmin=-bound,vmax=bound)
        ax.set_title('Delta '+title);ax.set_xticks(range(len(new_counts)),new_counts);ax.set_yticks(range(len(shots)),shots);ax.set_xlabel('New classes');ax.set_ylabel('K')
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(v)>.65*bound else 'black',fontsize=9)
    fig.suptitle(f'{methods[1]} minus {methods[0]} (percentage points); all registered class-count cells',fontsize=12)
    fig.colorbar(im,ax=axes,shrink=.7,label='percentage points',fraction=.025,pad=.025)
    fig.savefig(out/'delta_all_k_new.png',dpi=180,bbox_inches='tight');fig.savefig(out/'delta_all_k_new.svg',bbox_inches='tight');plt.close(fig)
    for name in ['old_new_h.svg','delta_all_k_new.svg']:normalize_svg(out/name)
    print(out/'old_new_h.png');print(out/'delta_all_k_new.png')


if __name__=='__main__':main()
