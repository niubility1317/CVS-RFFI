"""Plot aggregate-only same-packet source diagnostics;does not read IQ or models."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot(data,output):
    output.mkdir(parents=True,exist_ok=False)
    labels=['Coarse','Endpoint t=0','Path t=.25*','Path t=.50*','Path t=.75*','Euler16 prior','MAP6 output']
    colors=['#84909a','#2474a6','#e5a037','#e5a037','#e5a037','#2474a6','#27866e']
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.4))
    for ax,(name,row) in zip(axes,data['rows'].items()):
        values=[row['coarse_nmse']['mean'],*[v['endpoint_nmse']['mean'] for v in row['teacher_path']],row['free_running_prior_nmse']['mean'],row['full_restored_nmse']['mean']]
        bars=ax.bar(range(7),values,color=colors,width=.72,zorder=3)
        for i in (2,3,4):bars[i].set_hatch('//')
        ax.set_yscale('log');ax.set_ylim(7e-7,.14);ax.set_xticks(range(7),labels,rotation=28,ha='right',fontsize=8)
        ax.set_title(name+'  (sigma0='+str(row['sigma0'])+')',fontsize=12,pad=12)
        ax.set_ylabel('Mean IQ NMSE on the same 128 source packets');ax.grid(axis='y',alpha=.2,zorder=0)
        for x,y in enumerate(values):ax.text(x,y*1.16,f'{y:.2g}',ha='center',va='bottom',fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Low teacher-path error does not establish autonomous recovery',fontsize=15,y=.97)
    fig.text(.5,.895,'Frozen R20 checkpoints | same held-out source IQ and explicit epsilon | zero TX labels / F / target access',ha='center',fontsize=9)
    fig.legend(handles=[Patch(color='#84909a',label='Coarse baseline'),Patch(color='#2474a6',label='Current-IQ-only computation'),Patch(facecolor='#e5a037',hatch='//',label='* Clean information in path; diagnostic only'),Patch(color='#27866e',label='Soft MAP output')],loc='lower center',bbox_to_anchor=(.5,.02),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.065,right=.99,top=.81,bottom=.29,wspace=.22)
    fig.savefig(output/'path_diagnostic.png',dpi=170,facecolor='white');fig.savefig(output/'path_diagnostic.svg',facecolor='white');plt.close(fig)
    svg=output/'path_diagnostic.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plot(json.loads(args.input.read_text(encoding='utf-8')),args.output)
