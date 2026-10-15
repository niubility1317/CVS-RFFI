"""Report an already completed source-capacity calculation; no IQ/model input."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import design as d


def main():
    result=d.read(d.BASE/'results.json');completion=d.read(d.BASE/'complete.json')
    if completion['status']!='ANALYZED' or result['target_access'] is not False:
        raise ValueError('Need completed source-only analysis')
    out=d.BASE/'analysis';out.mkdir(exist_ok=False)
    p=result['primary'];rows=p['profiles']
    flat=[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]
    with (out/'per_fold_rx.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    rx=[]
    for index,label in enumerate([1,3,4,6,8]):
        selected=[r for r in rows if r['RX_index']==index]
        energy=sum(r['predicted_energy'] for r in selected)
        rx.append(dict(RX=label,weighted_direction_coverage=sum(r['direction_coverage']*r['predicted_energy'] for r in selected)/energy,
            observed_MSE_bounded=sum(r['observed_bounded_prediction_SSE'] for r in selected)/sum(r['observed_zero_SSE'] for r in selected),
            observed_MSE_full=sum(r['observed_full_prediction_SSE'] for r in selected)/sum(r['observed_zero_SSE'] for r in selected)))
    summary=dict(verdict=result['verdict'],weighted_direction_coverage=p['weighted_direction_coverage'],
        resolution_coverage=result['resolution_control']['weighted_direction_coverage'],
        resolution_difference=abs(p['weighted_direction_coverage']-result['resolution_control']['weighted_direction_coverage']),
        bounded_mismatch_to_predicted=p['bounded_mismatch_to_predicted'],relative_observed_MSE=p['relative_observed_MSE'],RX=rx,
        elapsed_seconds=completion['elapsed_seconds'],model_forwards=0,optimizer_updates=0,target_access=False)
    d.write(out/'summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    grid=np.array([r['direction_coverage'] for r in rows]).reshape(18,5)
    im=axes[0].imshow(grid*100,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    axes[0].set(xticks=range(5),xticklabels=[1,3,4,6,8],xlabel='Source RX',ylabel='Excluded TX/day fold',
        title='Single-response direction coverage (%)')
    axes[0].set_yticks(range(18),[f'TX{t}/D{day}' for t in range(6) for day in (1,2,3)],fontsize=7)
    fig.colorbar(im,ax=axes[0],shrink=.7)
    values=[1,p['relative_observed_MSE']['full_empirical'],p['relative_observed_MSE']['bounded_current_family'],p['relative_observed_MSE']['relaxed_shape_only']]
    axes[1].bar(['Zero','Source\ncrossfit','Bounded\n3-tap','Scalar\nrelaxed'],values,color=['gray','#377eb8','#e41a1c','#ff7f00'])
    axes[1].axhline(1,color='gray',ls='--',lw=1);axes[1].set(title='Held-out source contrast error',ylabel='MSE / zero-contrast MSE',ylim=(0,1.2))
    for i,value in enumerate(values):axes[1].text(i,value+.02,f'{value:.3f}',ha='center',fontsize=9)
    axes[2].bar([str(r['RX']) for r in rx],[100*r['weighted_direction_coverage'] for r in rx],color='#377eb8')
    axes[2].set(title='Energy-weighted coverage by source RX',xlabel='Source RX',ylabel='Coverage (%)',ylim=(0,100))
    fig.suptitle('Ideal magnitude proxy only: no identity-performance claim',fontsize=12)
    fig.savefig(out/'source_response_capacity.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
