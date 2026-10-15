"""Scientific figures from complete source-only teacher artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
ARMS=('SAMPLE','EXCLUDE_RX1');COLORS=('#2878b5','#368657')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def save(fig,out,name):fig.savefig(out/name,dpi=180);plt.close(fig)


def run(root):
    result=read(root/'comparison/results.json');resources=read(root/'resources.json');curves=read(root/'training_curves.json')
    if (result['status']!='ANALYZED' or not result['source_only'] or result['query_read'] or result['U_truth_read']
            or result['new_models_trained_by_comparison'] or result['student_models_trained'] or result['head_fit_updates']
            or result['comparison_forward_packets']!=1440 or result['teacher_evaluation_forward_packets']!=10084
            or len(resources['rows'])!=2 or resources['teacher_training_updates']!=22200):
        raise ValueError('Complete actual two-teacher source-only evidence required')
    records=result['records'];by={(r['view'],r['region']):r for r in records}
    if any(by[v,'RX1']['count']!=144 for v in VIEWS):raise ValueError('Fixed RX1 source144 required')
    out=root/'figures';out.mkdir(parents=True,exist_ok=True);x=np.arange(7);names=[v.replace('practical_','') for v in VIEWS]
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for arm,prefix,color in zip(ARMS,('sample','excluded'),COLORS):
        for ax,field,scale in ((axes[0,0],'accuracy',100),(axes[1,0],'true_class_probability',1),(axes[1,1],'highconfidence_wrong',100)):
            ax.plot(x,[scale*by[v,'RX1'][prefix+'_'+field] for v in VIEWS],'o-',label=arm,color=color)
    repairs=[by[v,'RX1']['repairs'] for v in VIEWS];damage=[by[v,'RX1']['new_errors'] for v in VIEWS]
    axes[0,1].bar(x-.18,repairs,.35,label='Repairs',color=COLORS[1]);axes[0,1].bar(x+.18,damage,.35,label='New errors',color=COLORS[0],hatch='///')
    titles=('Default-head accuracy (%)','EXCLUDE vs SAMPLE: repaired / damaged packets','Mean true-class probability','Confidence >= .95 wrong (%)')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title);ax.set_xticks(x,names,rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Held-out source RX1 only, n=144/view; equal4464-label E50 teachers; no target or student result')
    save(fig,out,'source_teacher_complementarity.png')

    clean=by['clean','RX1'];fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    maximum=max(np.max(clean[p+'_confusion']) for p in ('sample','excluded'))
    for ax,arm,prefix in zip(axes[0],ARMS,('sample','excluded')):
        matrix=np.asarray(clean[prefix+'_confusion']);obj=ax.imshow(matrix,vmin=0,vmax=maximum,cmap='Blues')
        if matrix.shape!=(6,6) or int(matrix.sum())!=144:raise ValueError('Actual full TX6 source confusion required')
        for i in range(6):
            for j in range(6):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8)
        ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xlabel('Predicted TX');ax.set_ylabel('True TX');ax.set_title(arm+' clean');fig.colorbar(obj,ax=ax,shrink=.75)
    for arm,prefix,color in zip(ARMS,('sample','excluded'),COLORS):
        axes[1,0].plot((1,2,3),[100*by['clean','RX1_day'+str(day)][prefix+'_accuracy'] for day in (1,2,3)],'o-',label=arm,color=color)
        categories=('RX1_old_gate_accepted','RX1_old_gate_rejected');values=[]
        for region in categories:
            v=by['clean',region];values.append(np.nan if not v['count'] else 100*v[prefix+'_accuracy'])
        axes[1,1].plot(range(2),values,'o-',label=arm,color=color)
    axes[1,0].set_xticks((1,2,3));axes[1,0].set_xlabel('Source day');axes[1,0].set_title('RX1 clean by day')
    axes[1,1].set_xticks(range(2),[f"{r.split('_')[-1]}\nn={by['clean',r]['count']}" for r in categories]);axes[1,1].set_title('Old-gate source replay regions')
    for ax in axes[1]:ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,105);ax.grid(alpha=.2)
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Actual source labels; gate replay uses fixed L_dev batching, not historical U decisions')
    save(fig,out,'source_teacher_day_confusion_gate.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm,color in zip(ARMS,COLORS):
        for ax,key in zip(axes,('train/loss','train/grad_before_clip')):
            values=curves[arm][key]
            if len(values)!=50 or not np.isfinite(values).all():raise ValueError('Full actual E50 curves required')
            ax.plot(range(1,51),values,label=arm,color=color);ax.set_title(key);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Complete11100 steps per teacher, no U/EMA/augmentation; budget end is not convergence')
    save(fig,out,'complete_teacher_training_curves.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(Path(a.root))
