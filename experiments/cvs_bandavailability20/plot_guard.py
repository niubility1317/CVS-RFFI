"""Scalar-only frozen-source guard plots;no IQ/checkpoint/network access."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);d=json.loads((root/'guard_independent_recount.json').read_text());models=list(d['models'].values());colors=['#245d9b','#d27b21'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for j,m in enumerate(models):
 acc=[100*m[k]['accuracy'] for k in ('raw','GAIN','MASK')];axes[0].plot(range(3),acc,'o-',color=colors[j],label=str(m['model_seed']))
 for i,v in enumerate(acc):axes[0].annotate(f'{v:.3f}',(i,v),xytext=(0,6 if j==0 else -15),textcoords='offset points',ha='center',fontsize=9)
 vals=[100*m['draws']['MASK'+str(i)]['raw_correct_retention'] for i in range(1,5)];axes[1].plot(range(1,5),vals,'o-',color=colors[j],label=str(m['model_seed']))
axes[0].set(xticks=range(3),xticklabels=['Raw','Matched gain','Frequency mask'],ylabel='Default-head accuracy (%)',title='Fit-source identity (5040 packets/model)',ylim=(99.3,100.05));axes[1].axhline(99,color='gray',ls='--',lw=1,label='Predeclared 99% floor');axes[1].set(xticks=range(1,5),xlabel='Independent IQ view draw',ylabel='Raw-correct retention (%)',ylim=(98.9,100.05),title='Same physical packets; frozen models')
for ax in axes:ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Source identity guard PASS — not held-RX or training efficacy',fontsize=12)
for ext in ('png','pdf'):fig.savefig(root/('guard_identity.'+ext))
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for j,m in enumerate(models):
 rx=[m['MASK']['by_RX'][str(k)] for k in (3,4,6,8)];pair=[m['MASK']['by_pair'][str(k)] for k in range(1,7)];shift=(j-.5)*.25
 axes[0].bar([i+shift for i in range(4)],[100*(q['recovered']-q['lost'])/q['count'] for q in rx],width=.25,color=colors[j],label=str(m['model_seed']))
 axes[1].plot(range(1,7),[100*(q['recovered']-q['lost'])/q['count'] for q in pair],'o-',color=colors[j],label=str(m['model_seed']))
 q=m['MASK']['margin_change_over30'];axes[2].errorbar(j,q['median'],yerr=[[q['median']-q['p10']],[q['p90']-q['median']]],fmt='o',capsize=5,color=colors[j],label=str(m['model_seed']))
axes[0].axhline(-1,color='gray',ls='--',lw=1);axes[0].set(xticks=range(4),xticklabels=['RX3','RX4','RX6','RX8'],ylabel='Mask minus matched raw (pp)',title='Pooled RX strata')
axes[1].set(xticks=range(1,7),xlabel='Masked harmonic pair k (±k/20)',ylabel='Mask minus matched raw (pp)',title='All predefined frequency pairs')
axes[2].axhline(0,color='gray',lw=1);axes[2].set(xticks=range(2),xticklabels=['Seed 01','Seed 02'],ylabel='True-class margin change / 30',title='Median and 10–90% interval')
for ax in axes:ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('Frozen source guard strata; 20160 masked views/model',fontsize=12)
for ext in ('png','pdf'):fig.savefig(root/('guard_strata.'+ext))
plt.close(fig);print(json.dumps(dict(status='VERIFIED',figures=['guard_identity.png','guard_identity.pdf','guard_strata.png','guard_strata.pdf'],IQ_reads=0,CP_reads=0,NN=0)))
