from pathlib import Path
import json,csv,statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
P=Path('E:/type10-7');O=P/'local_artifacts/phase1_transfer_analysis_20261008'
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
stats=json.loads((O/'analysis_stats.json').read_text(encoding='utf-8'))
meta=json.loads((O/'metadata_epochs.json').read_text(encoding='utf-8'))
fig,axs=plt.subplots(2,2,figsize=(13,8.3),layout='constrained')
ax=axs[0,0];x=np.arange(4)
for offset,values,name,color in [(-.18,[79.622,79.960,79.607,79.731],'clean','#3477a8'),(.18,[46.482,65.699,49.173,65.372],'混合satellite','#df8f44')]:
 bars=ax.bar(x+offset,values,.35,label=name,color=color)
 ax.bar_label(bars,fmt='%.2f',padding=2,fontsize=9)
ax.set(xticks=x,xticklabels=['CE','LEO','MixStyle','LEO＋MixStyle'],ylim=(0,91),ylabel='准确率（%）',title='A  LEO仍有大幅收益；其后叠加MixStyle无额外收益')
ax.legend(frameon=False,loc='lower right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax=axs[0,1]
for r in stats['summary']:
 if r['stage']!='r4':continue
 ax.scatter(r['source_score'],r['clean'],s=65,color='#277983' if r['arm']!='episode' else '#bd5451')
 ax.annotate(r['arm'],(r['source_score'],r['clean']),xytext=(5,4),textcoords='offset points',fontsize=9)
ax.set(xlim=(97.23,97.61),ylim=(77.1,79.65),xlabel='源选择分数（%）',ylabel='目标clean准确率（%）',title='B  R4源域小幅排名与目标收益并不一致');ax.grid(alpha=.15)
ax=axs[1,0]
rs=[r for r in meta['rows'] if r['config']['stage']=='r2' and r['config']['arm']=='pseudo']
es=range(131,201)
for key,name,color in [('train_temporal_pass','时间邻居通过','#3477a8'),('train_reliable_ratio','最终采用','#bd5451')]:
 ys=[100*st.mean(r['epochs'][e-1][key] for r in rs) for e in es];ax.plot(list(es),ys,label=name,color=color,lw=1.6)
ax.axhline(4*255/56699*100,color='#333333',ls='--',lw=1,label='随机batch邻居概率上界≈1.80%')
ax.set(xlabel='epoch',ylabel='U样本比例（%）',ylim=(0,2.5),title='C  伪标签门控受batch内邻居可用率限制');ax.legend(frameon=False,fontsize=9,loc='lower right');ax.grid(alpha=.15)
ax=axs[1,1]
for off,view,name,col in [(-.18,'clean','clean','#3477a8'),(.18,'practical_low_urban','完整低仰角城市','#df8f44')]:
 r=next(r for r in stats['paired'] if r['stage']=='r4' and r['arm']=='episode' and r['view']==view)
 bars=ax.bar(x+off,r['deltas'],.35,label=name,color=col);ax.bar_label(bars,fmt='%+.2f',padding=2,fontsize=9)
ax.axhline(0,color='#777',lw=.8);ax.set(xticks=x,xticklabels=['seed 01','seed 02','seed 03','seed 04'],ylabel='episode−base（百分点）',ylim=(-3,7),title='D  episode的正均值明显受seed 03影响');ax.legend(frameon=False,loc='upper left',fontsize=9)
fig.suptitle('新骨干叠加旧Phase1机制：收益、选择与有效训练信号',fontsize=17)
fig.savefig(O/'mechanism_analysis.png',dpi=170)
fig.savefig(O/'mechanism_analysis.pdf')
print(O/'mechanism_analysis.png')
