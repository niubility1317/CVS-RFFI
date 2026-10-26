"""Publication figures from fixed historical predictions; no learning or selection."""
import argparse,csv,json,statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NAMES={'sig_time':'PAMF','sig_time_removed':'PAMF w/o SIG-TIME','resnet18':'ResNet-18-1D','resnet34':'ResNet-34-1D',
 'resnet50':'ResNet-50-1D','vgg16_bn':'VGG-16-BN-1D','densenet121':'DenseNet-121-1D','convnext_tiny':'ConvNeXt-Tiny-1D',
 'transformer6':'Transformer (6 layers)','bilstm3':'BiLSTM (3 layers)','cvresnet18_1d':'Complex ResNet-18-1D',
 'cvresnet18_2d':'Complex ResNet-18-2D','resnet18_2d':'ResNet-18-2D','resnet50_2d':'ResNet-50-2D'}
COLORS=dict(zip(NAMES,['#D55E00','#234A6F','#C62828','#795548','#0072B2','#B8860B','#009E73','#303030','#737373','#009FA8','#8E44AD','#5364D9','#556B2F','#D64B8C']))
PARAMS={'sig_time':196001,'sig_time_removed':177025,'resnet18':3847430,'resnet34':7221766,'resnet50':15966982,
 'vgg16_bn':36406022,'densenet121':5526086,'convnext_tiny':26768550,'transformer6':18921990,'bilstm3':14743558,
 'cvresnet18_1d':7693958,'cvresnet18_2d':22346630,'resnet18_2d':11176454,'resnet50_2d':23517190}
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'axes.labelsize':7.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})


def validate_record(r):
 cm=np.asarray(r['confusion']);n=int(cm.sum());den=cm.sum(0)+cm.sum(1)
 f=np.divide(2*cm.diagonal(),den,out=np.zeros(6),where=den!=0).mean()
 if n!=r['query_count'] or abs(cm.trace()/n-r['accuracy'])>1e-12 or abs(f-r['macro_f1'])>1e-12:raise ValueError('Metric mismatch')


def csvwrite(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
 p=argparse.ArgumentParser();p.add_argument('--double',type=Path,required=True);p.add_argument('--full',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.double.read_text());full=json.loads(a.full.read_text())
 a.output.mkdir(parents=True,exist_ok=False)
 for name in ['png','preview','svg','pdf','source_data']:(a.output/name).mkdir()
 data=d['records'];all_records=data+full['records']
 for r in all_records:validate_record(r)
 variants=sorted({r['variant'] for r in data});seeds=range(2026092701,2026092709)
 summary=[];per_seed=[]
 for scope,records,count in [('双跨',data,42000),('综合',full['records'],168000)]:
  for v in variants:
   rs=[r for r in records if r['variant']==v and r['dimension']=='overall'];assert len(rs)==8
   assert {r['model_seed'] for r in rs}==set(seeds) and all(r['query_count']==count for r in rs)
   worst=[]
   for r in sorted(rs,key=lambda x:x['model_seed']):
    rx=[x for x in records if x['variant']==v and x['model_seed']==r['model_seed'] and x['dimension']=='receiver'];assert len(rx)==7
    tx=[x for x in records if x['variant']==v and x['model_seed']==r['model_seed'] and x['dimension']=='transmitter'];assert len(tx)==6
    assert np.array_equal(sum(np.asarray(x['confusion']) for x in rx),r['confusion'])
    assert np.array_equal(sum(np.asarray(x['confusion']) for x in tx),r['confusion'])
    low=min(x['accuracy'] for x in rx);worst.append(low)
    per_seed.append(dict(scope=scope,variant=v,network=NAMES[v],model_seed=r['model_seed'],query_count=count,
      accuracy=r['accuracy'],macro_f1=r['macro_f1'],worst_rx=low,total_parameters=PARAMS[v]))
   mean=lambda k:st.mean(r[k] for r in rs);sd=lambda k:st.stdev(r[k] for r in rs)
   summary.append(dict(scope=scope,variant=v,network=NAMES[v],actual_seed_count=8,query_count_per_seed=count,
    accuracy=mean('accuracy'),accuracy_sd_pp=sd('accuracy')*100,macro_f1=mean('macro_f1'),macro_f1_sd_pp=sd('macro_f1')*100,
    worst_rx=st.mean(worst),worst_rx_sd_pp=st.stdev(worst)*100,total_parameters=PARAMS[v]))
 order=sorted(variants,key=lambda v:next(r['accuracy'] for r in summary if r['scope']=='双跨' and r['variant']==v),reverse=True)
 layers={}
 for dimension in ('receiver','transmitter'):
  rows=[]
  for scope,records in [('双跨',data),('综合',full['records'])]:
   for v in order:
    strata=sorted({r['stratum'] for r in records if r['dimension']==dimension and r['variant']==v})
    for label in strata:
     rs=[r for r in records if r['dimension']==dimension and r['variant']==v and r['stratum']==label];assert len(rs)==8
     rows.append(dict(scope=scope,variant=v,network=NAMES[v],stratum=label,actual_seed_count=8,query_count_per_seed=rs[0]['query_count'],
      accuracy=st.mean(r['accuracy'] for r in rs),accuracy_sd_pp=st.stdev(r['accuracy'] for r in rs)*100,
      macro_f1=st.mean(r['macro_f1'] for r in rs),macro_f1_sd_pp=st.stdev(r['macro_f1'] for r in rs)*100))
  layers[dimension]=rows
 paired=[]
 if 'sig_time_removed' in variants:
  for scope in ('双跨','综合'):
   for seed in seeds:
    f=next(r for r in per_seed if r['scope']==scope and r['variant']=='sig_time' and r['model_seed']==seed)
    c=next(r for r in per_seed if r['scope']==scope and r['variant']=='sig_time_removed' and r['model_seed']==seed)
    paired.append(dict(scope=scope,model_seed=seed,full_accuracy=f['accuracy'],removed_accuracy=c['accuracy'],accuracy_difference_pp=100*(f['accuracy']-c['accuracy']),full_macro_f1=f['macro_f1'],removed_macro_f1=c['macro_f1'],f1_difference_pp=100*(f['macro_f1']-c['macro_f1'])))
 tables={'summary':summary,'per_seed':per_seed,'receiver':layers['receiver'],'transmitter':layers['transmitter'],'paired':paired}
 for name,rows in tables.items():
  if rows:csvwrite(a.output/'source_data'/f'{name}.csv',rows)
 figs=[]
 pdf=PdfPages(a.output/'all_results.pdf')
 def save(fig,stem,title,width=183,height=125):
  fig.savefig(a.output/'svg'/f'{stem}.svg');fig.savefig(a.output/'pdf'/f'{stem}.pdf')
  fig.savefig(a.output/'png'/f'{stem}.png',dpi=600);fig.savefig(a.output/'preview'/f'{stem}.png',dpi=180)
  pdf.savefig(fig);figs.append(dict(stem=stem,title=title,width_mm=width,height_mm=height));plt.close(fig)
 def footer(fig):fig.text(.5,.018,'Unseen date D0 + 7 unseen receivers; 42,000 fixed packets/model; E200 pure CE; n=8; mean ± sample SD.',ha='center',fontsize=6.5,color='#555555')
 def grid(ax):ax.set_axisbelow(True);ax.grid(axis='x',color='#E5E7EB',lw=.5)
 # Main performance: all models, both metrics, all seeds.
 fig,axes=plt.subplots(1,2,figsize=(183/25.4,140/25.4),sharey=True);fig.subplots_adjust(left=.31,right=.91,top=.91,bottom=.12,wspace=.25)
 for ax,key,title in zip(axes,['accuracy','macro_f1'],['a  Cross-day + cross-receiver accuracy','b  Macro-F1']):
  for i,v in enumerate(order):
   values=[r[key]*100 for r in per_seed if r['scope']=='双跨' and r['variant']==v]
   ax.scatter(values,i+np.linspace(-.12,.12,8),s=7,color='#A8ADB3',zorder=2)
   m=st.mean(values);sd=st.stdev(values);ax.errorbar(m,i,xerr=sd,fmt='D',color=COLORS[v],ms=3,capsize=2,lw=.9,zorder=3)
   ax.text(104,i,f'{m:.2f}',va='center',ha='left',fontsize=6.5,color=COLORS[v],clip_on=False)
  ax.set_xlim(0,101);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Score (%)');ax.set_title(title,loc='left',fontsize=7.4,pad=12);grid(ax)
 axes[0].set_yticks(range(len(order)),[NAMES[v] for v in order]);axes[0].invert_yaxis();footer(fig)
 save(fig,'fig01_overall_double_holdout','全部模型双跨准确率与Macro-F1',height=140)
 # Receiver and transmitter heatmaps.
 for dimension,stem,title in [('receiver','fig02_receiver_double_holdout','Receiver accuracy (%)'),('transmitter','fig03_transmitter_double_holdout','Transmitter recall (%)')]:
  labels=sorted({r['stratum'] for r in data if r['dimension']==dimension})
  matrix=np.array([[next(r['accuracy']*100 for r in layers[dimension] if r['scope']=='双跨' and r['variant']==v and r['stratum']==label) for label in labels] for v in order])
  fig,ax=plt.subplots(figsize=(183/25.4,130/25.4));fig.subplots_adjust(left=.31,right=.90,top=.92,bottom=.13)
  im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto');ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(len(order)),[NAMES[v] for v in order]);ax.set_title(title,loc='left',pad=12)
  for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=6,color='white' if value>65 else '#24303F')
  fig.colorbar(im,ax=ax,fraction=.03,pad=.04);footer(fig);save(fig,stem,title,height=130)
 # Parameter efficiency with explicit compressed small-model range.
 fig,ax=plt.subplots(figsize=(183/25.4,125/25.4));fig.subplots_adjust(left=.12,right=.96,top=.90,bottom=.29)
 for i,v in enumerate(order):
  rs=[r['accuracy']*100 for r in per_seed if r['scope']=='双跨' and r['variant']==v];p=PARAMS[v]/1e6;x=p/.3*2 if p<.3 else p
  ax.errorbar(x,st.mean(rs),yerr=st.stdev(rs),fmt='s' if v=='sig_time_removed' else 'D' if v=='sig_time' else 'o',ms=5,color=COLORS[v],capsize=2,lw=.8,label=f'{i+1} {NAMES[v]}')
  text=f'{NAMES[v]}\n{p:.3f} M' if v.startswith('sig_time') else str(i+1)
  offsets={'sig_time':(8,14),'sig_time_removed':(80,28),'cvresnet18_1d':(10,17),'resnet34':(7,-17),'densenet121':(-13,12),'resnet18':(-13,-17)}
  ax.annotate(text,(x,st.mean(rs)),xytext=offsets.get(v,(8,9)),textcoords='offset points',color=COLORS[v],fontsize=6.5,arrowprops=dict(arrowstyle='-',lw=.5,color=COLORS[v]))
 ax.set_xticks([3,5,10,20,30,40,50]);ax.set_xlim(0,51);ax.set_ylim(0,100);ax.set_xlabel('Total parameters (M); left range compressed, main axis linear');ax.set_ylabel('Accuracy (%)');ax.set_title('Capacity and cross-day + cross-receiver accuracy',loc='left',pad=12);ax.grid(color='#E5E7EB',lw=.5)
 ax.text(1.8,-.05,'//',transform=ax.get_xaxis_transform(),ha='center',fontsize=10)
 fig.legend(loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,fontsize=6,handletextpad=.3,columnspacing=.8);footer(fig)
 fig.text(.5,.045,'Parameter count does not measure inference latency or total computation.',ha='center',fontsize=6,color='#555555')
 save(fig,'fig04_parameter_double_holdout','参数量与双跨准确率')
 # One requested figure for every architecture: receiver, transmitter, confusion.
 for v in order:
  fig,axes=plt.subplots(1,3,figsize=(183/25.4,92/25.4),gridspec_kw={'width_ratios':[1,1,1.25]});fig.subplots_adjust(left=.07,right=.97,top=.83,bottom=.22,wspace=.55)
  fig.suptitle(f'{NAMES[v]} — cross-day + cross-receiver',fontsize=10,y=.96)
  for ax,dim,title in zip(axes[:2],['receiver','transmitter'],['a  Held-out receivers','b  Transmitter recall']):
   labels=sorted({r['stratum'] for r in data if r['dimension']==dim})
   for i,label in enumerate(labels):
    vals=[r['accuracy']*100 for r in data if r['dimension']==dim and r['variant']==v and r['stratum']==label]
    ax.scatter(vals,i+np.linspace(-.1,.1,8),s=5,c='#A8ADB3');ax.errorbar(st.mean(vals),i,xerr=st.stdev(vals),fmt='D',color=COLORS[v],capsize=2,ms=3,lw=.8)
   ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xticks([0,50,100]);ax.set_xlabel('Accuracy (%)');ax.set_title(title,loc='left',fontsize=7);grid(ax)
  cms=np.asarray([r['confusion'] for r in data if r['variant']==v and r['dimension']=='overall']);matrix=cms.mean(0)/7000*100
  ax=axes[2];im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues');ax.set_xticks(range(6),d['classes'],rotation=55,ha='right',fontsize=5.5);ax.set_yticks(range(6),d['classes'],fontsize=5.5);ax.set_title('c  Confusion (%)',loc='left',fontsize=7);ax.set_xlabel('Predicted identity');ax.set_ylabel('True identity')
  for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=4.5,color='white' if value>65 else '#24303F')
  footer(fig);save(fig,'model_'+v,NAMES[v]+'双跨分层与混淆矩阵',height=92)
 if paired:
  fig,axes=plt.subplots(1,2,figsize=(183/25.4,90/25.4));fig.subplots_adjust(left=.1,right=.95,top=.86,bottom=.22,wspace=.3)
  for ax,scope in zip(axes,['双跨','综合']):
   rs=[r for r in paired if r['scope']==scope];values=[r['accuracy_difference_pp'] for r in rs]
   ax.axhline(0,color='#8E949C',ls='--',lw=.8);ax.scatter(range(1,9),values,c=COLORS['sig_time'],s=20);ax.set_xticks(range(1,9));ax.set_xlabel('Paired seed (01–08)');ax.set_ylabel('Full minus w/o SIG-TIME (pp)')
   heading='Double holdout' if scope=='双跨' else 'All target dates'
   ax.set_title(heading+f'\nMean {st.mean(values):+.2f} ± {st.stdev(values):.2f} pp',loc='left',fontsize=8,pad=10)
  fig.text(.5,.035,'Same split, seed, E200 pure CE, default head and shared architecture; all eight differences retained.',ha='center',fontsize=6.5)
  save(fig,'fig05_sigtime_ablation','SIG-TIME严格消融配对差',height=90)
 pdf.close()
 bundle=dict(status='VERIFIED',subset=d['subset'],variants=order,names=NAMES,parameters=PARAMS,summary=summary,per_seed=per_seed,
  receiver=layers['receiver'],transmitter=layers['transmitter'],paired=paired,records=[dict(r,scope='双跨') for r in data]+[dict(r,scope='综合') for r in full['records']],figures=figs)
 (a.output/'workbook_input.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
 captions=['# Cross-day and cross-receiver clean-test figures','',
  'All main figures use D0 (2021-03-01) and seven receivers excluded from source training. Each frozen model predicts the same 42,000 packets: 6,000 per receiver and 7,000 per transmitter. Source training dates are D1–D3 (2021-03-08, 2021-03-15 and 2021-03-23). D0 predates those dates; this evaluates an unseen date, not prospective future-date extrapolation.',
  '', 'The eight model seeds are 2026092701–2026092708, with physical split seed392005 fixed. Grey dots show individual model seeds; colored diamonds and error bars show the mean and sample standard deviation (ddof=1). These are repeated model initializations on a fixed split, not eight independent datasets or a confidence interval. No collapsed seed or negative result is excluded.',
  '', 'PAMF denotes the historical 196,001-parameter SIG_TIME whole architecture trained with pure CE for E200 and its native default head. It does not denote the later T1/X3 semi-supervised recipe. The matched removal control retains reference response, internal compensation, explicit-CFO coordinate masking and the default head; only its 18,976-parameter SIG-TIME path is removed. Both use the same E200 from-scratch CE recipe.',
  '', 'Fig. 1. Accuracy and macro-F1 across all declared models. Model ordering is descriptive and is not used for checkpoint or candidate selection.',
  '', 'Fig. 2. Mean accuracy by held-out receiver. Each cell averages eight independently trained model seeds; every receiver contains all six transmitter identities.',
  '', 'Fig. 3. Mean recall by transmitter identity. Single-transmitter accuracy is class recall, not precision or class F1.',
  '', 'Fig. 4. Parameter count and double-holdout accuracy. The sub-0.3M region is explicitly compressed; the main axis spans 3–50M linearly. Small-model parameter counts are annotated. Parameters do not measure latency, memory or total network computation.',
  '', 'Individual model pages. Panels a and b retain per-seed receiver accuracy and transmitter recall. Panel c shows the mean row-normalized confusion matrix, with true identities in rows and predicted identities in columns. Confusion entries are descriptive averages of individual models, not ensemble decisions.',
  '', 'Matched ablation page, when complete. Each point is the same-seed accuracy difference, full model minus removal control. Double-holdout and all-target-date results are shown separately. All eight positive or negative differences are retained; no significance claim is inferred from their sign count.',
  '', 'The 168,000-packet historical overall evaluation uses D0–D3 on the same seven unseen receivers. It contains receiver-only and simultaneous date/receiver shifts; it does not contain a source-receiver, date-only test. The double-holdout results are post-hoc subgroup calculations on already frozen predictions from an exposed proxy benchmark, not a new independent blind test. No result is fed back into training, hyperparameter changes, model selection or selective reruns.']
 (a.output/'figure_captions.md').write_text('\n'.join(captions)+'\n',encoding='utf-8')
 print(json.dumps(dict(models=len(order),figures=len(figs),main_summary=[r for r in summary if r['scope']=='双跨' and r['variant'].startswith('sig_time')]),ensure_ascii=False))


if __name__=='__main__':main()
