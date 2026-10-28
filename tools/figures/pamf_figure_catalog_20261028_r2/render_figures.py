"""Render the supplied numerical catalog. All figure drawing uses Python."""
import argparse,csv,json,math,re,time,warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
plt.rcParams['svg.fonttype']='none'
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'savefig.facecolor':'white'})
from matplotlib.colors import to_rgb,LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch

COLORS={'RIEI':'#718FA8','DRIFT':'#6B9C97','IB-RFF':'#9A88AE','SDG':'#82A36E','PAMF':'#C4364E','PAMF+DURA':'#E29525','Supervised':'#84919D','Ordinary PL':'#3C718C','Tri-SIG-TIME SSL':'#C4364E','Without DURA':'#AEB9C3','With DURA':'#E29525','Clean':'#6F7C89','Six-scene mean':'#C4364E'}
PALETTE=['#718FA8','#6B9C97','#9A88AE','#82A36E','#C4364E','#E29525','#426A90','#B08376','#7989AE']
def color(label,i=0):
 if label in COLORS:return COLORS[label]
 if 'Three-view' in label:return '#C4364E'
 if 'Single-view' in label:return '#E29525'
 if 'SIG-TIME' in label:return '#377C87'
 return PALETTE[i%len(PALETTE)]
def tint(c):return tuple(.65+.35*x for x in to_rgb(c))
def rows_by(p):return {(str(r['category']),str(r.get('series',''))):r for r in p['rows']}

def label(s):
 s=str(s)
 exact={'Receiver split 1':'Rx split 1','Receiver split 2':'Rx split 2','PAMF-Sup':'Supervised','PAMF-PL':'PL (0.95)','Supervised':'Supervised','Ordinary PL':'PL (0.95)','Tri-SIG-TIME SSL':'Full SSL','raw':'Direct','matched_DURA':'DURA','DURA-LU':'DURA (L+U)','DURA-L':'DURA (L)','Six-scene mean':'Six-scene mean','High suburban':'Suburban-H','Mid suburban':'Suburban-M','Low suburban':'Suburban-L','High urban':'Urban-H','Mid urban':'Urban-M','Low urban':'Urban-L'}
 if s in exact:return exact[s]
 return s.replace(' source receivers',' source Rx').replace(' source receiver',' source Rx').replace('source RX','source Rx')
def limits(ax,p):
 if p.get('unit')=='percent':
  low,high=p.get('y_lim',[0,100]);ax.set_ylim(low,high)
  if p.get('y_tick_step'):ax.set_yticks(np.arange(low,high+.1,p['y_tick_step']))
 elif p.get('unit')=='delta':ax.axhline(0,color='#6E747A',ls='--',lw=.8,zorder=1)
 ax.grid(axis='y',color='#DCE2E6',lw=.55,zorder=0);ax.set_axisbelow(True)
def panel_render(ax,p,index):
 cats=list(map(str,p['categories']));series=list(map(str,p['series']));lookup=rows_by(p);kind=p['kind'];unit=p.get('unit','percent')
 ylabel=p.get('y_label','Accuracy (%)' if unit=='percent' else 'Change (pp)' if unit=='delta' else unit)
 if kind in ('bar','line'):
  x=np.arange(len(cats));w=.78/max(1,len(series))
  for j,ser in enumerate(series):
   rr=[lookup.get((c,ser)) for c in cats];ys=np.array([r['estimate'] if r else np.nan for r in rr],float);sd=np.array([r.get('sd') or 0 if r else np.nan for r in rr],float)
   cc=color(ser,j)
   if kind=='bar':
    xs=x+(j-(len(series)-1)/2)*w
    fills=[color(c,i) if c in COLORS else cc for i,c in enumerate(cats)]
    hatch='///' if ser in ['Supervised','Without DURA'] else '..' if ser=='Ordinary PL' else None
    if ser in ['Supervised','Without DURA','Ordinary PL']:fills=[tint(c) for c in fills]
    ax.bar(xs,ys,w*.90,yerr=sd,color=fills,edgecolor='#56616A' if hatch else '#FFFFFF',lw=.45,hatch=hatch,error_kw={'elinewidth':.8,'capsize':2},label=label(ser),zorder=3)
    if p.get('show_bar_values'):
     for xv,yv,sv in zip(xs,ys,sd):
      if np.isfinite(yv):ax.text(xv,yv+sv+.65,f'{yv:.2f}%',ha='center',va='bottom',fontsize=7.5,color='#233444',zorder=6)
   else:ax.errorbar(x,ys,yerr=sd,color=cc,label=label(ser),lw=1.9 if ser in ['PAMF','PAMF+DURA'] else 1.15,marker=['o','s','^','v','D','P','X','h','*'][j%9],ms=4.5,capsize=2,zorder=4 if ser.startswith('PAMF') else 2)
  rotation=p.get('label_rotation',45 if len(cats)>6 and any(len(c)>9 for c in cats) else 25 if any(len(c)>11 for c in cats) else 0)
  ax.set_xticks(x,[label(c) for c in cats],rotation=rotation,ha='right' if rotation else 'center');ax.set_ylabel(ylabel);limits(ax,p)
  if p.get('dura_arrows'):
   for i,cat in enumerate(cats):
    a,b=lookup[(cat,series[0])],lookup[(cat,series[1])];x1,x2=x[i]-w/2,x[i]+w/2;y1=a['estimate']+(a.get('sd') or 0)+2.2;y2=b['estimate']+(b.get('sd') or 0)+2.2;top=max(y1,y2)+5.2
    path=MplPath([(x1,y1),(x1,top),(x2,top),(x2,y2)],[MplPath.MOVETO,MplPath.LINETO,MplPath.LINETO,MplPath.LINETO])
    cc='#2F8179' if p.get('show_gain') else '#596E7A';ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=9,lw=.85,color=cc,zorder=5))
    if p.get('show_gain'):ax.text(x[i],top+.7,f"+{b['estimate']-a['estimate']:.2f} pp",ha='center',va='bottom',fontsize=8,color=cc,weight='bold')
  if series!=['']:
   handles=None
   if kind=='bar' and any(c in COLORS for c in cats) and all(s in ['Supervised','Ordinary PL','Tri-SIG-TIME SSL','Without DURA','With DURA'] for s in series):
    handles=[Patch(facecolor='#E3E7EA' if s in ['Supervised','Without DURA','Ordinary PL'] else '#687783',edgecolor='#56616A',hatch='///' if s in ['Supervised','Without DURA'] else '..' if s=='Ordinary PL' else None,label=label(s)) for s in series]
   ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.015),ncol=min(3,len(series)),frameon=False,borderaxespad=0)
  if p.get('x_label'):ax.set_xlabel(p['x_label'])
 elif kind=='point_table':
  y=np.arange(len(cats))
  for i,c in enumerate(cats):
   r=lookup[(c,'')];vals=[s['value'] for s in p.get('seeds',[]) if s['category']==c];cc=color(c,i)
   if vals:ax.scatter(vals,y[i]+np.linspace(-.08,.08,len(vals)),s=10,color='#A2A9B0',alpha=.65,zorder=2)
   ax.errorbar(r['estimate'],y[i],xerr=r['sd'],fmt='D' if 'PAMF' in c else 'o',color=cc,ms=5,lw=1.2,capsize=3,zorder=3)
   ax.text(1.025,y[i],f"{r['estimate']:.2f} ± {r['sd']:.2f}",transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8)
   ax.text(1.40,y[i],f"{r['variance_pp2']:.2f}",transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8)
  ax.text(1.025,1.015,'Mean ± SD (%)',transform=ax.transAxes,fontsize=8,weight='bold')
  ax.text(1.40,1.015,'Var (pp²)',transform=ax.transAxes,fontsize=8,weight='bold')
  ax.set_yticks(y,cats);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Accuracy (%)' if 'Accuracy' in p['title'] else 'Macro-F1 (%)');ax.grid(axis='x',color='#DFE4E8',lw=.6);ax.set_axisbelow(True)
  ax.legend(handles=[Line2D([],[],color='#566E7D',marker='o',lw=1,label=p['legend_label'])],loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=8)
 elif kind=='point':
  horizontal=p.get('horizontal',True);y=np.arange(len(cats));w=.45/max(1,len(series))
  for j,ser in enumerate(series):
   for i,c in enumerate(cats):
    r=lookup.get((c,ser))
    if not r:continue
    val=r['estimate'];sd=r.get('sd') or 0;pos=y[i]+(j-(len(series)-1)/2)*w
    cc=('#B9554F' if val<0 else '#2F8179') if unit=='delta' else color(c if not ser else ser,i if not ser else j)
    vals=r.get('seed_values') or [s['value'] for s in p.get('seeds',[]) if str(s['category'])==c and str(s.get('series',''))==ser]
    if vals:ax.scatter(vals,np.full(len(vals),pos)+np.linspace(-.08,.08,len(vals)),s=11,color='#A2A9B0',alpha=.65,zorder=2)
    ax.errorbar(val,pos,xerr=sd,fmt=['o','s','^','D','v','P'][j%6],color=cc,mfc='white' if j==0 and len(series)>1 else cc,ms=5,lw=1.2,capsize=3,zorder=3)
  ax.set_yticks(y,[label(c) for c in cats]);ax.invert_yaxis();ax.set_xlabel(p.get('x_label','Change (pp)' if unit=='delta' else label(p.get('title','Score'))+' (%)'));ax.grid(axis='x',color='#DFE4E8',lw=.6);ax.set_axisbelow(True)
  if unit=='percent':ax.set_xlim(0,100)
  if unit=='delta':ax.axvline(0,color='#68727B',ls='--',lw=.8)
  if len(series)>1:ax.legend(handles=[Line2D([],[],color='#6F7C89' if unit=='delta' else color(s,j),marker=['o','s','^','D','v','P'][j%6],mfc='white' if j==0 else '#6F7C89' if unit=='delta' else color(s,j),label=s,ls='none') for j,s in enumerate(series)],loc='lower left',bbox_to_anchor=(0,1.02),ncol=min(3,len(series)),frameon=False)
 elif kind=='heatmap':
  arr=np.full((len(series),len(cats)),np.nan)
  for i,s in enumerate(series):
   for j,c in enumerate(cats):
    r=lookup.get((c,s))
    if r:arr[i,j]=r['estimate']
  if unit=='delta':
   mx=max(1,float(np.nanmax(np.abs(arr))));im=ax.imshow(arr,cmap=LinearSegmentedColormap.from_list('signed_change',['#B9554F','#FFFFFF','#2F8179']),vmin=-mx,vmax=mx,aspect='auto')
  else:im=ax.imshow(arr,cmap='Blues',vmin=p.get('vmin',0),vmax=100 if unit=='percent' else p.get('vmax',float(np.nanmax(arr))),aspect='auto')
  ax.set_xticks(np.arange(len(cats)),[label(c) for c in cats],rotation=55 if len(cats)>12 else 35 if len(cats)>6 else 0,ha='right' if len(cats)>6 else 'center');ax.set_yticks(np.arange(len(series)),[label(s) for s in series])
  if p.get('show_values',len(cats)*len(series)<=130):
   for i in range(len(series)):
    for j in range(len(cats)):
     if np.isfinite(arr[i,j]):ax.text(j,i,f'{arr[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white' if (abs(arr[i,j])>mx*.6 if unit=='delta' else arr[i,j]>(65 if unit=='percent' else float(np.nanmax(arr))*.65)) else '#233444')
  cb=ax.figure.colorbar(im,ax=ax,fraction=.045,pad=.025);cb.ax.tick_params(labelsize=7)
  cb.set_label(p.get('scale_label','Change (pp)' if unit=='delta' else 'Score (%)'),fontsize=8)
 elif kind=='scatter':
  if p.get('parameter_break'):
   parameter_break(ax,p,index);return
  handles=[]
  for i,r in enumerate(p['rows']):
   c=color(r['category'],i);mk='D' if 'PAMF' in r['category'] else ['o','s','^','v','P','X','h','*'][i%8];ax.errorbar(r['x'],r['estimate'],yerr=r.get('sd') or 0,color=c,marker=mk,ms=6,lw=.9,capsize=2,ls='none');handles.append(Line2D([],[],ls='none',marker=mk,color=c,label=r['category']))
  if p.get('x_log'):ax.set_xscale('log')
  ax.set_xlabel(p.get('x_label','Parameters'));ax.set_ylabel(ylabel);limits(ax,p);ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False)
 else:raise ValueError(kind)
 if p.get('title') and kind not in ['point','point_table']:ax.set_title(label(p['title']),loc='left',pad=48 if kind in ['bar','line'] else 8,fontsize=10)
 if p.get('title') and kind=='point':ax.set_title(p['title'],loc='left',pad=52 if len(series)>3 else 30 if len(series)>1 else 9,fontsize=10)
 ax.text(-.02,1.02,chr(97+index),transform=ax.transAxes,ha='right',va='bottom',weight='bold',fontsize=10)

def parameter_break(ax,p,index):
 order=['PAMF-CE (Single-view)','PAMF-CE w/o SIG-TIME','Complex ResNet-18-1D','ResNet-50-1D','ResNet-34-1D','DenseNet-121-1D','ResNet-18-1D','VGG-16-BN-1D','BiLSTM (3 layers)','ConvNeXt-Tiny-1D','ResNet-18-2D','ResNet-50-2D','Complex ResNet-18-2D','Transformer (6 layers)','PAMF-CE (Three-view)']
 offsets={3:(10,12),4:(8,11),5:(8,-14),6:(-8,12),7:(-12,-14),8:(8,10),9:(-8,12),10:(8,10),11:(6,10),12:(10,8),13:(8,14),14:(8,-8)}
 handles=[]
 for i,n in enumerate(order,1):
  r=next(x for x in p['rows'] if x['category']==n);small='PAMF' in n;x=.65+.75*(r['x']-177025)/(232993-177025) if small else r['x']/1e6;cc=color(n,i-1);mk='D' if small else ['o','s','^','v','P','X','h','*'][i%8]
  ax.errorbar(x,r['estimate'],yerr=r['sd'],color=cc,marker=mk,ms=6,lw=.9,capsize=2,ls='none',zorder=4)
  handles.append(Line2D([],[],ls='none',marker=mk,color=cc,label=f'{i} '+n.replace('PAMF-CE','PAMF')))
  if small:
   tx,ty={1:(11,94),2:(2.0,85.5),15:(1.0,96.5)}[i]
   text={1:'PAMF (1-view)',2:'w/o SIG-TIME',15:'PAMF (3-view)'}[i]+f"\n{r['x']/1e6:.3f} M ({int(r['x']):,})"
   ax.annotate(text,xy=(x,r['estimate']),xytext=(tx,ty),fontsize=8,color=cc,ha='left',va='top',arrowprops=dict(arrowstyle='-',color=cc,lw=.75))
  else:ax.annotate(str(i),(x,r['estimate']),xytext=offsets[i],textcoords='offset points',fontsize=8,color=cc,arrowprops=dict(arrowstyle='-',color=cc,lw=.7))
 ax.set(xlim=(-.3,51),ylim=(0,101),xticks=[3,5,10,20,30,40,50],xlabel='Parameters (M)',ylabel='Accuracy (%)')
 ax.grid(color='#DCE2E6',lw=.55);ax.set_axisbelow(True)
 xx=np.linspace(.7,2.15,80);ax.plot(xx,-.035+.010*np.sin(np.linspace(0,4*np.pi,80)),transform=ax.get_xaxis_transform(),color='#4F5962',lw=1.1,clip_on=False)
 ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False,fontsize=8,columnspacing=1.3)
 ax.text(-.02,1.02,chr(97+index),transform=ax.transAxes,ha='right',va='bottom',weight='bold',fontsize=10)

def render(f,out):
 n=len(f['panels']);nc=2 if n!=3 else 3;nc=min(n,nc);nr=math.ceil(n/nc)
 if f['panels'][0]['kind']=='point_table':
  nc=1;nr=n;size=(10.8,5.7*n)
 elif f['panels'][0].get('parameter_break'):size=(10.5,6.4)
 elif any(p.get('dura_arrows') for p in f['panels']):size=(7.8,4.9*n);nc=1;nr=n
 elif n==1:
  p=f['panels'][0];height=4.4 if p['kind']=='scatter' else max(3.1,.28*len(p['series'])+1.0) if p['kind']=='heatmap' else max(3.3,.29*len(p['categories'])+1.1) if p['kind']=='point' else 3.9
  size=(max(7.6,.45*len(p['categories'])) if p['kind'] in ['bar','line'] else 7.6,height)
 elif n==3:size=(11.5,3.8)
 else:
  rowheight=max([3.3]+[.27*len(p['categories'])+1.2 for p in f['panels'] if p['kind']=='point']+[.28*len(p['series'])+1.0 for p in f['panels'] if p['kind']=='heatmap'])
  size=(10.5,rowheight*nr)
 fig,axes=plt.subplots(nr,nc,figsize=size,squeeze=False)
 for i,p in enumerate(f['panels']):panel_render(axes.flat[i],p,i)
 for ax in list(axes.flat)[n:]:ax.axis('off')
 # Leave enough room for labels and legends without descriptive paragraphs.
 fig.tight_layout(pad=1.2,h_pad=3.0,w_pad=2.1)
 for ext in ['pdf','svg','png']:
  fig.savefig(out/ext/(f['id']+'.'+ext),dpi=250 if ext=='png' else 300,bbox_inches='tight')
 # Save all actual numbers that were drawn, including the per-seed points.
 rows=[];seed=[]
 for i,p in enumerate(f['panels']):
  for r in p['rows']:
   a={k:v for k,v in r.items() if k not in ['seed_values','model_seeds']};rows.append(dict(panel=chr(97+i),panel_title=p['title'],unit=p['unit'],**a))
   for ri,v in enumerate(r.get('seed_values',[])):
    ss=r.get('model_seeds',[]);seed.append(dict(panel=chr(97+i),category=r['category'],series=r.get('series',''),model_seed=ss[ri] if ri<len(ss) else None,replicate_index=ri+1,value=v))
  seed.extend(dict(panel=chr(97+i),**r) for r in p.get('seeds',[]))
 pd.DataFrame(rows).to_csv(out/'source_data'/(f['id']+'_plot.csv'),index=False,encoding='utf-8-sig')
 if seed:pd.DataFrame(seed).to_csv(out/'source_data'/(f['id']+'_seeds.csv'),index=False,encoding='utf-8-sig')
 gains=[];gain_seeds=[]
 for i,p in enumerate(f['panels']):
  for g in p.get('gains',[]):
   a=next(r for r in p['rows'] if r['category']==g['category'] and r['series']=='Without DURA');b=next(r for r in p['rows'] if r['category']==g['category'] and r['series']=='With DURA')
   gains.append(dict(panel=chr(97+i),category=g['category'],direct_mean_pct=a['estimate'],DURA_mean_pct=b['estimate'],gain_pp=g['estimate'],gain_SD_pp=g['sd'],paired_seeds=g['seed_n']))
   gain_seeds.extend(dict(panel=chr(97+i),category=g['category'],model_seed=s,gain_pp=v) for s,v in zip(g['model_seeds'],g['seed_values']))
 if gains:
  pd.DataFrame(gains).to_csv(out/'source_data'/(f['id']+'_gains.csv'),index=False,encoding='utf-8-sig');pd.DataFrame(gain_seeds).to_csv(out/'source_data'/(f['id']+'_gain_seeds.csv'),index=False,encoding='utf-8-sig')
 plt.close(fig)
 return dict(id=f['id'],selected=f['selected'],panels=n,data_rows=len(rows),seed_rows=len(seed))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--selected-only',action='store_true');p.add_argument('--ids',nargs='*');p.add_argument('--force',action='store_true');args=p.parse_args()
 root=args.root;cat=json.loads((root/'catalog.json').read_text(encoding='utf-8'))
 for sub in ['pdf','svg','png','source_data']:(root/sub).mkdir(exist_ok=True)
 logs=[]
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always')
  for f in cat['figures']:
   if args.selected_only and not f['selected'] or args.ids and f['id'] not in args.ids:continue
   if (root/'pdf'/(f['id']+'.pdf')).exists() and not args.force:continue
   logs.append(render(f,root));print('RENDERED',f['id'],flush=True)
 warnings_out=[str(w.message) for w in ww]
 (root/('render_log_'+str(int(time.time()))+'.json')).write_text(json.dumps(dict(figures=logs,warnings=warnings_out),ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(rendered=len(logs),warnings=warnings_out[:8]),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
