"""Regenerate manuscript tables from frozen, independently audited CSV exports."""
import argparse,csv,json,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--data-root',type=Path,default=ROOT/'source_data')
args=parser.parse_args();D=args.data_root;T=ROOT/'tables';T.mkdir(exist_ok=True)
def load(name):
 with (D/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
data=load('primary_metrics_219x7.csv');training=load('training_costs_219.csv');cost=load('frozen_cost_summary_64.csv')
assert len(data)==219*7 and len(training)==219 and len(cost)==64
F=['ibrff','sdg','riei','drift'];N=dict(ibrff='IB-RFF',sdg='SDG',riei='RIEI',drift='DRIFT',resnet18='ResNet18',resnet50='ResNet50')
SEEDS=['2026092701','2026092702','2026092703']
V=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
def pick(group,family,var='ce',reg='supervised',route=None):
 rr=[r for r in data if r['group']==group and r['family']==family and r['variant']==var and r['training_regime']==reg and (route is None or r['route']==route)]
 assert len(rr)==21,(group,family,var,reg,route)
 return rr
def vals(rr,v='clean',m='rx_equal_accuracy'):
 a=sorted([r for r in rr if r['view']==v],key=lambda r:r['model_seed']);assert [r['model_seed'] for r in a]==SEEDS
 return [float(r[m]) for r in a]
def avgsc(rr):return [st.mean(values) for values in zip(*(vals(rr,v) for v in V[1:]))]
def delta(a,b,v='clean'):return [y-x for x,y in zip(vals(a,v),vals(b,v))]
def pd(a,b):return [y-x for x,y in zip(avgsc(a),avgsc(b))]
def pm(xs,scale=100,signed=False):
 return r'\pmv{'+(('%+.2f' if signed else '%.2f')%(scale*st.mean(xs)))+'}{%.2f}'%(scale*st.stdev(xs))
def tab(name,caption,label,cols,head,rows,wide=True,note=''):
 if name not in ('main','dg_effect','pl','receiver_effects','generic','inference_cost'):return
 env='table*' if wide else 'table'
 s=[r'\begin{'+env+r'}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',r'\footnotesize',r'\setlength{\tabcolsep}{4.1pt}',r'\begin{tabular}{'+cols+'}',r'\toprule',' & '.join(head)+r' \\',r'\midrule']
 for row in rows:s.append(r'\midrule' if row is None else ' & '.join(map(str,row))+r' \\')
 s += [r'\bottomrule',r'\end{tabular}']
 if note:s += [r'\par\smallskip\begin{minipage}{'+(r'\textwidth' if wide else r'\columnwidth')+r'}\footnotesize '+note+r'\end{minipage}']
 s += [r'\end{'+env+'}','']
 if name=='inference_cost':
  s[0]=r'\noindent\begin{minipage}{\columnwidth}'+'\n'+r'\makeatletter\def\@captype{table}\makeatother'
  s[-2]=r'\end{minipage}'
 (T/(name+'.tex')).write_text('\n'.join(s),encoding='utf-8')
MAIN=[]
for f in F:
 for pl in (False,True):
  for var in ('ce','full'):
   MAIN.append((N[f],var,'PL' if pl else 'Sup.',pick('F0-pl' if pl else 'F0',f,var,'pl' if pl else 'supervised')))
for g,f,reg in [('strongce','resnet18','supervised'),('strongce-pl','resnet18','pl'),('strongce','resnet50','supervised')]:MAIN.append((N[f],'ce','PL' if reg=='pl' else 'Sup.',pick(g,f,'ce',reg)))
rows=[]
for i,(name,var,reg,rr) in enumerate(MAIN):
 if i in (4,8,12,16):rows.append(None)
 rows.append([name,var.upper() if var=='ce' else 'Full',reg,pm(vals(rr)),pm(vals(rr,m='macro_f1')),pm(vals(rr,m='worst_rx')),pm(avgsc(rr))])
tab('main',r'F0 performance (\%). Every row uses the same satellite-concatenation policy. Values are mean $\pm$ sample SD over three model seeds.','main','lllrrrr',['Family','Version','Regime',r'Clean $A_{\rm RX}$','Macro-F1','Worst RX','Practical mean'],rows,
 note='Sup.: labeled-source supervision. PL: ordinary source pseudo-labeling jointly with satellite augmentation. Practical mean averages the six fixed stress conditions within each seed before computing seed statistics. Native training budgets differ across families; this is not an equal-compute ranking.')
rows=[];raw=[]
for f in F:
 for pl in (False,True):
  g='F0-pl' if pl else 'F0';reg='pl' if pl else 'supervised';a=pick(g,f,'ce',reg);b=pick(g,f,'full',reg);d=delta(a,b)
  rows.append([N[f],'PL' if pl else 'Sup.',pm(d,signed=True),str(sum(x>0 for x in d))+'/3',pm(pd(a,b),signed=True)])
  raw.append([N[f],'PL' if pl else 'Sup.']+['%+.4f'%(100*x) for x in d]+[pm(d,signed=True)])
tab('dg_effect','Within-seed DG effects: Full minus its own CE control (percentage points).','dg','llrrr',['Family','Regime','Clean difference','Positive seeds','Practical-mean difference'],rows)
tab('raw_pairs','Unrounded-source paired clean effects, displayed to four decimals (percentage points). Columns 01--03 denote the fixed model seeds.','raw','llrrrr',['Family','Regime','01','02','03',r'Mean $\pm$ SD'],raw)
rows=[];tr={r['row_id']:r for r in training}
for f,var in [(f,var) for f in F for var in ('ce','full')]+[('resnet18','ce')]:
 a=pick('strongce' if f=='resnet18' else 'F0',f,var);b=pick('strongce-pl' if f=='resnet18' else 'F0-pl',f,var,'pl');d=delta(a,b)
 ids=sorted({r['row_id'] for r in b});accept=[int(tr[x]['pl_accepted'])/int(tr[x]['pl_seen']) for x in ids]
 rows.append([N[f]+' '+('CE' if var=='ce' else 'Full'),pm(d,signed=True),str(sum(x>0 for x in d))+'/3',pm(pd(a,b),signed=True),pm(accept)])
tab('pl',r'Matched ordinary PL effects and cumulative acceptance (\%). Accuracy differences are in percentage points.','pl','lrrrr',['Identity model','Clean PL$-$Sup.','Positive seeds','Practical PL$-$Sup.','Accepted U'],rows)
rows=[];absolute=[]
for f in F:
 rows.append([N[f]]+[pm(delta(pick(g,f),pick(g,f,'full')),signed=True) for g in ['F0','F1','F2','chrono','ManyRx']])
 for var in ('ce','full'):
  absolute.append([N[f]+(' CE' if var=='ce' else ' Full')]+[pm(vals(pick(g,f,var))) for g in ['F0','F1','F2','chrono','ManyRx']])
tab('receiver_effects',r'Receiver-selection, future-date, and scale effects: Full$-$CE in clean $A_{\rm RX}$ (percentage points).','rxeffects','lrrrrr',['Family','F0','F1','F2','Future D3','ManyRx'],rows,
 note=r'Future D3 uses only the 42\,000 target-RX packets acquired on March 23. ManyRx uses receiver-equal accuracy over 16 target RX. The three F partitions overlap and are not independent datasets.')
tab('receiver_absolute',r'Absolute supervised clean accuracy across receiver/date protocols (\%).','rxabsolute','lrrrrr',['Model','F0','F1','F2','Future D3','ManyRx'],absolute)
rows=[]
routes=[('erm','ERM'),('mixstyle','MixStyle-crossRX'),('mixstyle_id','MixStyle-ID'),('fishr','Fishr-head'),('coral','Source-CORAL')]
a=pick('generic','resnet18',route='erm')
for route,name in routes:
 rr=pick('generic','resnet18',route=route);d=delta(a,rr)
 rows.append([name,pm(vals(rr)),pm(vals(rr,m='macro_f1')),pm(vals(rr,m='worst_rx')),r'---' if route=='erm' else pm(d,signed=True),r'---' if route=='erm' else str(sum(x>0 for x in d))+'/3'])
tab('generic',r'Same-backbone DG controls: ResNet18, 5000 updates, batch 150 (\%; differences in percentage points).','generic','lrrrrr',['Route',r'Clean $A_{\rm RX}$','Macro-F1','Worst RX',r'$\Delta$ versus ERM','Positive seeds'],rows)
rows=[]
for name,var,f in [('ResNet18 CE','ce','resnet18'),('IB-RFF CE','ce','ibrff'),('IB-RFF Full','full','ibrff')]:
 for g,percent in [('label01','1'),('label05','5'),('strongce' if f=='resnet18' else 'F0','10')]:
  rr=pick(g,f,var);ids=sorted({r['row_id'] for r in rr});updates=[int(tr[x]['updates']) for x in ids]
  u=str(updates[0]) if len(set(updates))==1 else str(min(updates))+'--'+str(max(updates))
  rows.append([name,percent,pm(vals(rr)),pm(vals(rr,m='macro_f1')),u])
tab('labels',r'Training-label efficiency with a separately labeled validation set of 27\,000 packets. Updates are total minibatch updates across all stages.','labels','lrrrr',['Model',r'Labeled training (\%)',r'Clean $A_{\rm RX}$ (\%)',r'Macro-F1 (\%)','Actual updates'],rows)
rows=[]
for name,var,reg,rr in MAIN:
 ids=sorted({r['row_id'] for r in rr});ts=[tr[x] for x in ids]
 epochs='/'.join('/'.join(t['epochs_by_stage'].split('/')) for t in ts)
 rows.append([name+' '+('CE' if var=='ce' else 'Full')+(' + PL' if reg=='PL' else ''),pm([float(t['source_seconds']) for t in ts],1),'/'.join(t['updates'] for t in ts),'/'.join(t['optimizer_steps'] for t in ts)])
tab('training_cost','Actual F0 training costs. Update triplets follow seeds 01/02/03; source seconds include the source-stage procedure and validation, excluding queue time and target inference.','training','lrrr',['Configuration','Source seconds','Minibatch updates','Optimizer updates'],rows)
rows=[]
for f,var in [(f,v) for f in F for v in ('ce','full')]+[('resnet18','strongce:erm'),('resnet50','strongce:erm')]:
 a=next(x for x in cost if x['origin']=='supervised' and x['protocol_id']=='F0' and x['family']==f and x['variant']==var and x['batch']=='1')
 b=next(x for x in cost if x['origin']=='supervised' and x['protocol_id']=='F0' and x['family']==f and x['variant']==var and x['batch']=='128')
 rows.append([N[f]+' '+('CE' if var in ('ce','strongce:erm') else 'Full'),'%.3f'%(int(a['identity_parameters'])/1e6),'%.2f'%(float(a['batch1_macs'])/1e6),r'\pmv{%.2f}{%.2f}'%(float(a['median_ms_mean']),float(a['median_ms_sample_sd'])),r'\pmv{%.2f}{%.2f}'%(float(b['median_ms_mean']),float(b['median_ms_sample_sd']))])
tab('inference_cost','Frozen identity-path cost on one RTX 3090. FP32, TF32 off; 100 warmups/1000 synchronized calls per batch and checkpoint. Latency is in ms.','inference','lrrrr',['Model','Params (M)','MACs (M)','Batch1','Batch128'],rows,wide=False,
 note='Latency summarizes three checkpoint medians (mean/sample SD). MACs exclude elementwise work; transfer and preprocessing are not timed. Memory and the full precision measurements remain in the source CSV.')
rows=[]
for name,var,reg,rr in MAIN:rows.append([name+' '+('CE' if var=='ce' else 'Full')+(' + PL' if reg=='PL' else '')]+['%.2f'%(100*st.mean(vals(rr,v))) for v in V])
tab('all_views',r'All seven F0 views: receiver-equal accuracy means (\%). H/M/L indicate elevation presets; S/U denote suburban/urban. Full seed SD and stratified scores accompany the source data.','views','lrrrrrrr',['Configuration','Clean','H-S','M-S','L-S','H-U','M-U','L-U'],rows)
rows=[]
for g in sorted({r['group'] for r in data if r['group'].startswith('count-')}):
 rr=pick(g,'resnet18');pieces=g.split('-');chain=pieces[1][1:];m=pieces[2][1:];budget=pieces[3];chain='A/B/C' if m=='5' else chr(64+int(chain))
 rows.append([chain,m,budget,pm(vals(rr)),pm(vals(rr,m='worst_rx'))])
tab('source_counts',r'Deduplicated source-count configurations: ResNet18, 10\,000 updates and batch180. Natural/fixed labeled sets are $1260m$/$1260$; validation is $5400m$.','counts','lllrr',['RX chain','Source RX $m$','L budget',r'Clean $A_{\rm RX}$ (\%)',r'Worst RX (\%)'],rows,
 note='The $m=1$ natural/fixed endpoints coincide; $m=5$ is shared by all chains. Eleven configurations and three seeds give 33 trained models, despite 54 plotted seed locations.')
print(json.dumps(dict(status='VERIFIED',source_rows=219,view_rows=1533,tables=len(list(T.glob('*.tex'))),statistics='mean/sample SD; matched seed deltas; no new scoring')))
