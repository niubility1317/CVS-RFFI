"""Recount fixed source-V logits by full practical state; no model or IQ access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,csv,gzip,json,math,time
from pathlib import Path
import numpy as np

SEED=2026092701
RUN30='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
RUN50='20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01'
PANEL_RUN='20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CONDITIONS=('full','no_noise/fixed','no_noise/total','flat_H/fixed','flat_H/total','no_carrier/fixed','no_carrier/total','no_RX_IQ/fixed','no_RX_IQ/total','actual_power_quality','edge_boundary')
STATES=('LOS','PARTIAL','BLOCKED');LOCKS=('locked','degraded','unlocked')
ARMS=('NOAS30','AS30','NOAS50','AS50')
PAIRS=(('AS50','AS30'),('NOAS50','NOAS30'),('AS50','NOAS50'),('AS30','NOAS30'))
LOGIT_SCALE=30.
# Original P52 bins. These describe added noise or quality proxies, never total SNR.
QUALITY_EDGES=(-math.inf,3.,8.,15.,math.inf)
QUALITY_LABELS=('<3','3-8','8-15','>=15')
NUMERIC_BINS={
 'signal_to_added_noise_db':(QUALITY_EDGES,QUALITY_LABELS,'dB; signal/added-noise only'),
 'actual_waveform_quality_db':(QUALITY_EDGES,QUALITY_LABELS,'dB; waveform power/added-noise variance'),
 'white_proxy_quality_db':(QUALITY_EDGES,QUALITY_LABELS,'dB; white-input power proxy/added-noise variance'),
 'waveform_minus_white_power_db':((-math.inf,-3.,-1.,1.,3.,math.inf),('<-3','-3--1','-1-1','1-3','>=3'),'dB; waveform minus white proxy'),
 'abs_output_frequency_hz':((0.,100.,1000.,10000.,math.inf),('<100','100-1000','1000-10000','>=10000'),'absolute residual frequency, Hz'),
 'noise_coloration_dynamic_range_db':((-math.inf,1e-9,3.,10.,math.inf),('flat(<1e-9)','1e-9-3','3-10','>=10'),'10log10(max/min positive equalizer power spectrum), dB'),
 'residual_kernel_energy_db':((-math.inf,-20.,-10.,0.,math.inf),('<-20','-20--10','-10-0','>=0'),'10log10(sum(abs(residual kernel)**2)), dB; zero energy enters lowest bin'),
 'interior_count':((0.,128.,192.,224.,256.,math.inf),('0-127','128-191','192-223','224-255','256'),'unaffected outputs out of IQ256; producer max(0,256-kernel_length+1)'),
}
PHYSICS_FIELDS=('state','lock','signal_to_added_noise_db','actual_waveform_quality_db','white_proxy_quality_db','waveform_minus_white_power_db','output_frequency_hz','agc_limited','noise_coloration_dynamic_range_db','residual_kernel_energy','interior_count','quality_snr_db','estimated_frequency_hz','agc_gain_db','kernel_length','numerical_delay','residual_scale')


def read(path):
 return json.loads(Path(path).read_text(encoding='utf8'))

def write(path,value):
 with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def csv_write(path,records):
 fields=list(dict.fromkeys(k for row in records for k in row))
 with Path(path).open('x',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)

def rows(project):
 runs=Path(project)/'runs'
 return {arm:runs/(RUN50 if arm.endswith('50') else RUN30)/(( 'B01' if arm.startswith('AS') else 'B00')+'-s'+str(SEED)) for arm in ARMS}

def check_roles(role,n=1800):
 if role.get('role')!='V' or role.get('counts')!={'L_s':6300,'U_s':56700,'V':27000} or role.get('V_panel')!=n or role.get('held_RX') is not None or role.get('U_truth_used') is not False or role.get('target_access') is not False or role.get('updates')!=0 or role.get('extra_Ldev',False):
  raise ValueError('Original legal V role only; no U truth, held RX, Ldev or updates')
 ids=np.asarray(role.get('physical_ids'))
 if ids.shape!=(n,) or ids.dtype.kind not in 'US' or len(set(ids.tolist()))!=n:raise ValueError('Unique ordered source physical IDs required')
 return ids

def check_completion(completion,n=1800):
 if completion.get('status')!='SOURCE_OBSERVATIONS_COMPLETE' or completion.get('unique_V')!=n or completion.get('draws')!=2 or tuple(completion.get('scenes',()))!=SCENES or tuple(completion.get('conditions',()))!=CONDITIONS or completion.get('target_access') is not False or completion.get('training_updates')!=0:
  raise ValueError('Fixed source draw/scene/condition order required; condition0 is full')

def validate_predictions(x,ids,reference=None,n=1800):
 keys=('ids','TX','RX','day','clean','satellite')
 if any(k not in x for k in keys) or x['clean'].shape!=(n,6) or x['satellite'].shape!=(2,3,n,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Finite default-head source logits required')
 if not np.array_equal(x['ids'],ids):raise ValueError('Source physical-ID order differs')
 for k in ('TX','RX','day'):
  if x[k].shape!=(n,) or x[k].dtype.kind not in 'iu':raise ValueError('Integer source role metadata required')
  if reference is not None and not np.array_equal(x[k],reference[k]):raise ValueError('Four-row source '+k+' alignment differs')
 if np.any((x['TX']<0)|(x['TX']>=6)) or not set(x['RX'].tolist())<={1,3,4,6,8} or not set(x['day'].tolist())<={1,2,3}:raise ValueError('Original six TX/all-source RX/day roles required')
 return x

def validate_physics(physics,n=1800):
 for key,value in physics.items():
  if value.shape!=(2,3,len(CONDITIONS),n):raise ValueError('Physics shape differs: '+key)
 for key,limit in [('state',3),('lock',3)]:
  if key not in physics or not np.isfinite(physics[key]).all() or np.any((physics[key]<0)|(physics[key]>=limit)) or np.any(physics[key]!=physics[key].astype(int)):raise ValueError('State/lock indices differ')
 for key in ('signal_to_added_noise_db','actual_waveform_quality_db','white_proxy_quality_db','agc_limited','interior_count'):
  if key not in physics:raise ValueError('Required source physics field absent: '+key)
 return physics

def check_records(records,physics,x,n=1800):
 """Validate named draws against ordered predictions and condition0 metadata."""
 total=0;scalars={k:set() for k in ('boundary','numerical_delay','kernel_length','interior_count','channel_equalization_applied','phase_tracking_applied','true_total_SNR_known')}
 for record in records:
  draw=total//(3*n);si=total//n%3;i=total%n
  if draw>=2 or record.get('draw')!=draw or record.get('scene')!=SCENES[si] or record.get('index')!=i or record.get('physical_id')!=str(x['ids'][i]):raise ValueError('Source JSONL draw/scene/index/physical-ID order differs')
  if any(record.get(k)!=int(x[k][i]) for k in ('TX','RX','day')):raise ValueError('Source JSONL TX/RX/day differs')
  if record.get('session')!='rx'+str(int(x['RX'][i]))+':day'+str(int(x['day'][i]))+':eq1' or type(record.get('seed')) is not int:raise ValueError('Source JSONL RX/day session or named draw seed differs')
  conditions=record.get('conditions',{})
  if tuple(conditions)!=CONDITIONS:raise ValueError('Source JSONL condition order differs; full must be condition0')
  full=conditions['full']
  if full.get('state')!=STATES[int(physics['state'][draw,si,0,i])] or full.get('receiver_lock_status')!=LOCKS[int(physics['lock'][draw,si,0,i])]:raise ValueError('JSONL state/lock differs from physics condition0')
  for key in PHYSICS_FIELDS:
   if key in ('state','lock') or key not in physics:continue
   if key not in full:raise ValueError('Source JSONL full metadata absent: '+key)
   a=physics[key][draw,si,0,i];b=full[key]
   if b is None:
    if not np.isnan(a):raise ValueError('Source JSONL missing/numeric physics differs: '+key)
   elif not math.isfinite(float(b)) or not np.isclose(a,float(b),atol=1e-10,rtol=1e-10):raise ValueError('Source JSONL full numeric metadata differs: '+key)
  if full.get('true_total_SNR_known') is not False:raise ValueError('Added-noise proxy cannot be relabelled total SNR')
  if full.get('channel_equalization_applied') is not False:raise ValueError('Existing source full condition requires original no-EQ residual')
  for key in scalars:
   if key in full:scalars[key].add(full[key])
  total+=1
 if total!=2*3*n:raise ValueError('Source JSONL full draw matrix incomplete')
 return {'records':total,'full_only':True,'producer_observed_values':{k:sorted(v,key=str) for k,v in scalars.items()}}

def load_inputs(project,n=1800):
 project=Path(project);panel=project/'runs'/PANEL_RUN/'diagnostic';role=read(panel/'data_roles.json');ids=check_roles(role,n);completion=read(panel/'observations/completion.json');check_completion(completion,n)
 with np.load(panel/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in PHYSICS_FIELDS if k in f.files}
 validate_physics(physics,n);data={};checks={}
 for arm,root in rows(project).items():
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as f:x={k:f[k].copy() for k in ('ids','TX','RX','day','clean','satellite')}
  validate_predictions(x,ids,data.get('NOAS30'),n);q=read(root/'source_qualification/complete.json')
  if q.get('status')!='SOURCE_READOUT_COMPLETE' or q.get('target_access') is not False or q.get('formal_scores_read') is not False or q.get('training_updates')!=0 or q.get('head_fit_updates')!=0 or q.get('NN')!=7*n:raise ValueError('Existing source-only default-head readout incomplete: '+arm)
  config=q.get('config',{});expected_run=RUN50 if arm.endswith('50') else RUN30
  if config.get('model_seed')!=SEED or config.get('row_id')!=root.name or config.get('run_id')!=expected_run:raise ValueError('Fixed source row/model seed/run differs: '+arm)
  clean=float((x['clean'].argmax(-1)==x['TX']).mean());sat=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if abs(clean-q['clean_accuracy'])>1e-12 or not np.allclose(sat,q['source_accuracies'],atol=1e-12,rtol=0):raise ValueError('Recount differs from source readout: '+arm)
  data[arm]=x;checks[arm]={'clean_accuracy':clean,'source_accuracies':sat.tolist(),'source_qualification':str(root/'source_qualification')}
 with gzip.open(panel/'observations/physical_records.jsonl.gz','rt',encoding='utf8') as f:audit=check_records((json.loads(line) for line in f),physics,data['NOAS30'],n)
 return data,physics,{'panel':str(panel),'source_role':'V','physical_count':n,'model_seed_count':1,'channel_draws':2,'condition':'full','row_recounts':checks,'record_alignment':audit,'missing_physics_fields':[k for k in PHYSICS_FIELDS if k not in physics]}

def bin_codes(values,edges):
 values=np.asarray(values,float);codes=np.full(values.shape,-1,np.int16)
 valid=~np.isnan(values)
 codes[valid]=np.searchsorted(np.asarray(edges[1:-1]),values[valid],side='right')
 return codes

def masks(physics):
 n=len(physics['state']);allmask=np.ones(n,bool)
 result=[('overall','ALL',allmask)]
 for key,labels in [('state',STATES),('lock',LOCKS)]:result.extend((key,label,physics[key]==i) for i,label in enumerate(labels))
 result.extend(('state_lock',s+'/'+l,(physics['state']==i)&(physics['lock']==j)) for i,s in enumerate(STATES) for j,l in enumerate(LOCKS))
 available=dict(physics)
 if 'output_frequency_hz' in available:available['abs_output_frequency_hz']=np.abs(available['output_frequency_hz'])
 if 'residual_kernel_energy' in available:
  e=available['residual_kernel_energy'];db=np.full(e.shape,np.nan);valid=(e>=0)&~np.isnan(e)
  with np.errstate(divide='ignore'):db[valid]=10*np.log10(e[valid])
  available['residual_kernel_energy_db']=db
 for key,(edges,labels,units) in NUMERIC_BINS.items():
  if key not in available:continue
  codes=bin_codes(available[key],edges)
  result.extend((key,label,codes==i) for i,label in enumerate(labels));result.append((key,'MISSING',codes==-1))
 if 'agc_limited' in available:
  a=available['agc_limited'];result.extend([('agc_limited','not_limited',a==0),('agc_limited','limited',a==1),('agc_limited','MISSING',~np.isin(a,[0,1]))])
 if all(key in available for key in ('actual_waveform_quality_db','white_proxy_quality_db')):
  wave=bin_codes(available['actual_waveform_quality_db'],QUALITY_EDGES);white=bin_codes(available['white_proxy_quality_db'],QUALITY_EDGES);valid=(wave>=0)&(white>=0)
  result.extend([('waveform_vs_white_quality_bin','waveform_lower',valid&(wave<white)),('waveform_vs_white_quality_bin','same_bin',valid&(wave==white)),('waveform_vs_white_quality_bin','waveform_higher',valid&(wave>white)),('waveform_vs_white_quality_bin','MISSING',~valid)])
 return result

def metric(y,z,cohort):
 y=np.asarray(y);z=np.asarray(z);cohort=np.asarray(cohort,bool);p=z.argmax(-1);n=len(y);cm=np.bincount((y*6+p).astype(int),minlength=36).reshape(6,6);correct=int(np.trace(cm));den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 if n:
  other=z.copy();other[np.arange(n),y]=-np.inf;margin=z[np.arange(n),y]-other.max(1);mean=float(margin.mean());median=float(np.median(margin))
 else:mean=median=None
 cohort_n=int(cohort.sum());retained=int(((p==y)&cohort).sum())
 return dict(count=n,correct=correct,accuracy=correct/n if n else None,macro_f1=float(f1.mean()) if n else None,confusion=cm.tolist(),mean_logit_margin=mean,median_logit_margin=median,mean_margin_over30=mean/LOGIT_SCALE if mean is not None else None,median_margin_over30=median/LOGIT_SCALE if median is not None else None,fixed_NOAS30_clean_correct_count=cohort_n,fixed_NOAS30_clean_correct_retained=retained,satellite_retention=retained/cohort_n if cohort_n else None,small_cell=n<30,empty_cell=n==0)

def paired(y,old,new,cohort):
 old=np.asarray(old)==y;new=np.asarray(new)==y;n=len(y);repair=int((~old&new).sum());damage=int((old&~new).sum());cohort=np.asarray(cohort,bool);nc=int(cohort.sum())
 return dict(count=n,control_correct=int(old.sum()),new_correct=int(new.sum()),repairs=repair,new_errors=damage,net_repairs=repair-damage,accuracy_delta_pp=100*(repair-damage)/n if n else None,fixed_NOAS30_clean_correct_count=nc,cohort_repairs=int((~old&new&cohort).sum()),cohort_new_errors=int((old&~new&cohort).sum()),control_retention=float(old[cohort].mean()) if nc else None,new_retention=float(new[cohort].mean()) if nc else None)

def analyze(data,physics):
 x=data['NOAS30'];n=len(x['TX']);cohort=x['clean'].argmax(1)==x['TX'];strata=[];effects=[];interactions=[];overalls={arm:{} for arm in ARMS}
 for scene in ('clean',*SCENES):
  si=SCENES.index(scene) if scene!='clean' else None
  for draw in ((-1,) if scene=='clean' else (0,1,-1)):
   ds=(0,1) if draw==-1 else (draw,);copies=1 if scene=='clean' else len(ds);y=np.tile(x['TX'],copies);c=np.tile(cohort,copies)
   logits={arm:row['clean'] if scene=='clean' else np.concatenate([row['satellite'][d,si] for d in ds]) for arm,row in data.items()}
   groups=[('overall','ALL',np.ones(len(y),bool))] if scene=='clean' else masks({key:np.concatenate([value[d,si,0] for d in ds]) for key,value in physics.items()})
   for key,values,allowed in [('RX',np.tile(x['RX'],copies),(1,3,4,6,8)),('day',np.tile(x['day'],copies),(1,2,3)),('TX',y,range(6))]:groups.extend((key,str(value),values==value) for value in allowed)
   for dimension,label,mask in groups:
    context=dict(scene=scene,draw=draw,dimension=dimension,stratum=label)
    current={}
    for arm in ARMS:
     m=metric(y[mask],logits[arm][mask],c[mask]);strata.append(dict(row_id=arm,model_seed=SEED,**context,**m));current[arm]=m
     if dimension=='overall':overalls[arm][scene+'__'+str(draw)]=m
    for new,old in PAIRS:
     q=paired(y[mask],logits[old][mask].argmax(-1),logits[new][mask].argmax(-1),c[mask]);q.update(new=new,control=old,**context,macro_f1_delta_pp=100*(current[new]['macro_f1']-current[old]['macro_f1']) if mask.any() else None,mean_margin_over30_delta=current[new]['mean_margin_over30']-current[old]['mean_margin_over30'] if mask.any() else None);effects.append(q)
    gains={arm:current[arm]['accuracy'] for arm in ARMS};value=100*((gains['AS50']-gains['NOAS50'])-(gains['AS30']-gains['NOAS30'])) if mask.any() else None
    # Per-observation integer correctness difference is the interaction numerator.
    correct={arm:current[arm]['correct'] for arm in ARMS};net=correct['AS50']-correct['NOAS50']-correct['AS30']+correct['NOAS30']
    interactions.append(dict(**context,count=int(mask.sum()),budget_x_As_net_correct=net,budget_x_As_accuracy_pp=value,not_Ag_interaction=True))
 return strata,effects,interactions,overalls

def verify_source_summary(summary,overalls):
 if summary.get('target_scores_read') is not False or summary.get('status')!='SOURCE_ANALYZED':raise ValueError('Existing source-only summary required')
 expected=summary.get('summary',{})
 for arm in ARMS:
  old=expected[arm]['panel']
  for scene in ('clean',*SCENES):
   new=overalls[arm][scene+'__-1']
   for before,after in [('accuracy','accuracy'),('macro_F1','macro_f1')]:
    if not math.isclose(old[scene][before],new[after],abs_tol=1e-12,rel_tol=0):raise ValueError('Existing source_summary recount differs: '+arm+'/'+scene+'/'+before)
 return {'status':'MATCH','rows':4,'views_per_row':4,'accuracy_and_macro_f1_reconciliations':32,'fullV_27000_not_recounted':True}

def bins_description():
 out={key:{'edges':[('-inf' if x==-math.inf else 'inf' if x==math.inf else x) for x in edges],'labels':list(labels)+['MISSING'],'interval':'[low,high); final upper open-ended','units':units} for key,(edges,labels,units) in NUMERIC_BINS.items()}
 out.update(state=list(STATES),lock=list(LOCKS),state_lock=[s+'/'+l for s in STATES for l in LOCKS],agc_limited=['not_limited','limited','MISSING'],waveform_vs_white_quality_bin=['waveform_lower','same_bin','waveform_higher','MISSING'],small_cell_count_below=30)
 return out

def plot(out,strata):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 dimensions=['state_lock','signal_to_added_noise_db','actual_waveform_quality_db','waveform_vs_white_quality_bin']
 colors={'NOAS30':'#94a3b8','AS30':'#d97706','NOAS50':'#475569','AS50':'#2563eb'}
 for scene in SCENES[1:]:
  fig,axes=plt.subplots(2,2,figsize=(18,10));pooled=[r for r in strata if r['scene']==scene and r['draw']==-1]
  for ax,dim in zip(axes.flat,dimensions):
   records=[r for r in pooled if r['dimension']==dim];labels=list(dict.fromkeys(r['stratum'] for r in records));xx=np.arange(len(labels))
   counts=[]
   for i,arm in enumerate(ARMS):
    seq=[next(r for r in records if r['row_id']==arm and r['stratum']==label) for label in labels]
    vals=[100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in seq]
    ax.bar(xx+(i-1.5)*.2,vals,.2,label=arm,color=colors[arm])
    if not counts:counts=[r['count'] for r in seq]
   ax.set(xticks=xx,xticklabels=[label+'\nn='+str(count) for label,count in zip(labels,counts)],ylim=(0,103),ylabel='Source V accuracy (%)',title=dim);ax.tick_params(axis='x',labelrotation=30 if len(labels)>5 else 0);ax.grid(axis='y',alpha=.2)
  axes.flat[0].legend(ncol=2);fig.suptitle(scene+' / one model seed / pooled two fixed source draws\nFull residual only; common physical packets; n includes repeated draws; no formal target states');fig.tight_layout()
  for ext in ('png','pdf'):fig.savefig(out/(scene+'_source_states.'+ext),dpi=170)
  plt.close(fig)

def run(project,out,source_summary=None,code_commit=None):
 out=Path(out)
 if out.exists():raise FileExistsError('Immutable new output root required; reconcile existing results')
 out.mkdir(parents=False);started=time.perf_counter();data,physics,audit=load_inputs(project);strata,effects,interaction,overalls=analyze(data,physics)
 summary_path=Path(source_summary) if source_summary else Path(project)/'runs'/RUN50/'analysis_auto_r01/source_summary.json'
 permitted=Path(project)/'runs'/RUN50
 if summary_path.name!='source_summary.json' or permitted.resolve() not in summary_path.resolve().parents:raise ValueError('Only existing P50 source_summary.json permitted')
 reconciled=verify_source_summary(read(summary_path),overalls)
 csv_write(out/'source_strata.csv',[{k:v for k,v in row.items() if k!='confusion'} for row in strata]);write(out/'source_confusions.json',strata);csv_write(out/'paired_effects.csv',effects);csv_write(out/'budget_interaction.csv',interaction)
 summary=dict(status='SOURCE_STATE_ANALYZED',model_seed=SEED,model_seed_count=1,channel_draws=2,source_unique_V=1800,condition='full',fixed_clean_cohort='NOAS30 clean correct; reused unchanged for every row and scene',fixed_clean_cohort_count=int((data['NOAS30']['clean'].argmax(-1)==data['NOAS30']['TX']).sum()),rows=overalls,alignment=audit,source_summary_reconciliation=reconciled,bins=bins_description(),logit_scale=LOGIT_SCALE,logit_scale_confirmed_from_code=True,logit_scale_evidence=['experiments/cvs_residual_identity/model.py:ResidualPhysicalHead.scale=30.0;classify(normalized linear)','experiments/cvs_reference_identity/model.py:ReferenceCVS uses residual_fusion default head (dependency confirmed by source audit)'],source_scene_count=3,source_high_urban_available=False,source_total_SNR_known=False,fullV_27000_not_recounted=True,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,training=0,scoring=0,formal_files_read=False,no_promotion=True)
 write(out/'source_summary.json',summary);plot(out,strata)
 report=dict(status='SOURCE_STATE_ANALYZED',code_commit=code_commit,model_seed_count=1,physical_source_V=1800,channel_draws=2,draws_are_not_model_seeds=True,source_strata=len(strata),paired_strata=len(effects),interaction_strata=len(interaction),NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,training=0,scoring=0,formal_files_read=False,no_promotion=True,target_feedback=False,counterfactual_classification=False,physical_causality_claimed=False,wall_seconds=time.perf_counter()-started,missing_physics_fields=audit['missing_physics_fields'],artifacts=['source_strata.csv','source_confusions.json','paired_effects.csv','budget_interaction.csv','source_summary.json']+[s+'_source_states.'+e for s in SCENES[1:] for e in ('png','pdf')],limitations=['Single model seed; two repeated channel draws do not establish cross-seed reproducibility.','Source state associations do not establish a unique noise/CFO/multipath causal mechanism.','Full-condition logits only; no E30/E50 counterfactual classification effects are estimated.','Fixed bins include empty cells; small-cell n<30 flags are descriptive.','Source panel is 1800 physical samples; full V27000 is not individually recounted here.'])
 write(out/'report.json',report);return report

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--source-summary',type=Path);parser.add_argument('--code-commit');args=parser.parse_args();print(json.dumps(run(args.project,args.out,args.source_summary,args.code_commit),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
