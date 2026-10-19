"""Bounded same-IQ frozen source comparison; no target, fitted head or persistence update."""
from pathlib import Path
import importlib.util,json,time
import numpy as np
import torch
from . import design as d
from .channel import tensor_iq
from .quality import panel
METHODS=('PAMF','csil','mopc_hr')
def inherited(method):
 p=Path(d.PROJECT)/'paper_reproduction/runs/20260930-phase1-native-baselines-practical-manysig-m5-r01'/(method+'-s392005')
 init=d.read(p/'initialization.json');done=d.read(p/'completion.json');contract=d.read(p/'source_contract.json');current=d.read(d.SOURCE)
 if not init['scratch_only'] or init['ancestors'] or init['checkpoint_sources'] or init['target_contact'] or init['target_access'] or init['selection']!='fixed_last_epoch' or done['epoch']!=200 or done['steps']!=9800 or done['target_access'] or contract['role_ids']!=current['role_ids'] or contract['equalized']!=1 or contract['out_len']!=256 or not contract['normalize'] or contract['classes']!=d.CLASSES:raise ValueError('Historical source checkpoint provenance/physical roles mismatch')
 cp=torch.load(p/'last.pt',map_location='cpu',weights_only=False)
 if cp['source_contract']!=contract or cp['initialization']!=init or cp['epoch']!=200 or cp['selection']!='fixed_last_epoch' or cp['method']!=method:raise ValueError('Actual historical checkpoint differs from audited provenance')
 ref=Path(d.PROJECT)/'releases/native_comparisons_20260930_r01';s=importlib.util.spec_from_file_location('historical84_models',ref/'comparison_suite/models.py');mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
 m=mod.factory(method,cp['num_classes'],cp['num_receivers'],**cp['model_kwargs']);m.load_state_dict(cp['model'],strict=True)
 return m.eval().requires_grad_(False),dict(method=method,model_seed=392005,epoch=200,steps=9800,origin='scratch;fixed last;exact physical roles;no target ancestors',head='original native classifier',checkpoint=str(p/'last.pt'))
def logits(m,x,device,method):
 values=[]
 with torch.no_grad():
  for i in range(0,len(x),180):
   batch=tensor_iq(x[i:i+180],device) if not torch.is_tensor(x) else x[i:i+180].to(device)
   z=m.forward_identity_only(batch)['tx_logits'] if method=='PAMF' else m(batch)['tx_logits']
   if z.shape!=(len(batch),6) or not torch.isfinite(z).all():raise ValueError('Finite same-input six-class default-head logits required')
   values.extend(z.cpu().tolist())
 return np.array(values,np.float32)
def statistics(y,a,b=None):
 ok=a.argmax(-1)==y
 result=dict(n=len(y),correct=int(ok.sum()),accuracy=float(ok.mean()))
 if b is not None:
  bo=b.argmax(-1)==y;result.update(teacher_correct_PAMF_wrong=int((ok&~bo).sum()),teacher_wrong_PAMF_correct=int((~ok&bo).sum()),both_wrong=int((~ok&~bo).sum()),net_repair=int(ok.sum()-bo.sum()),union_accuracy=float((ok|bo).mean()))
 return result
def fresh(x,ids,rx,day,scene,draw,ch,residual):
 cfg=ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');out=[]
 for i,pid in enumerate(ids):
  iq=np.asarray(x[i].tolist(),np.float64);normalized,_=ch.normalize_input(iq[0]+1j*iq[1]);session=f'rx{int(rx[i])}:day{int(day[i])}:eq1';hw=ch.receiver_for_session(cfg,20261016423,session);seed=ch.stable_seed(20261019844,'source84/fresh',scene,draw,pid);v,_=residual.ResidualChannel(cfg,seed,hw).process(normalized);out.append(np.stack((v.real,v.imag)))
 return np.asarray(out,np.float32)
def run(device='cpu'):
 torch.set_num_threads(2);out=d.BASE/'source_comparison'
 if out.exists():raise FileExistsError('Reconcile existing source comparison')
 out.mkdir();tic=time.perf_counter();data,roles=panel(torch);obs=np.load(d.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False);physics=np.load(d.PANEL/'observations/physics.npz',allow_pickle=False);pred={};provenance={}
 from experiments.cvs_noise_shape_factorial43 import design as rd,model as rm
 for method in METHODS:
  if method=='PAMF':
   cfg=rd.config(next(r for r in rd.rows() if r['arm']=='B01'));m=rm.own_model(cfg,device);prov=dict(model_seed=2026092701,epoch=30,origin='own scratch20+own10;fixed final;exact source roles',checkpoint=cfg['output_root']+'/final_ssdg.pth',head='original scale30 cosine')
  else:m,prov=inherited(method);m=m.to(device)
  before={k:v.clone() for k,v in m.state_dict().items()};z=dict(clean=logits(m,data['x'],device,method),satellite=np.empty((2,2,1800,6),np.float32))
  for draw in range(2):
   for j,si in enumerate((1,2)):z['satellite'][draw,j]=logits(m,obs[draw,si,0],device,method)
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen source inference updated parameters or BatchNorm')
  pred[method]=z;provenance[method]=prov;np.savez_compressed(out/(method+'.npz'),ids=data['ids'],TX=data['y'].tolist(),clean=z['clean'],satellite=z['satellite'])
  if method=='PAMF':student=m
  else:del m
 y=np.asarray(data['y'].tolist());result={};strata=[]
 for method,z in pred.items():
  result[method]=dict(clean=statistics(y,z['clean'],pred['PAMF']['clean']),satellite={})
  for j,si in enumerate((1,2)):
   scene=('practical_mid_urban','practical_low_urban')[j];a=z['satellite'][:,j].reshape(-1,6);b=pred['PAMF']['satellite'][:,j].reshape(-1,6);yy=np.tile(y,2);result[method]['satellite'][scene]=statistics(yy,a,b)
   groups=[('overall','ALL',np.ones(3600,bool))]
   for key,names in [('state',('LOS','PARTIAL','BLOCKED')),('lock',('locked','degraded','unlocked'))]:
    vv=physics[key][:,si,0].reshape(-1);groups.extend((key,name,vv==i) for i,name in enumerate(names))
   q=physics['signal_to_added_noise_db'][:,si,0].reshape(-1);code=np.searchsorted([3.,8.,15.],q,side='right');groups.extend(('signal_to_added_noise_db',name,code==i) for i,name in enumerate(('<3','3-8','8-15','>=15')))
   for dim,name,mask in groups:
    if mask.any():strata.append(dict(method=method,scene=scene,dimension=dim,stratum=name,**statistics(yy[mask],a[mask],b[mask])))
 # Fresh source L and V: same source channel prior and new seed, never training updates.
 cache=torch.load(d.SOURCE_CACHE/'source_cache.pt',map_location='cpu',weights_only=False);lab=cache['labeled'];contract=d.read(d.SOURCE)
 if set(lab['ids'])!=set(contract['role_ids']['L_s']):raise ValueError('Original L cache physical roles differ')
 indices=[]
 for yy in range(6):
  for rr in d.RX:
   for dd in (1,2,3):
    pool=[i for i in range(len(lab['ids'])) if int(lab['y'][i])==yy and int(lab['rx'][i])==rr and int(lab['day'][i])==dd];indices.extend(sorted(pool,key=lambda i:lab['ids'][i])[:20])
 if len(indices)!=1800:raise ValueError('Balanced L1800 required')
 L={k:lab[k][indices] for k in ('x','y','rx','day')};L['ids']=[lab['ids'][i] for i in indices]
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules();import leo_practical.channel as ch;import leo_practical.residual as residual
 beforefresh={k:v.clone() for k,v in student.state_dict().items()};fitting={}
 for role,v in [('L',L),('V',data)]:
  yy=np.asarray(v['y'].tolist());measures={'clean':statistics(yy,logits(student,v['x'],device,'PAMF'))}
  for scene in ('practical_mid_urban','practical_low_urban'):
   allz=[]
   for draw in range(2):allz.append(logits(student,fresh(v['x'],v['ids'],v['rx'],v['day'],scene,draw,ch,residual),device,'PAMF'))
   zz=np.stack(allz);measures[scene]=statistics(np.tile(yy,2),zz.reshape(-1,6));np.savez_compressed(out/(role+'_'+scene+'_fresh.npz'),ids=v['ids'],logits=zz,TX=yy)
  fitting[role]=measures
 if any(not torch.equal(v,student.state_dict()[k]) for k,v in beforefresh.items()):raise ValueError('Fresh source inference changed frozen student')
 answer=dict(status='SOURCE_COMPARISON_COMPLETE',same_observation=result,strata=strata,fresh_L_V=fitting,provenance=provenance,channel_draws=2,model_seed_counts={'PAMF':1,'csil':1,'mopc_hr':1},new_NN=45000,training_updates=0,head_fit_updates=0,target_access=False,formal_feedback=False,teacher_training_authorized_by_this_artifact=False,observed_training_channel_fit='not measured;fresh L assesses seen physical samples under unseen channel',seconds=time.perf_counter()-tic)
 d.write(out/'complete.json',answer);print(json.dumps(dict(status=answer['status'],same_observation=result,fresh_L_V=fitting)),flush=True);return answer
