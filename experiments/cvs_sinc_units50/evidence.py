import gzip,json
from pathlib import Path
import torch
from . import design as d
from .optimizer import NAMES
from .observer import summarize_snapshots
from .predict import source_proof
from experiments.cvs_feature_diagnostics.sinc_change import compare
from experiments.cvs_phase1_overlay.model import native_modules

def recount(row):
 c=d.config(row);d.validate(c)
 if c['baseline_reuse']:raise ValueError('No new baseline training witness')
 p=source_proof(c);obs=d.read(p/'sinc_optimizer_observation.json')
 if obs['steps']!=11100 or obs['original_U_observation']['U_packets']!=963900 or obs['extra_model_forwards'] or obs['head_fit_updates'] or obs['U_truth_read']:raise ValueError('Original unit-run data budget differs')
 with gzip.open(p/'sinc_optimizer_steps.jsonl.gz','rt',encoding='utf8') as stream:records=[json.loads(line) for line in stream]
 if len(records)!=11100 or [r['step'] for r in records]!=list(range(1,11101)):raise ValueError('Actual effective optimizer records incomplete')
 mode=c['sinc_optimizer_units']['mode'];mult=25000000. if mode=='NORMALIZED_UNITS' else 1.
 from experiments.cvs_phase1_repair.design import cosine_lr
 a=d.make_args(c,'cpu')
 for r in records:
  expected=cosine_lr(float(a.lr),(r['step']-1)//222+1,50,float(a.phase1_lr_min))
  if abs(r['base_lr']-expected)>1e-15 or abs(r['sinc_lr']-expected*mult)>1e-9 or r['sinc_eps']!=1e-8/mult or r['sinc_weight_decay']!=float(a.weight_decay)/mult:raise ValueError('Actual unit optimizer schedule differs')
 raw=[]
 for epoch in d.SINC_DIAGNOSTIC_EPOCHS:
  q=torch.load(p/f'sinc_optimizer_gradients_updates_e{epoch}.pt',map_location='cpu',weights_only=False)
  if len(q)!=13 or any(r['epoch']!=epoch or (r['step']-(epoch-1)*222)%16 for r in q):raise ValueError('Sinc actual snapshot period differs')
  for r in q:
   if tuple(r['parameters'])!=NAMES:raise ValueError('Actual Sinc tensor coordinates differ')
   for v in r['parameters'].values():
    if int(v['optimizer_step'])!=r['step'] or any(v[k].shape!=(24,1) or not torch.isfinite(v[k]).all() for k in ('before','after','gradient','first_moment','second_moment')):raise ValueError('Actual Sinc derivative/update invalid')
  raw.extend(q)
 summary=summarize_snapshots(raw)
 if summary!=d.read(p/'sinc_optimizer_update_records.json')['records']:raise ValueError('Independent actual update recount differs')
 initial=torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];checkpoint=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if checkpoint['epoch']!=50 or checkpoint['a1_ema_successful_updates']!=11100 or checkpoint['ema_model'] is None:raise ValueError('Original full ownEMA budget differs')
 native_modules();from model import SincConv1d
 return dict(status='VERIFIED',row_id=row['row_id'],mode=mode,actual_steps=11100,actual_U_packets=963900,actual_Sinc_parameter_records=104,actual_snapshot_batches=52,student_Sinc=compare(initial,checkpoint['model'],SincConv1d),ownEMA_Sinc=compare(initial,checkpoint['ema_model'],SincConv1d),source_only=True,head_fit_updates=0,extra_model_forwards=0,U_truth_read=False,query_read=False)
