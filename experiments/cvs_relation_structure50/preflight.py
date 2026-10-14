import argparse,copy,json
from pathlib import Path
import torch
from . import design as d,model,components
from .checks import run as checks
from .objective import objective,RXS

def run():
 from experiments.cvs_time_factorial50.provenance import source_proof
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_phase1_overlay.model import native_modules
 torch.set_num_threads(2);L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V
 gen=torch.Generator().manual_seed(d.SEEDS[0]+93217);anchors=[];probes=[]
 for r in RXS:
  for tx in range(6):
   for day in (1,2,3):
    cell=torch.nonzero((L['rx']==r)&(L['y']==tx)&(L['day']==day)).flatten()
    if len(cell)!=70:raise ValueError('Exact source L cell differs')
    chosen=cell[torch.randperm(70,generator=gen)[:4]];anchors.extend(chosen[:2].tolist());probes.extend(chosen[2:].tolist())
 if len(set(anchors+probes))!=360:raise ValueError('Source anchor/probe overlap')
 records=[]
 for arm in ('COMPACT_RESP','LONG_RESP'):
  row=next(r for r in d.rows() if r['model_seed']==d.SEEDS[0] and r['arm']==arm);c=d.config(row);tc=d.temporal_config(row);p=source_proof(tc);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if ck['epoch']!=50 or ck['candidate_id']!=tc['row_id'] or ck['run_id']!=d.temporal.RUN or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=11100:raise ValueError('Historical temporal source checkpoint differs')
  m=model.build(c,'cpu');m.load_state_dict(ck['model'],strict=True);t=copy.deepcopy(m);t.load_state_dict(ck['ema_model'],strict=True);m.eval();t.eval().requires_grad_(False)
  with torch.no_grad():
   out=m.forward_identity_only(torch.zeros(2,2,256))
   if out['id_feat_joint'].shape!=(2,160) or not torch.isfinite(out['tx_logits']).all():raise ValueError('Real temporal checkpoint no-query smoke')
   bank=t.forward_identity_only(L['x'][anchors])['id_feat_joint'].reshape(5,6,6,160);ot=t.forward_identity_only(L['x'][probes]);tp=components.parts(t,ot)
  m.train();torch.manual_seed(d.SEEDS[0]+94119);os=m.forward_identity_only(native_modules()._strong_augment(L['x'][probes],.015));sp=components.parts(m,os);loss,raw,facts=objective(sp,tp,bank,L['rx'][probes],'CROSS_RESP');h=model.classifier(m);e=m.id_backbone.encoder
  grads=torch.autograd.grad(loss,[h.weight,h.gain,e.response_gain,next(e.core.id_backbone.t1.parameters()),next(e.response_projection.parameters())],allow_unused=True)
  if any(g is not None for g in grads[:3]) or grads[3] is None or not torch.isfinite(grads[3]).all() or grads[3].norm()<=0 or facts['nonzero_relation_fraction']<.5:raise ValueError('Actual temporal relation auxiliary wiring failed')
  records.append(dict(arm=arm,checkpoint=str(p/'final_ssdg.pth'),source_roles='EXACT_MATCH',probe_count=180,extra_source_forward_packets=540,relation=float(raw['relation'].detach()),response=float(raw['response'].detach()),actual_t1_aux_gradient=float(grads[3].norm()),ref_projection_aux_gradient=float(grads[4].norm()) if grads[4] is not None else 0.,class_W_aux_gradient=None,PA_gain_aux_gradient=None,ref_gain_aux_gradient=None))
 return dict(status='PASS',real_checkpoint_noquery=True,strict_student_ownEMA=True,no_inherited_training=True,query_access=False,U_IQ_read=False,U_truth_read=False,head_fit_updates=0,checks=checks(),source_L_diagnostic=records,extra_source_forward_packets=1080,source_seen_only=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);d.write(p.parse_args().output,run());print(json.dumps(dict(status='PASS')))
