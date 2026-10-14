import argparse,copy,json
from pathlib import Path
import torch
from . import design as d,model,components
from .checks import run as checks
from .objective import objective,anchor_groups,distribution,RXS
def run():
 from experiments.cvs_feature_multiscale50.predict import source_proof
 source=source_proof(d.baseline_config(d.SEEDS[0]));ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 student=model.build(d.config(d.rows()[0]),'cpu');student.load_state_dict(ck['model'],strict=True);teacher=copy.deepcopy(student);teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False);student.eval()
 with torch.no_grad():z=student.forward_identity_only(torch.zeros(2,2,256))
 if z['id_feat_joint'].shape!=(2,160) or not torch.isfinite(z['tx_logits']).all():raise ValueError('Real checkpoint no-query smoke failed')
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_phase1_overlay.model import native_modules
 L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V
 torch.set_num_threads(2);gen=torch.Generator().manual_seed(d.SEEDS[0]+93217);anchors=[];probes=[]
 for r in RXS:
  for tx in range(6):
   for day in (1,2,3):
    cell=torch.nonzero((L['rx']==r)&(L['y']==tx)&(L['day']==day)).flatten();chosen=cell[torch.randperm(70,generator=gen)[:10]];anchors.extend(chosen[:2].tolist());probes.extend(chosen[2:].tolist())
 if len(set(anchors+probes))!=900:raise ValueError('L anchor/probe overlap')
 with torch.no_grad():bank=teacher.forward_identity_only(L['x'][anchors])['id_feat_joint'].reshape(5,6,6,160);ot=teacher.forward_identity_only(L['x'][probes]);tp=components.parts(teacher,ot)
 student.train();torch.manual_seed(d.SEEDS[0]+94119);os=student.forward_identity_only(native_modules()._strong_augment(L['x'][probes],.015));sp=components.parts(student,os);loss,raw,facts=objective(sp,tp,bank,L['rx'][probes],'CROSS_RESP')
 head=model.classifier(student);e=student.id_backbone.encoder;params=[head.weight,head.gain,e.response_gain,next(e.core.id_backbone.t1.parameters()),next(e.response_projection.parameters())];grads=torch.autograd.grad(loss,params,allow_unused=True)
 if any(g is not None for g in grads[:3]) or grads[3] is None or grads[3].norm()<=0 or facts['nonzero_relation_fraction']<.5:raise ValueError('Source auxiliary classW/gain/learnability smoke failed')
 semantic=[]
 with torch.no_grad():
  for r in RXS:
   ix=L['rx'][probes]==r;y=L['y'][probes][ix];mass=[]
   for other,a in anchor_groups(bank,r,True):
    weights=distribution(tp['joint'][ix],a);anchor_TX=torch.arange(6).repeat_interleave(5);mass.append((weights*(anchor_TX[None,:]==y[:,None])).sum(-1))
   semantic.append(dict(RX=r,same_TX_relation_mass=float(torch.stack(mass).mean()),chance=1/6,source_RX_seen=True))
 simple={k:v for k,v in facts.items() if not torch.is_tensor(v)}
 return dict(status='PASS',real_checkpoint_noquery=True,no_inherited_initialization=True,strict_student_and_ownEMA=True,query_access=False,head_fit_updates=0,checks=checks(),source_L_diagnostic=dict(anchors=180,disjoint_probe_packets=720,forward_packets=1620,V_fit=False,U_IQ_read=False,U_truth_read=False,relation=float(raw['relation'].detach()),response=float(raw['response'].detach()),actual_t1_aux_gradient=float(grads[3].norm()),ref_projection_aux_gradient=float(grads[4].norm()) if grads[4] is not None else 0.,class_W_aux_gradient=None,PA_gain_aux_gradient=None,ref_gain_aux_gradient=None,facts=simple,semantic=semantic,limitation='Legal source L seen during historical source training. Not unseen RX evidence or actual online hidden U correctness; sampling labels do not enter loss.'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);d.write(p.parse_args().output,run());print(json.dumps(dict(status='PASS')))
