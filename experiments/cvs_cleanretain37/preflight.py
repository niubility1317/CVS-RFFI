"""Compatibility-only real CP smoke; no source IQ, target query or initialization."""
import torch
from . import design as d,model as api
def _checkpoint_smoke():
 out=d.SMOKE_SOURCE;original=d.read(d.SOURCE);contract=d.read(out/'source_contract.json');done=d.read(out/'completion.json');init=d.read(out/'initialization.json');freeze=d.read(out/'source_frozen.json')
 if contract['role_ids']!=original['role_ids'] or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Smoke CP must match full original physical training roles')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access'] or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Full-source scratch FINAL20 compatibility CP provenance differs')
 ck=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or ck['checkpoint_selection']!='final_only':raise ValueError('Smoke inheritance differs')
 m=api.build(d.config(d.rows()[0]),'cpu').eval();m.load_state_dict(ck['model'],strict=True)
 with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(2202)))['tx_logits']
 if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or m._full_calls or m._full_pending is not None:raise ValueError('Real default CP no-query smoke failed')
 r=dict(status='PASS',CP=str(out/'final_ssdg.pth'),historical_CP_cost='E20/4440; compatibility only, not parent training',new_training_init='scratch only',strict_load=True,NN=2,real_IQ=0,target_query=False,training_updates=0,head_fit_updates=0);d.write(d.BASE/'real_checkpoint_smoke.json',r);return r

def smoke():
 r=_checkpoint_smoke()
 from .synthetic import run
 run()
 return r
