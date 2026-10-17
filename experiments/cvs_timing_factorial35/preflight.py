"""Compatibility-only real CP smoke; no source IQ, target query or initialization."""
import torch
from . import design as d,model as api
def smoke():
 d.write(d.BASE/'matched_native_arguments.json',d.matching_arguments());verify_controls()
 out=d.SMOKE_SOURCE;original=d.read(d.SOURCE);contract=d.read(out/'source_contract.json');done=d.read(out/'completion.json');init=d.read(out/'initialization.json');freeze=d.read(out/'source_frozen.json')
 if contract['role_ids']!=original['role_ids'] or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Smoke CP must match full original physical training roles')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access'] or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Full-source scratch FINAL20 compatibility CP provenance differs')
 ck=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or ck['checkpoint_selection']!='final_only':raise ValueError('Smoke inheritance differs')
 m=api.build(d.config(d.rows()[0]),'cpu').eval();m.load_state_dict(ck['model'],strict=True)
 with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(2202)))['tx_logits']
 if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or m._full_calls or m._full_pending is not None:raise ValueError('Real default CP no-query smoke failed')
 r=dict(status='PASS',CP=str(out/'final_ssdg.pth'),historical_CP_cost='E20/4440; compatibility only, not parent training',new_training_init='scratch only',strict_load=True,NN=2,real_IQ=0,target_query=False,training_updates=0,head_fit_updates=0);d.write(d.BASE/'real_checkpoint_smoke.json',r);return r

def verify_controls():
 for arm in ('B00','B01'):
  root=d.source_row_root(arm);c=d.control_config(arm);done=d.read(root/'source/completion.json');init=d.read(root/'source/initialization.json');frozen=d.read(root/'source/source_frozen.json');prediction=d.read(root/'prediction/complete.json');score=d.read(root/'scoring/complete.json')
  if done['config']!=c or done['epoch']!=30 or done['optimizer_steps']!=6660 or done['target_access'] or done['head_fit_updates'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or frozen['config']!=c or frozen['epoch']!=30 or frozen['selection']!='final_only' or frozen['target_access']:raise ValueError('Reuse control provenance/FINAL30 differs')
  if d.read(root/'source/source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Reused control source IDs differ')
  if prediction['status']!='PREDICTIONS_COMPLETE' or prediction['config']!=c or prediction['views']!=list(d.VIEWS) or prediction['count']!=168000 or prediction['truth_read'] or prediction['query_fit'] or not prediction['all_parameters_frozen'] or prediction['head_fit_updates'] or score['status']!='SCORED_COMPLETE' or not score['truth_last']:raise ValueError('Reuse immutable complete7view/default-head control only')
