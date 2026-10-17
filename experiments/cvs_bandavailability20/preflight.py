"""Matching effective-training-contract checkpoint, compatibility only; never init."""
import torch
from . import design as d,model as api

def smoke():
 torch.set_num_threads(2);s=d.SMOKE_SOURCE;done=d.read(s/'completion.json');init=d.read(s/'initialization.json');contract=d.read(s/'source_contract.json');fold=d.read(s/'source_fold.json');freeze=d.read(s/'source_frozen.json');carrier=d.read(d.SOURCE)
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Verified source-only final20 CP provenance required')
 if contract['held_RX']!=d.HELD or contract['source_receivers']!=list(d.FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600} or fold['held_RX']!=d.HELD or any(not set(v)<=set(carrier['role_ids'][k]) for k,v in contract['role_ids'].items()):raise ValueError('Exact current sourceRX1 effective data roles required')
 ck=torch.load(s/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=20 or ck['args']['seed']!=d.SEED or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or ck['checkpoint_selection']!='final_only':raise ValueError('CP inheritance differs')
 m=api.build(d.config(d.rows()[0]),'cpu').eval();m.load_state_dict(ck['model'],strict=True)
 with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(20261016204)))['tx_logits']
 if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Real CP no-query smoke failed')
 result=dict(status='PASS',checkpoint_source=s.as_posix(),role='Matching model-seed B00 6L source-only CP;compatibility smoke only,never initialization or teacher',same_effective_LUV_contract=True,strict_load=True,synthetic_NN_forward_packets=2,source_IQ_reads=0,target_access=False,training_updates=0,head_fit_updates=0);d.write(d.BASE/'real_checkpoint_smoke.json',result);return result
