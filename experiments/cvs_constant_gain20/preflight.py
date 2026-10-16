"""One real matching CP smoke plus its no-query engineering input audit."""
import torch
from . import design as d,model as api
from .input_audit import run
def smoke():
 torch.set_num_threads(2);src=d.SMOKE_SOURCE;done=d.read(src/'completion.json');contract=d.read(src/'source_contract.json');fold=d.read(src/'source_fold.json');carrier=d.read(d.SOURCE);init=d.read(src/'initialization.json');frozen=d.read(src/'source_frozen.json')
 if not init['scratch_only'] or init['ancestors'] or init['checkpoint_sources'] or init['target_contact'] or init['seed']!=2026092701 or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['target_access'] or frozen['selection']!='final_only' or frozen['target_access']:raise ValueError('Known matching CP provenance differs')
 if contract['held_RX']!=1 or set(contract['source_receivers'])!=set(d.FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600} or fold['held_RX']!=1 or any(not set(v)<=set(carrier['role_ids'][k]) for k,v in contract['role_ids'].items()):raise ValueError('Known matching source CP roles differ')
 ck=torch.load(src/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['args']['run_id']!='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01' or ck['args']['seed']!=2026092701 or ck['args']['source_fold_held_rx']!=1 or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=4440:raise ValueError('Matching real CP payload differs')
 m=api.build(d.config(d.rows()[0]),'cpu').eval();m.load_state_dict(ck['model'],strict=True);m.requires_grad_(False);ref=m.id_backbone.encoder
 if len(ref.sequence_pool.views)!=3 or ref.sequence_pool.modes!=('SIG_TIME',)*3 or ref.response.contract()['explicit_CFO_identity_input'] is not False or ref.core.id_backbone.cls_head.weight.shape!=(6,160) or ref.core.id_backbone.cls_head.scale!=30.:raise ValueError('Fixed F/default head boundary differs')
 with torch.no_grad():z=m.forward_identity_only(torch.randn(2,2,256,generator=torch.Generator().manual_seed(20261016204)))['tx_logits']
 if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Real CP no-query smoke failed')
 print('SMOKE PASS real matching checkpoint,2synthetic packets,0query;continue input audit',flush=True)
 audit=run(m);result=dict(status='PASS',role='Known own matching CP for synthetic no-query input audit only;all later workers scratch',strict_load=True,same_effective_LUV_contract=True,synthetic_NN_forward_packets=162,source_IQ_reads=0,target_access=False,training_updates=0,head_fit_updates=0,input_effect_decision=audit['qualification']['decision'])
 d.write(d.BASE/'real_checkpoint_smoke.json',result);print('INPUT AUDIT '+str(audit['qualification'])+';0source IQ/updates',flush=True);return result
