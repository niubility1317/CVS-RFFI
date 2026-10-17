"""One matching full-source trained checkpoint for compatibility only; never initialization."""
import torch
from . import design as d,model as api

def smoke():
 torch.set_num_threads(2);p=d.SMOKE_SOURCE;done=d.read(p/'completion.json');initial=d.read(p/'initialization.json');freeze=d.read(p/'source_frozen.json');args=d.read(p/'actual_native_args.json');args=args.get('args',args)
 if done['epoch']!=50 or done['optimizer_steps']!=11100 or done['target_access'] or done['checkpoint_sources'] or done['initialization']!='scratch' or initial['ancestors'] or initial['checkpoint_sources'] or initial['target_contact'] or not initial['scratch_only'] or freeze['selection']!='final_only' or args.get('source_fold_held_rx') is not None or args['baseline_ckpt'] or args['teacher_ckpt']:raise ValueError('Matching full-source reference provenance required')
 if d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
 c=d.config(d.rows()[0]);cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);m=api.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval()
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))
 if z['tx_logits'].shape!=(2,6) or z['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(z['tx_logits']).all()):raise ValueError('Real full-source checkpoint no-query smoke failed')
 d.write(d.BASE/'preflight.json',dict(status='PASS',full_source_contract_matches=True,reference_fixed_final50=True,checkpoint_used_for_training_initialization=False,real_CP_loads=1,NN_forward_packets=2,real_IQ=0,query_read=False,head_fit_updates=0,new_training_updates=0))
 print('PREFLIGHT PASS real full-source checkpoint/no-query;continuing',flush=True)
