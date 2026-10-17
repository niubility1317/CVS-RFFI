"""Strict fixed-parent cumulative30 provenance; source-only until prediction."""
from pathlib import Path
from . import design as d
def source(c):
 d.validate(c);p=Path(c['output_root']);v=d.read(p/'completion.json');i=d.read(p/'initialization.json');f=d.read(p/'source_frozen.json');a=d.read(p/'actual_native_args.json');a=a.get('args',a);parent=str(Path(c['parent_source'])/'final_ssdg.pth')
 if v['status']!='SOURCE_TRAINED' or v['config']!=c or (v['epoch'],v['logged_steps'],v['optimizer_steps'],v['cumulative_epochs'],v['cumulative_optimizer_steps'])!=(10,2220,2220,30,6660) or v['target_access'] or v['target_evaluated'] or v['checkpoint_sources']!=[parent] or v['ancestors']!=[c['parent_source']] or v['head_fit_updates'] or v['final_source_metrics']['source_val_count']!=27000:raise ValueError('Incomplete matched E20+10 source provenance')
 if i['scratch_only'] or i['checkpoint_sources']!=[parent] or i['ancestors']!=[c['parent_source']] or i['target_contact'] or i['source_roles']!='EXACT_MATCH' or not i['optimizer_reset'] or i['teacher_origin']!='strict own parent EMA':raise ValueError('Unknown/mismatched ancestor')
 if f!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=10,selection='final_only',target_access=False):raise ValueError('Frozen final10/cumulative30 differs')
 if (a['epochs'],a['label_epochs'],a['pseudo_epochs'],a['seed'])!=(10,0,10,d.SEED) or a['baseline_ckpt']!=parent or a['teacher_ckpt'] or a.get('source_fold_held_rx') is not None or a.get('source_fold_training_receivers') or a.get('a1_scratch_only') is not False or a['from_scratch'] or a['freeze_backbone'] or not a['use_aug'] or not a['use_ema_teacher'] or a['use_concat_sat_channel_aug']:raise ValueError('Actual full-source/head/U/view arguments differ')
 d.parent_proof(c['arm']);return p
def load(c,device):
 import torch
 from . import model as api
 p=source(c);cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if (cp['epoch'],cp['run_id'],cp['candidate_id'],cp['a1_ema_successful_updates'],cp['checkpoint_selection'])!=(10,d.RUN,c['row_id'],2220,'final_only'):raise ValueError('Checkpoint identity differs')
 m=api.build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
