"""Current full-source checkpoint identity and provenance, no query access."""
from pathlib import Path
from . import design as d

def source(c):
 d.validate(c);p=Path(c['output_root']);v=d.read(p/'completion.json');initial=d.read(p/'initialization.json');freeze=d.read(p/'source_frozen.json');args=d.read(p/'actual_native_args.json');args=args.get('args',args)
 if v['status']!='SOURCE_TRAINED' or v['config']!=c or v['epoch']!=20 or v['logged_steps']!=4440 or v['optimizer_steps']!=4440 or v['target_access'] or v['target_evaluated'] or v['checkpoint_sources'] or v['ancestors'] or v['initialization']!='scratch' or v['head_fit_updates'] or v['final_source_metrics']['source_val_count']!=27000:raise ValueError('Incomplete full-source scratch20 provenance')
 if not initial['scratch_only'] or initial['checkpoint_sources'] or initial['ancestors'] or initial['target_contact'] or initial['source_roles']!='EXACT_MATCH' or initial['seed']!=d.SEED:raise ValueError('Unknown/mismatched ancestor')
 if freeze!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=20,selection='final_only',target_access=False):raise ValueError('Frozen final20 differs')
 if (args['epochs'],args['label_epochs'],args['pseudo_epochs'],args['seed'])!=(20,13,7,d.SEED) or args['baseline_ckpt'] or args['teacher_ckpt'] or args.get('source_fold_held_rx') is not None or args.get('source_fold_training_receivers') or not args['use_aug'] or not args['use_ema_teacher'] or args['use_concat_sat_channel_aug']:raise ValueError('Actual full-source/U/view arguments differ')
 return p

def load(c,device):
 import torch
 from . import model as api
 p=source(c);cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=20 or cp['run_id']!=d.RUN or cp['candidate_id']!=c['row_id'] or cp['a1_ema_successful_updates']!=4440 or cp['checkpoint_selection']!='final_only':raise ValueError('Checkpoint identity differs')
 m=api.build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
