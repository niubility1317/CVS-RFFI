from types import SimpleNamespace
import torch
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint,infer_num_domains_from_state
from cvsrffi.game_tracking.runtime import build_model

def load_model(path,device,expected_epoch=200):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('epoch')!=expected_epoch:raise ValueError('incomplete final epoch')
    args=payload['args']
    if not args.get('from_scratch') or args.get('baseline_ckpt') or args.get('teacher_ckpt') or args.get('game_resume'):raise ValueError('checkpoint initialization is not scratch-only')
    if payload.get('target_contact',False):raise ValueError('checkpoint used target data')
    if payload.get('schema')=='adv3b02_xuc_v1':
        model=build_model(SimpleNamespace(**args),infer_num_domains_from_state(payload['model']),device)
        model.load_state_dict(payload['model'],strict=True)
    else:model,_=build_exact_ssdg_model_from_checkpoint(payload,input_len=256,device=device)
    return model.eval(),payload
