from pathlib import Path
import torch
from . import design as d

def check(c):
    d.validate(c);root=Path(c['checkpoint_origin']);done=d.read(root/'completion.json');init=d.read(root/'initialization.json')
    contract=d.read(root/'source_contract.json');freeze=d.read(root/'source_frozen.json');result=d.read(root/'result.json')
    expected=d.parent_model_config(c['arm']);f=done['feature_extractor']
    if (done['config']!=expected or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch'
        or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
        or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact']
        or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
        or result['target_access'] or result['U_truth_read'] or not result['default_head_joint_training']
        or contract['held_RX']!=1 or contract['source_receivers']!=[3,4,6,8]
        or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}
        or f['arm']!='SIG_TIME_X3' or f['sinc_coordinates']!='CENTER_25K' or f['Sinc_optimizer']!='CENTER_UNITS'):
        raise ValueError('Fixed F source-fold final20 parent provenance differs')
    return expected,contract

def load(c,device):
    expected,contract=check(c)
    if c['arm']=='A0':from experiments.cvs_raw_anchor20 import model as api
    else:from experiments.cvs_white_observation20 import model as api
    model=api.build(expected,'cpu')
    payload=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
    if payload['epoch']!=20 or payload['checkpoint_selection']!='final_only' or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt']:
        raise ValueError('Original scratch student final20 required')
    model.load_state_dict(payload['model'],strict=True);del payload
    model.eval().to(device).requires_grad_(True)
    with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256,device=device))['tx_logits']
    if out.shape!=(2,6) or not bool(torch.isfinite(out).all()):raise ValueError('Real frozen default-head no-query checkpoint smoke failed')
    if tuple(model.id_backbone.encoder.core.id_backbone.cls_head.weight.shape)!=(6,160):raise ValueError('Original default classifier changed')
    snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    print('SMOKE PASS '+c['arm']+' actual final20/defaulthead; query_read=false',flush=True)
    return model,snapshot,contract
