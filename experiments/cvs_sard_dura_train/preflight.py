import gc
import numpy as np
import torch
from . import design as d, model
from .boundary import training_tensor

def smoke(c):
    from experiments.cvs_t1_e50.source import original_parent
    from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
    from experiments.cvs_crossview_supcon41 import model as base
    m = model.build(c, 'cuda:0')
    parent = original_parent()  # Verified same-source E30, smoke only, never initializer.
    m.load_state_dict(parent['model'], strict=True)
    m.eval().requires_grad_(False)
    cache = d.PROJECT/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache'
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    ix = np.flatnonzero(roles=='L_s')[:2]
    x = array_tensor(np.load(cache/'epochs/01/received.npy', mmap_mode='r', allow_pickle=False)[ix], 'cuda:0')
    before = {k:v.clone() for k,v in m.state_dict().items()}
    engine = None
    if c['restored_augmentation']['enabled']:
        from .channel import load_dura
        engine = load_dura(x.device)
        g_before = {k:v.clone() for k,v in engine.pipeline.model.state_dict().items()}
        x = training_tensor(engine, x)
    with torch.no_grad():
        logits = m.forward_identity_only(x)['tx_logits']
    if logits.shape!=(2,6) or not torch.isfinite(logits).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):
        raise ValueError('Real source checkpoint no-query smoke failed')
    del m, before
    torch.manual_seed(6007)
    m = model.build(c, 'cuda:0').train()
    if engine is not None:
        m._full_channel.engine = engine
    m._feature_l_steps = 1333
    m._full_calls = 1332
    x = torch.randn(128,2,256,device='cuda:0')
    y = torch.arange(128,device=x.device)%6
    domains = torch.arange(128,device=x.device)%15
    rx = torch.tensor(base.supcon.RX_BY_DOMAIN,device=x.device)[domains]
    meta = dict(eq_i=torch.ones(128,dtype=torch.long,device=x.device), rx_i=rx,
        day_i=domains%3+1, sig_i=torch.arange(128,device=x.device)+2000, base_index=torch.arange(128,device=x.device))
    first, maps = base.capture_forward(m,x,domains,True)
    m._supcon_pending = dict(first=maps,second={})
    m._legacy_actual_L_input = x.clone()
    ce = torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
    extra,_ = m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,
        extra=(domains,meta),domains=domains,optimizer_step=1333,identity_loss=ce,labeled_count=128,labeled_inputs=x)
    total = ce+extra
    total.backward()
    head = m.id_backbone.encoder.core.id_backbone.cls_head.weight
    if not torch.isfinite(total) or head.grad is None or not bool(head.grad.abs().sum()>0) or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):
        raise ValueError('Joint default-head training through detached G output failed')
    if (m._full_channel.packets!=64 or m.id_backbone.encoder._full_train_packets!=768
            or m._temperature_records[-1]['As_KL']!=0. or m._supcon_handles or m._supcon_pending is not None):
        raise ValueError('SARD objective/slot/SupCon budget changed')
    g_state_unchanged = True
    if engine is not None:
        g = engine.pipeline.model
        g_state_unchanged = all(torch.equal(v,g.state_dict()[k]) for k,v in g_before.items())
        if not g_state_unchanged or any(p.grad is not None or p.requires_grad for p in g.parameters()) or g.training:
            raise ValueError('DURA must remain frozen during F gradient probe')
    d.write(d.row_root(c['row_id'])/'smoke.json', dict(status='PASS',real_source_Ls=2,
        real_F_checkpoint=str(d.old.PARENT_SOURCE/'final_ssdg.pth'), checkpoint_training_initializer=False,
        F_NN=770, synthetic_L_NN=768, actual_As_IQ=64, G_frozen=g_state_unchanged,
        G_source_IQ=2 if engine else 0, G_synthetic_IQ=64 if engine else 0,
        G_setup=[] if engine is None else engine.setup(), target_reads=0, updates=0,
        head_joint_gradient=True, As_KL=0., calendar=dict(L=13,U=37)))
    del m, engine, total, extra, ce, logits, first, maps
    gc.collect()
    torch.cuda.empty_cache()
    print('REAL_SOURCE_CP_NO_QUERY_AND_SARD_DURA_GRADIENT_SMOKE PASS;continue training',flush=True)
