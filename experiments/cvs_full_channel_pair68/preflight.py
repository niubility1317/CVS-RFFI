"""Launcher-only AS_CE20 student/ownEMA smoke and one real sixL synthetic graph."""
import torch
from . import design as d, model
from .source import parent_payload
from .checks import run as focused_checks
from experiments.cvs_crossview_supcon41 import model as base


def smoke():
    torch.set_num_threads(2)
    d.write(d.BASE / 'focused_checks.json', focused_checks())
    c = d.config(d.rows()[0])
    cp = parent_payload(c)
    student = model.build(c, 'cpu').eval()
    student.load_state_dict(cp['model'], strict=True)
    before = {k: v.clone() for k, v in student.state_dict().items()}
    if sum(p.numel() for p in student.parameters()) != 900722 or len(before) != 206:
        raise ValueError('Original fixed F/default-head architecture differs')
    with torch.no_grad():
        z = student.forward_identity_only(torch.zeros(2, 2, 256))['tx_logits']
    if z.shape != (2, 6) or not bool(torch.isfinite(z).all()):
        raise ValueError('Actual AS_CE20 student/default-head no-query smoke failed')
    ema = model.build(c, 'cpu').eval()
    ema.load_state_dict(cp['ema_model'], strict=True)
    with torch.no_grad():
        z = ema.forward_identity_only(torch.zeros(2, 2, 256))['tx_logits']
    if z.shape != (2, 6) or not bool(torch.isfinite(z).all()):
        raise ValueError('Actual AS_CE20 ownEMA/default-head no-query smoke failed')
    del ema
    student.train()
    student._feature_l_steps, student._full_calls = 1333, 1332
    generator = torch.Generator().manual_seed(68007)
    x = torch.randn((128, 2, 256), generator=generator)
    y, domains = torch.arange(128) % 6, torch.arange(128) % 15
    rx = torch.tensor(base.supcon.RX_BY_DOMAIN)[domains]
    metadata = dict(eq_i=torch.ones(128, dtype=torch.long), rx_i=rx,
        day_i=domains % 3 + 1, sig_i=torch.arange(128) + 68000, base_index=torch.arange(128))
    first, maps = base.capture_forward(student, x, domains, True)
    student._supcon_pending = dict(first=maps, second={})
    student._legacy_actual_L_input = x.clone()
    ce = torch.nn.functional.cross_entropy(first, y, label_smoothing=.01)
    delta, _ = student._source_labeled_feature_hook(model=student, labeled_output=dict(tx_logits=first),
        labels=y, extra=(domains, metadata), domains=domains, optimizer_step=1333,
        identity_loss=ce, labeled_count=128, labeled_inputs=x)
    loss = ce + delta
    loss.backward()
    head = student.id_backbone.encoder.core.id_backbone.cls_head.weight
    sinc = student.id_backbone.encoder.core.id_backbone.sinc
    if (not bool(torch.isfinite(loss)) or head.grad is None or float(head.grad.abs().sum()) <= 0
            or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None) <= 0
            or any(not bool(torch.isfinite(p.grad).all()) for p in student.parameters() if p.grad is not None)
            or any(not torch.equal(v, student.state_dict()[k]) for k, v in before.items())):
        raise ValueError('Joint default-head/Sinc finite synthetic graph or no-update state failed')
    record = student._pair_records[-1]
    if (student.id_backbone.encoder._full_train_packets != 768 or student._shape_NN != 256
            or student._full_channel.packets != 64 or student._pair_generated != 64
            or len(student._pair_records) != 1 or record['forward_count'] != 5
            or record['bypassed_elements'] or record['As_IQ_pair_MSE'] <= 0
            or not record['same_hardware'] or record['cross_channel_KL']
            or abs(student._full_records[-1]['total_L'] - float(loss.detach())) > 3e-6
            or student._supcon_pending is not None or student._supcon_handles
            or student._sat_engine.active or getattr(student, '_source_unlabeled_feature_hook', None) is not None):
        raise ValueError('SixL/two full As/sharedhardware/CE-only/oldU synthetic scope failed')
    result = dict(status='PASS', NN=772, real_checkpoint_NN=4, synthetic_remote_NN=768,
        checkpoint=c['parent_source'] + '/final_ssdg.pth', student_and_ownEMA=True,
        training_initializer=True, source_IQ=0, target_query=False, optimizer_updates=0,
        head_fit_updates=0, head_joint_gradient=True, Sinc_joint_gradient=True,
        state_unchanged=True, full_As_generations=128, record=record)
    d.write(d.BASE / 'real_checkpoint_smoke.json', result)
    print('ACTUAL_AS_CE20_STUDENT_OWNEMA+SIXL_INDEPENDENT_FULL_AS_SYNTHETIC PASS772NN;continue training', flush=True)
    return result
