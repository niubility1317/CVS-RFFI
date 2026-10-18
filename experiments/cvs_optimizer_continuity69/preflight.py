"""Both legal own20 student/EMA states and real AdamW load;8NN,zero updates."""
from copy import deepcopy
import torch
from . import design as d, model
from .source import parent_payload
from .optimizer_state import restore_optimizer
from .checks import run as focused_checks


def smoke():
    torch.set_num_threads(2)
    d.write(d.BASE / 'focused_checks.json', focused_checks())
    from experiments.cvs_phase1_overlay.model import native_modules
    from experiments.cvs_rdrop_sinc_units.optimizer import build as build_optimizer
    native = native_modules()
    records = []
    for row in d.rows():
        c = d.config(row)
        cp = parent_payload(c)
        m = model.build(c, 'cpu').eval()
        m.load_state_dict(cp['model'], strict=True)
        ema = deepcopy(m).eval().requires_grad_(False)
        ema.load_state_dict(cp['ema_model'], strict=True)
        if len(m.state_dict()) != 206 or sum(p.numel() for p in m.parameters()) != 900722:
            raise ValueError('Fixed original F/default-head shape differs')
        before = {k: v.clone() for k, v in m.state_dict().items()}
        a = d.make_args(c, 'cpu')
        opt = build_optimizer(m, native._optimizer_parameters(m, None), 'CENTER_UNITS',
            lr=float(a.lr), weight_decay=float(a.weight_decay))
        state = restore_optimizer(opt, cp['optimizer'], expected_steps=4440,
            named_parameters=m.named_parameters(), parent_model=cp['model'])
        with torch.no_grad():
            logits = [v.forward_identity_only(torch.zeros(2, 2, 256))['tx_logits'] for v in (m, ema)]
        if (any(z.shape != (2, 6) or not torch.isfinite(z).all() for z in logits)
                or any(not torch.equal(v, m.state_dict()[k]) for k, v in before.items())
                or opt._sinc_unit_steps or opt._sinc_physical_clip_calls
                or opt._sinc_unit_lr_records or any(p.grad is not None for p in m.parameters())):
            raise ValueError('No-query parent/model/optimizer load smoke altered state or performed update')
        records.append(dict(arm=c['arm'], checkpoint=c['checkpoint_sources'][0],
            student_NN=2, EMA_NN=2, strict_load=True, optimizer_load=state, optimizer_updates=0))
        del m, ema, opt, cp, before
    result = dict(status='PASS', NN=8, real_checkpoints=2, rows=records, source_IQ=0,
        target_query=False, optimizer_updates=0, head_fit_updates=0, extra_teacher_training=0,
        same_source_contract=True, training_initializers=True, optimizer_state_loaded_no_step=True)
    d.write(d.BASE / 'real_checkpoint_smoke.json', result)
    print('OWN20_STUDENT_EMA_OPTIMIZER_NO_QUERY_PASS8NN;continue training', flush=True)
    return result
