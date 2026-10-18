"""One no-query smoke: student and one fixed As20 teacher,4NN/one own20 CP."""
from copy import deepcopy
import torch
from . import design as d, model
from .source import parent_payload
from .checks import run as focused_checks


def smoke():
    torch.set_num_threads(2)
    d.write(d.BASE / 'focused_checks.json', focused_checks())
    c = d.config(d.rows()[0])
    as_cp = parent_payload(c)
    m = model.build(c, 'cpu').eval()
    m.load_state_dict(as_cp['model'], strict=True)
    ema = deepcopy(m).eval().requires_grad_(False)
    ema.load_state_dict(as_cp['ema_model'], strict=True)
    if len(m.state_dict()) != 206 or len(ema.state_dict()) != 206 or sum(p.numel() for p in m.parameters()) != 900722:
        raise ValueError('Own student/EMA deployment architecture differs')
    before = {k: v.clone() for k, v in m.state_dict().items()}
    bank = model.prepare_frozen_teachers(m, c, payloads={'B01': as_cp})
    x = torch.zeros(2, 2, 256)
    domains = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        outputs = {'student': m.forward_identity_only(x)['tx_logits'], **bank.forward(x, domains)}
    if any(z.shape != (2, 6) or not torch.isfinite(z).all() for z in outputs.values()):
        raise ValueError('Actual own20 student/default-head/fixed-teacher smoke failed')
    bank.assert_frozen()
    b = bank.summary()
    if (any(not torch.equal(v, m.state_dict()[k]) for k, v in before.items())
            or len(m.state_dict()) != 206 or b['forward_NN'] != {'B01': 2}
            or b['total_forward_NN'] != 2 or b['registered_in_student']):
        raise ValueError('Frozen auxiliary teacher changed student identity or smoke cost')
    result = dict(status='PASS', NN=4, student_NN=2, frozen_teacher_NN=2,
                  real_checkpoints=1, unique_checkpoint_sources=c['checkpoint_sources'],
                  student_checkpoint=c['parent_source'] + '/final_ssdg.pth',
                  own_EMA_strictly_loaded=True, frozen_teacher_payload='own20 ema_model',
                  frozen_teachers=b, training_initializer=True, same_source_contract=True,
                  source_IQ=0, target_query=False, optimizer_updates=0, synthetic_remote_NN=0,
                  teacher_parameters_registered_in_student=False, deployment_state_keys=206,
                  deployment_parameters=900722)
    d.write(d.BASE / 'real_checkpoint_smoke.json', result)
    print('ACTUAL_AS20_STUDENT_AND_OWN20_EMA_TEACHER_NO_QUERY PASS4NN;continue training', flush=True)
    return result
