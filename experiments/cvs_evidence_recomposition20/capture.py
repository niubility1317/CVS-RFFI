"""One source-eval forward with real normalized branches and raw residuals."""
import torch
from torch import nn

from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng

HEAD_PATH = 'id_backbone.encoder.core.id_backbone.cls_head'
SEQUENCE_PATH = 'id_backbone.encoder.sequence_pool'
REFERENCE_PATH = 'id_backbone.encoder.response_projection'


def _tensor(value, shape, device, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.shape != shape or value.device != device
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError('Actual finite branch shape/device differs: ' + name)


def forward(model, x):
    """Return attached native branches from exactly one identity-only call.

    The caller must already have selected eval mode. This function never changes
    training flags or binds a callback to a model/teacher. The adapter directly
    returns base_norm/pa_norm outputs; call-local hooks capture the existing
    sequence output_norm BEFORE its gain and the existing reference projection.
    The latter is a real fourth deployment term, including when its gain is zero.
    No component is inferred by subtracting branches from the deployed joint.
    """
    if not isinstance(model, nn.Module) or any(m.training for m in model.modules()):
        raise ValueError('An already eval-mode source model is required; flags are caller-owned')
    if (not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.ndim != 3
            or x.shape[1:] != (2, 256) or len(x) < 1 or not x.is_floating_point()
            or not bool(torch.isfinite(x).all())):
        raise ValueError('Finite real source IQ[N,2,256] required')
    head = model.get_submodule(HEAD_PATH)
    pool = model.get_submodule(SEQUENCE_PATH)
    reference = model.get_submodule(REFERENCE_PATH)
    if getattr(pool, 'arm', None) != 'SIG_TIME':
        raise ValueError('Only the deployed SIG_TIME output_norm branch is supported')
    parameters = [(p, p.detach().clone(), p._version, p.grad,
                   None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    buffers = [(b, b.detach().clone(), b._version) for b in model.buffers()]
    flags = [(m, m.training) for m in model.modules()]
    observed, handles = {}, []

    def hook(name):
        def remember(module, inputs, output):
            if name in observed:
                raise ValueError('Expected one same-call branch output: ' + name)
            _tensor(output, (len(x), 160), x.device, name)
            observed[name] = output
        return remember

    try:
        handles.append(pool.output_norm.register_forward_hook(hook('sequence_raw')))
        handles.append(reference.register_forward_hook(hook('reference_raw')))
        with _preserve_rng(), fresh_parameter_caches(model):
            output = model.forward_identity_only(x)
        if set(observed) != {'sequence_raw', 'reference_raw'}:
            raise ValueError('Existing raw sequence/reference branches were not both called')
        result = dict(identity=output.get('id_feat_cls'), physical=output.get('id_feat_pa'),
                      sequence_raw=observed['sequence_raw'], reference_raw=observed['reference_raw'],
                      joint=output.get('id_feat_joint'), logits=output.get('tx_logits'))
        for name, value in result.items():
            _tensor(value, (len(x), 6 if name == 'logits' else 160), x.device, name)
        # Keep exactly the deployed addition and multiplication order.
        encoder = model.id_backbone.encoder
        reconstructed = ((result['identity'] + head.gain.tanh() * result['physical'])
                         + encoder.response_gain.tanh() * result['reference_raw'])
        reconstructed = reconstructed + (.2 * pool.gain_raw.tanh()) * result['sequence_raw']
        if not torch.equal(reconstructed.detach(), result['joint'].detach()):
            raise ValueError('Direct real-branch reconstruction differs from deployed joint')
        return result
    finally:
        for handle in handles:
            handle.remove()
        changed = any(p._version != version or not torch.equal(p.detach(), value)
                      or p.grad is not grad
                      or (grad is not None and not torch.equal(grad, grad_value))
                      for p, value, version, grad, grad_value in parameters)
        changed |= any(b._version != version or not torch.equal(b, value)
                       for b, value, version in buffers)
        changed |= any(m.training != flag for m, flag in flags)
        if changed:
            raise RuntimeError('Source branch capture altered live parameters/buffers/gradients/flags')
