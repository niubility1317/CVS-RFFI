"""Read four actual additive contributions and reclassify frozen joint160.

All changes are temporary observation hooks.  A contribution replacement is
applied before the existing cosine head's final unit normalization, never by
adding, subtracting or averaging logits.  TF denotes the entire normalized
base contribution; it does not split the nonlinear native T/F fusion.
"""
import torch

BRANCHES = ('TF', 'PA', 'REF', 'SIG')
OPERATIONS = ('zero', 'swap_nonoise', 'swap_clean')


def _layout(model):
    if any(module.training for module in model.modules()):
        raise ValueError('Only a frozen eval model may be observed')
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('All checkpoint parameters must remain frozen')
    try:
        ref = model.id_backbone.encoder
        head = ref.core.id_backbone.cls_head
        targets = dict(TF=head.base_norm, PA=head.pa_norm,
            REF=ref.response_projection, SIG=ref.sequence_pool)
    except AttributeError as error:
        raise ValueError('Original F reference/core/default-head layout required') from error
    if (head.weight.shape != (6, 160) or head.scale != 30.
            or head.gain.shape != (160,) or ref.response_gain.shape != (160,)
            or not callable(getattr(head, 'classify', None))
            or not callable(getattr(model, 'forward_identity_only', None))):
        raise ValueError('Fixed joint160/default cosine6 scale30 required')
    return ref, head, targets


def _state(model):
    """Small read-only identity/version snapshot; never clone the whole model."""
    values = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        values.append(('parameter', name, id(parameter), parameter._version,
            parameter.requires_grad, id(grad), None if grad is None else grad._version))
    values.extend(('buffer', name, id(value), value._version)
        for name, value in model.named_buffers())
    values.extend(('mode', id(module), module.training) for module in model.modules())
    return tuple(values)


def _vector(value, batch, device, dtype, label):
    if (not torch.is_tensor(value) or value.shape != (batch, 160)
            or value.device != device or value.dtype != dtype
            or value.requires_grad or not bool(torch.isfinite(value).all())):
        raise ValueError('One finite detached Bx160 contribution required: ' + label)
    return value


def _logits(value, batch, device, dtype):
    if (not torch.is_tensor(value) or value.shape != (batch, 6)
            or value.device != device or value.dtype != dtype
            or value.requires_grad or not bool(torch.isfinite(value).all())):
        raise ValueError('One finite detached default-head Bx6 output required')
    return value


def _joint(parts):
    # Preserve the actual native addition order, including floating-point order.
    joint = parts['TF'] + parts['PA']
    joint = joint + parts['REF']
    return joint + parts['SIG']


@torch.no_grad()
def observe(model, x):
    """One real model readout plus one counted parity-only head readout.

    The caller keeps physical IDs in the same order across observations.  This
    API checks shapes and same-model origin; it does not infer IDs from IQ.
    NoExplicitCFO, three SIG views and checkpoint identity are also checked by
    the source launcher's real-checkpoint smoke.
    """
    ref, head, targets = _layout(model)
    if (not torch.is_tensor(x) or x.ndim != 3 or x.shape[1:] != (2, 256)
            or len(x) == 0 or x.dtype != torch.float32
            or x.device != head.weight.device or not bool(torch.isfinite(x).all())):
        raise ValueError('Finite nonempty float32 source IQ Bx2x256 required')
    batch = len(x); before = _state(model); values = {}; handles = []
    native = {}; original = head.classify
    had_override = 'classify' in vars(head); previous = vars(head).get('classify')

    def record(name, output):
        if name in values:
            raise ValueError('Repeated real contribution module: ' + name)
        values[name] = _vector(output, batch, x.device, x.dtype, name).detach().clone()

    def classify(joint):
        if native:
            raise ValueError('Exactly one native default-head classification required')
        native['joint'] = _vector(joint, batch, x.device, x.dtype, 'native joint').detach().clone()
        output = original(joint)
        native['logits'] = _logits(output, batch, x.device, x.dtype).detach().clone()
        return output

    try:
        for name, module in targets.items():
            def hook(module, args, output, name=name):
                record(name, output)
            handles.append(module.register_forward_hook(hook))
        head.classify = classify
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = model.forward_identity_only(x)
    finally:
        for handle in handles:
            handle.remove()
        if had_override:
            head.classify = previous
        elif 'classify' in vars(head):
            delattr(head, 'classify')
        if _state(model) != before:
            raise ValueError('Observation changed model state, gradients or training modes')
    if set(values) != set(BRANCHES) or set(native) != {'joint', 'logits'}:
        raise ValueError('All four actual additive paths and one head call required')
    if not isinstance(output, dict) or 'tx_logits' not in output:
        raise ValueError('Original identity-only default-head output required')
    logits = _logits(output['tx_logits'], batch, x.device, x.dtype)
    if not torch.equal(logits, native['logits']):
        raise ValueError('Model logits differ from captured native default-head logits')
    parts = dict(TF=values['TF'], PA=head.gain.tanh() * values['PA'],
        REF=ref.response_gain.tanh() * values['REF'], SIG=values['SIG'])
    reconstructed = _joint(parts)
    if not torch.allclose(reconstructed, native['joint'], atol=1e-6, rtol=1e-6):
        raise ValueError('Four contributions do not reconstruct the actual native joint')
    if 'id_feat_joint' in output and not torch.equal(output['id_feat_joint'], native['joint']):
        raise ValueError('Reported deployed joint differs from the head input')
    with torch.autocast(device_type=x.device.type, enabled=False):
        parity = _logits(original(reconstructed), batch, x.device, x.dtype)
    if not torch.allclose(parity, logits, atol=1e-5, rtol=1e-6):
        raise ValueError('Reconstructed joint does not reproduce original cosine logits')
    if _state(model) != before:
        raise ValueError('Parity head readout changed model state or gradients')
    return dict(logits=logits.detach().clone(), joint=native['joint'],
        contributions={key: value.detach().clone() for key, value in parts.items()},
        native_model_examples=batch, native_head_examples=batch,
        parity_head_examples=batch, head_examples=batch, extra_head_examples=batch,
        joint_max_abs_error=float((reconstructed-native['joint']).abs().max()),
        logit_max_abs_error=float((parity-logits).abs().max()),
        native_joint_parity=True, native_logit_parity=True,
        _model_token=id(model), _state_signature=before)


def _pack(pack, model, batch, device, dtype, state):
    if (not isinstance(pack, dict) or pack.get('_model_token') != id(model)
            or pack.get('_state_signature') != state
            or not isinstance(pack.get('contributions'), dict)
            or set(pack['contributions']) != set(BRANCHES)):
        raise ValueError('Aligned same frozen model observations required')
    for name in BRANCHES:
        _vector(pack['contributions'][name], batch, device, dtype, name)
    _vector(pack.get('joint'), batch, device, dtype, 'joint')
    _logits(pack.get('logits'), batch, device, dtype)
    if not torch.allclose(_joint(pack['contributions']), pack['joint'], atol=1e-6, rtol=1e-6):
        raise ValueError('Stored contribution joint parity differs')


@torch.no_grad()
def interventions(model, full, no_noise_fixed, clean):
    """Twelve whole-contribution replacements; zero model/encoder forwards."""
    _, head, _ = _layout(model); before = _state(model)
    if not isinstance(full, dict) or not torch.is_tensor(full.get('joint')):
        raise ValueError('A complete full observation is required')
    batch = len(full['joint']); device = head.weight.device; dtype = head.weight.dtype
    if batch == 0:
        raise ValueError('Nonempty aligned source observations required')
    for pack in (full, no_noise_fixed, clean):
        _pack(pack, model, batch, device, dtype, before)
    result = {}; original = head.classify
    with torch.autocast(device_type=device.type, enabled=False):
        for name in BRANCHES:
            for operation in OPERATIONS:
                if operation == 'zero':
                    replacement = torch.zeros_like(full['contributions'][name])
                else:
                    source = no_noise_fixed if operation == 'swap_nonoise' else clean
                    replacement = source['contributions'][name]
                parts = dict(full['contributions']); parts[name] = replacement
                result[operation+'_'+name] = _logits(
                    original(_joint(parts)), batch, device, dtype).detach().clone()
    if _state(model) != before:
        raise ValueError('Head-only interventions changed state or gradients')
    return dict(logits=result, head_examples=12*batch, extra_head_examples=12*batch,
        native_model_examples=0, native_head_examples=0, parity_head_examples=0,
        interventions=12, final_normalization_recomputed=True,
        whole_postnorm_gain_contributions=True)
