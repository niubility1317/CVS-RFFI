"""Independent disturbance networks supervise the unchanged CVS identity path.

The real E/G split is immediately before native physical fusion LayerNorms and
the reference-response projection. No new identity features or classifiers are
introduced. Auxiliary networks are training-only and do not participate in the
normal identity forward. Source RX group construction belongs to the runtime.
"""
import torch
from torch import nn
from experiments.cvs_reference_identity.model import ExcitationResponse

INTERMEDIATE_DIM = 349
IDENTITY_DIM = 160
KINDS = ('linear', 'temporal', 'receiver', 'interaction')


def _encoder(identity):
    encoder = getattr(identity, 'encoder', identity)
    if not all(hasattr(encoder, key) for key in ('core', 'response', 'response_projection', 'response_gain')):
        raise TypeError('Expected ReferenceCVS or Phase1IdentityAdapter')
    return encoder


def intermediate(identity, x):
    """Capture actual pre-fusion tensors, retaining gradients through E.

Calling the existing identity path preserves its implementation and hooks. The
captured physical tensor already includes the native pa_delta addition, so G
must not add it again. Hooks are local and removed on all exit paths.
"""
    encoder = _encoder(identity)
    head = encoder.core.id_backbone.cls_head
    captured = {}
    hooks = [
        head.base_norm.register_forward_pre_hook(lambda m, a: captured.update(base=a[0])),
        head.pa_norm.register_forward_pre_hook(lambda m, a: captured.update(physical=a[0])),
        encoder.response.register_forward_hook(lambda m, a, o: captured.update(response=o)),
    ]
    try:
        identity(x)
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != {'base', 'physical', 'response'}:
        raise RuntimeError('Reference identity intermediate capture incomplete')
    return torch.cat((captured['base'], captured['physical'], captured['response']), dim=1)


def identity_from_intermediate(identity, h):
    """Replay existing learned G exactly; no auxiliary parameters are used."""
    if h.ndim != 2 or h.shape[1] != INTERMEDIATE_DIM:
        raise ValueError('Expected intermediate [B,349]')
    encoder = _encoder(identity)
    base, physical, response = h.split((160, 160, 29), dim=1)
    joint = encoder.core.id_backbone.cls_head.components(base, physical)[2]
    return joint + encoder.response_gain.tanh() * encoder.response_projection(response)


def classify_intermediate(identity, h):
    encoder = _encoder(identity)
    return encoder.classify_features(identity_from_intermediate(identity, h))


class FixedViews(nn.Module):
    """Separate observable views, without learned shared nuisance encoder."""
    def __init__(self):
        super().__init__()
        self.response = ExcitationResponse('reference_response')

    def linear(self, x):
        return self.response(x)

    def temporal(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (2, 256):
            raise ValueError('Expected IQ [B,2,256]')
        window = x[:, :, 80:160]
        scale = torch.sqrt(window.square().mean((1, 2), keepdim=True) + 1e-6)
        # Four ordered cycles, with both quadratures retained; no phase removal.
        return (window / scale).reshape(len(x), 8, 20)

    def receiver(self, x):
        z = torch.complex(x[:, 0], x[:, 1])
        power = z.abs().square()
        scale = power.mean(1, keepdim=True) + 1e-6
        normalized = power / scale
        moments = torch.stack((scale[:, 0].log(), normalized.square().mean(1),
                               normalized.pow(3).mean(1), normalized.max(1).values), dim=1)
        correlations = []
        for lag in (1, 20):
            cross = (z[:, lag:] * z[:, :-lag].conj()).mean(1) / scale[:, 0]
            correlations.extend((cross.real, cross.imag))
        stats = torch.cat((moments, torch.stack(correlations, dim=1)), dim=1)
        # Smoothly bound high-order moments from rare impulsive packets.
        stats = 8 * torch.tanh(stats / 8)
        return torch.cat((self.linear(x), stats), dim=1)

    def interaction(self, x):
        return torch.cat((self.linear(x), self.temporal(x).flatten(1)), dim=1)


class BoundedConditionalOperator(nn.Module):
    """Low-rank state-conditioned action; q=0 produces exactly zero delta."""
    def __init__(self, dimension=INTERMEDIATE_DIM, rank=16, state_dim=16, radius=.25):
        super().__init__()
        self.radius = float(radius)
        self.left = nn.Linear(rank, dimension, bias=False)
        self.right = nn.Linear(dimension, rank, bias=False)
        self.shift = nn.Linear(rank, dimension, bias=False)
        self.coefficients = nn.Linear(state_dim, 2 * rank, bias=False)
        nn.init.normal_(self.left.weight, std=.025)
        nn.init.normal_(self.shift.weight, std=.025)

    def forward(self, h, q):
        a, b = self.coefficients(q).tanh().chunk(2, dim=1)
        raw = self.left(self.right(h) * a) + self.shift(b)
        cap = self.radius * torch.sqrt(h.square().sum(1, keepdim=True) + 1e-8)
        # Smooth norm bound, finite at h=0 and q=0.
        return raw / torch.sqrt(1 + raw.square().sum(1, keepdim=True) / cap.square())


class DisentanglementBranch(nn.Module):
    """Independent learned view encoder and action model, no TX classifier."""
    def __init__(self, kind, dimension=INTERMEDIATE_DIM, rank=16, state_dim=16):
        super().__init__()
        self.kind = kind
        self.views = FixedViews()
        width = {'linear': 29, 'receiver': 37, 'interaction': 189}.get(kind)
        if kind == 'temporal':
            self.encoder = nn.Sequential(nn.Conv1d(8, 32, 3, padding=1), nn.SiLU(),
                nn.Conv1d(32, 32, 3, padding=1), nn.SiLU(), nn.Flatten(),
                nn.Linear(32 * 20, state_dim))
        elif width is not None:
            self.encoder = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, state_dim))
        else:
            raise ValueError('Unknown disentanglement branch')
        self.operator = BoundedConditionalOperator(dimension, rank, state_dim)
        self.parameter_dim = {'linear': 4, 'temporal': 3, 'receiver': 0, 'interaction': 0}[kind]
        self.parameter_head = (nn.Linear(state_dim, self.parameter_dim, bias=False)
                               if self.parameter_dim else None)

    def view(self, x):
        return getattr(self.views, self.kind)(x)

    def encode_pair(self, view0, view1):
        # This guarantees identical observations have q=0 at every parameter state.
        return self.encoder(view1) - self.encoder(view0)

    def forward_views(self, view0, view1, h):
        q = self.encode_pair(view0, view1)
        return dict(q=q, delta=self.operator(h, q),
                    parameters=self.parameter_head(q) if self.parameter_head is not None else None)

    def forward(self, x0, x1, h):
        return self.forward_views(self.view(x0), self.view(x1), h)


class AuxiliaryNetworks(nn.Module):
    """Three independent main branches and an optional factorial interaction."""
    def __init__(self, dimension=INTERMEDIATE_DIM, rank=8, state_dim=8, include_interaction=False,
                 active=None, dim=None, qdim=None):
        super().__init__()
        dimension = dimension if dim is None else dim
        state_dim = state_dim if qdim is None else qdim
        if dimension != INTERMEDIATE_DIM:
            raise ValueError('Registered identity split requires dimension=349')
        self.dimension = dimension
        selected = tuple(active) if active is not None else KINDS[:3] + (('interaction',) if include_interaction else ())
        if any(k not in KINDS for k in selected):
            raise ValueError('Unknown active branch')
        for kind in KINDS:
            setattr(self, kind, DisentanglementBranch(kind, dimension, rank, state_dim) if kind in selected else None)

    def branch(self, kind):
        if kind not in KINDS or getattr(self, kind) is None:
            raise ValueError('Unregistered or disabled auxiliary branch: ' + str(kind))
        return getattr(self, kind)

    def predict(self, kind, x0, x1, h, descriptor=None):
        branch = self.branch(kind)
        if kind != 'interaction':
            return branch(x0, x1, h)
        if descriptor is None or not all(k in descriptor for k in ('x_linear', 'x_temporal')):
            raise ValueError('Interaction requires separate observed linear and temporal views')
        view0 = branch.view(x0)
        q_l = branch.encode_pair(view0, branch.view(descriptor['x_linear']))
        q_t = branch.encode_pair(view0, branch.view(descriptor['x_temporal']))
        q = q_l * q_t
        return dict(q=q, delta=branch.operator(h, q), parameters=None)

    def propose(self, kind, h, q):
        return self.branch(kind).operator(h, q)

    def contract(self):
        return dict(intermediate_dimension=self.dimension,
                    split=['native_base_pre_layernorm160', 'native_pa_pre_layernorm160', 'reference_observables29'],
                    independent_branches=[k for k in KINDS if getattr(self, k) is not None],
                    identity_forward_unchanged=True, training_only=True,
                    oracle_parameters_in_action=False, receiver_supervision='matched source group statistics',
                    operator_radius=.25, zero_intervention_exact_identity=True,
                    physical_factor_recovery_claim=False)


class UnifiedBranch(nn.Module):
    """Single combined-view encoder/operator for the matched-capability control."""
    def __init__(self, dimension, rank, state_dim):
        super().__init__()
        self.views = FixedViews()
        # 160 hidden units approximately match the total three-branch capacity.
        self.encoder = nn.Sequential(nn.Linear(197, 160), nn.SiLU(), nn.Linear(160, state_dim))
        self.operator = BoundedConditionalOperator(dimension, rank, state_dim)
        self.conditions = nn.Embedding(3, state_dim)
        self.parameter_head = nn.Linear(state_dim, 7, bias=False)

    def view(self, x):
        return torch.cat((self.views.receiver(x), self.views.temporal(x).flatten(1)), dim=1)

    def forward_views(self, view0, view1, h, kind='receiver'):
        if kind not in KINDS[:3]:
            raise ValueError('Unified control supports L/T/R main effects only')
        index = KINDS.index(kind)
        q = (self.encoder(view1) - self.encoder(view0)) * (1 + self.conditions.weight[index].tanh())
        parameters = self.parameter_head(q)
        parameters = parameters[:, :4] if kind == 'linear' else parameters[:, 4:] if kind == 'temporal' else None
        return dict(q=q, delta=self.operator(h, q), parameters=parameters)


class UnifiedAuxiliaryNetworks(nn.Module):
    """One type-conditioned nuisance network, same observed inputs and task targets."""
    def __init__(self, dimension=INTERMEDIATE_DIM, rank=8, state_dim=8, dim=None, qdim=None, **kwargs):
        super().__init__()
        dimension = dimension if dim is None else dim
        state_dim = state_dim if qdim is None else qdim
        if dimension != INTERMEDIATE_DIM:
            raise ValueError('Registered identity split requires dimension=349')
        self.shared = UnifiedBranch(dimension, rank, state_dim)

    def branch(self, kind):
        if kind not in KINDS[:3]:
            raise ValueError('Unified control does not include interaction')
        return self.shared

    def predict(self, kind, x0, x1, h, descriptor=None):
        b = self.branch(kind)
        return b.forward_views(b.view(x0), b.view(x1), h, kind=kind)

    def propose(self, kind, h, q):
        return self.branch(kind).operator(h, q)

    def contract(self):
        return dict(intermediate_dimension=INTERMEDIATE_DIM, shared_encoder_and_operator=True,
                    tasks=list(KINDS[:3]), oracle_parameters_in_action=False,
                    identity_forward_unchanged=True, training_only=True,
                    zero_intervention_exact_identity=True, physical_factor_recovery_claim=False)
