"""Fixed ordinary support-only cosine readout diagnostic on frozen joint160.

This parameterizes the existing six-class source readout over an explicitly
registered class count. It is not a new classifier optimization algorithm or a
claim of validated Phase2 performance. The API accepts frozen feature tensors
and support labels; it provides no file loader, teacher or fitted feature
transform. A protocol-aware caller must authorize support or permitted frozen
knowledge before calling it. Only fresh W is returned as learned inference
state; audit values describe the head budget.
"""
import math
import torch
from torch.nn import functional as F

FEATURE_DIM = 160
EPSILON = 1e-4
SCALE = 30.0
UPDATES = 20


def _class_count(num_classes):
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError('Registered num_classes must be an explicit positive integer')


def _features(value, name):
    if (not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != FEATURE_DIM
            or len(value) == 0):
        raise ValueError(name + ' must be nonempty features[N,160]')
    if value.dtype != torch.float32 or value.requires_grad:
        raise ValueError(name + ' must be frozen float32 features')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' contains nonfinite features')


def _scores(normalized, weight):
    return SCALE * F.linear(normalized, F.normalize(weight, dim=1, eps=EPSILON))


def fit(features, labels, seed, *, num_classes):
    """Fit fresh W with exactly 20 complete support-batch Adam updates.

    The private generator reproduces the existing six-class Xavier draws and
    full-batch permutations. Row normalization precedes the unchanged cosine CE
    loop. Labels describe only supplied registered support samples.
    """
    _class_count(num_classes)
    _features(features, 'support')
    if (not torch.is_tensor(labels) or labels.dtype != torch.int64
            or labels.shape != (len(features),) or labels.device != features.device):
        raise ValueError('Support labels must be matching int64 labels[N] on the same device')
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError('Support label lies outside the registered class table')
    if bool((torch.bincount(labels, minlength=num_classes) == 0).any()):
        raise ValueError('Every registered class requires at least one support sample')
    if type(seed) is not int:
        raise ValueError('The fixed readout seed must be an integer')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    bound = math.sqrt(6. / (num_classes + FEATURE_DIM))
    initial = torch.empty(num_classes, FEATURE_DIM, dtype=torch.float32, device='cpu').uniform_(
        -bound, bound, generator=generator)
    weight = torch.nn.Parameter(initial.to(device=features.device))
    optimizer = torch.optim.Adam([weight], lr=.01, weight_decay=0.)
    losses = []
    with torch.autocast(device_type=features.device.type, enabled=False):
        normalized = F.normalize(features, dim=1, eps=EPSILON)
        for step in range(UPDATES):
            order = torch.randperm(len(normalized), generator=generator).to(device=features.device)
            scores = _scores(normalized[order], weight)
            loss = F.cross_entropy(scores, labels[order])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Ordinary support cosine loss is nonfinite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if weight.grad is None or not bool(torch.isfinite(weight.grad).all()):
                raise FloatingPointError('Ordinary support cosine W gradient is nonfinite')
            optimizer.step()
            # Retain the existing full-batch loss aggregation exactly.
            losses.append(float(loss.detach()) * len(order) / len(normalized))
    if not bool(torch.isfinite(weight).all()):
        raise FloatingPointError('Ordinary support cosine W is nonfinite')
    return weight.detach(), dict(epochs=UPDATES, updates=UPDATES, losses=losses,
            feature_updates=0, head_updates=UPDATES, num_classes=num_classes,
            feature_dimension=FEATURE_DIM, support_count=len(features),
            initialization='fresh seeded Xavier; no inherited W', bias=False,
            normalization_epsilon=EPSILON, scale=SCALE, optimizer='Adam',
            learning_rate=.01, weight_decay=0., full_support_batch=True,
            inference_trainable_state='W only', query_fit=False,
            scope='fixed ordinary cosine readout diagnostic; no Phase2 validation claim')


def logits(features, weight, *, num_classes):
    """Score each frozen query independently against every registered W row."""
    _class_count(num_classes)
    _features(features, 'query')
    if (not torch.is_tensor(weight) or weight.shape != (num_classes, FEATURE_DIM)
            or weight.dtype != torch.float32 or weight.device != features.device
            or weight.requires_grad or not bool(torch.isfinite(weight).all())):
        raise ValueError('Frozen finite float32 W must match the registered class table and device')
    with torch.autocast(device_type=features.device.type, enabled=False):
        return _scores(F.normalize(features, dim=1, eps=EPSILON), weight)
