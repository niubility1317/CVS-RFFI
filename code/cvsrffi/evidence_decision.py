"""Frozen source-V calibration and selective decision diagnostics.

Model mismatch is a diagnostic hypothesis, never an unknown-transmitter label.
No source examples are retained and query prediction cannot update calibration.
"""
import math
import torch
from torch.nn import functional as F


def _validate(logits, mahalanobis, observed_count, quality):
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError('nonempty logits [N,C] required')
    n = logits.shape[0]
    if mahalanobis.shape != logits.shape or observed_count.shape != (n,) or quality.shape[0] != n:
        raise ValueError('incompatible calibration tensors')
    if not torch.isfinite(logits).all() or not torch.isfinite(mahalanobis).all() or (mahalanobis < 0).any():
        raise ValueError('finite logits and nonnegative finite mahalanobis required')
    if not torch.isfinite(observed_count).all() or (observed_count < 0).any() or (observed_count != observed_count.floor()).any():
        raise ValueError('observed_count must be nonnegative integers')
    if not torch.isfinite(quality).all() or quality.numel() == 0:
        raise ValueError('finite nonempty quality required')


class ConditionAwareCalibrator:
    """Temperature and class-agnostic consistency limits fitted only on role V.

    Quality means are clipped to [0,1] and assigned fixed bins. Limits use the
    true-class squared Mahalanobis divided by observed dimension in each group.
    Groups with insufficient V support yield no consistency claim and defer.
    This is empirical calibration, not a finite-sample risk guarantee.
    """
    def __init__(self, quality_bins=3, min_group_samples=20,
                 consistency_quantile=.95, min_confidence=.5):
        if quality_bins < 1 or min_group_samples < 1 or not 0 < consistency_quantile < 1 or not 0 <= min_confidence <= 1:
            raise ValueError('invalid calibration configuration')
        self.quality_bins = int(quality_bins)
        self.min_group_samples = int(min_group_samples)
        self.consistency_quantile = float(consistency_quantile)
        self.min_confidence = float(min_confidence)
        self.temperature = 1.0
        self.groups = {}
        self.fitted = False

    def _bins(self, quality):
        q = quality.reshape(quality.shape[0], -1).mean(-1).clamp(0, 1)
        return (q * self.quality_bins).long().clamp_max(self.quality_bins-1)

    @staticmethod
    def _coverage(state_coverage, n, device):
        if state_coverage is None:
            return torch.ones(n, dtype=torch.bool, device=device)
        if state_coverage.shape != (n,) or not torch.isfinite(state_coverage).all():
            raise ValueError('state_coverage must be finite [N]')
        # Coverage is an externally defined supported-state indicator, not a
        # learned TX feature. Numeric values use the explicit 0.5 boundary.
        return state_coverage >= .5

    def fit(self, logits, mahalanobis, observed_count, quality, labels, *, role='V', state_coverage=None):
        if role != 'V':
            raise ValueError('calibration fitting is permitted only for source V')
        if self.fitted:
            raise RuntimeError('calibration is frozen; construct a new calibrator to refit')
        _validate(logits, mahalanobis, observed_count, quality)
        if labels.shape != (logits.shape[0],) or labels.dtype != torch.long or (labels < 0).any() or (labels >= logits.shape[1]).any():
            raise ValueError('labels must be valid int64 [N]')
        covered = self._coverage(state_coverage, logits.shape[0], logits.device)
        # A bounded deterministic scalar search avoids optimizer state or sample
        # retention. One global temperature cannot encode class identity.
        x, y = logits.detach().double(), labels.detach()
        def loss(t):
            return F.cross_entropy(x / math.exp(t), y).item()
        left, right = math.log(.05), math.log(20.)
        for _ in range(48):
            a, b = left+(right-left)/3, right-(right-left)/3
            if loss(a) <= loss(b):
                right = b
            else:
                left = a
        temperature = math.exp((left+right)/2)
        bins = self._bins(quality)
        standardized = mahalanobis.detach().gather(1, labels[:,None]).squeeze(1) / observed_count.clamp_min(1)
        groups = {}
        for dim in observed_count.unique().tolist():
            if dim == 0:
                continue
            for qbin in range(self.quality_bins):
                mask = (observed_count == dim) & (bins == qbin) & covered
                count = int(mask.sum().item())
                if count >= self.min_group_samples:
                    limit = torch.quantile(standardized[mask].double(), self.consistency_quantile, interpolation='higher').item()
                    groups[f'{int(dim)}:{qbin}'] = dict(limit=float(limit), count=count)
        self.temperature, self.groups, self.fitted = temperature, groups, True
        return self

    def predict(self, logits, mahalanobis, observed_count, quality, *, state_coverage=None):
        if not self.fitted:
            raise RuntimeError('fit source-V calibration before prediction')
        _validate(logits, mahalanobis, observed_count, quality)
        probabilities = (logits / self.temperature).softmax(-1)
        confidence, top = probabilities.max(-1)
        bins = self._bins(quality)
        limits = logits.new_full((logits.shape[0],), float('nan'))
        available = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        for key, group in self.groups.items():
            dim, qbin = map(int, key.split(':'))
            mask = (observed_count == dim) & (bins == qbin)
            limits[mask] = group['limit']
            available[mask] = True
        standardized = mahalanobis.gather(1,top[:,None]).squeeze(1) / observed_count.clamp_min(1)
        consistent = available & (standardized <= limits)
        coverage = self._coverage(state_coverage, logits.shape[0], logits.device)
        sufficient = (observed_count > 0) & coverage & (confidence >= self.min_confidence)
        accepted = sufficient & consistent
        mismatch = available & (observed_count > 0) & coverage & ~consistent
        status = ['identify' if a else 'model_mismatch_candidate' if m else 'defer'
                  for a,m in zip(accepted.tolist(),mismatch.tolist())]
        return dict(top_class=top, accepted=accepted, status=status,
                    evidence_sufficiency=sufficient, consistency=consistent,
                    calibration_available=available, consistency_limit=limits,
                    standardized_mahalanobis=standardized, probabilities=probabilities)

    def state_dict(self):
        return dict(version=1, quality_bins=self.quality_bins,
                    min_group_samples=self.min_group_samples,
                    consistency_quantile=self.consistency_quantile,
                    min_confidence=self.min_confidence, temperature=self.temperature,
                    fitted=self.fitted, groups={k:dict(v) for k,v in self.groups.items()})

    @classmethod
    def from_state_dict(cls, state):
        if state.get('version') != 1:
            raise ValueError('unsupported calibration version')
        obj = cls(**{k:state[k] for k in ('quality_bins','min_group_samples','consistency_quantile','min_confidence')})
        if not math.isfinite(state['temperature']) or state['temperature'] <= 0:
            raise ValueError('invalid temperature')
        groups = {}
        for key, value in state['groups'].items():
            dim,qbin = map(int,key.split(':'))
            if dim <= 0 or not 0 <= qbin < obj.quality_bins or not math.isfinite(value['limit']) or value['limit'] < 0 or value['count'] < obj.min_group_samples:
                raise ValueError('invalid condition calibration group')
            groups[key] = dict(limit=float(value['limit']),count=int(value['count']))
        obj.temperature, obj.fitted, obj.groups = float(state['temperature']),bool(state['fitted']),groups
        return obj


def selective_metrics(logits, labels, accepted, confidence=None, ece_bins=15):
    """All-row probabilities plus selective risk. Rejects count as wrong.

    The curve varies a confidence threshold over all rows and groups ties, so
    it does not depend on row order. It describes ranking, not recalibration.
    """
    if logits.ndim != 2 or logits.shape[0] == 0 or labels.shape != logits.shape[:1] or accepted.shape != labels.shape or accepted.dtype != torch.bool or ece_bins < 1:
        raise ValueError('invalid metric tensors')
    if not torch.isfinite(logits).all() or labels.dtype != torch.long or (labels < 0).any() or (labels >= logits.shape[1]).any():
        raise ValueError('finite logits and valid int64 labels required')
    prob = logits.softmax(-1)
    conf, top = prob.max(-1)
    correct = top == labels
    ranking = conf if confidence is None else confidence
    if ranking.shape != labels.shape or not torch.isfinite(ranking).all():
        raise ValueError('ranking confidence must be finite [N]')
    n, na = labels.numel(), int(accepted.sum())
    ece = 0.0
    bins = (conf * ece_bins).long().clamp_max(ece_bins-1)
    for i in range(ece_bins):
        mask = bins == i
        if mask.any():
            ece += (mask.float().mean() * (correct[mask].float().mean()-conf[mask].mean()).abs()).item()
    curve = [dict(coverage=0.,risk=None,threshold=None)]
    sorted_conf, order = ranking.sort(descending=True)
    cumulative_correct = correct[order].long().cumsum(0)
    endpoints = torch.cat((sorted_conf[:-1] != sorted_conf[1:], torch.ones(1,dtype=torch.bool,device=logits.device))).nonzero().flatten()
    for end in endpoints.tolist():
        kept = end + 1
        curve.append(dict(coverage=kept/n,risk=1-float(cumulative_correct[end])/kept,threshold=float(sorted_conf[end])))
    return dict(accuracy=float((correct & accepted).sum())/n,
                closed_set_accuracy=float(correct.float().mean()),
                accepted_risk=1-float(correct[accepted].float().mean()) if na else None,
                coverage=na/n, nll=float(F.cross_entropy(logits,labels)),
                brier=float((prob-F.one_hot(labels,logits.shape[1])).square().sum(-1).mean()),
                ece=ece, count=n, accepted_count=na, risk_coverage=curve)
