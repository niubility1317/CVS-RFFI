"""Support-only registration and immutable-query H4/H5 prediction.

No query is accepted during fitting. Class names are explicit, never offsets.
Source parameters must already satisfy the experiment's lineage contract.
"""
import copy
import json
from dataclasses import asdict

import torch
from torch.nn import functional as F

from .conditional_response import SupportResponsePosterior, fit_support_response
from .evidence_head import EvidenceHead
from .partial_gaussian_head import gaussian_scores
from .pairwise_evidence import project_pairwise


def _labels(labels):
    labels = tuple(labels)
    if any(type(v) not in (int, str) for v in labels) or len(set(labels)) != len(labels):
        raise ValueError('class labels must be distinct integers or strings')
    if not labels:
        raise ValueError('at least one registered class required')
    return labels


class FrozenRegisteredEvidence:
    """Frozen model snapshot; predict scores every row against every class.

    Result observations must come from the same frozen source observation
    pipeline. Only z, observed, state and observation_variance are consumed;
    query means, source scores and labels are never used for registration.
    """
    def __init__(self, head, class_labels, source_class_labels, posteriors):
        self._head = copy.deepcopy(head).eval()
        self._head.requires_grad_(False)
        self.class_labels = _labels(class_labels)
        self._source_labels = _labels(source_class_labels)
        self._posteriors = tuple(SupportResponsePosterior.from_dict(p.to_dict()) for p in posteriors)
        if len(self._posteriors) != len(self.class_labels):
            raise ValueError('one posterior per class required')

    def _base_covariance(self, result):
        head = self._head
        state, obsvar = result['state'], result['observation_variance']
        if obsvar.shape != result['z'].shape or not torch.isfinite(obsvar).all() or (obsvar <= 0).any():
            raise ValueError('positive finite observation_variance[B,D] required')
        response_cov = torch.diag(F.softplus(head.raw_diagonal)+1e-6)+head.factor@head.factor.T
        jac = head.response.jacobian(state)
        shared_jac = jac.mean(1)
        u = head.state_error
        covariances = []
        for label in self.class_labels:
            j = jac[:, self._source_labels.index(label)] if label in self._source_labels else shared_jac
            state_cov = j@u@j.transpose(-1,-2)
            outside=self._head.response.domain_diagnostics(state)['outside_distance'].square()*head.config.response_variance
            state_cov=state_cov+outside[:,None,None]*torch.eye(head.feature_dim,device=state.device,dtype=state.dtype)
            covariances.append(response_cov+torch.diag_embed(obsvar)+state_cov)
        return torch.stack(covariances,1)

    @torch.no_grad()
    def predict(self, result, *, include_support_uncertainty=True):
        """The switch is an explicit ablation; production default includes it."""
        phi = self._head.response.basis(result['state'])
        predictions = [p.predict(phi) for p in self._posteriors]
        means = torch.stack([p[0] for p in predictions],1)
        support_cov = torch.stack([p[1] for p in predictions],1)
        covariance = self._base_covariance(result)
        if include_support_uncertainty:
            covariance = covariance+support_cov
        scored = gaussian_scores(result['z'],means,covariance,result['observed'])
        base = scored['scores']
        scores = base
        b,c = base.shape
        residual = base.new_zeros(b,c,c)
        if self._head.pair is not None:
            residual = self._head.config.pair_strength*self._head.pair(
                result['z'],means,result['state'],result['observation_variance'],
                covariance.diagonal(dim1=-2,dim2=-1),result['observed'])
            differences = base[:,:,None]-base[:,None,:]+residual
            weights = torch.ones_like(differences)-torch.eye(c,device=base.device)[None]
            scores = project_pairwise(base,differences,weights,self._head.config.pair_anchor)
        scores = torch.where(result['observed'].any(-1,keepdim=True),scores,torch.zeros_like(scores))
        return {**scored,'scores':scores,'base_scores':base,'class_labels':self.class_labels,
                'means':means,'covariance':covariance,'support_covariance':support_cov,
                'pair_residual':residual,'state_domain':self._head.response.domain_diagnostics(result['state']),
                'coverage':tuple(p.coverage(phi) for p in self._posteriors)}

    def to_dict(self):
        """Only tensors; exact physical IDs belong in separate run metadata."""
        metadata = {'version':1,'config':asdict(self._head.config),'feature_dim':self._head.feature_dim,
                    'class_labels':self.class_labels,'source_class_labels':self._source_labels}
        result = {'metadata':torch.tensor(list(json.dumps(metadata).encode('utf-8')),dtype=torch.uint8)}
        result.update({f'head.{k}':v.detach().clone() for k,v in self._head.state_dict().items()})
        for i,p in enumerate(self._posteriors):
            result.update({f'posterior.{i}.{k}':v for k,v in p.to_dict().items()})
        return result

    @classmethod
    def from_dict(cls, state):
        metadata = json.loads(bytes(state['metadata'].cpu().tolist()).decode('utf-8'))
        if metadata['version'] != 1:
            raise ValueError('unsupported registration snapshot version')
        hstate = {k[5:]:v for k,v in state.items() if k.startswith('head.')}
        reference = hstate['raw_diagonal']
        head = EvidenceHead(len(metadata['source_class_labels']),metadata['feature_dim'],metadata['config'])
        head = head.to(device=reference.device,dtype=reference.dtype)
        head.load_state_dict(hstate,strict=True)
        posts = []
        for i in range(len(metadata['class_labels'])):
            prefix=f'posterior.{i}.'
            posts.append(SupportResponsePosterior.from_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}))
        return cls(head,metadata['class_labels'],metadata['source_class_labels'],posts)


@torch.no_grad()
def fit_registered_evidence(head, support_result, support_labels, physical_ids, class_labels,
                            *, source_class_labels):
    """Register all classes with legal support only, using identical Gaussian updates.

    Known source classes inherit their learned prior mean. New labels receive
    the source class-average prior (shared slopes and mean), independent of any
    query or nearest-class selection. Noise is frozen shared response + supplied
    per-support observation variance + prior state propagation.
    """
    if head.stage not in (4,5):
        raise ValueError('registered evidence requires H4 or H5')
    classes, source = _labels(class_labels), _labels(source_class_labels)
    if len(source) != head.num_classes:
        raise ValueError('source_class_labels must map every source class')
    labels = tuple(support_labels)
    n = len(support_result['z'])
    if len(labels) != n or len(physical_ids) != n or len(set(physical_ids)) != n:
        raise ValueError('support labels and distinct physical IDs required')
    if any(label not in classes for label in labels):
        raise ValueError('support labels must belong to registered classes')
    # Temporary snapshot provides the same old/new covariance policy as predict.
    frozen_head = copy.deepcopy(head).eval()
    frozen_head.requires_grad_(False)
    basis = frozen_head.response.basis(support_result['state'])
    coefficients = frozen_head.response.coefficients()
    jac = frozen_head.response.jacobian(support_result['state'])
    obsvar = support_result['observation_variance']
    if obsvar.shape != support_result['z'].shape or not torch.isfinite(obsvar).all() or (obsvar <= 0).any():
        raise ValueError('positive finite support observation variance required')
    response_cov = torch.diag(F.softplus(frozen_head.raw_diagonal)+1e-6)+frozen_head.factor@frozen_head.factor.T
    posts = []
    for label in classes:
        take = [i for i,y in enumerate(labels) if y == label]
        if not take:
            raise ValueError('each registered class requires at least one support shot')
        old = source.index(label) if label in source else None
        prior = coefficients[old] if old is not None else coefficients.mean(0)
        j = jac[take,old] if old is not None else jac[take].mean(1)
        covariance = response_cov+torch.diag_embed(obsvar[take])+j@frozen_head.state_error@j.transpose(-1,-2)
        posts.append(fit_support_response(support_result['z'][take],basis[take],covariance,
                     support_result['observed'][take],[physical_ids[i] for i in take],
                     prior_mean=prior,prior_precision=head.config.prior_precision))
    return FrozenRegisteredEvidence(frozen_head,classes,source,posts)
