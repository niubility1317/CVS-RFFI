"""H0-anchored, shared bounded angular metric. No covariance or class bias."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _weights(w, tau):
    if w.ndim != 2 or min(w.shape) < 2 or not torch.isfinite(w).all():
        raise ValueError('finite nonempty class directions required')
    if not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError('positive finite H0 scale required')
    if (w.norm(dim=-1) < 1e-4).any():
        raise ValueError('degenerate H0 class direction')


def metric_logits(h, w, q, a, tau, eps=1e-4):
    if h.ndim != 2 or h.shape[1] != w.shape[1] or not torch.isfinite(h).all():
        raise ValueError('finite features with matching width required')
    c = torch.expm1(a)
    hq, wq = h @ q, w @ q
    numerator = h @ w.T + (hq * c) @ wq.T
    hn = (h.square().sum(-1) + (hq.square() * c).sum(-1)).clamp_min(eps**2).sqrt()
    wn = (w.square().sum(-1) + (wq.square() * c).sum(-1)).clamp_min(eps**2).sqrt()
    return tau * numerator / (hn[:, None] * wn[None, :])


class AnchoredMetricHead(nn.Module):
    def __init__(self, w0, tau0, rank=4, rho=math.log(2)/2):
        super().__init__(); _weights(w0, tau0)
        if not 1 <= rank <= w0.shape[1] or not math.isfinite(rho) or rho <= 0:
            raise ValueError('invalid metric rank/rho')
        self.register_buffer('w0', w0.detach().clone())
        self.tau0, self.rank, self.rho = float(tau0), int(rank), float(rho)
        q = torch.linalg.qr(torch.randn(w0.shape[1], rank, device=w0.device, dtype=w0.dtype), mode='reduced').Q
        self.B = nn.Parameter(q)
        self.b = nn.Parameter(w0.new_zeros(rank))

    def spectral(self):
        if not torch.isfinite(self.B).all() or not torch.isfinite(self.b).all():
            raise ValueError('nonfinite metric parameters')
        q, r = torch.linalg.qr(self.B, mode='reduced')
        if (r.diagonal().abs() < 1e-7).any():
            raise ValueError('metric basis is rank deficient')
        # Fix arbitrary QR signs, while preserving a smooth local parameterization.
        signs = torch.where(r.diagonal() >= 0, 1., -1.)
        return q * signs, self.rho * self.b.tanh()

    def forward(self, h):
        q, a = self.spectral()
        return metric_logits(h.to(self.w0), self.w0, q, a, self.tau0)

    @torch.no_grad()
    def geometry_diagnostics(self):
        q, a = self.spectral()
        eigen = torch.cat((a.exp(), a.new_ones(self.w0.shape[1]-self.rank)))
        return dict(eigenvalues=eigen.cpu().tolist(), condition_number=float(eigen.max()/eigen.min()),
                    orthogonality_error=float((q.T@q-torch.eye(self.rank,device=q.device,dtype=q.dtype)).abs().max()),
                    a=a.cpu().tolist(), direction_angles_deg=[0.] * len(self.w0))

    @torch.no_grad()
    def export_state(self):
        q, a = self.spectral()
        return dict(kind='bounded_metric_v1', w0=self.w0.cpu().clone(), tau0=self.tau0,
                    rank=self.rank, rho=self.rho, q=q.cpu().clone(), a=a.cpu().clone())

    @classmethod
    def from_export(cls, state):
        if set(state) != {'kind','w0','tau0','rank','rho','q','a'} or state['kind'] != 'bounded_metric_v1':
            raise ValueError('invalid metric export schema')
        _weights(state['w0'],state['tau0'])
        if not isinstance(state['rank'],int) or not 1<=state['rank']<=state['w0'].shape[1] or not math.isfinite(state['rho']) or state['rho']<=0:
            raise ValueError('invalid exported metric rank/rho')
        q, a = state['q'], state['a']
        if q.shape != (state['w0'].shape[1],state['rank']) or a.shape != (state['rank'],):
            raise ValueError('invalid metric export shapes')
        if not torch.isfinite(q).all() or not torch.isfinite(a).all() or (a.abs() > state['rho']+1e-7).any():
            raise ValueError('invalid bounded metric export')
        if not torch.allclose(q.T@q,torch.eye(len(a),dtype=q.dtype,device=q.device),atol=1e-5,rtol=1e-5):
            raise ValueError('exported basis must be orthonormal')
        return FrozenMetricHead(state)


class FrozenMetricHead(nn.Module):
    """Canonical deployment uses Q,a and precomputed class projections."""
    def __init__(self,state):
        super().__init__(); self.tau0=float(state['tau0']);self.rank=int(state['rank']);self.rho=float(state['rho'])
        for key in ('w0','q','a'):self.register_buffer(key,state[key].detach().clone())
        self.register_buffer('wq',self.w0 @ self.q)
        self.register_buffer('wn',(self.w0.square().sum(-1)+(self.wq.square()*torch.expm1(self.a)).sum(-1)).clamp_min(1e-8).sqrt())

    def forward(self,h):
        h=h.to(self.w0)
        if h.ndim!=2 or h.shape[1]!=self.w0.shape[1] or not torch.isfinite(h).all():
            raise ValueError('finite matching features required')
        hq=h@self.q;c=torch.expm1(self.a)
        hn=(h.square().sum(-1)+(hq.square()*c).sum(-1)).clamp_min(1e-8).sqrt()
        return self.tau0*(h@self.w0.T+(hq*c)@self.wq.T)/(hn[:,None]*self.wn[None])

    def export_state(self):
        return dict(kind='bounded_metric_v1',w0=self.w0.cpu().clone(),q=self.q.cpu().clone(),a=self.a.cpu().clone(),
                    tau0=self.tau0,rank=self.rank,rho=self.rho)


class OrdinaryAngleHead(nn.Module):
    def __init__(self,w0,tau0):
        super().__init__();_weights(w0,tau0)
        self.weight=nn.Parameter(w0.detach().clone());self.tau0=float(tau0)
        self.register_buffer('initial_weight',w0.detach().clone())

    def forward(self,h):
        if h.ndim!=2 or h.shape[1]!=self.weight.shape[1] or not torch.isfinite(h).all():
            raise ValueError('finite matching features required')
        return self.tau0*F.normalize(h.to(self.weight),dim=-1,eps=1e-4)@F.normalize(self.weight,dim=-1,eps=1e-4).T

    def geometry_diagnostics(self):
        cosine=(F.normalize(self.weight.detach(),dim=-1)*F.normalize(self.initial_weight,dim=-1)).sum(-1).clamp(-1,1)
        return dict(eigenvalues=[1.]*self.weight.shape[1],condition_number=1.,orthogonality_error=0.,a=[],
                    direction_angles_deg=(cosine.acos()*180/math.pi).cpu().tolist())

    def export_state(self):
        return dict(kind='ordinary_angle_v1',weight=self.weight.detach().cpu().clone(),tau0=self.tau0)


def load_angle_head(state):
    if state.get('kind')=='bounded_metric_v1':return AnchoredMetricHead.from_export(state)
    if state.get('kind')=='ordinary_angle_v1' and set(state)=={'kind','weight','tau0'}:
        return OrdinaryAngleHead(state['weight'],state['tau0']).requires_grad_(False).eval()
    raise ValueError('unsupported angular head state')
