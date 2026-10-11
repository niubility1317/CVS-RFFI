"""Source-only descriptive geometry; fitted readouts never select identity weights."""
from itertools import combinations
import math
import torch
from torch.nn import functional as F
from .features import identity
from experiments.cvs_stage2_decouple.actions import apply_physical_action
from experiments.cvs_multi_action_risk.physics import phase_process

EPS = .01
OMEGA = 1. / 6.
RIDGE = 1e-5


def summarize(v):
    v = torch.as_tensor(v).detach().cpu().double().flatten()
    if not len(v):
        return dict(count=0, mean=None)
    if not torch.isfinite(v).all():
        raise FloatingPointError('Nonfinite source geometry observations')
    return dict(count=len(v), mean=float(v.mean()), rms=float(v.square().mean().sqrt()),
        quantiles=torch.quantile(v, torch.tensor([0., .05, .5, .95, 1.], dtype=v.dtype)).tolist())


def physical(x, a, kind, noise=None):
    if kind == 'linear':
        p = a
    else:
        p = torch.stack((a[:,0]*.25, a[:,1]*3120., a[:,2]*.025,
                         torch.zeros_like(a[:,0]), a[:,3]*.25), 1)
    return apply_physical_action(x, p, kind, noise)


@torch.no_grad()
def extract_responses(model, cache, seed, device, batch_size=128):
    """All L_s, central signed local derivatives; limited explicit convergence probe."""
    model.eval()
    features, logits_all, jacobians = [], [], []
    convergence, linearization, interactions = [], [], []
    for start in range(0, len(cache['x']), batch_size):
        x = cache['x'][start:start+batch_size].to(device)
        logits, z = identity(model, x, verify=True)
        g = torch.Generator().manual_seed(seed+180001+start)
        noise = phase_process(len(x), g, x)
        js = []
        for kind in ('linear','temporal'):
            for axis in range(4):
                a = x.new_zeros((len(x),4));a[:,axis] = EPS
                plus = identity(model, physical(x,a,kind,noise))[1]
                minus = identity(model, physical(x,-a,kind,noise))[1]
                j = (plus-minus)/(2*EPS);js.append(j)
                if start == 0:
                    half_plus = identity(model,physical(x,a/2,kind,noise))[1]
                    half_minus = identity(model,physical(x,-a/2,kind,noise))[1]
                    half_j = (half_plus-half_minus)/EPS
                    convergence.append(dict(kind=kind,axis=axis,samples=len(x),
                        absolute_error=summarize((j-half_j).norm(dim=1)),
                        relative_error=summarize((j-half_j).norm(dim=1)/(half_j.norm(dim=1)+1e-8))))
        j = torch.stack(js,2)
        # Independent signed local scale prior; finite actions also audited at
        # normalized amplitude .25 and1, without choosing a good-looking scale.
        a = (2*torch.rand(len(x),4,generator=g)-1).to(x)
        a[:,2] = a[:,2].abs()
        for kind, offset in (('linear',0),('temporal',4)):
            for scale in (.25,1.):
                actual = identity(model,physical(x,a*scale,kind,noise))[1]-z
                predicted = torch.einsum('ndk,nk->nd',j[:,:,offset:offset+4],a*scale)
                linearization.append(dict(kind=kind,scale=scale,start=start,samples=len(x),
                    absolute_error=summarize((actual-predicted).norm(dim=1)),
                    actual_change=summarize(actual.norm(dim=1)),
                    relative_error=summarize((actual-predicted).norm(dim=1)/(actual.norm(dim=1)+1e-8))))
        # Same parameters and same phase realization in both orders. Central
        # crop is a boundary sensitivity diagnostic, not a pure decomposition.
        lx=physical(x,a,'linear');tx=physical(x,a,'temporal',noise)
        ltx=physical(tx,a,'linear');tlx=physical(lx,a,'temporal',noise)
        zl=identity(model,lx)[1];zt=identity(model,tx)[1]
        zlt=identity(model,ltx)[1];ztl=identity(model,tlx)[1]
        iq_delta=(ltx-tlx).square().sum(1)
        interactions.append(dict(start=start,samples=len(x),
            iq_order_relative=summarize((ltx-tlx).flatten(1).norm(dim=1)/(x.flatten(1).norm(dim=1)+1e-8)),
            central_order_rms=summarize(iq_delta[:,32:224].mean(1).sqrt()),
            edge_order_rms=summarize(torch.cat((iq_delta[:,:32],iq_delta[:,224:]),1).mean(1).sqrt()),
            feature_order=summarize((zlt-ztl).norm(dim=1)),
            feature_mixed_curvature=summarize((zlt-zl-zt+z).norm(dim=1))))
        features.append(z.cpu());logits_all.append(logits.cpu());jacobians.append(j.cpu())
        if start == 0 or start+batch_size >= len(cache['x']) or (start//batch_size+1)%10==0:
            print('GEOMETRY_SOURCE_PROGRESS',seed,min(start+batch_size,len(cache['x'])),flush=True)
    return dict(parent_feature=torch.cat(features),parent_logits=torch.cat(logits_all),
        jacobian=torch.cat(jacobians), convergence=convergence,linearization=linearization,
        interactions=interactions,contract=dict(method='central finite difference',normalized_step=EPS,
            convergence_scope='first128 L_s only, epsilon versus epsilon/2',
            dimensions=['L tap1.real','L tap1.imag','L tap3.real','L tap3.imag',
                        'T phase/.25rad','T CFO/3120Hz','T signed carrier/.025rad','T delay/.25sample'],
            signed_noise_tangent='negative coefficient is equivalent to sign reversal of the same fixed zero-mean realization',
            omega='I8/6 fixed zero-mean local scale prior; not an estimated true RX distribution or the learned policy covariance'))


def scatter(z,y,rx,day):
    z=z.double(); keys=sorted(set(zip(y.tolist(),rx.tolist(),day.tolist())))
    means=[];labels=[];residuals=[]
    for ty,r,dy in keys:
        part=z[(y==ty)&(rx==r)&(day==dy)]
        mean=part.mean(0);means.append(mean);labels.append(ty)
        centered=part-mean;residuals.append(centered.T@centered/len(part))
    means=torch.stack(means);labels=torch.tensor(labels)
    mus=torch.stack([means[labels==ty].mean(0) for ty in sorted(set(labels.tolist()))])
    centered=mus-mus.mean(0)
    return centered.T@centered/len(mus),torch.stack(residuals).mean(0)


def nuisance(j,selection=None):
    if selection is not None:j=j[selection]
    j=j.double()
    return [OMEGA*torch.einsum('ndk,nek->de',j[:,:,start:start+4],j[:,:,start:start+4])/len(j) for start in (0,4)]


def spectrum(stx,cn):
    eigen,basis=torch.linalg.eigh((cn+cn.T)/2)
    if eigen.min()<=0:raise ValueError('Nuisance metric must be positive definite')
    inv=(basis*eigen.rsqrt()[None])@basis.T
    values,u=torch.linalg.eigh(inv@stx@inv)
    order=torch.argsort(values,descending=True);values=values[order];v=inv@u[:,order]
    rank=int((values>max(float(values.max())*1e-7,1e-10)).sum())
    if rank>5:raise ValueError('Six TX mean scatter rank exceeds five')
    residual=stx@v-cn@v*values[None]
    euclidean=torch.linalg.qr(v[:,:rank]).Q if rank else v[:,:0]
    return dict(eigenvalues=values.tolist(),rank=rank,max_eigen_residual=float(residual.abs().max()),
                metric_min_eigenvalue=float(eigen.min())),v[:,:rank],euclidean


def ridge_readout(fit_x,fit_y,test_x,test_y):
    mean=fit_x.double().mean(0);scale=fit_x.double().std(0,unbiased=False).clamp_min(1e-5)
    a=(fit_x.double()-mean)/scale;b=(test_x.double()-mean)/scale
    a=torch.cat((a,torch.ones(len(a),1)),1);b=torch.cat((b,torch.ones(len(b),1)),1)
    weight=torch.linalg.solve(a.T@a/len(a)+torch.eye(a.shape[1],dtype=a.dtype),
                              a.T@F.one_hot(fit_y,6).double()/len(a))
    pred=(b@weight).argmax(1)
    return dict(count=len(test_y),correct=int((pred==test_y).sum()),accuracy=float((pred==test_y).double().mean()))


def analyze(cache,response,r_cov,physical,head_weight,validation=None,physical_groups=None):
    z=response['parent_feature'];j=response['jacobian'];y=cache['y'];rx=cache['rx'];day=cache['day']
    stx,cres=scatter(z,y,rx,day);cl,ct=nuisance(j)
    cr=r_cov.double();cn=cl+ct+cr+cres+RIDGE*torch.eye(160,dtype=torch.double)
    result,v,u=spectrum(stx,cn)
    weight=F.normalize(head_weight.double(),dim=1,eps=1e-4);weight-=weight.mean(0)
    _,singular,right=torch.linalg.svd(weight,full_matrices=False)
    hrank=int((singular>singular.max()*1e-8).sum());h=right[:hrank].T
    result.update(covariance_traces={name:float(a.trace()) for name,a in [('TX',stx),('L',cl),('T',ct),('R',cr),('packet_residual',cres)]},
        head_rank=hrank,head_top_direction_overlap=torch.linalg.svdvals(h.T@u).square().tolist(),
        nuisance_overlap={},generalized_transfer=[],readouts=[])
    for name,cov in [('L',cl),('T',ct),('R',cr)]:
        values,basis=torch.linalg.eigh(cov)
        order=torch.argsort(values,descending=True);values=values[order].clamp_min(0);basis=basis[:,order]
        k=min(5,int((values>max(float(values.max())*1e-7,1e-12)).sum()))
        result['nuisance_overlap'][name]=dict(top_rank=k,tx_overlap_singular_squared=torch.linalg.svdvals(u.T@basis[:,:k]).square().tolist(),eigenvalues=values.tolist())
    feature_sets=[('joint160',z),('physical',physical),('joint160_plus_physical',torch.cat((z,physical),1))]
    for name,indices in (physical_groups or {}).items():
        feature_sets.extend([(name,physical[:,indices]),('joint160_plus_'+name,torch.cat((z,physical[:,indices]),1))])
    # Cross-condition diagnostic fits its own source complement. Does not mean
    # unseen-TX registration: the identity parent was trained on every TX.
    for name,condition in [('RX',rx),('day',day)]:
        for held in sorted(set(condition.tolist())):
            train=condition!=held;test=~train
            ts,tr=scatter(z[train],y[train],rx[train],day[train]);tl,tt=nuisance(j,train)
            _,tv,_=spectrum(ts,tl+tt+tr+RIDGE*torch.eye(160,dtype=torch.double))
            hs,hr=scatter(z[test],y[test],rx[test],day[test]);hl,ht=nuisance(j,test)
            hn=hl+ht+hr+RIDGE*torch.eye(160,dtype=torch.double)
            ratios=torch.diag(tv.T@hs@tv)/torch.diag(tv.T@hn@tv).clamp_min(1e-12)
            result['generalized_transfer'].append(dict(condition=name,heldout=int(held),ratios=ratios.tolist(),
                note='R covariance omitted in both fit and heldout transfer diagnostic to avoid RX/day information in learned R'))
            for feature_name,x in feature_sets:
                result['readouts'].append(dict(condition=name,heldout=int(held),features=feature_name,
                    **ridge_readout(x[train],y[train],x[test],y[test])))
    if validation is not None:
        validation_sets=[('joint160',z,validation['feature']),('physical',physical,validation['physical']),
                         ('joint160_plus_physical',torch.cat((z,physical),1),torch.cat((validation['feature'],validation['physical']),1))]
        for name,indices in (physical_groups or {}).items():
            validation_sets.extend([(name,physical[:,indices],validation['physical'][:,indices]),
                ('joint160_plus_'+name,torch.cat((z,physical[:,indices]),1),torch.cat((validation['feature'],validation['physical'][:,indices]),1))])
        for name,fx,vx in validation_sets:
            result['readouts'].append(dict(condition='independent_source_V',features=name,
                **ridge_readout(fx,y,vx,validation['y'])))
            for condition in ('rx','day'):
                for held in sorted(set(validation[condition].tolist())):
                    mask=validation[condition]==held
                    result['readouts'].append(dict(condition='independent_source_V_'+condition,heldout=int(held),features=name,
                        **ridge_readout(fx,y,vx[mask],validation['y'][mask])))
    result['contract']=dict(feature_dimension=160,TX_mean_rank_bound=5,hard_projection_training=False,
        covariance_weights=dict(L=1.,T=1.,R=1.,packet_residual=1.),ridge=RIDGE,
        independent_readout_regularization=1.,readout_fit='L_s only, no hyperparameter search',
        physical_interpretation='bounded local response and empirical received-condition geometry, not identified TX/RX hardware',
        unknown_TX_evaluation=False)
    return result
