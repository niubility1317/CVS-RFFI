"""Closest U direction satisfying the full and two component L half-spaces."""
from itertools import combinations
import math
import torch
from .u_gradprotect_objective import parameter_group

GROUPS=('t2','cls_head','other')
ANCHORS=('L_full','L_clean','L_As')


def cone_coefficients(gram,dots):
    """Solve at most three homogeneous inequalities by exhaustive active sets."""
    gram=torch.as_tensor(gram,dtype=torch.float64,device='cpu')
    dots=torch.as_tensor(dots,dtype=torch.float64,device='cpu')
    count=len(dots)
    if (not 1<=count<=3 or gram.shape!=(count,count)
            or not bool(torch.isfinite(gram).all()) or not bool(torch.isfinite(dots).all())):
        raise ValueError('One to three finite source anchor constraints required')
    if not torch.allclose(gram,gram.T,rtol=1e-10,atol=1e-12):raise ValueError('Symmetric source anchor Gram matrix required')
    if float(torch.linalg.eigvalsh(gram).min()) < -1e-10:raise ValueError('Positive semidefinite source anchor Gram matrix required')
    scale=max(float(dots.abs().max()),1e-12);tol=1e-8*scale
    best=None
    for size in range(count+1):
        for selected in combinations(range(count),size):
            multipliers=torch.zeros(count,dtype=torch.float64)
            if size:
                indices=torch.tensor(selected,dtype=torch.long)
                active=gram[indices][:,indices]
                multipliers[indices]=torch.linalg.pinv(active,rtol=1e-10,hermitian=True)@(-dots[indices])
                if bool((multipliers[indices] < -tol).any()):continue
                if bool(((active@multipliers[indices]+dots[indices]).abs()>tol).any()):continue
            residual=dots+gram@multipliers
            if bool((residual < -tol).any()):continue
            cost=float(multipliers@(gram@multipliers))
            if best is None or cost<best[0]:best=(cost,multipliers,residual,selected)
    if best is None:raise ValueError('No numerically valid source anchor active set;stop this diagnostic')
    _,multipliers,residual,selected=best
    return dict(multipliers=multipliers.tolist(),constraint_dots=residual.tolist(),active_set=list(selected),
                correction_squared_norm=best[0],KKT_tolerance=tol)


def basis_candidates(basis,coordinates):
    """Avoid squared conditioning:enumerate active constraints in a thin QR basis."""
    basis=torch.as_tensor(basis,dtype=torch.float64,device='cpu')
    coordinates=torch.as_tensor(coordinates,dtype=torch.float64,device='cpu')
    count,width=basis.shape
    if not 1<=count<=3 or not 1<=width<=3 or coordinates.shape!=(width,):
        raise ValueError('At most three finite source anchor coordinates required')
    if not bool(torch.isfinite(basis).all() and torch.isfinite(coordinates).all()):
        raise ValueError('Finite source anchor basis required')
    dots=basis@coordinates;tol=1e-8*max(float(dots.abs().max()),1e-12);solutions=[]
    for size in range(count+1):
        for selected in combinations(range(count),size):
            multipliers=torch.zeros(count,dtype=torch.float64);projected=coordinates.clone()
            if size:
                indices=torch.tensor(selected,dtype=torch.long);active=basis[indices]
                correction=torch.linalg.pinv(active,rtol=1e-10)@(-dots[indices])
                projected=coordinates+correction
                multipliers[indices]=torch.linalg.pinv(active.T,rtol=1e-10)@correction
                if bool((multipliers[indices]<-tol).any()):continue
                if bool((active@projected).abs().gt(tol).any()):continue
                if not torch.allclose(active.T@multipliers[indices],correction,rtol=1e-8,atol=tol):continue
            residual=basis@projected
            if bool((residual<-tol).any()):continue
            solutions.append(dict(projected_coordinates=projected.tolist(),active_set=list(selected),
                correction_squared_norm=float((projected-coordinates).square().sum()),
                multipliers=multipliers.tolist(),constraint_dots=residual.tolist(),KKT_tolerance=tol))
    if not solutions:raise ValueError('No valid source anchor active set;stop the affected step')
    return sorted(solutions,key=lambda v:v['correction_squared_norm'])


@torch.no_grad()
def branch_projected_gradients(named,labeled,unlabeled,clean,physical,*,max_u_ratio=2.):
    """Keep complete L unchanged; project only U onto all three source constraints."""
    if not isinstance(max_u_ratio,(int,float)) or not math.isfinite(max_u_ratio) or max_u_ratio<=0:
        raise ValueError('Finite positive U/L norm cap required')
    if not len(named)==len(labeled)==len(unlabeled)==len(clean)==len(physical) or not named:
        raise ValueError('Matching complete and component source gradient vectors required')
    for values in (labeled,unlabeled,clean,physical):
        for (_,parameter),gradient in zip(named,values):
            if gradient is not None and (gradient.shape!=parameter.shape or gradient.device!=parameter.device
                    or gradient.dtype!=parameter.dtype or not bool(torch.isfinite(gradient).all())):
                raise ValueError('Finite aligned source parameter derivatives required')
    result=[None]*len(named);details={}
    for group in GROUPS:
        indices=[i for i,(name,_) in enumerate(named) if parameter_group(name)==group
                 and (labeled[i] is not None or unlabeled[i] is not None)]
        if not indices:
            details[group]=dict(L_norm=0.,U_norm=0.,applied_U_norm=0.,dot_before=0.,dot_after=0.,
                conflict=False,capped=False,tiny_L=True,anchor_cosines_before={a:None for a in ANCHORS},
                anchor_cosines_after={a:None for a in ANCHORS},constraint_dots_after={a:0. for a in ANCHORS},
                active_set=[],cone_correction_norm=0.,numerical_zero_projection=True,
                near_zero_component_anchors_omitted=list(ANCHORS))
            continue
        def vector(values):
            return [torch.zeros_like(named[i][1],dtype=torch.float64) if values[i] is None
                    else values[i].detach().double() for i in indices]
        gl,gu=vector(labeled),vector(unlabeled)
        anchor_vectors=[gl,vector(clean),vector(physical)]
        def dot(a,b):return sum(float((left*right).sum()) for left,right in zip(a,b))
        norms=[math.sqrt(dot(a,a)) for a in anchor_vectors]
        lnorm,unorm=norms[0],math.sqrt(dot(gu,gu));tiny=lnorm*lnorm<=1e-12
        valid=[i for i,norm in enumerate(norms) if norm*norm>1e-12]
        raw_dots=[dot(a,gu) for a in anchor_vectors]
        solution=dict(active_set=[],correction_squared_norm=0.)
        if tiny:
            candidates=[solution];q=perpendicular=None
        else:
            columns=[torch.cat([(value/norms[i]).reshape(-1) for value in anchor_vectors[i]]) for i in valid]
            matrix=torch.stack(columns,dim=1);flat=torch.cat([v.reshape(-1) for v in gu])
            q,r=torch.linalg.qr(matrix,mode='reduced');coordinates=q.T@flat
            perpendicular=flat-q@coordinates
            candidates=basis_candidates(r.T.cpu(),coordinates.cpu())
        chosen=None
        for candidate in candidates:
            if tiny:projected=[torch.zeros_like(v) for v in gu]
            else:
                projected_flat=perpendicular+q@torch.tensor(candidate['projected_coordinates'],dtype=q.dtype,device=q.device)
                projected=[];offset=0
                for value in gu:
                    projected.append(projected_flat[offset:offset+value.numel()].view_as(value));offset+=value.numel()
            pnorm=math.sqrt(dot(projected,projected));numerical_zero=pnorm<=1e-12*max(unorm,lnorm,1e-12)
            if numerical_zero:projected=[torch.zeros_like(v) for v in gu];pnorm=0.
            scale=min(1.,max_u_ratio*lnorm/pnorm) if pnorm else 1.
            applied=[(value*scale).to(named[i][1].dtype) for i,value in zip(indices,projected)]
            actual=[value.double() for value in applied];anorm=math.sqrt(dot(actual,actual))
            after=[dot(a,actual) for a in anchor_vectors]
            # A coarse Gram residual is insufficient near a zero projection.
            # Choose the closest active set that passes the actual cast constraints.
            if any(after[i]<-1e-6*max(norms[i]*anorm,1e-12) for i in valid):continue
            if anorm>max_u_ratio*lnorm*(1+1e-6)+1e-12:continue
            chosen=candidate;break
        if chosen is None:raise ValueError('No source active set passes the actual cast L constraints and norm cap')
        solution=chosen
        for index,lvalue,uvalue in zip(indices,gl,applied):result[index]=lvalue.to(named[index][1].dtype)+uvalue
        details[group]=dict(L_norm=lnorm,U_norm=unorm,applied_U_norm=anorm,
            dot_before=raw_dots[0],dot_after=after[0],conflict=raw_dots[0]<0.,capped=scale<1.,tiny_L=tiny,
            anchor_cosines_before={name:raw/(norm*unorm) if norm and unorm else None
                for name,norm,raw in zip(ANCHORS,norms,raw_dots)},
            anchor_cosines_after={name:product/(norm*anorm) if norm and anorm else None
                for name,norm,product in zip(ANCHORS,norms,after)},
            constraint_dots_after=dict(zip(ANCHORS,after)),active_set=[ANCHORS[valid[i]] for i in solution['active_set']],
            cone_correction_norm=math.sqrt(max(0.,solution['correction_squared_norm'])),
            solver='thin_QR_and_active_set_SVD',
            numerical_zero_projection=numerical_zero,
            near_zero_component_anchors_omitted=[ANCHORS[i] for i in range(3) if i not in valid])
    return tuple(result),details
