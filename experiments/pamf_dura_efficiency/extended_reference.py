"""Source-stable feature coordinates outside the original frozen head span."""
import numpy as np


def extended_basis(cell_means,head_basis):
    if (cell_means.shape!=(5,3,6,160) or head_basis.shape!=(160,5)
        or not np.isfinite(cell_means).all() or not np.isfinite(head_basis).all()
        or not np.allclose(head_basis.T@head_basis,np.eye(5),atol=2e-6,rtol=2e-6)):
        raise ValueError('Original fiveRX/three-day/six-soft-coordinate means and rank5 head basis required')
    # Every RX/day has equal weight. Centering the six soft means removes a
    # shared source offset; only five independent between-coordinate axes remain.
    centers=np.asarray(cell_means,dtype=np.float64).mean((0,1));centered=centers-centers.mean(0)
    q=np.asarray(head_basis,dtype=np.float64)
    residual=centered-(centered@q)@np.linalg.solve(q.T@q,q.T)
    _,singular,vectors=np.linalg.svd(residual,full_matrices=False)
    if singular[4]<=1e-6 or singular[5]>singular[0]*1e-6:
        raise ValueError('Five defined source-stable directions outside the frozen head required')
    auxiliary=vectors[:5].T
    signs=np.where(auxiliary[np.argmax(abs(auxiliary),axis=0),np.arange(5)]<0,-1.,1.)
    auxiliary*=signs
    result=np.column_stack([head_basis,auxiliary]).astype(np.float32)
    if np.max(abs(result.T@result-np.eye(10)))>3e-6:raise ValueError('Aligned orthonormal source10 basis required')
    if not np.array_equal(result[:,:5],head_basis):raise ValueError('Original cached head coordinates must remain unchanged')
    return result,dict(dimensions=10,original_head_dimensions=5,source_stable_null_dimensions=5,
        fit_cells=90,fit_RX_day_equal=True,center_singular_values=singular.tolist(),
        auxiliary_head_overlap=float(np.max(abs(result[:,:5].T@result[:,5:]))))


def auxiliary_decision(views):
    if set(views)!={'1','20'}:raise ValueError('Both fixed source views required')
    rows=[]
    for view,r in views.items():
        if r['count']!=512 or set(r['RX_mean_cosine'])!=set(map(str,[1,3,4,6,8])):
            raise ValueError('Complete512/fiveRX auxiliary geometry required')
        passed=(r['defined']==512 and r['defined_IQ']==512 and r['mean_cosine']>=.5
            and r['mean_cosine']>=r['mean_IQ_cosine']+.1 and r['nonnegative']>=384
            and all(value is not None and value>=0 for value in r['RX_mean_cosine'].values()))
        rows.append(dict(view=view,passed=bool(passed)))
    return dict(eligible=all(r['passed'] for r in rows),views=rows,
        scope='Additional five-coordinate direction validity;IQ gradient cap is checked on combined guide only')
