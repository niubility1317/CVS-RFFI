"""Route source H/G identity tasks and cross-RX relation gradients separately."""
import torch
from .feature_task_loss import task_loss
from .restoration_bridge import relation_loss

METHODS=('HJoint','HAnchor','GJoint','BothJoint')


def terms(h,g,target,accepted,receivers,method):
    if method not in METHODS:raise ValueError('Fixed four task/relation routing controls required')
    hz,gz=h['z_id'],g['z_id']
    task=[]
    if method in ('HJoint','HAnchor','BothJoint'):
        task.append(task_loss(hz,h['tx_logits'],target,accepted,True))
    if method in ('GJoint','BothJoint'):
        task.append(task_loss(gz,g['tx_logits'],target,accepted,True))
    ce=torch.stack(task).mean()
    relation,detail=relation_loss(hz,gz.detach() if method=='HAnchor' else gz,target,accepted,receivers,'bridge')
    detail.update(CE_H_accepted=int(accepted.sum()) if method in ('HJoint','HAnchor','BothJoint') else 0,
        CE_G_accepted=int(accepted.sum()) if method in ('GJoint','BothJoint') else 0,
        CE_loss=ce.detach(),relation_loss=relation.detach(),restored_feature_detached_in_CL=method=='HAnchor')
    return ce+relation,detail


def extra(student,views,domain,target,accepted,receivers,method,check=False):
    if len(views)!=2:raise ValueError('Exactly H and its paired G student forwards required')
    outputs=[student.forward_identity_only(v,domain_labels=domain) for v in views]
    value,detail=terms(*outputs,target,accepted,receivers,method)
    if check:
        grads=torch.autograd.grad(value,[o['z_id'] for o in outputs],retain_graph=True,allow_unused=True)
        norms=[0. if v is None else float(v.detach().norm()) for v in grads]
        if norms[0]<=0 or (norms[1]!=0 if method=='HAnchor' else norms[1]<=0):
            raise ValueError('Actual H/G feature-gradient route differs')
        detail.update(H_feature_gradient_norm=norms[0],G_feature_gradient_norm=norms[1])
    return value,detail
