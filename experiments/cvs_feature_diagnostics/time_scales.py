"""Read-only TIME branch diagnostics; default trained classifier, no fitting."""
import torch
from torch.nn import functional as F


def equivalent_kernel(block):
    if len(block.branches) != 3 or block.kind != 'time':
        raise ValueError('Expected historical three-branch time block')
    if tuple(block._settings) not in (((3,1),(5,1),(5,2)), ((3,1),(5,1),(5,1))):
        raise ValueError('Unregistered temporal support')
    ci=block.skip.in_channels;co=block.skip.out_channels
    if block.fuse.weight.shape != (co,3*ci,1) or block.skip.weight.shape!=(co,ci,1):
        raise ValueError('Unexpected temporal fusion dimensions')
    if getattr(block,'branch_activation','none') != 'none':
        raise ValueError('Nonlinear branches cannot be folded')
    kernel=torch.zeros(co,ci,9,dtype=torch.float64)
    for j,(branch,(width,dilation)) in enumerate(zip(block.branches,block._settings)):
        if branch.groups!=ci or branch.weight.shape!=(ci,1,width) or branch.bias is not None:
            raise ValueError('Expected unbiased depthwise branches')
        dw=branch.weight.detach().cpu().double()[:,0,:]
        pw=block.fuse.weight.detach().cpu().double()[:,j*ci:(j+1)*ci,0]
        for index in range(width):kernel[:,:,4+(index-width//2)*dilation]+=pw*dw[:,index][None,:]
    kernel[:,:,4]+=block.skip.weight.detach().cpu().double()[:,:,0]
    if not torch.isfinite(kernel).all():raise ValueError('Nonfinite actual weights')
    return kernel


def kernel_report(block):
    kernel=equivalent_kernel(block)
    singular=torch.linalg.svdvals(kernel.permute(1,0,2))
    rank=(singular>singular[:,:1]*1e-10).sum(1)
    if int(rank.max())>4:raise ValueError('Constrained basis rank unexpectedly exceeded four')
    energy=kernel.square().sum((0,1));total=energy.sum().clamp_min(1e-30)
    return dict(shape=list(kernel.shape),offsets=list(range(-4,5)),offset_energy=energy.tolist(),
                offset_energy_fraction=(energy/total).tolist(),per_input_singular_values=singular.tolist(),
                per_input_kernel_rank=rank.tolist(),maximum_kernel_rank=int(rank.max()),
                interpretation='Kernel channel-rank and offset energy; not joint160 feature rank or measured effective receptive field')


def sample_labeled(data,per_cell=8):
    if len(data['ids'])!=6300 or per_cell!=8:raise ValueError('Fixed labeled source diagnostic differs')
    selected=[]
    for y in range(6):
        for rx in (1,3,4,6,8):
            for day in (1,2,3):
                candidates=torch.where((data['y']==y)&(data['rx']==rx)&(data['day']==day))[0].tolist()
                candidates.sort(key=lambda i:data['ids'][i])
                if len(candidates)!=70:raise ValueError('Expected 70 labeled physical packets per TX/RX/day cell')
                selected.extend(candidates[:per_cell])
    if len(selected)!=720 or len(set(selected))!=720:raise ValueError('Source sample coverage differs')
    ix=torch.tensor(selected)
    return {k:([v[i] for i in selected] if k=='ids' else v[ix]) for k,v in data.items()}


def margin(logits,y):
    true=logits.gather(1,y[:,None]).squeeze(1)
    other=logits.clone();other.scatter_(1,y[:,None],-torch.inf)
    return true-other.amax(1)


@torch.no_grad()
def scale_removal(model,data):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Diagnostic model must be frozen')
    backbone=model.id_backbone.encoder.core.id_backbone
    head=backbone.cls_head
    if type(head).__name__!='ResidualPhysicalHead' or head.weight.shape!=(6,160) or head.scale!=30:
        raise ValueError('Default classifier differs')
    before={k:v.clone() for k,v in model.state_dict().items()}
    def collect():
        z=[];logits=[]
        for start in range(0,len(data['x']),64):
            output=model.forward_identity_only(data['x'][start:start+64])
            z.append(output['id_feat_joint']);logits.append(output['tx_logits'])
        return torch.cat(z),torch.cat(logits)
    full_z,full_logits=collect();y=data['y'].long();full_margin=margin(full_logits,y);full_ok=full_logits.argmax(1)==y
    records=[]
    for layer in ('t1','t2','t3'):
        block=getattr(backbone,layer);ci=block.skip.in_channels
        for branch_index in range(3):
            contributions=[]
            def contribution(module,args,output):
                value=args[0][:,branch_index*ci:(branch_index+1)*ci]
                fused=F.conv1d(value,module.weight[:,branch_index*ci:(branch_index+1)*ci])
                contributions.append((fused.square().mean((1,2)).sqrt().cpu(),output.square().mean((1,2)).sqrt().cpu()))
            observer=block.fuse.register_forward_hook(contribution)
            try:collect()
            finally:observer.remove()
            def remove(module,args,output):return torch.zeros_like(output)
            hook=block.branches[branch_index].register_forward_hook(remove)
            try:z,logits=collect()
            finally:hook.remove()
            difference=(full_z-z).norm(dim=1);cos=F.cosine_similarity(full_z,z,dim=1,eps=1e-4)
            m=margin(logits,y);ok=logits.argmax(1)==y
            rms=torch.cat([v[0] for v in contributions]);fused_rms=torch.cat([v[1] for v in contributions])
            for dimension,values in [('overall',[None]),('receiver',[1,3,4,6,8]),('day',[1,2,3])]:
                for value in values:
                    mask=torch.ones(len(y),dtype=torch.bool) if value is None else data['rx' if dimension=='receiver' else 'day']==value
                    records.append(dict(layer=layer,branch=branch_index,kernel_dilation=list(block._settings[branch_index]),dimension=dimension,stratum=value,count=int(mask.sum()),
                        full_accuracy=float(full_ok[mask].float().mean()),removed_accuracy=float(ok[mask].float().mean()),
                        corrected_errors=int((ok[mask]&~full_ok[mask]).sum()),introduced_errors=int((~ok[mask]&full_ok[mask]).sum()),
                        default_true_margin_change_mean=float((m-full_margin)[mask].mean()),joint160_change_L2_mean=float(difference[mask].mean()),
                        joint160_cosine_mean=float(cos[mask].mean()),post_fusion_contribution_RMS_mean=float(rms[mask].mean()),
                        contribution_to_fused_RMS_mean=float((rms/fused_rms.clamp_min(1e-8))[mask].mean())))
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Read-only diagnostic changed model state')
    return dict(records=records,source_packets=720,feature_updates=0,head_updates=0,U_opened=False,target_IQ_read=False,truth_read=False,
                limitation='Inference-time branch removal from jointly trained weights; not a separately trained ablation or causal attribution')


def checks():
    from types import SimpleNamespace
    from torch import nn
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.kind='time';self._settings=((3,1),(5,1),(5,2));self.skip=nn.Conv1d(3,4,1,bias=False)
            self.fuse=nn.Conv1d(9,4,1,bias=False);self.branches=nn.ModuleList([nn.Conv1d(3,3,k,padding=k//2*d,dilation=d,groups=3,bias=False) for k,d in self._settings])
    torch.set_num_threads(2);torch.manual_seed(173);b=Block().double();x=torch.randn(2,3,23,dtype=torch.float64,requires_grad=True)
    direct=b.fuse(torch.cat([v(x) for v in b.branches],1))+b.skip(x);fold=F.conv1d(x,equivalent_kernel(b),padding=4)
    error=float((direct-fold).abs().max());g1=torch.autograd.grad(direct.square().sum(),x,retain_graph=True)[0];g2=torch.autograd.grad(fold.square().sum(),x)[0]
    grad_error=float((g1-g2).abs().max());assert error<1e-12 and grad_error<1e-12
    r=kernel_report(b);assert r['offset_energy'][1]==0 and r['offset_energy'][7]==0
    negatives=0
    for mutation in ('activation','support','kind'):
        import copy
        q=copy.deepcopy(b)
        if mutation=='activation':q.branch_activation='silu'
        elif mutation=='support':q._settings=((3,1),(5,1),(7,1))
        else:q.kind='PA'
        try:equivalent_kernel(q)
        except ValueError:negatives+=1
    assert negatives==3
    return dict(status='PASS',synthetic_only=True,linear_pre_norm_output_error=error,input_gradient_error=grad_error,negative_checks=negatives,head_training=False,real_checkpoint_evidence=False)

if __name__=='__main__':
    import json
    print(json.dumps(checks()))
