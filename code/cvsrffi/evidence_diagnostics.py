"""Source-only falsification diagnostics; never called to rank target candidates."""
import torch
import torch.nn.functional as F


def source_condition_probe(train_features,train_labels,val_features,val_labels,*,train_role="L_s",val_role="V",ridge=1.0):
    """Same ridge classifier for TX or RX leakage probes of e/q/missing patterns."""
    if train_role!="L_s" or val_role!="V": raise ValueError("condition probes fit L_s and evaluate only source V")
    if ridge<=0: raise ValueError("positive ridge required")
    classes=torch.unique(train_labels,sorted=True)
    if not torch.isin(val_labels,classes).all(): raise ValueError("probe class not present in training")
    center=train_features.mean(0)
    scale=train_features.std(0,unbiased=False).clamp_min(1e-5)
    def design(x):
        z=(x-center)/scale
        return torch.cat((z,torch.ones_like(z[:,:1])),1)
    x=design(train_features)
    y=F.one_hot(torch.searchsorted(classes,train_labels),len(classes)).to(x)
    weight=torch.linalg.solve(x.T@x+ridge*torch.eye(x.shape[-1],device=x.device,dtype=x.dtype),x.T@y)
    pred=classes[(design(val_features)@weight).argmax(-1)]
    return {"accuracy":float((pred==val_labels).float().mean()),"classes":len(classes),
            "majority_reference":float(torch.bincount(torch.searchsorted(classes,val_labels)).max()/len(val_labels)),
            "interpretation":"acquisition association diagnostic, not physical identity disentanglement"}


def score_change_diagnostics(base_scores,new_scores,truth,*,role):
    if role not in {"V","sealed_final_scoring"}: raise ValueError("labels only for source V or independent final scoring")
    if base_scores.shape!=new_scores.shape: raise ValueError("matched score shape required")
    def margin(scores):
        top=scores.topk(min(2,scores.shape[-1]),dim=-1).values
        return top[:,0]-top[:,1] if top.shape[-1]==2 else torch.zeros_like(top[:,0])
    a,b=base_scores.argmax(-1),new_scores.argmax(-1)
    return {"score_delta_l2":(new_scores-base_scores).norm(dim=-1),
            "margin_delta":margin(new_scores)-margin(base_scores),
            "prediction_changed":a!=b,"rescue":(a!=truth)&(b==truth),"harm":(a==truth)&(b!=truth)}


def covariance_contributions(result):
    """Keep measurement, legitimate response, state and support uncertainty separate."""
    out={"measurement":result["observation_variance"].sum(-1),
         "response":result["response_covariance"].diagonal().sum(),
         "state":result["state_covariance"].diagonal(dim1=-2,dim2=-1).sum(-1)}
    if "support_covariance" in result:
        out["support"]=result["support_covariance"].diagonal(dim1=-2,dim2=-1).sum(-1)
    return out


class FixedReadoutBaseline:
    """Uniform prototype cosine / shared-diagonal Gaussian control on identical z.

    Fitting role is explicit: L_s for source head comparison, support for target
    registration. This object must not be used to transport empirical source stats.
    """
    def __init__(self,z,labels,*,role,kind="cosine",ridge=.01):
        if role not in {"L_s","support"}: raise ValueError("readout fit cannot consume V/query")
        if kind not in {"cosine","diagonal_gaussian","linear_ridge"} or ridge<=0: raise ValueError("invalid baseline")
        self.classes=torch.unique(labels,sorted=True); self.kind=kind
        self.means=torch.stack([z[labels==c].mean(0) for c in self.classes]).detach()
        target=torch.searchsorted(self.classes,labels)
        self.var=((z-self.means[target]).square().mean(0)+ridge).detach()
        x=torch.cat((z,torch.ones_like(z[:,:1])),1)
        self.weight=torch.linalg.solve(x.T@x+ridge*torch.eye(x.shape[-1],device=x.device),
                    x.T@F.one_hot(target,len(self.classes)).to(z)).detach()
    def predict(self,z):
        if self.kind=="cosine": return F.normalize(z,dim=-1)@F.normalize(self.means,dim=-1).T
        if self.kind=="diagonal_gaussian": return -.5*((z[:,None]-self.means[None]).square()/self.var).sum(-1)
        return torch.cat((z,torch.ones_like(z[:,:1])),1)@self.weight
