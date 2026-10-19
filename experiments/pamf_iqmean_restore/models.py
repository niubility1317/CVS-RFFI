import json
import torch
from torch import nn
from . import design as d
from experiments.pamf_identity_priority.models import load_checkpoint as load_base
from experiments.pamf_layerwise_refine.models import load_checkpoint as load_partner


class FixedIQMean(nn.Module):
    def __init__(self,base,partner):
        super().__init__();self.base=base;self.partner=partner;self.eval().requires_grad_(False)

    def train(self,mode=True):
        if mode:raise ValueError('Composed restorer is inference-only')
        return super().train(False)

    def forward(self,received):
        return .5*self.base(received)+.5*self.partner(received)


def validate_members(run_root):
    records=[]
    for run,commit,arms,epoch,updates,l,u,coarse,corr in (
        (d.BASE_RUN,d.BASE_COMMIT,('posterior03',),10,2470,63000,567000,30,30),
        (d.PARTNER_RUN,d.PARTNER_COMMIT,('final_guidance','layerwise_guidance'),30,7410,189000,1701000,50,30)):
        root=run_root/run;launch=json.loads((root/'launch.json').read_text())
        done=json.loads((root/'complete.json').read_text());source=json.loads((root/'source_complete.json').read_text())
        if (launch.get('commit')!=commit or done.get('status')!='ANALYZED' or not done.get('source_only') or
            source.get('status')!='ANALYZED' or not source.get('source_only') or source.get('target_reads')!=0):
            raise ValueError('Completed source-only member provenance required')
        for arm in arms:
            t=json.loads((root/arm/'training_complete.json').read_text())
            expected=dict(epochs=epoch,updates=updates,coarse_updates=updates,L_physical_exposures=l,
                U_physical_exposures=u,IQ_examples=2*(l+u),coarse_epochs=coarse,total_correction_epochs=corr,
                F_seeds=list(d.F_SEEDS),F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0)
            if any(t.get(k)!=v for k,v in expected.items()):raise ValueError('Member budget, data roles or frozen-head contract differ')
            records.append(dict(run=run,commit=commit,arm=arm,epochs=epoch,updates=updates,source_only=True))
    return records


def load_model(run_root,arm,device):
    config=d.configuration(arm);validate_members(run_root)
    base,old=load_base(run_root/d.BASE_RUN/'posterior03/final.pt','posterior03',device)
    partner,new=load_partner(run_root/d.PARTNER_RUN/d.PARTNERS[arm]/'final.pt',d.PARTNERS[arm],device)
    if (old['epoch'],old['updates'],new['epoch'],new['updates'])!=(10,2470,30,7410):
        raise ValueError('Fixed final member checkpoints required')
    model=FixedIQMean(base,partner).to(device)
    if sum(p.numel() for p in model.parameters())!=config['inference_parameters_expected']:
        raise ValueError('Expected two original restoration models required')
    return model
