"""Controlled synthetic MoPC stress test; no real data or predictions."""
import json
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_cil_bridge import from_protonet,make_anchor
from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import RegistrationConfig,register_new_classes


class Encoder(nn.Module):
    def __init__(self):super().__init__();self.layer=nn.Linear(2,4)
    def forward(self,x):return {'features':self.layer(x)}


def trial(seed,cap):
    torch.manual_seed(seed);model=from_protonet(Encoder().eval(),torch.randn(2,4)*.01)
    old_x=torch.randn(8,2)*100;old_y=torch.tensor([0]*4+[1]*4)
    anchor=make_anchor(model,old_x,old_y,method='mopc_hr')
    new_x=torch.randn(8,2)*100;new_y=torch.tensor([2]*4+[3]*4)
    cfg=RegistrationConfig(seed=seed,mopc_semantics='public_code',mopc_epochs=20,
                           mopc_batch_size=4,mopc_max_grad_norm=cap)
    try:
        state=register_new_classes(anchor,new_x,new_y,[2,3],config=cfg)
        return dict(seed=seed,clip=cap,status='COMPLETE',updates=len(state.loss_trace),
                    clipped=state.protocol_metadata.get('gradient_clipped_updates',0),
                    max_loss=max(x['loss'] for x in state.loss_trace))
    except (FloatingPointError,ValueError) as exc:
        return dict(seed=seed,clip=cap,status='FAILED',error=str(exc))


def main():
    torch.set_num_threads(1)
    print(json.dumps(dict(scope='synthetic_scaled_linear_encoder_CPU_only',
        results=[trial(seed,cap) for seed in (1731,1732,1733) for cap in (None,1.)],
        actual_support_read=False,query_read=False,truth_read=False,
        note='clipping changes optimizer updates; no claim of author parity or real-failure repair')))


if __name__=='__main__':main()
