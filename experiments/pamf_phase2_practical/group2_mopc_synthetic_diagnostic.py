"""Reproduce the ProtoNet/MoPC synthetic instability with and without clipping."""
import argparse
from collections import deque
import json
import os
from unittest.mock import patch
import torch

from .group2_cil_bridge import ProtoClassifier, from_protonet, make_anchor, register
from .group2_proto_mopc_affine import ProtoMoPCAffine
from .group2_savedb_candidate_diagnostic import historical_record
from .vendor_native.comparison_suite.models import ProtoNet
from .vendor_native.comparison_suite import registration as native


def finite_absmax(value):
    if value is None:return None
    value=value.detach().abs()
    return float(value.max()) if bool(torch.isfinite(value).all()) else None


def run_variant(clip,semantics='paper',seed=1731,new_count=10,bridge='original'):
    if new_count not in (2,5,10,20):raise ValueError('Invalid number of new classes')
    if bridge not in ('original','affine'):raise ValueError('Invalid diagnostic bridge')
    torch.manual_seed(seed)
    base=ProtoNet(6).cuda().eval();prototypes=torch.randn(6,256,device='cuda')*.001
    model=(from_protonet(base,prototypes) if bridge=='original' else ProtoMoPCAffine(base,prototypes)).cuda()
    old_x=torch.randn(60,2,256,device='cuda');old_y=torch.arange(6,device='cuda').repeat_interleave(10)
    anchor=make_anchor(model,old_x,old_y,method='mopc_hr')
    x=torch.randn(new_count*10,2,256,device='cuda');y=torch.arange(6,6+new_count,device='cuda').repeat_interleave(10)
    bridge_class=ProtoClassifier if bridge=='original' else ProtoMoPCAffine
    forward=bridge_class.forward;features={};recent=deque(maxlen=8);peak=None
    def observe_forward(self,x):
        result=forward(self,x)
        if self.training:
            mapped=result['features'].detach()
            features.update(embedding_absmax=finite_absmax(mapped[:,:-1] if bridge=='original' else mapped),
                            quadratic_absmax=finite_absmax(mapped[:,-1]) if bridge=='original' else None)
        return result
    def observe_record(trace,current,terms,callback,start,**fields):
        nonlocal peak
        grad=current.fc.weight.grad
        row=dict(stage=fields['stage'],iteration=fields['iteration'],loss=finite_absmax(terms['loss']),
                 quadratic_absmax=features.get('quadratic_absmax'),embedding_absmax=features.get('embedding_absmax'),
                 quadratic_weight_gradient_absmax=finite_absmax(grad[:,-1] if grad is not None and bridge=='original' else None),
                 other_weight_gradient_absmax=finite_absmax(grad[:,:-1] if grad is not None and bridge=='original' else grad))
        recent.append(row)
        if row['quadratic_absmax'] is not None and (peak is None or row['quadratic_absmax']>peak['quadratic_absmax']):
            peak=row.copy()
        return historical_record(trace,current,terms,callback,start,**fields)
    try:
        with patch.object(bridge_class,'forward',observe_forward),patch.object(native,'_record',observe_record):
            state=register(anchor,x,y,seed=42,mopc_semantics=semantics,mopc_max_grad_norm=clip)
        status='COMPLETE';error=None;updates=len(state.loss_trace)
        clipped=state.protocol_metadata.get('gradient_clipped_updates',0)
    except FloatingPointError as exc:
        status='NONFINITE';error=str(exc);updates=None;clipped=None
    return dict(variant='no_clip' if clip is None else 'clip1',status=status,error=error,
                completed_updates=updates,clipped_updates=clipped,peak=peak,recent=list(recent),
                optimizer_changed=clip is not None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--semantics',choices=['paper','public_code'],default='paper')
    p.add_argument('--seed',type=int,action='append')
    p.add_argument('--new-count',type=int,choices=[2,5,10,20],default=10)
    p.add_argument('--bridge',choices=['original','affine'],default='original')
    args=p.parse_args()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
        raise ValueError('Deterministic CUBLAS_WORKSPACE_CONFIG is required before startup')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    for seed in args.seed or [1731]:
        results=[run_variant(None,args.semantics,seed,args.new_count,args.bridge),run_variant(1.,args.semantics,seed,args.new_count,args.bridge)]
        print(json.dumps(dict(scope='synthetic_only',seed=seed,new_count=args.new_count,bridge=args.bridge,semantics=args.semantics,guard='historical_fp32',
            results=results,query_read=False,truth_read=False,predictions_written=False),allow_nan=False),flush=True)


if __name__=='__main__':main()
