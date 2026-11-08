"""Short source-only full-model comparison of native and polyphase training."""
import argparse
import json
import os
import struct
from pathlib import Path
import time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.benchmark_cross_expert_resident import tree_cpu
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.polyphase_cross_expert import PolyphaseCrossExpertRefiner
from experiments.phase1_dura_orthogonal.shared_source_benchmark import same_bits
from experiments.phase1_dura_orthogonal.signal_confidence_gate import source_gate_loss
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import load_direct
from experiments.phase1_dura_orthogonal.train_source import extract_context,load_source_training


def tree_bits(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,torch.Tensor):return same_bits(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_bits(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tree_bits(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return struct.pack('!d',a)==struct.pack('!d',b)
    return a==b


def compare_tensors(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Matching finite tensors required')
    diff=(a-b).abs()
    return dict(bitwise_equal=same_bits(a,b),max_abs=float(diff.max()),relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-20)),
                allclose_1e4=bool(torch.allclose(a,b,rtol=1e-4,atol=1e-5)))


def compare_model(a,b):
    if a.keys()!=b.keys():raise ValueError('Matching state keys required')
    max_abs=0.;sq_diff=0.;sq_base=0.;unequal=0;outside=0
    for key in a:
        x,y=a[key],b[key]
        if x.shape!=y.shape or x.dtype!=y.dtype or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Matching finite state required')
        delta=(x-y).double();max_abs=max(max_abs,float(delta.abs().max()))
        sq_diff+=float(delta.square().sum());sq_base+=float(x.double().square().sum())
        unequal+=int(not same_bits(x,y));outside+=int(not torch.allclose(x,y,rtol=1e-4,atol=1e-5))
    return dict(bitwise_equal=unequal==0,different_tensors=unequal,outside_tolerance_tensors=outside,
                max_abs=max_abs,relative_l2=(sq_diff/max(sq_base,1e-40))**.5)


def train(fields, cls, steps):
    torch.manual_seed(2026110830);model=cls().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    generator=torch.Generator().manual_seed(2026110831);losses=[]
    torch.cuda.synchronize();start=time.perf_counter();order=torch.randperm(len(fields[0]),generator=generator)
    for step in range(steps):
        idx=order[step*128:(step+1)*128]
        x,anchor,direction,variance,clean,proposal=(v[idx].cuda() for v in fields)
        out=model(x,anchor,direction,proposal,variance);loss=source_gate_loss(out,anchor,clean)
        if not torch.isfinite(loss):raise FloatingPointError('Finite source loss required')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        losses.append(loss.detach())
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    with torch.no_grad():
        x,anchor,direction,variance,_,proposal=(v[:8].cuda() for v in fields)
        output=model(x,anchor,direction,proposal,variance).cpu()
    snapshot=dict(model=tree_cpu(model.state_dict()),optimizer=tree_cpu(optimizer.state_dict()),
        losses=tree_cpu(torch.stack(losses)),order_rng=generator.get_state(),cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),source_output=output)
    return snapshot,elapsed


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    ids,*_=source_inputs(args.root);_,_,_,fit=load_source_training(args.root,ids,fit_size=32768)
    data,_=load_cache(args.cache,fit,32,'cuda:0',deterministic=True)
    selected=tuple(torch.cat([v[i:i+512] for i in (0,32768,65536)]) for v in data);del data
    direct=load_direct(args.root,ids).cuda();dura=frozen_dura(args.root,'cuda:0')
    with torch.no_grad():
        actual=extract_context(dura,selected[0][:8].cuda())
        for a,b in zip(actual,selected[1:4]):torch.testing.assert_close(a,b[:8].cuda(),rtol=1e-4,atol=1e-5)
        proposal=[]
        for i in range(0,len(selected[0]),32):proposal.append(direct(selected[0][i:i+32].cuda(),validate=False).cpu())
    fields=(*selected,torch.cat(proposal));del direct,dura,proposal
    torch.manual_seed(2026110830);native=CrossExpertRefiner();rng_native=torch.get_rng_state()
    torch.manual_seed(2026110830);poly=PolyphaseCrossExpertRefiner();rng_poly=torch.get_rng_state()
    if not tree_bits(native.state_dict(),poly.state_dict()) or not torch.equal(rng_native,rng_poly):
        raise ValueError('Exact identical initial state and RNG required')
    del native,poly
    print('POLY_MODEL_REAL_SOURCE_SMOKE_PASS',len(fields[0]),flush=True)
    for cls in (CrossExpertRefiner,PolyphaseCrossExpertRefiner):train(fields,cls,1)
    pairs=[];classes={'native':CrossExpertRefiner,'polyphase':PolyphaseCrossExpertRefiner}
    for first in ('native','polyphase'):
        order=(first,'polyphase' if first=='native' else 'native')
        results={name:train(fields,classes[name],4) for name in order}
        (a,ta),(b,tb)=results['native'],results['polyphase']
        pair=dict(first=first,native_seconds=ta,polyphase_seconds=tb,steps=4,
            model=compare_model(a['model'],b['model']),losses=compare_tensors(a['losses'],b['losses']),
            source_output=compare_tensors(a['source_output'],b['source_output']),
            optimizer_bitwise_equal=tree_bits(a['optimizer'],b['optimizer']),
            cpu_rng_equal=torch.equal(a['cpu_rng'],b['cpu_rng']),cuda_rng_equal=torch.equal(a['cuda_rng'],b['cuda_rng']),
            order_rng_equal=torch.equal(a['order_rng'],b['order_rng']))
        pairs.append(pair);print('POLY_MODEL_PAIR',first,'native',ta,'polyphase',tb,flush=True)
    result=dict(status='POLY_MODEL_COMPLETE',commit=args.commit,source_ids=512,views=3,records=len(fields[0]),
        batch=128,pairs=pairs,gpu=torch.cuda.get_device_name(),TX_labels_read=0,F_weights_read=0,F_outputs_read=0,
        V_reads=0,target_reads=0,weights_saved=False,
        caveat='Two reversed-order four-step source comparisons on shared GPU. No saved model, no complete training, target predictions or identity accuracy; floating point states may differ.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
