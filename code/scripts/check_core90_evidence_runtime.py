"""Bounded source-only runtime check, independent of pytest and formal training."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root',required=True)
    parser.add_argument('--run-root',required=True)
    args=parser.parse_args();root=Path(args.run_root)
    output=root/'remote_head_runtime.json'
    if output.exists():raise FileExistsError(output)
    sys.path.insert(0,args.code_root)
    from post_stage_common import build_baseline_model
    from cvsrffi.evidence_head_training import attach_evidence_head,initialize_from_source,supervised_evidence_loss
    from cvsrffi.evidence_pipeline import class_interleaved_order
    source=torch.load(root/'source_training.pt',map_location='cpu',weights_only=True)
    # This run-owned scratch smoke is a technical fixture, never a formal base.
    checkpoint=torch.load(root/'scratch_smoke.pt',map_location='cpu',weights_only=True)
    take=class_interleaved_order(source['y'],392005)[:24]
    x=source['x'][take].cuda();y=source['y'][take].cuda()
    ids=[source['physical_ids'][i] for i in take]
    results=[]
    for variant in ('H1','H2','H3','H4'):
        torch.manual_seed(392005)
        model=build_baseline_model(SimpleNamespace(**checkpoint['baseline_args']),torch.device('cuda:0'))
        model.load_state_dict(checkpoint['model'],strict=True);model.requires_grad_(False)
        attach_evidence_head(model,dict(variant=variant,covariance_rank=0 if variant=='H1' else 4))
        initialize_from_source(model,[(x,y)])
        model.eval();model.evidence_head.train();torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        out=model(x,y_tx=y,return_aux=True)
        extra,stats=supervised_evidence_loss(model,out,y,{'physical_sample_id':ids},len(y))
        loss=torch.nn.functional.cross_entropy(out['tx_logits'],y)+extra;loss.backward()
        if not torch.isfinite(loss):raise FloatingPointError(variant+' loss')
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(variant+' gradient')
        if not model.evidence_head.response.mean.grad.abs().sum()>0:raise RuntimeError('mean inactive')
        if variant=='H4' and stats['evidence/support_effective_queries']<=0:raise RuntimeError('support inactive')
        if any(p.grad is not None for p in model.id_backbone.parameters()):raise RuntimeError('backbone not frozen')
        torch.cuda.synchronize()
        row={'variant':variant,'loss':float(loss),'seconds':time.monotonic()-started,
             'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2,'finite_gradients':True,
             'support_effective_queries':stats['evidence/support_effective_queries']}
        results.append(row);print(json.dumps(row),flush=True)
        del model,out,loss,extra
        torch.cuda.empty_cache()
    output.write_text(json.dumps({'status':'PASS','torch':str(torch.__version__),'role':'L_s_only',
        'formal_checkpoint_modified':False,'formal_training_initialization':False,'results':results},indent=2),encoding='utf-8')


if __name__=='__main__':main()
