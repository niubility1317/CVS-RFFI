"""Bounded paired CUDA first-divergence audit; no target/checkpoint access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.data import build_source
from cvsrffi.game_tracking.runtime import build_model
from cvsrffi.game_tracking.step_context import prepare_context,Core90Objective
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.schedule import build_stage_state


def difference(a,b,path=''):
    """First exact difference and first preset tolerance failure, independently."""
    exact=[]; failed=[]
    def walk(x,y,key):
        if torch.is_tensor(x):
            if y is None or x.shape!=y.shape:
                row={'path':key,'kind':'shape_or_presence'};exact.append(row);failed.append(row);return
            if torch.equal(x,y):return
            delta=(x.detach().double()-y.detach().double()).abs()
            row={'path':key,'max_abs':float(delta.max()),'different_values':int((x!=y).sum())}
            exact.append(row)
            if not torch.allclose(x,y,atol=1e-6,rtol=1e-5):failed.append(row)
        elif isinstance(x,dict):
            if x.keys()!=y.keys():
                row={'path':key,'kind':'keys'};exact.append(row);failed.append(row);return
            for k in x:walk(x[k],y[k],key+'/'+str(k))
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y):
                row={'path':key,'kind':'length'};exact.append(row);failed.append(row);return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,key+'/'+str(i))
        elif isinstance(x,np.ndarray):
            if not np.array_equal(x,y):
                row={'path':key,'kind':'numpy'};exact.append(row);failed.append(row)
        elif x!=y:
            row={'path':key,'kind':'value'};exact.append(row);failed.append(row)
    walk(a,b,path)
    return {'exact':not exact,'within_tolerance':not failed,'first_exact_difference':exact[:1],
            'first_tolerance_failure':failed[:1],'different_tensors':len(exact)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wisig-pkl',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8);p.add_argument('--stage',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=90)
    p.add_argument('--determinism',choices=['both','off','on'],default='both')
    p.add_argument('--right',choices=['both','reference','head_grad_only','graph_reuse'],default='both')
    cli=p.parse_args();torch.set_num_threads(2);device=torch.device('cuda')
    args=parse_args(['--output_dir','unused','--wisig_pkl',cli.wisig_pkl,'--num_workers','0'])
    source=build_source(args)
    batch=next(iter(source.loader('train',cli.batch_size,seed=921,shuffle=True)))
    ub=next(iter(source.loader('unlabeled',cli.batch_size,seed=921,shuffle=True))) if cli.stage>=131 else None
    torch.manual_seed(12345)
    base=build_model(args,len(source.domains),device).train()
    teacher=copy.deepcopy(base).eval();teacher.requires_grad_(False)
    proto=PrototypeMemoryBank(args.num_classes,len(source.domains));proto._lazy_init(160,device,torch.float32)
    template=prepare_context(batch,ub,base,teacher,args,cli.stage,1,_loss_weights(args,build_stage_state(cli.stage,args)),torch.Generator(device=device).manual_seed(777))
    start_rng=RNGState.capture()
    report={'schema':'core90_b8_first_divergence_v2','torch':torch.__version__,'device':torch.cuda.get_device_name(),
            'atol':1e-6,'rtol':1e-5,'stage':cli.stage,'steps_budget':cli.steps,
            'source_path':cli.wisig_pkl,'source_counts':source.info['counts'],'rows':[]}
    cli.output.parent.mkdir(parents=True,exist_ok=True)
    def write():cli.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    deterministic_values=[False,True] if cli.determinism=='both' else [cli.determinism=='on']
    right_values=['reference','head_grad_only'] if cli.right=='both' else [cli.right]
    for deterministic in deterministic_values:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic=deterministic;torch.backends.cudnn.benchmark=False
        for right in right_values:
            row={'deterministic':deterministic,'left':'reference','right':right,'steps':[]}
            report['rows'].append(row);write()
            a=copy.deepcopy(base);b=copy.deepcopy(base);pa=copy.deepcopy(proto);pb=copy.deepcopy(proto)
            ca=copy.deepcopy(template);cb=copy.deepcopy(template)
            oa=torch.optim.AdamW(a.parameters(),lr=args.lr,weight_decay=args.weight_decay)
            ob=torch.optim.AdamW(b.parameters(),lr=args.lr,weight_decay=args.weight_decay)
            sa=GameSolver(a,oa,'head_lookahead',max_grad_norm=5,telemetry_interval=1)
            sb=GameSolver(b,ob,'head_lookahead',max_grad_norm=5,b8_impl=right,telemetry_interval=1)
            fa=Core90Objective(a,args,pa);fb=Core90Objective(b,args,pb);start_rng.restore()
            try:
                for step in range(cli.steps):
                    ca.origin_features=None;ca.origin_terms={};cb.origin_features=None;cb.origin_terms={}
                    before_rng=RNGState.capture()
                    ra=sa.step(lambda:fa(ca));after_a=RNGState.capture()
                    before_rng.restore()
                    if right=='graph_reuse':
                        from cvsrffi.game_tracking.head_lookahead import Core90ReusableGraph
                        rb=sb.step(Core90ReusableGraph(fb,cb))
                    else:rb=sb.step(lambda:fb(cb))
                    after_b=RNGState.capture()
                    checks={'parameters_buffers':difference(a.state_dict(),b.state_dict()),
                            'optimizer':difference(oa.state_dict(),ob.state_dict()),
                            'RNG':difference(vars(after_a),vars(after_b)),
                            'mask':difference(ca.strong_mask,cb.strong_mask),
                            'prototype':difference(vars(pa),vars(pb))}
                    for key in ('predictor_clipped','corrector','formal_clipped','virtual_update','formal_update'):
                        checks[key]=difference(sa.last_trace[key],sb.last_trace[key])
                    ai={name:g for name,p,g in zip(sa.names,sa.parameters,sa.last_trace['origin']) if id(p) in sa.head_ids}
                    bi={name:g for name,p,g in zip(sb.names,sb.parameters,sb.last_trace['origin']) if id(p) in sb.head_ids}
                    checks['origin_head_raw']=difference(ai,bi)
                    result={'step':step,'loss_left':ra.loss,'loss_right':rb.loss,'checks':checks}
                    row['steps'].append(result);write()
                    print(json.dumps({'deterministic':deterministic,'right':right,'step':step,
                                      'different':[k for k,v in checks.items() if not v['exact']],
                                      'failed':[k for k,v in checks.items() if not v['within_tolerance']]}),flush=True)
                    if any(not v['within_tolerance'] for v in checks.values()):
                        artifact=cli.output.with_name(cli.output.stem+f'_{deterministic}_{right}_first_failure.pt')
                        torch.save({'step':step,'names':sa.names,'left':a.state_dict(),'right':b.state_dict(),
                                    'trace_left':sa.last_trace,'trace_right':sb.last_trace,
                                    'optimizer_left':oa.state_dict(),'optimizer_right':ob.state_dict(),
                                    'rng_left':after_a,'rng_right':after_b},artifact)
                        row['first_failure_artifact']=str(artifact);break
            except Exception as exc:
                row['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
                print(json.dumps(row['error']),flush=True)
            write()


if __name__=='__main__':main()
