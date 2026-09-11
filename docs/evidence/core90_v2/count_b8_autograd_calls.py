"""Separate untimed CPU API-call audit on fixed legal source contexts."""
import copy
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
import torch
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.data import build_source
from cvsrffi.game_tracking.runtime import build_model
from cvsrffi.game_tracking.step_context import prepare_context,Core90Objective
from cvsrffi.game_tracking.head_lookahead import Core90ReusableGraph
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.schedule import build_stage_state

torch.set_num_threads(2)
args=parse_args(['--output_dir','unused','--wisig_pkl','E:/type10-7/local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl','--num_workers','0'])
source=build_source(args)
batch=next(iter(source.loader('train',90,seed=921,shuffle=True)))
ub=next(iter(source.loader('unlabeled',90,seed=921,shuffle=True)))
report={'schema':'b8_untimed_autograd_api_count_v2','device':'cpu','batch_size':90,
        'source_counts':source.info['counts'],'rows':[],
        'scope':'API call counts, not CUDA kernels or all internal backward operators',
        'explicit_extra_hvp_calls':0}
output=Path(__file__).with_name('b8_autograd_call_counts.json')
for epoch in (1,80,131):
    torch.manual_seed(12345)
    base=build_model(args,len(source.domains),torch.device('cpu')).train()
    teacher=copy.deepcopy(base).eval();teacher.requires_grad_(False)
    proto=PrototypeMemoryBank(args.num_classes,len(source.domains));proto._lazy_init(160,torch.device('cpu'),torch.float32)
    template=prepare_context(batch,ub if epoch>=131 else None,base,teacher,args,epoch,1,
        _loss_weights(args,build_stage_state(epoch,args)),torch.Generator().manual_seed(777))
    rng=RNGState.capture()
    for implementation in ('reference','head_grad_only','graph_reuse'):
        model=copy.deepcopy(base);bank=copy.deepcopy(proto);ctx=copy.deepcopy(template)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        solver=GameSolver(model,opt,'head_lookahead',b8_impl=implementation,max_grad_norm=5)
        objective=Core90Objective(model,args,bank);prepared_forward=ctx.forward_calls
        calls=[];tensor_backward=[]
        original=torch.autograd.grad;original_backward=torch.Tensor.backward
        head_ids={id(p) for p in model.adv_head.parameters()};all_ids={id(p) for p in model.parameters()}
        def counted(*pos,**kw):
            frames=inspect.stack(context=0)
            stage=None;scope='objective_internal';isolated_ids=set()
            for frame in frames[1:]:
                if frame.function=='_evaluate' and frame.filename.endswith('solvers.py'):
                    stage=frame.frame.f_locals['stage'];scope='solver_'+stage;break
                if frame.function=='corrector' and frame.filename.endswith('head_lookahead.py'):
                    scope='solver_graph_corrector';isolated_ids={id(p) for p in frame.frame.f_locals.get('mapping',{}).values()};break
            # Innermost objective call wins over enclosing solver _evaluate.
            direct=frames[1]
            if not direct.filename.endswith(('solvers.py','head_lookahead.py')):scope='objective_internal'
            inputs=kw.get('inputs',pos[1] if len(pos)>1 else None)
            inputs=[inputs] if torch.is_tensor(inputs) else inputs
            ids=[id(v) for v in inputs] if isinstance(inputs,(tuple,list)) else None
            create_graph=bool(kw.get('create_graph',pos[4] if len(pos)>4 else False))
            calls.append(dict(scope=scope,stage=stage,create_graph=create_graph,
                targets=len(ids) if ids is not None else None,
                head_targets=sum(i in head_ids or i in isolated_ids for i in ids) if ids is not None else None,
                caller=f'{Path(direct.filename).name}:{direct.function}:{direct.lineno}'))
            del frames,direct
            return original(*pos,**kw)
        def backward(self,*pos,**kw):
            tensor_backward.append(True);return original_backward(self,*pos,**kw)
        rng.restore();torch.autograd.grad=counted;torch.Tensor.backward=backward
        try:result=solver.step(Core90ReusableGraph(objective,ctx) if implementation=='graph_reuse' else lambda:objective(ctx))
        finally:torch.autograd.grad=original;torch.Tensor.backward=original_backward
        row=dict(stage=epoch,implementation=implementation,field_evaluations=result.field_evaluations,
                 autograd_grad_total=len(calls),create_graph_true=sum(c['create_graph'] for c in calls),
                 objective_internal_grad_calls=sum(c['scope']=='objective_internal' for c in calls),
                 tensor_backward_api_calls=len(tensor_backward),explicit_extra_hvp_calls=0,
                 model_forward=ctx.forward_calls,prepare_teacher_forward=prepared_forward,
                 solver_model_forward=ctx.forward_calls-prepared_forward,
                 calls=calls,origin_terms=ctx.origin_terms,pseudo_selected=ctx.pseudo_selected)
        report['rows'].append(row)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k not in ('origin_terms','calls')}),flush=True)
