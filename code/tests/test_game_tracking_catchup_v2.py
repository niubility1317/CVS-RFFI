from copy import deepcopy
import torch
from scripts.accept_core90_v2_source import acceptance_args
from cvsrffi.game_tracking import runtime
from cvsrffi.game_tracking.data import build_source
from cvsrffi.game_tracking.step_context import prepare_context
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.schedule import build_stage_state


def test_v2_catchup_uses_combined_training_head_and_only_commits_head():
    torch.set_num_threads(2)
    args=acceptance_args(synthetic=True)
    source=build_source(args)
    model=runtime.build_model(args,len(source.domains),torch.device('cpu')).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
    batch=next(iter(source.loader('train',18,seed=1,shuffle=False,workers=0,drop_last=True)))
    ctx=prepare_context(batch,None,model,None,args,131,1,_loss_weights(args,build_stage_state(131,args)),
                        torch.Generator().manual_seed(2))
    before=deepcopy(model.state_dict());seen=[]
    hook=model.adv_head.register_forward_pre_hook(lambda m,a:seen.append((len(a[0]),m.training)))
    steps,forwards=runtime.head_catchup(model,optimizer,ctx,1,args)
    hook.remove()
    assert steps==1 and forwards==1
    assert seen and all(n==36 and training for n,training in seen)
    for name,value in model.state_dict().items():
        if not name.startswith('adv_head.'):
            assert torch.equal(value,before[name]),name
    assert any(not torch.equal(value,before[name]) for name,value in model.state_dict().items()
               if name.startswith('adv_head.'))
