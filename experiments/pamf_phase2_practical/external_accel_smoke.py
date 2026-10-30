"""Real external checkpoints and source-only input; no target/query access."""
import copy
from pathlib import Path
import numpy as np
import torch
from . import external_additional as extra
from .external_acceleration import issl_ssl,issl_transfer
from .features import tensor
from .group1 import source_scope
from .group1_source import load_source
from .io import activate,read,require,write


def run(rt, output):
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from experiments.cric_dura_comparison.f_data import SourceDataset
    source=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']))
    x=source.x[:4];require(len(x)==4,'Insufficient real source-only smoke inputs')
    raw=np.asarray(x.tolist(),dtype=np.float32)
    old_tensor=torch.tensor(raw.tolist(),device='cuda:0',dtype=torch.float32)
    new_tensor=tensor(raw,'cuda:0')
    require(torch.equal(old_tensor,new_tensor),'Buffered source input differs')
    checks={};seed=rt['model_seeds'][0]
    for profile in rt['method_profiles']:
        bundle=load_source(source_scope(rt),profile,seed,'cuda:0')
        model=bundle['model'].eval()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            scores=bundle['score_fn'](model,new_tensor[:2])
        require(scores.shape==(2,6) and bool(torch.isfinite(scores).all()),'Invalid source checkpoint: '+profile)
        require(all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'Source anchor mutated: '+profile)
        checks[profile]='PASS'
        if profile=='issl':
            old=copy.deepcopy(model).requires_grad_(True);new=copy.deepcopy(model).requires_grad_(True)
            spec=dict(C_SSL_epochs=1,C_SSL_lr=1e-5);oldlog=[];newlog=[]
            was_deterministic=torch.backends.cudnn.deterministic
            torch.backends.cudnn.deterministic=True
            try:
                extra._seed(42);extra._issl_ssl(old,x,6,spec,42,oldlog.append)
                extra._seed(42);issl_ssl(new,x,6,spec,42,newlog.append)
                require(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()),
                        'ISSL accelerated source SSL update differs')
                require(oldlog==newlog,'ISSL source SSL log differs')
                y=torch.arange(4,dtype=torch.long)%6;oldlog=[];newlog=[]
                extra._seed(51);extra._train_ce(old,x,y,1,4,.03,51,oldlog.append,crop=218,stage='C_transfer')
                extra._seed(51);issl_transfer(new,x,y,1,4,.03,51,newlog.append,crop=218)
                require(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()),
                        'ISSL accelerated source transfer differs')
                require(oldlog==newlog,'ISSL source transfer log differs')
            finally:
                torch.backends.cudnn.deterministic=was_deterministic
            del old,new
        del before,model,bundle,scores
    write(Path(output),dict(status='PASS',actual_checkpoints=checks,source_rows=4,
        target_rows=0,query_rows=0,truth_read=False,input_exact=True,
        issl_source_training_equal=True,deterministic_setting='verification_only_restored_before_launch'))
