"""Public random-input execution check, without formal data or weight reuse."""
import argparse,json
from pathlib import Path
import torch
from experiments.cvs_channel_response_identity.model import build,VARIANTS
from experiments.cvs_channel_response_identity.source import validate_config
from experiments.cvs_channel_response_identity.collect import validate_response_diagnostics

def run(output):
    torch.set_num_threads(2);rows=[]
    configs=Path(__file__).parent/'configs'
    for p in configs.glob('response*-s*.json'):validate_config(json.loads(p.read_text(encoding='utf-8')))
    if len(list(configs.glob('response*-s*.json')))!=12:raise ValueError('Missing actual source configs')
    for variant in VARIANTS:
        torch.manual_seed(20261003);model=build(variant).train()
        x=torch.randn(8,2,256);y=torch.arange(8)%6
        initial=model.response_branch(x)['residual']
        assert torch.count_nonzero(initial)==0
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
        losses=[]
        for _ in range(3):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();losses.append(loss.item())
        assert all(p.grad.norm()>0 for p in model.response_parameters())
        d=model.response_diagnostics(x);validate_response_diagnostics(d,model.contract(),8)
        assert d['response_projection_relative']>0
        rows.append(dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),losses=losses,diagnostics=d))
    result=dict(status='VERIFIED',torch_version=torch.__version__,formal_data_access=False,target_access=False,
                original_CE_only=True,actual_configs=12,rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
