"""Finite public-only CE and frozen-property evidence for all registered seeds."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.cvs_spectral_relation_identity.model import build,VARIANTS,relation_contract,BASE_PARAMETERS,NEW_PARAMETERS
from experiments.cvs_spectral_relation_identity.physics import public_inputs,frozen_synthetic_diagnostics
from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY


def run(output):
    torch.set_num_threads(2);records=[]
    with numerical_context(FULL_FP32_POLICY):
        x=public_inputs(torch.device('cpu'),torch.float32);labels=torch.arange(30)%6
        for seed in range(2026092701,2026092705):
            for variant in VARIANTS:
                torch.manual_seed(seed);model=build(variant)
                assert model.contract()==relation_contract(variant)
                assert sum(p.numel() for p in model.parameters())==BASE_PARAMETERS+NEW_PARAMETERS
                optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
                steps=[];tic=time.perf_counter()
                initial={n:p.detach().clone() for n,p in model.core.spectral_relation.named_parameters()}
                for step in range(1,4):
                    model.train();optimizer.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(x),labels);assert torch.isfinite(loss)
                    loss.backward()
                    parameters=list(model.parameters());new=model.spectral_relation_parameters()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                    norm=float(torch.stack([p.grad.norm() for p in new]).norm())
                    optimizer.step()
                    steps.append(dict(step=step,ce=float(loss.detach()),ce_weight=1.,extra_losses=[],new_gradient_norm=norm,
                        gradient_used_parameters=sum(p.numel() for p in parameters if p.grad is not None)))
                changed={n:bool(not torch.equal(p,initial[n])) for n,p in model.core.spectral_relation.named_parameters()}
                assert all(changed.values())
                physics=frozen_synthetic_diagnostics(model);ideal=physics['ideal_gain']
                assert ideal['minimum_original_frequency_energy']>1e-6 and ideal['minimum_changed_frequency_energy']>1e-6
                assert ideal['varying_gain_eligible_frequency_count']>0
                assert ideal['common_complex_gain_max_error']<2e-5 and ideal['frequency_phase_only_max_error']<2e-5
                if variant=='relation_frequency_energy':assert ideal['varying_frequency_gain_eligible_max_error']<2e-5
                else:assert ideal['varying_frequency_gain_relative_distance']>1e-3
                assert physics['whole_model_phase']['ordinary_max_abs_error']<5e-4
                assert physics['whole_model_phase']['zero_finite']
                model.eval()
                with torch.no_grad():
                    for _ in range(2):model(x)
                    begin=time.perf_counter()
                    for _ in range(5):model(x)
                    forward_ms=(time.perf_counter()-begin)*1000/5
                records.append(dict(variant=variant,seed=seed,total_parameters=BASE_PARAMETERS+NEW_PARAMETERS,
                    new_parameters=NEW_PARAMETERS,steps=steps,all_new_tensors_changed=changed,
                    public_cpu_forward_batch30_ms=forward_ms,elapsed_seconds=time.perf_counter()-tic,physics=physics))
    result=dict(status='VERIFIED',models=8,public_ce_updates=24,records=records,real_data_access=False,
        target_access=False,scope='Public correctness and activity only; no formal training or recognition evidence',
        hardware=dict(device='cpu',torch_version=torch.__version__,threads=torch.get_num_threads()))
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Preserve previous public validation evidence')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
