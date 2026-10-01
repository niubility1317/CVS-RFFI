"""Read-only precision diagnosis of eight already-frozen source models."""
import argparse
import copy
import json
from pathlib import Path
import torch
from experiments.cvs_observable_identity.model import VARIANTS, build, observable_contract, frozen_synthetic_diagnostics


def diagnose(source_root, expected_contract, expected_commit, output, gpu=False):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    contract=json.loads(expected_contract.read_text())
    rows=[]
    initial_backend=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    for seed in range(2026092701,2026092705):
        for variant in VARIANTS:
            rid=f'{variant}-s{seed}';folder=source_root/rid/'source'
            resolved=json.loads((folder/'resolved_config.json').read_text())
            actual=json.loads((folder/'source_contract.json').read_text())
            initial=json.loads((folder/'initialization.json').read_text())
            done=json.loads((folder/'completion.json').read_text())
            assert actual==contract
            assert all(k in actual for k in ('classes','role_ids','source_rxs','source_days','ratios','split_seed','equalized','out_len','normalize'))
            assert resolved['commit']==expected_commit and resolved['method']=='cvs_observable_identity'
            assert resolved['variant']==variant and resolved['model_seed']==seed
            assert resolved['observables_actual']==observable_contract(variant) and resolved['observables']==observable_contract(variant) and resolved['observables_active']
            assert resolved['selection']=='fixed_last_epoch' and resolved['classifier_scale']==30.0
            assert not resolved['target_access'] and not resolved['augmentation'] and not resolved['domain_backbone'] and not resolved['extra_losses']
            assert initial['scratch_only'] and not initial['ancestors'] and not initial['checkpoint_sources'] and not initial['target_access']
            assert initial['checkpoint'] is None and not initial['target_contact'] and initial['selection']=='fixed_last_epoch'
            assert done['epoch']==200 and done['steps']==10000 and not done['target_evaluated']
            payload=torch.load(folder/'last.pt',map_location='cpu')
            assert payload['source_contract']==actual and payload['initialization']==initial
            assert payload['config']==resolved and payload['epoch']==200 and payload['variant']==variant
            assert payload['method']=='cvs_observable_identity' and payload['selection']=='fixed_last_epoch'
            assert payload['classes']==actual['classes'] and payload['num_classes']==6
            model=build(variant).eval();model.load_state_dict(payload['model'],strict=True)
            state={k:v.clone() for k,v in model.state_dict().items()}
            results=dict(cpu_float32=frozen_synthetic_diagnostics(model),
                         cpu_float64=frozen_synthetic_diagnostics(copy.deepcopy(model).double()))
            if gpu:
                device=torch.device('cuda:0');device_model=copy.deepcopy(model).to(device)
                torch.backends.cudnn.allow_tf32=initial_backend['cudnn_allow_tf32']
                torch.backends.cuda.matmul.allow_tf32=initial_backend['matmul_allow_tf32']
                results['gpu_float32_default']=frozen_synthetic_diagnostics(device_model)
                torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
                results['gpu_float32_tf32_disabled']=frozen_synthetic_diagnostics(device_model)
                del device_model
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
            rows.append(dict(row_id=rid,model_state_unchanged=True,diagnostics=results,
                             original_gpu_diagnostic=json.loads((folder/'source_physical_diagnostics.json').read_text())))
    result=dict(status='DIAGNOSED',rows=rows,expected_source_commit=expected_commit,
                torch_version=torch.__version__,initial_backend=initial_backend,
                hardware=torch.cuda.get_device_name(0) if gpu else 'cpu',tolerance=1e-3,
                original_results_preserved=True,source_selection_changed=False,
                formal_samples_access=False,target_access=False,training=False,
                claim='Precision diagnosis only;no retraining, target feedback or hardware identification.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'precision_diagnostics.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],rows=8,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--expected-contract',type=Path,required=True);p.add_argument('--expected-commit',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',action='store_true')
    a=p.parse_args();diagnose(a.source_root,a.expected_contract,a.expected_commit,a.output,a.gpu)
