import json
from types import SimpleNamespace

import pytest
import torch

from experiments.pamf_phase2_practical.group2_b_state import save_b_state


def test_saved_pilot_b_reused_only_with_matching_author_prediction(tmp_path):
    from experiments.pamf_phase2_practical.mopc_author_b_reuse import pilot_b
    model=torch.nn.Linear(2,2)
    anchor=SimpleNamespace(method='mopc_hr',model=torch.nn.Linear(2,2),prototypes=torch.ones(6,2),
                           protocol_metadata={})
    runtime={'data_root':'/data','source_run_root':'/source','source_contract':'/contract',
             'source_cache':'/cache','input_condition':'satellite_received'}
    (tmp_path/'pilot_runtime.json').write_text(json.dumps(runtime))
    job={'key':'same','profile':'mrior_sda','seed':2026092701,'capsule':'capsules/P01/MIX01/1-1',
         'methods':['mopc_hr'],'reuse_author_b':str(tmp_path),'reuse_author_runtime':str(tmp_path/'pilot_runtime.json')}
    save_b_state(tmp_path/'B-K5.pt',job,5,'oldsplit',123,model,anchor)
    row={'stage':'C','arm':'mrior_sda__mopc_hr','k':5,'new_count':5,'split_id':'newsplit','prediction':'p.npz',
         'input_condition':'satellite_received','resources':{'B_state_ref':str(tmp_path/'B-K5.pt'),
         'C_protocol':{'mopc_version':'public_code_semantics_matched_IQ_extension',
         'mopc_sampling_version':'author_drop_last_v1','config':{'mopc_max_grad_norm':None}}}}
    (tmp_path/'p.npz').touch()
    (tmp_path/'predictions.json').write_text(json.dumps({'capsule':job['capsule'],'profile':job['profile'],
         'model_seed':job['seed'],'rows':[row]}))
    (tmp_path/'complete.json').write_text(json.dumps({'status':'PREDICTIONS_COMPLETE','truth_read':False,
         'query_fit':False,'profile':job['profile'],'model_seed':job['seed'],'rows':1}))
    result=pilot_b(job,runtime,5,'oldsplit','newsplit',123,'cpu')
    assert result['path']==str(tmp_path/'B-K5.pt')
    assert pilot_b(job,runtime,10,'other','newsplit',123,'cpu') is None
    with pytest.raises(ValueError):pilot_b(job,runtime,5,'oldsplit','wrong',123,'cpu')
    with pytest.raises(ValueError):pilot_b(job,dict(runtime,source_cache='/other'),5,'oldsplit','newsplit',123,'cpu')
    job['seed']=2026092702
    with pytest.raises(ValueError):pilot_b(job,runtime,5,'oldsplit','newsplit',123,'cpu')
