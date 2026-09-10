import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_core90_evidence_experiment as experiment
from cvsrffi.evidence_head_training import validate_data_contract
from SSDG.train_ssdg import _run_final_heldout_evaluation,_resolve_phase1_terminal_status


def test_real_prepare_contract_source_order_and_target_union(tmp_path):
    rng=np.random.default_rng(392005)
    data=[[[[rng.normal(size=(100,256,2)).astype(np.float32)] for d in range(4)] for r in range(12)] for t in range(3)]
    payload={'dataset_id':'SYNTHETIC_RELEASE_FIXTURE','data':data,'tx_list':['a','b','c'],
             'rx_list':[f'r{i}' for i in range(12)],'capture_date_list':[f'd{i}' for i in range(4)],'equalized_list':[1]}
    pkl=tmp_path/'dataset.pkl'
    with pkl.open('wb') as handle:pickle.dump(payload,handle)
    root=tmp_path/'run';root.mkdir()
    plan=dict(dataset=str(pkl),seed=392005,source_rxs='1,3,4,6,8',source_days='1,2,3',
              target_rxs='0,2,5,7,9,10,11',heldout_days='0')
    experiment.write_json(root/'plan.json',plan)
    experiment.prepare(root)
    args,ctx=experiment.build_context(plan,root)
    contract=validate_data_contract(ctx,root/'data_contract.json')
    assert contract['split_seed']==392005 and contract['actual_target_days']==[0,1,2,3]
    assert contract['tx_mapping']==['a','b','c'] and contract['equalized']==1
    source=torch.load(root/'source_training.pt',weights_only=True)
    assert source['physical_ids']==contract['roles']['L_s']
    assert all(int(key.split(':')[0])==int(y) for key,y in zip(source['physical_ids'],source['y']))
    target=experiment.TargetRows(ctx,contract)
    assert len(target)==len(contract['roles']['target'])
    assert {i.rx_i for i in target.index}==set(contract['target_receivers'])
    with pytest.raises(FileExistsError):experiment.prepare(root)
    changed=dict(contract,equalized=0)
    experiment.write_json(root/'wrong.json',changed)
    with pytest.raises(ValueError):validate_data_contract(ctx,root/'wrong.json')


def test_external_evaluation_requires_contract_and_defers_target():
    args=SimpleNamespace(evidence_external_final_eval=True,evidence_data_contract='contract.json',
                         phase1_terminal_policy='core90_research',checkpoint_selection='final_only')
    result=_run_final_heldout_evaluation(args,None,None,None,'final_ssdg.pth')
    assert result['status']=='DELEGATED_TO_EVIDENCE_LAUNCHER'
    assert _resolve_phase1_terminal_status(tail_stopped=False,export_failed=False,final_blocked=False,
          selected_checkpoint_exists=True,heldout_eval_status=result['status'],external_final_eval=True)=='COMPLETE'
    args.evidence_data_contract=''
    with pytest.raises(ValueError):_run_final_heldout_evaluation(args,None,None,None,'final_ssdg.pth')


def test_scoring_refuses_partial_comparison_before_truth(tmp_path):
    with pytest.raises(FileNotFoundError):experiment.score_all(tmp_path)
    assert not (tmp_path/'target_truth.json').exists()


def test_readout_wrapper_consumes_frozen_parameters():
    class Backbone(torch.nn.Module):
        def forward(self,x,return_aux=True):return {'z_id':x,'tx_logits':x*0}
    head={'weight':torch.eye(2),'bias':torch.zeros(2),'normalize_input':torch.tensor(True)}
    model=experiment.FrozenControl(Backbone(),head)
    result=model(torch.tensor([[3.,4.]]))
    torch.testing.assert_close(result['tx_logits'],torch.tensor([[.6,.8]]))
