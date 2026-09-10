"""One real SSDG epoch on explicitly synthetic IQ; no scientific result claim."""
import copy
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from SSDG.train_ssdg import build_arg_parser,_build_ssdg_wisig_data,train
from cvsrffi.core90_evidence_profile import core90_arguments
from cvsrffi.evidence_head_training import validate_data_contract
from cvsrffi.evidence_pipeline import export_completed_training,seal_predictions
from post_stage_common import build_baseline_model


def test_production_train_one_epoch_and_tensor_only_export(tmp_path):
    torch.set_num_threads(2)
    gen=np.random.default_rng(89)
    data=[[[[gen.normal(size=(100,256,2)).astype(np.float32)] for day in range(3)] for rx in range(3)] for tx in range(3)]
    ds={'dataset_id':'SYNTHETIC_TECHNICAL_FIXTURE','data':data,'tx_list':['a','b','c'],
        'rx_list':['r0','r1','r2'],'capture_date_list':['d0','d1','d2'],'equalized_list':[0]}
    pkl=tmp_path/'synthetic.pkl'
    with pkl.open('wb') as f:pickle.dump(ds,f)
    output=tmp_path/'run';contract_path=tmp_path/'contract.json'
    args=build_arg_parser().parse_args(core90_arguments(str(pkl),str(contract_path),str(output),variant='H5',device='cpu',
                 source_rxs='0,1',source_days='0,1',target_rxs='2',target_days='2'))
    args.epochs=1;args.label_epochs=1;args.pseudo_epochs=0
    args.batch_size=84;args.eval_batch_size=12;args.num_workers=0;args.amp=False
    args.eval_max_batches=1;args.sat_eval_max_batches=1
    args.wisig_equalized='0'
    ctx=_build_ssdg_wisig_data(args,torch.device('cpu'))
    def ids(dataset):return [f'{i.tx_i}:{i.rx_i}:{i.day_i}:{i.sig_i}' for i in dataset.index]
    roles={k:ids(ctx[v].dataset) for k,v in [('L_s','train_loader'),('U_s','unlabeled_loader'),('V','val_loader')]}
    target=set()
    for loader in ctx['named_test_loaders'].values():
        for i in loader.dataset.index:
            if i.rx_i==2:target.add(f'{i.tx_i}:{i.rx_i}:{i.day_i}:{i.sig_i}')
    roles['target']=sorted(target)
    contract={'dataset_id':ds['dataset_id'],'source_receivers':[0,1],'target_receivers':[2],'roles':roles}
    contract_path.write_text(json.dumps(contract),encoding='utf-8')
    validate_data_contract(ctx,contract_path)
    bad=copy.deepcopy(contract);bad['roles']['target']=bad['roles']['target'][:-1]
    badpath=tmp_path/'bad.json';badpath.write_text(json.dumps(bad),encoding='utf-8')
    with pytest.raises(ValueError,match='target physical'):validate_data_contract(ctx,badpath)
    assert train(args)==0
    checkpoint=torch.load(output/'final_ssdg.pth',map_location='cpu',weights_only=False)
    assert checkpoint['training_data_contract']==contract
    assert 'rng_state' in checkpoint
    assert checkpoint['checkpoint_lineage']['from_scratch']
    bundle=export_completed_training(output)
    safe=torch.load(output/'ground_checkpoint.pt',map_location='cpu',weights_only=True)
    assert 'rng_state' not in safe
    model=build_baseline_model(SimpleNamespace(**bundle['baseline_args']),torch.device('cpu'))
    model.load_state_dict(bundle['model'],strict=True)
    received={'x':torch.randn(2,2,256),'query_ids':['fixture-q0','fixture-q1'],'protocol_schema':'p2_min_v1',
              'phase2_data_status':'VALIDATED_ONCE','capsule_id':'test-only','split_id':'fixture'}
    result=seal_predictions(model,received,tmp_path/'pred.json')
    assert len(result['rows'])==2
    logs=[json.loads(line) for line in (output/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert logs
    # The real training reducer must contain the support-episode activation;
    # this detects a CLI/optimizer/loss path bypass that math unit tests cannot.
    assert logs[0]['train_evidence_support_queries']>0
    assert logs[0]['train_evidence_observation_active']>0
    assert logs[0]['train_evidence_correlation_energy']>0
    terminal=json.loads((output/'phase1_terminal_status.json').read_text(encoding='utf-8'))
    assert terminal['status']=='COMPLETE' and not terminal['promotion_ready']
