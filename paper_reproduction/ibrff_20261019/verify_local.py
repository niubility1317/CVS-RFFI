"""Focused author-loop equivalence and no-query protocol checks on synthetic IQ."""
import argparse
import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import pipeline as p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=pathlib.Path,required=True)
    parser.add_argument('--model-root',type=pathlib.Path,required=True)
    parser.add_argument('--runtime-dependencies',type=pathlib.Path)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args()
    if args.runtime_dependencies:sys.path.insert(0,str(args.runtime_dependencies))
    torch.set_num_threads(2)
    checks=[]
    from timm.layers import trunc_normal_ as timm_init
    torch.manual_seed(2023);a=torch.empty(4096);timm_init(a,std=.02);rng=torch.get_rng_state()
    torch.manual_seed(2023);b=torch.empty(4096);torch.nn.init.trunc_normal_(b,std=.02)
    assert torch.equal(a,b) and torch.equal(rng,torch.get_rng_state())
    checks.append('seeded_timm_torch_initializer_and_rng_identical')
    conf=p.read_json(pathlib.Path(__file__).with_name('candidate.json'))
    fixture_conf=copy.deepcopy(conf);fixture_conf['epochs_per_stage']=1
    ct=p.contract(conf)
    with tempfile.TemporaryDirectory(prefix='ibrff-verify-') as tmp:
        root=pathlib.Path(tmp);run=root/'run';run.mkdir()
        candidate=root/'candidate.json';p.write_json(candidate,fixture_conf)
        roles=root/'roles.json'
        p.write_json(roles,{'rx_author_order':conf['rx_author_order'],'source':ct['source_rx'],'target':ct['target_rx'],'asknet_wavemlp':{'packet_rows':ct['packet_rows']}})
        runtime={'candidate':str(candidate),'model_root':str(args.model_root),'run_root':str(run),'physical_roles':str(roles),'data_root':str(root/'unused'),'device':'cpu'}
        calls=[]
        def source_spy(data,rx,day):
            assert rx in ct['source_rx'] and day in [1,2]
            calls.append((rx,day))
            return np.zeros((600,2,256),dtype=np.float32),np.repeat(np.arange(6),100)
        with mock.patch.object(p,'read_block',side_effect=source_spy):
            train,val=p.source_data(runtime,conf)
        assert len(calls)==18 and len(train[0])==7560 and len(val[0])==3240
        checks.append('source_loader_opens_only_18_source_rx_day_blocks')
        bad=dict(ct,source_rx=ct['target_rx'])
        torch.save({'source_contract':bad,'upstream_commit':conf['upstream_commit']},root/'bad.pt')
        try:p.load_checkpoint(root/'bad.pt',ct,'cpu')
        except ValueError:pass
        else:raise AssertionError('Mismatched checkpoint accepted')
        checks.append('checkpoint_contract_mismatch_rejected')
        with mock.patch.object(p.np,'load',side_effect=AssertionError('Truth accessed')):
            try:p.score(run)
            except FileNotFoundError:pass
            else:raise AssertionError('Truth-side scorer ran before predictions')
        checks.append('scorer_cannot_open_truth_before_prediction_completion')
        try:p.prepare_queries(runtime)
        except ValueError:pass
        else:raise AssertionError('Target export ran before training freeze')
        checks.append('target_export_requires_frozen_training')
        gen=np.random.RandomState(91)
        tx=gen.randn(16,2,256).astype(np.float32);ty=np.arange(16)%6
        vx=gen.randn(8,2,256).astype(np.float32);vy=np.arange(8)%6
        fixture=((tx,ty),(vx,vy))
        # Import the untouched author's actual training function, not a rewrite.
        sys.path.insert(0,str(args.upstream))
        spec=importlib.util.spec_from_file_location('author_ibrff_main',args.upstream/'main.py')
        author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
        for stage in ['stage1','stage2']:
            with mock.patch.object(p,'source_data',return_value=fixture):p.train_stage(runtime,stage)
            p.seed_everything(conf['model_seed'])
            dl=DataLoader(TensorDataset(torch.Tensor(tx),torch.Tensor(ty)),batch_size=128,shuffle=True)
            expected=p.model(runtime,conf)
            if stage=='stage2':expected.load_state_dict(p.load_checkpoint(run/'stage1/best.pt',ct,'cpu')['state_dict'])
            opt=torch.optim.Adam(expected.parameters(),lr=conf['lr'],weight_decay=0)
            with mock.patch.object(torch.cuda,'is_available',return_value=False):
                author.train(expected,torch.nn.NLLLoss(),dl,opt,1,conf[stage+'_alpha'])
            actual=p.load_checkpoint(run/stage/'best.pt',ct,'cpu')['state_dict']
            assert all(torch.equal(value,actual[key]) for key,value in expected.state_dict().items()),stage
        checks.append('both_stage_updates_match_untouched_author_train_bitwise')
        with mock.patch.object(p,'source_data',return_value=fixture):p.smoke(runtime)
        checks.append('source_only_real_checkpoint_roundtrip_smoke')
        p.write_json(root/'exclusive.json',{'first':1})
        try:p.write_json(root/'exclusive.json',{'overwrite':1})
        except FileExistsError:pass
        else:raise AssertionError('Immutable artifact overwritten')
        checks.append('existing_artifact_overwrite_rejected')
    result={'status':'PASS','checks':checks,'input':'synthetic CPU IQ only; no source or target real data; no author weights loaded','torch':torch.__version__}
    p.write_json(args.output,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
