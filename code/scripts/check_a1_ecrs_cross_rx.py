"""Fresh-source synthetic check of the actual A1/ECRS training entrypoint."""
from copy import deepcopy
from pathlib import Path
import argparse
import contextlib
import json
import pickle
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from SSDG import train_ssdg as train
from cvsrffi import a1_ecrs_cross_rx as cross
from run_a1_ecrs_cross_rx import ecrs_matrix
from run_a1_fast_v2 import v2_command


def run_check(device, folder, matrix_factory=ecrs_matrix, expect_cross_rx=True):
    torch.set_num_threads(2)
    folder.mkdir(parents=True,exist_ok=False)
    matrix=matrix_factory()
    initial=[]
    for row in matrix['rows']:
        args=train.build_arg_parser().parse_args(v2_command(matrix,project_root=folder,run_root=folder,row=row)[3:])
        train._validate_daot_config(args); train._validate_a1_scratch_only(args); train._validate_a1_ecrs_config(args)
        torch.manual_seed(args.seed)
        merged=train._apply_model_cli_args(train.merge_checkpoint_args({},args,input_len=256,num_domains=15),args)
        model=train.build_baseline_model(merged,device)
        state=train._initialize_muse_training_state(args,model,device)
        initial.append((model,state,args,torch.get_rng_state().clone()))
    first=initial[0]
    for model,state,_,rng in initial[1:]:
        for a,b in ((first[0],model),(first[1]['heads'],state['heads'])):
            assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        assert torch.equal(first[3],rng)
    model=first[0]
    x=torch.randn(30,2,256,device=device)
    labels=torch.arange(30,device=device)%6
    rx=torch.arange(30,device=device)//6
    day=torch.ones(30,device=device,dtype=torch.long)
    checks=[]
    for scope in ('clean','clean_leo'):
        for applied in (False,True):
            for amp in ([False,True] if device.type=='cuda' else [False]):
                model.zero_grad(set_to_none=True); model.train()
                with torch.autocast(device_type=device.type,enabled=amp):
                    clean=model(x,return_aux=True)['z_id']
                    leo=model(x+.02*torch.randn_like(x),return_aux=True)['z_id']
                    objective,logs=cross.labeled_cross_rx_objective(clean,labels,rx,day,
                        z_leo=leo,leo_applied=applied,scope=scope)
                assert logs['valid_anchors']>0 and objective.item()>0
                objective.backward()
                identity=[p.grad for n,p in model.named_parameters() if n.startswith('id_backbone.') and p.grad is not None]
                assert identity and all(torch.isfinite(g).all() for g in identity)
                assert sum(float(g.float().square().sum()) for g in identity)>0
                assert all(p.grad is None for n,p in model.named_parameters() if n.startswith('dom_backbone.'))
                checks.append({'scope':scope,'leo_applied':applied,'amp':amp,
                    'valid_anchors':logs['valid_anchors'],'leo_count':logs['leo_count'],
                    'loss':float(objective.detach()),'identity_gradient':'nonzero_finite'})
    del initial,model,state,first
    # Build only synthetic source IQ. All target cells are empty: the actual
    # dataset builder and L/U/V metadata path are exercised without target IQ.
    generator=np.random.default_rng(741)
    source_rxs={1,3,4,6,8}; source_days={1,2,3}
    data=[[[[generator.standard_normal((100,256,2)).astype(np.float32)
             if r in source_rxs and d in source_days else np.empty((0,256,2),np.float32)]
             for d in range(4)] for r in range(12)] for _ in range(6)]
    fixture=folder/'synthetic_source.pkl'
    with fixture.open('wb') as stream:
        pickle.dump({'data':data,'tx_list':list(range(6)),'rx_list':list(range(12)),
                     'capture_date_list':list(range(4)),'equalized_list':[1]},stream)
    del data
    row=matrix['rows'][-1]
    args=train.build_arg_parser().parse_args(v2_command(matrix,project_root=folder,run_root=folder/'entry',row=row)[3:])
    args.wisig_pkl=str(fixture); args.device=str(device); args.num_workers=0
    observations=[]; steps=[]
    original_objective=cross.labeled_cross_rx_objective
    original_step=torch.optim.AdamW.step
    class CheckedTrainingStop(Exception): pass
    def observed_objective(*pos,**kw):
        value,logs=original_objective(*pos,**kw)
        observations.append({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in logs.items()})
        return value,logs
    def observed_step(opt,*pos,**kw):
        gradients=[p.grad for group in opt.param_groups for p in group['params'] if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        value=original_step(opt,*pos,**kw)
        steps.append(True)
        if len(steps)==4: raise CheckedTrainingStop()
        return value
    reached=False
    with (folder/'actual_entry.log').open('x',encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), \
             patch.object(train,'load_checkpoint',side_effect=AssertionError('Unexpected checkpoint load')), \
             patch.object(train,'make_wisig_trainval_test_by_day_rx',
                          side_effect=AssertionError('Source screen constructed target loader') if getattr(args,'a1_source_screen_only',False) else None,
                          return_value=(None,None,None,{},{},{})), \
             patch.object(cross,'labeled_cross_rx_objective',side_effect=observed_objective), \
             patch.object(torch.optim.AdamW,'step',observed_step):
            try:
                train.train(args)
            except CheckedTrainingStop:
                reached=True
    assert reached and len(steps)==4
    if expect_cross_rx:
        assert observations and all(r['valid_anchors']>0 and r['weighted_loss']>0 for r in observations)
        assert all(r['leo_count']==0 for r in observations), 'E1 has no separately executed LEO student under the fixed course'
    else:
        assert not observations, 'Reference budget must keep cross-RX disabled'
    return {'status':'PASS','checkpoint_loads':0,'target_inputs':0,'initialization':'fresh_random',
        'paired_model_heads_rng':'EXACT','synthetic_gradient_checks':checks,
        'actual_train_entry_successful_steps':len(steps),'actual_train_entry_objectives':observations,
        'entry_scope':'actual matrix combined candidate E1 source loader; target-loader builder replaced with empty mapping; stop after four successful optimizer steps; no formal training result'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run_check(torch.device(args.device),args.output.with_suffix(''))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
