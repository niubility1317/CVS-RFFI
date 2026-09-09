"""Validate both compressed budgets on actual scratch source training entrypoints."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np  # Initialize MKL before torch/libgomp on the N607 runtime.
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi import a1_r3_objective as r3
from cvsrffi.a1_budget_schedule import reference_epoch
from check_a1_ecrs_cross_rx import run_check
from run_a1_r3_budgets import budget_matrix
from run_a1_fast_v2 import v2_command
from SSDG import train_ssdg as train


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True);cli=parser.parse_args()
    device=torch.device(cli.device);matrix=budget_matrix();results=[]
    original=r3.r3_pair_objective
    for row in matrix['rows']:
        calls=[]
        def observed(*pos,**kw):
            loss,logs=original(*pos,**kw)
            calls.append({'role':kw['role'],'epoch':kw['epoch'],'loss':float(loss.detach())})
            return loss,logs
        # Keep both candidates in initialization comparison; train the selected last row.
        ordered=dict(matrix,rows=[r for r in matrix['rows'] if r is not row]+[row])
        with patch.object(r3,'r3_pair_objective',side_effect=observed):
            result=run_check(device,cli.output.with_suffix('')/row['id'],
                matrix_factory=lambda:ordered,expect_cross_rx=False)
        assert {c['role'] for c in calls}=={'L_s','U_s'} and all(c['loss']>0 for c in calls)
        args=train.build_arg_parser().parse_args(v2_command(matrix,project_root=Path('/unused'),run_root=Path('/new'),row=row)[3:])
        assert args.from_scratch and not args.baseline_ckpt and not args.teacher_ckpt
        assert args.stage1_epochs+1==args.muse_s2a_start
        assert args.stage2_epochs+1==args.muse_s3a_start
        assert args.source_episode_structural_warmup_epochs==-1
        assert reference_epoch(args,args.epochs)==200
        cfg=train._apply_model_cli_args(train.merge_checkpoint_args({},args,input_len=256,num_domains=15),args)
        model=train.build_baseline_model(cfg,device)
        x=torch.randn(12,2,256,device=device);domains=torch.arange(12,device=device)//6
        late=[]
        for epoch in [args.epochs//4+1,args.epochs//2+1,3*args.epochs//4+1,args.epochs]:
            for role in ('L_s','U_s'):
                model.zero_grad(set_to_none=True)
                loss,logs=original(model=model,clean_iq=x,domains=domains,
                    physical_ids=[f'source:{i}' for i in range(12)],role=role,args=args,
                    epoch=epoch,batch_idx=1,optimizer_step=2,apply_sat_fn=train.apply_sat_channel_for_scenario)
                assert torch.isfinite(loss) and loss>0
                loss.backward()
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                late.append({'epoch':epoch,'reference_epoch':reference_epoch(args,epoch),'role':role,'loss':float(loss.detach())})
        result.update(row=row['id'],actual_entry_r3_calls=calls,late_r3_checks=late)
        results.append(result)
    cli.output.write_text(json.dumps({'status':'PASS','budgets':results},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','rows':[r['row'] for r in results],'successful_steps':[r['actual_train_entry_successful_steps'] for r in results]}),flush=True)


if __name__=='__main__':main()
