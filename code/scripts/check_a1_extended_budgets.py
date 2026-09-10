"""Actual source-only entry and late-clock gradient checks for each extension."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_a1_extended_budgets import extended_matrix
from run_a1_fast_v2 import v2_command
from check_a1_ecrs_cross_rx import run_check
from SSDG import train_ssdg as train
from cvsrffi.a1_budget_schedule import reference_epoch
from cvsrffi.a1_periodic_target import due
from cvsrffi.a1_r3_objective import r3_pair_objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    matrix = extended_matrix(); results = []; device = torch.device(cli.device)
    for row in matrix['rows']:
        one = dict(matrix, rows=[row])
        result = run_check(device, cli.output.with_suffix('')/row['id'], matrix_factory=lambda: one)
        args = train.build_arg_parser().parse_args(v2_command(matrix,
            project_root=Path('/p'),run_root=Path('/r'),row=row)[3:])
        cfg = train._apply_model_cli_args(train.merge_checkpoint_args({},args,input_len=256,num_domains=15),args)
        model = train.build_baseline_model(cfg,device)
        optimizer = torch.optim.AdamW([{'params': model.parameters(), 'fasttrust_role':'backbone'}],lr=args.lr)
        late = []
        for epoch in (201, args.epochs//2+1, args.epochs-1, args.epochs):
            ref = reference_epoch(args,epoch)
            train._apply_fasttrust_lr(optimizer,base_lr=args.lr,epoch=ref,tail_mode=args.a1_tail_lr)
            assert optimizer.param_groups[0]['lr'] > 0
            model.zero_grad(set_to_none=True)
            x = torch.randn(12,2,256,device=device)
            out = model(x,return_aux=True)
            loss = out['tx_logits'].float().square().mean()
            if args.use_a1_r3:
                extra, _ = r3_pair_objective(model=model,clean_iq=x,
                    domains=torch.arange(12,device=device)//6, physical_ids=[f'source:{i}' for i in range(12)],
                    role='L_s',args=args,epoch=epoch,batch_idx=1,optimizer_step=2,
                    apply_sat_fn=train.apply_sat_channel_for_scenario)
                loss = loss + extra
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            optimizer.step()
            late.append({'epoch':epoch,'reference_epoch':ref,'lr':optimizer.param_groups[0]['lr'],'loss':float(loss.detach())})
        assert [e for e in range(1,args.epochs+1) if due(e,args.a1_periodic_target_start,args.epochs,args.a1_periodic_target_interval)] == row['evaluation_epochs']
        result.update(row=row['id'], late_checks=late, evaluation_epochs=row['evaluation_epochs'])
        results.append(result)
    cli.output.write_text(json.dumps({'status':'PASS','rows':results},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','rows':[r['row'] for r in results],'entry_steps':[r['actual_train_entry_successful_steps'] for r in results]}))


if __name__ == '__main__': main()
