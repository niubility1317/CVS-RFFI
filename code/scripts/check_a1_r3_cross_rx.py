"""Exercise combined R3/cross-RX gradients and the real source training entry."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from SSDG import train_ssdg as train
from cvsrffi.a1_ecrs_cross_rx import labeled_cross_rx_objective
from cvsrffi import a1_r3_objective as r3
from run_a1_fast_v2 import v2_command
from run_a1_r3_cross_rx import r3_cross_rx_matrix
from check_a1_ecrs_cross_rx import run_check


def combined_checks(device):
    torch.set_num_threads(2)
    matrix=r3_cross_rx_matrix(); row=matrix['rows'][0]
    args=train.build_arg_parser().parse_args(v2_command(matrix,project_root=Path('/unused'),run_root=Path('/new'),row=row)[3:])
    train._validate_a1_scratch_only(args);train._validate_a1_ecrs_config(args)
    torch.manual_seed(args.seed)
    cfg=train._apply_model_cli_args(train.merge_checkpoint_args({},args,input_len=256,num_domains=15),args)
    model=train.build_baseline_model(cfg,device)
    x=torch.randn(12,2,256,device=device); labels=torch.arange(12,device=device)%6
    rx=torch.arange(12,device=device)//6; day=torch.ones_like(rx)
    # Capture the real fingerprint encoder output; the loss must see that
    # same object rather than the raw identity trunk or a detached copy.
    captured=[]
    hook=model.a1_r3.fingerprint.register_forward_hook(lambda module,pos,out:captured.append(out.z_f_id))
    model.eval();out=model(x,return_aux=True);hook.remove()
    assert out['z_id'] is captured[-1]
    direct,_=labeled_cross_rx_objective(out['z_id'],labels,rx,day,scope='clean',weight=.05,margin=.2)
    model.zero_grad(set_to_none=True);direct.backward()
    def grad_norm(prefix):
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        value=sum(float(g.detach().float().square().sum()) for g in grads)**.5
        assert value>0;return value
    cross_grad={'identity_backbone':grad_norm('id_backbone.'),'fingerprint':grad_norm('a1_r3.fingerprint.')}
    assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(('dom_backbone.','a1_r3.nuisance.','a1_r3.decoder.')))
    stages=[]
    for epoch,step in [(1,0),(41,0),(65,0),(92,2),(92,3),(161,0)]:
        for amp in ([False,True] if device.type=='cuda' else [False]):
            model.train();model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                output=model(x,y_tx=labels,return_aux=True)
                cross,stats=labeled_cross_rx_objective(output['z_id'],labels,rx,day,scope='clean',weight=.05,margin=.2)
                rloss,rlogs=r3.r3_pair_objective(model=model,clean_iq=x,domains=rx,
                    physical_ids=[f'source:{i}' for i in range(len(x))],role='L_s',
                    args=args,epoch=epoch,batch_idx=1,optimizer_step=step,
                    apply_sat_fn=train.apply_sat_channel_for_scenario)
                total=torch.nn.functional.cross_entropy(output['tx_logits'],labels)+cross+rloss
            assert torch.isfinite(total) and stats['valid_anchors']==12 and stats['leo_count']==0
            total.backward();norm=grad_norm('id_backbone.')
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            stages.append({'epoch':epoch,'optimizer_step':step,'amp':amp,'cross_rx':float(cross.detach()),
                'r3':float(rloss.detach()),'identity_grad_norm':norm,
                'r3_logs':{k:float(v) for k,v in rlogs.items()}})
    return {'identity_output':'EXACT_SAME_TENSOR_AS_z_f_id','cross_only_gradients':cross_grad,'stages':stages}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    device=torch.device(args.device);details=combined_checks(device)
    calls=[];original=r3.r3_pair_objective
    def observed(*pos,**kw):
        loss,logs=original(*pos,**kw)
        calls.append({'role':kw['role'],'epoch':kw['epoch'],'loss':float(loss.detach())})
        return loss,logs
    with patch.object(r3,'r3_pair_objective',side_effect=observed):
        result=run_check(device,args.output.with_suffix(''),matrix_factory=r3_cross_rx_matrix)
    assert {c['role'] for c in calls}=={'L_s','U_s'} and all(c['loss']>0 for c in calls)
    result.update(combined_checks=details,actual_entry_r3_calls=calls,
        candidate='R3_REFERENCE_CLEAN_CROSS_RX',final_evaluation='source_only')
    result['entry_scope']='actual R3 reference plus clean cross-RX E1; synthetic source only; four successful optimizer steps; no formal performance result'
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'stage_checks':len(details['stages']),
        'identity_output':details['identity_output'],'entry_steps':result['actual_train_entry_successful_steps'],
        'r3_roles':sorted({c['role'] for c in calls}),'checkpoint_loads':0,'target_inputs':0}),flush=True)


if __name__=='__main__':main()
