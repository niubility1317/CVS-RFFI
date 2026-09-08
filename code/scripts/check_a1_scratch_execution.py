"""Source-shaped synthetic checks for the real scratch/EMA/RC4 path."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from SSDG import train_ssdg as train
from run_a1_scratch_no_checkpoint import scratch_matrix, scratch_command, RELEASE


def run_check(device):
    torch.set_num_threads(2)
    matrix=scratch_matrix()
    initial=[]
    for row in matrix['rows']:
        args=train.build_arg_parser().parse_args(scratch_command(matrix,project_root=Path('/unused'),
                                    run_root=Path('/unused_new'),row=row)[3:])
        train._validate_daot_config(args); train._validate_a1_scratch_only(args)
        torch.manual_seed(args.seed)
        if device.type=='cuda': torch.cuda.manual_seed_all(args.seed)
        merged=train._apply_model_cli_args(train.merge_checkpoint_args({},args,input_len=256,num_domains=15),args)
        model=train.build_baseline_model(merged,device)
        state=train._initialize_muse_training_state(args,model,device)
        initial.append((model,state,args,torch.get_rng_state().clone()))
    first=initial[0]
    for second in initial[1:]:
        for scope in ('model','heads'):
            a=first[0] if scope=='model' else first[1]['heads']
            b=second[0] if scope=='model' else second[1]['heads']
            assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        assert torch.equal(first[3],second[3])
    model,state,args,_=first
    ema=deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad=False
    n=60
    domains=torch.arange(n)%15
    receiver=torch.tensor([1,3,4,6,8])[domains//3]
    batch=(torch.randn(n,2,256),torch.arange(n)%6,domains,{'rx_i':receiver})
    # Any implicit baseline/teacher load in the exercised path must fail.
    with patch.object(train,'load_checkpoint',side_effect=AssertionError('Unexpected checkpoint load')):
        calibration=train._calibrate_rc4_vcal(None,ema,[batch],args=args,
            domain_label_map={i:i for i in range(15)},device=device,num_classes=6,num_domains=15)
        optimizer=torch.optim.AdamW(train._optimizer_parameters(model,state),lr=float(args.lr))
        steps=[]
        for epoch in (1,21):
            train._configure_muse_epoch_state(state,epoch)
            x=torch.randn(8,2,256,device=device); d=torch.arange(8,device=device)%15
            rx=torch.tensor([1,3,4,6,8],device=device)[d//3]
            with torch.no_grad():
                weak=ema(x,domain_labels=d,return_aux=True)
                weak2=ema(x+0.001*torch.randn_like(x),domain_labels=d,return_aux=True)
                route=train.route_fasttrust_rc4(weak['tx_logits'],weak['tx_logits'],weak2['tx_logits'],
                    domains=d,receivers=rx,z_norm=weak['z_id'].norm(dim=-1),calibration=calibration,
                    total_identity_effective_budget=float(args.rc4_total_identity_effective_budget),
                    use_calibrated_partial_threshold=True,enable_negative=False)
            model.train(); optimizer.zero_grad(set_to_none=True)
            student=model(x,domain_labels=d,return_aux=True)
            losses=train._compute_rc4_unlabeled_losses(route=route,ema_outputs=weak,anchor_outputs=None,
                student_views={'clean':student,'satellite':None,'satellite_indices':torch.empty(0,dtype=torch.long,device=device)},
                domains=d,model=model,muse_state=state,epoch=epoch)
            assert torch.isfinite(losses['total'])
            assert losses['feature_anchor'].item()==0.0
            if epoch==1:
                assert losses['rc4_gradient_losses']['hard'].item()==0.0
                assert losses['rc4_gradient_losses']['partial_set'].item()==0.0
            losses['total'].backward()
            grads=[p.grad for p in train._optimizer_parameters(model,state) if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            optimizer.step()
            steps.append({'epoch':epoch,'loss':float(losses['total'].detach()),'gradients_finite':True,
                          'feature_anchor':0.0,'identity_enabled':epoch>=21})
    from cvsrffi.a1_r3_objective import r3_pair_objective
    args.a1_r3_aux_scale=1.
    amp_steps=[]
    for epoch,step in ((1,2),(90,2),(91,3),(161,2)):
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            x=torch.randn(8,2,256,device=device)
            supervised=model(x,y_tx=torch.arange(8,device=device)%6,return_aux=True)
            auxiliary,logs=r3_pair_objective(model=model,clean_iq=x,domains=d,
                physical_ids=tuple(str(i) for i in range(8)),role='U_s',args=args,
                epoch=epoch,batch_idx=2,optimizer_step=step,apply_sat_fn=train.apply_sat_channel_for_scenario)
            loss=torch.nn.functional.cross_entropy(supervised['tx_logits'],torch.arange(8,device=device)%6)+auxiliary
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        amp_steps.append({'epoch':epoch,'optimizer_step':step,'amp':device.type=='cuda',
            'loss':float(loss.detach()),'r3_loss':float(auxiliary.detach()),'finite':True})
    return {'status':'PASS','checkpoint_loads':0,'initialization':'fresh_random','r3_amp_steps':amp_steps,
            'paired_initial_parameters_and_cpu_rng_equal':True,'model_variant':'lite_d',
            'calibration_anchor':'EMA weak1 in existing RC4 no-anchor formula',
            'source_shaped_synthetic_no_query':True,'rc4_steps':steps}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    result=run_check(torch.device(args.device))
    execution_path=args.output.with_name(args.output.stem+'_daot.json')
    subprocess.run([sys.executable,str(RELEASE/'code/scripts/check_a1_fast_execution.py'),
                    '--model-variant','lite_d','--use-a1-r3','--device',args.device,'--output',str(execution_path)],check=True)
    result['daot_execution']=json.loads(execution_path.read_text(encoding='utf-8'))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
