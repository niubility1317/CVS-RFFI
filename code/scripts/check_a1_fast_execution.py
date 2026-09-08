"""No-query execution parity and teacher timing with fresh or this-run weights."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import SSDG.train_ssdg as train
from post_stage_common import build_baseline_model, merge_checkpoint_args
from cvsrffi.orbit_teacher import EMALossScaleNormalizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--model-variant',default='lite_c')
    p.add_argument('--use-a1-r3',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--include-batched',action='store_true',help='Separate numerical experiment; excluded from strict first-round parity')
    cli=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(392005)
    device=torch.device(cli.device)
    args=train.build_arg_parser().parse_args(['--output_dir','unused','--daot_ablation','A1','--num_classes','6','--model_variant',cli.model_variant])
    train._validate_daot_config(args)
    args.use_a1_r3=cli.use_a1_r3
    args.daot_diagnostic_epochs=''
    model_args=train._apply_model_cli_args(merge_checkpoint_args({},args,input_len=256,num_domains=15),args)
    template=build_baseline_model(model_args,device)
    if cli.checkpoint is not None:
        ckpt=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
        if int(ckpt.get('epoch',-1))!=200 or not ckpt.get('args',{}).get('from_scratch'):
            raise ValueError('Expected an E200 checkpoint originally trained from scratch')
        loaded=template.load_state_dict(ckpt['model'],strict=False)
        # A1 adds this diagnostic head to the selected base checkpoint.
        if loaded.unexpected_keys or any(not k.startswith('daot_nuisance_head.') for k in loaded.missing_keys):
            raise ValueError(f'Incomplete backbone load: {loaded}')
        for key,value in ckpt['model'].items():
            if not torch.equal(template.state_dict()[key].cpu(),value.cpu()):
                raise ValueError(f'Checkpoint tensor not loaded exactly: {key}')
        del ckpt
    teacher=deepcopy(template).eval()
    for param in teacher.parameters(): param.requires_grad=False
    inputs=[torch.randn(8,2,256,device=device),torch.randn(8,2,256,device=device)]
    domains=torch.arange(8,device=device)%15
    labels=torch.arange(8,device=device)%6
    result={'status':'PASS','input_role':'source_shaped_synthetic_no_query','target_inputs':0,
            'checkpoint':str(cli.checkpoint) if cli.checkpoint else None,
            'initialization':'supplied_E200_checkpoint' if cli.checkpoint else 'fresh_random',
            'new_a1_parameters':loaded.missing_keys if cli.checkpoint else [],
            'torch':torch.__version__,'device':str(device),'teacher_comparisons':[], 'step_comparisons':[],
            'timing_scope':'teacher-only microbenchmark; not end-to-end speedup'}
    modes=['legacy','identity_sequential'] + (['identity_batched'] if cli.include_batched else [])
    for amp in ([False,True] if device.type=='cuda' else [False]):
        outputs={}
        rng=torch.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
        buffers=deepcopy(dict(teacher.named_buffers()))
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=amp):
            for mode in modes:
                outputs[mode],_=train._forward_daot_teacher_views(teacher,inputs,domain_labels=domains,efficiency_mode=mode)
            tolerance=(3e-3,3e-3) if amp else (3e-5,3e-6)
            for mode in modes[1:]:
                errors={}
                for key in ['z_id','tx_logits']:
                    for a,b in zip(outputs['legacy'],outputs[mode]):
                        torch.testing.assert_close(a[key],b[key],rtol=tolerance[0],atol=tolerance[1])
                    errors[key]=max(float((a[key]-b[key]).abs().max()) for a,b in zip(outputs['legacy'],outputs[mode]))
                result['teacher_comparisons'].append({'mode':mode,'amp':amp,'max_abs_error':errors})
        assert torch.equal(rng,torch.get_rng_state())
        if cuda_rng is not None: assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
        for name,b in teacher.named_buffers(): torch.testing.assert_close(b,buffers[name],rtol=0,atol=0)
    # Compare actual L-then-U objective, model gradients, AdamW updates and BN state.
    for epoch in [1,21,161]:
        states=[]
        for mode in modes:
            torch.manual_seed(771)
            model=deepcopy(template).train()
            ema=deepcopy(teacher)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
            norm=EMALossScaleNormalizer(batched_readback=mode!='legacy')
            args.daot_efficiency_mode=mode; args.daot_skip_mean_metadata=mode!='legacy'
            clean=model(inputs[0],y_tx=labels,return_aux=True,domain_labels=domains)
            l=train._compute_daot_labeled_step(model=model,ema_model=ema,student_clean=clean,
                x_clean=inputs[0],y_clean=labels,d_clean=domains,args=args,epoch=epoch,batch_idx=2,
                apply_sat_fn=train.apply_sat_channel_for_scenario,prototype_matrix=None,loss_normalizer=norm)
            with torch.no_grad():
                teacher_clean=ema(inputs[1],y_tx=None,return_aux=True,domain_labels=domains)
            strong=model(inputs[1],y_tx=None,return_aux=True,domain_labels=domains)
            u=train._compute_daot_unlabeled_step(model=model,ema_model=ema,teacher_clean=teacher_clean,
                student_strong=strong,x_unlabeled=inputs[1],d_unlabeled=domains,args=args,epoch=epoch,batch_idx=2,
                apply_sat_fn=train.apply_sat_channel_for_scenario,prototype_matrix=None,loss_normalizer=norm)
            loss=l['loss']+u['loss']+torch.nn.functional.cross_entropy(clean['tx_logits'],labels)
            if cli.use_a1_r3:
                from cvsrffi.a1_r3_objective import r3_pair_objective
                args.a1_r3_aux_scale=1.
                for i,role in enumerate(('L_s','U_s')):
                    auxiliary,_=r3_pair_objective(model=model,clean_iq=inputs[i],domains=domains,
                        physical_ids=tuple(str(j) for j in range(8)),role=role,args=args,epoch=epoch,
                        batch_idx=2,optimizer_step=2,apply_sat_fn=train.apply_sat_channel_for_scenario)
                    loss=loss+auxiliary
            loss.backward()
            grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
            opt.step(); train._update_ema_model(ema,model,.99)
            states.append({'loss':loss.detach(),'grads':grads,'model':deepcopy(model.state_dict()),
                           'ema':deepcopy(ema.state_dict()),'scale':norm.state_dict(),'rng':torch.get_rng_state().clone(),
                           'mask_l':l['diagnostics'].get('consensus_mask'),'mask_u':u['diagnostics'].get('consensus_mask')})
            del model,ema,opt,clean,strong,l,u,loss
        for mode,state in zip(modes[1:],states[1:]):
            reference=states[0]
            torch.testing.assert_close(state['loss'],reference['loss'],rtol=5e-5,atol=5e-6)
            assert state['scale']==reference['scale'] if mode=='identity_sequential' else True
            assert torch.equal(state['rng'],reference['rng'])
            for key in ['grads','model','ema']:
                for name,value in state[key].items():
                    expected=reference[key][name]
                    if value is None or expected is None: assert value is expected
                    else: torch.testing.assert_close(value,expected,rtol=2e-4,atol=2e-5)
            for key in ['mask_l','mask_u']:
                if state[key] is not None: assert torch.equal(state[key],reference[key])
            result['step_comparisons'].append({'epoch':epoch,'mode':mode,'status':'PASS'})
        del states
    result['teacher_timing_ms']={}
    with torch.no_grad():
        for batch in [128,256]:
            views=[torch.randn(batch,2,256,device=device) for _ in range(2)]
            for mode in modes:
                for _ in range(2): train._forward_daot_teacher_views(teacher,views,domain_labels=None,efficiency_mode=mode)
                if device.type=='cuda': torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(4): train._forward_daot_teacher_views(teacher,views,domain_labels=None,efficiency_mode=mode)
                if device.type=='cuda': torch.cuda.synchronize()
                result['teacher_timing_ms'][f'{mode}_b{batch}']=(time.perf_counter()-start)*250
    cli.output.parent.mkdir(parents=True,exist_ok=True)
    cli.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
