"""Fresh-model A1-V2 checks; synthetic source shapes only, no checkpoint inputs."""
import argparse
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SSDG import train_ssdg as train
from run_a1_fast_v2 import v2_matrix, v2_command
from cvsrffi.a1_fast_runtime import GradientSnapshot


def run_check(device):
    torch.set_num_threads(2)
    matrix = v2_matrix()
    initial = []
    with patch.object(train, 'load_checkpoint', side_effect=AssertionError('Unexpected checkpoint load')):
        for row in matrix['rows']:
            args = train.build_arg_parser().parse_args(v2_command(matrix, project_root=Path('/unused'),
                run_root=Path('/unused_new'), row=row)[3:])
            train._validate_daot_config(args)
            train._validate_a1_scratch_only(args)
            assert args.from_scratch and not args.use_a1_r3
            assert not args.baseline_ckpt and not args.teacher_ckpt
            torch.manual_seed(args.seed)
            merged = train._apply_model_cli_args(train.merge_checkpoint_args({}, args, input_len=256, num_domains=15), args)
            model = train.build_baseline_model(merged, device)
            state = train._initialize_muse_training_state(args, model, device)
            initial.append((model, state, args, torch.get_rng_state().clone()))
        model, state, args, rng = initial[0]
        for other, other_state, _, other_rng in initial[1:]:
            for a, b in ((model, other), (state['heads'], other_state['heads'])):
                assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
            assert torch.equal(rng, other_rng)
        del initial
        ema = deepcopy(model).eval()
        for p in ema.parameters(): p.requires_grad = False
        n = 60
        domains = torch.arange(n) % 15
        receiver = torch.tensor([1,3,4,6,8])[domains // 3]
        batch = (torch.randn(n,2,256), torch.arange(n) % 6, domains, {'rx_i':receiver})
        calibration = train._calibrate_rc4_vcal(None, ema, [batch], args=args,
            domain_label_map={i:i for i in range(15)}, device=device, num_classes=6, num_domains=15)
        checks = []
        timing = {}
        # Preserve the actual combined weak batch shape (2*U=512).
        for amp in ([False, True] if device.type == 'cuda' else [False]):
            x = torch.randn(512,2,256,device=device)
            d = torch.arange(512,device=device) % 15
            buffers = deepcopy(dict(ema.named_buffers()))
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
            outputs = []
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                for mode in ('legacy', 'identity_sequential'):
                    for _ in range(2):
                        train._forward_daot_teacher_views(ema,[x],domain_labels=d,efficiency_mode=mode)
                    if device.type == 'cuda': torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(3):
                        out = train._forward_daot_teacher_views(ema,[x],domain_labels=d,efficiency_mode=mode)[0][0]
                    if device.type == 'cuda': torch.cuda.synchronize()
                    timing[f'{mode}_amp{amp}_b512_ms'] = (time.perf_counter()-start)*1000/3
                    outputs.append(out)
            for key in ('z_id','tx_logits'):
                torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
            assert torch.equal(rng, torch.get_rng_state())
            if cuda_rng is not None: assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
            for name, value in ema.named_buffers(): assert torch.equal(value, buffers[name])
            routes = []
            for fast, out in zip((False,True), outputs):
                a,b = train._split_muse_output(out,256,512)
                route = train.route_fasttrust_rc4(a['tx_logits'], a['tx_logits'], b['tx_logits'],
                    domains=d[:256], receivers=torch.tensor([1,3,4,6,8],device=device)[d[:256]//3],
                    z_norm=a['z_id'].float().norm(dim=-1), calibration=calibration,
                    total_identity_effective_budget=float(args.rc4_total_identity_effective_budget),
                    use_calibrated_partial_threshold=True, enable_negative=False, batched_readback=fast)
                routes.append(route)
            for field in fields(routes[0]):
                assert torch.equal(getattr(routes[0],field.name), getattr(routes[1],field.name)), field.name
            checks.append({'amp':amp,'batch':512,'teacher_route_buffers_rng':'EXACT'})
        optimizer = torch.optim.AdamW(train._optimizer_parameters(model,state),lr=float(args.lr))
        steps = []
        for epoch in (1,21,61,161):
            train._configure_muse_epoch_state(state,epoch)
            x = torch.randn(8,2,256,device=device); d = torch.arange(8,device=device)%15
            with torch.no_grad():
                weak = train._forward_daot_teacher_views(ema,[x],domain_labels=d,efficiency_mode='identity_sequential')[0][0]
                route = train.route_fasttrust_rc4(weak['tx_logits'],weak['tx_logits'],weak['tx_logits'],
                    domains=d,receivers=torch.tensor([1,3,4,6,8],device=device)[d//3],
                    z_norm=weak['z_id'].norm(dim=-1),calibration=calibration,
                    total_identity_effective_budget=float(args.rc4_total_identity_effective_budget),
                    use_calibrated_partial_threshold=True,enable_negative=False,batched_readback=True)
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                student = model(x,domain_labels=d,return_aux=True)
                losses = train._compute_rc4_unlabeled_losses(route=route,ema_outputs=weak,anchor_outputs=None,
                    student_views={'clean':student,'satellite':None,'satellite_indices':torch.empty(0,dtype=torch.long,device=device)},
                    domains=d,model=model,muse_state=state,epoch=epoch)
                loss = losses['total'] + torch.nn.functional.cross_entropy(student['tx_logits'],torch.arange(8,device=device)%6)
            assert torch.isfinite(loss) and losses['feature_anchor'].item()==0
            loss.backward()
            snapshot = GradientSnapshot.capture(model)
            assert snapshot.first_nonfinite() is None
            assert snapshot.norm() == train._grad_norm(model)
            optimizer.step()
            train._update_ema_model(ema,model,.99,versioned=True)
            steps.append({'epoch':epoch,'loss':float(loss.detach()),'finite':True,'feature_anchor':0.0})
    return {'status':'PASS','initialization':'fresh_random','checkpoint_loads':0,'target_inputs':0,
            'paired_initial_parameters_and_rng':'EXACT','model_variant':'lite_d',
            'teacher_checks':checks,'rc4_steps':steps,'teacher_timing_ms':timing,
            'timing_scope':'teacher only; end-to-end improvement requires training measurements'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run_check(torch.device(args.device))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
