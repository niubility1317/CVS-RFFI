"""One directly trained R69-size student from the frozen Adaptive R17 teacher."""

import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison import g_lineage, g_train
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_iq_joint_compact.models import transplant_core
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_repl_view_shuffle.runner import activate
from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
from experiments.pamf_stage2_inference.engine import InferenceEngine

from . import wisig_dura_estimated as workers
from . import wisig_dura_stress as evaluation
from . import wisig_original_dura as replay


RUN = '20261126-phase1-wisig-r17-direct-r69-adaptive-m1-r01'
METHODS = ('raw', 'R69', 'trained_direct_distill')
SOURCE_KEYS = ('run_root', 'code_commit', 'action_iq', 'teacher_root')
PARAMETERS = 266951


def teacher_from(rt):
    parent = Path(rt['teacher_root'])
    folder = parent / 'G/r17'
    configuration = read(parent / 'G/config.json')
    source = read(parent / 'inputs/source_contract.json')
    proof = read(folder / 'complete.json')
    if (parent.name != replay.OPERATOR_RUN or configuration['action_supervision'] != 'operator'
            or source['historical_received_epochs'] != str(Path(rt['action_iq']) / 'epochs')
            or read(parent / 'source_smoke.json')['simulator_supervision_reads'] != 0
            or (proof['status'], proof['stage'], proof['updates']) != ('SOURCE_FROZEN', 'r17', 4940)):
        raise ValueError('Expected the fixed source-only Adaptive R17 teacher')
    teacher, payload = g_lineage.load(folder / 'final.pt', source, 'LU')
    expected = [str(parent / 'G' / k / 'final.pt') for k in ('coarse','action')]
    if (payload['stage'] != 'r17' or payload['ancestors'] != expected
            or payload['model_seed'] != replay.SEED or teacher.candidate != 'unroll_unshared'):
        raise ValueError('Wrong R17 teacher or ancestry')
    return teacher.requires_grad_(False), folder / 'final.pt', source


def initialize(teacher):
    """Slice the untouched R17 weights straight into the native two-stage topology."""
    student = StageDistilled()
    old = teacher.frozen
    new = student.frozen
    maps = {}
    for name, source, target in (
        ('coarse_spectral', old.coarse.spectral, new.coarse.spectral),
        ('coarse_temporal', old.coarse.temporal, new.coarse.temporal),
        ('action', old.estimator.core, new.estimator.core),
    ):
        maps[name] = transplant_core(source, target)
    with torch.no_grad():
        new.estimator.global_head.weight.copy_(old.estimator.global_head.weight[:, maps['action'][-1]])
        new.estimator.global_head.bias.copy_(old.estimator.global_head.bias)
        new.estimator.initial_kernel.copy_(old.estimator.initial_kernel)
    # Copy each prior independently before averaging the shared long convolutions.
    priors = [deepcopy(core) for core in student.prior]
    for index, (target, source_index) in enumerate(zip(priors, student.STAGE_TARGETS, strict=True)):
        maps[f'prior{index}'] = transplant_core(teacher.prior[source_index], target)
    student.prior = torch.nn.ModuleList(share_long_weights(priors))
    if sum(p.numel() for p in student.parameters()) != PARAMETERS:
        raise ValueError('Direct distillation must have the original R69 parameter count')
    student.initialization_report = dict(mode='direct_equal_GN_group_R17_slices', parameters=PARAMETERS,
        teacher_steps=list(student.STAGE_TARGETS), CG_steps=4, optimizer_updates=0, IQ_examples=0,
        channel_maps={k:[q.tolist() for q in v] for k,v in maps.items()})
    return student


def source_models(rt):
    replay.setup_torch()
    torch.manual_seed(replay.SEED)
    torch.cuda.manual_seed_all(replay.SEED)
    teacher, path, contract = teacher_from(rt)
    student = initialize(teacher).cuda()
    return student, teacher.cuda(), path, contract


def smoke(rt):
    student, teacher, path, _ = source_models(rt)
    source = Path(rt['action_iq'])
    x = tensor(replay.targets.load(source/'train','clean')[:4], 'cuda:0')
    y = tensor(replay.targets.received(source, 1)[:4], 'cuda:0')
    loss = g_lineage.objective(student, teacher, 'r69', y, x, 1, original_recipe=True).mean()
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
    if not torch.isfinite(loss) or not float(gradient) > 0. or any(p.grad is not None for p in teacher.parameters()):
        raise FloatingPointError('Direct distillation smoke has invalid student or teacher gradients')
    student.eval().requires_grad_(False)
    torch.testing.assert_close(InferenceEngine(student, reuse=True, graph=True, native=False)(y)[0],
                              student(y), rtol=5e-5, atol=5e-6)
    write(Path(rt['run_root'])/'source_smoke.json',dict(status='PASS',teacher=str(path),parameters=PARAMETERS,
          source_packets=len(y),loss=float(loss.detach()),gradient_norm=float(gradient),target_reads=0))


def fit(rt):
    model, teacher, teacher_path, contract = source_models(rt)
    root = Path(rt['run_root'])
    folder = root/'student'
    folder.mkdir()
    view = replay.OriginalSourceView(dict(rt,action_supervision='operator'))
    if not view.ready() or len(view) != 63000:
        raise ValueError('Original R69 source training view is required')
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    rng = np.random.default_rng(2026102001)
    started, updates, seen_l, seen_u = perf_counter(), 0, 0, 0
    with (folder/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,21):
            model.train()
            total = 0.
            for selected in g_train.batches(len(view),rng):
                wave = view.batch(selected,epoch,coarse=False,action=False)
                clean, observed = (tensor(wave[key],'cuda:0') for key in ('clean','received'))
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = g_train.learning_rate(updates)
                loss = g_lineage.objective(model,teacher,'r69',observed,clean,epoch,original_recipe=True)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError('Nonfinite direct R17 to R69 loss')
                loss.mean().backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                total += float(loss.detach().sum())
                labeled = int(view.is_L[selected].sum())
                seen_l += labeled
                seen_u += len(selected)-labeled
                if updates == 1:
                    first = dict(updates=updates, loss=float(loss.detach().mean()), gradient_norm=float(gradient))
                    write(folder/'first_update.json',first)
                    print('DIRECT_DISTILL_FIRST_UPDATE', first, flush=True)
            row = dict(epoch=epoch,updates=updates,loss=total/len(view),seconds=perf_counter()-started,
                       L_exposures=seen_l,U_exposures=seen_u)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            log.flush()
            write(root/'progress.json',row,exclusive=False)
            print('DIRECT_DISTILL_EPOCH', row, flush=True)
    if (updates,seen_l,seen_u) != (4940,126000,1134000) or any(p.grad is not None for p in teacher.parameters()):
        raise ValueError('Incorrect one-stage budget or teacher gradients')
    path = folder/'final.pt'
    torch.save(dict(schema='contract_matched_dura_stage_v1',stage='r69',source_contract=contract,allowed_roles='LU',
        model_seed=replay.SEED,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        initialization_report=model.initialization_report,ancestors=[str(teacher_path)],epoch=20,updates=updates,
        L_exposures=seen_l,U_exposures=seen_u,TX_labels_read=0,F_reads=0,target_reads=0,code_commit=rt['code_commit']),path)
    fresh, _ = g_lineage.load(path,contract,'LU','cuda:0')
    with torch.no_grad():
        torch.testing.assert_close(model.eval()(observed[:2]),fresh(observed[:2]),rtol=1e-5,atol=1e-6)
    write(folder/'complete.json',dict(status='SOURCE_FROZEN',epoch=20,updates=updates,training_stages=1,
          teacher=str(teacher_path),parameters=PARAMETERS,seconds=perf_counter()-started,contract=contract,
          TX_labels_read=0,F_reads=0,target_reads=0))


def frozen_fit(root, arm):
    if arm != 'direct_distill':
        raise ValueError('Expected the one-stage student')
    folder = Path(root)/'student'
    marker = read(folder/'complete.json')
    if (marker['status'],marker['updates'],marker['parameters']) != ('SOURCE_FROZEN',4940,PARAMETERS):
        raise ValueError('Expected the complete one-stage student checkpoint')
    model, payload = g_lineage.load(folder/'final.pt',marker['contract'],'LU','cuda:0')
    if payload['ancestors'] != [marker['teacher']] or len(model.prior) != 2:
        raise ValueError('Unexpected student provenance or stage count')
    return InferenceEngine(model.requires_grad_(False),reuse=True,graph=True,native=False), None


def readout_smoke(rt):
    activate(rt)
    replay.setup_torch()
    engine, _ = frozen_fit(rt['run_root'],'direct_distill')
    _, heads = evaluation.readers(rt)
    iq = tensor(replay.targets.received(Path(rt['action_iq']),1)[:4],'cuda:0')
    decisions = evaluation.classify(heads,engine(iq)[0])
    if len(decisions) != 3 or any(len(row)!=len(iq) for row in decisions):
        raise ValueError('Expected the three frozen Main readers')
    write(Path(rt['run_root'])/'evaluation_smoke.json',dict(status='PASS',source_packets=len(iq),target_reads=0))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root/'logs').mkdir()
    state = root/'state.json'
    try:
        write(root/'source_runtime.json',{k:rt[k] for k in SOURCE_KEYS})
        allowed, _ = workers.gpu_selection(minimum=0)
        one = dict(rt,allowed_gpus=allowed,max_workers=1,max_compute_processes=None)
        write(state,dict(status='SOURCE_SMOKE'))
        schedule(one,[('fit-smoke','direct')],workers.launch)
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        schedule(one,[('fit','direct')],workers.launch)
        write(state,dict(status='FROZEN_CHECKPOINT_SMOKE'),exclusive=False)
        schedule(one,[('smoke','readout')],workers.launch)
        write(state,dict(status='TARGET_PREDICTING'),exclusive=False)
        schedule(dict(one,max_workers=2),[('predict',case) for case in evaluation.CASES],workers.launch)
        write(state,dict(status='PREDICTIONS_FROZEN'),exclusive=False)
        workers.complete(workers.launch(rt,'score','cpu'),'independent truth-last paired scoring')
        write(state,dict(status='ANALYZED'),exclusive=False)
    except Exception as e:
        write(state,dict(status='FAILED',error=str(e)),exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    settings = read(context)
    rt = {k:settings[k] for k in ('project','python','frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project/'runs'
    paths = dict(action_iq=str(runs/replay.action_design.RUN/'cache'),teacher_root=str(runs/replay.OPERATOR_RUN),
                 baseline_root=str(runs/evaluation.BASELINE),recognizer_run=str(runs/evaluation.F_RUN),
                 strong_view_root=str(runs/evaluation.RUN/'strong_target_iq'),
                 decision_replay=str(runs/replay.baseline.RUN/'predictions'))
    external = [*paths.values(),str(runs/replay.OPERATOR_RUN/'G/r17/final.pt'),
                str(runs/replay.OPERATOR_RUN/'predictions')]
    rt.update(run_id=RUN,runner=__spec__.name,code_commit=release_commit(repo),run_root=str(runs/RUN),
              release_root=str(project/'releases'/f'{RUN}-release'),external_releases=external,**paths)
    release(repo,output,rt,ssh_config,archive_extras=(
        'tools/wisig_paired_main_uncertainty.py','tools/r17_activation_factorization.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    jobs = parser.add_subparsers(dest='action',required=True)
    pub = jobs.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    commands = {'dispatch':dispatch,'fit-smoke':smoke,'fit':fit,'smoke':readout_smoke}
    for key in (*commands,'predict','score'):
        worker = jobs.add_parser(key)
        worker.add_argument('runtime',type=Path)
        worker.add_argument('name',nargs='?')
    args = parser.parse_args()
    if args.action == 'publish':
        return publish(args.repo,args.output,args.context,args.ssh_config)
    rt = read(args.runtime)
    if args.action == 'predict':
        return evaluation.predict(rt,args.name,methods=METHODS,load=frozen_fit,force_practical=True,
                                  restore=replay.restore,clean_passthrough=False)
    if args.action == 'score':
        from .wisig_dura_stress_score import score

        peer = ('adaptive_multistage',Path(rt['teacher_root'])/'predictions','trained_adaptive_operator')
        return score(rt,methods=METHODS,force_practical=True,clean_passthrough=False,reference=peer)
    return commands[args.action](rt)


if __name__ == '__main__':
    main()
