"""Replay frozen WiSig DURA with restoration active for every satellite observation."""

import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main import publish as transport
from . import wisig_dura_estimated as parent


RUN = '20261125-phase1-wisig-dura-force-main-m2-r01'
evaluation = parent.evaluation


def dispatch(rt):
    root,frozen = Path(rt['run_root']),Path(rt['frozen_root'])
    if read(frozen/'state.json')['status']!='ANALYZED':
        raise ValueError('Expected the original frozen Main comparison and paired baselines')
    root.mkdir()
    (root/'logs').mkdir()
    state = root/'state.json'
    try:
        allowed,load = parent.gpu_selection(minimum=0)
        rt = dict(rt,allowed_gpus=allowed,max_workers=4,max_compute_processes=None)
        write(root/'gpu_selection.json',dict(allowed=allowed,samples=load,maximum_workers=4))
        write(state,dict(status='FROZEN_CHECKPOINT_SMOKE'))
        schedule(rt,[('smoke','forced')],parent.launch)
        write(state,dict(status='TARGET_PREDICTING',force_practical=True),exclusive=False)
        schedule(rt,[('predict',case) for case in evaluation.CASES],parent.launch)
        write(state,dict(status='PREDICTIONS_FROZEN',force_practical=True),exclusive=False)
        parent.complete(parent.launch(rt,'score','cpu'),'independent truth-last scoring')
        write(state,dict(status='ANALYZED',force_practical=True),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def publish(repo,output,context,ssh_config):
    cfg = read(context)
    rt = {key:cfg[key] for key in ('project','python','frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project/'runs'
    frozen = runs/parent.RUN
    paths = dict(source_root=str(runs/'20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'/'cache'),
                 parent_root=str(runs/'20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'),
                 baseline_root=str(runs/evaluation.BASELINE),recognizer_run=str(runs/evaluation.F_RUN),
                 strong_view_root=str(runs/evaluation.RUN/'strong_target_iq'),
                 frozen_root=str(frozen),decision_replay=str(frozen/'predictions'))
    rt.update(run_id=RUN,runner=__spec__.name,code_commit=transport.release_commit(repo),
              run_root=str(runs/RUN),release_root=str(project/'releases'/f'{RUN}-release'),
              external_releases=list(paths.values()),**paths)
    extras = ('tools/wisig_paired_main_uncertainty.py','tools/r17_activation_factorization.py')
    transport.release(repo,output,rt,ssh_config,archive_extras=extras)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    release = sub.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        release.add_argument('--'+field,type=Path,required=True)
    for command in ('dispatch','smoke','predict','score'):
        stage = sub.add_parser(command)
        stage.add_argument('runtime',type=Path)
        stage.add_argument('name',nargs='?')
    args = parser.parse_args()
    if args.action=='publish':
        publish(args.repo,args.output,args.context,args.ssh_config)
        return
    rt = read(args.runtime)
    if args.action=='dispatch':
        dispatch(rt)
    elif args.action=='smoke':
        evaluation.smoke(rt,arms=parent.ARMS,load=parent.frozen_fit,force_practical=True)
    elif args.action=='predict':
        evaluation.predict(rt,args.name,methods=parent.METHODS,load=parent.frozen_fit,force_practical=True)
    else:
        from .wisig_dura_stress_score import score

        score(rt,methods=parent.METHODS,force_practical=True)


if __name__=='__main__':
    main()
