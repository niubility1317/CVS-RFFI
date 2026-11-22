"""Publish the full LoRa continuations and seven-view predictor on N607."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_adaptive import PRACTICAL_CE_RUN, PRACTICAL_GAP_RUN, PRACTICAL_SET_RUN, RUN, SET_RUN, WARMUP_RUN
from .full_packet_publish import DATA_RUN
from .full_supervised import RUN as SUPERVISED
from .full_tri_repl_lora import ADAPT_RUN, REPL_RUN, WARM_RUN
from .full_restore_draw import RUN as DRAW_RUN
from .publish import release, release_commit


def publish(repo,output,context,ssh_config,data_submit,supervised_submit,prior_submit=None,warmup_submit=None,
            draw_submit=None,set_submit=None,native_repl=False,native_warm=False,native_adapt=False,
            practical_ce=False,practical_gap=False,practical_submit=None):
    ctx=read(context)
    rt={k:ctx[k] for k in ('project','python','frozen_sources')}
    p=PurePosixPath(rt['project'])
    dependencies=[(data_submit,DATA_RUN),(supervised_submit,SUPERVISED)]
    run=RUN
    if (native_repl or native_warm or native_adapt) and (warmup_submit is None or draw_submit is not None):
        raise ValueError('Native Tri requires the matched supervised continuation')
    if practical_ce or practical_gap:
        if warmup_submit is None or set_submit is None or draw_submit is not None or prior_submit is not None:
            raise ValueError('Practical CE requires the frozen E140 and clean-observation set references')
        dependencies.extend(((warmup_submit,WARMUP_RUN),(set_submit,SET_RUN)))
        run=PRACTICAL_CE_RUN
        if practical_gap:
            if practical_submit is None:
                raise ValueError('The source gap test needs the matched fixed-weight comparison')
            dependencies.append((practical_submit,PRACTICAL_CE_RUN))
            run=PRACTICAL_GAP_RUN
    elif draw_submit is not None:
        if warmup_submit is None or set_submit is None or prior_submit is not None:
            raise ValueError('Independent-draw Tri requires E140, candidate-set, and source draw receipts')
        dependencies.extend(((warmup_submit,WARMUP_RUN),(draw_submit,DRAW_RUN),(set_submit,SET_RUN)))
        run=PRACTICAL_SET_RUN
    elif warmup_submit is not None:
        dependencies.append((warmup_submit,WARMUP_RUN))
        if native_adapt:
            run=ADAPT_RUN
        else:
            run=WARM_RUN if native_warm else REPL_RUN if native_repl else SET_RUN
    elif prior_submit is not None:
        dependencies.append((prior_submit,RUN))
        run=WARMUP_RUN
    for path,name in dependencies:
        r=read(path)
        if (r['archive_transfer']!='VERIFIED' or PurePosixPath(r['run_root'])!=p/'runs'/name
                or PurePosixPath(r['cwd'])!=p/'releases'/f'{name}-release'):
            raise ValueError('Expected exact complete release handle for the full LoRa dependency')
    data=p/'runs'/DATA_RUN
    rt.update(run_id=run,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.full_adaptive',
              release_root=str(p/'releases'/f'{run}-release'),run_root=str(p/'runs'/run),
              parent_root=str(data),source_root=str(p/'releases'/f'{DATA_RUN}-release/data/source'),
              target_views=str(data/'target_views'),allowed_gpus=list(range(7)),max_compute_processes=4)
    if run in (WARMUP_RUN,SET_RUN,PRACTICAL_SET_RUN,REPL_RUN,WARM_RUN,ADAPT_RUN):
        rt['max_workers']=4
    if run==PRACTICAL_SET_RUN:
        rt['allowed_gpus']=[4,1,5,6,0,2,3,7]
    if run in (PRACTICAL_CE_RUN,PRACTICAL_GAP_RUN):
        rt.update(allowed_gpus=[1,4,5,0,6,2,3],max_workers=2)
    if run==REPL_RUN:
        rt['allowed_gpus']=[6,5,0,2,1,3,4,7]
    if run==WARM_RUN:
        rt['allowed_gpus']=[5,2,4,1,3,0,6,7]
    if run==ADAPT_RUN:
        rt['allowed_gpus']=[0,5,6,4,2,1,3,7]
    return release(repo,output,rt,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config','data-submit','supervised-submit'):
        p.add_argument('--'+k,type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prior-submit',type=Path)
    mode.add_argument('--warmup-submit',type=Path)
    p.add_argument('--draw-submit',type=Path)
    p.add_argument('--set-submit',type=Path)
    p.add_argument('--practical-submit',type=Path)
    native=p.add_mutually_exclusive_group()
    native.add_argument('--native-repl',action='store_true')
    native.add_argument('--native-warm',action='store_true')
    native.add_argument('--native-adapt',action='store_true')
    native.add_argument('--practical-ce',action='store_true')
    native.add_argument('--practical-gap',action='store_true')
    publish(**vars(p.parse_args()))
