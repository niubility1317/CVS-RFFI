"""Release the full LoRa recovery comparison using established source roles."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_adaptive import RUN as TRI, SET_RUN, WARMUP_RUN
from .full_dura import COVERAGE_RUN, PROJECTED_RUN, RUN, SHAPE_RUN, STACK_RUN, STAGED_RUN, SWEEP_RUN
from .full_dura_detail import (COMPLEX_RESPONSE_RUN, DETAIL_CONFIRM_RUN, DETAIL_DOSE_RUN, DETAIL_RUN,
                               END_PHASE_RUN, PHASE_CONFIRM_RUN,
                               PHASE_DETAIL_RUN, SHARED_DETAIL_RUN, SPECTRAL_PHASE_RUN)
from .full_dura_r69_lora import R69_ACTION_RUN, R69_LORA_RUN
from .full_dura_trace_source import RETAINED_RUN, TRACE_RUN
from .full_native_joint import ADAPTED_JOINT_RUN, JOINT_RUN, parents
from .full_packet_publish import DATA_RUN
from .full_restore_draw import RUN as DRAW_RUN
from .publish import release, release_commit


def publish(repo,output,context,ssh_config,data_submit,tri_submit,dose_submit=None,shape=False,projected=False,
            coverage_submit=None,stack=False,draw_submit=None,projected_submit=None,stack_submit=None,
            coherent_detail=False,detail_dose_submit=None,confirm_dose=False,r69_aligned=False,
            native_tri_submit=None,r69_submit=None,r69_action=False,r69_trace=False,
            adapted_joint=False,r69_retain=False,shared_detail=False,phase_detail=False,
            phase_confirm=False,phase_submit=None,spectral_phase=False,ends_phase=False,complex_response=False):
    x=read(context)
    rt={k:x[k] for k in ('project','python','frozen_sources')}
    p=PurePosixPath(x['project'])
    if (coherent_detail or detail_dose_submit is not None) and draw_submit is None:
        raise ValueError('High-coherence detail requires the frozen independent source draw')
    single_post_half=phase_detail or spectral_phase or ends_phase or complex_response
    if (confirm_dose or r69_aligned or shared_detail or single_post_half or phase_confirm) and detail_dose_submit is None:
        raise ValueError('Dose confirmation requires the frozen single-seed dose matrix')
    if phase_confirm!=(phase_submit is not None):
        raise ValueError('Three-seed phase requires the frozen seed-02 phase run')
    native_joint=native_tri_submit is not None
    if adapted_joint and not native_joint:
        raise ValueError('Adapted LoRa joint scoring requires both frozen source model releases')
    r69_source=r69_action or r69_trace or r69_retain
    if r69_source and (r69_submit is None or draw_submit is None or native_joint or detail_dose_submit is not None):
        raise ValueError('Original DURA action calibration needs one frozen matching R69 source run and draw')
    if not r69_source and native_joint!=(r69_submit is not None):
        raise ValueError('Joint LoRa inference requires both native releases')
    if native_joint and (confirm_dose or r69_aligned or coherent_detail):
        raise ValueError('Joint LoRa inference uses already released models')
    joint=stack or draw_submit is not None or native_joint
    new_run=shape or projected or coverage_submit is not None or joint
    if new_run and dose_submit is None and not native_joint:
        raise ValueError('Shape protection requires the frozen full LoRa G')
    handles=[(data_submit,DATA_RUN),(tri_submit,SET_RUN if joint else WARMUP_RUN if new_run else TRI)]
    if dose_submit is not None:
        handles.append((dose_submit,RUN))
    if coverage_submit is not None:
        handles.append((coverage_submit,PROJECTED_RUN))
    if joint and not native_joint:
        if projected_submit is None:
            raise ValueError('Joint LoRa comparisons require the same frozen projected G control')
        handles.append((projected_submit,PROJECTED_RUN))
    if draw_submit is not None:
        if stack_submit is None:
            raise ValueError('Staged DURA requires the joint frozen Tri control experiment')
        handles.extend(((draw_submit,DRAW_RUN),(stack_submit,STACK_RUN)))
    if detail_dose_submit is not None:
        parent=(DETAIL_CONFIRM_RUN if (r69_aligned or phase_confirm) else
                DETAIL_DOSE_RUN if (confirm_dose or shared_detail or single_post_half) else DETAIL_RUN)
        handles.append((detail_dose_submit,parent))
    if phase_confirm:
        handles.append((phase_submit,PHASE_DETAIL_RUN))
    if native_joint:
        g,f=parents(PurePosixPath(ADAPTED_JOINT_RUN if adapted_joint else JOINT_RUN))
        handles.extend(((native_tri_submit,f),(r69_submit,g)))
    if r69_source:
        handles.append((r69_submit,TRACE_RUN if r69_retain else R69_LORA_RUN))
    for path,name in handles:
        o=read(path)
        if (o['archive_transfer']!='VERIFIED' or PurePosixPath(o['run_root'])!=p/'runs'/name
                or PurePosixPath(o['cwd'])!=p/'releases'/f'{name}-release'):
            raise ValueError('The full LoRa release receipts differ')
    data=p/'runs'/DATA_RUN
    if r69_retain:
        name=RETAINED_RUN
    elif r69_trace:
        name=TRACE_RUN
    elif r69_action:
        name=R69_ACTION_RUN
    elif native_joint:
        name=ADAPTED_JOINT_RUN if adapted_joint else JOINT_RUN
    elif joint:
        if detail_dose_submit is not None:
            name=(COMPLEX_RESPONSE_RUN if complex_response else END_PHASE_RUN if ends_phase else SPECTRAL_PHASE_RUN if spectral_phase else
                  PHASE_CONFIRM_RUN if phase_confirm else PHASE_DETAIL_RUN if phase_detail else
                  SHARED_DETAIL_RUN if shared_detail else R69_LORA_RUN if r69_aligned else
                  DETAIL_CONFIRM_RUN if confirm_dose else DETAIL_DOSE_RUN)
        else:
            name=DETAIL_RUN if coherent_detail else STAGED_RUN if draw_submit is not None else STACK_RUN
    elif coverage_submit is not None:
        name=COVERAGE_RUN
    else:
        name=PROJECTED_RUN if projected else SHAPE_RUN if shape else SWEEP_RUN if dose_submit is not None else RUN
    rt.update(run_id=name,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.full_dura',
        release_root=str(p/'releases'/f'{name}-release'),run_root=str(p/'runs'/name),
        parent_root=str(data),source_root=str(p/'releases'/f'{DATA_RUN}-release/data/source'),
        target_views=str(data/'target_views'),allowed_gpus=[7,5,6,4,1,3,0,2] if joint else
        [5,6,1,4,0,2,3] if new_run else list(range(7)),
        max_compute_processes=4)
    if joint or coverage_submit is not None:
        rt['max_workers']=4
    if r69_retain:
        rt['allowed_gpus']=[5,6,2,0,4,1,3,7]
    if shared_detail or single_post_half or phase_confirm:
        rt.update(allowed_gpus=[0,6,2,1,4,5,3],max_workers=2 if phase_confirm else 1)
    return release(repo,output,rt,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config','data-submit','tri-submit'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--dose-submit',type=Path)
    p.add_argument('--projected-submit',type=Path)
    p.add_argument('--stack-submit',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--shape',action='store_true')
    mode.add_argument('--projected',action='store_true')
    mode.add_argument('--coverage-submit',type=Path)
    mode.add_argument('--stack',action='store_true')
    mode.add_argument('--draw-submit',type=Path)
    detail=p.add_mutually_exclusive_group()
    detail.add_argument('--coherent-detail',action='store_true')
    detail.add_argument('--detail-dose-submit',type=Path)
    extension=p.add_mutually_exclusive_group()
    extension.add_argument('--confirm-dose',action='store_true')
    extension.add_argument('--r69-aligned',action='store_true')
    extension.add_argument('--r69-action',action='store_true')
    extension.add_argument('--r69-trace',action='store_true')
    extension.add_argument('--r69-retain',action='store_true')
    extension.add_argument('--shared-detail',action='store_true')
    extension.add_argument('--phase-detail',action='store_true')
    extension.add_argument('--phase-confirm',action='store_true')
    extension.add_argument('--spectral-phase',action='store_true')
    extension.add_argument('--ends-phase',action='store_true')
    extension.add_argument('--complex-response',action='store_true')
    p.add_argument('--phase-submit',type=Path)
    p.add_argument('--native-tri-submit',type=Path)
    p.add_argument('--r69-submit',type=Path)
    p.add_argument('--adapted-joint',action='store_true')
    publish(**vars(p.parse_args()))
