"""Strict E20 source parent and two native segments;no target-directed resume."""
from pathlib import Path
import shutil
import time
import torch
from experiments.cric_dura_comparison.io import read,write
from .hard_pl import PARENT_RUN
from .hard_pl_early import validate
from .hard_pl_objective import HardPLObjective
from .hard_pl_no_as_mixstyle_objective import NoAsMixStyleObjective


def objective(c):
    return (HardPLObjective if c['hard_pl']['method']=='HPL2' else NoAsMixStyleObjective)(c)


def source_parents(c,*,validate_config=validate):
    validate_config(c);contract=read(c['source_contract']);path=Path(c['fork_parent_source'])/'segment20'
    done=read(path/'complete.json');pc=done['config'];progress=read(path/'progress.json')
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=20 or done['updates']!=4440
        or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
        or pc['run_id']!=PARENT_RUN or pc['row_id']!='F21-s'+str(c['model_seed'])
        or pc.get('parent_source') is not None or pc.get('epoch_offset')!=0 or pc.get('step_offset')!=0
        or pc.get('model_seed')!=c['model_seed'] or pc.get('ablation')!='F21' or pc.get('regime')!='native_cric' or pc.get('satellite_concat') is not False
        or read(pc['source_contract'])!=contract or progress['PL_seen']!=396900):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH:original fixed F21 E20')
    payload=torch.load(path/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if (payload['run_id']!=pc['run_id'] or payload['candidate_id']!=pc['row_id'] or payload['epoch']!=20
        or payload['a1_ema_successful_updates']!=4440 or payload['args']['baseline_ckpt']
        or payload['args']['teacher_ckpt'] or payload['args']['use_ema_teacher'] or payload['args']['use_unlabeled']):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED:actual source-only F21 E20 payload')
    return [dict(folder=str(path),**done)],contract


def aggregate(states,parents,*,weight=2.):
    mechanisms=states[0]['method']=='HPLT1U';factor=6 if mechanisms else 1
    keys=('calls','seen','accepted','student_views','physical_seen','physical_student_views','raw_As_seen')
    total={key:sum(s[key] for s in states) for key in keys}
    anchors={name:sum(s['contrast_anchors'][name] for s in states) for name in ('time_down','t2.pool')}
    if (total['seen']!=sum(s['calls'] for s in states)*56700//222 or total['student_views']!=factor*total['seen']
        or total['physical_seen'] or total['physical_student_views']
        or total['raw_As_seen']!=(total['seen']//2 if mechanisms else 0)
        or any(s['EMA_updates'] or s['U_truth_read'] or s['label_smoothing']!=.01 for s in states)):
        raise ValueError('Earlier-U segment calls,mask,view exposure or permissions differ')
    prior=read(Path(parents[0]['folder'])/'progress.json')
    actual=dict(method='earlier_original_F21_E20_plus_native10_plus20',native_U_loop=False,own_EMA=False,
        cumulative_EMA_updates=0,U_seen=prior['PL_seen']+total['seen'],U_accepted=prior['PL_accepted']+total['accepted'],
        physical_U_seen=0,physical_student_U_views=0,raw_U_As_seen=total['raw_As_seen'],
        pseudo_label_smoothing=.01,student_U_views=prior['PL_seen']+total['student_views'],
        teacher_eval_U_views=prior['PL_seen']+total['seen'],U_truth_read=False,prototype_updates=0,
        hard_PL_calls=total['calls'],hard_PL_tail=dict(total,method=states[0]['method'],contrast_anchors=anchors,
            label_smoothing=.01,EMA_updates=0,U_truth_read=False),
        threshold=.95,weight=weight,default_head_joint=True,new_inference_parameters=0,
        new_mechanism_epochs=total['calls']//222,source_parent_epoch=20)
    if mechanisms:
        style=dict(states[0]['U_style'])
        for key in ('calls','seen','eligible','plan_calls'):style[key]=sum(s['U_style'][key] for s in states)
        style['applied']={name:sum(s['U_style']['applied'][name] for s in states) for name in anchors}
        if (style['calls']!=6*total['calls'] or style['seen']!=6*total['seen']
            or style['plan_calls']!=total['calls'] or style['eligible']>6*total['accepted']
            or style['persistent_hooks'] or style['teacher_or_L_style_calls'] or style['partner_TX_truth_reads']
            or not style['shared_six_forward_plan'] or not style['contrast_captures_after_style']):
            raise ValueError('Earlier supervised-U shared style execution differs')
        actual['U_style']=style;actual['hard_PL_tail']['U_style']=style
    if 'U_gradient_guard' in states[0]:
        guards=[s['U_gradient_guard'] for s in states];guard=dict(guards[0])
        if any(g['recipe']!=guards[0]['recipe'] or g['pending_U_loss'] or g['calls']!=s['calls'] for s,g in zip(states,guards)):
            raise ValueError('Every earlier-U loss must consume exactly one fixed gradient guard')
        guard['calls']=sum(g['calls'] for g in guards)
        for key in ('conflicts','caps'):guard[key]={group:sum(g[key][group] for g in guards) for group in guard[key]}
        actual['U_gradient_guard']=guard;actual['hard_PL_tail']['U_gradient_guard']=guard
    if 'U_joint_contrast' in states[0]:
        values=[s['U_joint_contrast'] for s in states];joint=dict(values[0])
        if any(v['recipe']!=joint['recipe'] or v['calls']!=s['calls'] or v['captures']!=2*s['calls'] or v['captured_packets']!=2*s['seen']
            or v['extra_forward_calls'] or v['teacher_or_L_captures'] or v['persistent_feature_wrapper'] for s,v in zip(states,values)):
            raise ValueError('Current U joint objective must reuse exactly two existing ordinary student forwards')
        for key in ('calls','captures','captured_packets','anchors'):joint[key]=sum(v[key] for v in values)
        actual['U_joint_contrast']=joint;actual['hard_PL_tail']['U_joint_contrast']=joint
    return actual


def train(c,device='cuda:0',*,validate_config=validate,objective_cls=objective):
    parents,contract=source_parents(c,validate_config=validate_config)
    from .hard_pl_train import installed
    from experiments.cric_dura_comparison.f_train import train as segment
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    states=[];segments=list(parents);previous=parents[-1]['folder'];start=time.perf_counter()
    for end,epochs,offset in ((30,10,20),(50,20,30)):
        folder=root/('segment'+str(end));cfg=dict(c,output_root=str(folder),segment_epochs=epochs,
            epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        holder={}
        with installed(cfg,holder,objective_cls=objective_cls):segment(cfg,device=device)
        obj=holder['objective'];state=obj.state_dict()
        if state['calls']!=epochs*222 or state['seen']!=epochs*56700 or obj.capture_active:
            raise ValueError('Complete fixed native segment required')
        states.append(state);write(folder/'hard_pl_state.json',state);actual=aggregate(states,parents,weight=c['hard_pl']['weight']);actual['method']=c['algorithm_id']
        done=read(folder/'complete.json');done['actual_adapter_SSL']=actual
        write(folder/'complete.json',done,exclusive=False);segments.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SOURCE_TRAINING',epoch=end,updates=end*222,
            incremental_epochs=end-20,incremental_updates=(end-20)*222,actual_adapter_SSL=actual),exclusive=False)
    actual=aggregate(states,parents,weight=c['hard_pl']['weight']);actual['method']=c['algorithm_id'];seconds=time.perf_counter()-start;parent_seconds=parents[0]['seconds']
    if (actual['hard_PL_calls']!=6660 or actual['U_seen']!=2097900
        or actual['student_U_views']!=(10602900 if c['hard_pl']['method']=='HPLT1U' else 2097900)):
        raise ValueError('Fixed cumulative E50 earlier-U exposure differs')
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',contract)
    done=dict(status='SOURCE_FROZEN',method_id=c['algorithm_id'],scaffold_id=c['scaffold_id'],epoch=50,updates=11100,
        segments=segments,source_contract=c['source_contract'],config=c,initialization='same_contract_scratch20_own10_own20',
        ancestors=[s['folder'] for s in segments],actual_adapter_SSL=actual,head_fit_updates=0,U_truth_read=False,
        target_access=False,seconds=seconds,incremental_epochs=30,incremental_updates=6660,
        parent_prefix_native_seconds=parent_seconds,parent_prefix_training_seconds=parent_seconds,
        cumulative_training_seconds=parent_seconds+seconds,parent_run_id=PARENT_RUN,
        parent_selection='Fixed original F21 E20 before current candidate targets;full sourceV .95 precision>=99% all3;no target selection')
    write(root/'complete.json',done)
    write(root/'progress.json',{k:done[k] for k in ('status','epoch','updates','incremental_epochs','incremental_updates','seconds','actual_adapter_SSL')},exclusive=False)
