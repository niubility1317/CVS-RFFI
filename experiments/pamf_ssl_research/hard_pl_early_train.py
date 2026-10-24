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
    reliable='U_reliable_As' in states[0]
    if any(('U_reliable_As' in s)!=reliable for s in states):raise ValueError('Matching As policy across both native segments required')
    native_as='U_native_As' in states[0]
    if (native_as and (reliable or weight!=6.) or any(('U_native_As' in s)!=native_as for s in states)):
        raise ValueError('Matching native As policy and fixed weight6 across native segments required')
    if any(('U_branch_guard' in s)!=('U_branch_guard' in states[0]) for s in states):
        raise ValueError('Matching source branch policy across both native segments required')
    if any(('U_parameter_step_guard' in s)!=('U_parameter_step_guard' in states[0]) for s in states):
        raise ValueError('Matching actual parameter step policy across native segments required')
    allrd='U_all_RDrop' in states[0]
    if (allrd and not native_as or any(('U_all_RDrop' in s)!=allrd for s in states)):
        raise ValueError('Matching rejected-U RDrop policy across both native segments required')
    balanced='U_balanced_CE' in states[0]
    if (balanced and not native_as or any(('U_balanced_CE' in s)!=balanced for s in states)):
        raise ValueError('Matching nativeAs bounded CE policy across native segments required')
    keys=('calls','seen','accepted','student_views','physical_seen','physical_student_views','raw_As_seen')
    total={key:sum(s[key] for s in states) for key in keys}
    anchors={name:sum(s['contrast_anchors'][name] for s in states) for name in ('time_down','t2.pool')}
    if (total['seen']!=sum(s['calls'] for s in states)*56700//222 or total['student_views']!=factor*total['seen']
        or total['physical_seen']!=(total['seen']//2 if reliable or native_as else 0)
        or total['physical_student_views']!=(2*sum(s['U_reliable_As']['used'] for s in states) if reliable else total['seen'] if native_as else 0)
        or total['raw_As_seen']!=(sum(s['U_reliable_As']['raw_fallback'] for s in states) if reliable else 0 if native_as else total['seen']//2 if mechanisms else 0)
        or any(s['EMA_updates'] or s['U_truth_read'] or s['label_smoothing']!=.01 for s in states)):
        raise ValueError('Earlier-U segment calls,mask,view exposure or permissions differ')
    prior=read(Path(parents[0]['folder'])/'progress.json')
    actual=dict(method='earlier_original_F21_E20_plus_native10_plus20',native_U_loop=False,own_EMA=False,
        cumulative_EMA_updates=0,U_seen=prior['PL_seen']+total['seen'],U_accepted=prior['PL_accepted']+total['accepted'],
        physical_U_seen=total['physical_seen'],physical_student_U_views=total['physical_student_views'],raw_U_As_seen=total['raw_As_seen'],
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
    if 'U_branch_guard' in states[0]:
        from .hard_pl_branch_guard import BRANCH_GUARD
        if any('U_branch_guard' not in s for s in states):raise ValueError('Matching branch policy across both native segments required')
        values=[s['U_branch_guard'] for s in states];branch=dict(values[0])
        expected_recipe=dict(BRANCH_GUARD,U_to_L_norm_cap=None) if states[0]['U_gradient_guard']['recipe']['U_to_L_norm_cap'] is None else BRANCH_GUARD
        if any(v['recipe']!=expected_recipe or v['calls']!=s['calls'] or v['gradient_calls']!=4*s['calls']
            or v['extra_autograd_calls']!=2*s['calls'] or v['L_pair_captures']!=4*s['calls']
            or v['pending_capture'] or v['pending_U_loss'] or v['extra_teacher_or_student_forward']
            or v['new_head'] or v['U_truth_reads'] or v['target_reads'] for s,v in zip(states,values)):
            raise ValueError('Original source branch captures/four derivatives/closed contexts or budgets differ')
        for key in ('calls','gradient_calls','extra_autograd_calls','L_pair_captures'):branch[key]=sum(v[key] for v in values)
        branch['anchor_conflicts']={group:{anchor:sum(v['anchor_conflicts'][group][anchor] for v in values)
            for anchor in branch['anchor_conflicts'][group]} for group in branch['anchor_conflicts']}
        actual['U_branch_guard']=branch;actual['hard_PL_tail']['U_branch_guard']=branch
    if 'U_parameter_step_guard' in states[0]:
        from .hard_pl_step_guard import STEP_GUARD
        values=[s['U_parameter_step_guard'] for s in states];step=dict(values[0])
        if any(v['recipe']!=STEP_GUARD or v['calls']!=s['calls'] or v['anchor_captures']!=s['calls']
            or v['reference_optimizer_steps']!=s['calls'] or v['reference_physical_clip_calls']!=s['calls']
            or v['pending_anchors'] or not v['copied_existing_moments'] or not v['live_joint_optimizer_state_retained']
            or not v['reference_has_no_model_forward'] or v['extra_autograd_calls'] or v['extra_model_forwards']
            or v['U_truth_reads'] or v['target_reads'] or v['new_head'] or v['new_inference_parameters'] for s,v in zip(states,values)):
            raise ValueError('Every real source step must consume one current-moment L reference and actual FP32 projection')
        for key in ('calls','anchor_captures','reference_optimizer_steps','reference_physical_clip_calls','fallback_groups','removed_material_conflicts','seconds'):
            step[key]=sum(v[key] for v in values)
        step['minimum_whole_increment_retention']=min(v['minimum_whole_increment_retention'] for v in values)
        actual['U_parameter_step_guard']=step;actual['hard_PL_tail']['U_parameter_step_guard']=step
    if allrd:
        from .rejected_rdrop import RECIPE as ALLRD
        values=[s['U_all_RDrop'] for s in states];rd=dict(values[0])
        if any(v['recipe']!=ALLRD or v['calls']!=s['calls'] or v['seen']!=s['seen'] or v['rejected']!=s['seen']-s['accepted']
            or not 0<=v['nonphysical_rejected']<=min(v['rejected'],v['seen']//2)
            or v['pairs']!=2*v['rejected']+v['nonphysical_rejected'] or v['reused_prediction_views']!=2*v['pairs']
            or v['extra_KL_evaluations']!=3*v['calls'] or v['hard_label_or_CE_on_rejected'] or v['raw_acceptance_changed']
            or v['physical_As_KL_weight'] or v['pending_logits'] or v['extra_teacher'] or v['extra_student'] or v['extra_head']
            or v['new_head'] or v['U_truth_reads'] or v['target_reads'] for s,v in zip(states,values)):
            raise ValueError('Rejected U may only reuse the fixed nonphysical RDrop logits without changing hard acceptance or any source permission')
        for key in ('calls','seen','rejected','nonphysical_rejected','pairs','reused_prediction_views','extra_KL_evaluations'):
            rd[key]=sum(v[key] for v in values)
        actual['U_all_RDrop']=rd;actual['hard_PL_tail']['U_all_RDrop']=rd
    if 'U_complement' in states[0]:
        values=[s['U_complement'] for s in states];complement=dict(values[0])
        if any(v['recipe']!=complement['recipe'] or v['calls']!=s['calls'] or v['extra_head_calls']!=2*s['calls']
            or v['extra_head_packet_views']!=2*s['seen'] or v['extra_backbone_forwards'] or v['teacher_or_L_head_calls']
            or v['new_head'] or v['persistent_hooks'] or not v['full_deployment_signature_retained'] or v['CE_dose_normalizer']!=1.25
            for s,v in zip(states,values)):
            raise ValueError('Complementary CE must reallocate the same dose using two existing-head views')
        for key in ('calls','extra_head_calls','extra_head_packet_views'):complement[key]=sum(v[key] for v in values)
        actual['U_complement']=complement;actual['hard_PL_tail']['U_complement']=complement
    if 'U_path_complement' in states[0]:
        values=[s['U_path_complement'] for s in states];paths=dict(values[0])
        if any(v['recipe']!=paths['recipe'] or v['calls']!=s['calls'] or v['extra_head_calls']!=4*s['calls']
            or v['extra_head_packet_views']!=4*s['seen'] or v['extra_backbone_forwards'] or v['teacher_or_L_head_calls']
            or v['new_head'] or v['persistent_hooks'] or not v['full_deployment_signature_retained']
            or not v['full_deployment_response_retained'] or v['CE_dose_normalizer']!=1.25 for s,v in zip(states,values)):
            raise ValueError('Path complementary CE must keep the same dose with four existing-head views')
        for key in ('calls','extra_head_calls','extra_head_packet_views'):paths[key]=sum(v[key] for v in values)
        actual['U_path_complement']=paths;actual['hard_PL_tail']['U_path_complement']=paths
    if 'U_receiver_extrapolation' in states[0]:
        values=[s['U_receiver_extrapolation'] for s in states];rxe=dict(values[0])
        if any(v['recipe']!=rxe['recipe'] or v['calls']!=s['calls'] or v['extra_head_calls']!=2*s['calls']
            or v['extra_head_packet_views']!=2*s['seen'] or not 0<=v['eligible_packet_views']<=2*s['accepted']
            or v['extra_backbone_forwards'] or v['teacher_or_L_head_calls'] or v['new_head'] or v['persistent_hooks']
            or not v['all_raw_accepted_retained'] or not v['full_deployment_signature_retained']
            or not v['full_deployment_response_retained'] or v['CE_dose_normalizer']!=1.25 for s,v in zip(states,values)):
            raise ValueError('Full-feature receiver extrapolation must conserve CE dose,all rawaccepted samples and two existing-head views')
        for key in ('calls','extra_head_calls','extra_head_packet_views','eligible_packet_views'):rxe[key]=sum(v[key] for v in values)
        actual['U_receiver_extrapolation']=rxe;actual['hard_PL_tail']['U_receiver_extrapolation']=rxe
    if native_as:
        from .hard_pl_native_as import NATIVE_AS
        values=[s['U_native_As'] for s in states];nas=dict(values[0])
        if any(v['recipe']!=NATIVE_AS or v['calls']!=s['calls'] or v['generated']!=s['seen']//2
            or v['student_physical_views']!=s['seen'] or s['physical_seen']!=v['generated']
            or s['physical_student_views']!=v['student_physical_views'] or s['raw_As_seen']!=0
            or not 0<=v['raw_accepted_half']<=min(s['accepted'],v['generated'])
            or v['accepted_student_views']!=2*v['raw_accepted_half'] or v['raw_fallback'] or v['extra_As_teacher']
            or not v['all_raw_accepted_retained'] or v['extra_student_forwards'] or v['new_head']
            or v['EMA_updates'] or v['U_truth_read'] or v['CE_dose_normalizer']!=1.25 for s,v in zip(states,values)):
            raise ValueError('Native As must retain raw acceptance/losses with original physical budget and no extra teacher')
        for key in ('calls','generated','student_physical_views','raw_accepted_half','accepted_student_views'):
            nas[key]=sum(v[key] for v in values)
        actual['U_native_As']=nas;actual['hard_PL_tail']['U_native_As']=nas
    if balanced:
        from .hard_pl_balanced_ce import BALANCED_CE
        values=[s['U_balanced_CE'] for s in states];bce=dict(values[0])
        if any(v['recipe']!=BALANCED_CE or v['calls']!=s['calls'] or v['captures']!=6*s['calls']
            or v['captured_packet_views']!=6*s['seen'] or v['extra_CE_kernels']!=6*s['calls'] or v['accepted']!=s['accepted']
            or abs(v['accepted_weight_mass']-s['accepted'])>1e-7*max(1,s['accepted'])
            or not .5-1e-12<=v['minimum_weight']<=v['maximum_weight']<=1.5+1e-12
            or not 0<=v['max_batch_dose_error']<=1e-6 or not v['all_raw_accepted_retained'] or v['raw_acceptance_changed']
            or v['extra_teacher'] or v['extra_backbone'] or v['extra_head'] or v['new_head'] or v['persistent_context']
            or v['EMA_updates'] or v['prototype_updates'] or v['U_truth_reads'] or v['target_reads'] or v['CE_dose_normalizer']!=1.25
            for s,v in zip(states,values)):
            raise ValueError('Bounded CE contribution must preserve accepted mass,masks and original nativeAs forward cost')
        for key in ('calls','captures','captured_packet_views','extra_CE_kernels','accepted','accepted_weight_mass'):
            bce[key]=sum(v[key] for v in values)
        bce['minimum_weight']=min(v['minimum_weight'] for v in values);bce['maximum_weight']=max(v['maximum_weight'] for v in values)
        bce['max_batch_dose_error']=max(v['max_batch_dose_error'] for v in values)
        actual['U_balanced_CE']=bce;actual['hard_PL_tail']['U_balanced_CE']=bce
    if reliable:
        values=[s['U_reliable_As'] for s in states];ras=dict(values[0])
        if any(v['recipe']!=ras['recipe'] or v['calls']!=s['calls'] or v['generated']!=s['seen']//2
            or v['teacher_As_views']!=v['generated'] or v['used']+v['raw_fallback']!=v['generated']
            or not 0<=v['used']<=v['raw_accepted_half']<=v['generated']
            or not v['used']<=min(v['view_confident'],v['view_argmax_agree'])<=v['generated']
            or s['physical_seen']!=v['generated'] or s['physical_student_views']!=2*v['used'] or s['raw_As_seen']!=v['raw_fallback']
            or not v['all_raw_accepted_retained'] or v['extra_student_forwards'] or v['new_head'] or v['EMA_updates']
            or v['U_truth_read'] or v['teacher_eval_norm_updates'] for s,v in zip(states,values)):
            raise ValueError('Reliable As must retain all raw accepted U,exactraw fallback and actual eval/physical use counts')
        for key in ('calls','generated','teacher_As_views','used','raw_fallback','raw_accepted_half','view_confident','view_argmax_agree'):
            ras[key]=sum(v[key] for v in values)
        actual['U_reliable_As']=ras;actual['hard_PL_tail']['U_reliable_As']=ras
        actual['teacher_eval_U_raw_views']=actual['teacher_eval_U_views'];actual['teacher_eval_U_As_views']=ras['teacher_As_views']
        actual['teacher_eval_U_views']+=ras['teacher_As_views']
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
