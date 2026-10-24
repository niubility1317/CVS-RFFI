"""Real F21 parent smoke, two hard-PL forks, automatic dual-readout truth-last."""
import argparse
import gc
import math
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write


def smoke(rt,*,validate_config=None,max_rows=2,objective_cls=None,source_parents_fn=None,parent_epoch=30):
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    activate(rt);numerical_policy()
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import context
    from experiments.cric_dura_comparison.f_train import arguments,original_config,class_dimension
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison import t1_model
    from .hard_pl_train import source_parents,installed
    if source_parents_fn is not None:source_parents=source_parents_fn
    if objective_cls is None:
        from .hard_pl_objective import HardPLObjective as objective_cls
    if validate_config is None:
        from .hard_pl import validate as validate_config
    checked=[]
    for row in rt['rows'][:max_rows]:
        c=read(row['train_config']);parents,_=source_parents(c,validate_config=validate_config);parent=parents[-1]['folder']
        cfg=dict(c,segment_epochs=10 if parent_epoch==20 else 20,epoch_offset=parent_epoch,
            step_offset=parent_epoch*222,parent_source=parent,hard_pl_smoke=True)
        payload=torch.load(Path(parent)/'final_ssdg.pth',map_location='cpu',weights_only=False)
        ctx=context(cfg);holder={};device='cuda:0';torch.manual_seed(c['model_seed'])
        with installed(cfg,holder,objective_cls=objective_cls),class_dimension(cfg),native_installed(original_config(c['model_seed'])) as native:
            args=arguments(cfg,device);args.input_len=256;args.baseline_ckpt=args.teacher_ckpt=''
            args.from_scratch=args.a1_scratch_only=True
            if args.use_unlabeled or args.use_ema_teacher:raise ValueError('Native U/EMA must be disabled')
            model=t1_model.install(native.build_baseline_model(args,device),cfg).to(device)
            model.load_state_dict(payload['model'],strict=True);model.eval()
            with torch.no_grad():logits=model(ctx['train_loader'].dataset.x[:2].to(device))
            if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Real F21 checkpoint inference failed')
            model.train();model._cd_U_loader=ctx['unlabeled_loader'];model._cd_U_iter=iter(model._cd_U_loader)
            batch=next(iter(ctx['train_loader']));x,y,domain=[v.to(device) for v in batch[:3]]
            meta={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch[3].items()}
            with model._phase1_labeled_feature_context():output=model(x,return_aux=True,domain_labels=domain)
            ce=F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
            correction,metrics=model._source_labeled_feature_hook(model=model,labeled_output=output,labels=y,
                extra=(None,meta),domains=domain,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            total=ce+correction;total.backward();obj=holder['objective'];state=obj.state_dict()
            head=model.id_backbone.encoder.core.id_backbone.cls_head.weight;mechanisms=c['hard_pl']['method']=='HPLT1U'
            no_uas=bool(c.get('hard_pl_no_uas'));physical_enabled=mechanisms and not no_uas
            reliable=bool(c.get('hard_pl_reliable_as'));ras=state.get('U_reliable_As',{})
            smoothing=c.get('hard_pl_label_smoothing',0.) if not mechanisms else .01
            if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all())
                or float(head.grad.norm())<=0 or metrics['T1_L_packets']!=768 or state['calls']!=1 or state['seen']!=256
                or not state['accepted'] or state['student_views']!=(1536 if mechanisms else 256)
                or state['physical_seen']!=(128 if physical_enabled else 0)
                or state['physical_student_views']!=(2*ras['used'] if reliable else 256 if physical_enabled else 0)
                or state['raw_As_seen']!=(ras['raw_fallback'] if reliable else 128 if no_uas else 0) or state['EMA_updates']!=0 or obj.capture_active
                or state['label_smoothing']!=smoothing or metrics['HPL_label_smoothing']!=smoothing
                or not torch.allclose(metrics['HPL_loss'],c['hard_pl']['weight']*metrics['HPL_base_loss'])):
                raise ValueError('Actual hard-PL weight/default-head/L/U execution failed')
            if mechanisms and (any(v<=0 for v in state['contrast_anchors'].values())
                or not all(h.grad is not None and bool(torch.isfinite(h.grad).all()) and float(h.grad.norm())>0
                           for maps in obj.smoke_maps for h in maps.values())
                or metrics['HPL_As_KL_weight']!=0 or metrics['HPL_ordinary_KL']<=0 or metrics['HPL_raw_KL']<=0):
                raise ValueError('Actual shallow/middle contrast or paired-dropout mechanism absent')
            if no_uas and not obj.smoke_U_As_input_raw:raise ValueError('Actual U As branch must use exact raw IQ')
            if no_uas and sum(model._cd_practical.coverage.values())!=64:
                raise ValueError('Original labeled As64 must remain;U Practical must not execute')
            if c.get('hard_pl_label_smoothing') and sum(model._cd_practical.coverage.values())!=64:
                raise ValueError('Smoothed ordinary PL must keep L Practical64 and U physical0')
            if c.get('hard_pl_mixstyle'):
                style=state['U_style']
                style_passes=6 if mechanisms else 1
                if (style['calls']!=style_passes or style['seen']!=256*style_passes or style['eligible']<=0 or obj.style_active
                    or not all(v>0 for v in style['applied'].values())
                    or not all(h.grad is not None and bool(torch.isfinite(h.grad).all())
                               and float(h.grad.norm())>0 for h in obj.style_smoke_maps.values())):
                    raise ValueError('Actual accepted-only shallow/middle U style or finite gradient absent')
            if c.get('hard_pl_gradient_guard'):
                guard=state['U_gradient_guard'];details=obj.smoke_guard_details
                if guard['calls']!=1 or guard['pending_U_loss']:
                    raise ValueError('Exactly one consumed real supervised-anchored U gradient required')
                for name,p in model.named_parameters():
                    expected=obj.smoke_guard_expected.get(name)
                    if (expected is None and p.grad is not None) or (expected is not None and
                        (p.grad is None or not torch.allclose(p.grad,expected,rtol=2e-5,atol=2e-6))):
                        raise ValueError('Actual backward differs from protected parameter gradients')
                if any(d['applied_U_norm']>d['L_norm']*(1+1e-6)+1e-8 or
                    d['dot_after'] < -1e-5*(d['L_norm']*d['applied_U_norm']+1e-12) for d in details.values()):
                    raise ValueError('Real gradient projection or norm cap differs')
            if c.get('hard_pl_joint_contrast'):
                joint=state['U_joint_contrast']
                if (joint['calls']!=1 or joint['captures']!=2 or joint['captured_packets']!=512 or joint['anchors']<=0 or joint['extra_forward_calls']
                    or joint['teacher_or_L_captures'] or joint['persistent_feature_wrapper'] or obj.joint_active
                    or obj.smoke_joint_signature_gradient<=0 or len(obj.smoke_joint_features)!=2
                    or any(h.shape!=(256,160) or h.grad is None or not bool(torch.isfinite(h.grad).all())
                        or float(h.grad.norm())<=0 for h in obj.smoke_joint_features)):
                    raise ValueError('Actual U deployed joint features/signature gradients or original forward budget absent')
            if c.get('hard_pl_complement'):
                complement=state['U_complement'];g=obj.smoke_complement_gradients
                if (complement['calls']!=1 or complement['extra_head_calls']!=2 or complement['extra_head_packet_views']!=512
                    or complement['extra_backbone_forwards'] or complement['teacher_or_L_head_calls'] or complement['new_head']
                    or complement['persistent_hooks'] or not complement['full_deployment_signature_retained']
                    or complement['CE_dose_normalizer']!=1.25 or obj.complement_active
                    or g['signature']>1e-7 or g['head']<=0 or g['core']<=0
                    or len(obj.smoke_complement_features)!=2 or any(h.shape!=(256,160) or h.grad is None
                        or not bool(torch.isfinite(h.grad).all()) or float(h.grad.norm())<=0 for h in obj.smoke_complement_features)):
                    raise ValueError('Same-head complementary CE dose/bypass gradient or actual head-view budget differs')
            if c.get('hard_pl_path'):
                paths=state['U_path_complement'];g=obj.smoke_path_gradients
                if (paths['calls']!=1 or paths['extra_head_calls']!=4 or paths['extra_head_packet_views']!=1024
                    or paths['extra_backbone_forwards'] or paths['teacher_or_L_head_calls'] or paths['new_head'] or paths['persistent_hooks']
                    or not paths['full_deployment_signature_retained'] or not paths['full_deployment_response_retained']
                    or paths['CE_dose_normalizer']!=1.25 or obj.path_active
                    or max(g['signature'],g['response'])>1e-7 or min(g['head'],g['core'])<=0
                    or len(obj.smoke_path_features)!=4 or any(h.shape!=(256,160) or h.grad is None
                        or not bool(torch.isfinite(h.grad).all()) or float(h.grad.norm())<=0 for h in obj.smoke_path_features)):
                    raise ValueError('Existing-head full/masked/core CE dose,exclusive-branch bypass or actual view budget differs')
            if c.get('hard_pl_receiver_extrapolation'):
                rxe=state['U_receiver_extrapolation'];norms=obj.smoke_rxe_gradients
                if (rxe['calls']!=1 or rxe['extra_head_calls']!=2 or rxe['extra_head_packet_views']!=512
                    or not 0<rxe['eligible_packet_views']<=2*state['accepted'] or rxe['extra_backbone_forwards']
                    or rxe['new_head'] or not rxe['all_raw_accepted_retained'] or rxe['CE_dose_normalizer']!=1.25
                    or min(norms.values())<=0 or not all(math.isfinite(v) for v in norms.values())):
                    raise ValueError('Actual fullfeature receiver CE must retain all branch/head gradients and declared cost')
            if c.get('hard_pl_native_as'):
                nas=state['U_native_As']
                if (nas['calls']!=1 or nas['generated']!=128 or nas['student_physical_views']!=256
                    or not 0<=nas['raw_accepted_half']<=min(state['accepted'],128)
                    or nas['accepted_student_views']!=2*nas['raw_accepted_half'] or nas['raw_fallback'] or nas['extra_As_teacher']
                    or not nas['all_raw_accepted_retained'] or nas['extra_student_forwards'] or nas['new_head']
                    or nas['EMA_updates'] or nas['U_truth_read'] or nas['CE_dose_normalizer']!=1.25
                    or not obj.smoke_native_As_matches or not obj.smoke_native_nonphysical_ordinary_matches
                    or sum(model._cd_practical.coverage.values())!=192):
                    raise ValueError('Native source As half/accepted views/unchanged CE or original forward cost differs')
            if c.get('hard_pl_balanced_ce'):
                from .source_balanced_pseudo_ce import contribution_weights
                bce=state['U_balanced_CE'];w=obj.smoke_bce_weights;a=obj.smoke_bce_accepted
                expected=contribution_weights(obj.smoke_bce_pseudo,a,obj.smoke_bce_rx)
                if (bce['calls']!=1 or bce['captures']!=6 or bce['captured_packet_views']!=1536 or bce['extra_CE_kernels']!=6
                    or bce['accepted']!=state['accepted'] or abs(bce['accepted_weight_mass']-state['accepted'])>1e-6
                    or bce['minimum_weight']<.5-1e-12 or bce['maximum_weight']>1.5+1e-12 or bce['max_batch_dose_error']>1e-6
                    or not torch.equal(w,expected) or w.requires_grad or not bool((w[a]>0).all())
                    or not bce['all_raw_accepted_retained'] or bce['raw_acceptance_changed'] or bce['extra_teacher']
                    or bce['extra_backbone'] or bce['extra_head'] or bce['new_head'] or bce['persistent_context']
                    or bce['EMA_updates'] or bce['prototype_updates'] or bce['U_truth_reads'] or bce['target_reads']
                    or obj._bce_logits is not None or obj._bce_current is not None
                    or not torch.allclose(obj.smoke_bce_after-obj.smoke_bce_before,obj.smoke_bce_delta,rtol=1e-4,atol=1e-6)):
                    raise ValueError('Real bounded CE reallocation,mass,original mask or forward cost differs')
            if reliable:
                from .hard_pl_reliable_as_objective import view_mask
                expected=view_mask(obj.smoke_ras_q,obj.smoke_ras_pseudo,obj.smoke_ras_raw_accepted,.95)
                if (ras['calls']!=1 or ras['generated']!=128 or ras['teacher_As_views']!=128 or ras['used']+ras['raw_fallback']!=128
                    or ras['used']!=int(expected.sum()) or not torch.equal(obj.smoke_ras_used,expected)
                    or not 0<=ras['used']<=ras['raw_accepted_half']<=128 or not ras['all_raw_accepted_retained']
                    or ras['extra_student_forwards'] or ras['new_head'] or ras['EMA_updates'] or ras['U_truth_read']
                    or ras['teacher_eval_norm_updates'] or obj.ras_active or not obj.smoke_ras_exact_fallback
                    or not obj.smoke_ras_eval_norm_unchanged or not obj.smoke_ras_training_restored
                    or sum(model._cd_practical.coverage.values())!=192):
                    raise ValueError('Actual As eligibility/exactraw fallback,teacher eval mode or added forward cost differs')
            checked.append(dict(method=row['method'],model_seed=c['model_seed'],parent_epoch=parent_epoch,parent_updates=parent_epoch*222,L_forward_packets=768,
                U_seen=256,U_accepted=state['accepted'],student_U_views=state['student_views'],
                physical_U_generated=state['physical_seen'],physical_student_views=state['physical_student_views'],
                raw_U_As_seen=state['raw_As_seen'],U_As_raw_input_matches=bool(no_uas and obj.smoke_U_As_input_raw),
                source_Practical_generated=sum(model._cd_practical.coverage.values()),
                contrast_anchors=state['contrast_anchors'],shallow_middle_gradient=mechanisms,
                ordinary_raw_RDrop=mechanisms,As_KL_weight=0,weight=c['hard_pl']['weight'],threshold=.95,
                EMA_updates=0,finite_backward=True,default_head_gradient=True,feature_hooks_removed=True))
            checked[-1]['U_label_smoothing']=smoothing
            if c.get('hard_pl_mixstyle'):checked[-1]['U_style']=state['U_style']
            if c.get('hard_pl_gradient_guard'):
                checked[-1].update(U_gradient_guard=state['U_gradient_guard'],gradient_guard_details=obj.smoke_guard_details,
                    explicit_backward_matches=True,complete_L_anchor=True)
            if c.get('hard_pl_joint_contrast'):
                checked[-1].update(U_joint_contrast=state['U_joint_contrast'],
                    U_joint_signature_gradient_norm=obj.smoke_joint_signature_gradient,
                    joint160_gradient=True,joint_feature_wrapper_removed=True)
            if c.get('hard_pl_complement'):
                checked[-1].update(U_complement=state['U_complement'],complementary_gradient_norms=obj.smoke_complement_gradients,
                    no_signature_masked_gradient=True,existing_core_and_head_masked_gradient=True,CE_total_dose_unchanged=True)
            if c.get('hard_pl_path'):
                checked[-1].update(U_path_complement=state['U_path_complement'],path_core_gradient_norms=obj.smoke_path_gradients,
                    core_CE_bypasses_exclusive_signature_and_response=True,existing_core_and_head_masked_gradient=True,CE_total_dose_unchanged=True)
            if c.get('hard_pl_receiver_extrapolation'):
                checked[-1].update(U_receiver_extrapolation=state['U_receiver_extrapolation'],receiver_extrapolation_gradient_norms=obj.smoke_rxe_gradients,
                    full_signature_response_core_head_gradient=True,CE_total_dose_unchanged=True,all_raw_accepted_retained=True)
            if c.get('hard_pl_native_as'):
                checked[-1].update(U_native_As=state['U_native_As'],native_As_input_matches=True,nonphysical_ordinary_input_matches=True,
                    teacher_U_raw_views=256,teacher_U_As_views=0,teacher_U_total_views=256,all_raw_accepted_retained=True,
                    CE_and_KL_coefficients_unchanged=True)
            if c.get('hard_pl_balanced_ce'):
                checked[-1].update(U_balanced_CE=state['U_balanced_CE'],current_accepted_weights_match=True,
                    accepted_CE_mass_preserved=True,all_six_logits_reused=True,CE_delta=float(obj.smoke_bce_delta),
                    extra_teacher_backbone_head_forwards=0)
            if reliable:
                checked[-1].update(U_reliable_As=ras,teacher_U_raw_views=256,teacher_U_As_views=128,teacher_U_total_views=384,
                    all_raw_accepted_retained=True,As_gate_matches_frozen_predicate=True,exactraw_fallback=True,
                    extra_teacher_eval_normalization_unchanged=True,student_training_mode_restored=True,CE_and_KL_coefficients_unchanged=True)
        del model,obj,holder,payload,ctx;gc.collect();torch.cuda.empty_cache()
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',methods=checked,target_reads=0,U_truth_reads=0,
        probe_updates_discarded=True,formal_reload_original_parent_epoch=parent_epoch,
        formal_reload_original_F21_E30=parent_epoch==30,immediate_continuation=True))


def row(c):
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    cfg=read(c['train_config']);activate(cfg);numerical_policy()
    from .hard_pl_train import train
    train(cfg);gc.collect()
    import torch
    torch.cuda.empty_cache()
    from .source_diagnostics import run as diagnostic
    diagnostic(cfg,Path(cfg['output_root'])/'source_V_diagnostic.json')
    gc.collect();torch.cuda.empty_cache()
    from experiments.cric_dura_comparison.evaluation import predict
    predict(read(c['prediction_config']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('smoke','row','score'));parser.add_argument('--config',required=True)
    args=parser.parse_args();value=read(args.config)
    if args.action=='smoke':smoke(value)
    elif args.action=='row':row(value)
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
