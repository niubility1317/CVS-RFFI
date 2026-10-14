"""FakeNative + actual SIG_TIME synthetic smoke; no data reads or launches."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F

from . import design as d, model as api
from .observe import Observer, labeled_context
from experiments.cvs_u_schedule50.counterfactual import _live_snapshot, _live_readback, _rng_snapshot, _equal


def _raises(callback):
    try:callback()
    except (ValueError, RuntimeError):return
    raise AssertionError('Expected stale/double/non-L observation rejection')


def check():
    torch.set_num_threads(2)
    configs=[d.config(next(row for row in d.rows() if row['arm']==arm and row['model_seed']==d.SEEDS[0]))
             for arm in d.ARMS]
    base=api.build(configs[0], 'cpu')
    generator=torch.Generator().manual_seed(140771)
    # Seven synthetic actual updates create nontrivial moments. These are local
    # smoke fixtures, not research training or packet data.
    initial_optimizer=torch.optim.AdamW(base.parameters(),lr=2e-4,weight_decay=1e-4)
    for _ in range(7):
        logits=base(torch.randn(8,2,256,generator=generator))
        F.cross_entropy(logits,torch.arange(8)%6).backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(),5.)
        initial_optimizer.step();initial_optimizer.zero_grad(set_to_none=True)
    saved_optimizer=copy.deepcopy(initial_optimizer.state_dict())
    y=torch.arange(128)%6
    rx=torch.tensor([1,3,4,6,8])[(torch.arange(128)//6)%5]
    day=1+(torch.arange(128)//30)%3
    domain=torch.searchsorted(torch.tensor([1,3,4,6,8]),rx)*3+day-1
    meta=dict(rx_i=rx,day_i=day,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
    x=torch.randn(128,2,256,generator=generator);u=torch.randn(256,2,256,generator=generator)
    summaries=[]
    with tempfile.TemporaryDirectory(prefix='credit-observer-') as temp:
        for c in configs:
            m=copy.deepcopy(base).train()
            opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4);opt.load_state_dict(saved_optimizer)
            out=Path(temp)/c['arm'];out.mkdir()
            ob=Observer()
            native_calls=[]
            def core(output,*args,**kwargs):
                native_calls.append(output)
                return {'loss_cls':F.cross_entropy(output['tx_logits'],y)}
            sentinel_builder=lambda:None
            n=SimpleNamespace(compute_core_losses=core,_meta_from_extra=lambda extra:extra[1],
                              _build_ssdg_wisig_data=sentinel_builder)
            original_step=torch.optim.AdamW.step
            with ob.native_scope(n):
                # Matches real native ordering: observer initialization, then
                # EMA deepcopy, then first labelled context/forward/core.
                ob.initialize(m,c,out)
                assert m._phase1_labeled_feature_context.__func__ is labeled_context
                teacher=copy.deepcopy(m).eval().requires_grad_(False)
                assert teacher._phase1_labeled_feature_context.__self__ is teacher
                assert teacher._credit_pre_gain is None
                assert not hasattr(teacher,'_source_labeled_feature_hook')
                assert not hasattr(teacher,'_source_optimizer_update_probe_hook')
                assert not teacher.get_submodule('id_backbone.encoder.sequence_pool.output_norm')._forward_hooks
                assert n._build_ssdg_wisig_data is sentinel_builder
                m._feature_l_steps=7775
                with m._phase1_labeled_feature_context():
                    output=m(x,y_tx=y,return_aux=True,domain_labels=domain)
                assert m._feature_l_steps==7776 and m._credit_pre_gain.requires_grad
                assert not m.get_submodule('id_backbone.encoder.sequence_pool.output_norm')._forward_hooks
                result=n.compute_core_losses(output)
                assert native_calls==[output] and getattr(m._source_labeled_feature_hook,'_accepts_labeled_inputs',False)
                early,metrics=m._source_labeled_feature_hook(model=m,labeled_output=output,labels=y,
                    extra=(domain,meta),domains=domain,optimizer_step=7776,identity_loss=result['loss_cls'],
                    labeled_count=128,labeled_inputs=x)
                assert early.item()==0 and m._credit_pre_gain is None and ob.pending is not None
                assert torch.equal(ob.pending['x'],x)
                u_logits=m(u);pseudo=u_logits.detach().argmax(1)
                entropy=-(u_logits.log_softmax(1)*u_logits.softmax(1)).sum(1).mean()
                unlabeled=.16*F.cross_entropy(u_logits,pseudo)+.01*entropy
                total=result['loss_cls']+early+unlabeled
                parameters=[p for g in opt.param_groups for p in g['params']]
                full_base=torch.autograd.grad(total,parameters,retain_graph=True,allow_unused=True)
                L_only=torch.autograd.grad(result['loss_cls'],parameters,retain_graph=True,allow_unused=True)
                assert any(g is not None and h is not None and not torch.equal(g,h) for g,h in zip(full_base,L_only))
                captured=[]
                original_assess=__import__('experiments.cvs_crossrx_credit50.observe',fromlist=['assess']).assess
                observe_module=__import__('experiments.cvs_crossrx_credit50.observe',fromlist=['assess'])
                def inspected_assess(model,optimizer,base_grads,*args,**kwargs):
                    assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
                               for a,b in zip(base_grads,full_base))
                    captured.append(True)
                    return original_assess(model,optimizer,base_grads,*args,**kwargs)
                observe_module.assess=inspected_assess
                before=_live_snapshot(opt);before_rng=_rng_snapshot();before_buffers={k:v.clone() for k,v in m.named_buffers()}
                try:
                    late=m._source_optimizer_update_probe_hook(model=m,optimizer=opt,epoch=36,batch_idx=6,
                        phase='pseudo',optimizer_step=7776,total_loss=total,unlabeled_loss=unlabeled,
                        labeled_count=128,unlabeled_count=256)
                finally:observe_module.assess=original_assess
                assert len(captured)==5 and all(_live_readback(opt,before).values())
                assert _equal(before_rng,_rng_snapshot()) and all(torch.equal(before_buffers[k],v) for k,v in m.named_buffers())
                assert ob.pending is None and m._credit_pre_gain is None
                assert ob.counts['temporary_AdamW_steps']==12 and ob.counts['extra_eval_IQ_packets']==1024
                assert set(late)=={'post_core_auxiliary_loss','post_core_auxiliary_metrics'}
                if c['arm']=='BASE':assert late['post_core_auxiliary_loss'].item()==0
                _raises(lambda:m._source_optimizer_update_probe_hook(model=m,optimizer=opt,epoch=36,batch_idx=6,
                    phase='pseudo',optimizer_step=7776,total_loss=total,unlabeled_loss=unlabeled,
                    labeled_count=128,unlabeled_count=256))
                # Consume the returned loss exactly once, as the new native late
                # extension does, then one main backward/globalclip/AdamW step.
                (total+late['post_core_auxiliary_loss']).backward()
                torch.nn.utils.clip_grad_norm_(parameters,5.)
                opt.step();opt.zero_grad(set_to_none=True)
                assert ob.pending_update is None and ob.probe_records==[7776]
                assert ob.counts['actual_optimizer_steps']==1
                actual=json.loads((out/'actual_credit_optimizer_increment.jsonl').read_text())
                assert actual['actual_update']['actual_optimizer_steps']==1
                assert actual['selected_W_gain_gradients_None'] and all(v['W_gain_gradients_None'] for v in actual['auxiliary_gradients'].values())
                ob._write_report(36)
                report=json.loads((out/'source_training_mechanism.json').read_text())
                assert report['auxiliary_calls']==1 and report['eligible_calls']==1 and report['eligible_groups']==5
                assert report['optimizer_probe_steps']==[7776] and report['original_U_sampler_unchanged']
                summaries.append(dict(arm=c['arm'],auxiliary_loss=actual['selected_auxiliary_loss'],
                    selected_gradient_norm=actual['selected_auxiliary_gradient_norm'],
                    shadow_max_abs_error=actual['actual_update']['actual_shadow_max_abs_error'],
                    temporary_steps=ob.counts['temporary_AdamW_steps'],extra_eval_IQ=ob.counts['extra_eval_IQ_packets']))
                assert teacher._credit_pre_gain is None and not hasattr(teacher,'_source_labeled_feature_hook')
            assert torch.optim.AdamW.step is original_step and n.compute_core_losses is core
            assert n._build_ssdg_wisig_data is sentinel_builder
            assert not hasattr(m,'_source_labeled_feature_hook') and not hasattr(m,'_credit_pre_gain')
            assert m._phase1_labeled_feature_context.__func__ is not labeled_context
            assert ob.pending is None and ob.pending_update is None and ob.native is None
        # A scheduled missing-RX call is counted and skipped without altering
        # sampling, yet the fixed actual probe still measures the base update.
        c=configs[-1];m=copy.deepcopy(base).train()
        opt=torch.optim.AdamW(m.parameters());opt.load_state_dict(saved_optimizer)
        out=Path(temp)/'missing_RX';out.mkdir();ob=Observer()
        n=SimpleNamespace(compute_core_losses=lambda output:{'loss_cls':F.cross_entropy(output['tx_logits'],y)},
                          _meta_from_extra=lambda extra:extra[1],_build_ssdg_wisig_data=sentinel_builder)
        reduced_meta={k:v.clone() for k,v in meta.items()};reduced_meta['rx_i'][rx==8]=6
        reduced_domain=torch.searchsorted(torch.tensor([1,3,4,6,8]),reduced_meta['rx_i'])*3+day-1
        with ob.native_scope(n):
            ob.initialize(m,c,out);m._feature_l_steps=1119
            with m._phase1_labeled_feature_context():output=m(x,y_tx=y,return_aux=True,domain_labels=reduced_domain)
            loss=n.compute_core_losses(output)['loss_cls']
            early,_=m._source_labeled_feature_hook(model=m,labeled_output=output,labels=y,
                extra=(reduced_domain,reduced_meta),domains=reduced_domain,optimizer_step=1120,identity_loss=loss,
                labeled_count=128,labeled_inputs=x)
            late=m._source_optimizer_update_probe_hook(model=m,optimizer=opt,epoch=6,batch_idx=10,
                phase='label',optimizer_step=1120,total_loss=loss+early,unlabeled_loss=loss*0,
                labeled_count=128,unlabeled_count=0)
            assert late['post_core_auxiliary_loss'].item()==0
            assert ob.counts['active_calls']==1 and ob.counts['eligible_calls']==0
            assert ob.counts['skipped_missing_RX_calls']==1 and ob.counts['extra_eval_IQ_packets']==0
            assert ob.counts['temporary_AdamW_steps']==2
            (loss+early+late['post_core_auxiliary_loss']).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
            assert ob.probe_records==[1120] and ob.pending_update is None
            assert any(name.endswith('.cls_head.weight') for name in ob.records[0]['protected_parameter_names'])
            assert any(name.endswith('sequence_pool.gain_raw') for name in ob.records[0]['protected_parameter_names'])
            assert n._build_ssdg_wisig_data is sentinel_builder
        # A failed assessment must release pending graph/context references and
        # restore the real state even when a defective evaluator mutated it.
        m=copy.deepcopy(base).train();opt=torch.optim.AdamW(m.parameters());opt.load_state_dict(saved_optimizer)
        out=Path(temp)/'failed_assess';out.mkdir();ob=Observer()
        n=SimpleNamespace(compute_core_losses=lambda output:{'loss_cls':F.cross_entropy(output['tx_logits'],y)},
                          _meta_from_extra=lambda extra:extra[1])
        observe_module=__import__('experiments.cvs_crossrx_credit50.observe',fromlist=['assess'])
        original_assess=observe_module.assess;original_step=torch.optim.AdamW.step;original_core=n.compute_core_losses
        def defective_assess(*args,**kwargs):
            with torch.no_grad():next(m.parameters()).add_(1);next(m.buffers()).add_(1)
            opt.param_groups[0]['lr']=.9
            torch.rand(3)
            raise RuntimeError('injected actual credit failure')
        try:
            with ob.native_scope(n):
                ob.initialize(m,c,out);m._feature_l_steps=1119
                with m._phase1_labeled_feature_context():output=m(x,y_tx=y,return_aux=True,domain_labels=domain)
                loss=n.compute_core_losses(output)['loss_cls']
                early,_=m._source_labeled_feature_hook(model=m,labeled_output=output,labels=y,
                    extra=(domain,meta),domains=domain,optimizer_step=1120,identity_loss=loss,
                    labeled_count=128,labeled_inputs=x)
                before=_live_snapshot(opt);rng=_rng_snapshot();buffers={k:v.clone() for k,v in m.named_buffers()}
                observe_module.assess=defective_assess
                m._source_optimizer_update_probe_hook(model=m,optimizer=opt,epoch=6,batch_idx=10,
                    phase='label',optimizer_step=1120,total_loss=loss+early,unlabeled_loss=loss*0,
                    labeled_count=128,unlabeled_count=0)
        except RuntimeError as error:assert str(error)=='injected actual credit failure'
        else:raise AssertionError('Defective assessment silently accepted')
        finally:observe_module.assess=original_assess
        assert all(_live_readback(opt,before).values()) and _equal(rng,_rng_snapshot())
        assert all(torch.equal(buffers[k],v) for k,v in m.named_buffers())
        assert ob.pending is None and ob.pending_update is None and ob.native is None
        assert not hasattr(m,'_credit_pre_gain') and not hasattr(m,'_source_labeled_feature_hook')
        assert torch.optim.AdamW.step is original_step and n.compute_core_losses is original_core
        # Explicit fault guard restores accidentally changed live state and RNG;
        # it is never used to hide a failed actual credit observation.
        m=copy.deepcopy(base);opt=torch.optim.AdamW(m.parameters());out=Path(temp)/'guard';out.mkdir()
        ob=Observer();ob.initialize(m,configs[0],out)
        before=_live_snapshot(opt);rng=_rng_snapshot();buffer={k:v.clone() for k,v in m.named_buffers()}
        try:
            with ob._guard(m,opt):
                with torch.no_grad():next(m.parameters()).add_(1)
                next(m.parameters()).grad=torch.ones_like(next(m.parameters()))
                torch.rand(4)
                m.eval()
                raise RuntimeError('injected evaluation failure')
        except RuntimeError:pass
        assert all(_live_readback(opt,before).values()) and _equal(rng,_rng_snapshot())
        assert m.training and all(torch.equal(buffer[k],v) for k,v in m.named_buffers())
    return dict(status='VERIFIED',torch=torch.__version__,device='cpu',
        actual_SIG_TIME_model=True,seven_synthetic_moment_steps=True,EMA_deepcopy_no_observer_closure=True,
        capture_installed_before_first_core=True,Lhook_zero_late_single_addition=True,
        full_base_including_U_used_by_all_five_assessments=True,all_controls_same_eval_budget=True,
        live_parameters_optimizer_buffers_gradients_RNG_unchanged=True,
        W_gain_auxiliary_gradients_None=True,actual_once_step_shadow_match=True,
        guard_restores_on_exception=True,scope_wrappers_callbacks_cleaned=True,
        missing_RX_scheduled_skip_and_actual_base_probe=True,failed_assessment_pending_and_state_restored=True,
        source_IQ_reads=0,U_truth_read=False,target_access=False,head_fit_updates=0,
        rows=summaries,boundary='Synthetic callback smoke only, not actual research training or efficacy')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
