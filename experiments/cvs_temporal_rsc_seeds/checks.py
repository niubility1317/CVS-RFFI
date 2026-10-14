import ast
import copy
import importlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .model import build
from .dispatch import restore_project
from experiments.cvs_temporal_rsc50.checks import native_call
from experiments.cvs_temporal_rsc50.observer import Observer
from experiments.cvs_temporal_rsc50.objective import ce

def equal(a,b):
    return a==b or (isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b))

def checks():
    torch.set_num_threads(2);contexts=[];negatives=0
    mechanism_keys=('feature_combination','temporal_rsc','hypothesis_basis','source_falsification','goal','cost_claim','training_hypothesis_parent_commit')
    public=json.loads(json.dumps([d.config(r) for r in d.rows()]).replace(d.PROJECT,'PROJECT'))
    assert restore_project(public,d.PROJECT)==[d.config(r) for r in d.rows()]
    for seed in d.SEEDS:
        initial=None
        for row in d.training_rows():
            if row['model_seed']!=seed:continue
            c=d.config(row);d.validate(c)
            first=d.first.config(next(r for r in d.first.rows() if r['arm']==row['arm']))
            assert all(c[k]==first[k] for k in mechanism_keys)
            a,original=d.make_args(c,'cpu'),d.first.make_args(first,'cpu')
            changed={k for k in vars(a).keys()|vars(original).keys() if not equal(vars(a).get(k),vars(original).get(k))}
            assert not changed-{'seed','run_id','candidate_id','output_dir'},sorted(changed)
            m=build(c,'cpu').train();values={k:v.clone() for k,v in m.state_dict().items()}
            if initial is None:initial=values
            else:assert set(initial)==set(values) and all(torch.equal(v,values[k]) for k,v in initial.items())
            gen=torch.Generator().manual_seed(2026307403)
            x=torch.randn(12,2,256,generator=gen);y=torch.arange(12)%6;rx=torch.tensor([1,3,4,6,8,1]*2)
            with tempfile.TemporaryDirectory(prefix='rsc-seed-context-') as temporary:
                obs=Observer(design=d);obs.initialize(m,c,Path(temporary));obs.steps=15
                n=SimpleNamespace(_labeled_identity_forward_hook=None,_labeled_identity_loss_hook=None,_trusted_pseudo_loss_hook=None)
                ns,compiled,tree=native_call()
                loss_hook=next(node for node in ast.walk(tree) if isinstance(node,ast.If) and ast.unparse(node.test)=='_labeled_identity_loss_hook is not None')
                compiled_loss=compile(ast.Module(body=[loss_hook],type_ignores=[]),'<actual-native-L-loss-hook>','exec')
                with obs.native_scope(n):
                    ns.update(vars(n));ns.update(forward_model=m,model=m,args=a,epoch=1,batch_idx=16,x_l_main=x,y_l=y,
                        receiver_l_base=rx,labeled_clean_count=len(y),d_l=torch.zeros_like(y),concat_sat_clean_bsz=0,concat_sat_full_batch=False)
                    exec(compiled,ns)
                    selected=ce(ns['out_l']['tx_logits'],y);ns['loss_tx_l']=selected
                    exec(compiled_loss,ns)
                    assert ns['loss_tx_l'] is selected and obs.steps==16 and len(obs.grad)==1
                    assert m._feature_l_steps==1 and m._feature_extra_l_steps==2
                    vectors=obs.grad[0]['vectors']
                    assert all(torch.isfinite(g).all() and g.norm()>1e-12 for g in vectors['selected'])
                    assert all(float(g.norm())==0 for g in vectors['delta']) if row['arm']=='SIG_TIME' else all(g.norm()>1e-12 for g in vectors['delta'])
                    U=selected*0
                    assert n._trusted_pseudo_loss_hook(model=m,args=a,epoch=1,batch_idx=16,loss_l=selected,loss_u=U,loss_ent=U,active=False,logits=None,pseudo=None,mask=None) is U
                assert all(v is None for v in vars(n).values())
                assert not m.id_backbone.encoder.core.id_backbone.t2._forward_hooks and not getattr(m,'_feature_extra_l_forward',False)
            contexts.append(dict(row_id=row['row_id'],native_difference_keys=sorted(changed),scientific_dictionary_identical=True,
                actual_same_seed_initial_state=True,actual_native_forward_and_loss_hooks=True,actual_t1_t2_SIG_W_joint_gradients=True,
                actual_original_U_loss_object=True,temporary_mask_and_hooks_cleared=True,public_full_config_roundtrip=True))
            bad=copy.deepcopy(c);bad['temporal_rsc']['mask_seed']+=1
            try:d.validate(bad)
            except ValueError:negatives+=1
            else:raise AssertionError('Changed remaining RSC formula accepted')
    qualification=dict(status='PASS',source_only=True,target_metrics_read_for_decision=False,
        fixed_rules=d.first.config(d.first.training_rows()[0])['source_falsification'],
        predefined_remaining_seeds=list(d.SEEDS),expand_all_three_arms_unchanged=True)
    d.validate_qualification(qualification)
    for key,value in [('status','FAIL'),('source_only',False),('target_metrics_read_for_decision',True),
                       ('fixed_rules',{}),('predefined_remaining_seeds',[2026092702]),('expand_all_three_arms_unchanged',False)]:
        bad=copy.deepcopy(qualification);bad[key]=value
        try:d.validate_qualification(bad)
        except ValueError:negatives+=1
        else:raise AssertionError('Invalid original source expansion state accepted')
    for name in ('source','predict','score','feature_check','audit','dispatch','preflight','publish'):importlib.import_module(f'experiments.cvs_temporal_rsc_seeds.{name}')
    from .publish import configure,transport
    configure();compile(transport.REMOTE,'<rsc-seeds-remote>','exec');compile(transport.INSPECT,'<rsc-seeds-inspect>','exec')
    assert 'experiments.cvs_temporal_rsc_seeds.preflight' in transport.REMOTE
    assert 'experiments.cvs_temporal_rsc_seeds.dispatch' in transport.REMOTE
    assert all(p in transport.PATHS for p in ('experiments/cvs_backbone_fishr_seeds','experiments/cvs_source_self_challenge','experiments/cvs_temporal_rsc50','experiments/cvs_temporal_rsc_seeds'))
    return dict(status='PASS',adapter_only_checks=True,unchanged_first_candidate_tests_not_repeated=True,negatives=negatives,
        native_contexts=contexts,real_IQ_checkpoint_or_first_qualification_read=False,first_seed_retrained=False,
        new_scientific_hypothesis=False,new_models=6,historical_baselines_reused=2,head_fit_updates=0)

if __name__=='__main__':print(json.dumps(checks()))
