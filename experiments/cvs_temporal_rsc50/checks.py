import ast
import copy
import importlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .model import build
from .observer import Observer
from .objective import feature_mask,selected_masks,ce

def native_call():
    tree=ast.parse((d.ROOT/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py').read_text(encoding='utf8'))
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_labeled_identity')
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='out_l' and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_forward_labeled_identity')
    ns={'_labeled_identity_forward_hook':None,'torch':torch}
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'<actual-native-forward-helper>','exec'),ns)
    code=compile(ast.Module(body=[assignment],type_ignores=[]),'<actual-native-labelled-forward>','exec')
    return ns,code,tree

def invoke(m,a,x,y,rx,hook=None):
    ns,code,_=native_call()
    ns.update(_labeled_identity_forward_hook=hook,forward_model=m,model=m,args=a,epoch=1,batch_idx=1,
        x_l_main=x,y_l=y,receiver_l_base=rx,labeled_clean_count=len(y),d_l=torch.zeros_like(y),
        concat_sat_clean_bsz=0,concat_sat_full_batch=False)
    exec(code,ns);return ns['out_l']

def exercise(m,c,*,optimizer_update):
    a=d.make_args(c,'cpu');m.train()
    gen=torch.Generator().manual_seed(2026307403)
    x=torch.randn(12,2,256,generator=gen);y=torch.arange(12)%6;rx=torch.tensor([1,3,4,6,8,1]*2)
    plain=copy.deepcopy(m);teacher_plain=copy.deepcopy(m);teacher_probed=copy.deepcopy(m)
    for teacher in (teacher_plain,teacher_probed):teacher.requires_grad_(False)
    rng=torch.get_rng_state().clone()
    out_plain=invoke(plain,a,x,y,rx);loss_plain=ce(out_plain['tx_logits'],y)
    after_plain=torch.get_rng_state().clone()
    before_buffers={k:v.clone() for k,v in plain.named_buffers()}
    with tempfile.TemporaryDirectory(prefix='rsc-check-') as root:
        obs=Observer();obs.initialize(m,c,Path(root));torch.set_rng_state(rng)
        out=invoke(m,a,x,y,rx,obs.forward)
        assert torch.equal(after_plain,torch.get_rng_state())
        assert set(before_buffers)==set(dict(m.named_buffers()))
        assert all(torch.equal(v,dict(m.named_buffers())[k]) for k,v in before_buffers.items()),'Extra forwards changed model buffers'
        loss=ce(out['tx_logits'],y)
        returned=obs.labeled(model=m,args=a,epoch=1,batch_idx=1,logits=out['tx_logits'],labels=y,receivers=rx,clean_count=12,original_loss=loss,role='L_s')
        assert returned is loss and obs.steps==1 and obs.extra_forwards==2
        assert m._feature_l_steps==plain._feature_l_steps and m._feature_extra_l_steps==2
        named=dict(m.named_parameters());W=m.id_backbone.encoder.core.id_backbone.cls_head.weight
        grad=torch.autograd.grad(loss,[*[named[n] for n in d.PARAMETERS],W],retain_graph=True)
        assert all(torch.isfinite(g).all() and g.norm()>1e-12 for g in grad)
        U=loss*0
        assert obs.unlabeled(model=m,args=a,epoch=1,batch_idx=1,loss_l=loss,loss_u=U,loss_ent=U,active=False,logits=None,pseudo=None,mask=None) is U
        assert not getattr(m,'_feature_extra_l_forward',False) and not m.id_backbone.encoder.core.id_backbone.t2._forward_hooks
        if c['arm']=='SIG_TIME':
            assert torch.equal(out['tx_logits'],out_plain['tx_logits']) and torch.equal(loss,loss_plain)
            if optimizer_update:
                opt_plain=torch.optim.AdamW(plain.parameters(),lr=.0002,weight_decay=.0001)
                opt_probed=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001)
                loss_plain.backward();loss.backward()
                assert all((p.grad is None)==(q.grad is None) and (p.grad is None or torch.equal(p.grad,q.grad)) for p,q in zip(plain.parameters(),m.parameters()))
                opt_plain.step();opt_probed.step()
                assert all(torch.equal(v,m.state_dict()[k]) for k,v in plain.state_dict().items())
                _,_,tree=native_call();ema=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_update_ema_model')
                namespace={'torch':torch};exec(compile(ast.Module(body=[ema],type_ignores=[]),'<actual-native-EMA>','exec'),namespace)
                namespace['_update_ema_model'](teacher_plain,plain,.999)
                namespace['_update_ema_model'](teacher_probed,m,.999)
                assert all(torch.equal(v,teacher_probed.state_dict()[k]) for k,v in teacher_plain.state_dict().items())
                for s,t in zip(opt_plain.state.values(),opt_probed.state.values()):
                    assert set(s)==set(t) and all(torch.equal(v,t[k]) if torch.is_tensor(v) else v==t[k] for k,v in s.items())
        else:
            original_grad=torch.autograd.grad(loss_plain,[dict(plain.named_parameters())[n] for n in d.PARAMETERS],retain_graph=True)
            assert all((u-v).norm()>1e-12 for u,v in zip(original_grad,grad[:3]))
        obs.stream.close()
    return dict(arm=c['arm'],actual_native_forward_hook=True,ambient_rng_unchanged=True,model_buffers_equal_to_original=True,
        actual_t1_t2_SIG_and_W_gradients_nonzero=True,extra_forward_counter=2,native_context_counter=1,
        optimizer_and_EMA_equivalence=(optimizer_update and c['arm']=='SIG_TIME'),head_fit_updates=0)

def checks():
    torch.set_num_threads(2);negatives=0
    g=torch.randn(12,96,32,generator=torch.Generator().manual_seed(51));logits=torch.randn(12,6);y=torch.arange(12)%6
    mask=feature_mask(g)
    for role in ('U_s','V','query'):
        for call in (lambda:feature_mask(g,role=role),lambda:selected_masks(logits,logits*.7,y,mask,1,role=role)):
            try:call()
            except ValueError:negatives+=1
            else:raise AssertionError('Forbidden masking role accepted')
    for key,value in [('role','query'),('mask_seed',1),('enable','later'),('layer','joint160')]:
        bad=d.config(d.training_rows()[0]);bad['temporal_rsc'][key]=value
        try:d.validate(bad)
        except ValueError:negatives+=1
        else:raise AssertionError('Changed fixed RSC configuration accepted')
    from .dispatch import restore_project
    registered=[d.config(r) for r in d.rows()]
    public=json.loads(json.dumps(registered).replace(d.PROJECT,"PROJECT"))
    assert restore_project(public,d.PROJECT)==registered
    assert restore_project({"path":"OTHER_PROJECT/source"},d.PROJECT)!={"path":d.PROJECT+"/source"}
    initial=None;actual=[]
    for row in d.training_rows():
        c=d.config(row);a=d.make_args(c,'cpu')
        assert (a.epochs,a.label_epochs,a.pseudo_epochs,a.lr,a.weight_decay)==(50,33,17,.0002,.0001)
        assert a.use_unlabeled and a.use_ema_teacher and not a.freeze_backbone
        m=build(c,'cpu');values={k:v.clone() for k,v in m.state_dict().items()}
        if initial is None:initial=values
        else:assert set(initial)==set(values) and all(torch.equal(v,values[k]) for k,v in initial.items())
        actual.append(exercise(m,c,optimizer_update=True))
    for name in ('source','predict','score','feature_check','audit','preflight','dispatch','publish'):importlib.import_module(f'experiments.cvs_temporal_rsc50.{name}')
    from .publish import configure,transport
    configure();compile(transport.REMOTE,'<rsc-remote>','exec');compile(transport.INSPECT,'<rsc-inspect>','exec')
    assert 'experiments/cvs_source_self_challenge' in transport.PATHS
    assert 'experiments.cvs_temporal_rsc50.preflight' in transport.REMOTE and 'experiments.cvs_temporal_rsc50.dispatch' in transport.REMOTE
    return dict(status='PASS',protocol_and_config_negatives=negatives,three_initial_states_equal=True,
        actual_native_call_checks=actual,real_IQ_or_checkpoint_read=False,head_fit_updates=0)

if __name__=='__main__':print(json.dumps(checks()))
