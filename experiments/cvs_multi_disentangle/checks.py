"""Focused synthetic checks for design parity; no source/query data reads."""
import argparse
from copy import deepcopy
import io
import json
import torch
from experiments.cvs_multi_disentangle import design as d
from experiments.cvs_multi_disentangle import model as m
from experiments.cvs_multi_disentangle import physics as p
from experiments.cvs_phase1_stack.runtime import installed as legacy_installed
from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags


def gradient_mass(module):
    gradients = [v.grad for v in module.parameters() if v.grad is not None]
    assert all(torch.isfinite(v).all() for v in gradients), 'Nonfinite gradient'
    return sum(float(v.abs().sum()) for v in gradients)


def identity(c, device):
    args = d.make_args(c, device)
    args.num_domains = 15; args.num_classes = 6; args.input_len = 256
    with legacy_installed(c) as native:
        torch.manual_seed(c['model_seed'])
        return native.build_baseline_model(args, torch.device(device))


def physics_checks(x):
    cpu = torch.random.get_rng_state().clone()
    cuda = [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    generator = torch.Generator(device='cpu').manual_seed(71403)
    views = p.factorial_batch(x, generator)
    assert torch.equal(cpu, torch.random.get_rng_state()), 'Physical draw changed native CPU RNG'
    assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
    assert torch.equal(p.apply_linear(x, torch.zeros(len(x),p.LINEAR_PARAMETER_DIM,device=x.device)),x)
    assert torch.equal(p.apply_temporal(x, torch.zeros(len(x),p.TEMPORAL_PARAMETER_DIM,device=x.device)),x)
    assert torch.equal(views['x11'],p.apply_temporal(views['x10'],views['temporal_parameters']))
    opposite = p.apply_linear(views['x01'],views['linear_parameters'])
    order_gap = float((opposite-views['x11']).abs().max())
    assert order_gap>1e-6, 'Chosen physical interventions accidentally commute'
    factorial = views['x11']-views['x10']-views['x01']+views['x00']
    assert float(factorial.abs().sum())>0 and torch.isfinite(factorial).all()
    for key in ('x10','x01','x11'):
        assert not torch.equal(x,views[key]) and torch.isfinite(views[key]).all()
    return views,dict(private_physical_rng=True,zero_interventions_exact=True,
                      ordered_factorial=True,order_difference_max=order_gap)


def component_checks(model,x,views):
    model.eval();core=model.id_backbone
    with torch.no_grad():
        h=m.intermediate(core,x)
        direct_features=core.encoder.features(x)
        replay=m.identity_from_intermediate(core,h)
        direct_logits=core.encoder.classify_features(direct_features)
        logits=m.classify_intermediate(core,h)
    assert h.shape==(len(x),m.INTERMEDIATE_DIM)
    assert torch.equal(replay,direct_features), dict(split_feature_difference=float((replay-direct_features).abs().max()))
    assert torch.equal(logits,direct_logits), 'Intermediate G changed native identity classifier'
    recipe=d.RECIPE
    auxiliary=m.AuxiliaryNetworks(rank=recipe['rank'],state_dim=recipe['code_dim'],include_interaction=True).to(x.device)
    branches=[auxiliary.branch(k) for k in m.KINDS]
    pointer_sets=[{v.data_ptr() for v in branch.parameters()} for branch in branches]
    assert all(not a&b for i,a in enumerate(pointer_sets) for b in pointer_sets[i+1:]), 'Shared auxiliary parameters'
    assert not set(v.data_ptr() for v in model.parameters())&set(v.data_ptr() for v in auxiliary.parameters())
    rows=[]
    targets={'linear':'x10','temporal':'x01','receiver':'x10','interaction':'x11'}
    for kind in m.KINDS:
        branch=auxiliary.branch(kind);model.zero_grad(set_to_none=True);auxiliary.zero_grad(set_to_none=True)
        with torch.no_grad(): target=m.intermediate(core,views[targets[kind]])-h
        descriptor=dict(x_linear=views['x10'],x_temporal=views['x01']) if kind=='interaction' else None
        predicted=auxiliary.predict(kind,x,views[targets[kind]],h.detach(),descriptor=descriptor)
        assert predicted['q'].shape==(len(x),recipe['code_dim'])
        assert float(predicted['q'].detach().abs().sum())>0
        known=(views['linear_parameters'] if kind=='linear' else views['temporal_parameters'] if kind=='temporal'
               else None)
        loss=(predicted['delta']-target.detach()).square().mean()
        if known is not None: loss=loss+(predicted['parameters']-known).square().mean()
        loss.backward()
        assert gradient_mass(model)==0, 'Auxiliary fit reached identity parameters'
        assert gradient_mass(branch.encoder)>0 and gradient_mass(branch.operator)>0, kind
        assert all(gradient_mass(other)==0 for other in branches if other is not branch)
        with torch.no_grad():
            zero_descriptor=dict(x_linear=x,x_temporal=x) if kind=='interaction' else None
            zero=auxiliary.predict(kind,x,x,h,descriptor=zero_descriptor)
            assert torch.count_nonzero(zero['q'])==0 and torch.count_nonzero(zero['delta'])==0
            if kind=='interaction':
                for one_factor in (dict(x_linear=x,x_temporal=views['x01']),
                                   dict(x_linear=views['x10'],x_temporal=x)):
                    absent=auxiliary.predict(kind,x,views['x11'],h,descriptor=one_factor)
                    assert torch.count_nonzero(absent['q'])==0 and torch.count_nonzero(absent['delta'])==0
            cap=recipe['residual_bound']*torch.sqrt(h.square().sum(1)+1e-8)
            big=auxiliary.propose(kind,h,1000*torch.randn_like(predicted['q']))
            assert (big.norm(dim=1)<=cap*(1+1e-5)).all(), 'Conditional action exceeds registered bound'
            reference=predicted['delta'].clone()
            branch.encoder[-1].weight.zero_()
            without_encoder=auxiliary.predict(kind,x,views[targets[kind]],h,descriptor=descriptor)['delta']
            assert not torch.equal(reference,without_encoder), 'Action bypassed the learned nuisance encoder'
        rows.append(dict(branch=kind,encoder_gradient=True,operator_gradient=True,independent=True,
                         zero_state_exact_identity=True,learned_code_used=True,bounded=True))
    # An identity objective uses the learned predicted change as frozen evidence.
    model.zero_grad(set_to_none=True);auxiliary.zero_grad(set_to_none=True)
    h_current=m.intermediate(core,x)
    with torch.no_grad(): delta=auxiliary.predict('temporal',x,views['x01'],h_current.detach())['delta']
    ce=torch.nn.functional.cross_entropy(m.classify_intermediate(core,h_current+delta.detach()),
                                        torch.arange(len(x),device=x.device)%6)
    ce.backward()
    assert gradient_mass(core)>0 and gradient_mass(auxiliary)==0
    return dict(native_intermediate_exact=True,auxiliary_components=rows,
                isolated_aux_fit=True,isolated_identity_constraint=True)


def inference_checks(model,x):
    model.eval();state={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        all_logits=model(x)
        separate=torch.cat([model(v[None]) for v in x])
        difference=float((all_logits-separate).abs().max())
        assert torch.allclose(all_logits,separate,atol=2e-5,rtol=2e-5), difference
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
        clone=deepcopy(model);blob=io.BytesIO();torch.save(state,blob);blob.seek(0)
        clone.load_state_dict(torch.load(blob,map_location=x.device,weights_only=False),strict=True)
        assert torch.equal(clone(x),all_logits)
    return dict(identity_only_stateless=True,batch_independence_max_difference=difference,strict_checkpoint_roundtrip=True)


def source_fixture(device='cpu'):
    """Distinct synthetic source packets with matched observable quality bins."""
    import numpy as np
    from experiments.cvs_rff_physics.known_excitation import template
    generator=torch.Generator().manual_seed(36014)
    base=.25*torch.randn(6,2,256,generator=generator)
    public=template(np.arange(80))
    base[:,0,80:160]=torch.tensor(public.real.tolist())
    base[:,1,80:160]=torch.tensor(public.imag.tolist())
    packets=[];labels=[];receivers=[]
    for receiver in (1,3):
        for label in range(6):
            for repetition in range(2):
                # Outside-window content makes packet IDs distinct; a mild
                # amplitude response is visible to the RX observable encoder.
                packet=base[label].clone()*(1 if receiver==1 else 1.06)
                packet[:,:80]+=.001*torch.randn(2,80,generator=generator)
                if receiver==3:
                    packet=p.apply_linear(packet[None],torch.tensor([[.2,-.1,.1,.15]]))[0]
                    packet=p.apply_temporal(packet[None],torch.tensor([[.2,.2,.1]]))[0]
                packets.append(packet);labels.append(label);receivers.append(receiver)
    return (torch.stack(packets).to(device),torch.tensor(labels,device=device),
            torch.tensor(receivers,device=device))


def runtime_checks(reference,device):
    from experiments.cvs_multi_disentangle.runtime import MultiTrainer, ReceiverStatistics, installed
    records=[];x,y,rx=source_fixture(device)
    reference.eval()
    with torch.no_grad(): baseline=reference(x)
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm))
        student=deepcopy(reference);teacher=deepcopy(reference)
        student.zero_grad(set_to_none=True);teacher.zero_grad(set_to_none=True)
        student.train();before={k:v.clone() for k,v in student.state_dict().items()}
        cpu=torch.random.get_rng_state().clone()
        cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        engine=MultiTrainer(c,student)
        assert torch.equal(cpu,torch.random.get_rng_state()), 'Auxiliary initialization changed native CPU RNG'
        assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())), 'Auxiliary initialization changed CUDA RNG'
        assert all(torch.equal(v,student.state_dict()[k]) for k,v in before.items())
        assert not any(k.startswith('auxiliary') for k in student.state_dict())
        loss=engine.loss(student,teacher,x,y,rx,epoch=21,step=4440)
        assert student.training, 'Auxiliary deterministic forward leaked eval mode'
        assert torch.equal(cpu,torch.random.get_rng_state()), 'Auxiliary fit changed native CPU RNG'
        assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())), 'Auxiliary fit changed CUDA RNG'
        assert gradient_mass(student)==0 and gradient_mass(teacher)==0, 'Actual auxiliary fit reached identity/EMA'
        if arm=='legacy_cosine':
            assert loss is None and engine.auxiliary is None and engine.updates==0 and not engine.statistics.groups
        else:
            assert loss is not None and torch.isfinite(loss)
            if engine.auxiliary is not None:
                assert engine.updates==1 and gradient_mass(engine.auxiliary)>0
                if arm!='unified':
                    for kind in engine.paths:
                        assert gradient_mass(engine.auxiliary.branch(kind))>0, (arm,kind,engine.last)
                engine.auxiliary.zero_grad(set_to_none=True)
            loss.backward()
            assert gradient_mass(student.id_backbone)>0, (arm,engine.last)
            assert gradient_mass(teacher)==0
            if engine.auxiliary is not None: assert gradient_mass(engine.auxiliary)==0, 'Identity objective reached auxiliary parameters'
            if 'receiver' in engine.paths:
                assert engine.last['receiver_pairs']>0 and engine.last['receiver_relations']>0, (arm,engine.last)
                assert engine.last['receiver_target_requires_grad']==0
                assert engine.last['receiver_main_subtraction_enabled']==int(arm!='receiver')
                if arm!='receiver': assert engine.last['receiver_main_action_norm']>0
                checkpoint=engine.statistics.state_dict()
                assert checkpoint['sample_cache'] is False and checkpoint['scope']=='source_L_aggregate_statistics_only'
                assert all(set(group)=={'view','h','count','step'} and group['count']>=2
                           for group in checkpoint['groups'].values())
                assert all(not group['h'].requires_grad and group['h'].device.type=='cpu'
                           for group in checkpoint['groups'].values())
                # Observe the target passed to the real fitting routine. It
                # must remove detached L/T predictions from the total grouped
                # RX difference, including in the unified capability control.
                from experiments.cvs_multi_disentangle import runtime as runtime_module
                pairs=engine.statistics.pairs(torch.device(device),32)
                explained=engine._receiver_main_actions(pairs)
                observed=torch.stack([pair['h1']-pair['h0'] for pair in pairs])
                wanted=(observed-explained).detach()
                captured=[];original_error=runtime_module.normalized_error
                def observed_error(prediction,target):
                    captured.append(target.detach().clone())
                    return original_error(prediction,target)
                with torch.no_grad():
                    batch=p.factorial_batch(x,engine.generator)
                    anchor={key:m.intermediate(teacher.id_backbone,batch[key]).detach()
                            for key in ('x00','x10','x01','x11')}
                runtime_module.normalized_error=observed_error
                try: engine._fit(batch,anchor,pairs,4444)
                finally: runtime_module.normalized_error=original_error
                assert torch.equal(captured[engine.paths.index('receiver')],wanted), 'DR fit did not remove L/T main actions'
            auxiliary_checkpoint=engine.checkpoint(200)
            assert auxiliary_checkpoint['inference_uses_auxiliary'] is False and not auxiliary_checkpoint['checkpoint_sources']
            if engine.auxiliary is not None:
                copy_engine=MultiTrainer(c,student)
                copy_engine.auxiliary.load_state_dict(auxiliary_checkpoint['auxiliary'],strict=True)
        # Query-side wrapper must allocate only the unchanged identity model.
        a=d.make_args(c,device);a.num_domains=15;a.num_classes=6;a.input_len=256
        with installed(c,training=False) as native:
            torch.manual_seed(c['model_seed'])
            infer=native.build_baseline_model(a,torch.device(device)).eval()
            with torch.no_grad(): assert torch.equal(infer(x),baseline), 'Registered arm changed deployment identity'
        records.append(dict(arm=arm,status='PASS',auxiliary_updates=engine.updates,
                            native_rng_preserved=True,aux_fit_identity_gradient=False,
                            identity_auxiliary_gradient=False,identity_only_inference=True,
                            receiver_relations=engine.last.get('receiver_relations')))
    # Metadata rejection and order invariance prove real sufficient statistics,
    # rather than pairing packets by minibatch position.
    stats=ReceiverStatistics(d.RECIPE)
    views=torch.zeros(12,197,device=device);views[:,24]=.5
    labels=torch.arange(12,device=device)%3;receivers=torch.tensor([1]*6+[3]*6,device=device)
    anchors=torch.arange(12*m.INTERMEDIATE_DIM,device=device,dtype=torch.float32).reshape(12,-1)/1000
    stats.update(labels,receivers,views,anchors,4440)
    reversed_stats=ReceiverStatistics(d.RECIPE);order=torch.arange(11,-1,-1,device=device)
    reversed_stats.update(labels[order],receivers[order],views[order],anchors[order],4440)
    assert stats.groups.keys()==reversed_stats.groups.keys()
    for key in stats.groups:
        assert torch.equal(stats.groups[key]['h'],reversed_stats.groups[key]['h'])
    try: stats.update(torch.full_like(labels,-1),receivers,views,anchors,4441)
    except ValueError: pass
    else: raise AssertionError('Hidden U truth was accepted by receiver statistics')
    return dict(rows=records,group_statistics_order_invariant=True,hidden_U_rejected=True)


def checks(device='cpu'):
    torch.set_num_threads(2)
    with numerical_context(d.FULL_FP32_POLICY):
        assert len(d.rows())==32 and len({r['row_id'] for r in d.rows()})==32
        c=d.config(d.rows()[0]);model=identity(c,device)
        x=torch.randn(12,2,256,generator=torch.Generator().manual_seed(7301)).to(device)
        views,physical=physics_checks(x)
        components=component_checks(model,x,views)
        inference=inference_checks(model,x)
        runtime=runtime_checks(identity(c,device),device)
        return dict(status='PASS',device=device,backend_flags=actual_flags(),physical=physical,
                    components=components,inference=inference,runtime=runtime,target_read=False,formal_weights=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=checks(args.device);d.write(args.output,result);print(json.dumps(result))
