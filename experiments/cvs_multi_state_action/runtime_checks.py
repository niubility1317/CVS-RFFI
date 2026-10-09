"""Actual-CVS integration checks, synthetic source-only IQ, no dataset access."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .runtime import OnlineTrainer, installed, physical_audit_role, gradient_vector
from .model import intermediate,classify_intermediate,reference_response
from experiments.cvs_multi_disentangle.checks import source_fixture,gradient_mass
from experiments.cvs_phase1_stack.runtime import blind_sample


def checks(device='cpu'):
    torch.set_num_threads(2)
    c=d.config(d.rows()[0]);args=d.make_args(c,device)
    args.num_domains=15;args.num_classes=6;args.input_len=256
    with installed(c,training=False) as native:
        torch.manual_seed(c['model_seed']);reference=native.build_baseline_model(args,torch.device(device)).eval()
    x,y,rx=source_fixture(device);x=x[:12];y=y[:12];rx=rx[:12]
    day=torch.zeros_like(y);ids=[(int(rx[i]),0,1,0,i) for i in range(len(x))]
    leo=x*1.04;records=[]
    baseline={k:v.clone() for k,v in reference.state_dict().items()}
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm))
        student=deepcopy(reference);teacher=deepcopy(reference)
        cpu=torch.random.get_rng_state().clone()
        engine=OnlineTrainer(c,student)
        assert torch.equal(cpu,torch.random.get_rng_state())
        assert all(torch.equal(v,student.state_dict()[k]) for k,v in baseline.items())
        native=F.cross_entropy(student(x),y)+F.cross_entropy(student(leo),y)
        # Include a source-U pseudo term without giving its labels to runtime.
        hidden=x.flip(0);pseudo=teacher(hidden).detach().argmax(1)
        native=native+.1*F.cross_entropy(student(hidden),pseudo)
        kwargs=dict(days=day,physical_ids=ids,leo_x=leo,native_loss=native)
        loss=engine.loss(student,teacher,x,y,rx,21,4440,**kwargs)
        assert torch.equal(cpu,torch.random.get_rng_state()),arm
        assert gradient_mass(student)==0 and gradient_mass(teacher)==0
        if arm=='native':
            assert loss is None and engine.updates==0
        else:
            assert loss is not None and torch.isfinite(loss)
            assert engine.last['physical_source_L_packets']<=32
            assert engine.last['full_native_gradient_reference']
            for kind in engine.paths:
                if kind=='receiver':continue
                assert engine.last[f'{kind}_E_real_native_norm']>0
                e_norm=engine.last[f'{kind}_E_learned_virtual_native_norm']
                assert (e_norm>0) if arm.endswith('_EG') else (e_norm==0)
                assert f'{kind}_G_C_exact_virtual_native_cosine' in engine.last
            assert gradient_mass(engine.auxiliary)==0
            (native+loss).backward()
            assert gradient_mass(student)>0 and gradient_mass(teacher)==0
            assert gradient_mass(engine.auxiliary)==0
            assert engine.last['effective_virtual_weight']+engine.last['effective_receiver_weight']<=.05+1e-9
            if engine.paths:assert engine.updates==1
            # Distinct coordinate versions reset the calibration counters and
            # old receiver contributions even after the EMA genuinely moves.
            if engine.paths:
                old_ref=engine.reference_version
                with torch.no_grad():next(teacher.id_backbone.parameters()).add_(.0001)
                engine.loss(student,teacher,x,y,rx,25,5328,days=day,physical_ids=ids,leo_x=leo)
                assert engine.reference_version==old_ref+1
                if engine.receiver is not None:
                    assert engine.receiver.reference_version==str(engine.reference_version)
                    assert all(e['step']==5328 for e in engine.receiver.entries.values())
            try:engine.loss(student,teacher,x,torch.full_like(y,-1),rx,21,4444,days=day,physical_ids=ids,leo_x=leo)
            except ValueError:pass
            else:raise AssertionError('Hidden U accepted')
        records.append(dict(arm=arm,status='PASS',updates=engine.updates,fit_steps=dict(engine.fit_steps),
            cost=engine.cost,source_only=True,native_rng_preserved=True))
    # Virtual-only detach boundary check independent of calibration admission.
    student=deepcopy(reference);h=intermediate(student.id_backbone,x)
    E=student.id_backbone.encoder.core.id_backbone
    loss=F.cross_entropy(classify_intermediate(student.id_backbone,h.detach()),y)
    loss.backward()
    assert gradient_mass(student.id_backbone)>0
    # Direct E weights exclude the existing G fusion/classifier modules.
    assert all(p.grad is None for name,p in E.named_parameters() if not name.startswith('cls_head.'))
    student.zero_grad(set_to_none=True)
    F.cross_entropy(classify_intermediate(student.id_backbone,intermediate(student.id_backbone,x)),y).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in E.named_parameters() if not name.startswith('cls_head.'))
    sample=blind_sample((x[0],5,1,{'tx':5,'tx_i':5,'rx_i':1}))
    assert sample[1]==-1 and 'tx' not in sample[3] and 'tx_i' not in sample[3]
    # The production installer compiles the exact native function and restores
    # it on exit; inference allocates no training auxiliary state.
    c=d.config(d.rows()[0])
    with installed(c):pass
    with installed(c,training=False) as n:assert not hasattr(n,'_state_action_step')
    return dict(status='PASS',device=device,arms=records,virtual_G_only_boundary=True,
        virtual_EG_boundary=True,hidden_U_mask=True,production_hook_compiles=True,
        native_reference='actual clean+LEO+source pseudo scalar',target_access=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
    a=p.parse_args();result=checks(a.device);Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
