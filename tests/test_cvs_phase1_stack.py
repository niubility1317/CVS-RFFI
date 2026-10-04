from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from experiments.cvs_phase1_stack.design import *
from experiments.cvs_phase1_stack.runtime import installed,blind_sample


def test_matrix_and_dependency_features():
    assert len(rows())==136 and len({r['row_id'] for r in rows()})==136
    assert features('r3','orth',['leo'])==['domain','leo','orth']
    assert 'muse' in features('r5','base',['leo'])
    assert not {'muse','rc4','pseudo'}&set(features('r6','no_pseudo',[]))


def test_source_config_rejects_unregistered_changes():
    c=config(rows()[0],['leo']);validate(c)
    for k,v in [('epochs',3),('target_access',True),('checkpoint_sources',['old.pt']),('output_root','bad')]:
        b=deepcopy(c);b[k]=v
        with pytest.raises(ValueError):validate(b)


def test_unlabeled_sample_has_no_tx_truth():
    s=blind_sample((torch.ones(2,256),4,2,{'tx_i':4,'tx':'secret','rx_i':3,'base_index':123}))
    assert s[1]==-1 and s[2]==2 and s[3]==dict(rx_i=3,base_index=123,tx_label_visible=False)


def test_incomplete_native_updates_cannot_be_frozen():
    from experiments.cvs_phase1_stack.source import require_budget
    require_budget(44400,44400)
    for logged,successful in [(44400,44399),(44399,44400),(0,0)]:
        with pytest.raises(ValueError,match='Incomplete update budget'):require_budget(logged,successful)


@pytest.mark.parametrize('stage,arm',[('r2','bridge'),('r2','pseudo'),('r3','all_dg'),('r4','all_open'),('r5','both'),('r6','native_full')])
def test_native_selected_identity_forward_gradient_and_roundtrip(stage,arm,tmp_path):
    torch.set_num_threads(2)
    row=next(r for r in rows() if r['stage']==stage and r['arm']==arm)
    c=config(row,['leo']);a=make_args(c,'cpu');a.input_len=256;a.num_domains=15
    with installed(c) as n:
        m=n.build_baseline_model(a,torch.device('cpu')).train()
        x=torch.randn(6,2,256);y=torch.arange(6);d=torch.arange(6)
        out=m(x,y_tx=y,domain_labels=d,return_aux=True)
        assert out['z_id'].shape==(6,160) and out['z_dom'].shape==(6,160)
        loss=torch.nn.functional.cross_entropy(out['tx_logits'],y)
        loss=loss+.1*torch.nn.functional.cross_entropy(out['dom_logits'],d)
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        m.eval();before=m(x).detach();path=tmp_path/'m.pt';torch.save(m.state_dict(),path)
        other=n.build_baseline_model(a,torch.device('cpu'));other.load_state_dict(torch.load(path,weights_only=True));other.eval()
        torch.testing.assert_close(before,other(x),atol=0,rtol=0)


def test_all_native_args_have_exact_mechanism_and_source_flags():
    for stage,arms in STAGES.items():
        for arm in arms:
            row=next(r for r in rows() if r['stage']==stage and r['arm']==arm)
            c=config(row,['leo']);a=make_args(c,'cpu')
            assert a.from_scratch and not a.baseline_ckpt and not a.teacher_ckpt and a.a1_source_screen_only
            assert a.checkpoint_selection=='final_only' and not a.amp
            assert a.use_adv3b02_daot_stn==('daot' in c['features'])
            assert a.fasttrust_rc4==('rc4' in c['features'])
            for feature,weights in WEIGHTS.items():
                for key,val in weights.items():assert getattr(a,key)==(val if feature in c['features'] else 0.)
