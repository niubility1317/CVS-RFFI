import torch
from torch import nn
import pytest
from types import SimpleNamespace

from experiments.pamf_lora_main import w24_joint_pamf as experiment


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 10)

    def forward(self, iq):
        return self.head(iq.mean((-1,1))),


class Tracker:
    def aperture(self, raw):
        return raw.unflatten(-1,(8,1024))[..., :432]


def test_unlabeled_iq_changes_only_restoration_losses(monkeypatch):
    torch.manual_seed(9)
    raw = torch.randn(4,8192,dtype=torch.complex64)
    labels = torch.tensor([0,6])
    g = nn.Parameter(torch.tensor(.2))
    f = Reader()
    anchor = Reader().requires_grad_(False)
    monkeypatch.setattr(experiment,'wide_delta',lambda g,tracker,iq: g * iq[..., :384])
    generic = lambda x,rng: x + .1 * torch.randn(x.shape, dtype=x.dtype, generator=rng)

    def parts(iq):
        return experiment.loss_parts(g,f,anchor,Tracker(),generic,iq,labels,torch.Generator().manual_seed(3))

    baseline = parts(raw)
    changed = raw.clone()
    changed[2:] *= 1j
    changed[2:, :4000] = -8 * changed[2:, :4000]
    replacement = parts(changed)
    torch.testing.assert_close(baseline[:3],replacement[:3],rtol=0.,atol=0.)
    assert not torch.isclose(baseline[3],replacement[3])
    gradients = torch.autograd.grad(baseline[0],(g,f.head.weight))
    assert all(x.abs().sum() > 0 for x in gradients)

    with pytest.raises(ValueError,match='equal labeled and unlabeled'):
        parts(raw[:3])


def test_only_the_frozen_pamf_candidate_can_open_the_formal_gate():
    fit = {k:k for k in experiment.FIT_FIELDS}
    fit.update(native_target_root='target', validation='V')
    out = experiment.allowed_fit_runtime(fit,'out')
    assert set(out) == set(experiment.FIT_FIELDS)
    assert out['run_root'] == 'out'
    scores = {m:dict(six_correct=1400,low_urban_correct=170,by_scene={'clean':297}) for m in experiment.MODES}
    scores['joint']['six_correct'] = 1600
    assert not experiment.qualifies(scores)
    scores[experiment.CANDIDATE]['six_correct'] = 1401
    assert experiment.qualifies(scores)
    for case in experiment.FULL_CASES:
        proof = experiment.formal_receipt(case)
        assert proof['modes'][-1] == experiment.CANDIDATE and 'joint' not in proof['modes']
        assert proof['processing_route'] == 'full_native' and not proof['target_truth_read']


def test_clean_pl_rejects_uncertain_teacher_and_trains_only_accepted_restored_iq():
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.))

        def forward(self, clean):
            return clean * self.weight,

    teacher = Teacher()
    reader = nn.Linear(3, 3)
    student_input = torch.randn(3,3,requires_grad=True)
    clean = torch.tensor([[10.,0.,0.],[0.,0.,0.],[0.,0.,10.]])
    loss, accepted = experiment.clean_pseudo_loss(lambda x:(reader(x),),teacher,clean,student_input)
    assert accepted.item()==2
    wanted = torch.nn.functional.cross_entropy(reader(student_input)[[0,2]],torch.tensor([0,2]),reduction='sum')/3.
    torch.testing.assert_close(loss,wanted)
    loss.backward()
    assert teacher.weight.grad is None
    assert student_input.grad[[0,2]].abs().sum()>0
    assert not student_input.grad[1].any()

    loss, accepted = experiment.clean_pseudo_loss(
        lambda _x: pytest.fail('No student U forward for an all-rejected batch'),
        teacher,torch.zeros_like(clean),student_input)
    assert loss.item()==accepted.item()==0


def test_collector_returns_when_source_does_not_qualify(monkeypatch):
    args=SimpleNamespace(submit='submit.json',ssh_config='ssh')
    monkeypatch.setattr(experiment,'read',lambda _path:{'run_root':f'/runs/{experiment.RUN}'})
    monkeypatch.setattr(experiment.subprocess,'run',lambda *a,**kw:SimpleNamespace(
        returncode=0,stdout=b'{"status":"SOURCE_SCORED","promote":false}'))
    experiment.collect(args)


def test_fixed_e104_clean_pl_requires_matched_gain_before_formal_full():
    control = experiment.CANDIDATE
    candidate = experiment.FIXED_PL
    scores = {m:dict(six_correct=1400,low_urban_correct=170,by_scene={'clean':297}) for m in experiment.MODES}
    scores[control].update(six_correct=1451,low_urban_correct=179)
    scores['unet_parent']['low_urban_correct'] = 181
    scores[candidate] = dict(six_correct=1460,low_urban_correct=180,by_scene={'clean':297})
    assert not experiment.qualifies(scores,candidate)
    scores[candidate]['low_urban_correct'] = 181
    assert experiment.qualifies(scores,candidate)
    scores[candidate].update(six_correct=1451)
    assert not experiment.qualifies(scores,candidate)
    scores[candidate].update(six_correct=1460,by_scene={'clean':296})
    assert not experiment.qualifies(scores,candidate)

    with pytest.raises(ValueError,match='frozen-E104'):
        experiment.qualifies(scores,experiment.PL)
    for case in experiment.FULL_CASES:
        receipt = experiment.formal_receipt(case,experiment.RUN_FIXED_PL)
        assert receipt['modes'][-1] == candidate and experiment.PL not in receipt['modes']
        assert receipt['candidate_run'] == experiment.RUN_FIXED_PL


def test_clean_pseudo_ce_can_update_restorer_with_fixed_reader():
    reader = Reader().eval().requires_grad_(False)
    with torch.no_grad():
        reader.head.bias[0] = 10
    clean = torch.randn(3,8,2,432)
    restored = torch.randn_like(clean,requires_grad=True)
    loss, accepted = experiment.clean_pseudo_loss(reader,reader,clean,restored)
    assert accepted.item() == 3
    loss.backward()
    assert restored.grad.abs().sum() > 0
    assert all(p.grad is None for p in reader.parameters())


def test_paired_feature_loss_keeps_uncertain_u_without_pseudo_labels():
    def readout(x):
        return torch.zeros(len(x), 10), x

    clean = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    restored = torch.tensor([[1., 1.], [1., 0.]], requires_grad=True)
    ce, accepted, feature = experiment.clean_pseudo_loss(
        readout, readout, clean, restored, preserve_feature=True)
    assert ce.item() == accepted.item() == 0
    torch.testing.assert_close(feature, torch.tensor(1. - .5**1.5))
    feature.backward()
    assert clean.grad is None
    assert (restored.grad.abs().sum(1) > 0).all()


def test_feature_candidate_requires_paired_gain_and_keeps_formal_e104_parent():
    candidate = experiment.FEATURE
    control = experiment.CONTINUED_PL
    parent = experiment.FIXED_PL
    modes = (*experiment.MODES,parent,control,candidate)
    scores = {m:dict(six_correct=1455,low_urban_correct=187,by_scene={'clean':297}) for m in modes}
    scores[candidate]['six_correct'] = 1456
    assert not experiment.qualifies(scores,candidate)
    scores[candidate]['low_urban_correct'] = 188
    assert experiment.qualifies(scores,candidate)
    scores[candidate]['by_scene']['clean'] = 296
    assert not experiment.qualifies(scores,candidate)
    for case in experiment.FULL_CASES:
        proof = experiment.formal_receipt(case,experiment.RUN_FEATURE)
        assert proof['reference_run'] == experiment.RUN_FIXED_PL
        assert proof['modes'][-2:] == [parent,candidate]
        assert control not in proof['modes'] and not proof['target_truth_read']


def test_second_stage_starts_at_parent_and_leaves_its_weights_frozen():
    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(.8))

        def propose(self, y, quality, rms, edges):
            return (y, y * self.scale),

    parent = Parent()
    plain = experiment.initialize_recovery(experiment.RAW_REFINE,Parent(),'cpu')
    stage = experiment.initialize_recovery(experiment.REFINE,parent,'cpu')
    assert all(torch.equal(v,plain.state_dict()[k]) for k,v in stage.state_dict().items())
    y = torch.randn(1,8,384,dtype=torch.complex64)
    quality = torch.zeros(1,8,1)
    rms = torch.ones(1,8,1)
    edges = torch.zeros(1,8,48,dtype=torch.complex64)
    (_, final), = stage.propose(y,quality,rms,edges)
    torch.testing.assert_close(final, y * parent.scale, rtol=0, atol=0)
    final.real.square().mean().backward()
    assert parent.scale.grad is None and not parent.scale.requires_grad
    assert stage.refiner.time_edit.bias.grad.abs().sum() > 0 or stage.refiner.time_edit.weight.grad.abs().sum() > 0


def test_second_stage_only_conditioned_candidate_can_promote():
    parent,control,candidate = experiment.FIXED_PL,*experiment.REFINE_ARMS
    scores = {m:dict(six_correct=1455,low_urban_correct=187,by_scene={'clean':297})
              for m in (*experiment.MODES,parent,control,candidate)}
    scores[candidate]['six_correct'] = 1456
    assert not experiment.qualifies(scores,candidate)
    scores[candidate]['low_urban_correct'] = 188
    assert experiment.qualifies(scores,candidate)
    for case in experiment.FULL_CASES:
        proof = experiment.formal_receipt(case,experiment.RUN_REFINE)
        assert proof['modes'][-2:] == [parent,candidate] and control not in proof['modes']


def test_strong_source_adds_exact_awgn_budget_and_preserves_other_pairs():
    mix = experiment.StrongSourceMix
    rng_seed = 74
    raw = torch.ones(32,8,128,dtype=torch.complex64) * torch.arange(1,33)[:,None,None]

    def generic(clean,rng):
        return clean * (.6 + .3 * torch.rand((len(clean),1,1),generator=rng))

    control_rng = torch.Generator().manual_seed(8)
    candidate_rng = torch.Generator().manual_seed(8)
    control = mix(generic,torch.Generator().manual_seed(rng_seed),'cpu',enabled=False)
    candidate = mix(generic,torch.Generator().manual_seed(rng_seed),'cpu',enabled=True)
    normal = control(raw,control_rng)
    paired = candidate(raw,candidate_rng)
    trial = torch.Generator().manual_seed(rng_seed)
    strong = torch.rand((len(raw),1,1),generator=trial).flatten() < experiment.STRONG_FRACTION
    low, high = experiment.STRONG_SNR_DB
    nominal = low + (high-low)*torch.rand((len(raw),1,1),generator=trial).flatten()
    assert 0 < strong.sum() == candidate.selected < len(raw) and not control.selected
    torch.testing.assert_close(paired[~strong],normal[~strong],rtol=0,atol=0)
    assert torch.equal(control_rng.get_state(),candidate_rng.get_state())
    measured = 10 * (raw.abs().square().mean((1,2)) / (paired-raw).abs().square().mean((1,2))).log10()
    torch.testing.assert_close(measured[strong],nominal[strong])


def test_strong_source_only_candidate_can_enter_formal_full():
    parent,control,candidate = experiment.FIXED_PL,*experiment.STRONG_ARMS
    scores = {m:dict(six_correct=1455,low_urban_correct=187,by_scene={'clean':297})
              for m in (*experiment.MODES,parent,control,candidate)}
    scores[candidate].update(six_correct=1460,low_urban_correct=188)
    assert experiment.qualifies(scores,candidate)
    scores[candidate]['by_scene']['clean'] = 296
    assert not experiment.qualifies(scores,candidate)
    for case in experiment.FULL_CASES:
        proof = experiment.formal_receipt(case,experiment.RUN_STRONG)
        assert proof['modes'][-2:] == [parent,candidate] and control not in proof['modes']


def test_capture_rate_coloring_changes_shape_at_matched_w24_center_power():
    producer = torch.Generator().manual_seed(35)
    white = torch.randn(4,8,1024,generator=producer,dtype=torch.complex64)
    transform = experiment.PassbandMatchedNoise(Tracker(),torch.Generator().manual_seed(17))
    colored = transform(white)
    p0 = transform.center_power(white.flatten(1))
    p1 = transform.center_power(colored.flatten(1))
    torch.testing.assert_close(p1,p0)
    assert not torch.equal(colored,white)
    freq = torch.fft.fftfreq(white.shape[1]*white.shape[2],d=1e-6).abs()

    def relative_outband(data):
        psd = torch.fft.fft(data.flatten(1)).abs().square()
        return psd[:,freq>300_000].sum(1)/psd.sum(1)

    assert (relative_outband(colored)<relative_outband(white)).all()


def test_color_candidate_uses_the_immediate_strong_source_parent():
    control,candidate = experiment.COLOR_ARMS
    parent = experiment.STRONG
    scores = {m:dict(six_correct=1455,low_urban_correct=187,by_scene={'clean':297})
              for m in (*experiment.parent_modes(experiment.RUN_STRONG),control,candidate)}
    scores[parent].update(six_correct=1487,low_urban_correct=188)
    scores[candidate].update(six_correct=1487,low_urban_correct=188)
    assert not experiment.qualifies(scores,candidate)
    scores[candidate]['six_correct'] += 1
    assert experiment.qualifies(scores,candidate)
    for case in experiment.FULL_CASES:
        proof = experiment.formal_receipt(case,experiment.RUN_COLOR)
        assert proof['reference_run']==experiment.RUN_STRONG
        assert proof['modes'][-3:]==[experiment.FIXED_PL,parent,candidate] and control not in proof['modes']


def test_task_gain_starts_at_half_and_backpropagates_only_to_calibration():
    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(.8))
            self.register_buffer('noise_band',torch.arange(384)>256,persistent=False)

        def propose(self,y,quality,rms,edges):
            return (y,y*self.scale),

    r = torch.Generator().manual_seed(8)
    y = torch.randn(2,8,384,generator=r,dtype=torch.complex64)
    q = torch.linspace(.2,.9,8)[None,:,None].expand(2,-1,-1)
    rms = torch.ones_like(q)
    edges = y[...,:48]
    for arm in experiment.GAIN_ARMS:
        model = experiment.initialize_recovery(arm,Parent(),'cpu')
        (_, output), = model.propose(y,q,rms,edges)
        torch.testing.assert_close(output,y*.8,rtol=0,atol=0)
        output.real.square().mean().backward()
        assert model.offset.grad.abs()>0 and model.parent.scale.grad is None
        if model.local is not None:
            assert model.local[-1].weight.grad.abs().sum()>0
            with torch.no_grad():
                model.local[-1].weight.fill_(.3)
            gains = model.gain(y,q,rms,edges)
            assert gains.shape==(2,8,1) and gains.std()>0
            assert ((gains>=.05)&(gains<=.95)).all()


def test_adaptive_gain_promotes_only_after_matching_parent_and_control():
    control,candidate=experiment.GAIN_ARMS
    parent=experiment.STRONG
    scores={m:dict(six_correct=1490,low_urban_correct=187,by_scene={'clean':297})
            for m in (*experiment.parent_modes(experiment.RUN_STRONG),control,candidate)}
    scores[candidate].update(six_correct=1492,low_urban_correct=188)
    assert experiment.qualifies(scores,candidate)
    scores[candidate]['by_scene']['clean']=296
    assert not experiment.qualifies(scores,candidate)
    for case in experiment.FULL_CASES:
        proof=experiment.formal_receipt(case,experiment.RUN_GAIN)
        assert proof['reference_run']==experiment.RUN_STRONG and proof['modes'][-2:]==[parent,candidate]



def test_local_covariance_only_adds_context_from_other_chirps_in_registered_bins():
    r = torch.Generator().manual_seed(8)
    y = torch.randn(2,8,384,generator=r,dtype=torch.complex64)
    diagonal, cross = (experiment.initialize_recovery(arm,nn.Identity(),'cpu')
                       for arm in experiment.COV_ARMS)
    independent = diagonal.observation_state(y,None)
    joint = cross.observation_state(y,None)
    revised = y.clone()
    revised[0,1] += y[0,0] * 2
    torch.testing.assert_close(diagonal.observation_state(revised,None)[0,0],independent[0,0])
    changed = cross.observation_state(revised,None)
    assert (changed[0,0]-joint[0,0]).abs().max() > 1e-4
    torch.testing.assert_close(changed[1],joint[1],rtol=0,atol=0)
    edit = torch.fft.fft(joint-independent,norm='ortho')
    untouched = torch.ones(384,dtype=torch.bool)
    untouched[cross.detail_bins] = False
    assert edit[...,untouched].abs().max() < 1e-6
    assert torch.count_nonzero(cross.observation_state(torch.zeros_like(y),None)) == 0


def test_cross_covariance_source_gate_and_formal_reference_use_strong_parent():
    control, candidate = experiment.COV_ARMS
    parent = experiment.STRONG
    scores = {m:dict(six_correct=1487,low_urban_correct=187,by_scene={'clean':297})
              for m in (*experiment.parent_modes(experiment.RUN_STRONG),control,candidate)}
    scores[candidate].update(six_correct=1489,low_urban_correct=188)
    assert experiment.qualifies(scores,candidate)
    scores[candidate]['low_urban_correct'] = 187
    assert not experiment.qualifies(scores,candidate)
    for case in experiment.FULL_CASES:
        proof = experiment.formal_receipt(case,experiment.RUN_COV)
        assert proof['reference_run']==experiment.RUN_STRONG and proof['modes'][-2:]==[parent,candidate]



def test_hard_view_risk_gives_more_gradient_to_the_harder_realization():
    loss = torch.tensor([[3., 2., 1.], [1., 2., 3.]],requires_grad=True)
    mean = experiment.reduce_view_risk(loss,0.)
    risk = experiment.reduce_view_risk(loss,experiment.VIEW_TILT)
    torch.testing.assert_close(mean,torch.full((3,),2.))
    torch.testing.assert_close(risk,experiment.reduce_view_risk(loss.flip(0),experiment.VIEW_TILT))
    assert risk[0] > mean[0] and risk[2] > mean[2] and risk[1] == mean[1]
    grad, = torch.autograd.grad(risk.sum(),loss)
    assert grad[0,0] > grad[1,0] > 0 and grad[1,2] > grad[0,2] > 0
    torch.testing.assert_close(grad.sum(0),torch.ones(3))


def test_same_clean_u_pseudo_label_applies_to_both_disturbed_views_without_true_labels():
    clean = torch.tensor([[10.,0.,0.],[0.,0.,0.],[0.,0.,10.]],requires_grad=True)
    disturbed = torch.randn(2,3,3,generator=torch.Generator().manual_seed(3),requires_grad=True)
    reader = lambda z: (z,)
    before, accepted = experiment.clean_pseudo_loss(reader,reader,clean,disturbed.flatten(0,1),view_tilt=0.)
    ce = torch.nn.functional.cross_entropy(disturbed[:, (0,2)].flatten(0,1),torch.tensor([0,2,0,2]),reduction='sum')
    torch.testing.assert_close(before,ce/6)
    risk, accepted_hard = experiment.clean_pseudo_loss(
        reader,reader,clean,disturbed.flatten(0,1),view_tilt=experiment.VIEW_TILT)
    assert accepted == accepted_hard == 2 and risk > before
    risk.backward()
    assert clean.grad is None
    assert disturbed.grad[:, (0,2)].abs().sum() > 0
    assert not disturbed.grad[:, 1].any()


def test_hard_view_loss_uses_identical_draws_and_keeps_other_losses_equal(monkeypatch):
    r = torch.Generator().manual_seed(8)
    raw = torch.randn(6,8192,generator=r,dtype=torch.complex64)
    label = torch.tensor([1,2,3])
    f = Reader()
    g = nn.Parameter(torch.tensor(.3))
    monkeypatch.setattr(experiment,'wide_delta',lambda g,tracker,iq: g * iq[..., :384])

    def parts(tilt):
        draws=[]
        def generic(x,rng):
            noisy = x + torch.randn(x.shape,dtype=x.dtype,generator=rng)
            draws.append(noisy)
            return noisy
        result = experiment.loss_parts(g,f,f,Tracker(),generic,raw,label,torch.Generator().manual_seed(10),
                                       pseudo=True,view_tilt=tilt)
        assert len(draws) == 2 and not torch.equal(*draws)
        return result,draws

    mean, common = parts(0.)
    hard, matched = parts(experiment.VIEW_TILT)
    for a,b in zip(common,matched,strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert hard[0] > mean[0]
    torch.testing.assert_close(hard[1:6],mean[1:6],rtol=0,atol=0)
    torch.testing.assert_close(hard[7:],mean[7:],rtol=0,atol=0)
    assert experiment.formal_receipt(experiment.FULL_CASES[0],experiment.RUN_VIEW)['reference_run']==experiment.RUN_STRONG
