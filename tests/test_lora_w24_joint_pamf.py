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
