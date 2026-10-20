import copy,json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_r46_gradient_geometry import design as d,gradients as g,runner

torch.set_num_threads(2)


class Tiny(torch.nn.Module):
    def __init__(self,scale=.9):
        super().__init__();self.frozen=torch.nn.Module()
        self.frozen.coarse=torch.nn.Linear(2,2,bias=False)
        self.frozen.estimator=torch.nn.Linear(2,2,bias=False)
        self.prior=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False)])
        with torch.no_grad():
            for p in self.parameters():p.copy_(torch.eye(2)*scale)
        self.eval().requires_grad_(False)
    def stage_outputs(self,x):
        a=self.frozen.coarse(x.transpose(1,2)).transpose(1,2)
        b=self.frozen.estimator(a.transpose(1,2)).transpose(1,2)
        c=self.prior[0](b.transpose(1,2)).transpose(1,2)
        return [a,b,c],None
    def forward(self,x):return self.stage_outputs(x)[0][-1]


def test_exact_parallel_opposed_and_zero_gradient_directions():
    p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
    losses=dict(base=p.sum(),moment_clean=-2*p.sum(),moment_teacher=0*p.sum())
    got=g.geometry(losses,[p],dict(whole=[0],coarse=[0]))
    for group in got.values():
        assert group['norms']['base']==pytest.approx(2**.5)
        pair=group['pairs']['base_vs_moment_clean']
        assert pair['valid'] and pair['cosine']==pytest.approx(-1.) and pair['norm_ratio']==pytest.approx(2.)
        zero=group['pairs']['base_vs_moment_teacher']
        assert not zero['valid'] and zero['cosine'] is None and zero['norm_ratio']==0.
    assert p.grad is None


def test_complete_unique_parameter_grouping_and_unknown_parameters_rejected():
    model=Tiny();params,groups=g.parameter_groups(model)
    assert len(params)==3 and set(groups)=={'whole','coarse','action','prior'}
    assert sorted(groups['coarse']+groups['action']+groups['prior'])==groups['whole']
    model.unknown=torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):g.parameter_groups(model)


def test_batch_measurement_preserves_weights_flags_and_references():
    torch.manual_seed(13);model=Tiny();snapshot=copy.deepcopy(model.state_dict())
    clean=(torch.randn(4,2,256)*.1).requires_grad_();received=clean.detach()+.005*torch.randn_like(clean)
    teacher=[(clean.detach()*.95).requires_grad_() for _ in range(3)]
    record=g.measure_batch(model,received,clean,teacher)
    assert set(record['losses'])==set(d.OBJECTIVES) and set(record['geometry'])==set(d.GROUPS)
    assert all(v>=0 for v in record['losses'].values())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(v,snapshot[k]) for k,v in model.state_dict().items())
    assert clean.grad is None and all(x.grad is None for x in teacher)


def test_exception_restores_original_requires_grad_flags(monkeypatch):
    model=Tiny();params=list(model.parameters());params[0].requires_grad_(True);flags=[p.requires_grad for p in params]
    def fail(*a,**k):raise RuntimeError('synthetic failure')
    monkeypatch.setattr(g,'objective',fail)
    with pytest.raises(RuntimeError):g.measure_batch(model,torch.zeros(2,2,256),torch.zeros(2,2,256),[torch.zeros(2,2,256)]*3)
    assert [p.requires_grad for p in params]==flags and all(p.grad is None for p in params)


def test_fixed_batches_cover_all_positions_with_eight_per_source_role():
    batches=runner.batch_positions();flat=[i for b in batches for i in b]
    assert len(batches)==16 and sorted(flat)==list(range(256))
    assert all(len(b)==16 and sum(i<128 for i in b)==8 for b in batches)


def test_real_restoration_graph_and_shared_prior_unique_norms_are_finite():
    from tests.test_pamf_r46_moment_fidelity import parent_model
    from experiments.pamf_r46_complex_fidelity.models import compact_from_parent
    torch.manual_seed(113);teacher=parent_model();model=compact_from_parent(teacher,'complex625_identity').eval().requires_grad_(False)
    x=torch.randn(2,2,256)*.1;received=x+torch.randn_like(x)*.01
    with torch.no_grad():desired=teacher.stage_outputs(received)[0]
    record=g.measure_batch(model,received,x,desired)
    for objective in d.OBJECTIVES:
        norms={group:v['norms'][objective] for group,v in record['geometry'].items()}
        assert all(np.isfinite(v) and v>=0 for v in norms.values())
        assert norms['whole']**2==pytest.approx(sum(norms[k]**2 for k in ('coarse','action','prior')),rel=1e-10)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())


def test_undefined_cosines_excluded_from_conflict_fraction():
    p=torch.nn.Parameter(torch.ones(2));raw=g.geometry(dict(base=p.sum(),moment_clean=-p.sum(),moment_teacher=p.sum()*0),[p],{n:[0] for n in d.GROUPS})
    rows=g.summarize_batches([dict(losses=dict(base=1.,moment_clean=2.,moment_teacher=0.),geometry=raw)]*16)
    assert len(rows)==27
    row=next(r for r in rows if r['kind']=='cosine' and r['group']=='whole' and r['metric']=='base_vs_moment_teacher')
    assert row['valid_count']==0 and row['invalid_count']==16 and row['negative_fraction'] is None and row['mean'] is None
    opposite=next(r for r in rows if r['kind']=='cosine' and r['group']=='whole' and r['metric']=='base_vs_moment_clean')
    assert opposite['valid_count']==16 and opposite['negative_fraction']==1.


@pytest.mark.parametrize('flag',['--reader-folder','--target-truth','--formal-reference'])
def test_F_or_target_arguments_rejected(flag):
    args=[v for key in ('compact-parent','factor-run','complex-run','parent-action','source-cache','output') for v in ('--'+key,'placeholder')]+['--expected-commit','method']
    with pytest.raises(SystemExit):runner.parser().parse_args(args+[flag,'forbidden'])


def test_existing_output_rejected_before_IO(tmp_path,monkeypatch):
    out=tmp_path/d.RUN;out.mkdir();monkeypatch.setattr(runner,'validate_sources',lambda _:pytest.fail('No IO'))
    with pytest.raises(ValueError):runner.diagnose(SimpleNamespace(output=out))


def test_full_CPU_source_fixture_has_6930_inputs_and_324_summary_rows(tmp_path,monkeypatch):
    torch.manual_seed(91);x=torch.randn(256,2,256)*.1;panel={v:x.clone() for v in d.VIEWS}
    models={a:Tiny(.9-.01*i) for i,a in enumerate(d.ARMS)}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(runner,'code_commit',lambda:'method')
    monkeypatch.setattr(runner,'validate_sources',lambda _:None)
    monkeypatch.setattr(runner,'load_model',lambda _,a:(models[a],dict(epoch=20,updates=d.SOURCE_BUDGET[a][1])))
    monkeypatch.setattr(runner,'load_panel',lambda _:(panel,np.arange(256),np.array(['L_s']*128+['U_s']*128)))
    args=SimpleNamespace(output=tmp_path/d.RUN,expected_commit='method',_owns_output=False)
    runner.diagnose(args);result=json.loads((args.output/'results.json').read_text(encoding='utf-8'))
    assert result['status']=='SOURCE_GRADIENT_GEOMETRY_COMPLETE' and result['optimizer_updates']==result['new_network_versions']==0
    assert result['all_checkpoint_states_unchanged'] and result['teacher_cache_IQ_inputs']==770 and result['total_actual_G_IQ_inputs']==6930
    assert result['batch_records']==192 and len(result['summary_rows'])==324
    assert all(row['actual_student_IQ_inputs']==dict(smoke_received=2,smoke_clean=2,received=768,clean_identity=768) for row in result['rows'].values())
    assert all(result[k]==0 for k in ('F_weight_reads','F_output_reads','TX_label_reads','V_reads','target_reads'))
    assert all(not p.requires_grad and p.grad is None for m in models.values() for p in m.parameters())
