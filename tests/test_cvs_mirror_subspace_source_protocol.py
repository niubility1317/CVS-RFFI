"""Synthetic-only training boundaries and independently recountable source evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_mirror_subspace_identity import source
from experiments.cvs_mirror_subspace_identity.model import VARIANTS, build, relation_contract


def config(variant=VARIANTS[0]):
    project = '/home/szu2070436088/2510044040/CV-SincNet'
    return dict(method='cvs_mirror_subspace_identity', variant=variant, epochs=200,
        batch_size=128, lr=.0002, lr_min=1e-6, weight_decay=.0001, drop_last=False,
        augmentation=False, domain_backbone=False, extra_losses=[], selection='fixed_last_epoch',
        split_seed=392005, model_seed=2026092701, mirror_relation=relation_contract(variant),
        numerical_policy=FULL_FP32_POLICY.copy(), dataset=project+'/Dataset_WigSig/ManySig.pkl',
        source_contract=project+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')


@pytest.mark.parametrize('variant', VARIANTS)
def test_single_ce_scratch_contract(variant):
    cfg = config(variant)
    assert source.validate_config(cfg) == cfg
    assert cfg['mirror_relation']['total_parameters'] == 247731
    assert cfg['mirror_relation']['new_trainable_parameters'] == 26744


def test_actual_prepared_matrix_uses_the_same_source_contract():
    directory = Path(source.__file__).parent/'configs'
    launch = json.loads((directory/'launch_spec.json').read_text(encoding='utf-8'))
    assert len(launch['rows']) == 8
    observed = set()
    for row in launch['rows']:
        cfg = json.loads((directory/(row['row_id']+'.json')).read_text(encoding='utf-8'))
        assert source.validate_config(cfg) == cfg
        observed.add((cfg['variant'], cfg['model_seed']))
    assert observed == {(v, s) for v in VARIANTS for s in range(2026092701, 2026092705)}


@pytest.mark.parametrize('key,value', [
    ('checkpoint','old.pt'), ('resume','old.pt'), ('teacher','old.pt'),
    ('initial_checkpoint','old.pt'), ('checkpoint_sources',['old.pt']), ('ancestors',['old.pt']),
    ('ema_checkpoint','old.pt'), ('teacher_checkpoint','old.pt'),
    ('target_inputs','query.npz'), ('target_truth','truth.json'), ('p1_truth','truth.json'),
    ('p1_capsule','capsule'), ('target_access',True), ('target_metrics',{'accuracy':1.}),
    ('epochs',199), ('batch_size',256), ('lr',.001), ('lr_min',0.), ('weight_decay',0.),
    ('drop_last',True), ('augmentation',True), ('domain_backbone',True), ('extra_losses',['aux']),
    ('selection','best_epoch'), ('split_seed',1), ('model_seed',1), ('model_seed',2026092701.0),
    ('source_contract','other_roles.json'), ('dataset','other_data.pkl'),
])
def test_changed_training_or_checkpoint_inheritance_is_rejected(key, value):
    cfg = config(); cfg[key] = value
    with pytest.raises(ValueError):
        source.validate_config(cfg)


@pytest.mark.parametrize('kind', ['architecture','precision','variant'])
def test_invalid_architecture_or_numerics_is_rejected(kind):
    cfg = config()
    if kind == 'architecture': cfg['mirror_relation']['relation_power_floor'] = .1
    elif kind == 'precision': cfg['numerical_policy']['cudnn_allow_tf32'] = True
    else: cfg['variant'] = 'unregistered'
    with pytest.raises(ValueError):
        source.validate_config(cfg)


def test_existing_output_is_rejected_before_model_or_data_access(tmp_path, monkeypatch):
    cfg = config(); cfg['output_root'] = str(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('No model or dataset access allowed')
    monkeypatch.setattr(source, 'build', forbidden)
    monkeypatch.setattr(source, 'build_contract_split', forbidden)
    with pytest.raises(FileExistsError):
        source.train(cfg)


@pytest.mark.parametrize('variant', VARIANTS)
def test_actual_snapshot_is_same_forward_without_state_rng_or_hook_changes(variant):
    torch.set_num_threads(2); torch.manual_seed(71)
    model = build(variant).eval()
    with torch.no_grad():
        model.core.mirror_relation.project.weight.normal_(0., .01)
        iq = torch.randn(3, 2, 256)
        expected = model.features(iq)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    modes = [module.training for module in model.modules()]
    hooks = [len(module._forward_hooks) for module in model.modules()]
    features, scalars = source.mirror_relation_forward_snapshot(model, iq)
    assert torch.equal(features, expected)
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert [module.training for module in model.modules()] == modes
    assert [len(module._forward_hooks) for module in model.modules()] == hooks
    assert set(scalars) == set(source.RELATION_SCALARS)
    assert all(value.shape == (3,) and value.dtype == torch.float64 for value in scalars.values())
    assert torch.all(scalars['branch_output_norm'] > 0)
    assert torch.allclose(scalars['relative_output_change'],
        scalars['branch_output_norm']/scalars['base_frequency_norm'], rtol=1e-6, atol=1e-8)
    # Rank<=2 PSD relation: trace/sqrt(2) <= Frobenius <= trace.
    assert torch.all(scalars['relation_frobenius_mean'] <= scalars['relation_trace_mean']+2e-5)
    assert torch.all(scalars['relation_frobenius_mean'] >= scalars['relation_trace_mean']/2**.5-2e-5)
    assert torch.all(scalars['relation_trace_max'] <= 1+2e-5)
    upper = 1/2**.5 if variant == VARIANTS[1] else 1
    assert torch.all(scalars['relation_frobenius_max'] <= upper+2e-5)
    with torch.no_grad():
        c=model.core.mirror_relation.components(model.core.mirror_relation.mix_spectra(model.core.mirror_relation.spectral(iq)))
    assert torch.equal(scalars['determinant_floor_fraction'],c['determinant_floor_active'].float().mean(1).double())
    assert torch.equal(scalars['alpha_mean'],c['alpha'].mean(1).double())



@pytest.mark.parametrize('kind', ['exception','bypass','duplicate','nonfinite'])
def test_snapshot_cleans_hooks_on_invalid_execution(kind):
    model = build(VARIANTS[0]).eval(); iq = torch.randn(2, 2, 256)
    original = model.features
    if kind == 'exception':
        def failed(x): raise RuntimeError('synthetic forward failure')
        model.features = failed
    elif kind == 'bypass': model.features = lambda x: torch.zeros(len(x), 160)
    elif kind == 'duplicate':
        def repeated(x):
            original(x)
            return original(x)
        model.features = repeated
    else: iq[0, 0, 0] = float('nan')
    hooks = [len(module._forward_hooks) for module in model.modules()]
    with pytest.raises((RuntimeError, ValueError)):
        source.mirror_relation_forward_snapshot(model, iq)
    assert [len(module._forward_hooks) for module in model.modules()] == hooks


def test_full_v_floor_telemetry_includes_relative_floor_active_frequencies():
    torch.manual_seed(74); model = build(VARIANTS[1]).eval()
    phase = torch.arange(256)*2*torch.pi*7/64
    iq = torch.stack((phase.cos(),phase.sin()))[None] + .003*torch.randn(2,2,256)
    _, scalars = source.mirror_relation_forward_snapshot(model, iq)
    branch = model.core.mirror_relation
    with torch.no_grad():
        v = branch.mix_spectra(branch.spectral(iq))
        absolute_only = (branch.energy(v) < branch.power_floor).float().mean(1).double()
        actual_floor = branch.floor_active(v).float().mean(1).double()
    assert torch.equal(scalars['floor_fraction'], actual_floor)
    assert torch.all(actual_floor > absolute_only)


def small_arrays():
    arrays = dict(ids=np.asarray(['a','b','c','d']), tx=np.asarray([0,0,1,1]),
        receiver=np.asarray([1,1,1,1]), day=np.asarray([0,0,0,0]))
    arrays.update({key: np.asarray([0., .25, .5, .75]) for key in source.RELATION_SCALARS})
    return arrays


def recount(arrays):
    return source.summarize_relation_scalars(arrays, ['a','b','c','d'],
        expected_count=4, expected_cells=2, expected_per_cell=2)


def test_per_cell_recount_uses_all_packets_and_no_large_vectors():
    arrays = small_arrays(); records = recount(arrays)
    assert len(records) == 2 and sum(row['count'] for row in records) == 4
    assert records[0]['branch_output_norm_mean'] == .125
    assert records[1]['branch_output_norm_min'] == .5
    assert records[1]['branch_output_norm_max'] == .75
    assert all(value.ndim == 1 for value in arrays.values())


@pytest.mark.parametrize('kind', ['duplicate','foreign','vector','nan','negative','floor','cell','float_tx','missing'])
def test_invalid_scalar_evidence_cannot_be_reported(kind):
    arrays = small_arrays()
    if kind == 'duplicate': arrays['ids'][1] = 'a'
    elif kind == 'foreign': arrays['ids'][1] = 'z'
    elif kind == 'vector': arrays['branch_output_norm'] = np.zeros((4,160))
    elif kind == 'nan': arrays['branch_output_norm'][0] = np.nan
    elif kind == 'negative': arrays['branch_output_norm'][0] = -1
    elif kind == 'floor': arrays['floor_fraction'][0] = 1.01
    elif kind == 'cell': arrays['tx'][0] = 2
    elif kind == 'float_tx': arrays['tx'] = arrays['tx'].astype(float)
    else: arrays.pop('relation_trace_mean')
    with pytest.raises(ValueError): recount(arrays)


class GeometryModel:
    """Scalar-only synthetic source V; no dataset loading and no learned update."""
    def __init__(self):
        self.synchronizer = SimpleNamespace(estimate=lambda x: (
            torch.zeros(len(x)), torch.ones(len(x), dtype=torch.bool), torch.ones(len(x))))
    def eval(self): return self
    def coordinates(self, x):
        return x, torch.zeros(len(x)), torch.ones(len(x), dtype=torch.bool), torch.ones(len(x)), 0.
    def alignment_strength(self): return torch.tensor(0.)
    def classify_features(self, x): return x[:, :6]


def complete_synthetic_loader():
    for tx in range(6):
        for rx in (1,3,4,6,8):
            for day in range(3):
                yield dict(iq=torch.full((300,2,256), float(tx)), label=torch.full((300,), tx),
                    receiver=torch.full((300,), rx), day=torch.full((300,), day),
                    meta=[dict(sample_id=f'{tx}/{rx}/{day}/{j}') for j in range(300)])


def test_complete_v_evidence_roundtrip_is_27000_scalars_and_90_balanced_cells(tmp_path, monkeypatch):
    calls = []
    def snapshot(model, iq):
        calls.append(len(iq)); tx = int(iq[0,0,0])
        features = torch.zeros(len(iq),160); features[:,tx] = 1
        scalars = {key: torch.full((len(iq),), .25, dtype=torch.float64) for key in source.RELATION_SCALARS}
        return features, scalars
    monkeypatch.setattr(source, 'mirror_relation_forward_snapshot', snapshot)
    expected_ids = [meta['sample_id'] for batch in complete_synthetic_loader() for meta in batch['meta']]
    path = tmp_path/'source_mirror_relation_scalars.npz'
    result = source.final_source_diagnostics(GeometryModel(), complete_synthetic_loader(), 'cpu', path, expected_ids)
    assert result['count'] == 27000 and len(result['groups']) == 90
    assert len(calls) == 90 and sum(calls) == 27000
    assert result['mirror_relation_scalars']['schema'] == source.RELATION_SCALAR_SCHEMA
    assert result['used_for_training'] is result['used_for_selection'] is result['target_access'] is False
    assert all(row['accuracy'] == 1. for row in result['groups'])
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    assert all(value.shape == (27000,) for value in arrays.values())
    assert source.summarize_relation_scalars(arrays, expected_ids) == result['mirror_relation_groups']
    assert len(result['mirror_relation_groups']) == 90
    assert all(row['count'] == 300 and row['branch_output_norm_mean'] == .25
               for row in result['mirror_relation_groups'])
    with pytest.raises(FileExistsError):
        source.final_source_diagnostics(GeometryModel(), complete_synthetic_loader(), 'cpu', path, expected_ids)


def test_last_batch_epoch_measurements_survive_compact_scalar_logs():
    torch.manual_seed(81); model = build(VARIANTS[1])
    iq = torch.randn(3,2,256); labels = torch.tensor([0,1,2])
    torch.nn.functional.cross_entropy(model(iq), labels).backward()
    compact = source.compact_relation_diagnostics(model.diagnostics(iq))
    assert compact['mirror_relation_relative_output_change_mean'] == 0.
    assert compact['mirror_relation_projection_norm'] == 0.
    assert compact['mirror_relation_projection_gradient_norm'] > 0.
    assert compact['mirror_relation_mix_gradient_norm'] == 0.
    assert compact['mirror_relation_encoder_gradient_norm'] == 0.
    assert all(isinstance(value, float) and np.isfinite(value) for value in compact.values())
    json.dumps(compact, allow_nan=False)


@pytest.mark.parametrize('seed',range(2026092701,2026092705))
@pytest.mark.parametrize('variant',VARIANTS)
def test_public_rank_deficient_snapshot_preserves_nonnegative_trace(seed,variant):
    from experiments.cvs_spectral_relation_identity.physics import public_inputs
    torch.manual_seed(seed);model=build(variant).eval()
    x=public_inputs(torch.device('cpu'),torch.float32)
    _,scalars=source.mirror_relation_forward_snapshot(model,x)
    branch=model.core.mirror_relation
    with torch.no_grad():
        v=branch.mix_spectra(branch.spectral(x));q=branch.relations(v)
    trace=q[:,0].diagonal(dim1=-2,dim2=-1).sum(-1)
    assert (trace>=0).all() and torch.isfinite(trace).all()
    assert all(torch.isfinite(a).all() and (a>=0).all() for a in scalars.values())
    assert scalars['determinant_floor_fraction'][24:].eq(1).all()
