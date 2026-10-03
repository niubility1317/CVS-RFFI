"""Synthetic metadata and public random tensors only; never registered source IQ."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.cvs_frontfilter_attribution import run
from experiments.cvs_frontfilter_identity.model import build, filter_contract
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from baselines.common.practical_source import physical_id

torch.set_num_threads(2)


def config(variant='frontfilter_static', seed=2026092701):
    row = variant+'-s'+str(seed)
    return dict(source_release=run.PROJECT+'/releases/'+run.SOURCE_RELEASE,
                source_commit=run.SOURCE_COMMIT,
                source_output=run.PROJECT+'/runs/'+run.SOURCE_RUN+'/'+row+'/source',
                variant=variant, model_seed=seed,
                output_root=run.PROJECT+'/runs/'+run.RUN+'/'+row,
                conditions=list(run.CONDITIONS), role='V', coefficient_role='L_s',
                launch_owner=run.OWNER)


def activated_model(variant):
    torch.manual_seed(604)
    model = build(variant).eval()
    with torch.no_grad():
        if variant == 'frontfilter_static':
            model.frontfilter.coeff_raw.normal_(std=.1)
        else:
            model.frontfilter.context[-1].weight.normal_(std=.13)
            model.frontfilter.context[-1].bias.normal_(std=.1)
    return model


@pytest.mark.parametrize('variant', run.VARIANTS)
def test_all_on_matches_model_and_identity_bypasses_only_filter(variant):
    model = activated_model(variant)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(5, 2, 256)
    fixed = model.frontfilter.coefficients(torch.randn(7, 2, 256)).detach().double().mean(0).float()
    logits, scalar = run.components(model, x, fixed)
    with torch.no_grad():
        torch.testing.assert_close(logits['all_on'], model(x), rtol=1e-6, atol=1e-6)
        identity = model.classify_features(NeuralResidualCVS.features(model, x))
        torch.testing.assert_close(logits['g_identity'], identity, rtol=0, atol=0)
        original = model.frontfilter.coefficients
        model.frontfilter.coefficients = lambda z: fixed[None].expand(len(z), -1, -1)
        try:
            torch.testing.assert_close(logits['mean_L'], model(x), rtol=0, atol=0)
        finally:
            model.frontfilter.coefficients = original
    assert set(logits) == set(run.CONDITIONS)
    assert len(scalar) == 14
    assert all(v.shape == (5,) and torch.isfinite(v).all() for v in scalar.values())
    assert (scalar['g_input_relative'] > 0).all()
    if variant == 'frontfilter_static':
        torch.testing.assert_close(logits['mean_L'], logits['all_on'], rtol=run.STATIC_RTOL, atol=run.STATIC_ATOL)
        assert torch.equal(logits['mean_L'].argmax(1), logits['all_on'].argmax(1))
    else:
        assert (scalar['coefficient_L_mean_distance'] > 0).all()
        assert (scalar['mean_L_all_on_input_relative'] > 0).all()
    run.no_source_smoke(model, 'cpu')
    run.assert_state_unchanged(model, before)


@pytest.mark.parametrize('variant', run.VARIANTS)
def test_zero_exit_collapses_all_interventions(variant):
    model = build(variant).eval()
    logits, stats = run.components(model, torch.randn(3, 2, 256), torch.zeros(2, 4))
    for value in logits.values():
        torch.testing.assert_close(value, logits['all_on'], rtol=0, atol=0)
    assert all(torch.count_nonzero(v) == 0 for v in stats.values())


def test_exact_state_check_includes_buffers_and_smoke_detects_wrong_full_forward():
    model = build(run.VARIANTS[0]).eval()
    model.register_buffer('audit_buffer', torch.zeros(1))
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    model.audit_buffer += 1
    with pytest.raises(ValueError, match='buffers'):
        run.assert_state_unchanged(model, snapshot)
    model.forward = lambda x: torch.ones(len(x), 6)*100
    with pytest.raises(ValueError, match='smoke'):
        run.no_source_smoke(model, 'cpu')


@pytest.mark.parametrize('variant', run.VARIANTS)
@pytest.mark.parametrize('seed', range(2026092701, 2026092705))
def test_fixed_eight_model_configuration(variant, seed):
    c = config(variant, seed)
    assert run.validate_config(c) == c


@pytest.mark.parametrize('key,value', [
    ('role', 'target'), ('coefficient_role', 'V'), ('coefficient_role', 'U_s'),
    ('target_truth', 'bad'), ('checkpoint', 'another.pt'), ('source_commit', 'bad'),
    ('source_release', 'bad'), ('source_output', 'bad'), ('output_root', 'bad'),
    ('conditions', ['all_on', 'g_identity']), ('variant', 'channel_order'),
    ('model_seed', 2026092701.), ('model_seed', 1), ('launch_owner', 'another')])
def test_permissions_and_paths_rejected(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError):
        run.validate_config(c)


def payload_fixture():
    c = config()
    architecture = filter_contract(c['variant'])
    resolved = dict(commit=run.SOURCE_COMMIT, frontfilter=copy.deepcopy(architecture),
                    frontfilter_actual=copy.deepcopy(architecture), frontfilter_active=True,
                    precision='float32', numerical_policy=copy.deepcopy(FULL_FP32_POLICY),
                    backend_flags=copy.deepcopy(FULL_FP32_POLICY), total_parameters=architecture['total_parameters'])
    contract = dict(classes=['14-10', '14-7', '20-15', '20-19', '6-15', '8-20'])
    initial = dict(status='SCRATCH', scratch_only=True, checkpoint=None, ancestors=[],
                   checkpoint_sources=[], target_access=False, target_contact=False,
                   model_seed=c['model_seed'], physical_roles='EXACT_MATCH', selection='fixed_last_epoch')
    payload = dict(epoch=200, selection='fixed_last_epoch', method='cvs_frontfilter_identity',
                   variant=c['variant'], config=copy.deepcopy(resolved),
                   source_contract=copy.deepcopy(contract), initialization=copy.deepcopy(initial),
                   classes=contract['classes'], num_classes=6, model={'weight': torch.ones(2)})
    return c, resolved, contract, initial, payload


def test_valid_payload_and_strict_model_roundtrip():
    run.validate_payload(*payload_fixture())
    for variant in run.VARIANTS:
        model, restored = activated_model(variant), build(variant)
        restored.load_state_dict(model.state_dict(), strict=True)
        assert restored.contract() == filter_contract(variant)
        state = dict(model.state_dict()); state.pop(next(iter(state)))
        with pytest.raises(RuntimeError):
            restored.load_state_dict(state, strict=True)


@pytest.mark.parametrize('corruption', ['commit', 'inactive', 'fp16_policy', 'tf32', 'parameters',
    'architecture', 'ancestor', 'target', 'epoch', 'method', 'payload_contract', 'fp16_weights',
    'nonfinite_weights', 'complex_weights'])
def test_source_provenance_payload_and_precision_rejected(corruption):
    c, resolved, contract, initial, payload = payload_fixture()
    if corruption == 'commit': resolved['commit'] = 'unfrozen'
    elif corruption == 'inactive': resolved['frontfilter_active'] = False
    elif corruption == 'fp16_policy': resolved['precision'] = 'float16'
    elif corruption == 'tf32': resolved['numerical_policy']['cudnn_allow_tf32'] = True
    elif corruption == 'parameters': resolved['total_parameters'] += 1
    elif corruption == 'architecture': resolved['frontfilter_actual']['frontfilter_taps'] = 9
    elif corruption == 'ancestor': initial['ancestors'] = ['unknown.pt']
    elif corruption == 'target': initial['target_contact'] = True
    elif corruption == 'epoch': payload['epoch'] = 199
    elif corruption == 'method': payload['method'] = 'another'
    elif corruption == 'payload_contract': payload['source_contract']['classes'] = ['different']
    elif corruption == 'fp16_weights': payload['model']['weight'] = torch.ones(2).half()
    elif corruption == 'nonfinite_weights': payload['model']['weight'][0] = float('nan')
    elif corruption == 'complex_weights': payload['model']['weight'] = torch.ones(2, dtype=torch.complex64)
    # Mirror changes in the envelope to test the independent resolved/scratch checks.
    if corruption in ('commit', 'inactive', 'fp16_policy', 'tf32', 'parameters', 'architecture'):
        payload['config'] = copy.deepcopy(resolved)
    if corruption in ('ancestor', 'target'):
        payload['initialization'] = copy.deepcopy(initial)
    with pytest.raises(ValueError):
        run.validate_payload(c, resolved, contract, initial, payload)


class IndexOnlyDataset:
    role = 'L_s'
    def __init__(self):
        self.indices = list(range(3))
        self.base = SimpleNamespace(index=[SimpleNamespace(tx_i=i, rx_i=1, day_i=1, eq_i=1, sig_i=i)
                                          for i in range(3)])
    def __len__(self): return len(self.indices)
    def __getitem__(self, _): raise AssertionError('No packet reads during physical ID check')


def test_physical_ids_checked_without_packet_or_label_reads():
    dataset = IndexOnlyDataset()
    ids = [physical_id(item) for item in dataset.base.index]
    assert len(run.checked_role_index(dataset, ids, 'L_s', 3)) == 3
    for wrong in (ids[:2], ids[:2]+[ids[0]], ids[:2]+['wrong']):
        with pytest.raises(ValueError): run.checked_role_index(dataset, wrong, 'L_s', 3)
    dataset.indices = [0, 1, 1]
    with pytest.raises(ValueError): run.checked_role_index(dataset, ids, 'L_s', 3)


class LabelForbidden(dict):
    def __getitem__(self, key):
        if key == 'label': raise AssertionError('L label access is forbidden')
        return super().__getitem__(key)


def mean_fixture():
    index = {'a': (0, 1, 1), 'b': (1, 3, 2), 'c': (2, 4, 3)}
    batch = LabelForbidden(iq=torch.randn(3, 2, 256), receiver=torch.tensor([1, 3, 4]),
                          day=torch.tensor([1, 2, 3]),
                          meta=[dict(sample_id=sid, rx_i=rx, day_i=day) for sid, (_, rx, day) in index.items()])
    return index, batch


def test_L_only_mean_float64_and_saved_FP32_cast_without_labels():
    model = activated_model('frontfilter_dynamic')
    index, batch = mean_fixture()
    ids, values, mean64, mean32 = run.coefficient_mean(model, [batch], index, 'cpu')
    assert ids == list(index)
    assert values.dtype == np.float32 and mean64.dtype == np.float64 and mean32.dtype == np.float32
    np.testing.assert_array_equal(mean64, values.astype(np.float64).mean(0))
    np.testing.assert_array_equal(mean32, mean64.astype(np.float32))


@pytest.mark.parametrize('corruption', ['id', 'day', 'rx', 'missing', 'repeat', 'nonfinite', 'bound'])
def test_bad_coefficient_inputs_fail(corruption):
    model = activated_model('frontfilter_dynamic'); index, batch = mean_fixture(); loader = [batch]
    if corruption == 'id': batch['meta'][0]['sample_id'] = 'V_packet'
    elif corruption == 'day': batch['meta'][0]['day_i'] = 2
    elif corruption == 'rx': batch['receiver'][0] = 8
    elif corruption == 'missing': index['missing'] = (0, 1, 1)
    elif corruption == 'repeat': loader = [batch, batch]
    elif corruption == 'nonfinite': model.frontfilter.coefficients = lambda x: torch.full((len(x), 2, 4), float('nan'))
    elif corruption == 'bound': model.frontfilter.coefficients = lambda x: torch.ones(len(x), 2, 4)
    with pytest.raises(ValueError): run.coefficient_mean(model, loader, index, 'cpu')


def test_metadata_checks_truth_and_actual_day_i():
    index, batch = mean_fixture(); batch = dict(batch); batch['label'] = torch.tensor([0, 1, 2])
    assert run.batch_metadata(batch, index, with_truth=True)[2] == [1, 2, 3]
    batch['label'][0] = 5
    with pytest.raises(ValueError): run.batch_metadata(batch, index, with_truth=True)


def test_paired_counts_and_stable_CE():
    truth = np.array([0, 1, 2, 3]); rx = np.array([1, 1, 3, 3])
    logits = {}
    for name, predictions in [('all_on', [0, 0, 2, 1]), ('mean_L', [1, 1, 2, 1]), ('g_identity', [0, 1, 2, 3])]:
        value = np.full((4, 6), -1000., dtype=np.float32)
        value[np.arange(4), predictions] = 1000.
        logits[name] = value
    rows, pred = run.classification_rows(logits, truth, rx)
    assert [r['ce'] for r in rows] == [1000., 1000., 0.]
    assert [r['accuracy'] for r in rows] == [.5, .5, 1.]
    assert rows[1]['prediction_changes'] == 2
    assert rows[1]['helped_by_all_on'] == rows[1]['hurt_by_all_on'] == 1
    assert rows[2]['helped_by_all_on'] == 0 and rows[2]['hurt_by_all_on'] == 2
    assert rows[0]['rx_accuracy'] == {'1': .5, '3': .5}
    assert np.array_equal(pred['g_identity'], truth)
    logits['all_on'][0, 0] = np.nan
    with pytest.raises(ValueError): run.classification_rows(logits, truth, rx)


@pytest.mark.parametrize('corruption', [None, 'missing', 'repeated_id', 'unbalanced', 'day'])
def test_complete_balanced_V_cell_coverage(corruption):
    cells = [(tx, rx, day) for tx in range(6) for rx in (1, 3, 4, 6, 8) for day in (1, 2, 3)]
    ids = [str(i) for i in range(90)]; index = dict(zip(ids, cells))
    truth, receiver, days = map(list, zip(*cells))
    if corruption == 'missing': ids.pop(); truth.pop(); receiver.pop(); days.pop()
    elif corruption == 'repeated_id': ids[0] = ids[1]
    elif corruption == 'unbalanced': truth[0] = 1
    elif corruption == 'day': days[0] = 0
    if corruption is None:
        run.validate_v_coverage(ids, truth, receiver, days, index, cell_count=1)
    else:
        with pytest.raises(ValueError): run.validate_v_coverage(ids, truth, receiver, days, index, cell_count=1)
