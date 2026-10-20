import copy

import pytest
import torch

from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_r17_compact.export import export_inference
from experiments.pamf_uncertainty_restore import design as ancestor
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_unrolled_restore.runtime import load_checkpoint, save_checkpoint

torch.set_num_threads(2)


@pytest.fixture(scope='module')
def original_payload(tmp_path_factory):
    folder = tmp_path_factory.mktemp('r17-original')
    path = folder/'original.pt'
    torch.manual_seed(17)
    model = UnrolledRestorer('unroll_unshared', ActionRestorer(ComplexTFUNet()))
    for core in model.prior:
        torch.nn.init.normal_(core.out.weight, std=.001)
    parent = dict(coarse_run=ancestor.COARSE_RUN, coarse_commit=ancestor.COARSE_COMMIT,
                  action_run=ancestor.ACTION_RUN, action_commit=ancestor.ACTION_COMMIT,
                  data_contract='original7/63/30', training_TX_F_reads=0, target_training=False)
    save_checkpoint(path, model, None, 20, 4940, parent)
    payload = torch.load(path, weights_only=False)
    payload['optimizer'] = {'state': {'step': torch.ones(10)}}
    torch.save(payload, path)
    return path, payload


def test_export_keeps_every_tensor_and_prediction_without_training_state(tmp_path, original_payload):
    source, payload = original_payload
    result = export_inference(source, tmp_path/'inference.pt')
    assert result['model_parameters'] == 4665073
    assert result['model_tensors_exact'] and result['original_R17_graph']
    assert result['target_IQ_reads'] == 0 and not result['new_target_performance_evidence']
    original, _ = load_checkpoint(source, 'cpu')
    compact, loaded = load_checkpoint(tmp_path/'inference.pt', 'cpu')
    assert loaded['optimizer'] is None
    assert torch.load(source, weights_only=False)['optimizer']['state']['step'].equal(torch.ones(10))
    assert all(torch.equal(value, loaded['model'][key]) for key, value in payload['model'].items())
    with torch.no_grad():
        value = torch.randn(2, 2, 256)*.2
        assert torch.equal(original(value), compact(value))


def test_export_never_overwrites_source_or_existing_output(tmp_path, original_payload):
    source, _ = original_payload
    with pytest.raises(ValueError, match='untouched'):
        export_inference(source, source)
    destination = tmp_path/'existing.pt'
    destination.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        export_inference(source, destination)
    assert destination.read_bytes() == b'preserve'


@pytest.mark.parametrize('change', [
    {'training_F_access': True}, {'training_TX_labels_read': 1}, {'target_training': True},
    {'epoch': 19, 'updates': 4693}, {'candidate': 'unroll_shared'},
    {'parent': {'target_training': True}},
])
def test_invalid_or_nonfinal_parent_rejected_before_destination_creation(tmp_path, original_payload, change):
    _, original = original_payload
    value = copy.copy(original)
    value.update(change)
    source, destination = tmp_path/'bad.pt', tmp_path/'never-created.pt'
    torch.save(value, source)
    with pytest.raises((ValueError, RuntimeError)):
        export_inference(source, destination)
    assert not destination.exists()
