"""Protect Euclidean decisions and source-only selection from numerical collapse."""
import pytest
import torch
import json
import numpy as np

from paper_reproduction.gaskin_tweak_2023.calibration import CalibrationState, closed_set_predict
from paper_reproduction.gaskin_tweak_2023.training import shared_triplet_loss
from paper_reproduction.gaskin_tweak_2023.official_lora_experiment import select_learning_rate_from_probes


def test_nearby_centroids_keep_the_same_prediction_for_small_and_large_query_batches():
    # MM distance cancellation must not turn distinct centroids into a zero-distance tie.
    centers = torch.tensor([[0.05, 0.05], [0.050001, 0.05]], dtype=torch.float32)
    state = CalibrationState(torch.tensor([1, 2]), centers, torch.zeros(2))
    point = centers[1:2].clone()
    assert closed_set_predict(point, state).tolist() == [2]
    assert closed_set_predict(point.repeat(32, 1), state).tolist() == [2] * 32


def test_shared_triplet_uses_squared_l2_equation_one():
    # Raw norms give 1.1, whereas the published squared-distance formula gives 3.1.
    loss = shared_triplet_loss(torch.nn.Identity(), torch.tensor([[0.0]]),
                               torch.tensor([[2.0]]), torch.tensor([[1.0]]))
    assert float(loss) == pytest.approx(3.1)


def test_source_accuracy_not_collapsed_training_loss_selects_learning_rate():
    # A margin-floor loss must not defeat a model that retains source identity.
    rows = [
        dict(learning_rate=0.01, first_window_mean_loss=.3, last_window_mean_loss=.1,
             mean_training_loss=.101, active_batches=10, source_monitor_accuracy=.1),
        dict(learning_rate=.001, first_window_mean_loss=.3, last_window_mean_loss=.15,
             mean_training_loss=.18, active_batches=10, source_monitor_accuracy=.8),
    ]
    assert select_learning_rate_from_probes(rows) == pytest.approx(.001)


class MeanIQ(torch.nn.Module):
    def forward(self, iq):
        return iq.mean(dim=2)


def test_source_monitor_never_reads_test_frames(tmp_path):
    # Test-region poison must not affect source-only model selection.
    from paper_reproduction.gaskin_tweak_2023.official_lora import load_configuration_records, split_record_frames
    from paper_reproduction.gaskin_tweak_2023.official_lora_experiment import _source_monitor
    split = split_record_frames(total_samples=1024 * 128)
    folder = tmp_path / 'Config2'
    folder.mkdir()
    for label in (1, 2):
        frames = np.full((1024, 128), complex(label, label), dtype=np.complex64)
        frames[split.physical_indices(split.testing)] = complex(float('nan'), float('nan'))
        path = folder / f'IQ_{label}.dat'
        frames.tofile(path)
        path.with_suffix('.sigmf-meta').write_text(json.dumps({'_metadata': {'global': {'core:datatype': 'cf32'}}}), encoding='utf-8')
    records = load_configuration_records(tmp_path, 'Config2', device_ids=[1, 2])
    result = _source_monitor(MeanIQ(), records, split, device=torch.device('cpu'))
    assert result['source_monitor_accuracy'] == 1.0
    assert result['source_monitor_centroid_distance'] == pytest.approx(2 ** .5)


def test_prediction_artifact_can_be_rescored_and_cannot_be_overwritten(tmp_path):
    from paper_reproduction.gaskin_tweak_2023.official_lora import load_configuration_records, split_record_frames
    from paper_reproduction.gaskin_tweak_2023.official_lora_experiment import _evaluate_configuration
    folder = tmp_path / 'Config2'
    folder.mkdir()
    path = folder / 'IQ_2.dat'
    np.full(1024 * 128, complex(2, 2), dtype=np.complex64).tofile(path)
    path.with_suffix('.sigmf-meta').write_text(json.dumps({'_metadata': {'global': {'core:datatype': 'cf32'}}}), encoding='utf-8')
    records = load_configuration_records(tmp_path, 'Config2', device_ids=[2])
    state = CalibrationState(torch.tensor([1, 2]), torch.tensor([[1., 1.], [2., 2.]]), torch.zeros(2))
    output = tmp_path / 'predictions.pt'
    kwargs = dict(device=torch.device('cpu'), batch_size=64, group_size=10, prediction_path=output)
    result = _evaluate_configuration(MeanIQ(), records, split_record_frames(total_samples=1024 * 128), state, **kwargs)
    artifact = torch.load(output, weights_only=True)
    truth = torch.load(output.with_suffix('.truth.pt'), weights_only=True)
    assert 'labels' not in artifact
    assert torch.equal(artifact['query_ids'], truth['query_ids'])
    assert int(artifact['predictions'].eq(truth['labels']).sum()) == result['correct_decisions'] == 25
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        _evaluate_configuration(MeanIQ(), records, split_record_frames(total_samples=1024 * 128), state, **kwargs)
    assert output.read_bytes() == before


def test_official_training_resets_probe_state_and_restores_best_source_epoch(monkeypatch):
    # Bound expensive GPU training at its boundary, exercise real orchestration and state copies.
    from paper_reproduction.gaskin_tweak_2023 import official_lora_experiment as runner
    seen_initial = []
    sequence = iter(range(1, 9))  # five fresh probes, then three continuous epochs

    def epoch(model, optimizer, records, split, **kwargs):
        number = next(sequence)
        if number <= 6:
            seen_initial.append({key: value.clone() for key, value in model.state_dict().items()})
            assert not optimizer.state  # momentum must not carry across probes/main restart
        with torch.no_grad():
            model.embedding[-1].bias.fill_(float(number))
        return dict(batches=100, active_batches=100, first_window_mean_loss=.4,
                    last_window_mean_loss=.1, mean_training_loss=1 / number)

    def monitor(model, records, split, **kwargs):
        marker = int(model.embedding[-1].bias[0])
        return {'source_monitor_accuracy': {1: .1, 2: .8, 3: .4, 4: .3, 5: .2,
                                             6: .6, 7: .9, 8: .5}[marker]}

    monkeypatch.setattr(runner, '_run_training_epoch', epoch)
    monkeypatch.setattr(runner, '_source_monitor', monitor)
    model, metadata = runner._train_encoder([], None, device=torch.device('cpu'), epochs=3,
                                            seed=20260908, max_batches_per_epoch=None)
    assert len(seen_initial) == 6
    assert all(all(torch.equal(value, seen_initial[0][key]) for key, value in state.items()) for state in seen_initial)
    assert metadata['best_learning_rate'] == .001
    assert metadata['best_epoch'] == 2
    assert metadata['best_source_monitor_accuracy'] == .9
    assert model.embedding[-1].bias.tolist() == [7.] * 12
