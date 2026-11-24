import pytest
import torch
from torch import nn

from experiments.pamf_lora_main.external_full8 import RUN, restore, role_dataset, source_runtime
from experiments.pamf_lora_main.external_full8_delivery import frozen_predictions
from experiments.pamf_lora_main.external_full8_models import PacketEdges, PartitionedPacketEncoder, packet_chirps


def test_physical_chirps_preserve_packet_order():
    x = torch.arange(2 * 8 * 2 * 384).reshape(2, 8, 2, 384)
    grouped = packet_chirps(x)
    assert torch.equal(grouped[7], x[0, 7])
    assert torch.equal(grouped[8], x[1, 0])
    with pytest.raises(ValueError, match='physical IQ'):
        packet_chirps(x[:, :, :, :256])


def test_packet_edges_use_ordered_first_and_last_relative_to_middle():
    edges = PacketEdges(2)
    inputs = torch.arange(16).reshape(8, 2).float()
    inputs[-1, 1] += 10
    captured = []
    hook = edges.project.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    output = edges(inputs, 1)
    hook.remove()
    interior = inputs[1:-1].mean(0)
    assert torch.equal(captured[0], torch.cat((inputs[0] - interior, inputs[-1] - interior))[None])
    assert output.shape == (1, 2)


def test_receiver_edges_do_not_mix_into_emitter_features():
    class Native(nn.Module):
        def forward(self, iq):
            return iq[:, :, :3].flatten(1)

    net = PartitionedPacketEncoder(Native(), 3, 3)
    x = torch.randn(4, 8, 2, 384)
    changed = x.clone()
    changed[:, 0, 1] += 3
    actual = net(x)
    other = net(changed)
    assert torch.equal(actual[:, :3], other[:, :3])
    assert not torch.equal(actual[:, 3:], other[:, 3:])


def test_source_and_scorer_refuse_illegal_or_incomplete_roles(tmp_path):
    with pytest.raises(ValueError, match='refuses target'):
        source_runtime({'run_id': RUN, 'native_target_root': tmp_path})
    with pytest.raises(ValueError, match='only authorized'):
        role_dataset(tmp_path, {}, tmp_path, 'U_s')
    with pytest.raises(ValueError, match='each registered'):
        frozen_predictions(tmp_path, ['riei', 'drift', 'ibrff'])


def test_owned_checkpoint_load_supports_legacy_pytorch_signature(tmp_path, monkeypatch):
    path = tmp_path/'stage1.pt'
    torch.save(dict(config={}, source_contract={}, stage=1, scratch=True,
                    target_reads=0, U_truth_reads=0, satellite_batches=0,
                    default_head_joint_training=True), path)
    current_load = torch.load

    def legacy_load(path, map_location):
        return current_load(path, map_location=map_location, weights_only=True)

    monkeypatch.setattr(torch, 'load', legacy_load)
    assert restore(path, {}, {}, 1, 'cpu')['stage'] == 1
    with pytest.raises(ValueError, match='provenance'):
        restore(path, {}, {}, 2, 'cpu')
