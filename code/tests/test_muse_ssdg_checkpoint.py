import torch
from torch import nn

from SSDG import train_ssdg


class _TinyMUSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 4
        self.num_classes = 3
        self.num_domains = 2
        self.weight = nn.Parameter(torch.ones(1))


def _args():
    return train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "true", "--muse_level", "M2"]
    )


def test_muse_checkpoint_round_trip_restores_training_state_without_deployment_heads():
    state = train_ssdg._initialize_muse_training_state(_args(), _TinyMUSEModel(), torch.device("cpu"))
    state["schedule_state"] = train_ssdg.muse_schedule_for_epoch(41, state["config"])
    state["temporal_memory"].observe(
        [(1, 2, 3, 4, 5)], torch.tensor([2]), torch.tensor([0.9]), epoch=39
    )
    state["temporal_memory"].observe(
        [(1, 2, 3, 4, 5)], torch.tensor([2]), torch.tensor([0.95]), epoch=40
    )
    state["classification_prototypes"].observe(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([2]),
        torch.tensor([1]),
        torch.tensor([True]),
        torch.tensor([True]),
    )
    payload = {"model": {"weight": torch.ones(1)}}
    payload.update(train_ssdg._muse_checkpoint_state(state))

    restored = train_ssdg._initialize_muse_training_state(
        _args(), _TinyMUSEModel(), torch.device("cpu")
    )
    train_ssdg._restore_muse_checkpoint_state(restored, payload)

    for name, value in state["heads"].training_state_dict().items():
        assert torch.equal(restored["heads"].training_state_dict()[name], value)
    assert restored["temporal_memory"].state_dict() == state["temporal_memory"].state_dict()
    assert restored["classification_prototypes"].state_dict()["counts"] == {2: 1.0}
    restored_prototypes = restored["classification_prototypes"].state_dict()
    original_prototypes = state["classification_prototypes"].state_dict()
    assert restored_prototypes["feature_dim"] == original_prototypes["feature_dim"]
    assert restored_prototypes["counts"] == original_prototypes["counts"]
    assert restored_prototypes["domain_counts"] == original_prototypes["domain_counts"]
    assert torch.equal(restored_prototypes["prototypes"][2], original_prototypes["prototypes"][2])
    assert restored["schedule_state"] == state["schedule_state"]
    assert "muse_training_heads" not in payload["model"]
