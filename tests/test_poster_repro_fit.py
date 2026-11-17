import torch
from experiments.poster_paper_repro.fit import fit_source, fit_support
from experiments.poster_paper_repro.model import PosterHomegrownAuthor


def test_source_fit_and_support_fit_do_not_modify_source_model():
    torch.manual_seed(90)
    model = PosterHomegrownAuthor(5, input_len=64)
    x = torch.randn(25, 2, 64)
    y = torch.arange(5).repeat_interleave(5)
    source, history = fit_source(model, x, y, x, y, seed=8, epochs=2, batch_size=25, patience=3)
    saved = {name: value.clone() for name, value in source.state_dict().items()}
    adapted, support_history = fit_support(source, x, y, mode='paper_last_layer', seed=8, epochs=2)
    assert len(history) == len(support_history) == 2
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, saved[name])
    torch.testing.assert_close(adapted.conv1.weight, source.conv1.weight)
    assert not torch.equal(adapted.head.weight, source.head.weight)
