import torch

from experiments.phase1_dura_orthogonal import source_pair_inference as diagnostic


def test_pair_path_preserves_logits_and_skips_only_anchor_reader(monkeypatch):
    x = torch.randn(4, 2, 256)
    monkeypatch.setattr(diagnostic, 'extract_context', lambda dura, value: (value*.9, value*.1, torch.ones(len(value))))
    class Candidate:
        def __call__(self, received, anchor, direction, variance): return anchor+direction*.2
    class Reader:
        def __init__(self): self.calls = 0
        def features(self, value): self.calls += 1; return value.mean((1, 2))[:, None]
        def classify_features(self, value): return value.expand(-1, 6)
    reader = Reader()
    original = diagnostic.logits(x, Candidate(), None, reader, include_anchor=True)
    assert reader.calls == 3
    reader.calls = 0
    pair = diagnostic.logits(x, Candidate(), None, reader, include_anchor=False)
    assert reader.calls == 2
    assert all(torch.equal(a, b) for a, b in zip(original, pair))
