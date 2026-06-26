def test_iter_train_batches_default_preserves_loader_once():
    from train import iter_train_batches_for_epoch

    batches = ["a", "b", "c"]

    assert list(iter_train_batches_for_epoch(batches, 0)) == [(0, "a"), (1, "b"), (2, "c")]


def test_iter_train_batches_repeats_until_requested_steps():
    from train import iter_train_batches_for_epoch

    batches = ["a", "b"]

    assert list(iter_train_batches_for_epoch(batches, 5)) == [
        (0, "a"),
        (1, "b"),
        (2, "a"),
        (3, "b"),
        (4, "a"),
    ]


def test_iter_train_batches_empty_loader_stops():
    from train import iter_train_batches_for_epoch

    assert list(iter_train_batches_for_epoch([], 4)) == []
