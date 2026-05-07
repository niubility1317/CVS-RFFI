from train import resolve_sat_eval_loader_names


def test_sat_target_alias_selects_all_wisig_ood_target_loaders():
    loaders = {
        "test_unseen_day_seen_rx": object(),
        "test_seen_day_unseen_rx": object(),
        "test_unseen_day_unseen_rx": object(),
        "test_day_0": object(),
    }

    assert resolve_sat_eval_loader_names(loaders, "target") == [
        "test_unseen_day_seen_rx",
        "test_seen_day_unseen_rx",
        "test_unseen_day_unseen_rx",
    ]


def test_sat_target_strict_alias_selects_only_strict_target_loader():
    loaders = {
        "test_unseen_day_seen_rx": object(),
        "test_seen_day_unseen_rx": object(),
        "test_unseen_day_unseen_rx": object(),
    }

    assert resolve_sat_eval_loader_names(loaders, "target_strict") == ["test_unseen_day_unseen_rx"]
