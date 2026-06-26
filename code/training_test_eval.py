from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping


WISIG_MAIN_TEST_KEYS = [
    "test_unseen_day_seen_rx",
    "test_seen_day_unseen_rx",
    "test_unseen_day_unseen_rx",
]


@dataclass(frozen=True)
class TrainingTestEvalResult:
    val_stats: Dict[str, float]
    named_test_stats: Dict[str, Dict[str, float]]
    test_stats: Dict[str, float]
    lines: List[str]


def should_run_training_test(
    policy: str,
    *,
    epoch: int,
    epochs: int,
    val_improved: bool,
    start_epoch: int = 1,
    interval: int = 0,
) -> bool:
    normalized = str(policy or "every_epoch").strip().lower()
    if normalized == "interval_final":
        if int(epoch) >= int(epochs):
            return True
        if int(epoch) < int(start_epoch):
            return False
        step = max(0, int(interval or 0))
        return step > 0 and int(epoch) % step == 0
    if int(epoch) < int(start_epoch):
        return False
    if normalized == "every_epoch":
        return True
    if normalized == "val_improved_final":
        return bool(val_improved) or int(epoch) >= int(epochs)
    raise ValueError(f"unknown test evaluation policy: {policy}")


def aggregate_named_stats(named_stats: Mapping[str, Mapping[str, float]], keys: List[str]) -> Dict[str, float]:
    total_correct = 0
    total_count = 0
    for key in keys:
        if key not in named_stats:
            continue
        total_correct += int(named_stats[key].get("tx_correct", 0))
        total_count += int(named_stats[key].get("tx_total", 0))
    return {
        "tx_acc": 100.0 * total_correct / max(1, total_count),
        "tx_correct": int(total_correct),
        "tx_total": int(total_count),
    }


def make_test_subset_label(name: str, meta: Mapping[str, Any]) -> str:
    if name == "test_unseen_day_seen_rx":
        return f"unseen_day_seen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name == "test_seen_day_unseen_rx":
        return f"seen_day_unseen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name == "test_unseen_day_unseen_rx":
        return f"unseen_day_unseen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name.startswith("test_day_"):
        return f"day={meta.get('days_label', ['?'])[0]} on seen_rxs={meta.get('rxs_idx', [])}"
    if name.startswith("test_unseen_day_rx_"):
        return f"rx={meta.get('rxs_idx', ['?'])[0]} on unseen_days={meta.get('days_label', [])}"
    if name.startswith("test_rx_"):
        return f"rx={meta.get('rxs_idx', ['?'])[0]} on seen_days={meta.get('days_label', [])}"
    return name


def format_named_test_lines(
    named_test_stats: Mapping[str, Mapping[str, float]],
    named_test_meta: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    ordered_names = list(named_test_stats.keys())
    ordered_names = [key for key in WISIG_MAIN_TEST_KEYS if key in named_test_stats] + [
        key for key in ordered_names if key not in WISIG_MAIN_TEST_KEYS
    ]
    lines = []
    for name in ordered_names:
        stats = named_test_stats[name]
        meta = named_test_meta.get(name, {})
        label = make_test_subset_label(name, meta)
        lines.append(f"          {label}: tx={stats['tx_acc']:.2f}% ({stats['tx_correct']}/{stats['tx_total']})")
    return lines


def select_main_test_keys(named_test_stats: Mapping[str, Mapping[str, float]], dataset: str) -> List[str]:
    if str(dataset).lower() == "wisig":
        keys = [key for key in WISIG_MAIN_TEST_KEYS if key in named_test_stats]
        if keys:
            return keys
    return list(named_test_stats.keys())


def summarize_training_tests(
    named_test_stats: Mapping[str, Mapping[str, float]],
    named_test_meta: Mapping[str, Mapping[str, Any]],
    *,
    dataset: str,
) -> tuple[Dict[str, float], List[str]]:
    test_keys = select_main_test_keys(named_test_stats, dataset)
    test_stats = aggregate_named_stats(named_test_stats, test_keys)
    lines = [
        f"[TEST]  overall_tx={test_stats['tx_acc']:.2f}% ({test_stats['tx_correct']}/{test_stats['tx_total']})",
        "[TEST-SPLIT]",
    ]
    lines.extend(format_named_test_lines(named_test_stats, named_test_meta))
    return test_stats, lines


def evaluate_training_tests(
    *,
    model,
    val_loader,
    named_test_loaders,
    device,
    domain_label_map,
    named_test_meta,
    dataset: str,
    max_batches: int = 0,
    evaluate_loader_fn: Callable[..., Dict[str, float]],
    evaluate_named_loaders_fn: Callable[..., Dict[str, Dict[str, float]]],
) -> TrainingTestEvalResult:
    val_stats = evaluate_loader_fn(
        model,
        val_loader,
        device,
        domain_label_map=domain_label_map,
        max_batches=max_batches,
    )
    named_test_stats = evaluate_named_loaders_fn(
        model,
        named_test_loaders,
        device,
        domain_label_map=domain_label_map,
        max_batches=max_batches,
    )
    test_stats, lines = summarize_training_tests(
        named_test_stats,
        named_test_meta,
        dataset=dataset,
    )
    return TrainingTestEvalResult(
        val_stats=dict(val_stats),
        named_test_stats={name: dict(stats) for name, stats in named_test_stats.items()},
        test_stats=dict(test_stats),
        lines=list(lines),
    )
