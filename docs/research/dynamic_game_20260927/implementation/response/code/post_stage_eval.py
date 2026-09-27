from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Tuple


WISIG_MAIN_TEST_KEYS = [
    "test_unseen_day_seen_rx",
    "test_seen_day_unseen_rx",
    "test_unseen_day_unseen_rx",
]


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
    ordered_names = [k for k in WISIG_MAIN_TEST_KEYS if k in named_test_stats] + [
        k for k in ordered_names if k not in WISIG_MAIN_TEST_KEYS
    ]
    lines = []
    for name in ordered_names:
        stats = named_test_stats[name]
        meta = named_test_meta.get(name, {})
        label = make_test_subset_label(name, meta)
        lines.append(f"          {label}: tx={stats['tx_acc']:.2f}% ({stats['tx_correct']}/{stats['tx_total']})")
    return lines


def select_post_stage_test_keys(named_test_stats: Mapping[str, Mapping[str, float]], dataset: str) -> List[str]:
    if str(dataset).lower() == "wisig":
        keys = [key for key in WISIG_MAIN_TEST_KEYS if key in named_test_stats]
        if keys:
            return keys
    return list(named_test_stats.keys())


def summarize_post_stage_tests(
    named_test_stats: Mapping[str, Mapping[str, float]],
    data_ctx: Mapping[str, Any],
    *,
    dataset: str,
) -> Tuple[Dict[str, float], List[str]]:
    keys = select_post_stage_test_keys(named_test_stats, dataset)
    test_stats = aggregate_named_stats(named_test_stats, keys)
    lines = [
        f"[TEST]  overall_tx={test_stats['tx_acc']:.2f}% ({test_stats['tx_correct']}/{test_stats['tx_total']})",
        "[TEST-SPLIT]",
    ]
    lines.extend(format_named_test_lines(named_test_stats, data_ctx.get("named_test_meta", {}) or {}))
    return test_stats, lines


def resolve_sat_eval_max_batches(sat_eval_max_batches: int, eval_max_batches: int) -> int:
    value = int(sat_eval_max_batches)
    return int(eval_max_batches) if value < 0 else value


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rank(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def spearman_rank_corr(records: Iterable[Mapping[str, Any]], left_key: str, right_key: str) -> float:
    pairs = []
    for record in records:
        left = _as_float(record.get(left_key))
        right = _as_float(record.get(right_key))
        if math.isfinite(left) and math.isfinite(right):
            pairs.append((left, right))
    if len(pairs) < 2:
        return float("nan")
    left_ranks = _rank([p[0] for p in pairs])
    right_ranks = _rank([p[1] for p in pairs])
    return _pearson(left_ranks, right_ranks)


def summarize_epoch_records(
    records: Iterable[Mapping[str, Any]],
    *,
    proxy_key: str = "proxy_val_rx_day",
    test_key: str = "unseen_day_unseen_rx",
    source_key: str = "val_source",
) -> Dict[str, float]:
    ordered = sorted([dict(record) for record in records], key=lambda item: int(item.get("epoch", 0)))
    if not ordered:
        return {}

    final = ordered[-1]

    def best_record(key: str) -> Dict[str, Any]:
        finite = [record for record in ordered if math.isfinite(_as_float(record.get(key)))]
        if not finite:
            return {}
        return max(finite, key=lambda record: _as_float(record.get(key), float("-inf")))

    best_source = best_record(source_key)
    best_proxy = best_record(proxy_key)
    best_test = best_record(test_key)
    final_test = _as_float(final.get(test_key))
    best_test_value = _as_float(best_test.get(test_key)) if best_test else float("nan")

    return {
        "final_epoch": int(final.get("epoch", 0)),
        "final_source": _as_float(final.get(source_key)),
        "final_proxy": _as_float(final.get(proxy_key)),
        "final_test": final_test,
        "best_source_epoch": int(best_source.get("epoch", 0)) if best_source else 0,
        "best_source_value": _as_float(best_source.get(source_key)) if best_source else float("nan"),
        "best_proxy_epoch": int(best_proxy.get("epoch", 0)) if best_proxy else 0,
        "best_proxy_value": _as_float(best_proxy.get(proxy_key)) if best_proxy else float("nan"),
        "best_proxy_test": _as_float(best_proxy.get(test_key)) if best_proxy else float("nan"),
        "best_test_epoch": int(best_test.get("epoch", 0)) if best_test else 0,
        "best_test_value": best_test_value,
        "final_minus_best_test": final_test - best_test_value if math.isfinite(final_test) and math.isfinite(best_test_value) else float("nan"),
        "proxy_test_rank_corr": spearman_rank_corr(ordered, proxy_key, test_key),
        "source_test_rank_corr": spearman_rank_corr(ordered, source_key, test_key),
        "harm_rate": _as_float(final.get("harm_rate")),
        "rescue_rate": _as_float(final.get("rescue_rate")),
        "net_gain_rate": _as_float(final.get("net_gain_rate")),
        "changed_pred_rate": _as_float(final.get("changed_pred_rate")),
    }
