"""Apply the user's performance-first priority before any new target prediction."""
from experiments.cvs_balanced_identity.dispatch import select_source_candidate


def select_performance_candidate(records):
    # Reuse the registered eight-row completeness and source-summary checks.
    initial = select_source_candidate(records)
    summary = initial['source_summaries']
    chosen = min(summary, key=lambda v: (
        -summary[v]['score'], -summary[v]['source_accuracy'],
        -summary[v]['worst_rx_accuracy'], summary[v]['conv_linear_macs'],
        summary[v]['parameters'], v))
    return dict(status='SOURCE_SELECTION_FROZEN', selected_variant=chosen,
        source_summaries=summary,
        selection_rule='performance first: maximize four-seed E200 0.5 source-V accuracy+0.5 worst source RX; exact score ties use V then worst RX; costs only after equal performance; no 0.2pp cost tolerance',
        selection_policy='performance_first_20261001', target_access=False,
        target_score_used=False, test_view='clean')
