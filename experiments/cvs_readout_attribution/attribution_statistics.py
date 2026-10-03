"""Frozen source-V descriptions: paired readout ablations and balanced factor scatter.

Only NumPy arrays are consumed. No model fitting, checkpoint selection, data access,
or inference about physical channel disentanglement is performed here.
"""
import numpy as np


FEATURES = ('time_skip', 'time_delta', 'behavior_skip', 'behavior_delta')
CONDITIONS = ('all_on', 'time_off', 'behavior_off', 'both_off')
FACTORS = ('tx', 'receiver', 'day', 'tx_receiver', 'tx_day', 'receiver_day',
           'tx_receiver_day', 'within_cell')
REQUIRED_FIELDS = frozenset(('ids', 'truth', 'receiver', 'day') + FEATURES + CONDITIONS)
RATIO_EPSILON = 1e-12


def _validate(q, require_complete):
    if set(q) != REQUIRED_FIELDS:
        raise ValueError('Attribution artifact fields differ from the exact source array contract')
    arrays = {key: np.asarray(q[key]) for key in REQUIRED_FIELDS}
    truth = arrays['truth']
    if truth.ndim != 1 or not len(truth):
        raise ValueError('Source labels must be nonempty one-dimensional arrays')
    n = len(truth)
    for key in ('ids', 'truth', 'receiver', 'day'):
        if arrays[key].shape != (n,):
            raise ValueError('Source metadata must have one paired value per sample')
    for key in ('truth', 'receiver', 'day'):
        if not np.issubdtype(arrays[key].dtype, np.integer):
            raise ValueError('Source factor labels must have integer dtype')
    if np.any((truth < 0) | (truth >= 6)):
        raise ValueError('Truth must index the six logits, from 0 to 5')
    ids = arrays['ids']
    if ids.dtype.kind not in 'iuUS' or (ids.dtype.kind in 'US' and np.any(np.char.str_len(ids) == 0)):
        raise ValueError('IDs must be integer or nonempty string values')
    if len(np.unique(ids)) != n:
        raise ValueError('Source IDs must be unique')
    for key in FEATURES + CONDITIONS:
        width = 320 if key in FEATURES else 6
        if arrays[key].shape != (n, width) or not np.issubdtype(arrays[key].dtype, np.floating):
            raise ValueError(f'{key} must be a floating array with shape ({n}, {width})')
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'{key} contains nonfinite values')
    levels = [np.unique(arrays[key]) for key in ('truth', 'receiver', 'day')]
    if require_complete and (n != 27000 or any(not np.array_equal(actual, expected) for actual, expected in
            zip(levels, (np.arange(6), np.array([1, 3, 4, 6, 8]), np.array([1, 2, 3]))))):
        raise ValueError('Complete source V requires 27000 samples, six TX, five fixed RX and three days')
    positions = [np.searchsorted(level, arrays[key]) for level, key in zip(levels, ('truth', 'receiver', 'day'))]
    shape = tuple(len(level) for level in levels)
    cell_index = np.ravel_multi_index(positions, shape)
    counts = np.bincount(cell_index, minlength=int(np.prod(shape)))
    if not np.all(counts == counts[0]) or counts[0] == 0:
        raise ValueError('Every TX/RX/day Cartesian cell must be present and equally populated')
    if require_complete and counts[0] != 300:
        raise ValueError('Complete source cells require 300 samples each')
    cells = [dict(tx=int(tx), receiver=int(rx), day=int(day), count=int(counts[0]))
             for tx in levels[0] for rx in levels[1] for day in levels[2]]
    return arrays, levels, shape, cell_index, cells


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0, .01, .1, .25, .5, .75, .9, .99, 1])
    return dict(count=len(values), mean=float(values.mean()), **dict(zip(
        ('minimum', 'p01', 'p10', 'p25', 'median', 'p75', 'p90', 'p99', 'maximum'),
        map(float, quantiles))))


def _logit_stats(logits, truth):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(truth)), truth]
    predictions = np.argmax(logits, axis=1)
    return predictions, predictions == truth, ce


def _strata(values, receiver, levels, cell_index, cells, metric):
    """Averages of a paired per-sample quantity, retaining the full source grid."""
    per_rx = {str(int(rx)): dict(count=int(np.count_nonzero(receiver == rx)),
        **{metric: float(values[receiver == rx].mean())}) for rx in levels[1]}
    per_cell = [dict(cell, **{metric: float(values[cell_index == index].mean())})
                for index, cell in enumerate(cells)]
    return per_rx, per_cell


def _factor_scatter(features, shape, cell_index, cells):
    """Balanced three-way orthogonal ANOVA of raw feature means (sum of squares)."""
    z = np.asarray(features, dtype=np.float64)
    # Subtract a common reference first to avoid introducing scatter for a
    # constant feature whose decimal value is not exactly representable.
    centered = z - z[0]
    means = np.zeros((len(cells), z.shape[1]), dtype=np.float64)
    np.add.at(means, cell_index, centered)
    per_cell = cells[0]['count']
    means /= per_cell
    grid = means.reshape(shape + (z.shape[1],))
    grand = grid.mean(axis=(0, 1, 2), keepdims=True)
    tx = grid.mean(axis=(1, 2), keepdims=True)-grand
    rx = grid.mean(axis=(0, 2), keepdims=True)-grand
    day = grid.mean(axis=(0, 1), keepdims=True)-grand
    tx_rx = grid.mean(axis=2, keepdims=True)-grand-tx-rx
    tx_day = grid.mean(axis=1, keepdims=True)-grand-tx-day
    rx_day = grid.mean(axis=0, keepdims=True)-grand-rx-day
    tx_rx_day = grid-grand-tx-rx-day-tx_rx-tx_day-rx_day
    terms = (tx, rx, day, tx_rx, tx_day, rx_day, tx_rx_day)
    components = {name: float(per_cell*np.square(np.broadcast_to(term, grid.shape)).sum(dtype=np.float64))
                  for name, term in zip(FACTORS[:-1], terms)}
    components['within_cell'] = float(np.square(centered-means[cell_index]).sum(dtype=np.float64))
    total = float(np.square(centered-grand.reshape(-1)).sum(dtype=np.float64))
    reconstructed = sum(components.values())
    if not np.isfinite(total) or not np.isfinite(reconstructed):
        raise ValueError('Feature scatter exceeds finite float64 range')
    if not np.isclose(total, reconstructed, rtol=1e-10, atol=1e-10*max(total, np.finfo(np.float64).tiny)):
        raise ValueError('Orthogonal factor scatter does not reconstruct total scatter')
    return dict(total=total, components=components,
        fractions={name: None if total == 0 else value/total for name, value in components.items()},
        reconstructed_total=reconstructed, reconstruction_absolute_error=abs(total-reconstructed),
        reconstruction_relative_error=None if total == 0 else abs(total-reconstructed)/total,
        cells=len(cells), samples_per_cell=per_cell,
        definition='Raw 320-dimensional feature sum of squared Euclidean deviations. Balanced TX/RX/day main effects, pairwise interactions, three-way interaction and within-cell residual are orthogonal.',
        interpretation='Observed source-label associations; not causal factors or physical channel disentanglement.')


def analyze_arrays(q, require_complete=True):
    """Describe paired frozen-source arrays using float64 arithmetic.

    Complete mode requires the registered 6 x 5 x 3 cells of 300 samples.
    Synthetic mode accepts the complete balanced Cartesian product of observed
    factor levels. Truth always indexes the six logit columns. Returned values
    are JSON-serializable; accuracies are fractions and *_pp values are points.
    """
    arrays, levels, shape, cell_index, cells = _validate(q, require_complete)
    truth, receiver = arrays['truth'], arrays['receiver']
    n = len(truth)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            stats = {name: _logit_stats(arrays[name], truth) for name in CONDITIONS}
            classification = {}
            for name, (_, correct, ce) in stats.items():
                per_rx, per_cell = _strata(correct, receiver, levels, cell_index, cells, 'accuracy')
                for rx in levels[1]:
                    per_rx[str(int(rx))]['ce'] = float(ce[receiver == rx].mean())
                for index, cell in enumerate(per_cell):
                    cell['ce'] = float(ce[cell_index == index].mean())
                classification[name] = dict(count=n, accuracy=float(correct.mean()), ce=float(ce.mean()),
                                             receiver=per_rx, cells=per_cell)
            on_pred, on_correct, on_ce = stats['all_on']
            paired = {}
            for name in CONDITIONS[1:]:
                pred, correct, ce = stats[name]
                helped = int(np.count_nonzero(on_correct & ~correct))
                hurt = int(np.count_nonzero(~on_correct & correct))
                changed = int(np.count_nonzero(on_pred != pred))
                ce_delta = on_ce-ce
                paired[name] = dict(reference_condition=name, count=n, helped=helped, hurt=hurt,
                    both_correct=int(np.count_nonzero(on_correct & correct)),
                    both_wrong=int(np.count_nonzero(~on_correct & ~correct)),
                    prediction_changed=changed, helped_fraction=helped/n, hurt_fraction=hurt/n,
                    prediction_changed_fraction=changed/n, accuracy_delta_pp=100*(helped-hurt)/n,
                    ce_delta=float(ce_delta.mean()), ce_delta_summary=_summary(ce_delta),
                    definition='all_on minus reference; helped/hurt denote wrong-to-correct/correct-to-wrong on the same IDs; negative CE delta favors all_on.')
            interaction_values = 100*(stats['all_on'][1].astype(np.float64)
                -stats['time_off'][1]-stats['behavior_off'][1]+stats['both_off'][1])
            per_rx, per_cell = _strata(interaction_values, receiver, levels, cell_index, cells, 'value_pp')
            interaction = dict(count=n, value_pp=float(interaction_values.mean()), receiver=per_rx, cells=per_cell,
                formula='100 * (accuracy(all_on) - accuracy(time_off) - accuracy(behavior_off) + accuracy(both_off))',
                interpretation='Positive means the joint accuracy gain exceeds the sum of isolated-branch gains over both_off; fixed-model source-V intervention, not physical-channel causality.')
            ratios = {}
            for path in ('time', 'behavior'):
                skip = np.linalg.norm(arrays[path+'_skip'].astype(np.float64), axis=1)
                delta = np.linalg.norm(arrays[path+'_delta'].astype(np.float64), axis=1)
                ratios[path] = dict(_summary(delta/np.maximum(skip, RATIO_EPSILON)),
                    denominator_epsilon=RATIO_EPSILON, zero_skip_count=int(np.count_nonzero(skip == 0)),
                    floored_skip_count=int(np.count_nonzero(skip < RATIO_EPSILON)),
                    definition='Per-sample L2(delta) / max(L2(skip), 1e-12), all source-V samples; NumPy linear quantiles.')
            scatter = {name: _factor_scatter(arrays[name], shape, cell_index, cells) for name in FEATURES}
    except FloatingPointError as error:
        raise ValueError('Attribution statistic exceeds finite float64 range') from error
    return dict(count=n, complete_source_required=bool(require_complete),
        factor_levels=dict(tx=levels[0].tolist(), receiver=levels[1].tolist(), day=levels[2].tolist()),
        cell_count=len(cells), samples_per_cell=cells[0]['count'], classification=classification,
        paired_all_on=paired, accuracy_interaction=interaction, norm_ratios=ratios, factor_scatter=scatter,
        statistics_dtype='float64', source_only=True, model_fitting=False, model_selection=False,
        interpretation='Frozen source-V descriptions on paired IDs. Feature factors are observational associations, not causal attribution or physical channel disentanglement; no independent-target generalization claim.')
