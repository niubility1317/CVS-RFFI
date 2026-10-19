"""Plot completed frozen source-fit JSON; no model, IQ or remote access.

Named-stream and cached-origin subsets are displayed separately.  Their fresh
L partners retain the same physical IDs and scene assignment.  Method pairs
use common IQ within each condition; different condition compositions are not
interpreted as a cache effect.  Online training values are never plotted.
"""
import argparse
import json
import math
from pathlib import Path

RUN = '20261019-diagnostic-F-epochReplay-sourceFit-e20-s2026092701-m2-r01'
ARMS = ('B0', 'B1', 'EPOCH_REPLAY')
M = 'EPOCH_REPLAY'
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
SCENE_TITLES = ('Low suburban', 'Mid urban', 'Low urban')
ORIGINS = ('old_B1_observed', 'M_cached_used')
NAMES = {'B0': 'B0: ordinary NoAs', 'B1': 'B1: fresh As', M: 'M: epoch replay'}
COLORS = {'B0': '#7B858C', 'B1': '#286F9C', M: '#C13F4A'}
MARKERS = {'B0': 'o', 'B1': 's', M: 'D'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_metric(value):
    n, accuracy, ce = value['n'], value['accuracy'], value['hard_label_NLL']
    require(isinstance(n, int) and not isinstance(n, bool) and n > 0,
            'A positive source observation count is required')
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) for v in (accuracy, ce))
            and 0 <= accuracy <= 1 and ce >= 0,
            'Finite frozen accuracy and hard-label CE are required')
    correct = round(n * accuracy)
    require(abs(n * accuracy - correct) <= 1e-6
            and value.get('correct', correct) == correct,
            'Saved frozen correct count differs from accuracy')
    return dict(observations=n, correct=correct, accuracy_percent=100 * accuracy,
                standard_CE=ce)


def checked_pair(value, before, after, expected_n):
    require(value['n'] == expected_n, 'Paired source observation count differs')
    counts = [value[k] for k in ('corrected', 'damaged', 'both_correct', 'both_wrong')]
    require(all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts)
            and sum(counts) == expected_n, 'Paired correct/error counts do not partition observations')
    require(expected_n % before['observations'] == expected_n % after['observations'] == 0
            and before['correct'] * (expected_n // before['observations'])
                == value['both_correct'] + value['damaged']
            and after['correct'] * (expected_n // after['observations'])
                == value['both_correct'] + value['corrected'],
            'Paired correct counts differ from the repeated matched source sets')
    net = 100 * (value['corrected'] - value['damaged']) / expected_n
    require(math.isclose(net, value['net_pp'], abs_tol=1e-9, rel_tol=0)
            and math.isclose(net, after['accuracy_percent'] - before['accuracy_percent'],
                             abs_tol=1e-9, rel_tol=0),
            'Frozen paired accuracy difference differs from common-IQ metrics')
    ce = after['standard_CE'] - before['standard_CE']
    if 'CE_delta' in value:
        require(math.isclose(ce, value['CE_delta']['hard_label_NLL'], abs_tol=1e-10, rel_tol=0),
                'Frozen paired CE difference differs')
    return dict(observations=expected_n, mean_accuracy_delta_pp=net,
                mean_CE_delta=ce, corrected=value['corrected'], new_errors=value['damaged'],
                both_correct=value['both_correct'], both_wrong=value['both_wrong'],
                model_seed_count=1, sample_SD=None)


def prepare(data):
    """Expose saved metrics only, preserving each observation composition."""
    require(data['status'] == 'SOURCE_ONLY_FROZEN_READOUT_COMPLETE' and data['run_id'] == RUN
            and data['model_seed_count'] == 1 and data['target_access'] is False
            and data['source_promotion_changed'] is False
            and all(data[k] == 0 for k in ('optimizer_updates', 'EMA_updates', 'head_fit_updates')),
            'A completed frozen single-seed source-fit diagnostic is required')
    metadata = data['metadata']
    require(metadata['historical_IQ_byte_comparison'] is False
            and metadata['source_promotion_changed'] is False and metadata['target_access'] is False
            and tuple(metadata['scenes']) == SCENES
            and metadata['raw180_is_not_fullL6300'] and metadata['V1800_is_not_fullV27000'],
            'Frozen source-fit scope or reconstruction boundary differs')
    raw = {arm: checked_metric(data['L'][arm]['raw180']['overall']) for arm in ARMS}
    require(all(value['observations'] == 180 for value in raw.values()),
            'Raw L reference must retain all180 original physical IDs')
    panels = {}
    for origin in ORIGINS:
        arms = ARMS if origin == 'old_B1_observed' else ('B1', M)
        scenes = {}
        for scene in SCENES:
            values, physical_n = {}, None
            for arm in arms:
                observed = data['L'][arm][origin].get('by_scene', {}).get(scene)
                fresh = data['L'][arm]['new_paired_to_observed_scene'][origin].get('by_scene', {}).get(scene)
                v = checked_metric(data['V'][arm]['new1800_by_scene'][scene]['overall'])
                require(v['observations'] == 3600, 'Fresh source V must use1800 IDs by2 draws')
                require((observed is None) == (fresh is None), 'Matched scene subset is incomplete')
                if observed is None:
                    points, n = [None, None, v], 0
                else:
                    a, b = checked_metric(observed), checked_metric(fresh)
                    n = a['observations']
                    require(b['observations'] == 2 * n,
                            'Fresh L must retain exactly the observed scene subset by2 draws')
                    points = [a, b, v]
                if physical_n is None:
                    physical_n = n
                require(n == physical_n, 'Methods must share identical physical/scene groups within a panel')
                values[arm] = points
            scenes[scene] = dict(L_physical_IDs=physical_n, L_fresh_draws=2,
                                 V_physical_IDs=1800, V_draws=2, arms=values)
        require(sum(value['L_physical_IDs'] for value in scenes.values()) == 180,
                'Each named/cached scene composition must partition all180 L IDs')
        panels[origin] = scenes

    method_pairs = data['common_IQ_method_pairs'][M + '_minus_B1']
    common = {}
    matched_seen_to_new = {origin: {} for origin in ORIGINS}
    for scene in SCENES:
        comparisons = {}
        for origin in ORIGINS:
            panel = panels[origin][scene]
            if panel['L_physical_IDs']:
                before, after = panel['arms']['B1'][0], panel['arms'][M][0]
                pair = method_pairs[origin]['by_scene'][scene]
                comparisons[origin] = checked_pair(pair, before, after, panel['L_physical_IDs'])
            else:
                comparisons[origin] = None
            matched_seen_to_new[origin][scene] = {}
            for arm in ('B1', M):
                if panel['L_physical_IDs']:
                    before, after = panel['arms'][arm][:2]
                    pair = data['observed_to_new_sameL_pairs'][arm][origin]['by_scene'][scene]
                    matched_seen_to_new[origin][scene][arm] = checked_pair(
                        pair, before, after, 2 * panel['L_physical_IDs'])
                else:
                    matched_seen_to_new[origin][scene][arm] = None
        for label, family, n in (('fresh_L_all180', 'L', 360), ('fresh_V1800', 'V', 3600)):
            key = 'new_all180_by_scene' if family == 'L' else 'new1800_by_scene'
            before = checked_metric(data[family]['B1'][key][scene]['overall'])
            after = checked_metric(data[family][M][key][scene]['overall'])
            require(before['observations'] == after['observations'] == n,
                    'Fresh common-IQ method comparison count differs')
            pair = method_pairs[key][scene] if family == 'L' else method_pairs['V'][key][scene]
            comparisons[label] = checked_pair(pair['overall'], before, after, n)
            if 'CE_delta' in pair:
                require(math.isclose(after['standard_CE'] - before['standard_CE'],
                                    pair['CE_delta']['hard_label_NLL'], abs_tol=1e-10, rel_tol=0),
                        'Fresh common-IQ CE delta differs')
        common[scene] = comparisons
    return dict(raw_all180=raw, panels=panels, common_IQ_M_minus_B1=common,
                matched_seen_to_new=matched_seen_to_new,
                scene_composition={origin: {scene: panels[origin][scene]['L_physical_IDs']
                                            for scene in SCENES} for origin in ORIGINS},
                fullV27000_clean=data['fullV27000_clean'],
                origin_counts={key: metadata[key] for key in (
                    'same_origin_packets', 'different_origin_packets', 'old_and_cached_IQ_equal_packets')},
                frozen_readout_only=True, online_training_values_used=False,
                standard_CE='Mean hard-label CE on default logits; no temperature or label smoothing',
                named_stream_reconstruction_is_historical_byte_comparison=False,
                B0_has_no_cached_IQ_readout=True, model_seed_count=1, sample_SD=None)


def setup_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#C4CBD2')
    ax.spines['bottom'].set_color('#C4CBD2')
    ax.grid(axis='y', color='#D8DDE2', alpha=.75, linewidth=.7)
    ax.tick_params(labelsize=9.5)


def fit_figure(payload, metric, prefix, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator
    accuracy = metric == 'accuracy_percent'
    kind = 'accuracy' if accuracy else 'CE'
    fig = plt.figure(figsize=(17.0, 11.6), facecolor='white')
    grid = fig.add_gridspec(3, 3, height_ratios=[.55, 2.4, 2.4], left=.065, right=.985,
                           top=.81, bottom=.255, hspace=.72, wspace=.21)
    fig.text(.065, .96, 'Frozen source fit: '+('accuracy' if accuracy else 'standard CE'),
             fontsize=21, weight='bold')
    fig.text(.065, .925, 'E20 default heads | source-only | original180 L IDs | one model seed',
             fontsize=12, color='#3E4650')
    cards = fig.add_subplot(grid[0, :]); cards.set_axis_off()
    cards.text(0, 1.35, 'Raw L reference: all180 IDs; separate from the scenario subsets below',
               fontsize=11, color='#3E4650', weight='bold', transform=cards.transAxes)
    for i, arm in enumerate(ARMS):
        left = .003+i/3
        cards.add_patch(FancyBboxPatch((left, -.08), .316, 1.03, transform=cards.transAxes,
            boxstyle='round,pad=.006,rounding_size=.025', facecolor='#FAFBFC',
            edgecolor=COLORS[arm], linewidth=1.1, clip_on=False))
        value = payload['raw_all180'][arm]
        label = (f"{value[metric]:.2f}% ({value['correct']}/180)" if accuracy
                 else f"CE = {value[metric]:.4f}")
        cards.text(left+.158, .64, NAMES[arm], ha='center', va='center', fontsize=11,
                   color=COLORS[arm], weight='bold', transform=cards.transAxes)
        cards.text(left+.158, .15, label, ha='center', va='center', fontsize=12,
                   color='#1D2732', transform=cards.transAxes)
    maximum = max(point[metric] for scenes in payload['panels'].values() for panel in scenes.values()
                  for values in panel['arms'].values() for point in values if point is not None)
    ymax = 100 if accuracy else max(.1, maximum*1.18)
    for row_index, origin in enumerate(ORIGINS):
        for column, scene in enumerate(SCENES):
            ax = fig.add_subplot(grid[row_index+1, column]); setup_axes(ax)
            if column == 2:
                ax.set_facecolor('#FFF5F4')
            panel = payload['panels'][origin][scene]
            n = panel['L_physical_IDs']
            for arm, points in panel['arms'].items():
                width = 2.3 if arm == M else 1.7
                if n:
                    ax.plot([0, 1], [value[metric] for value in points[:2]],
                            color=COLORS[arm], marker=MARKERS[arm], markersize=5.5,
                            linewidth=width, zorder=4 if arm == M else 3)
                # V has different physical IDs: no connector to the L pair.
                ax.plot([2], [points[2][metric]], color=COLORS[arm], marker=MARKERS[arm],
                        markersize=6, linestyle='none')
            title = 'B1 named-stream groups' if origin == 'old_B1_observed' else 'M cached-origin groups (B1 and M)'
            ax.set_title(SCENE_TITLES[column]+' | '+title, fontsize=11.2, pad=13)
            seen = 'Named As' if origin == 'old_B1_observed' else 'Cached origin'
            ax.set_xticks([0, 1, 2], [f'{seen}\n{n} L IDs', f'Fresh, matched L\n{n} IDs x 2 draws',
                                     'Fresh V\n1800 IDs x 2 draws'])
            ax.set_xlim(-.18, 2.18); ax.set_ylim(0, ymax)
            ax.tick_params(axis='x', pad=7)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if not n:
                ax.text(.20, .5, 'No L IDs in this scene group', transform=ax.transAxes,
                        fontsize=9, color='#5C6670', ha='center')
            if column == 0:
                ax.set_ylabel('Accuracy (%)' if accuracy else 'Standard hard-label CE', fontsize=11)
    handles = [Line2D([0], [0], color=COLORS[arm], marker=MARKERS[arm], linewidth=2,
                      markersize=6, label=NAMES[arm]) for arm in ARMS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .172),
               ncol=3, frameon=False, fontsize=11, columnspacing=2.5)
    composition = payload['scene_composition']
    counts = lambda origin: '/'.join(str(composition[origin][s]) for s in SCENES)
    notes = [
        'Named and cached groups have separate scene compositions: named '+counts('old_B1_observed')+
        '; cached '+counts('M_cached_used')+' (low suburban / mid urban / low urban).',
        'Only each solid L segment shares physical IDs and scene assignment. Channel state/noise may still differ; no cross-composition net cache effect is claimed.',
        'V points use different IDs and are descriptive. B0 has no cached-IQ readout. All points use frozen heads; online training accuracy is excluded.',
        'Named streams are deterministic reconstructions, without historical IQ byte comparison. Two channel draws are not model seeds; no cross-seed error bars.',
        'CE is standard true-label CE on default logits (T=1, smoothing=0); training CE used smoothing=0.01. Raw180 and V1800 are distinct from full L6300/V27000.'
    ]
    for i, note in enumerate(notes):
        fig.text(.065, .146-i*.025, note, fontsize=9.2, color='#4A525C')
    files = [Path(str(prefix)+'_'+kind+suffix) for suffix in ('.png', '.pdf')]
    for path in files:
        fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return [str(path) for path in files]


def pair_figure(payload, prefix, plt):
    from matplotlib.ticker import MaxNLocator
    keys = ('old_B1_observed', 'M_cached_used', 'fresh_L_all180', 'fresh_V1800')
    labels = ('B1 named As', 'M cached As', 'Fresh L180', 'Fresh V1800')
    shades = ('#7B858C', '#A66D25', '#286F9C', '#C13F4A')
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.6), facecolor='white')
    fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.24, hspace=.47, wspace=.23)
    fig.text(.065, .955, 'M minus B1: paired frozen readouts on common IQ', fontsize=20, weight='bold')
    fig.text(.065, .911, 'Each column category retains its own identical IQ, physical IDs and scene group across both models.',
             fontsize=12, color='#3E4650')
    for column, scene in enumerate(SCENES):
        values = payload['common_IQ_M_minus_B1'][scene]
        for row, (metric, ylabel) in enumerate((('mean_accuracy_delta_pp', 'Accuracy difference (pp)'),
                                                ('mean_CE_delta', 'Standard CE difference'))):
            ax = axes[row, column]; setup_axes(ax)
            if column == 2:
                ax.set_facecolor('#FFF5F4')
            for i, key in enumerate(keys):
                value = values[key]
                if value is None:
                    continue
                delta = value[metric]
                ax.vlines(i, 0, delta, color=shades[i], linewidth=2)
                ax.scatter([i], [delta], color=shades[i], marker='D', s=36, zorder=3)
                ax.annotate(f"{delta:+.2f}" if row == 0 else f"{delta:+.4f}", (i, delta),
                            xytext=(0, 6 if delta >= 0 else -13), textcoords='offset points',
                            ha='center', fontsize=9, color='#3E4650')
            ax.axhline(0, color='#7B858C', linewidth=.8)
            ax.set_xticks(range(4), [label+'\nn='+str(0 if values[key] is None else values[key]['observations'])
                                    for key, label in zip(keys, labels)])
            ax.set_xlim(-.45, 3.45); ax.margins(y=.30)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if row == 0:
                ax.set_title(SCENE_TITLES[column], fontsize=12.5, pad=12)
            if column == 0:
                ax.set_ylabel(ylabel, fontsize=11)
    notes = [
        'Positive accuracy differences favor M; negative CE differences favor M. Counts are observations, including both draws where applicable.',
        'Named/cached categories may use different L scene subsets. Compare M with B1 inside each category; differences between categories are not net treatment effects.',
        'n=1 model seed, SD unavailable. Channel draws and repeated physical IDs are not independent model seeds. No model, IQ or target data was read to draw these figures.'
    ]
    for i, note in enumerate(notes):
        fig.text(.065, .14-i*.033, note, fontsize=9.6, color='#4A525C')
    files = [Path(str(prefix)+'_common_IQ_pairs'+suffix) for suffix in ('.png', '.pdf')]
    for path in files:
        fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    return [str(path) for path in files]


def main(input_path, output_prefix):
    source, prefix = Path(input_path), Path(output_prefix)
    payload = prepare(json.loads(source.read_text(encoding='utf-8-sig')))
    files = [Path(str(prefix)+stem+suffix) for stem in ('_accuracy', '_CE', '_common_IQ_pairs')
             for suffix in ('.png', '.pdf')]
    metadata_path = Path(str(prefix)+'_metadata.json')
    if any(path.exists() for path in files+[metadata_path]):
        raise FileExistsError('Existing figure prefix; choose a new prefix')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                                'axes.unicode_minus': False, 'pdf.fonttype': 42,
                                'savefig.bbox': 'tight'})
    import matplotlib.pyplot as plt
    outputs = fit_figure(payload, 'accuracy_percent', prefix, plt)
    outputs += fit_figure(payload, 'standard_CE', prefix, plt)
    outputs += pair_figure(payload, prefix, plt)
    metadata = dict(schema='pure_F_epochReplay20_frozen_source_fit_plot_v1', source_run=RUN,
        input=str(source.resolve()), figures=outputs, model_access=False, model_forward_packets=0,
        IQ_access=False, checkpoint_access=False, formal_data_access=False, remote_actions=0,
        online_training_values_used=False, computations='Saved source JSON metric/count/pair checks; '
        'accuracy x100; matched CE subtraction; no fitted quantities', plotted=payload)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                        allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(status='FIGURES_WRITTEN', files=outputs,
                          metadata=str(metadata_path), model_forward_packets=0)))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-prefix', required=True)
    args = parser.parse_args()
    main(args.input, args.output_prefix)
