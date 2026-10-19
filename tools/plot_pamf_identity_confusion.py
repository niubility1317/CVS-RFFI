"""Plot existing source-only R16 aggregates; never load IQ, labels or predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


TX = ('14-10', '14-7', '20-15', '20-19', '6-15', '8-20')
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def extract(summary):
    assert summary['status'] == 'ANALYZED'
    assert summary['method_commit'] == '50587946b6c83b7b32500696af7fa7b9086484b5'
    sources = {
        'Native MAP': summary['source_native_reference'],
        'U-Net + IQ/FFT loss': summary['paired_R15_source']['unet_control'],
        'U-Net + IQ-only loss': summary['source']['unet_control'],
    }
    result = {}
    expected_counts = None
    for name, source in sources.items():
        assert source['role'] == 'source' and source['status'] == 'ANALYZED'
        assert source['count'] == 1800 and source['F_seed'] == 2026092701
        assert source['all_predictions_before_truth'] and source['original_default_head']
        assert source['target_access'] is False
        assert source['training_TX_labels_read'] == 0 and not source['training_F_access']
        assert source['head_fit_updates'] == 0
        row = {}
        for view in VIEWS:
            selected = [r for r in source['records'] if r['view'] == view
                        and r['dimension'] == 'overall' and r['readout'] == 'restored']
            assert len(selected) == 1
            record = selected[0]
            confusion = np.asarray(record['confusion'], dtype=np.int64)
            assert confusion.shape == (6, 6) and (confusion >= 0).all()
            counts = confusion.sum(1)
            assert (counts > 0).all() and confusion.sum() == record['count'] == 1800
            if expected_counts is None:
                expected_counts = counts
            assert np.array_equal(counts, expected_counts)
            accuracy = float(confusion.trace() / confusion.sum() * 100)
            assert abs(accuracy - record['accuracy']) < 1e-9
            diagnostics = source['views'][view]['restored']
            feature = diagnostics['paired_clean_vs_other_TX']
            assert feature['reference_pool_count'] == 512
            assert feature['inference_or_selection_use'] is False
            assert feature['full_global_nearest_other_TX'] is False
            row[view] = dict(count=1800, accuracy_percent=accuracy,
                TX_counts=counts.tolist(), confusion=confusion.tolist(),
                TX_accuracy_percent=(confusion.diagonal() / counts * 100).tolist(),
                IQ_NMSE=diagnostics['waveform']['iq_nmse'],
                paired_clean_feature_cosine_distance=diagnostics['paired_clean_feature_cosine_distance'],
                paired_feature_margin_mean=feature['mean'],
                own_clean_pair_above_512_pool_other_TX_percent=feature['own_pair_higher_percent'])
        result[name] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--summary', type=Path, default=root / 'automation_reports/CV-SincNet/'
        '20261020-phase1-restoration-iqonly-selfsup-s2026092701-m2-r16/summary.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory; existing results are never overwritten.')
    args = parser.parse_args()
    data = extract(json.loads(args.summary.read_text(encoding='utf-8-sig')))
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1), sharey=True)
    fig.subplots_adjust(left=.07, right=.94, bottom=.20, top=.81, wspace=.16)
    for ax, (name, scenes) in zip(axes, data.items()):
        row = scenes['practical_low_urban']
        confusion = np.asarray(row['confusion'])
        percent = confusion / confusion.sum(1, keepdims=True) * 100
        im = ax.imshow(percent, cmap='Blues', vmin=0, vmax=100)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f'{confusion[i,j]}\n{percent[i,j]:.1f}%',
                        ha='center', va='center', fontsize=7.7,
                        color='white' if percent[i,j] > 55 else '#172b4d')
        # Both directions of the source-identified confusion; no inference routing.
        for i, j in ((1, 3), (3, 1)):
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                   edgecolor='#dd6b20', linewidth=2))
        ax.set_xticks(range(6), TX, rotation=40, ha='right')
        ax.set_yticks(range(6), TX)
        ax.set_xlabel('Predicted TX')
        ax.set_title(f'{name}\nAccuracy {row["accuracy_percent"]:.2f}%', fontsize=11, pad=12)
        ax.tick_params(length=0)
    axes[0].set_ylabel('True TX')
    cax = fig.add_axes((.955, .24, .012, .52))
    fig.colorbar(im, cax=cax, label='Within-TX percentage')
    fig.suptitle('Average improvement can coexist with transmitter-specific damage',
                 x=.49, y=.97, fontsize=15)
    fig.text(.07, .91, 'Source V only | low urban | same 1,800 physical records | frozen original F50 / seed 2026092701',
             fontsize=10, color='#334155')
    fig.text(.07, .035,
             'Cell: count and row percentage. Orange: TX 14-7 / 20-19 confusion. '
             'One model seed; no new predictions or target access.\n'
             'IQ-only vs IQ/FFT is the matched R15/R16 loss ablation. Native MAP is a historical reference.',
             fontsize=9, color='#334155')
    for suffix in ('png', 'svg'):
        figure_path = args.output / f'lowurban_confusions.{suffix}'
        fig.savefig(figure_path, dpi=220,
                    metadata={'Creator': 'CVS source-aggregate diagnostic'})
        if suffix == 'svg':
            figure_path.write_text('\n'.join(line.rstrip() for line in
                figure_path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)
    try:
        source_reference = str(args.summary.resolve().relative_to(root)).replace('\\', '/')
    except ValueError:
        source_reference = args.summary.name
    payload = dict(scope='Existing fixed source-only aggregates; no training or new predictions',
        data_source=source_reference,
        TX_order=list(TX), model_seed=2026092701, training_updates=0,
        target_reads=0, selected_after_diagnostic=False, methods=data,
        limits=['Confusion-count differences are not paired individual prediction transitions.',
                'Feature margins use a fixed 512-position pool, not global retrieval.',
                'Accuracy or feature similarity does not prove each generated detail is true.',
                'Single-seed source evidence cannot establish the formal 70% / 5pp target.'])
    (args.output / 'metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n',
                                            encoding='utf-8')
    print(json.dumps(dict(status='SOURCE_AGGREGATES_VALIDATED_AND_PLOTTED',
        methods=list(data), scenes=len(VIEWS), records_per_scene=1800,
        output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
