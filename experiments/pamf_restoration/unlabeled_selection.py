"""Pure-Python, TX/F-free selection on source reconstruction metrics only.

The input schema rejects label-derived recognition/feature fields. The main
seven-network experiment's labeled-V comparison remains a separate diagnostic.
"""
import argparse
import json
import math
from pathlib import Path

CANDIDATES = ('rescnn', 'unet', 'dual_domain', 'diffusion', 'flow', 'unrolled', 'transformer')
POLICY = dict(version='restoration_label_free_selection_v1',
              clean_nmse_max=.02, primary='minimum mean raw IQ NMSE across six source practical views',
              constraints='clean->clean NMSE<=.02; no TX labels/F scores/features/stochastic classification',
              tie='fixed original candidate order',
              fallback='If no clean-qualified model, minimum clean NMSE, then minimum practical NMSE; negative confirmation only')


def select(packet):
    if set(packet) != {'source_run', 'code_commit', 'rows'}:
        raise ValueError('Only waveform source-selection input fields are accepted')
    rows = packet['rows']
    if len(rows) != len(CANDIDATES) or {row.get('candidate') for row in rows} != set(CANDIDATES):
        raise ValueError('All seven fixed source candidates required')
    for row in rows:
        if set(row) != {'candidate', 'clean_nmse', 'mean_practical_nmse', 'identity_mean_nmse'}:
            raise ValueError('TX/F/recognition fields are forbidden in label-free selection')
        if any(not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0
               for key in ('clean_nmse', 'mean_practical_nmse', 'identity_mean_nmse')):
            raise ValueError('Finite nonnegative reconstruction metrics required')
    qualified = [row for row in rows if row['clean_nmse'] <= POLICY['clean_nmse_max']]
    if qualified:
        chosen = min(qualified, key=lambda row: (row['mean_practical_nmse'], CANDIDATES.index(row['candidate'])))
    else:
        chosen = min(rows, key=lambda row: (row['clean_nmse'], row['mean_practical_nmse'], CANDIDATES.index(row['candidate'])))
    return dict(status='LABEL_FREE_SOURCE_SELECTION_FROZEN', candidate=chosen['candidate'],
                policy=POLICY, source_run=packet['source_run'], source_code_commit=packet['code_commit'],
                source_waveform_metrics=rows, clean_qualified=bool(qualified),
                reconstruction_improves=chosen['mean_practical_nmse'] < chosen['identity_mean_nmse'],
                training_TX_label_access=False, selection_TX_label_access=False,
                selection_F_prediction_access=False, target_feedback=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = select(json.loads(args.input.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
