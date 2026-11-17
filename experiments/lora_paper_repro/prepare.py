"""Pack author's residential enrollment and unlabeled identification records."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--author-preprocessing', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('lora_author_preprocessing', args.author_preprocessing)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader, converter = module.LoadDataset(), module.ChannelIndSpectrogram()
    devices = np.arange(30, 40, dtype=int)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for name, packet_range in [('enrollment', np.arange(0, 100)), ('query', np.arange(100, 200))]:
        signals, labels = loader.load_iq_samples(args.dataset, devices, packet_range)
        labels = labels.reshape(-1)
        if len(signals) != 1000 or not np.array_equal(np.unique(labels, return_counts=True)[0], devices):
            raise ValueError('author dataset does not match the fixed ten-device experiment')
        if not np.all(np.unique(labels, return_counts=True)[1] == 100):
            raise ValueError('expected 100 packets per device')
        features = converter.channel_ind_spectrogram(signals).astype(np.float32)
        if features.shape != (1000, 102, 62, 1) or not np.isfinite(features).all():
            raise ValueError('author preprocessing produced invalid spectrograms')
        np.save(root / f'{name}_input.npy', features, allow_pickle=False)
        label_name = 'enrollment_labels.npy' if name == 'enrollment' else 'truth_side_labels.npy'
        np.save(root / label_name, labels, allow_pickle=False)
    (root / 'preparation.json').write_text(json.dumps({'devices_one_based': [31, 40],
        'enrollment_packets_per_device': [0, 99], 'query_packets_per_device': [100, 199],
        'paper_exact_packet_indices': 'not independently specified in paper',
        'source_preprocessing': str(Path(args.author_preprocessing).resolve())}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
