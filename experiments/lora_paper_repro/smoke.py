"""Real author-checkpoint smoke on one enrollment record, without query data."""
import argparse
import importlib.util
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from experiments.lora_paper_repro.checkpoint import load_author_weights
from experiments.lora_paper_repro.predict import embeddings
from experiments.pamf_phase2_practical.vendor_extra.metric_models import LoRaExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--author-preprocessing', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('lora_author_preprocessing', args.author_preprocessing)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    with h5py.File(args.dataset, 'r') as data:
        if int(data['label'][0, 0]) != 31:
            raise ValueError('first real enrollment record does not belong to device 31')
        raw = data['data'][0:1]
    iq = author.LoadDataset()._convert_to_complex(raw)
    features = author.ChannelIndSpectrogram().channel_ind_spectrogram(iq).astype(np.float32)
    model = LoRaExtractor(height=102, width=62).to(args.device)
    load_author_weights(model, args.checkpoint)
    value = embeddings(model, features, device=args.device)
    if value.shape != (1, 512):
        raise ValueError('unexpected author-checkpoint embedding shape')
    with Path(args.output).open('x', encoding='utf8') as out:
        json.dump({'status': 'PASS', 'real_checkpoint': True, 'enrollment_records': 1,
                   'query_records': 0, 'device': args.device}, out, indent=2)


if __name__ == '__main__':
    main()
