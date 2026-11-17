"""Run fixed author Extractor 1 and 15-NN without reading query labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.lora_paper_repro.checkpoint import load_author_weights, load_project_weights
from experiments.lora_paper_repro.knn import classify
from experiments.pamf_phase2_practical.vendor_extra.metric_models import LoRaExtractor


def embeddings(model, inputs, *, device, batch_size=64):
    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (102, 62, 1) or not len(inputs):
        raise ValueError('expected nonempty [N,102,62,1] author spectrograms')
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = torch.tensor(np.asarray(inputs[start:start + batch_size]), device=device, dtype=torch.float32)
            z = model(x.permute(0, 3, 1, 2).contiguous())
            if z.shape != (len(x), 512) or not bool(torch.isfinite(z).all()):
                raise ValueError('invalid or nonfinite author-model embedding')
            result.append(z.cpu().numpy())
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-format', choices=['author-h5', 'project-pt'], default='author-h5')
    parser.add_argument('--enrollment-input', required=True)
    parser.add_argument('--enrollment-labels', required=True)
    parser.add_argument('--query-input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    model = LoRaExtractor(height=102, width=62).to(args.device)
    if args.checkpoint_format == 'author-h5':
        load_author_weights(model, args.checkpoint)
    else:
        load_project_weights(model, args.checkpoint, device=args.device)
    enrollment = np.load(args.enrollment_input, mmap_mode='r', allow_pickle=False)
    enrollment_labels = np.load(args.enrollment_labels, allow_pickle=False)
    query = np.load(args.query_input, mmap_mode='r', allow_pickle=False)
    reference = embeddings(model, enrollment, device=args.device)
    features = embeddings(model, query, device=args.device)
    predictions = classify(reference, enrollment_labels, features)
    with (root / 'predictions.npy').open('xb') as file:
        np.save(file, predictions, allow_pickle=False)
    (root / 'predictions_complete.json').write_text(json.dumps({'status': 'PREDICTIONS_FIXED',
        'count': len(predictions), 'neighbors': 15}), encoding='utf8')


if __name__ == '__main__':
    main()
