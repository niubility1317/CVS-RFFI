"""Lossless final-R17 export: same FP32 model and graph, no optimizer state."""
import argparse
import json
from pathlib import Path

import torch

from experiments.pamf_unrolled_restore.runtime import load_checkpoint


def export_inference(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Source checkpoint must remain untouched')
    if destination.exists():
        raise FileExistsError(destination)
    model, original = load_checkpoint(source, 'cpu')
    if (original['candidate'] != 'unroll_unshared' or original['epoch'] != 20
            or original['updates'] != 4940):
        raise ValueError('Original final E20/4940 R17 unshared baseline required')
    exported = dict(original, optimizer=None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(exported, stream)
    restored, value = load_checkpoint(destination, 'cpu')
    if value['optimizer'] is not None:
        raise ValueError('Inference artifact must omit optimizer state')
    for key in original:
        if key not in ('model', 'optimizer') and original[key] != value[key]:
            raise ValueError('Original provenance and configuration must remain identical')
    before, after = model.state_dict(), restored.state_dict()
    if before.keys() != after.keys() or any(
            before[key].dtype != after[key].dtype or not torch.equal(before[key], after[key])
            for key in before):
        raise ValueError('All model tensors must remain exactly equal')
    return dict(status='VERIFIED_LOSSLESS_EXPORT', source_bytes=source.stat().st_size,
                inference_bytes=destination.stat().st_size,
                model_parameters=sum(p.numel() for p in restored.parameters()),
                model_tensors_exact=True, original_R17_graph=True,
                optimizer_removed=True, new_training=False,
                source_IQ_reads=0, validation_IQ_reads=0, target_IQ_reads=0,
                new_target_performance_evidence=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(export_inference(args.source, args.destination)))


if __name__ == '__main__':
    main()
