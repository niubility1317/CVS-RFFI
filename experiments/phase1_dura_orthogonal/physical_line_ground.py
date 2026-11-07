"""Fixed physical-line DURA combination for Phase1 frozen seven-view evaluation."""
import argparse
from pathlib import Path
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal import ground
from experiments.phase1_dura_orthogonal.physical_line_source_probe import bounded_step, SOURCE, SOURCE_COMMIT

MODE = 'physical_line_v1'


def apply_physical_line(received, anchor, proposal, action):
    def c(v):
        return torch.complex(v[:, 0], v[:, 1])
    y, a, p = map(c, (received, anchor, proposal))
    delta = p - a
    alpha = bounded_step(y - action(a), action(delta))
    restored = a + alpha[:, None] * delta
    return torch.stack((restored.real, restored.imag), dim=1)


@torch.no_grad()
def decisions(x, model, dura, reader, *, inference_mode=MODE):
    if inference_mode != MODE:
        raise ValueError('Only the fixed physical-line inference mode is supported')
    context = dura.estimate_context(x)
    stages, _, action, _ = dura.stage_outputs(x, context)
    anchor = stages[-1]
    observed = torch.complex(x[:, 0], x[:, 1])
    a = torch.complex(anchor[:, 0], anchor[:, 1])
    direction = action.adjoint(observed - action(a))
    direction = torch.stack((direction.real, direction.imag), dim=1)
    proposal = model(x, anchor, direction, context[2])
    restored = apply_physical_line(x, anchor, proposal, action)
    outputs = []
    for signal in (x, anchor, restored):
        logits = reader.classify_features(reader.features(signal))
        if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
            raise ValueError('Finite frozen reader outputs required')
        outputs.append(logits.argmax(1))
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('smoke', 'predict', 'score'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ground.decisions = decisions
    fixed = dict(source_commit=SOURCE_COMMIT, inference_mode=MODE)
    if args.mode == 'smoke':
        ground.smoke(args.root, args.out, SOURCE, args.commit, **fixed)
    elif args.mode == 'predict':
        ground.predict(args.root, args.out, SOURCE, args.commit, **fixed)
    else:
        core.score(args.root, args.out, args.commit)
        ground.paired_score(args.root, args.out, args.commit)


if __name__ == '__main__':
    main()
