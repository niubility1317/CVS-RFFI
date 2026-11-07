"""Source-only phase and gain alignment of fixed DURA phase-orbit outputs."""
import argparse
import os
from pathlib import Path
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import wait_for_idle


def align_paths(paths):
    """Align each complex path to the unrotated DURA output without clean IQ."""
    import torch
    if paths.ndim != 3 or paths.shape[1:] != (4, 256) or not torch.is_complex(paths):
        raise ValueError('Four complex 256-sample paths required')
    reference = paths[:, :1]
    inner = (reference*paths.conj()).sum(-1)
    power = paths.abs().square().sum(-1)
    if not torch.isfinite(inner).all() or (inner.abs() <= 1e-12).any() or (power <= 1e-12).any():
        raise ValueError('Finite nonzero path alignment statistics required')
    phase = inner/inner.abs()
    gain = inner/power
    return (paths*phase[..., None]).mean(1), (paths*gain[..., None]).mean(1), phase, gain


def evaluate_batch(dura, x, clean):
    import torch
    from experiments.phase1_dura_orthogonal.source_phase_orbit_probe import rotate
    with torch.no_grad():
        iq_paths = [rotate(dura(rotate(x, k)), (-k) % 4) for k in range(4)]
        paths = torch.stack([torch.complex(v[:, 0], v[:, 1]) for v in iq_paths], 1)
        phase, gain, phase_weights, gain_weights = align_paths(paths)
        target = torch.complex(clean[:, 0], clean[:, 1])
        energy = target.abs().square().mean(-1).clamp_min(1e-8)
        outputs = (paths[:, 0], paths.mean(1), phase, gain)
        errors = torch.stack([(v-target).abs().square().mean(-1)/energy for v in outputs], 1)
        changes = torch.stack([(v-paths[:, 0]).abs().square().mean(-1)/energy for v in outputs[1:]], 1)
        stats = torch.stack((phase_weights.real.mean(1), gain_weights.abs().mean(1)), 1)
        result = torch.cat((errors, changes, stats), 1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Finite source alignment metrics required')
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args()
    uuid=wait_for_idle(args.out,args.gpu,allow_shared=True)
    if 'torch' in sys.modules:
        raise RuntimeError('Select GPU before importing torch')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs

    if not torch.cuda.is_available():
        raise ValueError('Registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    ids,rx,scene,_,received,clean=source_inputs(args.root)
    dura=frozen_dura(args.root,'cuda:0')
    x=core.tensor(received[ids[:8]],'cuda:0')
    y=core.tensor(clean[ids[:8]],'cuda:0')
    smoke=evaluate_batch(dura,x,y)
    if smoke.shape != (8,9) or any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen real-checkpoint source smoke required')
    print('SOURCE_PHASE_ALIGNMENT_SMOKE_PASS',len(x),flush=True)
    chunks=[]
    for start in range(0,len(ids),16):
        selected=ids[start:start+16]
        chunks.append(evaluate_batch(dura,core.tensor(received[selected],'cuda:0'),
                                     core.tensor(clean[selected],'cuda:0')).cpu())
    values=torch.cat(chunks)
    groups={}
    for name,receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in core.VIEWS:
            mask=torch.as_tensor((np.isin(rx,receivers)&(scene==view)).tolist(),dtype=torch.bool)
            cell=values[mask]
            if len(cell)!=48:
                raise ValueError('48 records per source receiver/view group required')
            groups[name+'/'+view]=dict(count=48,mean_metrics=cell.mean(0).tolist(),
                beats_dura=[int((cell[:,i]<cell[:,0]).sum()) for i in (1,2,3)])
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json',dict(status='SOURCE_PHASE_ALIGNMENT_COMPLETE',commit=args.commit,
        records=len(ids),quarter_turns=[0,1,2,3],groups=groups,
        metric_order=['dura_nmse','naive_mean_nmse','phase_aligned_nmse','complex_gain_aligned_nmse',
                      'naive_change_nmse','phase_change_nmse','gain_change_nmse','mean_phase_weight_real','mean_gain_weight_abs'],
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        physical_gpu=args.gpu,physical_gpu_uuid=uuid,
        caveat='Alignments use unrotated DURA output only; clean IQ is used only for source metrics. No actual TX identity or ground performance is measured.'))
    print('SOURCE_PHASE_ALIGNMENT_COMPLETE',len(ids),flush=True)


if __name__=='__main__':
    main()
