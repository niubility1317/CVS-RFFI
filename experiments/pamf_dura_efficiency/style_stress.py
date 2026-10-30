"""Source-only stress of frozen noVar restoration under acquisition response shifts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_dura_direct_train import runner as training,no_variance,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values,action_from_arrays
from experiments.pamf_restoration.models import complex_iq,pair
from .acquisition_style import Style,sample_style,paired_style

RUN='20261030-diagnostic-DURA-paired-acquisition-stress-s2026092701-r01'
COMMIT='cf40edea90667d9645a724b1e8f333027cdf7869'
SEED=2026103014
RULE=('A single-stage same-budget paired-style training contrast may be preregistered only if BOTH cached views '
      'frozen noVar common-reference-power MSE rises>=10%, per-sample power relative error<=2e-6, '
      'and exact-A noise-replay error/common clean power<=1e-10. F clean-choice/style fidelity is reported, '
      'not a threshold or parameter-selection objective; no target or TX truth. Otherwise no training from this probe.')


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU output required')
    no_variance.configure();p=training.paths(args.root)
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,training_F_access=False,
        V_reads=0,target_reads=0,initialization_checkpoint_loads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed original noVar source-final required')
    model=training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=COMMIT).eval().requires_grad_(False)
    clean,roles=training.source_training(p);f_guided.source_ids(p,roles);indices=source_indices(roles)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before=[{key:v.clone() for key,v in item.state_dict().items()} for item in (model,reader)]
    generator=torch.Generator().manual_seed(SEED);styles=sample_style(512,generator)
    args.output.mkdir();training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0))
    smoke=reader(model(tensor(clean[indices[:2]],'cpu')))
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Actual source2Ls G/F smoke failed')
    print(json.dumps(dict(event='SOURCE_STYLE_STRESS_SMOKE_PASS',real_Ls=2,TX_labels_read=0,target_reads=0)),flush=True)
    summary={};decisions=[];started=time.perf_counter()
    for epoch in (1,20):
        folder=p['cache']/'epochs'/f'{epoch:02d}'
        arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients')}
        with np.load(folder/'groups.npz',allow_pickle=False) as groups:labels={key:groups[key][indices] for key in ('scene','state','lock')}
        accumulated={}
        for offset in range(0,512,32):
            take=indices[offset:offset+32];x=tensor(clean[take],'cpu');y=tensor(arrays['received'][take],'cpu')
            action=action_from_arrays(tensor(arrays['kernel'][take],'cpu'),tensor(arrays['coefficients'][take],'cpu'))
            style=Style(styles.taps[offset:offset+32],styles.image[offset:offset+32])
            shifted,reference=paired_style(y,x,action,style)
            power=x.square().mean((1,2)).clamp_min(1e-4)
            residual=y-pair(action(complex_iq(x)))
            shifted_residual=shifted-pair(action(complex_iq(reference)))
            original=model(y);restored=model(shifted)
            fclean=reader.features(x);fstyle=reader.features(reference)
            first=reader.features(original);second=reader.features(restored)
            choose=reader.classify_features(fclean).argmax(1);styled_choice=reader.classify_features(fstyle).argmax(1)
            values=dict(original_common_nmse=(original-x).square().mean((1,2))/power,
                styled_common_nmse=(restored-reference).square().mean((1,2))/power,
                power_relative_error=(reference.square().mean((1,2))-power).abs()/power,
                noise_replay_error=(shifted_residual-residual).square().mean((1,2))/power,
                style_delta_energy=(reference-x).square().mean((1,2))/power,
                clean_style_F_choice_agreement=(choose==styled_choice).float(),
                original_F_choice_agreement=(reader.classify_features(first).argmax(1)==choose).float(),
                styled_F_choice_agreement=(reader.classify_features(second).argmax(1)==styled_choice).float(),
                original_Fcos_error=1-F.cosine_similarity(first,fclean,dim=1,eps=1e-8),
                styled_Fcos_error=1-F.cosine_similarity(second,fstyle,dim=1,eps=1e-8))
            for key,value in values.items():
                if value.shape!=(len(take),) or not torch.isfinite(value).all():raise FloatingPointError('Nonfinite source style stress')
                accumulated.setdefault(key,[]).append(numpy_values(value))
            print(json.dumps(dict(event='SOURCE_STYLE_STRESS_BATCH',epoch=epoch,count=offset+32,seconds=time.perf_counter()-started)),flush=True)
        values={key:np.concatenate(v) for key,v in accumulated.items()}
        np.savez(args.output/f'epoch{epoch:02d}.npz',source_positions=indices,roles=roles[indices],**labels,**values)
        masks={'all':np.ones(512,dtype=bool)}
        for key,group in dict(role=roles[indices],**labels).items():
            for label in np.unique(group):masks[f'{key}:{label}']=group==label
        summary[str(epoch)]={key:dict(count=int(mask.sum()),metrics={name:dict(mean=float(v[mask].mean()),median=float(np.median(v[mask])),max=float(v[mask].max())) for name,v in values.items()}) for key,mask in masks.items()}
        allmetrics=summary[str(epoch)]['all']['metrics'];ratio=allmetrics['styled_common_nmse']['mean']/allmetrics['original_common_nmse']['mean']
        decisions.append(dict(epoch=epoch,common_MSE_ratio=ratio,power_error_max=allmetrics['power_relative_error']['max'],
            noise_replay_max=allmetrics['noise_replay_error']['max'],passed=ratio>=1.1 and allmetrics['power_relative_error']['max']<=2e-6 and allmetrics['noise_replay_error']['max']<=1e-10))
    for item,state in zip((model,reader),before):
        if any(not torch.equal(v,state[key]) for key,v in item.state_dict().items()) or any(v.grad is not None for v in item.parameters()):raise ValueError('Source stress changed model or accumulated gradient')
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,decisions=decisions,
        training_contrast_eligible=all(v['passed'] for v in decisions),source_records=512,source_L=128,source_U=384,epochs=[1,20],
        style=dict(seed=SEED,tap_real_imag_std=.03,tap_L1_max=.08,image_real_imag_std=.005,image_abs_max=.015,preserve_RMS=True,
            pairing='y+A(Sx-x),same actual A/noise',training_if_eligible='single style draw p.5 per visit,one Gforward;no antithetic three-view loss'),
        F_unchanged=True,G_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        source_rule=RULE,scope='Synthetic source acquisition response robustness;not real RX/day causal proof or TX identity truth;F metrics descriptive/not selection;no trained version or targetprediction'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,
        training_contrast_eligible=all(v['passed'] for v in decisions),goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
