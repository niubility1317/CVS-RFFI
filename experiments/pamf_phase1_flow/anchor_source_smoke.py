"""Actual Phase1 DURA + Tri-SIG-TIME source-only anchor-flow smoke."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import experiment as core
from .anchor_flow import FixedAnchoredFlow,source_objective

RUN='20261101-diagnostic-phase1-dura-anchor-flow-smoke-s2026092701-r01'
DURA='runs/20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt'


def frozen_dura(root,device):
    import experiments
    from importlib import import_module
    release=root/'releases/pamf_stage2_inference_20261021_r01'
    if (release/'release_commit.txt').read_text(encoding='ascii').strip()!='47c264286ca689f060a3f36aafb763fe209a6f06':
        raise ValueError('Exact original Phase1 DURA release required')
    package=str(release/'experiments')
    if package not in experiments.__path__:experiments.__path__.append(package)
    runtime=import_module('experiments.pamf_r65_four_route.runtime')
    if Path(runtime.__file__).resolve()!=(release/'experiments/pamf_r65_four_route/runtime.py').resolve():
        raise ValueError('Original DURA loader path differs')
    model,p=runtime.load_checkpoint(root/DURA,device)
    if (p['arm'],p['epoch'],p['updates'],p['data_contract'],p['training_F_access'],p['training_TX_labels_read'],p['target_training']) != ('stage2',20,4940,'original7/63/30',False,0,False):
        raise ValueError('Exact Phase1 source-only DURA checkpoint required')
    if sum(v.numel() for v in model.parameters())!=266951:raise ValueError('Exact original DURA architecture required')
    return model.eval().requires_grad_(False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);f,contract=core.teacher(a.root,'cpu');cache,clean,roles=core.data(a.root,contract)
    with np.load(cache/'epochs/01/groups.npz',allow_pickle=False) as groups:scene=groups['scene']
    if scene.shape!=(core.COUNT,):raise ValueError('Original source scene metadata required')
    take=np.flatnonzero((roles=='L_s')&(scene!='clean'))[:2]
    if len(take)!=2:raise ValueError('Two original corrupted L_s inputs required')
    raw=np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    x=core.tensor(raw[take],'cpu');target=core.tensor(clean[take],'cpu')
    dura=frozen_dura(a.root,'cpu');dura_before={k:v.clone() for k,v in dura.state_dict().items()};f_before={k:v.clone() for k,v in f.state_dict().items()}
    with torch.no_grad():anchor=dura(x)
    if anchor.shape!=x.shape or not torch.isfinite(anchor).all():raise ValueError('Finite original DURA source output required')
    field=core.make_model();initial=FixedAnchoredFlow(field)
    with torch.no_grad():
        original=initial(x,anchor);torch.testing.assert_close(original,anchor,rtol=1e-6,atol=3e-7)
    noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(2026110112))
    loss,terms=source_objective(field,f,x,anchor,target,noise)
    gradients={key:torch.autograd.grad(terms[key].sum(),field.output.weight,retain_graph=True)[0] for key in ('iq','semantic')}
    if any(not torch.isfinite(v).all() or not v.abs().sum()>0 for v in gradients.values()):raise ValueError('Finite nonzero source IQ and semantic gradients required')
    loss.mean().backward();torch.optim.AdamW(field.parameters(),lr=.0003).step()
    if any(v.grad is not None for v in f.parameters()) or any(v.grad is not None for v in dura.parameters()):raise ValueError('Pretrained networks must not acquire gradients')
    if any(not torch.equal(v,f_before[k]) for k,v in f.state_dict().items()) or any(not torch.equal(v,dura_before[k]) for k,v in dura.state_dict().items()):raise ValueError('Pretrained state changed')
    out=a.root/'runs'/RUN
    if out.exists():raise FileExistsError('Unique source smoke output required')
    out.mkdir();path=out/'field-smoke.pt'
    with path.open('xb') as stream:torch.save(dict(schema='phase1_anchor_smoke_v1',commit=a.commit,model=field.state_dict()),stream)
    loaded=core.make_model();payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['schema']!='phase1_anchor_smoke_v1' or payload['commit']!=a.commit:raise ValueError('Exact diagnostic checkpoint required')
    loaded.load_state_dict(payload['model'],strict=True)
    with torch.no_grad():
        actual=FixedAnchoredFlow(loaded)(x,anchor);expected=FixedAnchoredFlow(field)(x,anchor)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);scores=f.classify_features(f.features(actual))
    if not torch.isfinite(actual).all() or not torch.isfinite(scores).all() or scores.shape!=(2,6):raise ValueError('Finite actual two-model output required')
    core.write(out/'complete.json',dict(status='SOURCE_SMOKE_PASS',run=RUN,commit=a.commit,source_records=2,
        F_run=core.F_RUN,F_row=core.F_ROW,DURA_checkpoint=DURA,DURA_parameters=266951,F_frozen=True,DURA_frozen=True,
        initial_anchor_max_abs=float((original-anchor).abs().max()),
        IQ_gradient_norm=float(gradients['iq'].norm()),semantic_gradient_norm=float(gradients['semantic'].norm()),
        checkpoint_reload=True,smoke_optimizer_steps=1,TX_labels_read=0,V_reads=0,test_IQ_reads=0))
    print(json.dumps(dict(status='SOURCE_SMOKE_PASS',run=RUN),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
