"""Single controlled source F-fidelity variant; unchanged MSAT/data/single30 budget."""
import argparse
import copy
import json
import subprocess
from pathlib import Path
import torch
from . import multiview_trial as trial
from .source_pair import read
from .main_split_eval import valid_identity

RUN='20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r02'
MODULE='experiments.pamf_phase3_restoration.variance_trial'
ARM='MSAT_VarF'
GEOMETRY_RUN='20261101-diagnostic-Restore-learned-prior-physics-s2026092701-r01'
GEOMETRY_COMMIT='f57c54cad7491e3e8a36f7ea51a9caa13a3a0ab9'
TEMPLATE=trial.configuration('MSAT_M20')


def recovery_gpu_choice(runner=None):
    """Query admitted healthy cards separately; GPU7 is unavailable after r01."""
    if runner is None:runner=subprocess.run
    result=[]
    for gpu in (4,6):
        p=runner(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
        q=runner(['nvidia-smi','-i',str(gpu),'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
        if p.returncode or q.returncode:
            print('RECOVERY_GPU_UNAVAILABLE',json.dumps(dict(GPU=gpu,gpu_exit=p.returncode,compute_exit=q.returncode,
                detail=(p.stderr+q.stderr)[-800:])),flush=True)
            continue
        fields=[x.strip() for x in p.stdout.strip().split(',')]
        if len(fields)!=3 or int(fields[0])!=gpu or not fields[1].startswith('GPU-'):
            raise ValueError('Unrecognized admitted GPU identity query')
        uuid=fields[1];pids=set()
        for line in q.stdout.splitlines():
            if not line.strip():continue
            identity,pid=[x.strip() for x in line.split(',')]
            if identity!=uuid:raise ValueError('Scoped compute query returned another GPU')
            pids.add(int(pid))
        free=int(fields[2])
        if free>=12000 and len(pids)<=2:
            result.append(dict(GPU=gpu,CUDA_ID=uuid,free_memory_MB=free,existing_compute_PIDs=len(pids)))
    return sorted(result,key=lambda x:(x['existing_compute_PIDs'],-x['free_memory_MB'],x['GPU']))


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed MSAT variance-F variant allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,objective='IQNMSE+.05STFT+.1(F0cos+source-variance-standardized-featureMSE)/perrecordIQgradientcap.5',
        feature_geometry_source=GEOMETRY_RUN,feature_geometry_code=GEOMETRY_COMMIT,
        feature_geometry_count=63000,feature_variance_floor='max(var,.01mean(var),1e-10)',
        geometry_uses_G_outputs=False,geometry_uses_TX_labels=False)
    return c


def load_scale(root):
    folder=root/'runs'/GEOMETRY_RUN;r=read(folder/'results.json');g=read(folder/'feature_geometry.json')
    valid_identity(r['F0_identity'])
    required=dict(status='LEARNED_PRIOR_PHYSICS_COMPLETE',run_id=GEOMETRY_RUN,code_commit=GEOMETRY_COMMIT,
        source_feature_moment_rows=63000,G_training_updates=0,optimizer_updates=0,model_state_updates=0,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0)
    if any(r.get(k)!=v for k,v in required.items()) or (g.get('count'),g.get('source_only'),g.get('TX_labels_read'))!=(63000,True,0):
        raise ValueError('Frozen original63000 source-only F0 feature geometry required')
    scale=torch.tensor(g['scale'],dtype=torch.float32)
    variance=torch.tensor(g['variance'],dtype=torch.float64)
    if scale.shape!=(160,) or variance.shape!=(160,) or not torch.isfinite(scale).all() or not (scale>0).all():
        raise ValueError('Finite positive source fused160 scales required')
    floor=(.01*variance.mean()).clamp_min(1e-10)
    torch.testing.assert_close(scale.double(),variance.clamp_min(floor),rtol=1e-6,atol=1e-10)
    return scale


def variance_feature_guide(reader,restored,clean,target_features,scale):
    if target_features.shape!=(len(restored),160) or scale.shape!=(160,):
        raise ValueError('Matched source fused160 targets and scales required')
    actual=reader.features(restored);target=target_features.detach()
    cosine=1-torch.nn.functional.cosine_similarity(actual,target,dim=1,eps=1e-8)
    standardized=((actual-target).square()/scale).mean(1)
    raw=.1*(cosine+standardized)
    gradient=torch.autograd.grad(raw.sum(),restored,retain_graph=True)[0]
    norm=gradient.detach().flatten(1).norm(dim=1)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    IQ_norm=2*(restored.detach()-clean.detach()).flatten(1).norm(dim=1)/(512*energy)
    weight=(.5*IQ_norm/norm.clamp_min(1e-12)).clamp(max=1).detach()
    return weight*raw,dict(feature_error=cosine,feature_standardized_error=standardized,
        weight=weight,input_gradient_norm=norm,IQ_gradient_norm=IQ_norm)


def install(scale):
    """Bind one explicit recipe into the already tested, fresh-process trial runner."""
    by_device={}
    def guided(reader,restored,clean,target):
        if restored.device not in by_device:by_device[restored.device]=scale.to(restored.device)
        return variance_feature_guide(reader,restored,clean,target,by_device[restored.device])
    trial.RUN=RUN;trial.MODULE=MODULE;trial.ARMS=(ARM,)
    trial.configuration=configuration;trial.feature_guide=guided
    trial.gpu_choice=recovery_gpu_choice


def main():
    # Parse only the common root; the reused runner owns the full CLI/stage checks.
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True)
    args,_=p.parse_known_args();install(load_scale(args.root));trial.main()


if __name__=='__main__':main()
