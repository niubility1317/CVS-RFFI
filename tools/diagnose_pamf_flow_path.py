"""Same-source-IQ diagnostic; teacher-path points contain clean information."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch


def distribution(values):
    values=values.detach().double()
    if values.ndim!=1 or not len(values) or not torch.isfinite(values).all():raise ValueError('Finite packet metrics required')
    return dict(mean=float(values.mean()),median=float(values.median()),p90=float(torch.quantile(values,.9)))


@torch.no_grad()
def diagnose(model,clean,received,noise,*,sigma,steps,scale):
    if (model.training or any(p.requires_grad for p in model.parameters()) or clean.ndim!=3 or clean.shape[1]!=2
            or received.shape!=clean.shape or noise.shape!=clean.shape or type(steps)is not int or steps<=0
            or not 0<=sigma<float('inf') or not 0<scale<float('inf') or not all(torch.isfinite(v).all() for v in (clean,received,noise))):
        raise ValueError('Frozen eval model and aligned finite source IQ/explicit noise required')
    base,_,_,condition=model.context(received);endpoint=(clean-base)/scale;start=sigma*noise
    power=clean.double().square().mean((1,2)).clamp_min(1e-4)
    def nmse(value):return distribution((value.double()-clean.double()).square().mean((1,2))/power)
    teacher=[]
    for t in (0.,.25,.5,.75):
        time=received.new_full((len(received),),t);point=(1-t)*start+t*endpoint
        velocity=model.prior.field(point,condition,time)
        estimate=base+scale*(point+(1-t)*velocity)
        objective=scale**2*(velocity.double()-(endpoint-start).double()).square().mean((1,2))/power
        teacher.append(dict(time=t,endpoint_nmse=nmse(estimate),velocity_nmse=distribution(objective),contains_clean_information=t>0))
    trajectory=[];field=model.prior.field
    def observe(point,context,time):
        # Clean is used only for recording errors; the original field receives its original arguments unchanged.
        velocity=field(point,context,time);next_point=point+velocity/steps
        trajectory.append(dict(step=len(trajectory)+1,time=float(time[0]),prior_nmse=nmse(base+scale*next_point)))
        return velocity
    with patch.object(model.prior,'field',side_effect=observe):output=model(received,noise)
    if len(trajectory)!=steps:raise ValueError('Actual complete sampler call count differs')
    return dict(count=len(clean),sigma0=sigma,steps=steps,coarse_nmse=nmse(base),teacher_path=teacher,
        free_running_steps=trajectory,free_running_prior_nmse=trajectory[-1]['prior_nmse'],full_restored_nmse=nmse(output),
        full_restored_mean=float(output.mean()),teacher_path_uses_clean_endpoint=True,free_running_uses_clean=False,
        parameter_updates=0,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,target_reads=0,
        scope='Same held-out source packets and explicit epsilon;teacher-path errors with t>0 are not deployable predictions;not proof of a unique causal failure')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('inputs','weights_root','output'):parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    from experiments.pamf_flow_start_restore.runtime import load_checkpoint
    from experiments.pamf_flow_start_restore import design as d
    from experiments.pamf_action_restore.models import SCALE
    from experiments.pamf_restoration.runner import array_tensor
    with np.load(args.inputs,allow_pickle=False) as archive:
        if set(archive.files)!={'clean','received','physical_ids'}:raise ValueError('Only declared source IQ and physical IDs allowed')
        clean=array_tensor(archive['clean'],'cpu');received=array_tensor(archive['received'],'cpu');ids=archive['physical_ids'].copy()
    if clean.shape!=(128,2,256) or received.shape!=clean.shape or ids.shape!=(128,):raise ValueError('Fixed first128 held-out source panel required')
    generator=torch.Generator(device='cpu').manual_seed(202610202105)
    noise=torch.randn(clean.shape,generator=generator,dtype=clean.dtype);rows={}
    for candidate in d.ROWS:
        model,payload=load_checkpoint(args.weights_root/candidate/'final.pt','cpu');model.requires_grad_(False)
        if payload['candidate']!=candidate or payload['epoch']!=20 or payload['updates']!=4940:raise ValueError('Original frozen final R20 checkpoint required')
        rows[candidate]=diagnose(model,clean,received,noise,sigma=d.SIGMA[candidate],steps=d.SAMPLING_STEPS,scale=SCALE)
    result=dict(status='ANALYZED',source_only=True,view='practical_low_urban',count=128,source_panel_count=1800,
        CPU_epsilon_seed=202610202105,new_source_IQ_restorations=256,F_prediction_calls=0,TX_labels_read=0,target_reads=0,rows=rows,
        note='Additional descriptive held-out-source diagnostic;no selection/update/new target;CPU epsilon differs from original CUDA primary draw and is fixed for both rows.')
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()
