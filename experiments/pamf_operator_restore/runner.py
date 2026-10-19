"""Source-only R3 feasibility, not a deployable oracle or identity result."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from leo_practical.channel import Config,receiver_for_session,stable_seed
from experiments.pamf_residual_restore.runtime import load_coarse
from experiments.pamf_restoration.data import IQSource,SCENES,write_json
from experiments.pamf_restoration.runner import array_tensor,tensor_array,code_commit
from experiments.pamf_restoration.dispatcher import capacity
from .operators import real_pair,complex_pair,oracle_snapshot,GaussianSystem,estimate_action

ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-restoration-operator-source-s2026092701-r06'
MODEL_SEED=2026092701
DATA_SEED=2026101909
AUGMENTATION_SEED=2026101908
RECEIVER_SEED=2026101910
SCALE=0.1169459182403555
NAMESPACE='operator_feasibility/sourceV/draw0'
CAPACITY_LOCK='receiver_residual_capacity4.lock'
LOCAL_TAPS=(0,*range(20,29))
WIDE_TAPS=tuple(range(49))
METHODS=('received','coarse','oracle_white_prior','coarse_oracle_full','coarse_oracle_isotropic',
         'coarse_estimated10','coarse_estimated49')


def configuration():
    return dict(schema='source_operator_feasibility_v1',run=RUN,model_seed=MODEL_SEED,data_seed=DATA_SEED,
        augmentation_seed=AUGMENTATION_SEED,receiver_seed=RECEIVER_SEED,source_per_RX_day=4,
        count=60,views=['clean',*SCENES],methods=list(METHODS),training_updates=0,
        TX_labels_read=0,F_outputs_read=0,target_reads=0,physical_GPU=5,CPU_threads=2,
        prior_RMS=SCALE,prior_precision=1/SCALE**2,white_prior_precision=2.,
        scale_origin='R03 once-only 8192 sourceL/U scale,0V/TX/target/optimizer updates',
        estimation='Transient per-record widely-linear FIR LS using only received and frozen coarse; no true channel/scene/clean',
        fitting_ridge_trace_fraction=.01,estimated_noise_floor_fraction=1e-4,
        local_taps=list(LOCAL_TAPS),wide_taps=list(WIDE_TAPS),
        scope='Source waveform/operator feasibility only. Oracle is conditional on true action/AGC/lock and is not deployable.',
        scientific_threshold=dict(oracle_mean_six_scene_NMSE_gain_vs_coarse_min=.10,
            estimated_mean_six_scene_NMSE_gain_vs_coarse_min=.05,estimated_clean_NMSE_max=.02),
        threshold_use='Determine which source-only R3 problem to investigate,not identity qualification or automatic target access')


def waveform(estimate,clean):
    denominator=max(float(np.mean(abs(clean)**2)),1e-12)
    error=abs(estimate-clean)**2
    return dict(iq_nmse=float(np.mean(error)/denominator),
        prefix232_nmse=float(np.mean(error[:232])/max(np.mean(abs(clean[:232])**2),1e-12)),
        tail24_nmse=float(np.mean(error[-24:])/max(np.mean(abs(clean[-24:])**2),1e-12)),
        fft_log_mse=float(np.mean((np.log1p(abs(np.fft.fft(estimate,norm='ortho')))-
                                  np.log1p(abs(np.fft.fft(clean,norm='ortho'))))**2)))


def restorations(received,coarse,matrix,covariance):
    observation=real_pair(received);prior=real_pair(coarse);precision=1/SCALE**2
    exact=GaussianSystem(matrix,observation,covariance)
    isotropic=float(np.trace(covariance,axis1=1,axis2=2).mean()/2)
    white=GaussianSystem(matrix,observation,isotropic)
    outputs=dict(received=received,coarse=coarse,
        oracle_white_prior=complex_pair(exact.solve(np.zeros_like(prior),2.)),
        coarse_oracle_full=complex_pair(exact.solve(prior,precision)),
        coarse_oracle_isotropic=complex_pair(white.solve(prior,precision)))
    estimated={}
    for name,taps in (('coarse_estimated10',LOCAL_TAPS),('coarse_estimated49',WIDE_TAPS)):
        # This function receives no oracle fields or source clean reference.
        approximate,variance,details=estimate_action(received,coarse,taps)
        outputs[name]=complex_pair(GaussianSystem(approximate,observation,variance).solve(prior,precision))
        details['oracle_relative_operator_error_for_diagnostic_only']=float(
            np.sum((approximate-matrix)**2)/max(np.sum(matrix**2),1e-12))
        details['estimated_to_oracle_real_noise_ratio_for_diagnostic_only']=variance/isotropic
        estimated[name]=details
    if any(not np.isfinite(value).all() for value in outputs.values()):raise FloatingPointError('Nonfinite source restoration')
    return outputs,dict(oracle_geometry=exact.geometry(precision),oracle_isotropic_real_noise_variance=isotropic,estimated=estimated)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-cache','parent-restoration','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in ('source_cache','parent_restoration','output'):setattr(args,key,getattr(args,key).resolve())
    if args.output.exists():raise FileExistsError('Immutable source operator run exists; reconcile')
    if args.output.name!=RUN or code_commit()!=args.expected_commit:raise ValueError('Registered run/code differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='5':raise ValueError('This diagnostic owns only physical GPU5')
    args.output.mkdir();write_json(args.output/'resolved_config.json',configuration())
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=os.sys.argv,commit=args.expected_commit,
        owner='codex/root/source-operator-feasibility',started=time.time(),GPU=5))
    try:
        import fcntl
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        shared=args.output.parent/CAPACITY_LOCK
        while True:
            with shared.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[5]
                if len(state['pids'])<2 and state['free_mb']>=12000:
                    model,parent=load_coarse(args.parent_restoration,'cuda:0')
                    source=IQSource(args.source_cache,'L_s');x=array_tensor(source.iq[:2],'cuda:0')
                    with torch.no_grad():probe=model(x)
                    if not torch.isfinite(probe).all():raise FloatingPointError('Real source checkpoint smoke failed')
                    del source,x,probe;break
            time.sleep(2)
        write_json(args.output/'smoke.json',dict(status='PASS',source_samples=2,TX_labels_read=0,F_outputs_read=0,
            target_reads=0,parent=parent,persistent_state_updates=0))
        source=IQSource(args.source_cache,'V');indices=source.stratified_indices(4,DATA_SEED)
        if len(indices)!=60:raise ValueError('Fixed source panel count differs')
        ids=np.asarray(source.ids[indices]);np.save(args.output/'physical_ids.npy',ids,allow_pickle=False)
        pairs=args.output/'pairs';pairs.mkdir();summaries={};all_records=[];started=time.perf_counter()
        for scene in ('clean',*SCENES):
            rows=[];predictions={name:[] for name in METHODS};cleans=[];receiveds=[]
            cfg=Config(fs_hz=25e6,fc_hz=2462e6,scenario=SCENES[0] if scene=='clean' else scene,
                processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
            for position,index in enumerate(indices):
                iq=np.asarray(source.iq[index]);clean=iq[0].astype(float)+1j*iq[1].astype(float)
                if scene=='clean':
                    received=clean.copy();matrix=np.eye(512);covariance=np.broadcast_to(np.eye(2)*1e-8,(256,2,2)).copy()
                    meta=dict(state_start='clean',receiver_lock_status='not_applicable',
                        clean_identity_operator=True,oracle_signal_relative_error=0.,observation_decomposition_relative_error=0.)
                else:
                    session='source_RX'+str(source.rx[index])+'_day'+str(source.day[index])
                    receiver=receiver_for_session(cfg,RECEIVER_SEED,session)
                    seed=stable_seed(AUGMENTATION_SEED,NAMESPACE,scene,str(source.ids[index]))
                    received,matrix,covariance,meta=oracle_snapshot(clean,cfg,seed,receiver)
                tensor=array_tensor(np.stack((received.real,received.imag))[None].astype(np.float32),'cuda:0')
                with torch.no_grad():coarse=tensor_array(model(tensor))[0]
                coarse=coarse[0].astype(float)+1j*coarse[1].astype(float)
                output,details=restorations(received,coarse,matrix,covariance)
                record=dict(position=position,scene=scene,rx=int(source.rx[index]),day=int(source.day[index]),
                    state=meta['state_start'],lock=meta['receiver_lock_status'],
                    oracle_signal_relative_error=meta['oracle_signal_relative_error'],
                    observation_decomposition_relative_error=meta['observation_decomposition_relative_error'],
                    metrics={name:waveform(value,clean) for name,value in output.items()},operator=details)
                rows.append(record);all_records.append(record)
                for name,value in output.items():predictions[name].append(np.stack((value.real,value.imag)).astype(np.float32))
                cleans.append(iq);receiveds.append(np.stack((received.real,received.imag)).astype(np.float32))
                print(json.dumps(dict(event='SOURCE_OPERATOR_PROGRESS',scene=scene,count=position+1,total=60)),flush=True)
            np.savez(pairs/(scene+'.npz'),clean=np.stack(cleans),received=np.stack(receiveds),
                **{name:np.stack(values) for name,values in predictions.items() if name!='received'})
            summaries[scene]={name:{metric:float(np.mean([row['metrics'][name][metric] for row in rows]))
                for metric in rows[0]['metrics'][name]} for name in METHODS}
            write_json(args.output/(scene+'.json'),dict(count=len(rows),summary=summaries[scene],records=rows))
        mean={name:float(np.mean([summaries[scene][name]['iq_nmse'] for scene in SCENES])) for name in METHODS}
        oracle_gain=1-mean['coarse_oracle_full']/mean['coarse']
        estimated={name:dict(mean_six_scene_NMSE=mean[name],gain_vs_coarse=1-mean[name]/mean['coarse'],
            clean_NMSE=summaries['clean'][name]['iq_nmse'],waveform_feasible=bool(
             1-mean[name]/mean['coarse']>=.05 and summaries['clean'][name]['iq_nmse']<=.02))
            for name in ('coarse_estimated10','coarse_estimated49')}
        result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',count=60,observations=420,views=summaries,
            mean_six_scene_NMSE=mean,oracle_gain_vs_coarse=oracle_gain,oracle_potential=bool(oracle_gain>=.10),
            estimated=estimated,seconds=time.perf_counter()-started,TX_labels_read=0,F_outputs_read=0,target_reads=0,
            optimizer_updates=0,persistent_state_updates=0,transient_per_observation_parameter_fitting=True,
            interpretation='Waveform feasibility,not identity evidence or final target/deployment validation',goal_complete=False)
        write_json(args.output/'summary.json',result)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_only=True,TX_labels_read=0,
            F_outputs_read=0,target_reads=0,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),
            artifacts_preserved=True,automatic_retry=False));raise


if __name__=='__main__':main()
