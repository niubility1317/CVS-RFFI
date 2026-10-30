"""Scratch E20 with one-forward source-calibrated paired spectral augmentation."""
import time
import numpy as np
import torch

from . import no_variance
from experiments.pamf_dura_diagnostics.runner import tensor
from experiments.pamf_dura_efficiency.spectral_style import spectral_reference
from experiments.pamf_restoration.models import complex_iq,pair

RUN='20261030-phase1-DURA-source-spectralMix-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.spectral_mix'
CALIBRATION='20261030-diagnostic-DURA-source-spectral-stress-s2026092701-r01'
CALIBRATION_COMMIT='a55b378eb47a6796ab375d2884f2e71b541a74a6'
SEED=2026103016


class SpectralPairing:
    def __init__(self,gain,sessions,rx,day,roles,device):
        self.bank=tensor(gain,device)
        self.sessions=np.asarray(sessions,np.int64)
        lookup={tuple(v):i for i,v in enumerate(self.sessions.tolist())}
        self.origins=np.asarray([lookup[int(r),int(d)] for r,d in zip(rx,day)],np.int64)
        self.choices=np.stack([np.flatnonzero(self.sessions[:,0]!=r) for r in self.sessions[:,0]])
        self.roles=np.asarray(roles);self.rng=np.random.default_rng(SEED)
        self.seen=self.selected=self.calls=0
        self.by_role={'L_s':0,'U_s':0};self.draws=np.zeros((len(sessions),len(sessions)),np.int64)

    @torch.no_grad()
    def __call__(self,received,clean,action,positions):
        mask=self.rng.random(len(positions))<.5
        selected=np.flatnonzero(mask)
        self.seen+=len(positions);self.selected+=len(selected);self.calls+=1
        if not len(selected):return received,clean
        original=self.origins[positions[selected]]
        destination=self.choices[original,self.rng.integers(self.choices.shape[1],size=len(selected))]
        index=torch.tensor(selected.tolist(),device=clean.device,dtype=torch.long)
        origin_t=torch.tensor(original.tolist(),device=clean.device,dtype=torch.long)
        destination_t=torch.tensor(destination.tolist(),device=clean.device,dtype=torch.long)
        reference=clean.clone()
        reference.index_copy_(0,index,spectral_reference(clean.index_select(0,index),self.bank[origin_t,destination_t]))
        observed=received+pair(action(complex_iq(reference-clean)))
        np.add.at(self.draws,(original,destination),1)
        for role in self.by_role:self.by_role[role]+=int((self.roles[positions[selected]]==role).sum())
        return observed,reference

    def summary(self):
        return dict(seen=self.seen,selected=self.selected,actual_fraction=self.selected/self.seen if self.seen else None,
            calls=self.calls,selected_by_role=self.by_role,sessions=self.sessions.tolist(),cross_session_draw_counts=self.draws.tolist(),
            seed=SEED,probability=.5,numpy_version=np.__version__,F_guidance_inputs=0,TX_labels_read=0,teacher_forwards=0)


def load_pairing(p,device):
    root=p['output'].parent/CALIBRATION
    base=no_variance.base
    complete=base.read_json(root/'complete.json');meta=base.read_json(root/'results.json');launch=base.read_json(root/'launch.json')
    if (complete['status'],complete['training_contrast_eligible'],launch['commit'])!=('ANALYZED_SOURCE_ONLY',True,CALIBRATION_COMMIT):
        raise ValueError('Qualified fixed source spectral calibration required')
    if not meta['G_unchanged'] or not all(v['passed'] for v in meta['decisions']) or any(meta[k]!=0 for k in ('optimizer_updates','TX_labels_read','V_reads','target_reads','F_optimization_inputs','F_feature_analysis_inputs')):
        raise ValueError('Source-only IQ calibration without F guidance required')
    with np.load(root/'spectral_bank.npz',allow_pickle=False) as bank:
        gain=bank['gain'];sessions=bank['sessions'];counts=bank['session_counts']
    expected=np.asarray([(r,d) for r in (1,3,4,6,8) for d in (1,2,3)],np.int64)
    if not np.array_equal(sessions,expected) or not np.array_equal(counts,np.full(15,4200)) or gain.shape!=(15,15,256) or not np.isfinite(gain).all() or not (gain>0).all():
        raise ValueError('Fixed original source-session spectral bank required')
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    roles=np.load(p['cache']/'train/roles.npy',allow_pickle=False)
    if rx.shape!=(63000,) or day.shape!=(63000,) or roles.shape!=(63000,):raise ValueError('Original source metadata required')
    sampler=SpectralPairing(gain,sessions,rx,day,roles,device)
    sampler.calibration_source_seconds=meta['calibration']['seconds']
    return sampler


def configure():
    no_variance.configure();base=no_variance.base
    previous_configuration=base.configuration;previous_command=base.command
    previous_smoke=base.smoke;previous_train=base.train
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='single_stage_scratch_no_variance_aux_source_spectral_mix',
            source_pair_seed=SEED,source_pair_probability=.5,source_spectral_calibration=CALIBRATION,
            source_spectral_calibration_commit=CALIBRATION_COMMIT,
            source_pairing='Source PSD sqrt(dest/origin),uniform other-source-RX session;unit RMS;y+A(Sx-x),same A/noise',
            G_forwards_per_training_visit=1,learned_teachers=0,source_statistic_records=63000,
            source_statistic_scope='Original source L6300/U56700 empirical PSD;not a causal RX transfer estimate;no F features/labels',
            max_total_visible_GPU_processes=3,allowed_GPUs=[4,6,7],
            resource_scope='One own worker on GPU4/6/7,>=12GB free,at most2 existing compute PIDs before adding;other owners unchanged',
            host_numpy_conversion='DLPack_owned_FP32',comparison='Fixed original noVar E20,source cf40;no checkpoint initialization')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item==no_variance.MODULE else item for item in previous_command(args,mode,arm,role)]
    def smoke(args,p):
        sampler=load_pairing(p,'cpu');base.training_pair=sampler
        previous_smoke(args,p)
        if sampler.selected==0:raise ValueError('Real smoke did not exercise source spectral augmentation')
        base.write_json(p['output']/'spectral_smoke.json',dict(status='PASS',**sampler.summary()))
    def train(args,p):
        started=time.perf_counter();sampler=load_pairing(p,'cuda:0');preparation=time.perf_counter()-started
        base.training_pair=sampler;previous_train(args,p)
        if (sampler.seen,sampler.calls)!=(1260000,4940):raise ValueError('Incomplete registered spectral augmentation exposure')
        base.write_json(p['output']/args.arm/'source_pairing_summary.json',dict(status='SOURCE_PAIRING_COMPLETE',
            preparation_seconds=preparation,total_G_training_pipeline_seconds=time.perf_counter()-started,
            calibration_run=CALIBRATION,calibration_source_seconds=sampler.calibration_source_seconds,**sampler.summary()))
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke;base.train=train


if __name__=='__main__':
    configure();no_variance.base.main()
