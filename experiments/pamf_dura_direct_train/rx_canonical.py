"""Source F guidance toward a fixed head-protected RX feature reference."""
import numpy as np
import torch
from torch.nn import functional as F

from . import f_guided
from experiments.pamf_dura_efficiency import rx_geometry,canonical_gradient
from experiments.pamf_dura_diagnostics.runner import source_indices

RUN='20261031-phase1-DURA-source-RXcanonical-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.rx_canonical'
WEIGHT=.019528750294704578
METRIC='F_RXcanonical_cosine_error'
_mean=_basis=None
_device_constants={}
_visits=0


def prepare(p,root):
    global _mean,_basis,_device_constants,_visits
    base=f_guided.base;folder=root/'runs'/rx_geometry.RUN
    geometry=base.read_json(folder/'results.json');launch=base.read_json(folder/'launch.json')
    calibration=base.read_json(root/'runs'/canonical_gradient.RUN/'results.json')
    if (launch['commit']!=canonical_gradient.GEOMETRY_COMMIT or geometry['F_teacher']!=f_guided.teacher_reference()
        or not geometry['geometry']['source_canonical_feature_calibration_eligible'] or geometry['canonical_alpha']!=.25
        or not calibration['decision']['RX_canonical_training_study_eligible'] or calibration['decision']['alpha']!=WEIGHT
        or calibration['feature_RX_shrinkage']!=.25 or calibration['F_teacher']!=f_guided.teacher_reference()):
        raise ValueError('Fixed qualified original source RX reference and weight required')
    roles=np.load(p['cache']/'train/roles.npy',allow_pickle=False);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    with np.load(folder/'geometry.npz',allow_pickle=False) as a:
        for key,value in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices]).items():
            if not np.array_equal(a[key],value):raise ValueError('Source reference physical role mismatch')
        mean=a['fit_feature_mean'];basis=a['RX_basis'];probe=a['probe_features'];expected=a['canonical_features']
    if mean.shape!=(160,) or basis.shape!=(160,4) or not np.isfinite(mean).all() or not np.isfinite(basis).all():
        raise ValueError('Fixed source feature mean and rank4 RX basis required')
    _mean=base.tensor(mean,'cpu');_basis=base.tensor(basis,'cpu');_device_constants={};_visits=0
    actual=canonical_reference(base.tensor(probe,'cpu'))
    torch.testing.assert_close(actual,base.tensor(expected,'cpu'),atol=2e-6,rtol=2e-6)


def canonical_reference(reference):
    if _mean is None or _basis is None:raise ValueError('Source RX reference must be prepared')
    key=(reference.device,reference.dtype)
    if key not in _device_constants:
        _device_constants[key]=(_mean.to(reference),_basis.to(reference))
    mean,basis=_device_constants[key]
    unit=F.normalize(reference.detach(),dim=1,eps=1e-4)
    return unit-.25*((unit-mean)@basis)@basis.T


def objective(student,reader,received,clean,action,variance,probes,reference):
    global _visits
    _,terms,restored=f_guided.source_objective(student,received,clean,action,variance,probes,return_restored=True)
    target=canonical_reference(reference)
    guide=1-F.cosine_similarity(reader.features(restored),target,dim=1,eps=1e-8)
    _visits+=len(clean)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+WEIGHT*guide
    return loss,dict(terms,**{METRIC:guide}),restored


def configure():
    f_guided.configure();base=f_guided.base
    previous_configuration=base.configuration;previous_command=base.command
    previous_smoke=base.smoke;previous_train=base.train
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='source_F_guided_head_protected_RX_reference',F_guidance_metric=METRIC,F_feature_weight=WEIGHT,
            F_RX_reference=dict(geometry_run=rx_geometry.RUN,geometry_commit=canonical_gradient.GEOMETRY_COMMIT,
                calibration_run=canonical_gradient.RUN,calibration_commit='c5a6f6cbb21fa541d5e46f48e1f510875097975f',
                shrinkage=.25,head_rank=5,RX_rank=4,fit_records=62488,holdout_records=512),
            source_reference='Original paired IQ unchanged;only frozenFfeature teacher shrinks sourceRX span orthogonal to original centeredhead',
            objective='noVar fourbase losses+sourcefixedweight*cosine(Frestored,RXcanonicalFclean)',
            hard_pseudo_TX_labels=False,G_forwards_per_training_visit=1,additional_inference_branches=0,
            coefficient_comparison='Reference and weight both differ from old cos.1;not an isolated causal reference ablation',
            max_total_visible_GPU_processes=3,allowed_GPUs=[4,6,7],host_numpy_conversion='DLPack_owned_FP32')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==f_guided.MODULE else v for v in previous_command(args,mode,arm,role)]
    def smoke(args,p):
        prepare(p,args.root);previous_smoke(args,p)
        base.write_json(p['output']/'RX_reference_smoke.json',dict(status='PASS',reference_matches_qualified_source=True,
            shrinkage=.25,loss_weight=WEIGHT,head_parameters_updated=0,target_reads=0,TX_labels_read=0))
    def train(args,p):
        prepare(p,args.root);previous_train(args,p)
        if _visits!=1260000:raise ValueError('Full matched source RX reference visit budget required')
        base.write_json(p['output']/args.arm/'RX_reference_complete.json',dict(status='PASS',reference_visits=_visits,
            source_reference_fixed=True,shrinkage=.25,loss_weight=WEIGHT,reference_optimizer_updates=0,
            additional_G_forwards=0,additional_inference_branches=0,TX_labels_read=0,V_reads=0,target_reads=0))
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.smoke=smoke;base.train=train
    f_guided.GUIDANCE_METRIC=METRIC;f_guided.objective=objective


if __name__=='__main__':configure();f_guided.base.main()
