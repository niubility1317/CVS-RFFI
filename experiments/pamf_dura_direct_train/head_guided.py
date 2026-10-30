"""Source-calibrated fixed-head feature guidance, scratch G single-stage E20."""
from . import f_guided
from experiments.pamf_dura_efficiency.head_features import head_feature_basis,frozen_head_feature_loss

RUN='20261030-phase1-DURA-source-headFeature-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.head_guided'
CALIBRATION_RUN='20261030-diagnostic-DURA-head-feature-geometry-s2026092701-r01'
CALIBRATION_COMMIT='486c498c28abcbc87cef195c485181d962f1ae90'
FEATURE_WEIGHT=0.006607891036090233
_basis_reader=None
_basis=None


def objective(student,reader,received,clean,action,variance,probes,reference):
    global _basis_reader,_basis
    if reader is not _basis_reader:
        parameter=next(reader.parameters())
        _basis=head_feature_basis(reader).to(device=parameter.device,dtype=parameter.dtype)
        if _basis.shape!=(160,5):raise ValueError('Predesignated original F head rank5 required')
        _basis_reader=reader
    _,terms,restored=f_guided.source_objective(student,received,clean,action,variance,probes,return_restored=True)
    guide=frozen_head_feature_loss(reader,restored,reference,_basis)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+FEATURE_WEIGHT*guide
    return loss,dict(terms,**{f_guided.GUIDANCE_METRIC:guide}),restored


def configure():
    f_guided.configure();base=f_guided.base
    original_configuration=base.configuration;original_command=base.command
    def configuration():
        value=original_configuration()
        value.update(run=RUN,recipe='source_F_guided_head_feature_source_calibrated',
            F_feature_weight=FEATURE_WEIGHT,F_guidance_metric='F_head_projected_feature_error',F_head_basis_rank=5,
            F_head_basis='SVD centered normalized original six by160 weights;orthonormal rank5;fixed/no fitting',
            F_feature_normalization_eps=1e-4,F_projected_scale='160/(2*rank)',
            F_source_calibration=dict(run=CALIBRATION_RUN,commit=CALIBRATION_COMMIT,source_model='noVarAux finalE20',
                batches=32,positions=512,cached_epochs=[1,20],rule='alpha=.25/median(projected/base norm ratio)',
                observed_calibrated_median_ratios=[.30927746555628666,.23095659781471053],all_source_conditions_passed=True),
            objective='noVarAuxIQ+prior+spectral+action base+sourcecalibrated fixedoriginalhead-visible normalizedfeature residual',
            source_supervision='Paired source IQ/action and frozen F clean features/original head weights;inherits source-labeled F knowledge;direct TX labels0')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item==f_guided.MODULE else item for item in original_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command
    f_guided.GUIDANCE_METRIC='F_head_projected_feature_error';f_guided.objective=objective


if __name__=='__main__':
    configure();f_guided.base.main()
