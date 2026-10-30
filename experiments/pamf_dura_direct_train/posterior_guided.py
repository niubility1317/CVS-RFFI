"""Source feature-derived posterior guidance; scratch single-stage G, unchanged F."""
from . import f_guided
from experiments.pamf_dura_efficiency.posterior_loss import frozen_posterior_loss

RUN='20261030-phase1-DURA-source-posteriorKL010-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.posterior_guided'
WEIGHT=.1


def objective(student,reader,received,clean,action,variance,probes,reference):
    _,terms,restored=f_guided.source_objective(student,received,clean,action,variance,probes,return_restored=True)
    guide=frozen_posterior_loss(reader,restored,reference)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+WEIGHT*guide
    return loss,dict(terms,**{f_guided.GUIDANCE_METRIC:guide}),restored


def configure():
    f_guided.configure();base=f_guided.base
    previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='source_F_guided_original_head_soft_posterior_KL010',
            F_guidance_metric='F_posterior_KL',F_feature_weight=WEIGHT,F_posterior_weight=WEIGHT,F_posterior_temperature=1.,
            F_posterior_dimension=6,F_posterior_direction='KL(softmax(originalhead(cleanFfeature)) || softmax(originalhead(restoredFfeature)))',
            hard_pseudo_TX_labels=False,source_reference='Original paired clean IQ unchanged;no spectral or channel reassignment',
            objective='noVarAuxIQ+prior+spectral+action base+.1 continuous original-frozen-head posterior KL',
            source_supervision='Same frozen F source features mapped by its original frozen head;continuous distributions only;no hard pseudo TX/true TX labels',
            coefficient_comparison='Same numeric .1 as cosine control, different loss units;not gradient-strength matched or source-calibrated',
            G_forwards_per_training_visit=1,max_total_visible_GPU_processes=3,allowed_GPUs=[4,6,7],
            resource_scope='One own worker on GPU4/6/7,>=12GB free,at most2 existing compute PIDs before adding;other owners unchanged',
            host_numpy_conversion='DLPack_owned_FP32')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item==f_guided.MODULE else item for item in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command
    f_guided.GUIDANCE_METRIC='F_posterior_KL';f_guided.objective=objective


if __name__=='__main__':
    configure();f_guided.base.main()
