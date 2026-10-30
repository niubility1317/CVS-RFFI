"""User-authorized resource-only replacement of the unstarted R01 queue."""
from . import head_guided

RUN='20261030-phase1-DURA-source-headFeature-s2026092701-r02'
MODULE='experiments.pamf_dura_direct_train.head_guided_capacity3'


def configure():
    head_guided.configure();base=head_guided.f_guided.base
    previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,max_total_visible_GPU_processes=3,allowed_GPUs=[4,6,7],
            resource_scope='User2026-10-30explicitlyauthorized one own worker on one of GPU4/6/7 with>=12GB free and<=2 existing compute processes;oldqueue replaced onlybeforeanyGPUrowstart',
            host_numpy_conversion='DLPack_owned_FP32;actualsource1356scalarsbit-equal',
            science_unchanged_from='20261030-phase1-DURA-source-headFeature-s2026092701-r01')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item==head_guided.MODULE else item for item in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command


if __name__=='__main__':
    configure();head_guided.f_guided.base.main()
