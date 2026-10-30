"""Independent pure-G E30 control for thirty fresh source channel draws."""
import gc
from . import no_variance
from . import prior_aux_free as queue
from .coarse_hint import learning_rate
base=no_variance.base
RUN='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.repeat_views'
FRESH_RUN='20261031-phase1-DURA-freshViews-E30-s2026092701-r01'
FRESH_MODULE='experiments.pamf_dura_direct_train.fresh_views'
FRESH_COMMIT='32a32998d3c76105b017a0905300e8edbce64e16'
FRESH_RELEASE='20261031-DURA-freshViews-E30-r01'


def configure():
    no_variance.configure();previous_configuration=base.configuration;previous_command=base.command;previous_train=base.train
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_repeat20_matched_pureG_control',epochs=30,updates=7410,new_cumulative_epochs=30,
            L_exposures=189000,U_exposures=1701000,corruption_epochs=20,
            source_freeze='fixedfinalE30/7410;noearlystop/source-target-ranking',
            source_corruption_schedule='01..20then01..10',matched_fresh_run=FRESH_RUN,
            CUDA_probe_seed=2026092701,numpy_shuffle_seed=2026103004,
            teacher_model=None,training_F_access=False,F_guided_training=False,source_F_informed_method_research=True,
            source_supervision='Originalsourcepairedclean/action;no teacher/F/TX/V/targetoptimization',
            source_rationale='HistoricalIQenhancephysicalhasCUDAprobe seed2026103107;independentpureGcontrolmatchesfreshcodepath/seed2026092701',
            allowed_GPUs=[4],max_total_visible_GPU_processes=3,host_numpy_conversion='DLPack_owned_FP32',
            evaluation_resource_queue=dict(predecessor_module=FRESH_MODULE,predecessor_commit=FRESH_COMMIT,
                wait_until_actual_worker_and_scorer_exit=True))
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==no_variance.MODULE else v for v in previous_command(args,mode,arm,role)]
    def train(args,p):
        previous_train(args,p);gc.collect()
        import torch
        torch.cuda.empty_cache();queue.wait_evaluation(args,p)
    queue.PREDECESSOR_MODULE=FRESH_MODULE;queue.PREDECESSOR_COMMIT=FRESH_COMMIT;queue.PREDECESSOR_RELEASE=FRESH_RELEASE
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.learning_rate=learning_rate;base.train=train


if __name__=='__main__':configure();base.main()
