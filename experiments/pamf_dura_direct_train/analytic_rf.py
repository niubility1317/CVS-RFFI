"""Single-stage scratch trial with one source-calibrated analytic RF loss."""
import time
import torch

from . import no_variance
from . import runner as base
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.rf_geometry import analytic_terms, BRANCHES
from experiments.pamf_restoration.views import PAMFViews

RUN='20261030-phase1-DURA-source-analyticRF-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.analytic_rf'
ALPHA=.009527944261805803
CALIBRATION_RUN='20261030-diagnostic-DURA-source-analytic-RF-geometry-s2026092701-r01'
CALIBRATION_COMMIT='c785ebd8816b180445cbdfc721c3551bc845ddbc'
_extractor=None
_snapshot=None
_inputs=0


def extractor(device):
    global _extractor,_snapshot,_inputs
    if _extractor is None or next(_extractor.parameters()).device!=torch.device(device):
        _extractor=PAMFViews().eval().requires_grad_(False).to(device)
        _snapshot={k:v.detach().clone() for k,v in _extractor.state_dict().items()}
        _inputs=0
    return _extractor


def assert_frozen():
    if _extractor is None or any(p.grad is not None for p in _extractor.parameters()) or any(
            not torch.equal(v,_snapshot[k]) for k,v in _extractor.state_dict().items()):
        raise ValueError('Fixed analytic RF transforms changed')


def objective(model,received,clean,actual_action,actual_variance,probes):
    global _inputs
    _,terms,restored=source_objective(model,received,clean,actual_action,actual_variance,probes,return_restored=True)
    auxiliary=analytic_terms(restored,clean,extractor(restored.device));_inputs+=2*len(clean)
    rf=sum(auxiliary.values())
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+ALPHA*rf
    return loss,dict(terms,analytic_RF_nmse=rf,**{'analytic_'+key+'_nmse':value for key,value in auxiliary.items()})


def configure():
    no_variance.configure()
    original_configuration=base.configuration;original_command=base.command
    original_smoke=base.smoke;original_train=base.train
    def configuration():
        value=original_configuration()
        value.update(run=RUN,recipe='single_stage_scratch_source_analytic_RF',
            objective='finalIQNMSE+.1priorIQNMSE+.05multiscaleSTFT+.05sourceactionprobeNMSE+source_fixed_alpha*sum5analyticRFNMSE',
            analytic_RF_weight=ALPHA,analytic_RF_branches=list(BRANCHES),
            analytic_RF_reference=dict(run=CALIBRATION_RUN,commit=CALIBRATION_COMMIT,
                rule='.25/median32source_output_gradient_norm_ratios',output_gradient_only=True),
            analytic_RF_transform='FreshfixedPAMFViews;nominalSinc;no trainedFweights/embedding/head forward',
            extra_loss_metrics=['analytic_RF_nmse',*['analytic_'+key+'_nmse' for key in BRANCHES]],
            G_forwards_per_training_visit=1,analytic_transform_inputs_per_training_visit=2,
            analytic_transform_optimizer_updates=0,training_F_access=False,
            allowed_GPUs=[4,6,7],max_total_visible_GPU_processes=3,
            clean_identity_visits='OriginalR07cached15percentunchanged',target_feedback=False)
        return value
    def command(*args,**kwargs):
        result=original_command(*args,**kwargs);result[3]=MODULE;return result
    def smoke(args,p):
        original_smoke(args,p);assert_frozen()
        base.write_json(p['output']/'analytic_smoke.json',dict(status='PASS',alpha=ALPHA,
            analytic_transforms_unchanged=True,trained_F_weights_in_loss=False,analytic_transform_inputs=_inputs,
            TX_labels_read=0,V_reads=0,target_reads=0))
    def train(args,p):
        # Construct before the matched trainer resets its G/probe RNG. No F or
        # ancestor G checkpoint is loaded; targets are computed in each batch.
        started=time.perf_counter();extractor('cuda:0');prepared=time.perf_counter()
        original_train(args,p);assert_frozen()
        if _inputs!=2520000:raise ValueError('Full matched source RF visit budget required')
        base.write_json(p['output']/args.arm/'analytic_RF_complete.json',dict(status='PASS',
            alpha=ALPHA,analytic_transform_inputs=_inputs,analytic_parameters_unchanged=True,
            analytic_optimizer_updates=0,training_F_access=False,TX_labels_read=0,V_reads=0,target_reads=0,
            additional_inference_branches=0,additional_G_forwards=0,
            analytic_preparation_seconds=prepared-started,total_training_pipeline_seconds=time.perf_counter()-started))
        # Evaluation uses only the original G and F deployment graph.
        global _extractor,_snapshot
        _extractor=None;_snapshot=None
    base.RUN=RUN;base.configuration=configuration;base.command=command
    base.direct_objective=objective;base.smoke=smoke;base.train=train


if __name__=='__main__':configure();base.main()
