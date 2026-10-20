from experiments.pamf_restoration.data import SCENES

RUN='20261020-diagnostic-restoration-oracle-attribution-source-F50-m8-r01'
F_SEEDS=(2026092701,2026092702,2026092703)
F_COMMIT='4df7b94d0fd8ce29056bd8236364e45afa92e07f'
BASELINE_RUN='20261020-phase1-restoration-spatial-conditioning-source-s2026092701-m2-r01'
BASELINE_ARM='pooled_features'
PARENT_RUN='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
PARENT_COMMIT='b01b534f52daf9cae9e940ed6ec6f9243bccbdbe'
VIEWS=('clean',*SCENES)
READOUTS=('identity','parent_R03','clean_tail24','clean_prefix232','clean_low64','clean_high64','clean_gain','clean_reference')
RECEIVERS=(1,3,4,6,8);PER_RECEIVER=36;COUNT=180;BATCH=36
LOCK_RUNS=('20261020-phase1-restoration-scratch-guidance-source-s2026092701-m2-r01',
           '20261020-phase1-restoration-spectral-precision-source-s2026092701-m2-r01')


def configuration():
    return dict(run_id=RUN,source_only=True,source_role='V',count=COUNT,views=list(VIEWS),readouts=list(READOUTS),
        receivers=list(RECEIVERS),selection='36systematiclinspacepositionsinexistingorderperRX;noTXlabels',
        parent_run=PARENT_RUN,parent_commit=PARENT_COMMIT,baseline_source_run=BASELINE_RUN,baseline_arm=BASELINE_ARM,
        F_seeds=list(F_SEEDS),F_commit=F_COMMIT,oracle_diagnostic_only=True,formal_target_predictions=False,
        runtime_supported=False,source_selection=False,optimizer_updates=0,head_fit_updates=0,state_updates=0,
        target_reads=0,CPU_threads=2,GPU_workers=0,
        interventions='ReplaceR03tail24/prefix232/lowest64/highest64cleanFFTbins;oraclefullframecomplexgain;received/R03/cleanreferences',
        precision='FP32IQ;FP64NumPycomplexcounterfactuals;nativefrozenF/defaultheadargmax;novote',
        interpretation='Mechanismdiagnosison180sourcephysicalexamples,notdeployablecorrection,independenttargetsuccessorperformanceupperbound',
        CPU_locks=list(LOCK_RUNS),CPU_policy='Holdbothactiveownrunscore.lock inlistedorderthroughinference+independentchildscore;nohealthyjobedits',
        stop_rule='Illegalparent/source/config,collision,nonfinite,changedF,incompletepredictions:preserve/fail,noautoretry')
