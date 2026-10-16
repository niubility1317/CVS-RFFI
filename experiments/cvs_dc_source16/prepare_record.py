"""Use the maintained existing registration schema, for this source-only row."""
from copy import deepcopy
from . import design as d


def record():
    template=d.ROOT/'automation_reports/CV-SincNet/20261016-phase1-F-timing-sourceRX1-e20-s2026092701-m2-r01/experiment.json'
    c=d.read(template)
    c.update(run_id=d.RUN,group_id='phase1-F-DC-observation-Ag-As',display_name='F 后续 IQ 路线：源 DC 的跨 TX/day 重复性证伪',description='Test whether fit-source RX DC differences transport across held TX and day before proposing any bounded DC augmentation. One source-statistic diagnostic; no classifier/model/checkpoint/training/target.',kind='cvs',stage='Phase1-source-diagnostic',aliases=['source DC crossfit'],tags=['F','DC','IQ_augmentation','source_only','cross_TX_day','no_training'],parent_run_ids=[],replaces_run_id=None,authorization='Active user goal: pursue a distinct evidence-based IQ Ag/As route on fixed F/E20; prior fractional timing rejected. This first source-only DC question excludes all other-chat phase/origin/noise work.')
    c['code']=dict(commit='Actual execution commit in release_commit.txt/resolved_config',checkout=str(d.ROOT),environment='ordinary N607 existing CVS-RFFI Python; CPU2; CUDA hidden',cwd='PROJECT/releases/'+d.RELEASE)
    c['data'].update(version='Existing validated L6300 cache, four fit RX subset only',representation='Unchanged native RMS equalized1 IQ256; fixed reference[80:160] DC statistic',source_receivers=[3,4,6,8],source_days=[1,2,3],roles=dict(L_s='5040 fitRX used;1260 RX1 deserialized then excluded before any IQ statistic',U_s='Unused; TX truth never accessed',V='Unused'),physical_ids_ref='Existing source_contract L_s handles; exact fit5040 IDs retained privately',leo_config_ref=None)
    c['permissions']=dict(regime='source_only',query_use='none',external_method_exception=None,claim_scope='RX-associated complex DC repeatability only; no hardware identification, identity-recognition or performance claim')
    c['checkpoint']=dict(initialization='not_applicable_no_model',sources=[],contract_check_ref='No checkpoint constructed or loaded. Existing source raw cache roles matched before L extraction; no parameter inheritance.',provenance_verdict='Only existing legal L IQ; not a checkpoint experiment',selection_rule='No model/checkpoint/candidate selection; fixed source-statistic criteria before data')
    c['execution'].update(launch_owner=d.OWNER,gpu_policy='CPU only;2threads;CUDA_VISIBLE_DEVICES empty;no GPU or model allocation',remote_run_root='PROJECT/runs/'+d.RUN,remote_log_root='PROJECT/releases/'+d.RELEASE+'/source_DC.stdout.log',local_artifact_root='automation_reports/CV-SincNet/'+d.RUN,launch_command='CVS-RFFI/bin/python -u -m experiments.cvs_dc_source16.run',stop_rule='Wrongsource/role, missingcache, nonfinite observation, wrong matrix or exception: preserve all partial output; no automatic retry; never stop or modify unrelated jobs.')
    c['expected_artifacts']=['resolved_config/active/completion','DC_crossfit.json including all18folds/72cells and controls','DC_cells.json','fit_L_ids.json private','source_DC stdout private','independent crossfit recount/figures/report/registry event']
    c['metrics_plan']=dict(metric_names=['DC_cell_prediction_risk_gain','positive_folds','per_RX_risk_gain','wrong_RX_controls','TX_half_center_cosine','DC_cycle_repeatability','DC_power_fraction'],dimensions=['RX3/4/6/8','6TX','day1/2/3','18heldTXanddayfolds'],prediction_ref='No identity prediction; DC_crossfit.json source statistic only',scorer_ref='Independent saved72cell-means crossfit recalculation; no truth-side target scorer applies')
    row=deepcopy(c['rows'][0]);rowid='DC_SOURCE_CROSSFIT'
    row.update(row_id=rowid,method='Fixed reference complex DC; leave TX and day out, pooled/wrongRX controls',purpose='single_hypothesis',config_ref='experiments/cvs_dc_source16/design.py:config',resolved_config_ref='PROJECT/runs/'+d.RUN+'/resolved_config.json',data_overrides={},seeds={k:None for k in ('model','split','data','augmentation','support','evaluation')},seed_notes='No stochastic model/augmentation/support/evaluation. Source physical split392005 reused; cache directory seed2026092701 identifies historical preparation, not a newly trained model.',k=None,scenario='source_clean_statistic_no_model',optimizer=None,lr=None,epochs=0,fl_rounds=None,budget_ref='5040fitL/72cells/18folds;0model,0network forward,0optimizer,0CP;deserialized old cache unused U/V and1260heldL disclosed',output_root='PROJECT/runs/'+d.RUN,log_path='PROJECT/releases/'+d.RELEASE+'/source_DC.stdout.log',command='CVS-RFFI/bin/python -u -m experiments.cvs_dc_source16.run',expected_artifacts=c['expected_artifacts'])
    row['seeds']['split']=392005;c['rows']=[row]
    c['notes']=[
        'Reference[80,160) fixed before data. All70 samples in each of72 fitTX/RX/day cells; no alternate window/receiver/strength selected after result.',
        '18folds exclude heldTX and heldday jointly. Primary totalrisk gain>=.05,atleast14positivefolds,eachRXgain>=0,gain over bestof3wrongRX>=.02,TXhalf center cosine>=.8,and DC/odd-even difference power>=2. Zero baseline means undefined riskgain, not perfect performance.',
        'A PASS is prerequisite evidence for a separately fixed bounded input/defaulthead check and matched F E20 training; it is not Ag efficacy and cannot complete the joint objective. FAIL closes this fixedDC hypothesis without rescue.',
        'DC may contain TX LO leakage, finite waveform mean, CFO leakage or preprocessing; do not claim real receiver replacement or label preservation from hardware literature.',
        'Source-cache provenance/roles reused from existing validated preparation; no new hash/receipt/ledger. G0 constructs no network and consumes no checkpoint, hence no realCP smoke applies at this stage.',
        'No modifications to healthy phase/origin/WHITE30/ordinary or other tasks. Future fixedmodel remains F/defaultcos/NoExplicitCFO with necessary internal compensation; current analysis makes no deployment change.'
    ]
    c.pop('registered_at',None)
    return c


if __name__=='__main__':
    path=d.ROOT/'local_artifacts/F_DC_source_20261016/spec.json';d.write(path,record());print(path)
