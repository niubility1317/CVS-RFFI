"""Existing registration schema for one fixed, source-only statistical row."""
from . import design as d


def record():
    c=d.read(d.ROOT/'automation_reports/CV-SincNet/20261016-diagnostic-F-DC-source-crossfit-manysig-m1-r01/experiment.json')
    c.update(run_id=d.RUN,group_id='phase1-F-gain-envelope-Ag-As',display_name='F 后续 IQ 路线：包内缓变增益的跨 TX/day 源检验',description='Fixed cycle-power slope statistics and cross-TX/day transport before any gain augmentation; no model/checkpoint/training/target.',aliases=['gain envelope source crossfit'],tags=['F','gain_envelope','IQ_augmentation','source_only','cross_TX_day','no_training'],authorization='Active joint Ag/As goal; distinct from other-chat phase/origin and closed fractional-delay/DC hypotheses. Fixed source-only gain-envelope question first.')
    c['code'].update(commit='Actual execution commit in release_commit.txt/resolved_config',cwd='PROJECT/releases/'+d.RELEASE)
    c['data'].update(representation='Native RMS IQ256; fixed reference[80:160] four20sample power slope; no IQ changes')
    c['permissions']['claim_scope']='Source RX association of median absolute cycle-power slope only; no hardware, label-preservation or recognition claim'
    c['checkpoint']['contract_check_ref']='No checkpoint or model constructed. Same already validated original L6300 raw cache roles; fit5040 filtered before IQ statistics.'
    c['execution'].update(launch_owner=d.OWNER,remote_run_root='PROJECT/runs/'+d.RUN,remote_log_root='PROJECT/releases/'+d.RELEASE+'/source_gain.stdout.log',local_artifact_root='automation_reports/CV-SincNet/'+d.RUN,launch_command='CVS-RFFI/bin/python -u -m experiments.cvs_gain_envelope16.run')
    artifacts=['resolved_config/active/completion','gain_crossfit.json: all18folds and controls','gain_cells.json:72cells','packet_observations.json private:5040 slopes/R2/4cyclelevels','fit_L_ids.json private','source_gain stdout private','independent recount/report/registry']
    c['expected_artifacts']=artifacts
    c['metrics_plan'].update(metric_names=['median_abs_slope_crossfit_risk_gain','positive_folds','per_RX_risk_gain','wrong_RX_controls','TX_half_center_cosine','packet_linear_R2','RMS_slope_retention'],prediction_ref='No identity prediction; source gain_crossfit.json only',scorer_ref='Independent saved72cell and5040packet observation scalar recount, no target truth')
    row=c['rows'][0]
    row.update(row_id='GAIN_ENVELOPE_SOURCE_CROSSFIT',method='Per cell median absolute fourcycle dB power slope; leave TX and day out',config_ref='experiments/cvs_gain_envelope16/design.py:config',resolved_config_ref='PROJECT/runs/'+d.RUN+'/resolved_config.json',scenario='source_clean_statistic_no_model',output_root='PROJECT/runs/'+d.RUN,log_path='PROJECT/releases/'+d.RELEASE+'/source_gain.stdout.log',command=c['execution']['launch_command'],expected_artifacts=artifacts)
    c['notes']=[
        'Source-only5040 L on RX3/4/6/8;70 per TX/RX/daycell;RX1/U/V statistics excluded. Old cache otherroles/features deserialize unused, explicitly disclosed.',
        'Window80:160 fixed;four20sample powers;centered dB OLS q=(-1,-1/3,1/3,1). Primary median absolute slope per72cells.18folds simultaneously exclude held TX/day from both RX and pooled fits.',
        'All conditions fixed before real data:crossfitgain>=.05,14/18positive,allRXgain>=0,2pp above best3wrongRX,TXhalfcenter cosine>=.8,median perpacket linearR2>=.8,median abs slope>=.01dB,and RMS slope difference<=1e-10. No alternatewindow/statistic/receiver/strength rescue onFAIL.',
        'PASS only supports separately registered bounded positive commonIQ gain input and matched F/E20 defaulthead training. It does not prove AGC hardware, identity preservation or Ag efficacy. No automatic training or target.',
        'Ordinary wholepacket gain and receiverDG softclip are different operators. No new PA/DAC/IQ imbalance/clipping/CFO/phase/time or practical chain is applied.',
        'No model/checkpoint means realcheckpoint smoke is inapplicable. Reuse existing matching source handles; no new data hashes, ledger, permission layer or blanket review.',
        'Fixed later boundary F/SIG_TIME3/unifiedSinc/joint160/defaultcos/NoExplicitCFO, E20, old U rule. Current diagnostic consumes zero forward/update budget.'
    ]
    c.pop('registered_at',None)
    return c


if __name__=='__main__':
    p=d.ROOT/'local_artifacts/F_gain_envelope_20261016/spec.json';d.write(p,record());print(p)
