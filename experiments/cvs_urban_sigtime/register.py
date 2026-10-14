"""Maintain the existing experiment record, with the actual conditional row matrix."""
from pathlib import Path
import json
from . import design as d

def make():
    root=Path('E:/type10-7');template=root/'local_artifacts/cvs_urban_sigtime_20261014/registry_template.json'
    spec=json.loads(template.read_text(encoding='utf-8'))
    report=f'automation_reports/CV-SincNet/{d.RUN}'
    spec.update(run_id=d.RUN,group_id='phase1-fixed-sigtime-practical-urban',
        display_name='固定 SIG_TIME 的真实 urban 配对增强、状态课程与判别间隔',
        description='从头 E30，普通拼接基线 low suburban/mid urban/low urban；独立检验 paired CE、源域状态课程、单侧默认头间隔和有条件 U→urban 软监督。',
        aliases=[],tags=['phase1','sigtime','practical','urban','paired_CE','source_only','default_head','old_gate_cos','scratch30'],
        authorization='Current user goal and steering: fixed SIG_TIME, scratchE30, lowurban>=60%, ordinaryconcat3scene baseline, automaticclean+sixpractical. No administrator use or unrelated workload changes.',
        parent_run_ids=[],notes=['18 registered rows are a maximum conditional matrix, not18 completed models.',
            'FirstseedC0/B0/C1/C2/C3; remaining fixedseeds andC4 only from preregistered source-only qualification.',
            'E50 reference is frozen diagnostic-only; no training inheritance. Own EMA is initialized from own scratch network.',
            'B0 ordinaryconcat andC1 separate equalweightCE are algebraically equivalent at fixedlogits. Main comparisons isolate sourcecurriculum andmargin.',
            'U TX remains hidden; V is health-only and never backpropagated. All7 fixed predictions precede truth.',
            'support/capsule_id/split_id/K are Phase2 fields, not applicable to thisPhase1 experiment; reuse existing VALIDATED_ONCE Phase1 IQ.'])
    spec['code'].update(commit='TO_BE_FIXED_BY_GIT_COMMIT',checkout=str(d.ROOT),cwd=(Path(d.PROJECT)/'releases'/d.RELEASE).as_posix(),environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python; nativeLinuxN607; Windowsnative localvalidation')
    spec['data'].update(dataset='ManySig.pkl',version='existing fixed source contract 392005',representation='equalized=1 center256 native RMS; added residualpost_syncnoEQ25MHz',
        contract_ref=d.SOURCE,physical_ids_ref=d.SOURCE+'#/role_ids',label_map_ref=d.CAPSULE+'/manifest.json#/classes',
        source_receivers=[1,3,4,6,8],target_receivers=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],
        source_days=[1,2,3],target_days=['2021_03_01','2021_03_08','2021_03_15','2021_03_23'],
        tx_sets_ref=d.CAPSULE+'/manifest.json#/classes',roles=dict(L_s=6300,U_s=56700,V=27000),train_ratio=.7,
        validation_ref=str(d.FORMAL_VIEWS/'manifest.json'),leo_config_ref='experiments/cvs_urban_sigtime/design.py and preserved formal manifest',
        support_query_ref=None,capsule_id=None,split_id=None)
    spec['checkpoint'].update(initialization='scratch matched per modelseed; no parentloaded',sources=[],
        contract_check_ref='each source/source_contract.json and native exact role-ID comparison',provenance_verdict='SCRATCH_ONLY_EXACT_SOURCE_CONTRACT',selection_rule='fixed final30; no epoch or target selection')
    spec['permissions'].update(claim_scope='Phase1 fixed practicalproxy DG on exposedbenchmark; not Phase2/realinorbit validation')
    command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_urban_sigtime.dispatch'
    spec['execution'].update(host='N607 ordinaryaccount',launch_owner=d.OWNER,gpu_policy='max3 total experiment processes/GPU including otherowners; capacity checked before launch',
        remote_run_root=str(d.BASE),remote_log_root=str(d.BASE/'logs'),local_artifact_root='E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014',
        launch_command=command,stop_rule='Protocol/role/budget/output collision/nonfinite/unlaunchable failure stops that row. Halt pending dispatch aftertechnicalfailure; preserve all outputs, allow other healthy rows tofinish. No performance-basedstop/restart.')
    spec['expected_artifacts']=['source_diagnostic/results.json','source_diagnostic/sample_diagnostics.jsonl.gz','eachrow/source/final_ssdg.pth','eachrow/source/step_metrics.jsonl.gz','eachrow/source/training_coverage.json','eachrow/prediction/predictions.npz','eachrow/scoring/scores.json','summary.json','resources.json','analysis.md','figures/*.png']
    spec['metrics_plan'].update(metric_names=['urban meanaccuracy','low urbanaccuracy>=.60','midurbanaccuracy','worsturbanRX','macroF1','clean','suburban','fixedreferenceclean-correctsatelliteretention','states/quality/CFO/kernel/AGC/boundary','classifiermargin','gates/gradients','effectivecoverage'],
        dimensions=['row','modelseed','scene','RX','day','TX','sourcevisibilitystate','sourcereceiverlock','sourcequality'],
        prediction_ref='eachrow/prediction aftersourcefreeze; seven fixedIQ arrays',scorer_ref='experiments.cvs_urban_sigtime.evaluate.score separateexecstage')
    spec['rows']=[]
    for r in d.rows():
        c=d.config(r);row=dict(row_id=r['row_id'],method=r['arm'],purpose='primary' if r['arm']=='C3' else 'comparison',
            config_ref=report+'/configs/'+r['row_id']+'.json',resolved_config_ref=c['output_root']+'/resolved_config.json',data_overrides={},
            seeds={k:c['seeds'][k] for k in ('model','split','data','augmentation','support','evaluation')},
            seed_notes='support is not applicable; receiver-trainingseed2026092701 and formalreceiverseed2027 recorded separately; validation2026101402,diagnostic2026101403 streams independent.',
            k=None,scenario=','.join(d.TRAIN_SCENES),optimizer='AdamW',lr=.0002,epochs=30,fl_rounds=None,budget_ref='scratch30=6660updates;label20+pseudo10; parent0,total30',
            output_root=c['output_root'],log_path=str(d.BASE/'logs'/(r['row_id']+'.log')),
            command=command+' --mode worker --row '+r['row_id'],expected_artifacts=spec['expected_artifacts'],config=c)
        spec['rows'].append(row)
        d.write(d.ROOT/report/'configs'/(r['row_id']+'.json'),c)
    d.write(d.ROOT/'experiments/cvs_urban_sigtime/registration.json',spec)
    d.write(d.ROOT/report/'experiment.json',spec)
    return spec

if __name__=='__main__':
    v=make();print(json.dumps(dict(run_id=v['run_id'],registered_rows=len(v['rows']),firstseed_initial_rows=5),ensure_ascii=False))
