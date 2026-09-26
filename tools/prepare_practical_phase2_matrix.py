"""Freeze matched evaluation rows before any target predictions exist."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN='20260927-phase2-baselines-practical-manytx-m5-r01'
SOURCE='20260927-phase1-baselines-practical-manysig-m5-r01'
DATA='20260927-phase2-practical-data-manytx-s2026092705-r01'
REMOTE='/home/szu2070436088/2510044040/CV-SincNet'


def main():
    source=json.loads((ROOT/f'automation_reports/CV-SincNet/{SOURCE}/experiment.json').read_text(encoding='utf-8'))
    spec=copy.deepcopy(source)
    release=REMOTE+'/releases/phase2_practical_baselines_20260927_r01'
    runroot=REMOTE+'/runs/'+RUN
    logs=REMOTE+'/logs/'+RUN
    configs=ROOT/'configs/phase2_practical_baselines_20260927'
    configs.mkdir(exist_ok=False)
    spec.update(run_id=RUN,group_id='phase2-practical-baselines-matched',stage='Phase2',
        display_name='Phase2外部对比：固定DG、support NCM及POSTER/RadioNet原生微调',
        description='40个源模型分别预测共享2100个split；DG每RX/场景仅一次；作者微调只读support。',
        tags=['phase2','comparison','residual_noeq','poster','radionet'],status='PLANNED')
    spec['code'].update(commit='release_commit_recorded_in_launch',checkout=str(ROOT),cwd=release)
    spec['data'].update(dataset=REMOTE+'/runs/'+DATA+'/capsule',version='residual-noeq-ba667eee4fb061055e4c08b5',
        capsule_id='residual-noeq-ba667eee4fb061055e4c08b5',split_id='all2100 validated splits',
        contract_ref=REMOTE+'/runs/'+DATA+'/capsule/manifest.json',
        physical_ids_ref=REMOTE+'/runs/'+DATA+'/capsule/received.npz',
        label_map_ref=REMOTE+'/runs/'+DATA+'/capsule/splits',support_query_ref=REMOTE+'/runs/'+DATA+'/capsule/splits',
        validation_ref=REMOTE+'/runs/'+DATA+'/capsule/manifest.json')
    spec['checkpoint'].update(initialization='matched_source_final_epoch200',sources=[REMOTE+'/runs/'+SOURCE],
        provenance_verdict='runtime_requires_exact_role_ids_scratch_only_no_target_contact',
        selection_rule='fixed epoch200 for every seed; no source-best or target selection')
    spec['permissions'].update(regime='inductive_support_only',query_use='read-only pointwise prediction; no adaptation or selection',
        claim_scope='6old+20new ManyTx target; source ManySig. NCM is a common registration extension; RadioNet DF fine-tuning only.')
    spec['execution'].update(launch_owner='codex/root/practical-phase2-baselines-20260927',
        gpu_policy='one Phase2 lane per GPU; concurrent source lane means at most2 training jobs/GPU',
        remote_run_root=runroot,remote_log_root=logs,local_artifact_root=f'automation_reports/CV-SincNet/{RUN}')
    spec['execution']['launch_command']=f"{spec['code']['environment']} tools/launch_practical_phase2_baselines.py --spec automation_reports/CV-SincNet/{RUN}/experiment.json --detach --commit RELEASE_COMMIT"
    spec['execution']['stop_rule']='No retries. Dependency technical failure blocks its lane; preserve all predictions; no performance-based decisions.'
    rows=[]
    for source_row in source['rows']:
        row=copy.deepcopy(source_row)
        sid=row['row_id']
        method=json.loads((ROOT/source_row['config_ref']).read_text(encoding='utf-8'))['method']
        config_ref=f'configs/phase2_practical_baselines_20260927/{sid}.json'
        cfg=dict(method=method,model_seed=row['seeds']['model'],device='cuda:0',
            source_root=REMOTE+'/runs/'+SOURCE+'/'+sid,
            source_contract=REMOTE+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
            capsule=REMOTE+'/runs/'+DATA+'/capsule',output_root=runroot+'/'+sid)
        (ROOT/config_ref).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        row.update(purpose='DG and support-based adaptation/registration',config_ref=config_ref,
            resolved_config_ref=runroot+'/'+sid+'/resolved_config.json',output_root=runroot+'/'+sid,log_path=logs+'/'+sid+'.log',
            source_root=cfg['source_root'],source_state=REMOTE+'/runs/'+SOURCE+'/state.json',
            optimizer='author Adam eps1e-7' if method in ('poster','radionet') else None,
            lr=.001 if method in ('poster','radionet') else None,
            epochs=10 if method=='poster' else (30 if method=='radionet' else None),
            budget_ref=config_ref,k=[1,5,10,20],scenario=['practical_high','practical_mid','practical_low_urban'],
            expected_artifacts=['predictions.jsonl','predictions_complete.json','checkpoint_provenance.json','support_smoke.json'])
        row['seeds'].update(split=2026092705,data=2026092705,augmentation=2026092707,support=None,evaluation=2026092706)
        row['seed_notes']='Model initialization/FT seed matched across methods; five support draws2026092711..2026092715 are separately identified by split. NCM/DG deterministic. Historical392005 reported separately.'
        row['command']=f"{spec['code']['environment']} tools/run_practical_phase2_baseline.py --config {config_ref}"
        rows.append(row)
    spec['rows']=rows
    spec['expected_artifacts']=['state.json','dispatcher_complete.json','scored_results.json','*/predictions_complete.json']
    spec['metrics_plan'].update(metric_names=['accuracy','macro_accuracy','old_accuracy','new_accuracy','harmonic_mean'],
        dimensions=['model_seed','support_seed','receiver','scenario','K','registered_classes','mode'])
    spec['notes']=['All40 source rows must freeze predictions before independent scorer opens truth; no target-based changes/retries.',
        'DG21 predictions per row; support NCM2100 per row; author fine-tuning additional2100 for each POSTER/RadioNet row.',
        'POSTER uses author-code three-Dense replacement,10epochs,batch256. RadioNet uses DF head replacement,30epochs,batch128. BothAdam.001.',
        'Source budget200epochs is explicitly matched, rather than an exact paper source-budget reproduction.',
        'Other external native DA/new-class algorithms are not represented by the common NCM extension.',
        'No augmentation or extra equalization in Phase2: received IQ is fixed by capsule.']
    (ROOT/'practical_phase2_baselines_spec_20260927.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
