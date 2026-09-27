"""Materialize the user-approved five-seed original residual recipe, without target feedback."""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SUB = REPO / 'experiments/adv3b02_xuc'
RUN = '20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
RELEASE = PROJECT + '/releases/cvs_rc4_matched_20260927_r01'
SEEDS = [392005, 2026092701, 2026092702, 2026092703, 2026092704]

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) == value:
            return
        raise FileExistsError(path)
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    path.write_text(text, encoding='utf-8')
    assert json.loads(path.read_text(encoding='utf-8')) == value

def main():
    base_path = SUB / 'configs/rc4_practical_residual_noeq_20260918.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))
    spec = json.loads(Path('E:/type10-7/automation_reports/CV-SincNet/20260927-phase1-baselines-practical-manysig-m5-r01/experiment.json').read_text(encoding='utf-8'))
    template = copy.deepcopy(spec['rows'][0])
    spec.update(run_id=RUN, group_id='phase1-cvs-daot-rc4-original-practical-matched',
                display_name='CVS：DAOT＋FastTrust-RC4原始residual_noeq五种子实验',
                description='用户确认的原始DAOT A1＋FastTrust-RC4。与对比共享物理L/U/V和五个模型种子，200轮final-only；关闭历史周期目标评分。后续接D92 E0真实256维v2。',
                kind='cvs', authorization='2026-09-27用户：跑CVS实验；注意版本；是这个，以及后续的D92。',
                tags=['cvs','daot','fasttrust_rc4','residual_noeq','source_only','d92_parent'],
                rows=[], status='PLANNED')
    spec['code'].update(commit='RELEASE_COMMIT', checkout=str(REPO), cwd=RELEASE,
                        base_recipe='experiments/adv3b02_xuc/configs/rc4_practical_residual_noeq_20260918.json',
                        native_code_base='b63734157')
    spec['data']['leo_config_ref'] = 'experiments/adv3b02_xuc/code/cvsrffi/practical_adapter.py'
    spec['data']['label_map_ref'] = 'each row/source_contract.json#classes'
    spec['execution'].update(launch_owner='codex/root/cvs-daot-rc4-matched-20260927',
          remote_run_root=PROJECT+'/runs/'+RUN, remote_log_root=PROJECT+'/logs/'+RUN,
          local_artifact_root='automation_reports/CV-SincNet/'+RUN,
          gpu_policy='five separate GPU lanes selected from current free slots; at most2 total jobs/GPU',
          launch_command='python experiments/adv3b02_xuc/tools/publish_rc4_matched.py --output local_artifacts/cvs_matched_20260927',
          stop_rule='Nonzero exit/protocol violation/missing final checkpoint stops affected row; no automatic retry and no performance stop.')
    spec['checkpoint']['selection_rule'] = 'fixed final_ssdg.pth epoch200; no old checkpoint, teacher or ground bundle'
    spec['permissions']['claim_scope'] = 'Original native CVS recipe; matched physical roles/seeds/epochs, unequal optimizer-step budget and augmentation draws. Historical392005 reported separately from fresh4.'
    spec['expected_artifacts'] = ['each row/startup.json','each row/initialization.json','each row/source_contract.json','each row/resolved_config.json','each row/final_ssdg.pth','each row/completion.json','launch.json']
    spec['metrics_plan']['prediction_ref'] = 'independent final evaluator, existing clean/satellite VALIDATED_ONCE capsule'
    spec['next_stage'] = {'method':'D92 E0 identity160+FFT96 true256 v2', 'status':'PREPARING', 'ground_bundle':'must derive from this run final model and authorized source aggregates; old bundles forbidden'}
    for seed in SEEDS:
        doc = copy.deepcopy(base)
        row_id = 'cvs-daot-rc4-s'+str(seed)
        doc.update(id=row_id, seed=seed, status='PREPARED', reference='Original residual_noeq method, final-only source training; target evaluation separated')
        doc['options'].update({'--seed':str(seed),'--a1_periodic_target_start':'0',
                              '--a1_periodic_target_interval':'0','--a1_final_weak_reference':'false'})
        cfg = 'configs/matched_20260927/'+row_id+'.json'
        write(SUB/cfg, doc)
        row = copy.deepcopy(template)
        out = PROJECT+'/runs/'+RUN+'/'+row_id
        row.update(row_id=row_id, method='DAOT_A1+FastTrust_RC4_original_residual_noeq',
                   purpose='historical_seed_replication' if seed==392005 else 'fresh_model_seed',
                   gpu=None, config_ref='experiments/adv3b02_xuc/'+cfg, resolved_config_ref=out+'/resolved_config.json',
                   seeds=dict(model=seed,split=392005,data=None,augmentation=seed,support=None,evaluation=None),
                   seed_notes='Fixed deterministic physical split. Native augmentation generators derive from model seed/epoch/batch; receiver hardware seed2027. No target support/evaluation during training.',
                   optimizer='AdamW; native FastTrust schedule', lr=.0002,
                   budget_ref='native muse_epoch_basis=unlabeled_loader; Ubatch256; actual steps logged; not compute-matched to50-step baselines',
                   output_root=out, log_path=PROJECT+'/logs/'+RUN+'/'+row_id+'.train.log',
                   command=' '.join([spec['code']['environment'],'-u',RELEASE+'/code/scripts/train_rc4_matched.py','--config',RELEASE+'/'+cfg,'--dataset',spec['data']['dataset'],'--output',out,'--source-contract',spec['data']['contract_ref'],'--run-id',RUN,'--commit','RELEASE_COMMIT']),
                   expected_artifacts=['final_ssdg.pth','completion.json','startup.json','resolved_config.json'])
        spec['rows'].append(row)
    write(SUB/'configs/matched_20260927/experiment_spec.json',spec)
    smoke = (SUB/'code/scripts/check_rc4_practical.py').read_text(encoding='utf-8')
    smoke = smoke.replace('from scripts.train_rc4_practical import', 'from scripts.train_rc4_matched import')
    smoke = smoke.replace("'unused-contract','unused-inputs','unused-truth','smoke'", "'unused-contract','','','smoke'")
    (SUB/'code/scripts/check_rc4_matched.py').write_text(smoke,encoding='utf-8')
    print(SUB/'configs/matched_20260927/experiment_spec.json')

if __name__ == '__main__':
    main()
