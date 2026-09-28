"""Materialize this authorized launch registration; no remote side effects."""
import copy
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUN = '20260928-phase1-ir-eg-source-screen-s392005-r01'
BASE = '/home/szu2070436088/2510044040/CV-SincNet'
RELEASE = BASE + '/releases/ir_eg_source_screen_20260928_r01'
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
CONTRACT = BASE + '/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
out = REPO / 'automation_reports/CV-SincNet' / RUN
out.mkdir(parents=True, exist_ok=False)
original = json.loads((out.parent / '20260927-phase1-ir-eg-wisig-m3-r01/experiment.json').read_text(encoding='utf-8'))
doc = copy.deepcopy(original)
doc.update(run_id=RUN, display_name='IR-EG六方法单seed源侧筛查',
    description='SIM/EG/IR/G0/OR/encoder-off同物理曝光比较；六行从零训练，只访问源侧，E200为预算审计。',
    authorization='2026-09-28用户明确要求：发布实验。范围为已实现设计的首轮固定seed源侧筛查。',
    status='LOCAL_VERIFIED', parent_run_ids=[], replaces_run_id=None,
    preparation_run_id=original['run_id'])
doc['code'].update(commit='f11b785a42cc48a9e996f52468b627e546b080d1', cwd=RELEASE,
    environment='N607 Python 3.10.19; torch 2.1.0+cu121; local ssr-gpu torch 2.10.0+cu128')
doc['code']['commit_note'] = 'Implementation base; exact release commit supplied to launcher and effective_launch.json, recorded after commit.'
doc['data'].update(dataset='ManySig.pkl', version='existing N607 asset, 2359341461 bytes; no dataset transformation',
    contract_ref=CONTRACT, physical_ids_ref=CONTRACT + '#role_ids',
    label_map_ref='ManySig.pkl tx_list index order (6 classes); native loader preserves mapping',
    tx_sets_ref='ManySig.pkl tx_list (all six source-known classes)',
    validation_ref=CONTRACT, representation='equalized=1; 2x256; normalized center crop')
doc['checkpoint']['contract_check_ref'] = CONTRACT
doc['checkpoint']['selection_rule'] = 'Source V evidence only; final E200 is fixed budget audit, not convergence. No target inference in this launch.'
doc['permissions']['claim_scope'] = 'Source-only mechanism and same-exposure budget screening; no target performance or convergence claim.'
doc['permissions']['query_use'] = 'none'
doc['execution'].update(host='N607', launch_owner='/root thread 01a0e381-ac85-7b20-9a4e-d137ff563379',
    remote_run_root=BASE + '/runs/' + RUN, remote_log_root=BASE + '/logs/' + RUN,
    local_artifact_root=str(out), python=PYTHON, dataset_path=BASE + '/Dataset_WigSig/ManySig.pkl', source_contract=CONTRACT,
    launch_command=[PYTHON, RELEASE + '/tools/launch_source_screen.py', '--spec', RELEASE + '/configs/source_screen_20260928.json', '--commit', '<release Git HEAD>'],
    stop_rule='Fixed E200/44400 accepted-step budget. Fatal exception/nonfinite ordinary EG aborts only that row; no skipped batch, performance stop, automatic retry or target feedback.')
doc['expected_artifacts'] = ['source_contract.json', 'resolved_config.json', 'actions.jsonl', 'logs.jsonl', 'latest_ssdg.pth', 'final_ssdg.pth', 'completion.json']
doc['rows'] = [r for r in doc['rows'] if r['row_id'] in [m + '_s392005' for m in ('SIM','EG','IR','IR_G0','OR_EG','IR_ENCODER_OFF')]]
for row, gpu in zip(doc['rows'], [1,3,4,5,6,7]):
    rid = row['row_id']
    row.update(status='LOCAL_VERIFIED', gpu=gpu, purpose='single_seed_source_screen',
        output_root=doc['execution']['remote_run_root'] + '/' + rid,
        log_path=doc['execution']['remote_log_root'] + '/' + rid + '.log', expected_artifacts=doc['expected_artifacts'])
    row['resolved_config_ref'] = row['output_root'] + '/resolved_config.json'
    row['command'] = [PYTHON, '-u', RELEASE + '/code/scripts/train_response_games.py',
        '--config', RELEASE + '/configs/ir_rows/' + rid + '.json', '--output', row['output_root'],
        '--dataset', doc['execution']['dataset_path'], '--device', 'cuda:0', '--source-contract', CONTRACT, '--execute']
doc['release_smoke'] = dict(initialization='scratch', inherited_weights=False, target_access=False,
    purpose='Native initialized checkpoint serialization/restore plus source labeled batch finite forward, not scientific training',
    model_seed=392005, split_seed=392005, output=doc['execution']['remote_run_root'] + '/_release_smoke',
    launch_owner=doc['execution']['launch_owner'], gpu=1)
doc['notes'] = ['Remaining seeds and BR rows stay in preparation record; no dependent automatic launch.',
    'No convergence extension: continuation manifest remains unfrozen.',
    'Phase2 capsules, support/query and support seed are not applicable.',
    'Only source RX/day filtered tensors reach learner; monolithic pickle is the existing builder input.',
    'Single artifact archive SHA and remote compile; no new data validation or hash chain.']
for path in [out/'experiment.json', ROOT/'configs/source_screen_20260928.json']:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(out/'report.md').write_text('# IR-EG源侧筛查发布\n\n用户已授权发布实验。六行固定model seed392005，从零训练，保持L/U/V物理契约与DAOT/FastTrust原生配置。GPU1/3/4/5/6/7各一行，200epoch/44400接受步为预算审计，不代表收敛。\n\n实际配置与命令见experiment.json。三seed确认、BR效率和目标预测不在本次自动启动范围。\n\n发布状态：LOCAL_VERIFIED。启动后补充独立进程与日志读回。\n', encoding='utf-8')
(out/'events.jsonl').write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),status='LOCAL_VERIFIED',note='User authorized release; source screen registered'),ensure_ascii=False)+'\n', encoding='utf-8')
print(out)
