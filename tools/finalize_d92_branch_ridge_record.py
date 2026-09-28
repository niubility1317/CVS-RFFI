"""Archive complete BranchRidge evidence; no score access before both SCORED.

Dry-run by default. Reuses isolated, method-neutral cell arithmetic helpers,
never the previous method's fitting, cost assumptions or report generator.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from cvsrffi.d92_branch_ridge import FROZEN_CONFIG

RUNS = {c: f'20260929-phase2-d92-branch-ridge-repeat-{c}-m4-r01' for c in ('rx3','rx1')}
METHODS = ('D92', 'D92-BranchRidge-v1')
MARKER = '<!-- BRANCH_RIDGE_FINAL_ANALYSIS_20260929 -->'
# Separate module globals keep concurrent/other callers of the old finalizer intact.
_spec = importlib.util.spec_from_file_location('_branch_ridge_archive_arithmetic', ROOT/'tools/finalize_d92_mvridge_record.py')
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
_common.METHODS = METHODS
_common.MARKER = MARKER
BASELINES, KEY, METRICS, PRIMARY = _common.BASELINES, _common.KEY, _common.METRICS, _common.PRIMARY
check, read, table, preflight, persist = _common.check, _common.read, _common.table, _common.preflight, _common.persist
compare_baseline = _common.compare_baseline


def interpret(inputs, summary):
    result = _common.interpret(inputs, summary)
    result['limitations'] = [
        'Previously scored targets reused after authorized support-only branch diagnostics; not independent unseen-data confirmation.',
        'Shared query and overlapping support draws are correlated; no independent-replicate significance claim.',
        'One fixed full-support fit per episode for every K; no runtime OOF, grid or parameter selection. K1 has no independent same-class holdout evidence.',
        'Linear ridge scores are not calibrated probabilities.',
        'No automatic promotion, goal completion, parameter feedback or selective rerun.']
    result['fit_protocol'] = dict(feature_dim=736, full_fit_calls=4800, fold_fit_calls=0, oof=None,
        optimizer_steps=0, selection='fixed_no_selection', numeric_state_bytes='5896*C')
    return result


def baseline_binding(cohort, startup, original_startup, run_id):
    spec, original = startup['spec'], original_startup['spec']
    conf, prior = spec['confirmation'], original['confirmation']
    check(spec['run_id'] == run_id and original['run_id'] == BASELINES[cohort], 'Baseline run identity mismatch')
    check(tuple(conf[k] for k in ('candidate_method','candidate_folder','candidate_predictor','candidate_mode')) ==
        ('D92-BranchRidge-v1','branch_ridge','evaluate_d92_branch_ridge.py','d92_branch_ridge_registration'), 'BranchRidge alias mismatch')
    check(conf['candidate'] == FROZEN_CONFIG, 'Frozen BranchRidge formula mismatch')
    check(conf['capsule'] == prior['capsule'] and conf['reuse_validated_capsule_id'] == spec['data']['capsule_id'], 'Capsule binding mismatch')
    old_id = prior.get('reuse_validated_capsule_id') or original['data'].get('capsule_id')
    if old_id is not None: check(old_id == spec['data']['capsule_id'], 'Original capsule identity mismatch')
    rows, old_rows = spec['rows'], original['rows']
    check(len(rows) == len(old_rows) == 4 and len({r['row_id'] for r in rows}) == len({r['row_id'] for r in old_rows}) == 4, 'Four unique model bindings required')
    mapped = {r['row_id']:r for r in old_rows}
    for row in rows:
        old = mapped.get(row['row_id'])
        check(old is not None and row['seeds']['model'] == old['seeds']['model'] and row['source_root'] == old['source_root']
            and row['reuse_row_root'] == old['output_root'], 'Original baseline source/model binding mismatch')
        sha = row['expected_checkpoint_sha256']
        check(isinstance(sha,str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'Missing checkpoint SHA')
    check(sorted(r['seeds']['model'] for r in rows) == list(range(2026092701,2026092705)), 'Model seed matrix mismatch')
    return dict(status='VERIFIED',original_run_id=BASELINES[cohort],capsule_id=spec['data']['capsule_id'],
        model_checkpoints={r['row_id']:r['expected_checkpoint_sha256'] for r in rows})


def finite_tree(value):
    if isinstance(value,dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value,(list,tuple)):
        for item in value: finite_tree(item)
    elif type(value) in (int,float): check(math.isfinite(value),'Nonfinite cost evidence')


def costs(fit, startups, completions):
    """Bind the collector's two-cohort JSON; retain actual measured metadata."""
    finite_tree(fit)
    check(fit['status'] == 'VERIFIED' and fit['method'] == METHODS[1], 'Verified BranchRidge fit audit required')
    for key,value in dict(total_fits=4800,k1_fits=1200,full_fit_calls=4800,factorizations=4800,fold_fit_calls=0,
        optimizer_steps=0,query_rows_used_for_fit=0,source_rows_used_for_fit=0,new_source_payload_bytes=0,new_ground_statistics_bytes=0).items():
        check(type(fit[key]) is int and fit[key] == value, 'Fit count/access mismatch: '+key)
    check(fit['oof'] is None and fit['selection'] == 'fixed_no_selection' and fit['raw_traces_preserved'] is True
          and fit['parameter_feedback_forbidden'] is True and fit['model_incremental_transfer_bytes'] is None, 'Fit protocol/deployment mismatch')
    check(fit['feature_dim']==736 and fit['numeric_head_bytes_formula']=='8*(736+1)*C = 5896*C','Frozen numeric head dimension mismatch')
    run_ids={c:s['spec']['run_id'] for c,s in startups.items()}
    runs={r['run_id']:r for r in fit['runs']}
    check(len(runs) == len(fit['runs']) == 2 and set(runs) == set(run_ids.values()), 'Fit audit run mismatch')
    output={};by_sha={};all_models=[]
    for cohort,run_id in run_ids.items():
        run=runs[run_id];spec=startups[cohort]['spec'];models=run['models'];expected=900 if cohort=='rx3' else 300
        check(run['status']=='VERIFIED' and run['commit']==startups[cohort]['commit']==completions[cohort]['commit'], 'Fit commit/status mismatch')
        check(run['total_fits']==4*expected and run['k1_fits']==expected and len(models)==4
              and {m['row_id'] for m in models}=={r['row_id'] for r in spec['rows']}, 'Fit audit model coverage mismatch')
        rowmap={r['row_id']:r for r in spec['rows']}
        for model in models:
            row=rowmap[model['row_id']];resource=model['resources'];sha=row['expected_checkpoint_sha256']
            check(model['model_seed']==row['seeds']['model'] and resource['checkpoint_sha256']==sha, 'Fit checkpoint binding mismatch')
            check(model['fits']==model['full_fit_calls']==model['factorizations']==expected and model['k1_fits']==expected//4
                and model['status_counts']=={'CLOSED_FORM_SOLVED':expected}
                and model['solver_counts']=={'dual':expected}, 'Single full fit coverage mismatch')
            check(resource['new_source_payload_bytes']==resource['new_ground_statistics_bytes']==resource['support_cache_reuse_count']==0
                and resource['model_incremental_transfer_bytes'] is None, 'Source/transfer/reuse mismatch')
            count=resource['count'];check(type(count) is int and count>0,'Missing received count')
            check(resource['native_physical_forward_count']==resource['native_batch_calls']==count
                and resource['smoke_forward_count']==resource['smoke_batch_calls']==1
                and resource['native_total_physical_forward_count']==resource['native_total_batch_calls']==count+1
                and resource['feature_array_bytes']==2944*count, 'Feature forward/dimension mismatch')
            size=resource['existing_model_file_bytes'];check(type(size) is int and size>0,'Missing model package bytes')
            check(sha not in by_sha or by_sha[sha]==size,'Same SHA model byte mismatch');by_sha[sha]=size
            for key in ('head_bytes','persistent_state_bytes'):
                stat=model['state_bytes'][key]
                check(stat['min']==5896*6 and stat['max']==5896*26 and stat['min']<=stat['mean']<=stat['max']
                    and math.isclose(stat['total'],stat['mean']*expected,rel_tol=1e-10), '736D numeric state bytes mismatch')
            check(model['state_bytes']['head_bytes']==model['state_bytes']['persistent_state_bytes'],'W+b byte mismatch')
            for collection in ('timing','fit_measurements'):
                for key,stat in model[collection].items():
                    check(all(type(stat[k]) in (int,float) and stat[k]>=0 for k in ('min','mean','max','total'))
                        and stat['min']<=stat['mean']+1e-12<=stat['max']+1e-12
                        and math.isclose(stat['total'],stat['mean']*expected,rel_tol=1e-9,abs_tol=1e-10), 'Invalid measured statistics: '+key)
            d=model['fit_measurements'];check(math.isclose(d['loss_total']['total'],d['loss_data']['total']+d['loss_ridge']['total'],rel_tol=1e-10), 'Objective component mismatch')
        check(run['run_wall_seconds']>=0 and math.isclose(run['run_wall_seconds'],run['run_finished']-run['run_started'],abs_tol=1e-8), 'Cohort wall clock mismatch')
        output[cohort]=run;all_models.extend(models)
    check(len(by_sha)==fit['unique_model_count']==4 and sum(by_sha.values())==fit['existing_unique_model_file_bytes'], 'Unique model byte/count mismatch')
    for key in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds'):
        check(math.isclose(fit['summed_process_work_seconds'][key],sum(m['timing'][key]['total'] for m in all_models),rel_tol=1e-10), 'Process work total mismatch')
    for field,key in [('received_feature_array_bytes','feature_array_bytes'),('received_feature_file_bytes','feature_file_bytes'),
        ('received_native_forward_count','native_physical_forward_count'),('synthetic_smoke_forward_count','smoke_forward_count'),
        ('actual_total_native_forward_count','native_total_physical_forward_count'),
        ('actual_total_native_batch_calls','native_total_batch_calls'),('received_registry_array_bytes','registry_array_bytes')]:
        check(fit[field]==sum(m['resources'][key] for m in all_models),'Joint resource total mismatch: '+field)
    check(math.isclose(fit['summed_extraction_seconds'],sum(m['resources']['extraction_timing']['total_seconds'] for m in all_models),rel_tol=1e-10),'Extraction total mismatch')
    keys=set().union(*(m['resources']['extraction_timing'] for m in all_models))
    check(set(fit['summed_extraction_stage_seconds'])==keys,'Extraction stage coverage mismatch')
    for key in keys:
        check(math.isclose(fit['summed_extraction_stage_seconds'][key],sum(m['resources']['extraction_timing'][key] for m in all_models),rel_tol=1e-10),'Extraction stage sum mismatch')
    check(math.isclose(fit['joint_wall_span_seconds'],max(r['run_finished'] for r in runs.values())-min(r['run_started'] for r in runs.values()),abs_tol=1e-8),'Joint wall span mismatch')
    check(math.isclose(fit['summed_cohort_wall_seconds'],sum(r['run_wall_seconds'] for r in runs.values()),abs_tol=1e-8),'Cohort wall sum mismatch')
    return output


def cost_text(run):
    models=run['models'];resources=[m['resources'] for m in models]
    total=lambda key:sum(m['timing'][key]['total'] for m in models)
    return (f"共 {run['total_fits']} 次全 support 解析拟合，含 {run['k1_fits']} 次 K1；每任务一次分解，所有 K 均无运行时 OOF、网格或选参。"
        f"拟合调用累计 {total('fit_call_seconds'):.3f} s，query 打分累计 {total('query_score_seconds'):.3f} s，"
        f"提取累计 {sum(r['extraction_timing']['total_seconds'] for r in resources):.3f} s；本 cohort 墙钟 {run['run_wall_seconds']:.3f} s，累计进程时间不等于墙钟。"
        f" received forward {sum(r['native_physical_forward_count'] for r in resources)} 次，合成 smoke {sum(r['smoke_forward_count'] for r in resources)} 次分列。"
        "数值头为 W[736,C]+b[C]，5896C B，C=6 至 26 对应 35376 至 153296 B，不含 Python/registry/audit 开销。"
        f"最大记录梯度残差 {max(m['fit_measurements']['gradient_norm']['max'] for m in models):.3g}；没有优化器更新，不作迭代收敛声明。"
        "新增 source payload 与 ground statistics 均为 0；模型文件是已有完整训练 checkpoint 包，不是最小推理包，部署状态和增量模型传输未知。"
        "RSS 为单进程高水位，不能相加冒充并发峰值；所有实测分量和字节见产物索引。")


def prepare(workspace, runs=None, fit_audit=None):
    runs=dict(RUNS if runs is None else runs)
    check(set(runs)=={'rx3','rx1'} and len(set(runs.values()))==2 and all(isinstance(v,str) and v and '/' not in v and '\\' not in v and v not in ('.','..') for v in runs.values()),'Two distinct safe run IDs required')
    base=Path(workspace)/'automation_reports/CV-SincNet';folders={c:base/r for c,r in runs.items()}
    combined=folders['rx3']/'results/combined_rx4'
    fit_path=Path(fit_audit) if fit_audit is not None else folders['rx3']/'results/fit_audit.json'
    preflight([(combined/'interpretation_audit.json',None)]+[(p/'results/artifacts.json',None) for p in folders.values()],
        [(p/'report.md',None) for p in folders.values()]+[(combined/'report.md',None)])
    completions={c:read(p/'results/complete.json') for c,p in folders.items()}
    check(all(v['status']=='SCORED' for v in completions.values()),'Both cohorts must be SCORED before any score access')
    startups={c:read(p/'results/startup.json') for c,p in folders.items()}
    fit=read(fit_path);by_cohort=costs(fit,startups,completions)
    inputs={}
    for c,folder in folders.items():
        result=folder/'results';summary=read(result/'summary/summary.json')
        check(read(result/'arithmetic_audit.json')['status']=='VERIFIED','Independent complete arithmetic audit required')
        spec=startups[c]['spec'];d=spec['data'];registered=dict(model_seed=[r['seeds']['model'] for r in spec['rows']],
            receiver=d['target_receivers'],scenario=d['scenarios'],k=d['k'],new_count=d['new_class_counts'],support_seed=d['support_seeds'])
        check(summary['matrix']==registered,'Summary differs from preregistered matrix')
        original=base/BASELINES[c]/'results'
        binding=baseline_binding(c,startups[c],read(original/'startup.json'),runs[c])
        inputs[c]=dict(summary=summary,original=original,binding=binding,result=result)
    for c,data in inputs.items():
        data['scores']=read(data['result']/'scores.json')
        check(len(data['scores']['results'])==completions[c]['records']==(7236 if c=='rx3' else 2412),'Full score count mismatch')
        data['baseline_audit']=dict(data['binding'],**{k:v for k,v in compare_baseline(data['scores'],read(data['original']/'scores.json'),data['summary']['matrix']).items() if k!='status'})
    audit=interpret(inputs,read(combined/'summary.json'))
    audit['original_baseline_verification']={c:d['baseline_audit'] for c,d in inputs.items()}
    audit['fit_audit']=str(fit_path);audit['joint_cost']={k:v for k,v in fit.items() if k!='runs'}
    writes=[(combined/'interpretation_audit.json',audit)];appends=[]
    for c,folder in folders.items():
        data=inputs[c];cost=by_cohort[c]
        artifact=dict(run_id=runs[c],status='ANALYZED',execution_status='SCORED',evidence_status='VERIFIED',candidate=METHODS[1],
            candidate_promoted=False,goal_complete=False,joint_preregistered_guard_pass=audit['preregistered_guard_pass'],
            records=len(data['scores']['results']),release_commit=completions[c]['commit'],cost=cost,fit_audit=str(fit_path),
            original_baseline_verification=data['baseline_audit'],per_k=audit['per_cohort_k'][c],
            raw_scores=str(data['result']/'scores.json'),raw_predictions=f"/home/szu2070436088/2510044040/CV-SincNet/runs/{runs[c]}/<row_id>/branch_ridge/predictions.jsonl",
            raw_fit_logs='fit_logs/<row_id>/',raw_artifacts_preserved=True,model_already_deployed=None,model_incremental_transfer_bytes=None,
            old_guard_scope='All paired cells including old-only; new/H and joint old exclude old-only.',claim_scope=audit['claim_scope'],
            interpretation_audit=str(combined/'interpretation_audit.json'),limitations=audit['limitations'])
        writes.append((folder/'results/artifacts.json',artifact))
        text=f"\n\n{MARKER}\n\n## 最终结果、核验与成本\n\n当前 SCORED / ANALYZED，证据 VERIFIED；原文保留运行历史。release commit：{completions[c]['commit']}。完整 {len(data['scores']['results'])} 条评分，原始 D92/DG 逐记录保持一致。\n\n"
        text+=audit['conclusion']+'\n\n下表为候选减 D92，单位百分点；前三列仅新类存在任务，末列含 old-only。\n\n'+table(audit['per_cohort_k'][c])+'\n\n'+cost_text(cost)
        text+='\n\n验收采用完整四 RX 等单元权重，不能以本 cohort 替代联合结论。本轮为已评分数据透明重复基准，开发使用授权 support 诊断；没有新的独立泛化证据。结果不回流选参或选择性重跑。\n\n见[产物索引](results/artifacts.json)、[本 cohort 汇总](results/summary/report.md)。\n'
        appends.append((folder/'report.md',text))
    text=f"\n\n{MARKER}\n\n## 独立解释核验\n\n完整 9648 条记录、4800 对任务及 48 条 DG。独立复核 {audit['compared_summary_values']} 个汇总数值，最大差 {audit['maximum_absolute_error']:.3g}。每 K 联合任务为 720 rx3 + 240 rx1，3:1 等单元；旧类保护另含 old-only，共 1200 对。H 为单元 H 的均值。\n\n"
    text+=table(audit['per_k'])+'\n\n单位为百分点。'+audit['conclusion']+'\n\n| K | ΔH 正向模型数 | Δ新类正向模型数 | ΔH 模型范围（百分点） |\n|---|---:|---:|---:|\n'
    for row in audit['per_k']:
        h,n=row['seed_consistency']['harmonic_mean'],row['seed_consistency']['new_accuracy']
        text+=f"| {row['k']} | {h['improved']}/4 | {n['improved']}/4 | {h['range_pp'][0]:+.3f} 至 {h['range_pp'][1]:+.3f} |\n"
    text+=f"\n全部 4800 次任务各一次 full-support fit，无运行时 OOF、搜索或优化器步骤；736 维固定表示，数值头 5896C B。联合墙钟跨度 {fit['joint_wall_span_seconds']:.3f} s，累计 fit 调用 {fit['summed_process_work_seconds']['fit_call_seconds']:.3f} s，两者口径不同。4 个唯一完整训练 checkpoint 包合计 {fit['existing_unique_model_file_bytes']} B，不是最小推理包，新增传输未知；新增 source payload=0。\n"
    text+='\n完整 per-K、四模型 seed、两 cohort 和 K×new_count 见[解释审计](interpretation_audit.json)，真实拟合/提取/字节/峰值全部保留在原 collector JSON 及两 run 的 artifacts.json。独立算术审计已覆盖混淆矩阵，本工具不重复该原分析。原始基线逐记录不变；不将 ridge 分数称为校准概率，不把共享 query/重叠 support 抽样当作独立重复，不作新独立泛化或自动晋级结论。\n'
    appends.append((combined/'report.md',text));preflight(writes,appends)
    return writes,appends


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=Path('E:/type10-7'))
    p.add_argument('--fit-audit',type=Path);p.add_argument('--rx3-run',default=RUNS['rx3']);p.add_argument('--rx1-run',default=RUNS['rx1']);p.add_argument('--write',action='store_true')
    a=p.parse_args();writes,appends=prepare(a.workspace,dict(rx3=a.rx3_run,rx1=a.rx1_run),a.fit_audit)
    if a.write:persist(writes,appends)
    print(json.dumps(dict(status='VERIFIED' if a.write else 'PREFLIGHT_VERIFIED',outputs=[str(p) for p,_ in writes],
        reports=[str(p) for p,_ in appends],preregistered_guard_pass=writes[0][1]['preregistered_guard_pass']),ensure_ascii=False))


if __name__=='__main__':main()
