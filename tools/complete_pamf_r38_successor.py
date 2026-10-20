"""Collect and deliver the existing R13 run; never launch or modify remote jobs."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from experiments.pamf_r38_successor import design as d
from tools.analyze_pamf_r38_successor_results import completed_metrics, METHOD


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def remote(local, code):
    result = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', str(local/'tools/n607_ssh_config'),
        '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607', 'python3 -'],
        input=code.encode('utf-8'), capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    return json.loads(result.stdout.decode('utf-8'))


def collect(local, output):
    if (output/'analysis.json').exists():
        return json.loads((output/'analysis.json').read_text(encoding='utf-8'))
    startup = json.loads((output/'deployment.json').read_text(encoding='utf-8'))['startup']
    code = '\n'.join(k+'='+repr(v) for k, v in dict(RUN=startup['run'], PID=startup['pid'],
        METHOD=METHOD, ARMS=d.ARMS, R17=d.PARENT_RUN, R38=d.COMPACT_PARENT_RUN).items())+'\n'+'''
import json, os
from pathlib import Path
assert os.geteuid()!=0
root=Path(RUN)
def read(path):return json.loads(path.read_text())
failure=read(root/'failure.json') if (root/'failure.json').exists() else None
if failure or not (root/'complete.json').exists():
 print(json.dumps(dict(status='FAILED_TECHNICAL' if failure else 'PENDING',failure=failure,stages=read(root/'stages.json'))))
else:
 assert read(root/'launch.json')['commit']==METHOD
 workers=read(root/'workers.json')['workers'];live=[]
 for pid in {PID,*[w['pid'] for w in workers]}:
  path=Path('/proc')/str(pid)
  try:
   state=(path/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(path/'cmdline').read_bytes().decode().split(chr(0))
   if state!='Z' and RUN in argv and 'experiments.pamf_r38_successor.runner' in argv:live.append(pid)
  except FileNotFoundError:pass
 rows={}
 for arm in ARMS:
  row=root/arm;source=row/'native_map_prior';target=row/'target_confirmation/native_map_prior'
  log=root/'logs'/('worker-'+arm+'.log')
  rows[arm]=dict(training_records=[json.loads(line) for line in (row/'training.jsonl').read_text().splitlines() if line.strip()],
   training_complete=read(row/'training_complete.json'),initialization=read(row/'initialization.json'),
   matched_profile=read(row/'matched_batch1_profile.json'),source_summary=read(source/'summary.json'),
   source_prediction_complete=read(source/'prediction/complete.json'),target_summary=read(target/'summary.json'),
   target_prediction_complete=read(target/'prediction/complete.json'),complete=read(row/'complete.json'),
   failure=read(row/'failure.json') if (row/'failure.json').exists() else None,
   assessment=read(row/'assessment.json'),checkpoint_bytes=(source/'final.pt').stat().st_size,
   checkpoint_path=str(source/'final.pt'),stdout=log.read_text(errors='replace'))
 baselines={name:{role:read(root.parent/run/arm/('target_confirmation/' if role=='target' else '')/'native_map_prior/summary.json')
  for role in ('source','target')} for name,run,arm in (('R17',R17,'unroll_unshared'),('R38',R38,'dense_half'))}
 print(json.dumps(dict(run_id=root.name,method_commit=METHOD,complete=read(root/'complete.json'),failure=failure,
  actual_live_owned_PIDs=live,rows=rows,baselines=baselines,controller_stdout=Path(RUN).parent.parent.joinpath('logs',root.name,'runner.log').read_text(errors='replace'))))
'''
    value = remote(local, code)
    if value.get('status') in ('PENDING', 'FAILED_TECHNICAL'):
        print(json.dumps(value))
        return None
    folder = output/'collections'
    folder.mkdir(exist_ok=True)
    for arm, row in value['rows'].items():
        row['stdout_findings'] = [line for line in row['stdout'].splitlines()
            if re.search(r'Traceback|out of memory|killed|warning|\b(?:nan|inf)\b', line, re.I)]
    write(folder/(str(time.time_ns())+'.json'), value)
    analysis = completed_metrics(value)
    write(output/'analysis.json', analysis)
    print(json.dumps(dict(status='COLLECTED_COMPLETE', accepted_rows=analysis['accepted_rows'],
                         accepted_IQ_rows=analysis['accepted_IQ_rows'])))
    return analysis


def retrieve(local, output):
    saved = output/'model_artifacts.json'
    if saved.exists():
        return json.loads(saved.read_text(encoding='utf-8'))
    collections = sorted((output/'collections').glob('*.json'))
    raw = json.loads(collections[-1].read_text(encoding='utf-8'))
    import torch
    from experiments.pamf_r38_successor.runtime import load_checkpoint
    torch.set_num_threads(2)
    artifacts = {}
    for arm in d.ARMS:
        folder = output/'models'/arm
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'final.pt'
        row = raw['rows'][arm]
        if not path.exists():
            subprocess.run(['C:/Program Files/Git/usr/bin/scp.exe', '-F', str(local/'tools/n607_ssh_config'),
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607:'+row['checkpoint_path'], str(path)],
                check=True, timeout=55)
        if path.stat().st_size != row['checkpoint_bytes']:
            raise ValueError('Incomplete retrieved artifact; preserve and reconcile transfer')
        model, payload = load_checkpoint(path, 'cpu')
        if (payload['arm'], payload['epoch'], payload['updates']) != (arm, 10, 2470):
            raise ValueError('Wrong final checkpoint')
        if payload['optimizer'] is not None or payload['training_F_access'] != d.TASK[arm]:
            raise ValueError('Wrong inference payload or training permission')
        components = dict(coarse=sum(p.numel() for p in model.frozen.coarse.parameters()),
            action=sum(p.numel() for p in model.frozen.estimator.parameters()),
            prior=sum(p.numel() for p in model.prior.parameters()))
        macs = [0]
        def count(module, inputs, result):
            if isinstance(module, torch.nn.Conv1d):
                macs[0] += result.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
            elif isinstance(module, torch.nn.Linear):
                macs[0] += result.numel()*module.in_features
        hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear))]
        with torch.no_grad():
            restored = model(torch.zeros((1, 2, 256), dtype=torch.float32))
        for hook in hooks:
            hook.remove()
        if restored.shape != (1, 2, 256) or not torch.isfinite(restored).all() or sum(components.values()) != d.PARAMETERS[arm]:
            raise ValueError('Invalid full-chain inference artifact')
        artifacts[arm] = dict(path=str(path), bytes=path.stat().st_size, parameters=d.PARAMETERS[arm],
            components=components, FP32_parameter_bytes=4*d.PARAMETERS[arm], conv_linear_MACs_per_IQ=macs[0],
            strict_load=True, synthetic_zero_IQ_output_finite=True, actual_IQ_reads=0, optimizer_included=False)
    result = dict(status='VERIFIED', source_run=d.RUN, method_commit=METHOD, models=artifacts,
                  MAC_scope='Conv1d and Linear only; excludes FFT/complex CG/normalization/activation/memory traffic')
    write(saved, result)
    return result


def report(local, output, analysis, artifacts):
    folder = local/'automation_reports/CV-SincNet'/d.RUN
    report_path = folder/'report.md'
    body = report_path.read_text(encoding='utf-8').split('\n## R13 完整结果交付\n')[0]
    body = body.replace('状态：QUEUE_VERIFIED。', '状态：ANALYZED。', 1)
    lines = ['\n## R13 完整结果交付\n',
        '六行均完成固定 E10/2470、source 1800×7 和 target 168000×7；每行全部预测固定后独立 truth-last。'
        '实际所属进程已退出，六份完整恢复器已取回并严格加载。全部负结果保留，结果不回流训练或重跑。\n',
        '|版本|参数|较 R38 减少|最大下降 R38/R17（pp）|low urban（%）|双参考 1pp|训练 F 知识|',
        '|---|---:|---:|---:|---:|---|---|']
    for arm, row in analysis['rows'].items():
        drops = [max(-v['accuracy_delta_pp'] for v in row['acceptance'][b]['views'].values()) for b in ('R38', 'R17')]
        lines.append(f"|{arm}|{row['parameters']}|{row['reduction_from_R38_percent']:.4f}%|{drops[0]:.6f}/{drops[1]:.6f}|"
                     f"{row['performance_optimization']['low_urban_accuracy']:.6f}|{row['accepted_both']}|{row['training_F_access']}|")
    lines += ['\n1pp 通过仅表示满足退化容差。IQ 行训练不加载 F；task 行使用监督训练的冻结 F 知识，须单独标注。'
              'low urban≥70% 和其他五场景距 raw clean≤5pp 分别记录，不以压缩验收替代。单 seed、已暴露基准，不称独立盲测或全局最小。\n',
              '|版本|场景|准确率（%）|相对 R38（pp）|相对 R17（pp）|相对同预算原宽度（pp）|净纠错|',
              '|---|---|---:|---:|---:|---:|---:|']
    for arm, row in analysis['rows'].items():
        for view in d.VIEWS:
            item = row['target']['views'][view]
            lines.append(f"|{arm}|{view}|{item['accuracy']:.6f}|{row['target_vs_R38']['views'][view]['accuracy_delta_pp']:+.6f}|"
                f"{row['target_vs_R17']['views'][view]['accuracy_delta_pp']:+.6f}|"
                f"{row['matched_same_budget_control']['target']['views'][view]['accuracy_delta_pp']:+.6f}|{item['net_repaired']:+d}|")
    lines += ['\n|版本|coarse/action/prior 参数|Conv+Linear MAC/IQ|文件字节|R38/本版中位 ms|resident/增量 peak 字节|',
              '|---|---|---:|---:|---:|---:|']
    for arm, row in analysis['rows'].items():
        model = artifacts['models'][arm]
        profile = row['profile']['results']
        base, current = profile['parent_R38_794529'], profile['compact']
        lines.append(f"|{arm}|"+'/'.join(str(v) for v in model['components'].values())+
            f"|{model['conv_linear_MACs_per_IQ']}|{model['bytes']}|{base['median_ms']:.4f}/{current['median_ms']:.4f}|"
            f"{current['resident_before_forward_bytes']}/{current['incremental_peak_bytes']}|")
    lines += ['\nMAC 不含 FFT、复数 CG、归一化、激活、插值和访存。实测包含完整恢复链，使用 16 条 source V、batch 1、2 次预热；'
              '顺序和共存负载影响延迟，不从参数比例推断加速。原 R17 的同口径延迟和峰值显存保留在 results.json。\n']
    for arm, row in analysis['rows'].items():
        train = row['training']
        lines += ['\n### '+arm+'：完整训练与波形证据\n',
            '全量 10 轮损失首轮→末轮：'+'；'.join(k+f" {v['first']:.8g}→{v['last']:.8g}" for k,v in train['losses'].items())+
            f"。中位每轮 {train['epoch_seconds_median']:.3f} s。日志命中：{json.dumps(row['stdout_findings'], ensure_ascii=False)}。\n",
            '|场景|full256 NMSE|prefix232 NMSE|tail24 NMSE|FFT log-power L1|配对 clean 特征距离|',
            '|---|---:|---:|---:|---:|---:|']
        for view in d.VIEWS:
            item = row['target']['views'][view]
            lines.append('|'+view+'|'+'|'.join(f"{item['waveform'][k]:.9g}" for k in
                ('iq_nmse','prefix232_nmse','tail24_nmse','fft_log_power_L1'))+'|'+
                str(item['paired_clean_geometry'].get('paired_clean_feature_cosine_distance','未提供'))+'|')
    lines += ['\n逐 TX/RX/day 的准确率、Macro-F1、纠错/损伤、混淆矩阵、配对 clean 几何与完整训练曲线见 results.json 和 row_metrics.csv。'
        '这些指标不能证明恢复了全部真实 TX 细节。每组件累计 50 轮；祖先训练共 90 串行 pass，component-epoch 合计 150，F50 预训练另计。\n',
        '本地模型：local_artifacts/pamf_restoration/r38_successor_20261020_r13/models/<arm>/final.pt。'
        '加载 experiments.pamf_r38_successor.runtime.load_checkpoint；输入原生尺度 float32 [B,2,256] IQ，输出同形 IQ，接原 F50 默认头。'
        '权重不入 Git。目标状态依据全部识别指标保留，不以本轮完成宣称长期优化目标完成。\n']
    report_path.write_text(body+'\n'.join(lines), encoding='utf-8')
    write(folder/'results.json', analysis)
    records = []
    for arm, row in analysis['rows'].items():
        for role in ('source','target'):
            for view, item in row[role]['views'].items():
                for dimension, groups in [('overall', {'all':item}), *item['groups'].items()]:
                    for label, metric in groups.items():
                        records.append(dict(arm=arm, role=role, view=view, dimension=dimension, value=label,
                            **{k:metric[k] for k in ('count','correct','accuracy','macro_f1','repaired','damaged','net_repaired')}))
    with (folder/'row_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    spec = json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
    spec['scientific_outcome'] = dict(status='ANALYZED', accepted_rows=analysis['accepted_rows'],
        accepted_IQ_rows=analysis['accepted_IQ_rows'], all_six_rows_scored=True, goal_complete=False)
    write(folder/'experiment.json', spec)
    module_spec = importlib.util.spec_from_file_location('registry', local/'tools/experiment_registry.py')
    registry = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(registry)
    for arm, row in analysis['rows'].items():
        registry.event(local, d.RUN, 'ANALYZED', ['report.md','results.json','row_metrics.csv'],
            'VERIFIED E10/2470;source7;target168000x7;truth-last;models strict-loaded;dual1pp='+str(row['accepted_both']), row_id=arm)
    registry.event(local, d.RUN, 'ANALYZED', ['report.md','results.json','row_metrics.csv'],
                   'All six fixed rows complete; all negative results preserved; no target feedback')
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):
        destination = WORKSPACE/(folder/name).relative_to(local)
        if destination.resolve() != (folder/name).resolve():
            shutil.copy2(folder/name, destination)
    (output/'models/README.md').write_text('六行完整模型与七场景结果：'+str(report_path)+'\n'
        '加载 experiments.pamf_r38_successor.runtime.load_checkpoint(path, device)，原生 float32 [B,2,256] IQ 输入/输出。\n', encoding='utf-8')
    print(json.dumps(dict(status='REPORT_AND_MODELS_VERIFIED', report=str(report_path), accepted_rows=analysis['accepted_rows'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root', type=Path, default=Path('E:/type10-7'))
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    output = args.local_root/'local_artifacts/pamf_restoration/r38_successor_20261020_r13'
    analysis = collect(args.local_root, output)
    if analysis is not None and not args.collect_only:
        report(args.local_root, output, analysis, retrieve(args.local_root, output))


if __name__ == '__main__':
    main()
