"""Collect and deliver the existing R66 run; never launch or modify remote jobs."""
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
from experiments.pamf_r46_weight_mean import design as d
from tools.analyze_pamf_r46_weight_mean_results import completed_metrics, METHOD


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
        METHOD=METHOD, ARMS=d.ARMS, R17=d.PARENT_RUN, R44=d.parent.COMPACT_PARENT_RUN, R46=d.COMPACT_PARENT_RUN).items())+'\n'+'''
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
 for pid in {PID,read(root/'source_worker.json')['pid'],*[w['pid'] for w in workers]}:
  path=Path('/proc')/str(pid)
  try:
   state=(path/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(path/'cmdline').read_bytes().decode().split(chr(0))
   if state!='Z' and root.name in ' '.join(argv) and 'experiments.pamf_r46_weight_mean.runner' in argv:live.append(pid)
  except FileNotFoundError:pass
 rows={}
 for arm in ARMS:
  row=root/arm;source=row/'native_map_prior';target=row/'target_confirmation/native_map_prior'
  log=root/'logs'/('worker-'+arm+'.log')
  rows[arm]=dict(parameter_readout=read(row/'parameter_readout.json'),
   matched_profile=read(row/'matched_inference_profile.json'),source_summary=read(source/'summary.json'),
   source_prediction_complete=read(source/'prediction/complete.json'),target_summary=read(target/'summary.json'),
   target_prediction_complete=read(target/'prediction/complete.json'),complete=read(row/'complete.json'),
   failure=read(row/'failure.json') if (row/'failure.json').exists() else None,
   assessment=read(row/'assessment.json'),checkpoint_bytes=(source/'final.pt').stat().st_size,
   checkpoint_path=str(source/'final.pt'),stdout=log.read_text(errors='replace'))
 shared=root/'source_path'
 shared_source=dict(training_records=[json.loads(line) for line in (shared/'training.jsonl').read_text().splitlines() if line.strip()],
  training_complete=read(shared/'training_complete.json'),initialization=read(shared/'initialization.json'),completion=read(root/'shared_source_complete.json'),
  stdout=(root/'logs/worker-source_path.log').read_text(errors='replace'))
 baselines={name:{role:read(root.parent/run/arm/('target_confirmation/' if role=='target' else '')/'native_map_prior/summary.json')
  for role in ('source','target')} for name,run,arm in (('R17',R17,'unroll_unshared'),('R44',R44,'group_pruned'),('R46',R46,'prune_refine'))}
 print(json.dumps(dict(run_id=root.name,method_commit=METHOD,complete=read(root/'complete.json'),failure=failure,shared_source=shared_source,
  actual_live_owned_PIDs=live,rows=rows,baselines=baselines,controller_stdout=Path(RUN).parent.parent.joinpath('logs',root.name,'runner.log').read_text(errors='replace'))))
'''
    value = remote(local, code)
    if value.get('status') in ('PENDING', 'FAILED_TECHNICAL'):
        print(json.dumps(value))
        return None
    folder = output/'collections'
    folder.mkdir(exist_ok=True)
    for row in [value['shared_source'],*value['rows'].values()]:
        # Keep raw output in the private collection; public results contain only
        # finding categories and counts, not raw log lines or runtime paths.
        findings = {}
        for line in row['stdout'].splitlines():
            for kind, pattern in (('traceback', r'Traceback'), ('out_of_memory', r'out of memory'),
                                  ('killed', r'killed'), ('warning', r'warning'), ('nonfinite', r'\b(?:nan|inf)\b')):
                if re.search(pattern, line, re.I):
                    findings[kind] = findings.get(kind, 0) + 1
        row['stdout_findings'] = [dict(kind=kind, lines=count) for kind, count in findings.items()]
    write(folder/(str(time.time_ns())+'.json'), value)
    analysis = completed_metrics(value)
    write(output/'analysis.json', analysis)
    print(json.dumps(dict(status='COLLECTED_COMPLETE', accepted_rows=[a for a,r in analysis['rows'].items() if r['accepted_both']])))
    return analysis


def retrieve(local, output):
    saved = output/'model_artifacts.json'
    if saved.exists():
        return json.loads(saved.read_text(encoding='utf-8'))
    collections = sorted((output/'collections').glob('*.json'))
    raw = json.loads(collections[-1].read_text(encoding='utf-8'))
    import torch
    from experiments.pamf_r46_weight_mean.runtime import load_checkpoint
    torch.set_num_threads(2)
    artifacts = {}
    partial_path=output/'partial_model_artifacts.json'
    partial=json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else None
    if partial is not None and (partial.get('status')!='VERIFIED' or partial.get('source_run')!=d.RUN or partial.get('method_commit')!=METHOD):
        raise ValueError('Existing partial delivery belongs to another run or method')
    for arm in d.ARMS:
        folder = output/'models'/arm
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'final.pt'
        row = raw['rows'][arm]
        known=partial.get('models',{}).get(arm) if partial else None
        if known is not None:
            if not path.is_file() or path.stat().st_size!=row['checkpoint_bytes'] or known['bytes']!=row['checkpoint_bytes']:
                raise ValueError('Reconcile the existing verified partial model instead of overwriting')
            if known['parameters']!=d.PARAMETERS[arm] or not known['strict_load'] or not known['synthetic_zero_IQ_output_finite'] or known['optimizer_included'] or known['training_F_access']:
                raise ValueError('Partial model inference proof differs')
            artifacts[arm]=known
            continue
        if not path.exists():
            subprocess.run(['C:/Program Files/Git/usr/bin/scp.exe', '-F', str(local/'tools/n607_ssh_config'),
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607:'+row['checkpoint_path'], str(path)],
                check=True, timeout=55)
        if path.stat().st_size != row['checkpoint_bytes']:
            raise ValueError('Incomplete retrieved artifact; preserve and reconcile transfer')
        model, payload = load_checkpoint(path, 'cpu')
        if (payload['arm'], payload['epoch'], payload['updates']) != (arm, 20, 4940):
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
            strict_load=True, synthetic_zero_IQ_output_finite=True, actual_IQ_reads=0, optimizer_included=False,
            training_F_access=d.TASK[arm], training_TX_labels_read=0, parent_R46_run=d.COMPACT_PARENT_RUN)
    result = dict(status='VERIFIED', source_run=d.RUN, method_commit=METHOD, models=artifacts,
                  MAC_scope='Conv1d and Linear only; excludes FFT/complex CG/normalization/activation/memory traffic')
    write(saved, result)
    return result




def report(local,output,analysis,artifacts):
    folder=local/'automation_reports/CV-SincNet'/d.RUN
    training=analysis['shared_source_training'];t=training['training_complete']
    lines=['# R66：一次源训练的三个固定权重读出','',
        '**VERIFIED / ANALYZED**。一次源训练E20/4940，三个固定读出均完成source1800×7和正式168000×7；每读出全部七视图预测固定后独立truth-last。实际所属进程已退出，三份独立推理权重严格重载通过，全部正负结果保留。完整目标ACTIVE。','',
        '性能优先。三版同为273765参数/context185615，直接从指定M35固定GN4图裁剪，教师始终为原冻结M35。原pairedclean IQ NMSE+.05谱误差+线性1→.1三阶段IQ教师目标，全部coarse/action/prior联合更新。0TX/F/V/target优化、新校准或额外神经前向。允许的源模拟配对clean自监督，不称cleanfree。','',
        '训练一次后读取末次权重final、完整E16至20等权均值late_mean、update2471至4940的decay.99参数EMA step_ema。平均不是训练教师；714个唯一参数各平均一次，770别名保留共享，bias/GN affine独立，buffers固定，无推理附加参数。','',
        '三个读出相关，共享同一条源轨迹；主比较均值/EMA相对同一新final，不当作三次独立训练或三个seed。方案和窗口在读取R65识别率前固定；不得用已暴露目标反馈改变窗口、选择或重跑。','',
        '方法commit `'+METHOD+'`；精确矩阵、命令、环境/CWD与停止规则见[登记](experiment.json)，全部20轮及逐TX/RX/day分组见[结果](results.json)和[CSV](row_metrics.csv)。','',
        '|读出|参数|较M35减参|原R17逐景1pp|M35逐景1pp|low urban（%）|其他五景gap≤5pp|','|---|---:|---:|---|---|---:|---|']
    for arm,row in analysis['rows'].items():
        p=row['performance_optimization']
        lines.append(f"|{arm}|{row['parameters']}|{row['reduction_from_R46_percent']:.6f}%|{row['acceptance']['R17']['accepted']}|{row['acceptance']['R46']['accepted']}|{p['low_urban_accuracy']:.6f}|{p['other_five_within_5pp_of_raw_clean']}|")
    lines+=['','|读出|场景|准确率（%）|MacroF1（%）|较M35（pp）|较原R17（pp）|较同轨迹final（pp）|净修复|','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in analysis['rows'].items():
        for view,v in row['target']['views'].items():
            delta=0. if arm=='final' else analysis['same_trajectory_comparisons'][arm]['target']['views'][view]['accuracy_delta_pp']
            lines.append(f"|{arm}|{view}|{v['accuracy']:.6f}|{v['macro_f1']:.6f}|{row['target_vs_R46']['views'][view]['accuracy_delta_pp']:+.6f}|{row['target_vs_R17']['views'][view]['accuracy_delta_pp']:+.6f}|{delta:+.6f}|{v['net_repaired']:+d}|")
    lines+=['','|共享源训练成本（全矩阵只计一次）|实际值|','|---|---|',
        '|源轨迹/epoch/update|1 / 20 / 4940|',f"|L/U曝光|{t['L_exposures']} / {t['U_exposures']}|",
        f"|student/原M35 teacher IQ|{t['source_IQ_examples']} / {t['teacher_IQ_examples']}|",
        f"|训练墙钟秒（含GPU平均内核）|{t['seconds']:.6f}|",f"|包含三权重序列化的总秒|{t['seconds_including_all_readout_serialization']:.6f}|",
        f"|平均host enqueue秒|{t['averaging_host_enqueue_seconds']:.6f}|",f"|完整训练及读出导出peak allocated字节|{t['peak_allocated_bytes']}|",
        f"|实际EMA更新/末轮快照|{t['EMA_updates']} / {t['late_mean_snapshots']}|",'',
        '平均host enqueue时间只量到主机提交，不能当作隔离的新增GPU耗时；GPU平均内核已计入完整训练墙钟。峰值包括教师、优化器、激活、平均状态及权重导出。继承每组件80+本次20=100轮，组件轮次和300/串行140 pass只计一次；原F50训练另计。三个读出的source/profile/formal推理成本各计，不把三版共享训练成本重复相加。','',
        '|读出|coarse/action/prior参数|文件字节|Conv/Linear MAC每IQ|','|---|---|---:|---:|']
    for arm in d.ARMS:
        m=artifacts['models'][arm]
        lines.append('|'+arm+'|'+'/'.join(str(v) for v in m['components'].values())+f"|{m['bytes']}|{m['conv_linear_MACs_per_IQ']}|")
    lines+=['','|读出|source1800×7预测秒|正式168000×7预测秒|','|---|---:|---:|']
    for arm,row in analysis['rows'].items():
        lines.append(f"|{arm}|{row['prediction_cost']['source']['seconds']:.4f}|{row['prediction_cost']['target']['seconds']:.4f}|")
    lines+=['','上表是实际预测墙钟，不含profile和独立truth评分。各视图秒数在结果文件中；下表为独立profile测量，不将其推算为整套profile墙钟。','',
        '|读出|口径|父/本版median ms|本版P90/P95 ms|本版sample/s|父/本版增量峰值字节|','|---|---|---|---|---:|---|']
    for arm,row in analysis['rows'].items():
        current=row['profile']['results']['compact']['cases'];parent=row['profile']['results']['parent_R46_350521']['cases']
        for key,v in current.items():
            b=parent[key];lines.append(f"|{arm}|{key}|{b['median_ms']:.4f}/{v['median_ms']:.4f}|{v['p90_ms']:.4f}/{v['p95_ms']:.4f}|{v['throughput_samples_per_second']:.4f}|{b['incremental_peak_bytes']}/{v['incremental_peak_bytes']}|")
    lines+=['','同卡64条sourceV、G/G+原冻结F、B1/B32、三轮交替次序预热；同步预加载GPU输入，不含传输/前处理/loader。三G一F共享驻留suite，增量峰值对应当前前向，不当作独立部署显存。Conv/Linear MAC不含FFT/CG/norm/activation/F；不同GPU时段不据绝对时延排名。','',
        '完整共享20轮首→末：'+'；'.join(k+f" {v['first']:.8g}→{v['last']:.8g}" for k,v in training['losses'].items())+f"；每轮中位{training['epoch_seconds_median']:.4f}s。",
        '完整源训练日志类别：'+json.dumps(training['stdout_findings'],ensure_ascii=False)+'；各评测读出日志类别：'+json.dumps({a:r['stdout_findings'] for a,r in analysis['rows'].items()},ensure_ascii=False),'',
        'lowurban≥70%、其他五受损景距rawclean84.6761904762%≤5pp、识别率实际提升与原R17每景1pp退化上限分别报告。容差和减参不代表性能提升；单seed/相关轨迹/已暴露工程基准，不称新独立盲测。重建、识别率或F几何不能单独证明恢复的每个细节属于输入TX。完整目标ACTIVE。','',
        '独立权重位于local_artifacts/pamf_restoration/r46_weight_mean_r66_20261021/models/<readout>/final.pt。使用experiments.pamf_r46_weight_mean.runtime.load_checkpoint，输入float32[B,2,256]原生IQ，输出同形IQ；G推理无教师/均值器/历史状态/源缓存，原F50默认头从同一恢复IQ生成FFT/PA/时域/统计输入。','']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    public=dict(artifacts,models={a:dict(m,path=Path(m['path']).relative_to(local).as_posix()) for a,m in artifacts['models'].items()})
    write(folder/'results.json',dict(analysis,model_artifacts=public))
    records=[]
    for arm,row in analysis['rows'].items():
        for role in ('source','target'):
            for view,v in row[role]['views'].items():
                for dimension,groups in [('overall',{'all':v}),*v['groups'].items()]:
                    for label,m in groups.items():records.append(dict(arm=arm,role=role,view=view,dimension=dimension,value=label,**{k:m[k] for k in ('count','correct','accuracy','macro_f1','repaired','damaged','net_repaired')}))
    with (folder/'row_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'));spec['scientific_outcome']=dict(status='ANALYZED',all_fixed_rows_scored=True,
        shared_source_training_paths=1,source_training_counted_once=True,goal_complete=False)
    write(folder/'experiment.json',spec)
    reg=importlib.util.spec_from_file_location('registry',local/'tools/experiment_registry.py');registry=importlib.util.module_from_spec(reg);reg.loader.exec_module(registry)
    for arm in d.ARMS:registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'SharedE20/4940/fixedreadout/full7formal/independenttruthlast/strictweights/ownedEXITED',row_id=arm)
    registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'One shared source path/three formal readouts;all positives/negatives preserved;goalACTIVE')
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):
        dest=WORKSPACE/(folder/name).relative_to(local)
        if dest.resolve()!=(folder/name).resolve():shutil.copyfile(folder/name,dest)
    (output/'models/README.md').write_text('\n'.join(['# R66 独立IQ恢复器','','三个读出共享一次源E20，均完成七正式；每版273765参数，普通GN4。','',
        '```python','import torch','from experiments.pamf_r46_weight_mean.runtime import load_checkpoint',
        'model, metadata = load_checkpoint("path/to/models/late_mean/final.pt", "cuda:0")','model.eval().requires_grad_(False)',
        'with torch.no_grad():','    restored_iq = model(received_iq)  # float32[B,2,256]原生幅度','```','',
        '同样可加载final或step_ema。G只用当前IQ及普通权重；识别接原冻结F50默认头，所有PAMF特征来自同一恢复IQ。逐细节TX真实性尚未证明。','']),encoding='utf-8')
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,methods=list(d.ARMS),all7formal=True,shared_source_training_paths=1,goal_complete=False))
    print(json.dumps(dict(status='REPORT_AND_MODELS_VERIFIED',report=str(folder/'report.md'),source_training_paths=1,goal_complete=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collect-only',action='store_true');args=parser.parse_args()
    local=Path('E:/type10-7');output=local/'local_artifacts/pamf_restoration/r46_weight_mean_r66_20261021'
    analysis=collect(local,output)
    if analysis is not None and not args.collect_only:report(local,output,analysis,retrieve(local,output))


if __name__=='__main__':main()
