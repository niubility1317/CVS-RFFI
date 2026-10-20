"""Collect and deliver the existing R62 run; never launch or modify remote jobs."""
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
from experiments.pamf_r46_group_drop import design as d
from tools.analyze_pamf_r46_group_drop_results import completed_metrics, METHOD


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
 for pid in {PID,*[w['pid'] for w in workers]}:
  path=Path('/proc')/str(pid)
  try:
   state=(path/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(path/'cmdline').read_bytes().decode().split(chr(0))
   if state!='Z' and root.name in ' '.join(argv) and 'experiments.pamf_r46_group_drop.runner' in argv:live.append(pid)
  except FileNotFoundError:pass
 rows={}
 for arm in ARMS:
  row=root/arm;source=row/'native_map_prior';target=row/'target_confirmation/native_map_prior'
  log=root/'logs'/('worker-'+arm+'.log')
  rows[arm]=dict(training_records=[json.loads(line) for line in (row/'training.jsonl').read_text().splitlines() if line.strip()],
   training_complete=read(row/'training_complete.json'),initialization=read(row/'initialization.json'),
   matched_profile=read(row/'matched_inference_profile.json'),source_summary=read(source/'summary.json'),
   source_prediction_complete=read(source/'prediction/complete.json'),target_summary=read(target/'summary.json'),
   target_prediction_complete=read(target/'prediction/complete.json'),complete=read(row/'complete.json'),
   failure=read(row/'failure.json') if (row/'failure.json').exists() else None,
   assessment=read(row/'assessment.json'),checkpoint_bytes=(source/'final.pt').stat().st_size,
   checkpoint_path=str(source/'final.pt'),stdout=log.read_text(errors='replace'))
 baselines={name:{role:read(root.parent/run/arm/('target_confirmation/' if role=='target' else '')/'native_map_prior/summary.json')
  for role in ('source','target')} for name,run,arm in (('R17',R17,'unroll_unshared'),('R44',R44,'group_pruned'),('R46',R46,'prune_refine'))}
 control=root.parent/'20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49'
 assert read(control/'launch.json')['commit']=='e96397bc3c7036f085317f9cc0e59074fb6ee24f' and read(control/'complete.json')['status']=='ANALYZED'
 reused=dict(run=control.name,row='retain_refine',method_commit='e96397bc3c7036f085317f9cc0e59074fb6ee24f',
  training_records=[json.loads(line) for line in (control/'retain_refine/training.jsonl').read_text().splitlines() if line.strip()],
  training_complete=read(control/'retain_refine/training_complete.json'),
  source=read(control/'retain_refine/native_map_prior/summary.json'),target=read(control/'retain_refine/target_confirmation/native_map_prior/summary.json'))
 print(json.dumps(dict(run_id=root.name,method_commit=METHOD,complete=read(root/'complete.json'),failure=failure,reused_retain_control=reused,
  actual_live_owned_PIDs=live,rows=rows,baselines=baselines,controller_stdout=Path(RUN).parent.parent.joinpath('logs',root.name,'runner.log').read_text(errors='replace'))))
'''
    value = remote(local, code)
    if value.get('status') in ('PENDING', 'FAILED_TECHNICAL'):
        print(json.dumps(value))
        return None
    folder = output/'collections'
    folder.mkdir(exist_ok=True)
    for arm, row in value['rows'].items():
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
    from experiments.pamf_r46_group_drop.runtime import load_checkpoint
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
    lines=['# R62：整组压缩的正式性能与成本','',
        '**VERIFIED / ANALYZED**。两固定行完整 E20/4940、source1800×7 和正式168000×7；每行七景预测固定后独立 truth-last，实际所属 controller/worker 全退出，独立权重严格重载通过。全部正负结果保留，不反馈调参或选择性重跑。','',
        '性能优先。两行同248828参数、同选组和初始权重、同训练预算，只改变 context GN3/GN1；prior完整112594参数。权重选择删除一个完整原GN组，无校准IQ。GN3保留原每组通道宽度，GN1重新合组；层级同值等价不代表整个恢复器等价，归约方式和实测成本不同。','',
        '直接指定350521父初始化和冻结三阶段IQ教师；全coarse/action/prior联合，pairedclean IQ NMSE+.05谱误差+线性1→.1教师，全63000源记录/20缓存轮/globalmicro256。L126000/U1134000，0TX/F/V/target优化，0新增参数/校准IQ/额外G前向；合法源clean模拟配对的TX无标签自监督。','',
        '方法commit `'+METHOD+'`。完整命令/环境/矩阵/数据权限见[登记](experiment.json)，完整曲线、每TX/RX/day、修复损伤、NMSE/FFT/特征描述见[结果](results.json)与[CSV](row_metrics.csv)。','',
        '|行|完整参数|较M35减参|原R17逐景1pp|M35逐景1pp|low urban（%）|其他五景gap≤5pp|','|---|---:|---:|---|---|---:|---|']
    for arm,row in analysis['rows'].items():
        p=row['performance_optimization']
        lines.append(f"|{arm}|{row['parameters']}|{row['reduction_from_R46_percent']:.6f}%|{row['acceptance']['R17']['accepted']}|{row['acceptance']['R46']['accepted']}|{p['low_urban_accuracy']:.6f}|{p['other_five_within_5pp_of_raw_clean']}|")
    lines+=['','|行|场景|准确率（%）|MacroF1（%）|较M35（pp）|较原R17（pp）|较R49同预算（pp）|净修复|','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in analysis['rows'].items():
        for view,v in row['target']['views'].items():
            delta=analysis['matched_budget_comparisons'][arm]['target']['views'][view]['accuracy_delta_pp']
            lines.append(f"|{arm}|{view}|{v['accuracy']:.6f}|{v['macro_f1']:.6f}|{row['target_vs_R46']['views'][view]['accuracy_delta_pp']:+.6f}|{row['target_vs_R17']['views'][view]['accuracy_delta_pp']:+.6f}|{delta:+.6f}|{v['net_repaired']:+d}|")
    lines+=['','主消融为GN3相对GN1：相同初始权重、通道和预算，保留全部逐景差；同预算未裁剪R49 retain_refine只读复用，不重复训练/继承其权重。宽度和归一化同时区别于R49，不把全部收益归于单一因素。','',
        '|场景|GN3−GN1准确率（pp）|MacroF1（pp）|','|---|---:|---:|']
    for view,v in analysis['grouping_comparison']['target']['views'].items():lines.append(f"|{view}|{v['accuracy_delta_pp']:+.6f}|{v['macro_f1_delta_pp']:+.6f}|")
    lines+=['','|行|coarse/action/prior参数|文件字节|Conv/Linear MAC每IQ|完整训练秒|训练peak allocated字节|','|---|---|---:|---:|---:|---:|']
    for arm,row in analysis['rows'].items():
        m=artifacts['models'][arm];t=row['training']['training_complete']
        lines.append('|'+arm+'|'+'/'.join(str(v) for v in m['components'].values())+f"|{m['bytes']}|{m['conv_linear_MACs_per_IQ']}|{t['seconds']:.4f}|{t['peak_allocated_bytes']}|")
    lines+=['','|行|口径|父/本版median ms|本版P90/P95 ms|本版sample/s|父/本版增量峰值字节|','|---|---|---|---|---:|---|']
    for arm,row in analysis['rows'].items():
        current=row['profile']['results']['compact']['cases'];parent=row['profile']['results']['parent_R46_350521']['cases']
        for key,v in current.items():
            b=parent[key];lines.append(f"|{arm}|{key}|{b['median_ms']:.4f}/{v['median_ms']:.4f}|{v['p90_ms']:.4f}/{v['p95_ms']:.4f}|{v['throughput_samples_per_second']:.4f}|{b['incremental_peak_bytes']}/{v['incremental_peak_bytes']}|")
    lines+=['','成本使用同卡64条sourceV、G/G+冻结原F、B1/B32、3轮交替次序和预热。同步预加载GPU输入，不含传输/前处理/loader；三G一F共享驻留suite，增量峰值是当前前向，不当作独立部署显存。Conv/Linear MAC不含FFT/CG/norm/activation/F，训练peak包括教师/快照/优化器/激活；不同GPU时段不据绝对数值作速度排名。','']
    for arm,row in analysis['rows'].items():
        t=row['training'];p=row['performance_optimization']
        lines.append(arm+'完整20轮首→末：'+'；'.join(k+f" {v['first']:.8g}→{v['last']:.8g}" for k,v in t['losses'].items())+f"；每轮中位{t['epoch_seconds_median']:.4f}s；lowurban70={p['low_urban_at_least_70']}；otherfive5pp={p['other_five_within_5pp_of_raw_clean']}；日志类别"+json.dumps(row['stdout_findings'],ensure_ascii=False))
    lines+=['','原R17逐景1pp是退化上限，性能提升、lowurban≥70%与其他五受损场景距rawclean84.6761904762%≤5pp分别判断。单seed、相关视图、已暴露且由用户指定父的工程基准；全部正负结果保留，不称新独立盲测。重建、识别率或F几何不能单独证明每个恢复细节来自输入TX。完整目标 ACTIVE。','',
        '旧每组件80+本次20=100、组件轮次和300、串行140 pass，原F50训练另计。相对M35增加20轮，另报告同增量R49控制，不能把追加预算收益全部归于减参。','',
        '独立权重位于local_artifacts/pamf_restoration/r46_group_drop_r62_20261021/models/<arm>/final.pt；运行experiments.pamf_r46_group_drop.runtime.load_checkpoint后输入float32[B,2,256]原生IQ，输出同形IQ。恢复器无需教师/F/源缓存，识别时同一恢复IQ接原冻结F50默认头与FFT/PA/时域/统计分支，无另训头。','']
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
        accepted_R17=[a for a,r in analysis['rows'].items() if r['acceptance']['R17']['accepted']],accepted_R46=[a for a,r in analysis['rows'].items() if r['acceptance']['R46']['accepted']],goal_complete=False)
    write(folder/'experiment.json',spec)
    reg=importlib.util.spec_from_file_location('registry',local/'tools/experiment_registry.py');registry=importlib.util.module_from_spec(reg);reg.loader.exec_module(registry)
    for arm in d.ARMS:registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'E20/4940/full7formal/independenttruthlast/strictweights/ownedEXITED',row_id=arm)
    registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'Both fixed normalization-group rows complete;negative results preserved;goalACTIVE')
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):
        dest=WORKSPACE/(folder/name).relative_to(local)
        if dest.resolve()!=(folder/name).resolve():shutil.copyfile(folder/name,dest)
    usage=['# R62 独立恢复器','','两行全E20/4940及七景正式测试已完成；完整248828参数，严格加载绑定GN3/GN1。','',
        '```python','import torch','from experiments.pamf_r46_group_drop.runtime import load_checkpoint',
        'model, metadata = load_checkpoint("path/to/models/group_preserve/final.pt", "cuda:0")','model.eval().requires_grad_(False)',
        'with torch.no_grad():','    restored_iq = model(received_iq)  # float32[B,2,256]原生幅度','```','',
        '另一行regroup_control/final.pt。G推理只需当前IQ和本权重；识别使用原冻结F50默认头。同一恢复IQ供FFT/PA/时域/统计分支，无另训头。全部正负结果见正式报告，逐细节TX真实性未证明。','']
    (output/'models/README.md').write_text('\n'.join(usage),encoding='utf-8')
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,methods=list(d.ARMS),all7formal=True,goal_complete=False))
    print(json.dumps(dict(status='REPORT_AND_MODELS_VERIFIED',report=str(folder/'report.md'),goal_complete=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collect-only',action='store_true');args=parser.parse_args()
    local=Path('E:/type10-7');output=local/'local_artifacts/pamf_restoration/r46_group_drop_r62_20261021'
    analysis=collect(local,output)
    if analysis is not None and not args.collect_only:report(local,output,analysis,retrieve(local,output))


if __name__=='__main__':main()
