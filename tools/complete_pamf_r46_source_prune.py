"""Collect and deliver the existing R54 run; never launch or modify remote jobs."""
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
from experiments.pamf_r46_source_prune import design as d
from tools.analyze_pamf_r46_source_prune_results import completed_metrics, METHOD


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
   if state!='Z' and RUN in argv and 'experiments.pamf_r46_source_prune.runner' in argv:live.append(pid)
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
 control=root.parent/'20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53'
 assert read(control/'launch.json')['commit']=='df36c3d3da4e026b747776b65b58d4921dc1a71e' and read(control/'prune_decay/complete.json')['status']=='ANALYZED'
 reused=dict(run=control.name,row='prune_decay',method_commit='df36c3d3da4e026b747776b65b58d4921dc1a71e',
  training_records=[json.loads(line) for line in (control/'prune_decay/training.jsonl').read_text().splitlines() if line.strip()],
  training_complete=read(control/'prune_decay/training_complete.json'),
  source=read(control/'prune_decay/native_map_prior/summary.json'),target=read(control/'prune_decay/target_confirmation/native_map_prior/summary.json'))
 print(json.dumps(dict(run_id=root.name,method_commit=METHOD,complete=read(root/'complete.json'),failure=failure,reused_weight_control=reused,calibration=read(root/'calibration.json'),
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
    from experiments.pamf_r46_source_prune.runtime import load_checkpoint
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
    lines=['# R54：同一压缩拓扑的源通道筛选正式结果','',
        '状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。','',
        '唯一直接初始化与冻结IQ教师仍为用户指定R46 prune_refine350521。两行均沿用完整273765拓扑，区别是source激活能量与随机输出投影gain敏感度选择通道；原3prior/6CG、GN4组配额与共同prior map、全部coarse/action/prior联合、pairedclean IQ+.05谱误差+1→.1IQ教师、E20/global256micro256均保持。固定同宽weight-only控制复用R53 prune_decay，不重复训练。校准共用一次128源样本/256冻结G输入、0optimizer更新，另计计算与曝光；不称data-free初始化。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。属于有clean模拟配对的TX无标签源域自监督。','',
        '方法commit `'+METHOD+'`。完整登记、数据与祖先、命令和停止规则见[登记](experiment.json)；全部20轮与TX/RX/day见[结果](results.json)及[逐组CSV](row_metrics.csv)。','',
        '|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|','|---|---:|---:|---|---|---:|']
    for arm,row in analysis['rows'].items():
        lines.append(f"|{arm}|{row['parameters']}|{row['reduction_from_R46_percent']:.6f}%|{row['acceptance']['R17']['accepted']}|{row['acceptance']['R46']['accepted']}|{row['performance_optimization']['low_urban_accuracy']:.6f}|")
    lines+=['','逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。','',
        '|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in analysis['rows'].items():
        for view,item in row['target']['views'].items():
            lines.append(f"|{arm}|{view}|{item['accuracy']:.6f}|{item['macro_f1']:.6f}|{row['target_vs_R46']['views'][view]['accuracy_delta_pp']:+.6f}|{row['target_vs_R44']['views'][view]['accuracy_delta_pp']:+.6f}|{row['target_vs_R17']['views'][view]['accuracy_delta_pp']:+.6f}|{item['net_repaired']:+d}|")
    lines+=['','同宽度通道筛选对照：以下只比较预登记R53 prune_decay权重筛选控制，保留全部七景差值；额外共用源校准256次冻结G输入另计，不以正式结果重新选择父、宽度或筛选方式。','',
        '|源通道筛选行|权重筛选控制|场景|准确率变化（pp）|MacroF1变化（pp）|','|---|---|---|---:|---:|']
    for arm,comparison in analysis['same_width_channel_selection_comparisons'].items():
        for view,item in comparison['target']['views'].items():
            lines.append('|'+arm+'|'+comparison['weight_control']+'|'+view+f"|{item['accuracy_delta_pp']:+.6f}|{item['macro_f1_delta_pp']:+.6f}|")
    lines+=['','|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|','|---|---|---:|---:|']
    for arm,row in analysis['rows'].items():
        model=artifacts['models'][arm]
        lines.append('|'+arm+'|'+'/'.join(str(v) for v in model['components'].values())+f"|{model['bytes']}|{model['conv_linear_MACs_per_IQ']}|")
    lines+=['','|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|','|---|---|---|---:|---:|---|']
    for arm,row in analysis['rows'].items():
        profile=row['profile']['results'];current=profile['compact'];base=profile['parent_R46_350521']
        for key,item in current['cases'].items():
            reference=base['cases'][key]
            lines.append('|'+arm+'|'+key+f"|{reference['median_ms']:.4f}/{item['median_ms']:.4f}|{item['p95_ms']:.4f}|{item['throughput_samples_per_second']:.4f}|{reference['incremental_peak_bytes']}/{item['incremental_peak_bytes']}|")
    lines+=['','同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。','']
    cal=analysis['shared_source_calibration']
    lines+=['','共用源校准：'+str(cal['source_unique_IQ'])+'条物理样本（64L_s/64U_s）、'+str(cal['frozen_parent_IQ_inputs'])+'次冻结G输入、耗时'+f"{cal['seconds']:.4f}"+'s；source received01/20、两次Rademacher投影；0TX/F/V/target/optimizer更新、父状态不变。具体分数/map/行位置保持私有，不是运行时输入。','']
    for arm,row in analysis['rows'].items():
        perf=row['performance_optimization'];train=row['training']
        lines.append(arm+'首轮→末轮：'+'；'.join(k+f" {v['first']:.8g}→{v['last']:.8g}" for k,v in train['losses'].items())+f"；中位每轮{train['epoch_seconds_median']:.3f}s。lowurban70={perf['low_urban_at_least_70']}；其他五景gap5pp={perf['other_five_within_5pp_of_raw_clean']}。日志类别："+json.dumps(row['stdout_findings'],ensure_ascii=False))
    lines+=['','全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。','',
        '继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。两行同E20训练预算、同batch256/micro256/directbuffer、同1→.1教师；额外共用256次冻结G源校准、0optimizer更新，不重复计算为两份或隐藏为0数据初始化，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。','',
        '独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_source_prune_r54_20261021/models/<arm>/final.pt；使用experiments.pamf_r46_source_prune.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。','']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    public_artifacts=dict(artifacts,models={a:dict(m,path=Path(m['path']).relative_to(local).as_posix()) for a,m in artifacts['models'].items()})
    write(folder/'results.json',dict(analysis,model_artifacts=public_artifacts))
    records=[]
    for arm,row in analysis['rows'].items():
        for role in ('source','target'):
            for view,item in row[role]['views'].items():
                for dimension,groups in [('overall',{'all':item}),*item['groups'].items()]:
                    for label,metric in groups.items():
                        records.append(dict(arm=arm,role=role,view=view,dimension=dimension,value=label,**{k:metric[k] for k in ('count','correct','accuracy','macro_f1','repaired','damaged','net_repaired')}))
    with (folder/'row_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    usage = ['# R54 两套独立恢复器', '',
        '两行均已完成固定 E20/4940、168000 条记录的七场景正式预测及独立 truth-last 评分。全部正负结果见公开报告。CPU 严格加载、合成零 IQ 有限输出及参数核对已通过；不因此宣称输入 TX 的每个细节均已得到证明。', '',
        '|模型|完整参数|权重相对路径|', '|---|---:|---|']
    for arm in d.ARMS:
        usage.append(f"|{arm}|{artifacts['models'][arm]['parameters']}|{arm}/final.pt|")
    usage += ['', '```python', 'import torch',
        'from experiments.pamf_r46_source_prune.runtime import load_checkpoint',
        'model, metadata = load_checkpoint("path/to/models/activation_energy/final.pt", "cuda:0")',
        'model.eval().requires_grad_(False)', 'with torch.no_grad():',
        '    restored_iq = model(received_iq)  # float32 [B, 2, 256]，原生幅度', '```', '',
        '恢复器推理无需教师、源数据、校准文件或 F。身份识别时，将同一 restored_iq 交给原冻结 F50 默认头，由原 F 一致计算 FFT、PA、时域与统计分支；不另训或替换分类头。两版均直接继承用户指定的 R46 350521 父版。', '',
        '识别性能为主要指标，参数、时延、吞吐和显存为辅助指标。每景准确率、匹配同宽控制与原 70%/5pp 目标分别报告。权重不提交 Git。', '']
    (output/'models/README.md').write_text('\n'.join(usage),encoding='utf-8')
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'));spec['scientific_outcome']=dict(status='ANALYZED',all_fixed_rows_scored=True,
        accepted_R17=[a for a,r in analysis['rows'].items() if r['acceptance']['R17']['accepted']],accepted_R46=[a for a,r in analysis['rows'].items() if r['acceptance']['R46']['accepted']],goal_complete=False)
    write(folder/'experiment.json',spec)
    module=importlib.util.spec_from_file_location('registry',local/'tools/experiment_registry.py');registry=importlib.util.module_from_spec(module);module.loader.exec_module(registry)
    for arm in d.ARMS:registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'VERIFIED E20/4940;all7formaltruth-last;strictstandaloneIQmodel;ownedEXITED',row_id=arm)
    registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'Both fixed source-channel-selection rows complete;negative results preserved;goalactive')
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):
        destination=WORKSPACE/(folder/name).relative_to(local)
        if destination.resolve()!=(folder/name).resolve():shutil.copyfile(folder/name,destination)
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,methods=list(d.ARMS),all7formal=True,goal_complete=False))
    print(json.dumps(dict(status='REPORT_AND_MODELS_VERIFIED',report=str(folder/'report.md'),goal_complete=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collect-only',action='store_true');args=parser.parse_args()
    local=Path('E:/type10-7');output=local/'local_artifacts/pamf_restoration/r46_source_prune_r54_20261021'
    analysis=collect(local,output)
    if analysis is not None and not args.collect_only:report(local,output,analysis,retrieve(local,output))


if __name__=='__main__':main()
