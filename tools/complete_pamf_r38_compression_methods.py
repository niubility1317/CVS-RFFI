"""Collect and deliver the existing R44 run; never launch or modify remote jobs."""
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
from experiments.pamf_r38_compression_methods import design as d
from tools.analyze_pamf_r38_compression_methods_results import completed_metrics, METHOD


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
   if state!='Z' and RUN in argv and 'experiments.pamf_r38_compression_methods.runner' in argv:live.append(pid)
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
    print(json.dumps(dict(status='COLLECTED_COMPLETE', accepted_rows=[a for a,r in analysis['rows'].items() if r['accepted_both']])))
    return analysis


def retrieve(local, output):
    saved = output/'model_artifacts.json'
    if saved.exists():
        return json.loads(saved.read_text(encoding='utf-8'))
    collections = sorted((output/'collections').glob('*.json'))
    raw = json.loads(collections[-1].read_text(encoding='utf-8'))
    import torch
    from experiments.pamf_r38_compression_methods.runtime import load_checkpoint
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
            training_F_access=d.TASK[arm], training_TX_labels_read=0, parent_R38_run=d.COMPACT_PARENT_RUN)
    result = dict(status='VERIFIED', source_run=d.RUN, method_commit=METHOD, models=artifacts,
                  MAC_scope='Conv1d and Linear only; excludes FFT/complex CG/normalization/activation/memory traffic')
    write(saved, result)
    return result




def report(local,output,analysis,artifacts):
    folder=local/'automation_reports/CV-SincNet'/d.RUN
    lines=['# R44：低秩卷积与重要性剪枝的正式结果','',
        '状态：VERIFIED / ANALYZED。两方法均完成固定E20/4940、source1800×7及target168000×7；每行全部预测固定后独立CPU truth-last，所属controller/worker均已退出。全部结果保留，测试不回流训练或选择性重跑。','',
        '唯一直接父/冻结IQ教师为用户指定R38 dense_half794529；low_rank保留通道、长卷积半秩SVD并保持共享因子，group_pruned按GN4组内encoder权重重要性选通道。固定两方法各新增20轮，paired-clean IQNMSE+10到1三stageIQ教师约束。全部coarse/action/prior联合，L126000/U1134000曝光、0TX/F/V/target优化；原F50/defaulthead仅冻结评测。不是无clean盲学习。','',
        '方法commit：`'+METHOD+'`。完整矩阵、命令、数据契约、祖先来源、预算和停止规则见[登记](experiment.json)；统计、完整20轮曲线和逐TX/RX/day见[结果JSON](results.json)及[逐组CSV](row_metrics.csv)。','',
        '|方法|完整参数|较79.45万减参|相对R17逐景1pp通过|相对R38逐景1pp通过|low urban（%）|',
        '|---|---:|---:|---|---|---:|']
    for arm,row in analysis['rows'].items():
        lines.append(f"|{arm}|{row['parameters']}|{row['reduction_from_R38_percent']:.4f}%|{row['acceptance']['R17']['accepted']}|{row['acceptance']['R38']['accepted']}|{row['performance_optimization']['low_urban_accuracy']:.6f}|")
    lines+=['','1pp是逐场景退化容差，不等于性能优化成功；另列原lowurban≥70%、其他五景距rawclean84.6761904762%≤5pp。单seed、用户指定的已暴露工程基准，不称新独立盲测或干净泛化证明。','',
        '|方法|场景|准确率（%）|Macro-F1（%）|相对R38（pp）|相对R17（pp）|raw净修复|',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm,row in analysis['rows'].items():
        for view,item in row['target']['views'].items():
            lines.append(f"|{arm}|{view}|{item['accuracy']:.6f}|{item['macro_f1']:.6f}|{row['target_vs_R38']['views'][view]['accuracy_delta_pp']:+.6f}|{row['target_vs_R17']['views'][view]['accuracy_delta_pp']:+.6f}|{item['net_repaired']:+d}|")
    lines+=['','|方法|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|同卡R38/本版median ms|resident/峰值字节|',
        '|---|---|---:|---:|---:|---|']
    for arm,row in analysis['rows'].items():
        model=artifacts['models'][arm];profile=row['profile']['results'];current=profile['compact'];base=profile['parent_R38_794529']
        lines.append('|'+arm+'|'+'/'.join(str(v) for v in model['components'].values())+f"|{model['bytes']}|{model['conv_linear_MACs_per_IQ']}|{base['median_ms']:.4f}/{current['median_ms']:.4f}|{current['resident_before_forward_bytes']}/{current['peak_allocated_bytes']}|")
    lines+=['','延迟为同卡16条sourceV、batch1、2次预热、完整context/三prior/六CG；顺序及并存负载影响计时。MAC仅Conv/Linear，不含FFT/复数CG/归一化/激活/访存。参数减少不预先推定加速或峰值显存降低。','']
    for arm,row in analysis['rows'].items():
        train=row['training'];perf=row['performance_optimization']
        lines+=[arm+'完整20轮首轮→末轮：'+'；'.join(k+f" {v['first']:.8g}→{v['last']:.8g}" for k,v in train['losses'].items())+
            f"。中位每轮{train['epoch_seconds_median']:.3f}s。lowurban70通过={perf['low_urban_at_least_70']}；其他五景gap5pp通过={perf['other_five_within_5pp_of_raw_clean']}。日志发现："+json.dumps(row['stdout_findings'],ensure_ascii=False),'']
    lines+=['全帧/主体232/尾24的paired-clean NMSE、FFT log-power差距、配对clean身份特征几何及全部TX/RX/day保留在JSON。识别改善或这些几何指标均不能证明每个新增细节属于输入TX。恢复器0TX训练与评分时独立打开标签分别说明。','',
        '祖先60串行pass+R38joint20+新增joint20=100；每组件累计60，组件轮次和180，原F50预训练另计。与R13E10/constant teacher1/LR1e-4并非同预算单因素比较。','',
        '两份独立恢复器已取回并CPU严格加载、有限synthetic IQ输出。路径：local_artifacts/pamf_restoration/r38_compression_methods_r44_20261020/models/<arm>/final.pt。使用experiments.pamf_r38_compression_methods.runtime.load_checkpoint，原生float32[B,2,256]输入/同形输出；接原F50/defaulthead。权重不入Git。长期总目标继续active，不由本轮产物完整代替总目标达成。','']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write(folder/'results.json',dict(analysis,model_artifacts=artifacts))
    records=[]
    for arm,row in analysis['rows'].items():
        for role in ('source','target'):
            for view,item in row[role]['views'].items():
                for dimension,groups in [('overall',{'all':item}),*item['groups'].items()]:
                    for label,metric in groups.items():
                        records.append(dict(arm=arm,role=role,view=view,dimension=dimension,value=label,
                            **{k:metric[k] for k in ('count','correct','accuracy','macro_f1','repaired','damaged','net_repaired')}))
    with (folder/'row_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
    spec['scientific_outcome']=dict(status='ANALYZED',all_fixed_methods_scored=True,
        accepted_R17=[a for a,r in analysis['rows'].items() if r['acceptance']['R17']['accepted']],
        accepted_R38=[a for a,r in analysis['rows'].items() if r['acceptance']['R38']['accepted']],goal_complete=False)
    write(folder/'experiment.json',spec)
    module=importlib.util.spec_from_file_location('registry',local/'tools/experiment_registry.py');registry=importlib.util.module_from_spec(module);module.loader.exec_module(registry)
    for arm in d.ARMS:registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'VERIFIED E20/4940;all7formaltruth-last;strictstandaloneIQmodel;allownedEXITED',row_id=arm)
    registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'Both different compression methods complete;negative results preserved;goalactive')
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):
        shutil.copyfile(folder/name,WORKSPACE/(folder/name).relative_to(local))
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,methods=list(d.ARMS),all7formal=True,goal_complete=False))
    print(json.dumps(dict(status='REPORT_AND_MODELS_VERIFIED',report=str(folder/'report.md'),goal_complete=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collect-only',action='store_true');args=parser.parse_args()
    local=Path('E:/type10-7');output=local/'local_artifacts/pamf_restoration/r38_compression_methods_r44_20261020'
    analysis=collect(local,output)
    if analysis is not None and not args.collect_only:report(local,output,analysis,retrieve(local,output))


if __name__=='__main__':main()
