"""Collect and deliver the existing R58 run; never launch or modify remote jobs."""
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
from experiments.pamf_r46_stat_transport import design as d
from tools.analyze_pamf_r46_stat_transport_results import completed_metrics, METHOD


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
   if state!='Z' and root.name in ' '.join(argv) and 'experiments.pamf_r46_stat_transport.runner' in argv:live.append(pid)
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
 control=root.parent/'20261021-phase1-restoration-R46-fold-compensate-s2026092701-m2-r55'
 assert read(control/'launch.json')['commit']=='6bea50605a1c2cfb49645b5c2f7ba8306b39791d' and read(control/'fold_full/complete.json')['status']=='ANALYZED'
 reused=dict(run=control.name,row='fold_full',method_commit='6bea50605a1c2cfb49645b5c2f7ba8306b39791d',
  training_records=[json.loads(line) for line in (control/'fold_full/training.jsonl').read_text().splitlines() if line.strip()],
  training_complete=read(control/'fold_full/training_complete.json'),
  source=read(control/'fold_full/native_map_prior/summary.json'),target=read(control/'fold_full/target_confirmation/native_map_prior/summary.json'))
 print(json.dumps(dict(run_id=root.name,method_commit=METHOD,complete=read(root/'complete.json'),failure=failure,reused_fold_control=reused,calibration=read(root/'calibration.json'),
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
    from experiments.pamf_r46_stat_transport.runtime import load_checkpoint
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
    lines=['# R58：压缩图可学习组统计校正正式结果','',
        '**VERIFIED / ANALYZED**。两固定行完整 E20/4940、source1800×7、正式168000×7；逐行七预测固定后独立 truth-last，所属进程实际退出。全部正负结果保留，不反馈调参或选择性重跑。','',
        '性能优先。直接初始化和冻结 IQ 教师保持人选 R46 prune_refine350521；复用 R55 source-only full-fold 统计，无 R55/R57 训练后学生继承。既有 Conv/U-Net/3prior6CG 内168个GN加入2688个可学习组统计参数；完整276453、context186959，所有原参数与新参数联合训练。两行图、参数、计算、预算相同，仅 calibrated 为源拟合初始化，identity_control 为恒等初始化。原参数继承80+20=100轮，新统计参数只训练20轮/4940次；不能把新参数也记为100轮。','',
        '源训练使用6300L_s+56700U_s的模拟受损/clean配对，E20每行L126000/U1134000，全局256/micro256，原IQ+.05谱误差+1→.1三阶段IQ教师，0TX标签/F权重/F输出/V/target训练。属于TX无标签源域自监督，使用clean配对；原冻结F50默认头仅评测同一恢复IQ及其FFT/PA/时域/统计分支。方案及核心原型在R57正式分数读取前固定。','',
        '方法 commit `'+METHOD+'`。见[登记](experiment.json)、[完整结果](results.json)和[逐组CSV](row_metrics.csv)。单seed、已暴露且人选父的工程基准，不称新的独立盲测。','',
        '|行|完整参数|较350521减参|R17每景1pp容差|父每景1pp容差|low urban≥70%|其余五景距rawclean≤5pp|',
        '|---|---:|---:|---|---|---|---|']
    for a,r in analysis['rows'].items():
        q=r['performance_optimization'];lines.append(f"|{a}|{r['parameters']}|{r['reduction_from_R46_percent']:.6f}%|{r['acceptance']['R17']['accepted']}|{r['acceptance']['R46']['accepted']}|{q['low_urban_at_least_70']}|{q['other_five_within_5pp_of_raw_clean']}|")
    lines+=['','容差、减参和局部拟合均不等于性能优化。raw clean同F50准确率84.6761904762%，其他五个受损场景门槛79.6761904762%。','',
        '|行|场景|准确率（%）|MacroF1（%）|较指定父（pp）|较原R17（pp）|raw净修复|','|---|---|---:|---:|---:|---:|---:|']
    for a,r in analysis['rows'].items():
        for v in d.VIEWS:
            q=r['target']['views'][v];lines.append(f"|{a}|{v}|{q['accuracy']:.6f}|{q['macro_f1']:.6f}|{r['target_vs_R46']['views'][v]['accuracy_delta_pp']:+.6f}|{r['target_vs_R17']['views'][v]['accuracy_delta_pp']:+.6f}|{q['net_repaired']}|")
    lines+=['','主要匹配对照：同一276453参数图的 calibrated 相对 identity_control；差别仅初始化。','',
        '|场景|准确率差（pp）|MacroF1差（pp）|','|---|---:|---:|']
    for v in d.VIEWS:
        c=analysis['same_graph_initialization_comparison']['target']['views'][v];lines.append(f"|{v}|{c['accuracy_delta_pp']:+.6f}|{c['macro_f1_delta_pp']:+.6f}|")
    lines+=['','R55 fold_full仅作旧归一化参考：Conv宽度相同，但273765参数与归一化计算不同，不能当作完整参数及计算匹配控制。','',
        '|行|场景|较R55 full-fold准确率差（pp）|','|---|---|---:|']
    for a,c in analysis['full_fold_reference_comparisons'].items():
        for v in d.VIEWS:lines.append(f"|{a}|{v}|{c['target']['views'][v]['accuracy_delta_pp']:+.6f}|")
    lines+=['','|行|coarse/action/prior参数|权重文件字节|Conv+Linear MAC/IQ|','|---|---|---:|---:|']
    for a,r in analysis['rows'].items():
        m=artifacts['models'][a];lines.append('|'+a+'|'+'/'.join(str(v) for v in m['components'].values())+f"|{m['bytes']}|{m['conv_linear_MACs_per_IQ']}|")
    lines+=['','|行|口径|同卡父/本版median ms|本版P95 ms|本版条/s|父/本版增量峰值字节|','|---|---|---|---:|---:|---|']
    for a,r in analysis['rows'].items():
        p=r['profile']['results'];parent=p['parent_R46_350521']['cases']
        for k,c in p['compact']['cases'].items():
            b=parent[k];lines.append(f"|{a}|{k}|{b['median_ms']:.4f}/{c['median_ms']:.4f}|{c['p95_ms']:.4f}|{c['throughput_samples_per_second']:.4f}|{b['incremental_peak_bytes']}/{c['incremental_peak_bytes']}|")
    lines+=['','同卡64条sourceV、B1/B32、三轮交替顺序、同步预加载GPU输入，G及G+原F50分别计时，包含手写组统计计算；排除传输/预处理/loader。驻留为三G+一F共享suite；增量峰值对应当前前向，不称独立部署显存。MAC仅Conv/Linear，排除FFT/CG/norm/激活/访存和F。参数减少不能推定提速；不同卡绝对时间不作排名。','']
    cal=analysis['shared_source_calibration'];old=analysis['reused_fold_calibration']
    lines+=['源校准一次共用128物理样本（64L_s/64U_s），received01/20，256父+256恒等子G输入，共512，耗时'+f"{cal['seconds']:.4f}s"+'；0clean/TX/F/V/target/optimizer，父及恒等子收集期间不变。原R55同源256父G的'+f"{old['seconds']:.4f}s"+'折叠统计复用单列，不重复记两份。','',
        '完整pre-GN组均值和log(var+eps)配对，每组256样本，无空间抽样；identity-centered ridge1e-3、斜率[.25,4]、自由截距。新参数在校准后与原参数联合训练，系数和索引不作推理输入。下表两种统计量单位不同，不合并为一种误差；局部拟合下降不证明全G或身份收益。','',
        '|统计量|module等权MSE前/后|触界斜率数|','|---|---|---:|']
    for k,q in cal['statistics'].items():lines.append(f"|{k}|{q['module_unweighted_MSE_before']:.8g}/{q['module_unweighted_MSE_after']:.8g}|{q['slope_clipped_count']}|")
    for a,r in analysis['rows'].items():
        t=r['training'];lines+=['',a+'首轮→末轮：'+'；'.join(k+f" {q['first']:.8g}→{q['last']:.8g}" for k,q in t['losses'].items())+f"；每轮中位{t['epoch_seconds_median']:.3f}s。日志类别："+json.dumps(r['stdout_findings'],ensure_ascii=False)]
    lines+=['','完整20轮曲线、逐TX/RX/day、修复/损伤、全帧/主体232/尾24重建误差和F配对clean几何均保留。原组件轮次和300/串行140pass仅描述原参数继承；新2688统计参数仅20轮，新增校准及F50训练另计。追加训练预算、初始化、归一化计算和压缩分别披露。','',
        '两套独立权重已取回并CPU严格重载，合成零IQ有限输出及参数核对通过。使用experiments.pamf_r46_stat_transport.runtime.load_checkpoint，float32[B,2,256]原生幅度输入/同形输出；恢复器无需教师/F/源数据/校准文件。权重、数据和原始日志不入Git。','',
        '识别率、IQ误差和特征几何不能单独证明每个恢复细节属于输入TX；原70%/5pp和TX细节保真目标保持ACTIVE。','']
    (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    public_artifacts=dict(artifacts,models={a:dict(m,path=Path(m['path']).relative_to(local).as_posix()) for a,m in artifacts['models'].items()})
    write(folder/'results.json',dict(analysis,model_artifacts=public_artifacts))
    records=[]
    for a,r in analysis['rows'].items():
        for role in ('source','target'):
            for view,q in r[role]['views'].items():
                for dim,groups in [('overall',{'all':q}),*q['groups'].items()]:
                    for label,m in groups.items():records.append(dict(arm=a,role=role,view=view,dimension=dim,value=label,**{k:m[k] for k in ('count','correct','accuracy','macro_f1','repaired','damaged','net_repaired')}))
    with (folder/'row_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    usage=['# R58 两套独立恢复器','','两行均完成E20/4940和168000×7正式测试，正负结果见公开报告。完整参数各276453，新统计参数2688仅训练20轮。CPU严格重载及合成零IQ检查通过；不据此声明逐细节TX真实性。','',
        '```python','import torch','from experiments.pamf_r46_stat_transport.runtime import load_checkpoint',
        'model, metadata = load_checkpoint("path/to/models/calibrated/final.pt", "cuda:0")','model.eval().requires_grad_(False)',
        'with torch.no_grad():','    restored_iq = model(received_iq)  # float32[B,2,256]原生幅度','```','',
        '权重相对路径：calibrated/final.pt、identity_control/final.pt。恢复器推理无需teacher/F/源数据/校准文件；身份识别将同一恢复IQ交给原冻结F50默认头及其FFT/PA/时域/统计分支。性能优先，时延、吞吐、显存与参数次要比较。','']
    (output/'models/README.md').write_text('\n'.join(usage),encoding='utf-8')
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'));spec['scientific_outcome']=dict(status='ANALYZED',all_fixed_rows_scored=True,
        accepted_R17=[a for a,r in analysis['rows'].items() if r['acceptance']['R17']['accepted']],accepted_R46=[a for a,r in analysis['rows'].items() if r['acceptance']['R46']['accepted']],goal_complete=False)
    write(folder/'experiment.json',spec)
    module=importlib.util.spec_from_file_location('registry',local/'tools/experiment_registry.py');registry=importlib.util.module_from_spec(module);module.loader.exec_module(registry)
    for a in d.ARMS:registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'VERIFIED E20/4940/all7formaltruthlast/strictstandaloneIQweights/ownedEXITED;original100/newstats20',row_id=a)
    registry.event(local,d.RUN,'ANALYZED',['report.md','results.json','row_metrics.csv'],'Both matched statistic-initialization rows complete;negative results preserved;goalactive')
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):
        dest=WORKSPACE/(folder/name).relative_to(local)
        if dest.resolve()!=(folder/name).resolve():shutil.copyfile(folder/name,dest)
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,methods=list(d.ARMS),all7formal=True,goal_complete=False))
    print(json.dumps(dict(status='REPORT_AND_MODELS_VERIFIED',report=str(folder/'report.md'),goal_complete=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--collect-only',action='store_true');args=parser.parse_args()
    local=Path('E:/type10-7');output=local/'local_artifacts/pamf_restoration/r46_stat_transport_r58_20261021'
    analysis=collect(local,output)
    if analysis is not None and not args.collect_only:report(local,output,analysis,retrieve(local,output))


if __name__=='__main__':main()
