"""Read-only final collection and local delivery; never launches remote jobs."""
import argparse,csv,importlib.util,json,re,shutil,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parents[1];sys.path.insert(0,str(WORK))
from experiments.pamf_r46_cuda_graph import design as d
from tools.analyze_pamf_r46_cuda_graph_results import completed_metrics,METHOD


def write(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def remote(local,code):
    p=subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe','-F',str(local/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],
        input=code.encode('utf-8'),capture_output=True,timeout=45)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf-8',errors='replace'))
    return json.loads(p.stdout.decode('utf-8'))


def collect(local,output):
    if (output/'analysis.json').exists():return json.loads((output/'analysis.json').read_text(encoding='utf-8'))
    s=json.loads((output/'deployment.json').read_text(encoding='utf-8'))['startup']
    prefix='\n'.join(k+'='+repr(v) for k,v in dict(RUN=s['run'],PID=s['pid'],METHOD=METHOD,ARMS=d.ARMS,R17=d.PARENT_RUN,R46=d.COMPACT_PARENT_RUN).items())+'\n'
    code=prefix+'''import json,os
from pathlib import Path
assert os.geteuid()!=0
root=Path(RUN)
def read(p):return json.loads(p.read_text())
failure=read(root/'failure.json') if (root/'failure.json').exists() else None
if failure or not (root/'complete.json').exists():
 print(json.dumps(dict(status='FAILED_TECHNICAL' if failure else 'PENDING',failure=failure,stages=read(root/'stages.json'))))
else:
 assert read(root/'launch.json')['commit']==METHOD
 live=[]
 for pid in {PID,*[w['pid'] for w in read(root/'workers.json')['workers']]}:
  p=Path('/proc')/str(pid)
  try:
   state=(p/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(p/'cmdline').read_bytes().decode().split(chr(0))
   if state!='Z' and RUN in argv and 'experiments.pamf_r46_cuda_graph.runner' in argv:live.append(pid)
  except FileNotFoundError:pass
 if live:
  print(json.dumps(dict(status='PENDING',actual_live_owned_PIDs=live)))
 else:
  rows={}
  for arm in ARMS:
   row=root/arm;source=row/'native_map_prior';target=row/'target_confirmation/native_map_prior';origin=read(row/'checkpoint_source.json')
   native=Path(origin['source_checkpoint']).parent
   v=dict(complete=read(row/'complete.json'),failure=read(row/'failure.json') if (row/'failure.json').exists() else None,
     checkpoint_source=origin,smoke=read(row/'smoke.json'),runtime_ready=read(row/'runtime_ready.json'),profile=read(row/'matched_inference_profile.json'),
     standalone_graph_memory=read(row/'standalone_graph_memory.json'),stdout=(root/'logs'/('worker-'+arm+'.log')).read_text(errors='replace'))
   for role,folder,old in (('source',source,native),('target',target,native.parent/'target_confirmation/native_map_prior')):
    v[role+'_summary']=read(folder/'summary.json');v[role+'_prediction_complete']=read(folder/'prediction/complete.json')
    v[role+'_agreement']=read(folder/'native_agreement.json');v['native_'+role+'_summary']=read(old/'summary.json')
   rows[arm]=v
  baselines={name:{role:read(root.parent/run/arm/('target_confirmation/' if role=='target' else '')/'native_map_prior/summary.json') for role in ('source','target')}
    for name,run,arm in (('R17',R17,'unroll_unshared'),('R46',R46,'prune_refine'))}
  print(json.dumps(dict(run_id=root.name,method_commit=METHOD,complete=read(root/'complete.json'),failure=None,actual_live_owned_PIDs=live,rows=rows,baselines=baselines)))
'''
    raw=remote(local,code)
    if raw.get('status') in ('PENDING','FAILED_TECHNICAL'):
        print(json.dumps(raw));return None
    for row in raw['rows'].values():
        row['stdout_findings']=[name for name,pattern in (('Traceback',r'Traceback'),('OOM',r'out of memory'),('warning',r'warning'),('nonfinite',r'\b(?:nan|inf)\b')) if re.search(pattern,row['stdout'],re.I)]
    write(output/'collections'/(str(time.time_ns())+'.json'),raw)
    analysis=completed_metrics(raw);write(output/'analysis.json',analysis)
    return analysis


def retrieve(local,output):
    saved=output/'model_artifacts.json'
    if saved.exists():return json.loads(saved.read_text(encoding='utf-8'))
    import torch
    from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
    from experiments.pamf_r46_conditional_factor.runtime import load_checkpoint as load_factor
    from experiments.pamf_r46_cuda_graph.program import outputs
    from experiments.pamf_iq_joint_compact.models import convolution_linear_macs
    raw=json.loads(sorted((output/'collections').glob('*.json'))[-1].read_text(encoding='utf-8'))
    torch.set_num_threads(2);models={}
    for arm in d.ARMS:
        path=local/'local_artifacts/pamf_restoration'/('r44_pruned_refine_r46_20261020' if arm==d.ARMS[0] else 'r46_conditional_factor_20261020_r18')/'models'/d.SOURCE_ROWS[arm]/'final.pt'
        if not path.is_file() or path.stat().st_size!=raw['rows'][arm]['checkpoint_source']['checkpoint_bytes']:
            raise ValueError('Original already-retrieved full checkpoint missing; preserve and reconcile')
        model,payload=(load_parent if arm==d.ARMS[0] else load_factor)(path,'cpu')
        if (payload['arm'],payload['epoch'],payload['updates'])!=(d.SOURCE_ROWS[arm],*d.SOURCE_BUDGET[arm]) or payload['optimizer'] is not None or payload['training_F_access']:
            raise ValueError('Wrong original standalone full checkpoint')
        model.eval().requires_grad_(False)
        components=dict(coarse=sum(p.numel() for p in model.frozen.coarse.parameters()),action=sum(p.numel() for p in model.frozen.estimator.parameters()),prior=sum(p.numel() for p in model.prior.parameters()))
        with torch.no_grad():
            x=torch.zeros(1,2,256);native=model(x);static=outputs(model,x)[0]
            if not torch.equal(native,static) or not torch.isfinite(static).all() or sum(components.values())!=d.PARAMETERS[arm]:
                raise ValueError('Loaded original and static arithmetic disagree')
            macs=convolution_linear_macs(model,x)
        expected=raw['rows'][arm]['profile']['results']['parent_R46_350521' if arm==d.ARMS[0] else 'candidate']['ConvLinear_MAC_per_IQ']
        if macs!=expected:raise ValueError('Complete CNN/Linear cost differs')
        models[arm]=dict(path=str(path),bytes=path.stat().st_size,parameters=d.PARAMETERS[arm],components=components,
            ConvLinear_MAC_per_IQ=macs,strict_load=True,native_static_zero_IQ_equal=True,original_weights_reused=True,
            new_training_updates=0,optimizer_included=False,training_F_access=False,actual_IQ_reads=0)
    value=dict(status='VERIFIED',models=models,new_weight_copies=0,method_commit=METHOD)
    write(saved,value);return value


def report(local,output,analysis,artifacts):
    folder=local/'automation_reports/CV-SincNet'/d.RUN
    initial=(folder/'report.md').read_text(encoding='utf-8').split('\n## 完整正式结果\n')[0]
    initial=re.sub(r'状态：[^\n]+','状态：VERIFIED / ANALYZED。',initial,count=1)
    lines=['\n## 完整正式结果\n','三行均完成source1800×7和formal168000×7，逐行七预测固定后独立CPU truth-last；实际controller/三worker退出，无技术失败。三份原完整权重本地复用并严格加载，静态路径与原函数零IQ逐位一致，无新增权重或训练。\n',
      '|版本|完整参数|较350521减参|父逐景0.8pp|目标原native决策全一致|70%/5pp|','|---|---:|---:|---|---|---|']
    for arm,r in analysis['rows'].items():
        p=r['performance_optimization']
        lines.append(f"|{arm}|{r['parameters']}|{r['reduction_from_R46_percent']:.6f}%|{r['acceptance']['R46']['accepted']}|{r['agreement']['target']['all_argmax_equal']}|{p['low_urban_at_least_70']}/{p['other_five_within_5pp_of_raw_clean']}|")
    lines+=['','|版本|场景|准确率%|MacroF1%|较350521pp|较自身原native pp|raw净纠错|','|---|---|---:|---:|---:|---:|---:|']
    for arm,r in analysis['rows'].items():
        for view,v in r['target']['views'].items():
            lines.append(f"|{arm}|{view}|{v['accuracy']:.6f}|{v['macro_f1']:.6f}|{r['target_vs_R46']['views'][view]['accuracy_delta_pp']:+.6f}|{r['target_vs_own_native']['views'][view]['accuracy_delta_pp']:+.6f}|{v['net_repaired']:+d}|")
    lines+=['','0.8pp为逐场景退化容忍上限，不是性能优化目标。原R17历史1pp、逐TX/RX/day、clean损伤、波形/几何、全部原native差异和完整成本见results.json/CSV。单seed已暴露工程基准；不称新盲测、显著性结论或逐细节TX证明。',
        '', '|版本|口径|父native ms|本版native ms|父graph ms|本版graph ms|本版native→graph倍数|父graph→本版graph倍数|','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in analysis['rows'].items():
        p=r['profile']['raw']['results'];parent=p['parent_R46_350521']['cases'];cur=p['parent_R46_350521' if arm==d.ARMS[0] else 'candidate']['cases']
        for scope,c in r['profile']['comparisons'].items():
            lines.append(f"|{arm}|{scope}|{parent[scope+'_native']['median_ms']:.6f}|{cur[scope+'_native']['median_ms']:.6f}|{parent[scope+'_graph']['median_ms']:.6f}|{cur[scope+'_graph']['median_ms']:.6f}|{c['candidate_native_to_graph_median_speedup']:.4f}|{c['parent_graph_to_candidate_graph_median_speedup']:.4f}|")
    lines+=['','同卡64sourceV、三轮交替顺序；图计时包含输入copy、输出clone，预热/捕获单列，重放后数值检查按本候选调用位置执行。G含完整coarse/action/3prior/6CG，G+F含原默认头；输入预加载GPU，不含传输/loader/预处理。native原函数内部主机检查与图专用路径重放后检查的位置不同，实测改进同时包含该执行安排与图重放；不单独归因于kernel launch。两卡存在其他owner负载，不能以跨卡绝对时间比较结构。父native与父graph两套对照均保留，不能把父版保持native的差距全称为结构收益。',
        '', '|版本|checkpoint字节|ConvLinearMAC/IQ|单模型G+F五批量graph后allocated MiB|reserved MiB|capture peak MiB|','|---|---:|---:|---:|---:|---:|']
    for arm,r in analysis['rows'].items():
        m=r['standalone_graph_memory'];key='parent_R46_350521' if arm==d.ARMS[0] else 'candidate'
        lines.append(f"|{arm}|{r['checkpoint_bytes']}|{r['profile']['raw']['results'][key]['ConvLinear_MAC_per_IQ']}|{m['after_capture_allocated_bytes']/2**20:.3f}|{m['after_capture_reserved_bytes']/2**20:.3f}|{m['capture_peak_allocated_bytes']/2**20:.3f}|")
    lines+=['','上述独立部署内存含一个完整G+F及批量1/8/32/64/256的五套graph池，不能当作单批量B1内存，也不能把捕获前仅权重的allocated当作native峰值。匹配计时suite的驻留/peak/incremental、p90/p95、吞吐、各轮中位和每批量setup秒数全部保留于results.json。MAC仅Conv1d+Linear，不包括FFT/CG/norm/激活/访存/F。图不减少权重参数；27.82万减参来自已经固定的R18。',
        '', '|版本|场景|argmax差异条数|IQ最大差|features最大差|logits最大差|','|---|---|---:|---:|---:|---:|']
    for arm,r in analysis['rows'].items():
        for view,v in r['agreement']['target']['views'].items():
            lines.append(f"|{arm}|{view}|{v['argmax_disagreements']}|{v['iq']['max_abs_difference']:.9g}|{v['features']['max_abs_difference']:.9g}|{v['logits']['max_abs_difference']:.9g}|")
    lines+=['','原native预测复用，batch128与新256可能带来浮点差异；不同数值不自动等同于不同分类。逐位/分类一致性分开报告，任何差异都未回流改模型、rank或重跑目标。',
        '', '部署恢复器可加载原R46/R18完整权重，使用GraphCache(FrozenPipeline(G,graph_safe=True))(IQ)[0]，无需F或teacher；首次各批量捕获及显存开销必须计入具体部署方案。F只在本轮冻结评测的G+F路径使用。',
        '', '[完整结果](results.json) · [逐TX/RX/day CSV](row_metrics.csv)']
    (folder/'report.md').write_text(initial+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    write(folder/'results.json',analysis)
    with (folder/'row_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arm','role','view','dimension','group','count','accuracy','macro_f1','repaired','damaged','net_repaired'])
        for arm,r in analysis['rows'].items():
            for role in ('source','target'):
                for view,v in r[role]['views'].items():
                    groups=[('all','all',v)]+[(dim,name,g) for dim,values in v['groups'].items() for name,g in values.items()]
                    for dim,name,g in groups:writer.writerow([arm,role,view,dim,name,*[g[k] for k in ('count','accuracy','macro_f1','repaired','damaged','net_repaired')]])
    spec=json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
    spec['scientific_outcome']=dict(status='ANALYZED',every_version_full7formal=True,models_strict_loaded=3,
        accepted_rows=analysis['accepted_rows'],all_target_native_decisions_equal={a:r['agreement']['target']['all_argmax_equal'] for a,r in analysis['rows'].items()},
        new_training_updates=0,target_feedback=False,long_term_goal_complete=analysis['long_term_goal_complete'])
    spec['execution']['postlaunch_state']='VERIFIED_ALL_OWNED_HANDLES_EXITED';write(folder/'experiment.json',spec)
    ms=importlib.util.spec_from_file_location('registry',local/'tools/experiment_registry.py');reg=importlib.util.module_from_spec(ms);ms.loader.exec_module(reg)
    for arm in d.ARMS:reg.event(local,d.RUN,'ANALYZED',['experiment.json','report.md','results.json','row_metrics.csv'],
        'All3fixedweight graph versionssource7/formal7;independenttruthlast;actualownedhandlesexited;originalfullmodelsstrictloaded',row_id=arm)
    mirror=WORK/folder.relative_to(local);mirror.mkdir(exist_ok=True,parents=True)
    for name in ('experiment.json','report.md','events.jsonl','results.json','row_metrics.csv'):shutil.copy2(folder/name,mirror/name)
    doc=local/'docs/PAMF_R46_CUDA_GRAPH_R20_20261021.md'
    doc.write_text((folder/'report.md').read_text(encoding='utf-8').replace('](results.json)',f'](../automation_reports/CV-SincNet/{d.RUN}/results.json)').replace('](row_metrics.csv)',f'](../automation_reports/CV-SincNet/{d.RUN}/row_metrics.csv)'),encoding='utf-8')
    shutil.copy2(doc,WORK/doc.relative_to(local))
    write(output/'public_results_verified.json',dict(status='VERIFIED',run=d.RUN,all_versions_full7formal=True,models_strict_loaded=3))


def main():
    p=argparse.ArgumentParser();p.add_argument('--local',type=Path,default=Path('E:/type10-7'));p.add_argument('--output',type=Path,default=Path('E:/type10-7/local_artifacts/pamf_restoration/r46_cuda_graph_20261021_r20'));args=p.parse_args()
    analysis=collect(args.local,args.output)
    if analysis is None:return
    artifacts=retrieve(args.local,args.output);report(args.local,args.output,analysis,artifacts)
    print(json.dumps(dict(status='VERIFIED_ANALYZED',accepted_rows=analysis['accepted_rows'],all_versions_full7formal=True,models_strict_loaded=3)))


if __name__=='__main__':main()
