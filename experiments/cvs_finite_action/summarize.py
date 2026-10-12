"""Read every closed auxiliary epoch; source-only matrix report, no target."""
import csv,json,math,time,statistics
from . import design as d
def analyze(base):
 from pathlib import Path
 base=Path(base);rows=[]
 for row in d.rows():
  c=d.config(row);p=base/row['row_id'];done=d.read(p/'completion.json');result=d.read(p/'results.json');resolved=d.read(p/'resolved_config.json')
  if done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or done['config']!=c or result['config']!=c or resolved['config']!=c or done['identity_updates'] or done['head_updates'] or done['target_access'] or result['U_fit'] or result['U_truth'] or result['V_forwards'] or result['target_access']:raise ValueError('Source action completion differs')
  if len(result['fits'])!=117 or len(result['records'])!=630 or len(result['coverage'])!=45 or done['budget']['packet_occurrences']!=491400 or not done['budget']['reference_parameters_buffers_modes_gradients_unchanged']:raise ValueError('Incomplete finite matrix')
  epochs=[json.loads(line) for line in (p/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()];assert len(epochs)==2340
  with (p/'epoch_metrics.csv').open(encoding='utf-8',newline='') as stream:csvrows=list(csv.DictReader(stream))
  if len(csvrows)!=len(epochs) or any(any(str(value)!=csvrow[key] for key,value in record.items()) for record,csvrow in zip(epochs,csvrows)):raise ValueError('Auxiliary CSV/JSON logs differ')
  groups={}
  for v in epochs:
   key=(v['split'],v['kind'],v['level']);groups.setdefault(key,[]).append(v)
   if not all(not isinstance(value,float) or math.isfinite(value) for value in v.values()):raise ValueError('Nonfinite auxiliary log')
  updates=0
  for fit in result['fits']:
   key=tuple(fit[k] for k in ('split','kind','level'));log=groups[key];expected=c['fit_epochs']*math.ceil(fit['fit_observations']/c['fit_batch_size'])
   if [v['epoch'] for v in log]!=list(range(1,21)) or log[-1]['optimizer_steps']!=expected or fit['optimizer_steps']!=expected or not fit['fit_eval_PID_disjoint'] or fit['U_fit'] or not fit['fit_normalization_only']:raise ValueError('Incomplete or unlawful auxiliary fitting')
   updates+=expected
  if updates!=done['auxiliary_optimizer_steps'] or updates!=117180:raise ValueError('Fixed auxiliary budget differs')
  source_lines=0;closed=False
  with (base/'logs'/f"{row['row_id']}.log").open(encoding='utf-8') as stream:
   for line in stream:
    source_lines+=1
    if any(v in line for v in ('Traceback','CUDA out of memory','FloatingPointError','RuntimeError')):raise ValueError('Source stdout error')
    if line.startswith('SOURCE_DIAGNOSTIC_COMPLETE '):closed=True;break
  if not closed:raise ValueError('Source stdout not closed')
  rows.append(dict(row_id=row['row_id'],model_seed=row['model_seed'],completion=done,result=result,epoch_groups={':'.join(k):v for k,v in groups.items()},source_stdout_lines=source_lines))
 return dict(status='COMPLETE',run_id=d.RUN,rows=rows,source_only=True,identity_updates=0,target_access=False,U_truth=False,auxiliary_fits=351,auxiliary_optimizer_steps=sum(r['completion']['auxiliary_optimizer_steps'] for r in rows),source_model_packet_occurrences=1474200,reference_model_seeds=list(d.SEEDS))
def report(v):
 lines=['# 有限作用源端迁移：完整诊断','',f"{len(v['rows'])}/3种子；351个独立作用拟合，{v['auxiliary_optimizer_steps']}次辅助更新，1474200次实际源模型样本读出。身份更新0，U拟合0，V前向/训练0，target访问0。",'', '三个作用及三个实际层分别建网；已知频谱系数、分数延迟或噪声幅度输入，不学习自由状态。模型参考均为固定noCFO源CE E20。各层/种子的原编码器已见全部源RX，以下是作用模型留出迁移，不是身份测试准确率。','', '|作用|层|留出条件|zero|参数线性|双线性ridge|低秩8|MLP|','|---|---|---|---:|---:|---:|---:|---:|']
 for kind in d.KINDS:
  for level in d.LEVELS:
   for split in d.config(d.rows()[0])['splits']+['U_coverage']:
    values=[]
    for method in d.config(d.rows()[0])['methods']:
     records=[r for row in v['rows'] for r in row['result']['records'] if r['kind']==kind and r['level']==level and r['split']==split and r['method']==method]
     if len(records)!=3:raise ValueError('Missing source stratum seed')
     numbers=[r['normalized_repeat_MSE'] for r in records];values.append('N/A' if any(n is None for n in numbers) else f'{statistics.mean(numbers):.6f}')
    lines.append('|'+kind+'|'+level+'|'+split+'|'+'|'.join(values)+'|')
 lines+=['','数值为重复实现期望误差／真实有限差分总能量，越小越好；noise同时包括4次实现的样本内方差，不能把随机不可预测能量全归为预测失败。绝对误差、均值误差、方差/NLL、低能量退化、原型未配对RX残差覆盖与完整逐seed记录见analysis.json；不依据目标成绩决定新增身份约束。','', '|seed|源读出秒|全部辅助拟合秒|峰值显存MiB|辅助更新|观测缓存字节|','|---|---:|---:|---:|---:|---:|']
 for row in v['rows']:
  c=row['completion'];lines.append(f"|{row['model_seed']}|{c['observations_seconds']:.2f}|{c['seconds']-c['observations_seconds']:.2f}|{c['peak_cuda_memory_bytes']/1024**2:.2f}|{c['auxiliary_optimizer_steps']}|{c['source_observations_bytes']}|")
 lines+=['','每作用/层/留出独立20次fit遍历；只用L拟合。质量组采用全L coherence排序三分组，仅定义源分层，模型标准化和低秩基底均只在fit ID估计；不是SNR，也不证明重复前导。U只做同包已知操作读出，无TX标签/伪标签对。不同RX真实残差为独立L条件均值统计，不是逐包因果反事实。','', '前导任务N/A：未获得当前256窗口协议确认的重复区间；不凭lag16自相关添加前导损失。不称本批完成TX/RX解耦，更不称达成84%身份目标。全部正负结果保留，后续身份联合训练必须由这些源证据提出独立登记。']
 return '\n'.join(lines)+'\n'
def finish():
 v=analyze(d.BASE);text=report(v);d.write(d.BASE/'analysis.json',v);(d.BASE/'final_analysis.md').write_text(text,encoding='utf-8');(d.BASE/'report.md').write_text(text,encoding='utf-8')
 spec=d.read(d.BASE/'experiment.json');spec['status']='ANALYZED'
 for row in spec['rows']:row['status']='SOURCE_DIAGNOSTIC_COMPLETE'
 d.write(d.BASE/'experiment.json',spec)
 with (d.BASE/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(event='SOURCE_DIAGNOSTIC_ANALYZED',time=time.time(),identity_updates=0,target_access=False))+'\n')
 return dict(status='VERIFIED',rows=3,auxiliary_fits=351)
