"""Descriptive source-only audit report, no target loading or candidate selection."""
import argparse,csv,statistics as st
from pathlib import Path
from . import design as d
from .collect import validate

def mean(v):
 v=[x for x in v if x is not None];return st.mean(v) if v else None

def fmt(x,n=4):return 'N/A' if x is None else f'{x:.{n}f}'

def angle(row,a,b):
 values=row['pairwise_cosine'];return values.get(a+'|'+b,values.get(b+'|'+a))

def report(raw):
 v=validate(raw);records=[];baselines=[];controls=[]
 for row in v['rows']:
  seed=row['row']['model_seed']
  for name,m in row['models'].items():
   a=m['audit']
   for r in a['records']:
    for group,b in r['baseline_directions'].items():
     baselines.append(dict(seed=seed,model=name,round=r['round'],day=r['day'],group=group,clean_CE=r['clean']['ce'],clean_accuracy=r['clean']['accuracy'],clean_margin=r['clean']['margin_mean'],fishr_CE_cos=angle(b,'clean_CE','Fishr_both'),fishr_shallow_middle_cos=angle(b,'Fishr_shallow','Fishr_middle'),fishr_norm=b['norms']['Fishr_both'],CE_norm=b['norms']['clean_CE']))
    for s in r['styles']:
     q=s['comparison'];style=s['style']
     for group,g in s['directions'].items():
      ce_norm=g['norms']['clean_CE'];inc_norm=g['norms']['style_increment']
      records.append(dict(seed=seed,model=name,round=r['round'],day=r['day'],variant=s['variant'],group=group,
       clean_ce=r['clean']['ce'],styled_ce=q['styled']['ce'],ce_increase=q['styled']['ce']-r['clean']['ce'],clean_accuracy=r['clean']['accuracy'],styled_accuracy=q['styled']['accuracy'],accuracy_delta_pp=100*(q['styled']['accuracy']-r['clean']['accuracy']),
       clean_margin=r['clean']['margin_mean'],styled_margin=q['styled']['margin_mean'],margin_delta=q['styled']['margin_mean']-r['clean']['margin_mean'],repaired=q['repaired'],broken=q['broken'],changed=q['changed_prediction'],
       active_layers=sum(s0.get('changed_packets',0)>0 for s0 in style['levels'].values()),changed_packet_layers=sum(s0.get('changed_packets',0) for s0 in style['levels'].values()),
       style_CE_cos=angle(g,'clean_CE','styled_CE'),increment_CE_cos=angle(g,'clean_CE','style_increment'),increment_Fishr_cos=angle(g,'Fishr_both','style_increment'),combined_CE_cos=angle(g,'clean_CE','Fishr_both_plus_style_increment'),
       increment_CE_norm_ratio=inc_norm/ce_norm if ce_norm>0 else None,increment_norm=inc_norm,CE_norm=ce_norm))
   for c in a['dropout_controls']:
    for s in c['variants']:
     controls.append(dict(seed=seed,model=name,day=c['day'],variant=s['variant'],replay_exact=c['same_mask_replay_exact'],replay_close=c.get('same_mask_replay_close',c['same_mask_replay_exact']),same_mask_changed=s['same_mask_comparison']['changed_prediction'],independent_mask_changed=c['independent_mask']['changed_prediction'],style_delta_rms=s['style_delta_rms'],mask_delta_rms=s['mask_delta_rms'],interaction_delta_rms=s['interaction_delta_rms']))
 lines=['# 浅中层受控风格与真实梯度方向：三种子源诊断','',
  '**三个 worker、六个固定模型全部完成。身份参数零更新，未读取目标数据或评分。** 这是源域机制诊断，不是新的测试集成绩，也不表示 84% 目标已完成。','',
  'P0 为原生纯 CE＋旧门控＋cos E20；CE20 为固定纯 CE 续训 20 轮。两者按对照角色全量纳入，未按目标排名选择。每个模型在冻结身份特征上暖头 200 步、L 校准 12 对批次，再执行 36 对源 L 双批；每对包含同日均衡 TX×RX 结构，六个固定风格变体。源 RX 均已见过，不能称为未知 RX 泛化。','',
  '所有下表均由完整固定诊断表汇总。梯度 cosine 大于零表示两个损失当前局部下降方向相容，小于零表示局部冲突；它不是目标泛化改善的充分条件。','',
  '|模型|参数组|Fishr 与主 CE 平均 cosine|浅／中 Fishr cosine|Fishr／CE 范数比均值|有效配对数|',
  '|---|---|---:|---:|---:|---:|']
 for name in ('P0','CE20'):
  for group in ('full_identity','time_down','t2.pool'):
   rows=[r for r in baselines if r['model']==name and r['group']==group]
   lines.append(f"|{name}|{group}|{fmt(mean([r['fishr_CE_cos'] for r in rows]))}|{fmt(mean([r['fishr_shallow_middle_cos'] for r in rows]))}|{fmt(mean([r['fishr_norm']/r['CE_norm'] for r in rows if r['CE_norm']>0]))}|{len(rows)}|")
 lines+=['','## 主 eval 对照：身份变化及风格独有梯度','',
  '主对照关闭 dropout，clean 与 styled 都用相同的 120 个样本出现位置；没有混入 split60/combined120 数值差。修复／破坏按样本出现次数统计，不是独立物理样本总数。增量梯度是 ∇CE_style−∇CE_clean。准确率、margin、修复／破坏保留全部108对；梯度方向与范数比分布只解释实际发生扰动的行（changed_packets>0），未扰动行单列数值控制。','',
  '|模型|位置@强度|准确率变化（百分点）|margin 变化|修复／破坏|有效增量／CE 范数比中位数／最大|有效增量与 CE cosine|有效增量与 Fishr cosine|实际扰动数／108|',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
 variants=sorted({r['variant'] for r in records})
 for name in ('P0','CE20'):
  for variant in variants:
   rows=[r for r in records if r['model']==name and r['variant']==variant and r['group']=='full_identity']
   active=[r for r in rows if r['active_layers']>0];ratios=[r['increment_CE_norm_ratio'] for r in active if r['increment_CE_norm_ratio'] is not None]
   lines.append(f"|{name}|{variant}|{fmt(mean([r['accuracy_delta_pp'] for r in rows]))}|{fmt(mean([r['margin_delta'] for r in rows]))}|{sum(r['repaired'] for r in rows)}／{sum(r['broken'] for r in rows)}|{fmt(st.median(ratios) if ratios else None)}／{fmt(max(ratios) if ratios else None)}|{fmt(mean([r['increment_CE_cos'] for r in active]))}|{fmt(mean([r['increment_Fishr_cos'] for r in active]))}|{len(active)}/108|")
 lines+=['','### 未实际扰动的数值控制','',
  '不改变或删除原始梯度记录。未扰动行 logits 和 CE 均严格相同；重复反传存在微小浮点残差，其归一化方向不具有风格含义。下面单列残差范数，原始 cosine 保留于 CSV，但不进入有效风格方向均值。','',
  '|模型|未扰动行数|增量／CE 范数比最大值|logits 最大绝对差|CE 最大绝对差|',
  '|---|---:|---:|---:|---:|']
 for name in ('P0','CE20'):
  rows=[r for r in records if r['model']==name and r['group']=='full_identity' and r['active_layers']==0]
  deltas=[]
  for row in v['rows']:
   for r in row['models'][name]['audit']['records']:
    for q in r['styles']:
     if not any(layer.get('changed_packets',0)>0 for layer in q['style']['levels'].values()):deltas.append(q['comparison']['max_abs_logit_delta'])
  lines.append(f"|{name}|{len(rows)}|{max(r['increment_CE_norm_ratio'] for r in rows):.4g}|{max(deltas):.4g}|{max(abs(r['ce_increase']) for r in rows):.4g}|")
 lines+=['','## 每个种子的独立结果','',
  '|模型|seed|位置@强度|准确率变化（百分点）|增量与 CE cosine|增量与 Fishr cosine|',
  '|---|---:|---|---:|---:|---:|']
 for name in ('P0','CE20'):
  for seed in d.SEEDS:
   for variant in variants:
    rows=[r for r in records if r['model']==name and r['seed']==seed and r['variant']==variant and r['group']=='full_identity']
    active=[r for r in rows if r['active_layers']>0]
    lines.append(f"|{name}|{seed}|{variant}|{fmt(mean([r['accuracy_delta_pp'] for r in rows]))}|{fmt(mean([r['increment_CE_cos'] for r in active]))}|{fmt(mean([r['increment_Fishr_cos'] for r in active]))}|")
 lines+=['','## 同随机数 dropout 对照','',
  '每个模型每 day 一组；仅 dropout 模块使用训练模式，具有持久统计的模块保持 eval。风格与 dropout 的增量分别测量，并记录交互；不假设两者可以线性相加。单一模型／天的结果只用于这一固定局部对照。','',
  '|模型|位置@强度|风格同 mask 改判次数|独立 mask 改判次数|风格 delta RMS|mask delta RMS|交互 delta RMS|',
  '|---|---|---:|---:|---:|---:|---:|']
 for name in ('P0','CE20'):
  for variant in variants:
   rows=[r for r in controls if r['model']==name and r['variant']==variant]
   lines.append(f"|{name}|{variant}|{sum(r['same_mask_changed'] for r in rows)}|{sum(r['independent_mask_changed'] for r in rows)}|{fmt(mean([r['style_delta_rms'] for r in rows]))}|{fmt(mean([r['mask_delta_rms'] for r in rows]))}|{fmt(mean([r['interaction_delta_rms'] for r in rows]))}|")
 lines+=['','## 完整性与实际成本','',
  f"核验 {v['gradient_pairs']} 对批次、{v['style_trials']} 个主风格诊断单元、{v['dropout_controls']} 组 dropout 对照（108 个变体）。主诊断与 dropout 合计 {v['diagnostic_model_forwards']} 次模型前向、{v['diagnostic_L_readouts']} 次 L 样本读出；暖头额外 {v['warmup_L_readouts']}，校准额外 {v['calibration_L_readouts']}，合计 {v['total_L_readouts_excluding_preflight']}。发布前 CPU 无 query smoke 另读 6 次 L，未计入上述诊断成本。",'',
  '|seed|模型|诊断秒|暖头秒|校准秒|峰值显存 MiB|全身份参数|',
  '|---:|---|---:|---:|---:|---:|---:|']
 for row in v['rows']:
  for name,m in row['models'].items():
   a=m['audit'];peak=m['summary']['peak_cuda_memory_bytes'];lines.append(f"|{row['row']['model_seed']}|{name}|{a['seconds']:.2f}|{m['warm']['seconds']:.2f}|{m['calibration']['seconds']:.2f}|{fmt(peak/1024**2 if peak is not None else None,2)}|{a['group_parameters']['full_identity']}|")
 lines+=['','每个模型完整 state_dict、既有参数梯度、训练标志及 CPU/CUDA RNG 均保持。诊断零 optimizer 更新；仅预热读出头共 1,200 次更新，头不保存为新身份 checkpoint。','',
  '全身份组包含实际 id_backbone 的所有可训练参数及 Hz／分类头；浅中组沿用既有 Fishr 标准参数组（不含 Hz）。成本为本次共享 GPU 墙钟，不等于独占硬件速度。完整统计 [JSON](evidence/readback.json)，逐对梯度 [CSV](evidence/gradient_trials.csv)，dropout [CSV](evidence/dropout_controls.csv)。','',
  '固定 0.3／1.0 是预登记诊断强度，不通过本报告挑选配置。下一步设计需来自源证据与物理假设，另行冻结实验；不得从历史或本轮目标评分、类别错误、排名回流研发。']
 return '\n'.join(lines)+'\n',dict(style=records,baseline=baselines,dropout=controls)

def write(output):
 output=Path(output);v=d.read(output/'evidence/readback.json');text,tables=report(v)
 (output/'analysis.md').write_text(text,encoding='utf-8')
 for kind,name in [('style','gradient_trials'),('baseline','baseline_gradients'),('dropout','dropout_controls')]:
  with (output/'evidence'/f'{name}.csv').open('w',encoding='utf-8',newline='') as out:
   rows=tables[kind];w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return tables

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=write(a.output);print({k:len(v) for k,v in t.items()})
