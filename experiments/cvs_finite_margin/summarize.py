import statistics as st
from . import design as d
def analyze(base):
 values=[]
 for row in d.rows():
  v=d.read(base/row['row_id']/'completion.json');c=d.config(row)
  if v['config']!=c or v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['identity_updates']!=4440 or v['target_access'] or v['U_opened'] or v['checkpoint_sources'] or v['extra_mode_updates_normalization'] or len(v['epochs'])!=20:raise ValueError('Finite margin source row incomplete')
  if v['main_L_packets']!=568320 or v['extra_L_packets']!=710400 or [r['epoch'] for r in v['epochs']]!=list(range(1,21)):raise ValueError('Actual finite margin forward/update budget differs')
  if len(v['metrics']['V_views'])!=5:raise ValueError('Missing fixed finite stress source views')
  for m in [v['metrics']['L_held'],*v['metrics']['V_views']]:
   cm=m['confusion'];count=sum(map(sum,cm));a=sum(cm[i][i] for i in range(6))/count
   if count!=m['count'] or abs(a-m['accuracy'])>1e-7 or len(m['class_pairs'])!=15:raise ValueError('Finite source metric recount')
  values.append(dict(**row,metrics=v['metrics'],elapsed_seconds=v['elapsed_seconds']))
 result=[]
 for arm in d.ARMS:
  rows=[r for r in values if r['arm']==arm]
  result.append(dict(arm=arm,clean_mean=st.mean(r['metrics']['V_views'][0]['accuracy'] for r in rows),clean_worst_RX=min(r['metrics']['V_views'][0]['accuracy'] for r in rows),stress_mean=st.mean(m['accuracy'] for r in rows for m in r['metrics']['V_views'][1:]),per_RX={str(r['held_rx']):r['metrics']['V_views'][0]['accuracy'] for r in rows}))
 lookup={r['arm']:r for r in result};margin=lookup['FINITE_MARGIN'];action=lookup['ACTION_CE'];replay=lookup['CLEAN_REPLAY'];rule=d.config(d.rows()[0])['scientific_rule']
 dif=[margin['per_RX'][str(r)]-action['per_RX'][str(r)] for r in d.RX]
 eligible=margin['clean_mean']-action['clean_mean']>=rule['mean_gain_ACTION_CE'] and sum(v>=0 for v in dif)>=rule['nonnegative_RX'] and margin['clean_mean']>=replay['clean_mean']-rule['clean_mean_noninferiority_REPLAY'] and margin['clean_worst_RX']>=action['clean_worst_RX']-rule['worst_RX_noninferiority_ACTION_CE'] and margin['stress_mean']>=action['stress_mean']
 return dict(status='VERIFIED',run_id=d.RUN,source_only=True,target_access=False,models=15,identity_updates=66600,summary=result,paired_clean_margin_action=dif,eligible=eligible,rule=rule,records=values)
def finish():
 v=analyze(d.BASE);d.write(d.BASE/'analysis.json',v);d.write(d.BASE/'source_decision.json',dict(selected='FINITE_MARGIN' if v['eligible'] else None,eligible=v['eligible'],rule=v['rule'],target_access=False,automatic_target_launch=False))
 lines=['# 实际有限作用间隔：整源 RX 留出结果','','VERIFIED：单种子、五个源 RX、三个从零 CVS 身份训练对照，15 个模型各固定 E20、4440 更新。完全留出对应 RX，U 不加载，目标访问 0。','','|方案|留出 V clean 均值|最差 RX|四操作压力均值|','|---|---:|---:|---:|']
 for r in v['summary']:lines.append(f"|{r['arm']}|{r['clean_mean']:.4%}|{r['clean_worst_RX']:.4%}|{r['stress_mean']:.4%}|")
 lines+=['',f"FINITE_MARGIN 相对 ACTION_CE 逐 RX 差值：{v['paired_clean_margin_action']}；预登记晋级：{v['eligible']}。",'', '三个方案均计算同样的四个实际数字操作、主 clean128 和额外 eval160 前向；REPLAY 的额外前向只用五份 clean。ACTION_CE 用 clean＋四操作平均 CE，FINITE_MARGIN 用各竞争类别跨五视图的最小真实 logit 间隔；各类别使用同一规则，无固定难TX加权。额外 eval 前向保留梯度，不更新 dropout／BN。','', '间隔损失是有限集合上最坏实际 CE 的上界，不是任意连续 FIR／椭球扰动证明；数字操作不等于物理换 RX。整 RX 留出和同操作CE对照用于检验它是否提供独立身份收益。原 V 始终只评估，不用于更新或将角色拆分。','', '全部 RX/day、15 TX 类对、混淆和间隔保留于 analysis.json；完整 20 epoch/4440 步日志在各 row。没有目标数据或目标支持，没有新的三种子 clean 成绩，因此 84% 目标尚未证明。通过后另开原契约三种子匹配身份矩阵并逐 row 自动 clean＋六 practical 预测和独立 truth-last 评分。失败保留负结果。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return v
