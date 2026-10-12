import statistics as st
from . import design as d

def analyze(base):
 values=[]
 for row in d.rows():
  v=d.read(base/row['row_id']/'completion.json');c=d.config(row)
  if v['config']!=c or v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['target_access'] or v['U_opened'] or v['neural_backbone_updates'] or v['head_updates']!=3200:raise ValueError('Incomplete source head row')
  if [r['arm'] for r in v['records']]!=list(d.ARMS):raise ValueError('Missing source arm')
  for r in v['records']:
   if r['updates']!=800 or len(r['epochs'])!=20 or [e['epoch'] for e in r['epochs']]!=list(range(1,21)):raise ValueError('Source training budget differs')
   for role,n in [('L_held',1260),('V_held',5400)]:
    m=r[role];cm=m['confusion'];count=sum(map(sum,cm));acc=sum(cm[i][i] for i in range(6))/count
    if count!=n or len(m['class_pairs'])!=15 or abs(m['accuracy']-acc)>1e-7:raise ValueError('Source metric recount')
   values.append(r)
 result=[];rule=d.config(d.rows()[0])['scientific_rule']
 for arm in d.ARMS:
  rows=[r for r in values if r['arm']==arm];control=[r for r in values if r['arm']=='RAW_REL']
  dif=[next(r for r in rows if r['held_rx']==x['held_rx'])['V_held']['accuracy']-x['V_held']['accuracy'] for x in control]
  mean=st.mean(r['V_held']['accuracy'] for r in rows);worst=min(r['V_held']['accuracy'] for r in rows)
  eligible=arm in ('COMPLEX_REL','GL2_REL') and st.mean(dif)>=rule['mean_gain_vs_RAW_REL'] and sum(x>=0 for x in dif)>=rule['nonnegative_folds'] and worst>=min(r['V_held']['accuracy'] for r in control)-rule['worst_RX_noninferiority'] and mean>=rule['mean_accuracy_floor']
  result.append(dict(arm=arm,V_held_mean=mean,V_held_worst_RX=worst,L_held_mean=st.mean(r['L_held']['accuracy'] for r in rows),paired_gain_RAW_REL=dif,mean_gain_RAW_REL=st.mean(dif),eligible=eligible))
 return dict(status='VERIFIED',run_id=d.RUN,source_only=True,target_access=False,neural_backbone_updates=0,head_fits=20,head_updates=16000,summary=result,records=values,rule=rule)

def finish():
 v=analyze(d.BASE);d.write(d.BASE/'analysis.json',v)
 eligible=[r for r in v['summary'] if r['eligible']]
 selected=max(eligible,key=lambda r:(r['V_held_mean'],r['V_held_worst_RX'],r['arm']=='COMPLEX_REL'))['arm'] if eligible else None
 d.write(d.BASE/'source_decision.json',dict(selected=selected,eligible=[r['arm'] for r in eligible],rule=v['rule'],target_access=False,automatic_target_launch=False))
 lines=['# 波形关系表示：整源接收机留出诊断','','状态：VERIFIED。5 个 RX、4 种表示、20 份从零分类器。U 未打开，目标访问 0。仅一个模型种子，不能称三种子性能确认。','','每次仅在其余四个 RX 的 L 上训练 20 个完整遍历；保留原始物理角色，V 只评估。没有预训练骨干或教师。','','|表示|留出 V 均值|最差 RX|留出 L 均值|相对 RAW_REL 增益|通过条件|','|---|---:|---:|---:|---:|---|']
 for r in v['summary']:lines.append(f"|{r['arm']}|{r['V_held_mean']:.4%}|{r['V_held_worst_RX']:.4%}|{r['L_held_mean']:.4%}|{100*r['mean_gain_RAW_REL']:+.4f} 个百分点|{r['eligible']}|")
 lines+=['',f'按预登记源规则保留：{selected}。没有读取目标分数，也未据目标选择结构。','', '全部 15 个 TX 类对、5 个 RX、day 分层、混淆、NLL、间隔和 20 轮训练历史见 analysis.json。各头参数相同，但重复／零坐标使有效输入维数不同，不能把表示收益简单归为不变性。RAW_REL 使用四个 I/Q 外积块；COMPLEX_REL 使用归一化复 Gram；GL2_REL 使用共同固定线性投影后的行空间。','', '不使用前导、已知激励、CFO 数值或 TX 特定权重。原始 IQ 仍可能包含频偏信息，不能宣称完全消除 CFO。GL(2) 严格性质仅对满行秩和无数值截断成立；零包和近退化包全部保留并登记。该几何不涵盖任意 FIR、时变 AGC 或非线性接收链。','', '这里检验表示内的身份信息存活，尚不能证明加入 CVS 后有增益，也没有证明 84% clean 三种子目标。通过者下一步与普通同容量关系分支做匹配 E20 身份比较；失败者保留负结果，不扩展强不变分支。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return v
