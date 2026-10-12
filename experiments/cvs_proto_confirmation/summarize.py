import statistics as st
from . import design as d
def analyze(base):
 rows=[]
 for row in d.rows():
  v=d.read(base/row['row_id']/'completion.json');c=d.config(row)
  if v['config']!=c or v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['identity_updates']!=4440 or v['head_updates'] or v['target_access'] or v['U_access'] or not v['parameters_buffers_unchanged'] or v['fit_L_count']!=5040 or v['heldV_count']!=5400 or set(v['records'])!=set(d.ARMS):raise ValueError('Incomplete prototype confirmation')
  for m in v['records'].values():
   cm=m['confusion'];n=sum(map(sum,cm));a=sum(cm[k][k] for k in range(6))/n
   if n!=5400 or abs(a-m['accuracy'])>1e-7 or len(m['class_pairs'])!=15:raise ValueError('Prototype source metric recount')
  rows.append(dict(**row,records=v['records']))
 result=[]
 for seed in d.SEEDS:
  rr=[r for r in rows if r['model_seed']==seed];native=[r['records']['NATIVE_HEAD']['accuracy'] for r in rr];proto=[r['records']['COSINE_PROTO']['accuracy'] for r in rr];dif=[p-n for p,n in zip(proto,native)]
  result.append(dict(model_seed=seed,native_mean=st.mean(native),prototype_mean=st.mean(proto),paired_gain=dif,mean_gain=st.mean(dif),positive_RX=sum(x>=0 for x in dif),native_worst=min(native),prototype_worst=min(proto)))
 rule=d.config(d.rows()[0])['scientific_rule'];eligible=all(r['mean_gain']>=rule['each_seed_mean_gain'] and r['prototype_worst']>=r['native_worst']-rule['worst_RX_noninferiority_each_seed'] for r in result) and sum(r['positive_RX'] for r in result)>=rule['total_nonnegative_RX']
 return dict(status='VERIFIED',models=10,source_only=True,target_access=False,identity_updates=44400,summary=result,eligible=eligible,rows=rows,rule=rule)
def finish():
 v=analyze(d.BASE);d.write(d.BASE/'analysis.json',v);d.write(d.BASE/'source_decision.json',dict(selected='COSINE_PROTO' if v['eligible'] else None,eligible=v['eligible'],target_access=False,automatic_target_launch=False))
 lines=['# 余弦原型：另两个模型种子源确认','','VERIFIED：2026092702/03，两seed各五真源RX留出；10个原CVS随机E20，每个4440更新。原分类头与普通原型使用完全同一冻结骨干。源拟合L5040，V5400只评估，U和目标访问0。','','|种子|原头均值|原型均值|平均差|非负RX数|原头最差|原型最差|','|---|---:|---:|---:|---:|---:|---:|']
 for r in v['summary']:lines.append(f"|{r['model_seed']}|{r['native_mean']:.4%}|{r['prototype_mean']:.4%}|{100*r['mean_gain']:+.4f}个百分点|{r['positive_RX']}|{r['native_worst']:.4%}|{r['prototype_worst']:.4%}|")
 lines+=['',f"预登记独立源确认通过：{v['eligible']}。",'', '候选来自先前seed2701源端控制信号，未使用目标成绩；本批两个模型seed在设计冻结后完整执行，未改门槛。每seed平均至少+2个百分点、十RX比较至少8非负、各seed最差RX不低于原头超过1个百分点，全部通过才进入三seed全源最终确认。','', '原型仅将合法fitL实际joint160单位向量按六类取均值并归一化，统一余弦判决，无协方差、辅助网络、关系损失、学温度或支持集。输出完整15类对、RX/day和混淆；不同分数尺度不宣称NLL或间隔校准优势。','', '这仍是源端验证，不代表目标84%已完成。后续全源固定E20比较需使用三seed、同模型两读出，逐row自动clean与六practical全部预测固定后独立truth-last评分。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return v
