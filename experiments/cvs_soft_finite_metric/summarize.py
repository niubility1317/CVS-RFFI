import statistics as st
from . import design as d
def analyze(base):
 rows=[]
 for row in d.rows():
  v=d.read(base/row['row_id']/'completion.json')
  if v['config']!=d.config(row) or v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['identity_updates'] or v['head_updates'] or v['target_access'] or v['U_access'] or not v['parameters_buffers_unchanged'] or v['fit_L_count']!=5040 or v['heldV_count']!=5400 or set(v['records'])!=set(d.ARMS):raise ValueError('Incomplete source metric row')
  for m in v['records'].values():
   cm=m['confusion'];n=sum(map(sum,cm));acc=sum(cm[k][k] for k in range(6))/n
   if n!=5400 or abs(acc-m['accuracy'])>1e-7 or len(m['class_pairs'])!=15:raise ValueError('Source metric independent recount')
  rows.append(dict(**row,records=v['records'],metric_diagnostics=v['metric_diagnostics']))
 result=[]
 for arm in d.ARMS:result.append(dict(arm=arm,mean=st.mean(r['records'][arm]['accuracy'] for r in rows),worst_RX=min(r['records'][arm]['accuracy'] for r in rows),per_RX={str(r['held_rx']):r['records'][arm]['accuracy'] for r in rows}))
 lookup={r['arm']:r for r in result};soft=lookup['SOFT_FINITE'];mahal=lookup['MAHALANOBIS'];native=lookup['NATIVE_HEAD'];dif=[soft['per_RX'][str(rx)]-mahal['per_RX'][str(rx)] for rx in d.RX];rule=d.config(d.rows()[0])['scientific_rule']
 eligible=soft['mean']>=mahal['mean']+rule['clean_gain_MAHALANOBIS'] and sum(v>=0 for v in dif)>=rule['nonnegative_RX'] and soft['mean']>=native['mean']-rule['clean_mean_noninferiority_NATIVE'] and soft['worst_RX']>=mahal['worst_RX']-rule['worst_RX_noninferiority_MAHALANOBIS']
 return dict(status='VERIFIED',run_id=d.RUN,models=5,readouts=25,identity_updates=0,target_access=False,summary=result,paired_gain=dif,eligible=eligible,rows=rows)
def finish():
 v=analyze(d.BASE);d.write(d.BASE/'analysis.json',v);d.write(d.BASE/'source_decision.json',dict(selected='SOFT_FINITE' if v['eligible'] else None,eligible=v['eligible'],target_access=False,automatic_target_launch=False))
 lines=['# 软有限作用度量：整源RX留出','','VERIFIED：五份固定且未见heldRX的从零E20 CVS模型，25个冻结读出，无骨干／头更新，无U、目标访问。所有源拟合仅合法四RX L5040；V5400仅评估。','','|读出|源V clean均值|最差RX|','|---|---:|---:|']
 for r in v['summary']:lines.append(f"|{r['arm']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|")
 lines+=['',f"相对普通MAHALANOBIS的逐RX差值：{v['paired_gain']}；预登记晋级：{v['eligible']}。",'', '所有模型使用固定 CLEAN_REPLAY 原模型，不按分数挑老师。source-only父权重实际config、commit、角色、heldRX排除与全祖先逐模型核对。源头训练20epoch/4440更新成本另计，本批0更新。','', 'C_noise是合法L的类内共享协方差，加固定10%球形收缩和相对1e-6数值地板。C_finite是同包四种实际操作delta外积的平均；不把U总体协方差当作域噪声。SOFT_FINITE=(C_noise+C_finite)^-1，倍率固定1。MAHALANOBIS不加C_finite；HARD_FINITE在noise白化坐标中去掉解释80%有限作用能量的方向，rank只由fitL决定。全部类别统一中心／公式，无指定TX分支。','', '没有理论保证数字操作覆盖未知RX；只有源留出收益通过才进入正式三种子目标确认。完整15类对、day和混淆见analysis.json。不因微小收益改倍率、rank或选择性重跑。84%目标尚未证明。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return v
