import statistics as st
from . import design as d
def analyze(base):
 rows=[]
 for row in d.rows():
  v=d.read(base/row['row_id']/'completion.json');c=d.config(row)
  if v['config']!=c or v['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or v['identity_updates']!=6660 or len(v['epochs'])!=30 or v['target_access'] or v['U_opened'] or not v['parameters_buffers_unchanged'] or v['fit_L_count']!=5040 or v['held_V_count']!=5400:raise ValueError('Incomplete quality source row')
  if row['arm']!='CE30' and [r['level'] for r in v['actual_layers']]!=list(d.LEVELS):raise ValueError('Missing actual shallow/middle FiLM')
  m=v['metrics'];cm=m['confusion'];n=sum(map(sum,cm));a=sum(cm[k][k] for k in range(6))/n
  if n!=5400 or abs(a-m['accuracy'])>1e-7 or len(m['class_pairs'])!=15:raise ValueError('Quality source metric recount')
  rows.append(dict(**row,metrics=m,parameters=v['parameters'],actual_layers=v['actual_layers']))
 summary=[]
 for arm in d.ARMS:
  rr=[r for r in rows if r['arm']==arm];summary.append(dict(arm=arm,mean=st.mean(r['metrics']['accuracy'] for r in rr),worst_RX=min(r['metrics']['accuracy'] for r in rr),parameters=rr[0]['parameters'],per_RX={str(r['held_rx']):r['metrics']['accuracy'] for r in rr}))
 look={r['arm']:r for r in summary};q=look['QUALITY_FILM'];b=look['CE30'];f=look['FREE_FILM'];db=[q['per_RX'][str(rx)]-b['per_RX'][str(rx)] for rx in d.RX];df=[q['per_RX'][str(rx)]-f['per_RX'][str(rx)] for rx in d.RX];rule=d.config(d.rows()[0])['quality_film']['scientific_rule']
 eligible=q['mean']>=b['mean']+.01 and q['mean']>=f['mean']+.005 and sum(v>=0 for v in db)>=4 and sum(v>=0 for v in df)>=4 and q['worst_RX']>=b['worst_RX']-.01 and q['worst_RX']>=f['worst_RX']-.01
 return dict(status='VERIFIED',models=15,identity_updates=99900,source_only=True,target_access=False,summary=summary,paired_vs_CE=db,paired_vs_FREE=df,eligible=eligible,rule=rule,rows=rows)
def finish():
 v=analyze(d.BASE);d.write(d.BASE/'analysis.json',v);d.write(d.BASE/'source_decision.json',dict(selected='QUALITY_FILM' if v['eligible'] else None,eligible=v['eligible'],target_access=False,automatic_target_launch=False))
 lines=['# 浅中层质量条件调制：整源RX留出','','VERIFIED：单seed五源RX×三控制，从零各30epoch/6660updates，U和目标访问0，五份heldRX整机完全未见。','','|方案|源V clean均值|最差RX|身份可训练参数|','|---|---:|---:|---:|']
 for r in v['summary']:lines.append(f"|{r['arm']}|{r['mean']:.4%}|{r['worst_RX']:.4%}|{r['parameters']}|")
 lines+=['',f"预登记质量FiLM源晋级：{v['eligible']}；相对CE逐RX差{v['paired_vs_CE']}；相对FREE差{v['paired_vs_FREE']}。",'', '实际time_down/t2.pool位置和channel数保留，缩放exp(.2tanh)为正、介于0.8187至1.2214，平移最大0.1 activation unit；不删整条方向。最后generator0初始化使整个初始模型等同原native。三种quality代理从同包IQ产生，mean/populationstd仅fit四RX L5040、固定且clip4；不是纯RX参数或新信息，也不是独立目标支持适应。','', 'FREE_FILM以实际层channelmean/std学习三维context，使用同一有界调制，与quality机制区分新增普通自适应容量；参数相近但不完全相同，成本实报。没有CFO标量、验证前导、U标签、辅助loss或teacher。所有TX共用调制，原完整波形和身份结构保留；不宣称全模型物理不变量或RX完全可辨识。','', '所有原V只评估；保留15TX类对、day、混淆与30epoch/6660step日志。84%三seed目标尚未证明，只有源条件通过才进入固定三seed全源30最终七视图确认。设计来自报告条件坐标路线和source跨RX不稳定证据，不据既有target分数调整尺度、rank、参数或seed。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return v
