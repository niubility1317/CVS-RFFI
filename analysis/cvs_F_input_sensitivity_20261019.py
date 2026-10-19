"""Independently aggregate every saved source observation and batch; no target."""
import json
from pathlib import Path
import numpy as np

RUN='20261019-diagnostic-F-CSIL-inputJacobian-s2026092701-m1-r01'
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 cache=ROOT/'local_artifacts/F_sameIQ_sensitivity_20261019/complete_analysis'
 complete=read(cache/'complete.json');records=read(cache/'batch_geometry.json')['records'];z=np.load(cache/'source_derivatives.npz',allow_pickle=False);y=z['TX'].astype(int)
 assert len(y)==360 and len(set(z['ids'].tolist()))==360 and len(records)==20
 assert complete['total_NN']==10804 and complete['optimizer_updates']==0 and complete['weights_and_buffers_unchanged'] and complete['parameter_grads_none'] and not complete['target_access']
 cells={tuple(t):0 for t in zip(y,z['RX'],z['day'])}
 for t in zip(y,z['RX'],z['day']):cells[tuple(t)]+=1
 assert len(cells)==90 and set(cells.values())=={4}
 views={}
 for view in ('clean','d0_mid','d0_low','d1_mid','d1_low'):
  t=z[view+'_teacher'];s=z[view+'_student'];gt=z[view+'_g_teacher'].reshape(360,-1).astype(float);gs=z[view+'_g_student'].reshape(360,-1).astype(float);gw=z[view+'_g_wrong'].reshape(360,-1).astype(float)
  def cosine(a,b):return (a*b).sum(1)/(np.maximum(np.linalg.norm(a,axis=1),1e-12)*np.maximum(np.linalg.norm(b,axis=1),1e-12))
  tp=z[view+'_teacher_plus'];tm=z[view+'_teacher_minus'];sp=z[view+'_student_plus'];sm=z[view+'_student_minus'];tn=np.linalg.norm(gt,axis=1);u=gt/np.maximum(tn[:,None],1e-12)
  # Exact derivative ratio cancels the same h in both models' central differences.
  a=((tp-tm)[np.arange(360),y]/2).astype(float);b=((sp-sm)[np.arange(360),y]/2).astype(float);expected=(gs*u).sum(1)/np.maximum(tn,1e-12)
  nonzero=np.abs(a)>1e-6
  rs=[r for r in records if r['view']==view]
  assert sum(r['teacher_correct'] for r in rs)==int((t.argmax(1)==y).sum())
  assert sum(r['student_correct'] for r in rs)==int((s.argmax(1)==y).sum())
  views[view]=dict(n=360,teacher_accuracy_pct=float(100*np.mean(t.argmax(1)==y)),student_accuracy_pct=float(100*np.mean(s.argmax(1)==y)),teacher_correct_student_wrong=int(((t.argmax(1)==y)&(s.argmax(1)!=y)).sum()),teacher_wrong_student_correct=int(((t.argmax(1)!=y)&(s.argmax(1)==y)).sum()),input_student_teacher_cos_mean=float(cosine(gs,gt).mean()),input_true_wrong_cos_mean=float(cosine(gt,gw).mean()),body_true_CE_cos_mean=float(np.mean([r['body']['true_CE']['cosine'] for r in rs])),body_wrong_CE_cos_mean=float(np.mean([r['body']['wrong_CE']['cosine'] for r in rs])),teacher_finite_difference_error_mean=float(np.mean([r['teacher_finite_difference_relative_error_mean'] for r in rs])),student_to_teacher_derivative_ratio_abs_error_median=float(np.median(np.abs(b[nonzero]/a[nonzero]-expected[nonzero]))),nonzero_finite_difference_count=int(nonzero.sum()),teacher_input_norm_median=float(np.median(tn)),student_input_norm_median=float(np.median(np.linalg.norm(gs,axis=1))))
 urban=[r for r in records if r['view']!='clean'];clean=[r for r in records if r['view']=='clean']
 avg=lambda rs,key:float(np.mean([r['body'][key]['cosine'] for r in rs]))
 true,wrong,cl=avg(urban,'true_CE'),avg(urban,'wrong_CE'),avg(clean,'true_CE');adv=100*sum(r['teacher_correct']-r['student_correct'] for r in urban)/1440
 passed=adv>=1 and true>=0 and true-wrong>=.01 and cl>=-.05
 decision=read(cache/'source_decision.json')
 for k,value in dict(urban_teacher_advantage_pp=adv,urban_body_true_CE_cos=true,urban_body_wrong_CE_cos=wrong,urban_body_true_minus_wrong=true-wrong,clean_body_true_CE_cos=cl).items():assert abs(decision[k]-value)<1e-12
 assert decision['rationale_pass']==bool(passed)
 v=dict(run_id=RUN,status='ANALYZED',views=views,source_decision=decision,total_NN=10804,unique_source_V=360,source_observations=1800,optimizer_updates=0,all_weights_buffers_unchanged=True,model_seeds=1,wall_seconds=complete['wall_seconds'],qualification_independently_recomputed=True,target_access=False)
 write(cache/'analysis.json',v)
 out=ROOT/'automation_reports/CV-SincNet'/RUN
 lines=['# 同 IQ 输入敏感度诊断：完整源域结果','', '**固定真类导数对齐依据不成立，关闭本方案。** 此结果不支持启动该损失的 E20，也不改变总体研发目标。','',f'运行 `{RUN}`；实际发布 `83616fbabc86ff8d106a406aca53f2878f0f0467`，代码/config `9ba9d52afe179db13bafab3295efc336bfba7d76`。普通账户发布、单归档传输比较和远端编译 VERIFIED；实际 checkpoint 二阶梯度 smoke PASS。独立 PID/CWD/argv/GPU4 和日志增长核实，进程自然退出；无其他进程干预。','', '固定 360 个源 V 物理样本，每个 TX×RX×day 单元 4 个；clean 及既有两次 mid/low urban 实现，共 1800 观测。所有 20 批和保存的逐样本输出、输入导数已重算。未访问目标 IQ/truth，没有训练或新预测模型。','', '|源视图|CSIL 默认头 %|F B1 默认头 %|骨干真类导数损失/CE 余弦|错类对照/CE 余弦|','|---|---:|---:|---:|---:|']
 for name,a in views.items():lines.append(f"|{name}|{a['teacher_accuracy_pct']:.6f}|{a['student_accuracy_pct']:.6f}|{a['body_true_CE_cos_mean']:.6f}|{a['body_wrong_CE_cos_mean']:.6f}|")
 lines += ['', f'事前标准及实测：urban 教师优势 {adv:.6f} 个百分点（要求 ≥1，通过）；真类梯度余弦 {true:.6f}（要求 ≥0，失败）；真类减错类 {true-wrong:.6f}（要求 ≥0.01，失败）；clean 真类余弦 {cl:.6f}（要求 ≥−0.05，通过）。决定在任何后续训练与正式测试前固定，不改标签、归一化、阈值或筛选正向层以救回本方案。','', '虽然 CSIL 对这些受损源观测的分类更准确，其真类 logit 的局部输入导数未提供预定的、优于错类对照的 CE 一致骨干方向。这只否定本次固定 signed true-row 方法的推进依据，不证明所有 Jacobian 蒸馏都无效，也不证明物理性能上限。','',f"实际共 10804 次包前向，两个模型各 5402；0 次优化器/头拟合/EMA 更新，全部权重和缓冲区逐项相等，参数 .grad 均未填入，F 的 BatchNorm 模块数为 0。GPU 运行 {complete['wall_seconds']:.3f} 秒。中心差分仅是数学导数校验，不是身份保持的物理增强，也不计入正式识别性能。各视图教师差分相对误差均值："+', '.join(f"{name}={a['teacher_finite_difference_error_mean']:.6g}" for name,a in views.items())+'。','', '原始源观测输出、导数和梯度几何保存在不可覆盖运行根；Git 只提交实现与聚合报告。单模型 seed、源域诊断不构成性能或稳定性结论。总体目标尚未完成。','']
 (out/'final_analysis.md').write_text('\n'.join(lines),encoding='utf8')
 print(json.dumps(dict(decision=decision,views=views),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
