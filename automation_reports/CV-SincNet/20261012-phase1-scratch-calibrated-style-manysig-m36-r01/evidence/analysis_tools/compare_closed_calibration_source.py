"""Compare fixed completed source trajectories; never read target score files."""
from pathlib import Path
import json
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011')
base=R/'automation_reports/CV-SincNet'
new='20261012-phase1-scratch-calibrated-style-manysig-m36-r01'
old='20261012-phase1-scratch-boundary-style-manysig-m27-r01'
pairs=[(old,'M_MIDDLE-s2026092701'),(new,'DELAY_M_MIDDLE-s2026092701')]
rows=[]
for run,row_id in pairs:
    value=json.loads((base/run/'evidence/source_full_analysis.json').read_text(encoding='utf-8'))
    row=next(r for r in value['rows'] if r['row_id']==row_id)
    assert row['steps']==4440 and row['source_only'] and len(row['epochs'])==20
    ratios={k:v for k,v in row['combined_fishr_gradient_ratios'].items() if k.endswith('/ratio_to_main_ce')}
    assert all(v['observations']==999 for v in ratios.values())
    rows.append(dict(run=run,row_id=row_id,seed=row['seed'],steps=row['steps'],calibration_coefficient=row['calibration']['fixed_coefficients']['t2.pool'],ratios=ratios,last_CE=row['epochs'][-1]['ce'],last_KL=row['epochs'][-1]['KL'],last_weight_movement=row['epochs'][-1]['weight_movement']))
out=dict(status='VERIFIED',source_only=True,target_access=False,rows=rows,scope='two complete source trajectories of one preregistered seed; not shared intermediate weights or a causal proof; no runtime changes')
f=base/new
(f/'evidence/closed_calibration_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 首个闭合源轨迹：延后校准对照','','仅比较已完整结束的两行源训练日志，不读取目标预测或评分。每行4440次更新、20轮、999次辅助步骤。单个种子2026092701，不能推断三种子总体。','','|指标|初始化校准 M_MIDDLE|444步后校准 DELAY_M_MIDDLE|','|---|---:|---:|']
metrics=[('中层固定系数',lambda r:r['calibration_coefficient']),('中层联合辅助/主CE梯度比均值',lambda r:r['ratios']['t2.pool/ratio_to_main_ce']['mean']),('中层联合辅助/主CE梯度比最大值',lambda r:r['ratios']['t2.pool/ratio_to_main_ce']['max']),('浅层参数组接收的辅助/主CE梯度比均值',lambda r:r['ratios']['time_down/ratio_to_main_ce']['mean']),('末轮主CE',lambda r:r['last_CE']),('末轮U KL（未加权）',lambda r:r['last_KL']),('编码器相对初始化L2位移',lambda r:r['last_weight_movement']['encoder']['relative_l2']),('融合头相对初始化L2位移',lambda r:r['last_weight_movement']['fusion_head']['relative_l2'])]
for label,fn in metrics:lines.append(f'|{label}|{fn(rows[0]):.7g}|{fn(rows[1]):.7g}|')
lines+=['','延后校准这一行的辅助梯度平均尺度明显更小，编码器和融合头仍有非零更新。中层目标的梯度也传到浅层参数组；浅层没有独立统计项不等于浅层不更新。','','DELAY没有运行时限幅，中层局部梯度比最大值仍超过1；一次校准不能保证全程上限。该比值分母是辅助批上的主CE，参数组为旧局部定义，并非新CAP的全参数组、主CE+U KL联合分母，二者不能直接替换。','','同seed预热数值轨迹已有差异，见warmup_pairing_audit.md。这是已观察到的训练轨迹差异，不是固定同一中间权重的反事实试验；不证明梯度尺度是测试变化的原因，不证明更小一定更好。权重位移为相对各自初始化的距离，不代表两个最终模型彼此接近。','','继续执行已冻结的CAP、DELAYCAP及多层/MixStyle消融，完成全部三种子后统一报告；当前健康训练不修改。']
(f/'closed_calibration_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status='VERIFIED',source_rows=2,source_steps=8880,auxiliary_steps=1998,target_access=False)))
