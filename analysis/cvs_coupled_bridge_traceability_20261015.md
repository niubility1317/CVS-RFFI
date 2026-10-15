# 同信道 HQ 与 SIG 教学：阶段实现追踪

本表追踪用户 17f04499 附件的后续 HQ／分组 SIG 路线，不重新批准既有八位置研究，不构成启动审批或额外门槛。source-only 物理与冻结默认头测量已完成，D/E 训练仍待实现。代码位置：`experiments/cvs_urban_bridge/`；完整配置与输出路径见其 registration.json。

|ID|用户要求与实际映射|状态|证据／剩余项|
|---|---|---|---|
|Q01|固定合法 A0 final50＋同 checkpoint own EMA；L_dev720|verified|真实 CP smoke PASS，源契约及 fixed final50 检查通过|
|Q02|full 实现与原 practical residual 一致|verified|views.py；真实引擎单测逐值一致|
|Q03|冻结接收状态、补偿和 full AGC|verified|元数据及 S/H/K/G 数组相等；复用 full 增益的单测|
|Q04|EQ 噪声着色，EQ 关时不偷换弱信道|verified|EQ-on 着色与缩放单测；实际六场景 EQ=false/G=1|
|Q05|边界、context 和 FIR 延迟保持|verified|原算子复用；require_context 缺失拒绝／提供成功|
|Q06|共享与独立新增噪声，原录音噪声保留|verified|views.py 命名独立流及剂量检查，0 dB shared=full|
|Q07|实际 SIG112 的 6／15／6 组，常量 dt 排除|verified|capture/split；实际模型 layout 与坐标单测|
|Q08|SIG 梯度到 local projection 与 t2，非仅末端|verified|实际固定网络 autograd；额外 W／gain／output_projection 梯度为0；CE W非0|
|Q09|固定源尺度及 floor，先组内平均|verified|teacher_scales/grouped_loss；零输入 floor 单测|
|Q10|0 更新、无 U/query、V 前向0、36k前向|verified|完整 run 36k／34.409653s，student／ownEMA state unchanged，独立读回正常结束|
|Q11|原头、身份差、SIG 组与状态分层结果及图|verified|results／17,280条完整源记录／4,320配对检查／三张图已生成并查看|
|Q12|D/E 保留 direct clean→urban，另加 HQ CE 与 clean锚点|pending|明确设计；训练尚未实现|
|Q13|D/E 额外前向匹配，E 唯一新增 grouped SIG|pending|训练实现与实际预算验证尚未完成|
|Q14|D/E scratch E50＋自动 clean／六 practical truth-last|pending|本次零更新源诊断不冒充训练完成|
|Q15|预定父模型与源域选择；目标不可回流|verified|config 固定 A0；只源诊断，不读目标或依分支目标排名|
|Q16|新 U 特征教学|deferred|按附件第一版不叠加，原旧门控 U 训练未来保留|

计数：verified 12、pending 3、deferred 1、rejected 0、blocked 0。5 项针对性单测通过，P1 Linux 路径修复后定向复核 PASS，真实源 run 完成；训练 D/E 未完成。本源阶段发布 commit 为 baae811ff715831a51c86b13a488f5a2cab3a78c。完整报告见 automation_reports/CV-SincNet/20261015-phase1-sigtime-coupled-bridge-source-manysig-s2026092701-r01/report.md；实际产物在 E:/type10-7/local_artifacts/cvs_coupled_bridge_20261015/。

HQ 仍非可靠无误教师：源 low ownEMA full 64.8611%→独立HQ 70.9722%，BLOCKED 40.7407%→50.6173%；LOS不变。共享HQ更相似且损失更小，存在相关噪声解释。最高剩余风险是分组教学能否转化为原头的固定测试收益及 clean 保持；不能从波形配对或源反事实改善推导已训练方法有效或跨seed稳定。
