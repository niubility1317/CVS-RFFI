# ADVB02 NTRS设计追踪记录

## 来源和解释边界

- 用户指导原文：`E:/codex/home/attachments/90a3e579-e242-4aa9-b188-6596902e09d2/pasted-text.txt`
- 当前协议：`E:/type10-7/项目.md`
- 实现规格：`docs/superpowers/specs/2026-08-20-advb02-ntrs-leo-weak-design.md`
- 候选：`ADVB02_NTRS_LEO_WEAK_E200`
- 基线提交：`2650134dcd92493d3d6dcd74483b854ebf1786cf`

本追踪表把指导中的Phase1首版配置与后续研究路线分开。`deferred`表示指导本身列为后续优先级且不属于本次Phase1候选；`rejected`只用于与当前协议冲突的运行期行为。状态不得被误写成性能已验证。

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|NTRS-01|1、16、21|实现分层信道—接收机稳健系统，而非普通Adapter|`code/ntrs.py`、dual model|pending|待TDD|独立于CRRA|
|NTRS-02|2、12、20|拼接卫星CE为主，KL=0.01、margin=0.03、relation=0.02|训练器、losses|pending|待TDD|不使用强点一致性|
|NTRS-03|3、10|32–48维分组物理描述符与fast/slow上下文|`code/ntrs.py`|pending|待TDD|slow EMA=0.95|
|NTRS-04|4|消费同次增强可用物理metadata并显式处理缺失|视图metadata、model|pending|待TDD|不得二次生成LEO视图|
|NTRS-05|5|L=3有界广义复数FIR、共轭FIR和相位斜坡，近恒等初始化|`code/ntrs.py`|pending|待TDD|校正幅度受限|
|NTRS-06|5、16|时域读校正IQ、频域raw/corrected双视图、PA和domain读raw IQ|`model.py`、dual model|pending|待TDD|PA旁路强校正|
|NTRS-07|6|clean/LEO成对差分学习全局rank-8干扰切空间|`code/ntrs.py`、训练器|pending|待TDD|只由source训练更新|
|NTRS-08|6、7|修正量严格位于切空间，alpha最大0.20，零初始化有界残差|`code/ntrs.py`|pending|待TDD|160维z_id端|
|NTRS-09|7、8|raw/robust双头、source支持门、不确定度和correctability|dual model、training|pending|待TDD|robust头从raw原型初始化|
|NTRS-10|8、13|安全融合、修正能量门、默认禁止unknown rescue|dual model、evaluation|pending|待TDD|分歧回退raw|
|NTRS-11|9|receiver/day/channel三头和q的TX去泄漏|dual model、training|pending|待TDD|标签均来自source批次|
|NTRS-12|9|类别条件z_id/z_dom去相关|losses、training|pending|待TDD|权重0.01|
|NTRS-13|9|接收机公共偏移一致性能力|losses|pending|待TDD|首版仅source类共享估计|
|NTRS-14|12|最小修正、切空间、correctability和安全损失|losses、training|pending|待TDD|权重按规格|
|NTRS-15|13|raw/robust四类转移、类别吸引余弦和切空间一致性遥测|evaluation|pending|待TDD|逐场景保存|
|NTRS-16|17、18|S1/S2-a/S2-b/S3交替日程与骨干:NTRS=1:5学习率|training helper|pending|待TDD|S1 gate严格为0|
|NTRS-17|20|第一版配置rank=8、alpha=0.20、slow EMA=0.95、原始PA旁路|launcher、config|pending|待TDD|冻结配置|
|NTRS-18|当前协议|Core90、seed=392034、0.07/0.63/0.15/0.15和三种LEO_WEAK|launcher、负测|pending|待TDD|拒绝隐式mixed_orbit|
|NTRS-19|评测要求|final checkpoint独立clean和三种LEO逐场景测试|launcher、evaluation|pending|待TDD|训练完成不等于实验完成|
|NTRS-20|11.1–11.3、19 P2|扩展接收机链路模拟器和困难增强挖掘|后续独立候选|deferred|不阻断本run|本run保持Core90同一LEO增强，便于机制归因|
|NTRS-21|14、19 P3-1/P3-2|Phase2类共享低秩接收机映射和capture-level prototype calibration|后续Phase2候选|deferred|不阻断本run|不属于Phase1模型|
|NTRS-22|15|使用无标签query更新q_slow、映射或prototype|无|rejected|协议审查|当前query只用于测试，不得更新状态|
|NTRS-23|6.3|动态receiver-conditioned/slow-fast双切空间|后续消融|deferred|不阻断本run|指导明确建议首版先全局子空间|
|NTRS-24|Phase1声明|不得宣称真实在轨、Phase2或真实unknown结果|报告|pending|反向审查|source-only代理证据|

## 基线证据

- 独立worktree：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/advb02-ntrs-leo-weak-20260820`
- 基线聚焦测试：CRRA核心、模型、训练、协议、评估和launcher共46项通过。
- `use_ntrs`尚未实现；以上`pending`条目必须按失败测试→最小实现→通过测试的顺序推进。

