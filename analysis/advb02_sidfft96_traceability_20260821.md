# ADV3B02 SID-FFT96指导建议追踪表

设计规范：`docs/superpowers/specs/2026-08-21-advb02-sidfft96-design.md`  
指导来源：用户提供的《CVS项目优化的核心方向：从“后验修正特征”转向“前端可辨识分解”》  
当前阶段：设计规范待用户书面确认；尚未修改训练代码或发布N607实验。

|ID|来源章节|需求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|SID-001|总体判断、P0|先构建source-only频谱可辨识性地图|`code/scripts/audit_phase1_spectral_identifiability.py`、`code/cvsrffi/spectral_identifiability.py`|pending|P0聚焦测试与真实source smoke|只用`L_s`TX标签，不读取target或`U_s`真值|
|SID-002|4.2|计算并分别输出TX散度、跨RX/day/LEO散度、估计噪声与`J_b`|同上|pending|手工小样本统计fixture|禁止只输出最终比值|
|SID-003|4.1、9.1|支持中心带、相位带和P0多频带三种固定掩码|`code/cvsrffi/spectral_identifiability.py`|pending|掩码边界、空掩码和稳定排序测试|第一轮不学习频带拓扑|
|SID-004|8.1|实现严格96维SID-FFT96五组描述|`code/cvsrffi/spectral_identifiability.py`|pending|维数、有限值、相位、镜像和带边测试|带宽不足时称为带内边缘残差|
|SID-005|2.2、5.3、7.2|不再修改已形成的`z_id`；以前端SID证据形成零初始化加法残差|`code/model_dual_cvsincnet.py`|pending|初始化`z_sid==z_raw`与raw不变测试|不是NTRS式`q→Δz`后验修正|
|SID-006|7.2、P1|保留Sinc、PA、DAC和原分类头，冻结成熟ADV3B02路径|`code/model_dual_cvsincnet.py`、`code/SSDG/train_ssdg.py`|pending|可训练参数白名单测试|SID最后投影零初始化|
|SID-007|阶段B|同checkpoint同时保存raw与SID输出|同上、`code/eval_feature_diagnosis.py`|pending|同批raw数值一致与双输出schema测试|支持逐样本`rescued/harmed`|
|SID-008|P1|历史checkpoint只允许新增SID键缺失，其他漂移失败|`code/SSDG/train_ssdg.py`|pending|允许SID缺键与拒绝旧键/shape漂移测试|不得用通用`strict=False`掩盖错误|
|SID-009|12.1|保留Core90拼接式卫星TX CE与三场景日程|`code/SSDG/train_ssdg.py`、launcher|pending|参数流与dry-run测试|`0.68`、E80、三类LEO_WEAK|
|SID-010|Phase1协议|保持`0.07/0.63/0.15/0.15`和source-only边界|launcher、报告|pending|launcher参数解析与协议负测|不接入Phase2/Phase3数据|
|SID-011|阶段B、P1|首发S0–S3单seed矩阵|`code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh`|pending|`bash -n`、dry-run与不可覆盖测试|S0冻结评测；S1中心；S2相位；S3完整SID|
|SID-012|19|报告clean、逐LEO、Strict UDU、floor、转换和probe指标|`code/eval_feature_diagnosis.py`、实验报告|pending|受控预测fixture与scorer测试|不拼接不同row最优值|
|SID-013|17、16|Phase1 bundle增加SID内容并供Phase2适应|无，本轮不修改|deferred|Phase1晋级后建立独立设计|首发矩阵不声明Phase2改善|
|SID-014|5、6、P2|结构化幅相/CFO/IQ算子及LEO算子监督|无，本轮不修改|deferred|S3满足晋级门槛后再设计|防止一次堆叠导致归因失败|
|SID-015|10、11、13、P3|MUSE新路由、跨RX SupCon、episodic DG和pseudo-new几何|无，本轮不修改|deferred|P2前端证明有效后再设计|不作为本轮发布门|
|SID-016|20|禁止强GRL、强KL、NTRS放大、任意全矩阵Koopman和删除中高频|模型、训练配置、launcher|pending|配置负测和审查|属于明确的候选边界|
|SID-017|18|不从S0直接跳到完整S12|设计、launcher、报告|verified|本设计仅包含P0和S0–S3|后续阶段独立发布|
|SID-018|Exclusive Minimal Experiment Workflow|只执行八项白名单，不增加seal/receipt/SHA链|报告、发布流程|pending|发布前反向审计|release归档仅一次本地/远端SHA|

## 当前计数

- verified：1
- pending：14
- deferred：3
- rejected：0
- blocked：0

最高风险项是SID-008：历史checkpoint兼容加载必须精确允许新增SID参数，同时拒绝任何旧ADV3B02参数漂移。若该边界实现错误，S0与S1–S3将不再是同checkpoint可比实验。
