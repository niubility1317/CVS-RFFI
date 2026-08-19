# ADVB02 NTRS LEO_WEAK因果矩阵实验报告

## 当前结论

- 状态：`RUNNING`
- 实现提交：`b92648f2731ed39775a101ea74c52ecb85421371`
- 目标：在完全相同的Phase1数据协议、seed和LEO_WEAK增强下，分离NTRS整体、身份结构监督、干扰因子分解、有界嵌入残差和开放集安全损失的贡献。
- 历史`mixed_orbit`不参与本矩阵。CRRA运行虽然已产生最终文件，但200个epoch均为`train_optimizer_step_applied=0.0`，因此不作为有效对照行。

## 冻结公共条件

- base candidate：`ADV3B02_CORE90_SOFT_E200`
- seed：`392034`
- Phase1角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 训练方式：`concat_masked`
- 训练场景：仅`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- LEO日程：E1–40为`p=0.30`的clear；E41–90为`p=0.60`的low-elev/rain；E91–200为`p=0.80`的三场景并集。
- 最终独立测试：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`逐场景完整测试。
- 训练轮数：200；其他Core90结构、优化器、学习率、batch、source-only边界和选模规则不变。

## 矩阵

|臂|run ID|profile|唯一方法差异|GPU|
|---|---|---|---|---|
|M0|`phase1_advb02_ntrs_matrix_20260820_m0_control`|`control`|关闭NTRS，作为同协议Core90对照|0|
|M1|`phase1_advb02_ntrs_leo_weak_20260820_r3`|`full`|完整NTRS；已在独立run中运行|1|
|M2|`phase1_advb02_ntrs_matrix_20260820_m2_no_idstruct`|`no_identity_structure`|将sat-KL、margin、relation和class-conditional四项身份结构损失置0|2|
|M3|`phase1_advb02_ntrs_matrix_20260820_m3_no_nuisance`|`no_nuisance_factorization`|将receiver/day/channel、context-TX去泄漏、条件去相关和共享receiver损失置0|3|
|M4|`phase1_advb02_ntrs_matrix_20260820_m4_no_embedres`|`no_embed_residual`|将`alpha_max`、minimum-correction、alpha和subspace损失置0，关闭嵌入端有界残差修正|4|
|M5|`phase1_advb02_ntrs_matrix_20260820_m5_no_safety`|`no_safety_losses`|将correctability、score-stability和class-attraction三项安全损失置0；推理安全门结构仍保留|5|

主要比较为M1−M0、M1−M2、M1−M3、M1−M4和M1−M5。所有比较必须使用同一run行的clean和三场景最终结果，不拼接跨run峰值。

## 设计追踪

|ID|来源|要求|目标文件或artifact|状态|验证|
|---|---|---|---|---|---|
|T01|Phase1协议|固定四角色、seed和source-only边界|launcher及各run命令|verified|6个profile完整命令均由训练器parser解析|
|T02|LEO_WEAK默认|训练和最终测试只用三种LEO_WEAK，禁止`mixed_orbit`|launcher、final eval|verified|聚焦测试及独立审查|
|T03|指导第十九节|提供Core90对照和完整NTRS|M0、M1|verified|profile干跑测试|
|T04|指导身份锚定组|单独移除结构监督组|M2|verified|profile干跑测试|
|T05|指导干扰分解组|单独移除nuisance分解组|M3|verified|profile干跑测试|
|T06|指导有界校正组|单独关闭嵌入残差组|M4|verified|profile干跑测试|
|T07|指导开放集安全组|单独移除安全训练损失|M5|verified|profile干跑测试|
|T08|N607启动|5个新run不可覆盖且profile/GPU/CWD/日志绑定正确|5个run root及log root|verified|启动后PID/CWD/cmdline/GPU/log检查|
|T09|项目实验闭环|训练后独立测试clean及三种LEO_WEAK|5个`independent_final_eval`|pending|待N607运行闭合|

## 本地验证与审查

- TDD：新增profile测试先出现6项预期失败，再实现launcher参数化并全部通过。
- 6个profile的完整训练命令均由`build_arg_parser()`成功解析。
- NTRS核心、模型、训练、协议负测、评估和launcher共40项聚焦测试通过；launcher的`bash -n`及测试文件`py_compile`通过。
- 唯一一次独立P0/P1审查结论为通过；未发现会导致错误profile、协议越界、覆盖输出、启动失败或最终测试不闭合的问题。
- 本地和远端Git分支OID均为`b92648f2731ed39775a101ea74c52ecb85421371`。

## 发布映射

- 本地release归档：`E:\type10-7\local_artifacts\phase1_advb02_ntrs_matrix_20260820\phase1_advb02_ntrs_matrix_20260820_b92648f2.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/phase1_advb02_ntrs_matrix_20260820_b92648f2.tar.gz`
- 远端workspace：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace`
- 单次本地到远端SHA256：`f405932b3455294950a8a7d5a2ba106b0e8fb00102af000b4301d227c3369fe3`，两端一致。
- 远端相关Python编译、launcher语法检查及6个profile干跑均通过。
- 真实checkpoint与一个ManySig source样本的无query smoke通过：`source_samples=1 query_samples=0 missing_ntrs=63 unexpected=0`。

## 启动证据

|臂|launcher PID|trainer PID|GPU|启动时间|状态|
|---|---:|---:|---:|---|---|
|M0|3481635|3481663|0|2026-08-20T03:26:54+08:00|RUNNING|
|M2|3481710|3481735|2|2026-08-20T03:26:55+08:00|RUNNING|
|M3|3481784|3481807|3|2026-08-20T03:26:55+08:00|RUNNING|
|M4|3481852|3481935|4|2026-08-20T03:26:55+08:00|RUNNING|
|M5|3481983|3482067|5|2026-08-20T03:26:55+08:00|RUNNING|

五个run的launcher/trainer均绑定`b92648f2/workspace`及各自唯一run root；cmdline中的profile、seed、角色比例和三种LEO_WEAK参数匹配预登记，日志已增长。M1继续由原r3 launcher在GPU1运行，没有重启或覆盖。

## 初始真实训练证据

|臂|检查epoch|optimizer step rate|nonfinite skip rate|train TX|source val TX|
|---|---:|---:|---:|---:|---:|
|M0|5|1.0000|0.0000|28.7153%|65.7619%|
|M1|39|1.0000|0.0000|61.5625%|93.9683%|
|M2|3|1.0000|0.0000|2.6736%|82.6032%|
|M3|3|1.0000|0.0000|2.7083%|73.8571%|
|M4|3|1.0000|0.0000|2.7431%|73.5794%|
|M5|3|1.0000|0.0000|2.7604%|79.4762%|

截至该检查点，六个有效矩阵臂的完整当前日志均无Traceback、RuntimeError、CUDA error、OOM、`FAIL`或`ERROR`。这些早期值只用于证明训练真实推进，不用于性能排序。

## 技术停止规则

只在协议/seed/角色/场景错误、错误release或CWD、输出碰撞、进程归属不清、同类确定性预prediction异常至少重复两次、无法生成最终checkpoint或独立测试不能闭合时停止对应run。低性能不触发停止，不干预M1或其他无关任务。
