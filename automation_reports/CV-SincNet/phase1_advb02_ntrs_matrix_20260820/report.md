# ADVB02 NTRS LEO_WEAK因果矩阵实验报告

## 当前结论

- 状态：`LOCAL_VERIFIED`
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
|T08|项目实验闭环|每个新run不可覆盖并在训练后独立测试clean及三种LEO_WEAK|5个新run目录和`independent_final_eval`|pending|待N607运行闭合|

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
- 归档SHA、远端编译、真实checkpoint无query smoke和启动证据将在执行后追加。

## 技术停止规则

只在协议/seed/角色/场景错误、错误release或CWD、输出碰撞、进程归属不清、同类确定性预prediction异常至少重复两次、无法生成最终checkpoint或独立测试不能闭合时停止对应run。低性能不触发停止，不干预M1或其他无关任务。

