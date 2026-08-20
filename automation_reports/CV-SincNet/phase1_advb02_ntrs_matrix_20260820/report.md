# ADVB02 NTRS LEO_WEAK因果矩阵实验报告

## 当前结论

- 状态：`ANALYZED`
- 实现提交：`b92648f2731ed39775a101ea74c52ecb85421371`
- 目标：在完全相同的Phase1数据协议、seed和LEO_WEAK增强下，分离NTRS整体、身份结构监督、干扰因子分解、有界嵌入残差和开放集安全损失的贡献。
- 历史`mixed_orbit`不参与本矩阵。CRRA运行虽然已产生最终文件，但200个epoch均为`train_optimizer_step_applied=0.0`，因此不作为有效对照行。
- 六个矩阵臂均完成200轮训练和E200最终checkpoint独立测试；M0的LEO_WEAK三场景均值为`70.457%`，完整NTRS M1仅为`51.618%`，下降`18.839`个百分点。
- 当前结论是否定性的：本版NTRS不能晋级。四个NTRS消融也全部显著低于M0；其中表现最好的M4仍比M0低`16.506`个百分点。

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
|T09|项目实验闭环|训练后独立测试clean及三种LEO_WEAK|6个`independent_final_eval`|verified|6组均为E200、`eval_exit=0`，加载键差异为0|

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
|M0|3481635|3481663|0|2026-08-20T03:26:54+08:00|ARTIFACTS_COMPLETE|
|M1|3466737|3466758|1|2026-08-20T03:01:25+08:00|ARTIFACTS_COMPLETE|
|M2|3481710|3481735|2|2026-08-20T03:26:55+08:00|ARTIFACTS_COMPLETE|
|M3|3481784|3481807|3|2026-08-20T03:26:55+08:00|ARTIFACTS_COMPLETE|
|M4|3481852|3481935|4|2026-08-20T03:26:55+08:00|ARTIFACTS_COMPLETE|
|M5|3481983|3482067|5|2026-08-20T03:26:55+08:00|ARTIFACTS_COMPLETE|

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

## 最终测试闭环

- 六组训练状态均为`ARTIFACTS_COMPLETE`、`exit_code=0`，训练日志均完整包含E001–E200和终态标记。
- 六个独立评测均为`eval_exit=0`；checkpoint均为本run的`final_ssdg.pth`，`checkpoint_epoch=200`，加载时`missing_count=0`且`unexpected_count=0`。
- 每组测试204,000条clean样本，并对`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`各测试204,000条样本，共816,000条；六组共4,896,000条测试判定。
- 每个场景由`unseen day/seen receiver` 84,000条、`seen day/unseen receiver` 60,000条和严格`unseen day/unseen receiver` 60,000条组成。
- 训练和测试配置均只包含三种LEO_WEAK；没有`mixed_orbit`。

|臂|训练完成时间|独立测试完成时间|
|---|---|---|
|M0|2026-08-20T05:51:19+08:00|2026-08-20T05:53:09+08:00|
|M1|2026-08-20T06:13:53+08:00|2026-08-20T06:18:08+08:00|
|M2|2026-08-20T06:36:21+08:00|2026-08-20T06:40:28+08:00|
|M3|2026-08-20T06:33:48+08:00|2026-08-20T06:37:58+08:00|
|M4|2026-08-20T06:34:50+08:00|2026-08-20T06:38:51+08:00|
|M5|2026-08-20T06:40:06+08:00|2026-08-20T06:44:11+08:00|

## 总体结果

以下所有数值均来自同一行E200最终checkpoint，不使用中途最好值，也不跨run拼接。

|臂|clean总体|clear总体|low-elev总体|rain总体|LEO均值|LEO最差场景|严格LEO均值|严格LEO最差场景|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|M0 control|87.536%|72.490%|69.442%|69.439%|70.457%|69.439%|63.924%|62.945%|
|M1 full|84.313%|52.865%|50.796%|51.194%|51.618%|50.796%|44.736%|44.197%|
|M2 no identity structure|84.412%|53.274%|50.985%|51.416%|51.892%|50.985%|45.226%|44.620%|
|M3 no nuisance factorization|84.479%|54.804%|52.182%|52.623%|53.203%|52.182%|46.287%|45.650%|
|M4 no embedding residual|84.359%|55.646%|52.875%|53.332%|53.951%|52.875%|46.924%|46.167%|
|M5 no safety losses|84.211%|54.049%|51.495%|51.983%|52.509%|51.495%|45.188%|44.452%|

按LEO均值排序为M0>M4>M3>M5>M2>M1。M0在clean、三个LEO总体、LEO均值、最差场景、严格均值和严格最差场景上全部第一；完整NTRS M1在LEO均值和严格LEO均值上均为最后一名。

## clean测试细分

|臂|unseen day/seen RX|seen day/unseen RX|严格unseen day/unseen RX|严格RX floor|
|---|---:|---:|---:|---:|
|M0|92.181%|87.522%|81.048%|73.542%|
|M1|90.165%|82.737%|77.695%|64.975%|
|M2|90.174%|83.170%|77.588%|62.800%|
|M3|90.077%|83.208%|77.912%|66.092%|
|M4|90.089%|82.963%|77.733%|62.392%|
|M5|90.069%|82.667%|77.555%|64.575%|

`严格RX floor`是在clean严格集合中5个未见receiver各12,000条样本准确率的最小值。LEO评测当前保存的是三个命名集合及其总体值，没有逐receiver LEO切片，因此不虚构LEO receiver floor。

## LEO_WEAK逐场景、逐域细分

|臂|场景|unseen day/seen RX|seen day/unseen RX|严格unseen day/unseen RX|总体|
|---|---|---:|---:|---:|---:|
|M0|clear|79.945%|68.748%|65.795%|72.490%|
|M0|low-elev|76.369%|66.155%|63.032%|69.442%|
|M0|rain|76.398%|66.192%|62.945%|69.439%|
|M1|clear|59.748%|50.598%|45.497%|52.865%|
|M1|low-elev|57.139%|48.515%|44.197%|50.796%|
|M1|rain|57.608%|48.892%|44.515%|51.194%|
|M2|clear|60.127%|50.813%|46.140%|53.274%|
|M2|low-elev|57.207%|48.638%|44.620%|50.985%|
|M2|rain|57.824%|48.945%|44.917%|51.416%|
|M3|clear|62.018%|52.080%|47.428%|54.804%|
|M3|low-elev|58.726%|49.553%|45.650%|52.182%|
|M3|rain|59.289%|50.130%|45.782%|52.623%|
|M4|clear|62.775%|53.137%|48.175%|55.646%|
|M4|low-elev|59.489%|50.323%|46.167%|52.875%|
|M4|rain|60.001%|50.897%|46.432%|53.332%|
|M5|clear|61.719%|51.117%|46.242%|54.049%|
|M5|low-elev|58.377%|48.903%|44.452%|51.495%|
|M5|rain|59.093%|49.142%|44.872%|51.983%|

## 同行因果差分

正值表示完整NTRS M1优于比较对象，负值表示M1更差。

|比较|clean总体差值|LEO均值差值|严格LEO均值差值|结论|
|---|---:|---:|---:|---|
|M1−M0|−3.224 pp|−18.839 pp|−19.188 pp|NTRS整体显著负收益|
|M1−M2|−0.100 pp|−0.273 pp|−0.489 pp|身份结构组未显示正贡献|
|M1−M3|−0.166 pp|−1.585 pp|−1.551 pp|干扰分解组在当前权重下有害|
|M1−M4|−0.047 pp|−2.333 pp|−2.188 pp|嵌入残差是消融中最大负贡献来源|
|M1−M5|+0.101 pp|−0.891 pp|−0.452 pp|安全损失略保clean但损害LEO|

四个消融都能提高M1的LEO均值，但没有一个接近M0。最佳NTRS消融M4仍比M0低`16.506`个百分点，说明问题不是单一loss项，而是当前NTRS训练路径整体与LEO_WEAK增强发生了不利耦合。

## 训练稳定性与资源

|臂|训练参数|训练时长|峰值CUDA allocated|E200 train TX|E200 source val TX|梯度跳步/9000|优化步执行率|
|---|---:|---:|---:|---:|---:|---:|---:|
|M0|1,049,665|2:24:20|10.193 GB|90.885%|98.595%|11|99.878%|
|M1|1,138,089|3:12:22|10.322 GB|71.059%|97.762%|8|99.911%|
|M2|1,138,089|3:09:22|10.321 GB|70.799%|97.579%|8|99.911%|
|M3|1,138,089|3:06:48|10.322 GB|70.799%|97.913%|8|99.911%|
|M4|1,138,089|3:07:50|10.322 GB|70.781%|97.508%|8|99.911%|
|M5|1,138,089|3:13:06|10.321 GB|71.458%|97.690%|7|99.922%|

- 六组各有200行训练指标，epoch范围严格为1–200；六份训练日志各9,019行，六份测试日志各26行。
- 完整扫描54,114行训练日志和156行测试日志后，`Traceback`、`RuntimeError`、CUDA error、OOM、Killed、`FAIL/FAILED`、`ERROR`和warning计数均为0。
- 六组均没有非有限loss跳步。梯度跳步为AMP按batch恢复的零星事件，每组7–11个batch，占9,000个训练batch的0.078%–0.122%；没有形成连续失效，最终优化步执行率均高于99.87%。这与历史r2每轮100%跳步不同，本矩阵训练有效。
- M1的NTRS factor从E17启用，geometry/relation/margin/sat-KL从E41启用，safety从E69启用；对应消融项在M2–M5中按预登记为0，M0所有NTRS加权项为0，证明实际运行profile与矩阵设计一致。
- NTRS增加88,424个可训练参数，较M0增加8.42%；M1观察训练时长较M0增加33.3%。各run并发执行，因此时长只作为本次吞吐观察，不声明隔离延迟。

## checkpoint与可复现性

|臂|执行提交|E200 checkpoint SHA256|终态|
|---|---|---|---|
|M0|`b92648f2`|`248fe5aff24c342d6bff8e6e7d65cb5c8b250cfc318b18c655d3e57219e93690`|ARTIFACTS_COMPLETE|
|M1|`7c32ac84`|`5c124c7eca5b843cb11b16ae5f25ce9851808a4b7587b45a481f2df3e31fdef6`|ARTIFACTS_COMPLETE|
|M2|`b92648f2`|`4dc05d73111c8f3cc4691552469e6b95251274e689399bbd586dd52eb8718f12`|ARTIFACTS_COMPLETE|
|M3|`b92648f2`|`1ee982a4c8a7ecf7ccbf5eaacc7b193f840bd000c67a7293fab5ce3b9402d2b6`|ARTIFACTS_COMPLETE|
|M4|`b92648f2`|`875e990c949151c63ffef0f0c98e3a8bdc4c91a10c1ecf21cfbdbc1d98dae6cf`|ARTIFACTS_COMPLETE|
|M5|`b92648f2`|`f5e2f44292906e3e9fc263ce5fa8d9c8ff10976d61ffbd7ef7c7761d825839cb`|ARTIFACTS_COMPLETE|

M1使用先前已修复零残差梯度问题的r3提交`7c32ac84`；矩阵提交`b92648f2`仅把同一launcher参数化为六个profile并增加相应测试和报告。两提交之间没有模型或训练器实现差异；`full` profile保持M1原NTRS参数和值不变，因此M1可作为矩阵完整NTRS行。该版本差异已显式保留，不把它隐藏为同一Git提交。

## 结论与决策

1. 当前完整NTRS不是星地信道增强，不能替代M0/Core90控制路线；LEO均值和严格LEO均值分别下降18.839和19.188个百分点。
2. clean只下降约3.22个百分点，而LEO下降约18.84个百分点，说明主要问题是NTRS与LEO扰动下的表示/校正耦合，不是单纯的全局分类器崩溃。
3. 关闭嵌入残差带来的恢复最大，其次是关闭干扰分解；但所有消融仍远低于M0，因此不能只通过恢复某一个子项来宣称方法有效。
4. 本矩阵只支持Phase1 source-only、模拟LEO_WEAK证据，不代表真实卫星链路、Phase2或开放世界部署性能。六组`promotion_ready=false`，本报告不作晋级或真实星地成功声明。
5. 当前实验决策：保留M0作为同协议有效基线，NTRS完整版本及本轮四个消融均不晋级。
