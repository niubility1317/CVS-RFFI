# ECRS V1R/V2实验矩阵发布与第三次复核

日期：2026-09-07。状态：`LOCAL_VERIFIED / NOT_LAUNCHED`。本次发布矩阵与可展开命令，不表示已部署N607或训练完成。

代码基线：`f7956c57c10cec641c2cc3e8fad6f4b137b5aef4`。本次未修改模型、训练器、损失系数或协议。正式矩阵以[完整配置JSON](ECRS_V1R_V2_MATRIX_RELEASE_20260907.json)和本文件为准；旧日期配置保留为历史准备快照。

## 1.复核结论

对照[设计计划第10—11节](../plans/2026-09-06-adv3b02-ecrs-v1r-v2-implementation.md)、[第二次审查](ECRS_V1R_V2_REVIEW_20260907.md)、当前配置、launcher及训练阶段路由，本轮未发现新的阻断问题。36项相关测试通过，15行展开命令全部通过真实`train.py`参数解析和修订配置验证。参数解析没有读取数据或执行训练，也不等价于真实数据训练验证。

发布作如下明确化：B1是工程检查，不安排无必要的200轮正式任务；B3与B3c配置相同，只保留B3c；B3桥接是带边界的估计器包对照；B6及训练策略对照分批展开，B7不可用。不把所有可配置行当作一次性启动清单。

第二次审查修复的空目标AdamW更新、U跳过视图、恢复参数边界及分层rescue/harm已纳入本轮回归。既有57项第二次检查和127项早期实现检查是历史证据，本轮没有重跑这些完整集合。

## 2.执行矩阵

共同seed=392005；正式性能行epochs=200，均从头训练。表中权重仅指新增响应目标，实际执行仍受版本阶段、有效样本及开关约束。

|批次/行|父控制与改动|新增响应目标|预算/用途|
|---|---|---|---|
|首批B0|匹配现有ECRS V1训练入口，关闭ECRS|无|200轮，独立主基线|
|工程B1|B0＋旁路纯计算|全部为零，融合关闭|定点及短程零作用验证；不默认运行200轮|
|首批B2-V1|本轮匹配V1 R7非融合控制|旧目标与旧接线；额外raw CE=0.30；gate=0|200轮，整套修复包对照|
|首批B2|B2-V1→V1R修复包|独立response CE=0.15；保留旧物理目标；额外raw CE=0|200轮，不解释为单一头或单一梯度消融|
|第二批B3a|旧参考＋legacy28统一估计/读取/冻结物理包|response CE=0.15|200轮；相对B2不声称solver单因素因果效果|
|第二批B3b|B3a更换为estimated reference路径|response CE=0.15|200轮，保持其余估计约定|
|第二批B3c|B3b→compact8字典/尺度/去冗余包|response CE=0.15|200轮；与B3同配置，不重复执行|
|第二批B4|B3c的real8改为complex24锚点|response CE=0.15|200轮，读取方式对照|
|第二批B5-X|B4＋跨RX表示判别|response CE=0.15，cross-RX=0.05|200轮，保持旧采样|
|第二批B5-U|B4＋同物理样本clean/LEO编码器EMA|response CE=0.15，U=0.03|200轮，U不读取TX真值|
|第二批B5-XU|B4＋两项目标|response CE=0.15，cross-RX=0.05，U=0.03|200轮，组合判别对照|
|第三批B6|B5-XU＋固定切向融合|前三项＋fusion CE=1.0；rho=0.05|200轮，E91启用融合，初始隔离raw/response的融合梯度|
|策略S-LR|B5-XU仅改有效更新步LR|与B5-XU相同|200轮，独立对照|
|策略S-LEO|B5-XU仅改提前线性LEO CE课程|与B5-XU相同|200轮，独立对照|
|策略S-BATCH|B5-XU仅改TX/RX平衡采样|与B5-XU相同|200轮，独立对照|

B3a/b/c、B4及其后各行采用固定估计器，旧canonical/content/cycle/split-fit/pair等物理loss系数全部为0；不能把物理诊断算作已训练目标。配置中保留的非零候选系数，在对应开关关闭时也不代表执行。V2判别目标从训练开始按开关启用；V1/V1R遵循旧分阶段课程，因此跨版本差值不是同等响应训练暴露的单因素对比。

B2旧物理目标权重：canonical/content/cycle/split-fit/pair-cross各0.10，pair-surface=0.03，same-TX=0.05，different-TX=0.03，gate=0。各项目按R7阶段启用，不能仅据此表声称实际激活。

上述0.15/0.05/0.03/1.0均为原设计候选值，尚未由真实source结果验证为最优。B7动态gate、compile及多策略组合不在本次执行清单。

## 3.共同训练与选择约束

- Phase1 source-only：ManySig、equalized=1、RX=`1,3,4,6,8`、day=`1,2,3`，L/U/V=`0.07/0.63/0.30`；物理ID隔离，V只读，target不用于选择。
- 既有双骨干/PA训练入口：`s4_stagewise_full_dual`、CV-SincNet M/lite_d、no_dac/no_stats；属于`matched_ECRS_V1_train_py`，不宣称完整SSDG Core90目标复现。
- 常规batch=128、eval batch=256、LR=0.0002、最低LR=0.000001、weight decay=0.0001、label smoothing=0.01。S-BATCH单独改变采样结构，不能解释为仅loss改变。
- 默认LEO概率为E1起0.30、E41起0.60、E91起0.80；默认LEO CE在E80起、权重0.68；S-LEO为独立课程对照。
- E200正式行采用48次source验证时点。每次分别记录clean及三项`leo_*_weak`，保持物理样本与视图绑定。
- B0/B1/B2-V1/B2及非融合V2行按raw路径选择；B6按既定融合规则选择，保留同row raw/response/fused，不事后切换最高分的头。
- 恢复必须满足当前完整语义及状态检查；旧版本checkpoint不能冒充exact resume。改版沿用权重需另立run并说明，不进入从头主对照。

首批只安排B0/B2-V1/B2与B1必要工程检查。第二批依source证据展开桥接和判别目标；B6用于后续互补性检验。策略行分别相对B5-XU，不提前组合；不因低准确率停止健康训练。

## 4.科学判定及证据边界

沿用设计计划，所有数值规则在查看target前固定：B6相对同row raw，source-clean下降≤0.5pp、clean最弱RX下降≤1pp、三LEO均值提升≥1pp、最弱RX×LEO单元下降≤1pp、rescue−harm>0；还须独立对匹配B0报告同样的clean/floor退化界和LEO收益。记录batch=1的p95延迟，建议不超过B0的1.25倍。多seed确认只用于source筛选后的候选，不由本文件自动启动。

真实source参考匹配、独立预测增益、TX/RX/view probe、响应秩/覆盖及实际资源测量仍缺证据。当前参考为estimated reference，不能称协议参考已完成对齐。单seed、合成检查或同row融合收益均不足以晋级默认方案。没有收益时保留负结果，记`NO_PROMOTION_TO_DEFAULT`。

本launcher仅做source筛选，拒绝训练中开启最终target评估。完成source行记`SOURCE_SCREEN_COMPLETE`；正式Phase1最终闭合仍需冻结checkpoint后独立完成clean及三LEO结果，不把本次配置发布写成`ARTIFACTS_COMPLETE`。

## 5.可复现命令与部署说明

在核实的项目环境和Git根目录运行以下命令生成首批计划；`<...>`必须替换为部署主机的真实路径，输出root须尚不存在。默认只输出JSON。

```text
python code/scripts/launch_phase1_adv3b02_ecrs_v1r_v2.py --root <checkout> --run-root <new-run-root> --wisig-pkl <ManySig.pkl> --rows B0,B2-V1,B2 --seed 392005 --epochs 200
```

附带JSON保留Windows准备路径及每行完整argv，是展开模板，不能直接当N607落地命令。部署时在目标主机重新展开；现有launcher的`--execute`是本机顺序执行，不提供多GPU调度，也不替代N607最小发布流程。

N607启动时按仓库流程补齐精确CWD/解释器、数据/run/log路径、GPU分配、唯一launch owner、不可覆盖run ID、当前进程资源检查及一次真实checkpoint无query smoke，独立P0/P1审查仅检查直接正确性。发布后核对PID/CWD/cmdline/GPU与log增长。每GPU最多两个训练实验，不干预其他健康任务。

技术停止限定协议/输入/checkout错误、输出碰撞、无法执行或无法形成合法prediction等既定系统故障，保留产物，不因性能低而停机。本次没有访问、部署或启动N607。

## 6.本轮验证记录

环境：Windows原生`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，显式UTF-8。

```text
python -X utf8 -m pytest code/tests/test_ecrs_second_review.py code/tests/test_ecrs_v1r_v2_config.py code/tests/test_ecrs_v2_bridges.py -q
```

结果：36项通过，exit=0。另从真实`train.main()`捕获参数解析器，在读取数据前停止，对15行完整argv逐一解析并执行`validate_ecrs_revision_args`，全部通过；E200验证时点数=48。JSON按UTF-8写入后重新解析，未创建其中声明的run root。发布文件另做UTF-8、链接、配置一致性和Git diff检查。
