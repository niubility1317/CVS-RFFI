# CORE90锚定几何：复审问题修复与再审查

日期：2026-09-11。修复基线：`9e2a47fac5a663c15a302ff409da3c7e12fa6dcc`；分支：`codex/core90-evidence-head-20260911`。

**结论：上一轮4项缺口及独立审查新增2项边界问题均已修复，77项相关测试通过；本轮审查无剩余P0/P1/P2发现。**适用首轮设计实现恢复为42项verified、6项按计划deferred、0项implemented/pending/blocked/rejected。这里的verified只指实现与技术验证，不表示正式source矩阵、机制收益或独立确认已经完成。

依据：[设计原文](core90_anchored_geometry_design_source_20260911.md)、[实施计划](core90_anchored_geometry_implementation_plan_20260911.md)、[问题复查记录](core90_anchored_geometry_recheck_20260911.md)、[逐项追踪表](core90_anchored_geometry_traceability.md)。旧复查及测试产物保留为历史证据，本报告说明其后修复。

## 1.问题闭环

|问题|修复及当前行为|实际验证|
|---|---|---|
|F1/P1：OOF最终专家与校准/导出断链|`oof`直接使用返回的final_expert保存`system_state.pt`；同candidate/seed/config/cache身份的OOF产物可进入V校准|文件化source阶段测试执行5个4RX＋1个5RX计数拟合，随后禁用额外fit完成calibrate/export；导出专家状态与OOF全source专家逐状态身份一致|
|F2/P1：utility训练与部署数值路径不同|训练、固定动作审计、嵌套评价和部署均把原始logits交给`realize_actions`，统一在FP64执行概率变换|120行六类合成混合边界及保护区测试，5个动作逐元素概率完全相等，预测、实际alpha及utility一致；不是实际数据错误率估计|
|F3/P2：缺少三seed汇总入口|新增显式`aggregate_core90_anchored_source.py`，读取三个已完成A6嵌套source目录，核对配置、checkpoint/cache/contract、seed与完成状态；汇总不加载模型、不再拟合|真实子进程调用CLI读取合成文件产物，形成三个seed×16组=48行A6相对inner选定A5的差值；duplicate seed、checkpoint/contract/config不匹配及未激活状态被拒绝|
|F4/P2：缺TX仍可返回支持|对A6和A5_inner_selected核对冻结定义中的完整、唯一RX/TX/day/all/clean集合，检查必需指标与数值合法性|缺TX、重复RX、错误RX、缺day、缺A5、NaN、Inf均返回`INCOMPLETE_SOURCE_EVIDENCE`，不会进入支持结论|
|再审查P2：极小唯一H0分差被归一化抹平|alpha=0/1显式保留原专家决定；utility、嵌套输出、报告和部署共同消费`predictions`；最终温度只校准置信度并保留冻结决定|`[0,1e-20,-1e-20]`的原H0类别1在未校准/已校准alpha0仍为1；门控训练与部署一致；不通过扰动概率伪造间隔|
|再审查P2：数学上不可能的有限指标仍可支持|验证隐含H0准确率；按λH=2从net/utility反解rescue/harm的可行范围；RX/TX/day行数分区必须等于all，clean行数不能超过all|不可能baseline、utility=100、错误分区计数及缺失rows均返回不完整；合法三seed完整证据仍通过|

预算仍为5种专家×3head seeds×6次=90次基础拟合；A4条件性嵌套最多额外30次，共120次。这里验证的是调度复用和预算定义，没有实际执行90/120次正式拟合。汇总为新增独立CLI，不改已有七阶段配置schema或自动触发训练。

## 2.机制和参数复核

本次未修改配置、几何头或拟合模块。G仍为r=4、ρ=log(2)/2、条件数≤2，只训练B/b；W/τ冻结。clean/LEO仍各0.5，Adam学习率0.001、weight_decay=0，固定80轮和20/40/80诊断；λmetric=0.01，A4/C_angle_keep的λkeep=1/τ²、γmax=0.1τ。A6仍采用五个离散alpha、λH=2、ridge=0.001、utility_margin=0、保护分位数0.9；正净收益触发条件不变。V最终coverage=0.9，温度范围0.05–20；P1及延期分支参数未改。

先前真实两步试验的keep梯度很弱，不足以证明有效保护；本次没有擅自加大系数、开放延期模块或以旧target反馈调参。相关测试覆盖参数冻结和分支接线，但A6/P1在正式source上的激活和收益仍需完整实验产物。当前最高剩余风险是科学效果未证实，以及已有target启发不能转写为新的独立确认。

### 数值与产物语义

`log_probabilities`仍保存真实计算概率；`predictions/decision`保存实际执行的冻结决定。在极小唯一分差归一化后发生浮点并列时，概率的重新argmax不能代替原专家端点决定。source的utility、nested及报告入口已统一消费显式决定；最终校准只改变置信度，不重新选类。真实原始并列仍遵守原专家的固定tie policy。

旧版本融合门控、校准或汇总技术产物保持原样，不冒充本修复版本的验证证据。若使用旧版本生成的融合状态，需要在同一source授权范围内重新执行受影响的fuse/calibrate/export阶段；OOF专家可继续通过既有身份检查复用，不需要额外重训专家。当前没有已完成正式A6矩阵可据此声明升级成功。

## 3.入口与交付文件

独立专家直接复用OOF最终系统，输入文件是`<oof目录>/system_state.pt`，不是`all_source/expert.pt`。原有`run_core90_anchored_geometry.py`的`calibrate/export`参数保持不变。

三seed汇总示例（占位路径，未启动正式矩阵）：

```text
python -X utf8 code/scripts/aggregate_core90_anchored_source.py --inputs <A6_seed392005_fuse目录> <A6_seed392006_fuse目录> <A6_seed392007_fuse目录> --contract <本次数据契约.json> --output <新的汇总目录>
```

输入为三个已经完成嵌套评价的A6 fuse目录；契约必须给出当前六类`tx_mapping`。输出包括`all_seed_source_metrics.csv`、`a6_vs_inner_a5.csv`、`source_promotion.json`、`protocol_manifest.json`与`report.md`。缺组数据保存不完整判定，不伪装为指标未达标；跨身份、重复seed、未完成状态在写入输出前拒绝。

## 4.实际验证与独立审查

已验证解释器：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，PyTorch2.10.0+cu128。CPU执行，无N607访问、无真实target读取、无正式训练启动。

先补回归后修复：第一批原问题测试10项失败；修复后首轮完整相关测试72项通过。独立审查新增两处P2对应的4项反例先失败，修复并补充缺失rows及tiny-gap门控覆盖后，最终结果为**77项通过、0失败、0错误、0跳过，39.992秒**。[最终JUnit](core90_anchored_fix_evidence_20260911/final_tests.xml)是当前证据；[首轮72项](core90_anchored_fix_evidence_20260911/acceptance_tests.xml)及[定点回归](core90_anchored_fix_evidence_20260911/review_followup_tests.xml)保留过程记录。

实际运行：

```text
python -X utf8 -m pytest code/tests/test_anchored_cache.py code/tests/test_anchored_geometry.py code/tests/test_anchored_fit.py code/tests/test_anchored_crossfit.py code/tests/test_anchored_fusion.py code/tests/test_anchored_calibration.py code/tests/test_anchored_pipeline.py code/tests/test_anchored_source_stages.py code/tests/test_anchored_reporting.py code/tests/test_anchored_recheck_fixes.py code/tests/test_partial_evidence_patterns.py code/tests/test_partial_gaussian_head.py code/tests/test_pairwise_evidence.py code/tests/test_evidence_integration.py code/tests/test_evidence_pipeline.py -q --tb=short --junitxml=analysis/core90_anchored_fix_evidence_20260911/final_tests.xml
python -X utf8 code/scripts/run_core90_anchored_geometry.py --validate-config
python -X utf8 code/scripts/aggregate_core90_anchored_source.py --help
```

独立审查采用一轮变更审查及对新增两处问题的一次定点复核。审查者独立执行6组最终检查，确认生产utility、outer动作、nested预测、融合报告和校准均消费显式决定，数值/完整性反例不再通过；本轮无剩余发现。既有`model.py`的autocast弃用警告不影响结果，本次未扩展修改该无关模块。

正式代码和报告位于隔离Git承载面`E:/type10-7/code/snapshots/core90_evidence_20260911_wt`；根目录`analysis`镜像报告与小型证据。提交和push后需独立读回远端OID，结果记录在本次交付回复。
