# Phase1 LEO_WEAK默认策略

生效日期：2026-08-20

## 范围

本策略适用于新建Phase1训练、最终checkpoint独立测试和协同推理评测入口。默认星地信道增强族为`LEO_WEAK`，包含`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 行为

- 未提供`--sat_train_scenario`和`--sat_train_scenarios`时，训练轮换三种`LEO_WEAK`视图。
- 最终星地测试默认逐项评估并保留三种`LEO_WEAK`指标；clean结果仍作为无星地增强对照。
- 显式给出的单场景或多场景参数优先于默认值，因此历史`mixed_orbit`复现不会被静默改写。
- `mixed_orbit`不再是新建Phase1训练或测试的默认值，只能作为显式历史复现、对照或诊断压力路径。

## 实现与核验

共享默认值和场景解析位于`code/training_controls.py`。通用视图回退、`SSDG.train_ssdg`、`train.py`、CRRA配置、post-stage评测和协同评测入口均消费该策略。聚焦回归测试已通过，覆盖默认三场景、CRRA默认、显式`mixed_orbit`优先级、post-stage默认和协同评测CLI默认；相关模块`py_compile`通过。

本次聚焦组中`code/tests/test_phase1_jointp0_core.py::test_unlabeled_geometry_masks_do_not_reapply_pseudo_confidence_to_direct_or_invariance`存在可重复的既有断言失败。该函数与本次差异没有交集，因而未在本策略变更中修改；其余与LEO_WEAK默认直接相关的用例均通过。

本策略不改动已启动实验、历史checkpoint、历史日志或已显式冻结的命令。
