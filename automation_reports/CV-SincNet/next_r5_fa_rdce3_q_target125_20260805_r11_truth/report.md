# NEXT-R5 FA-RDCE3→qKNN Target125 truth/score报告（r11）

## 身份与目标

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r11_truth`；日期：2026-08-05；状态：`LOCAL_VERIFIED`。
- 目标：复用r10不可变完整prediction manifest，仅完成truth-open与score，不重跑125 prediction。
- r10 prediction闭合：125/125 outer、375/375 scene、1500/1500 logical、1350 unique+150 alias；manifest=`33027483`bytes，SHA=`29982326a4c3130abea223a56c19362f9cf7e583da186e3498de2b80758c0c49`。
- r10 truth-open失败根因：truth adapter传给D108 sidecar loader的outer漏`old_classes/new_classes`，在任何truth评分前触发KeyError。
- r11从source_context绑定的D92 before_apply/after_apply sealed package manifest取得预truth registry，验证包seal、receiver/seed/K、连续class_index、旧类前缀和new_count，并与r10四状态`registered_classes`元数据逐scene绑定；不读取predicted labels、query truth、role或IQ。
- 科学commit=`d17b29d0aa1840c6d4dffa32034745b952ff828f`；六入口编译和36项聚焦测试通过；独立Terra复核`P0=0，P1=0`。

## 闭包与冻结输入

- closure=`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r11_truth_closure_d17b29d0.tar`；73175040bytes；SHA=`94f106b7c237e034d3c1e50e176adaec528db392fac18660ae03246639d7b810`。
- truth adapter/CLI/D108 scorer/runtime SHA=`91bd6c0c80bad50ec431db37a7941ce8cf9edd62b10703c5b665c08072bfc797`/`6cd8c9c682e5cc4ac05b7c560c887858a5d5ec62147a4b3d535321806617c06c`/`ee64b32359599acba152487b8673ebae386f7d63e2d095ee8186275e5efad766`/`38a69778a3131a144fcece1d7bd066f50c829852aba1c43d265f2c1a29ccbb38`。
- r10远端prediction=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r10/merged/prediction_manifest.json`，必须先验证上述SHA与大小。
- r10 prepared plan/context实产SHA由r10报告与runner证据原样传入；不得重新prepare或修改prediction。

## N607执行

- RUN_ROOT=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r11_truth`，先为`ABSENT`；r1至r10只读。
- 只执行：preflight→closure land/hash/compile→读取并验证r10 prediction+prepared→detached truth-open→score→取回。无GPU prediction、无asset build、无prepare、无smoke、无shard、无merge。
- truth catalog和score output-dir必须为r11新路径；truth-open进程必须绑定r11 source CWD和r10 immutable manifest SHA。
- score必须覆盖125 outer、375 scene和四状态；最终按同一行报告DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1及DA/注册主效应和interaction。REG0的新类/H=`N/A`。
- 失败只按truth/score技术错误处理，不得修改或删除r10 prediction；fresh retry authority=`无`。

## 结果待填

|prediction|truth|score|结论|
|---|---|---|---|
|r10完整manifest已封存|未产生|未产生|`LOCAL_VERIFIED`|

