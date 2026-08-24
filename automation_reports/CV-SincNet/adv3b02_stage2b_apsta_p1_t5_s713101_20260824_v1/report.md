# ADV3B02 Stage2-B APSTA-P1 Target5实验报告

## 当前状态

- run ID：`adv3b02_stage2b_apsta_p1_t5_s713101_20260824_v1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 冻结checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 协议：`p2_min_v1`、`VALIDATED_ONCE`；support/query在打开query IQ前核对`protocol_schema/phase2_data_status/capsule_id/split_id`

## 候选、预算与停止规则

- 候选：`APSTA_P1_TIME_FUSION_ROBUST`；只训练原编码器`id_backbone.t3`、`id_backbone.t_proj`和`id_backbone.fuse`
- 冻结：teacher、CosFace分类头、原型、域分支、频率分支及其余网络；不新增或训练D92式协方差、LDA、目标分类头、temperature或bias
- support-only目标：冻结头CE、anchored leave-one-out原型CE、平滑worst-class tail、prototype topology、L2-SP；不使用旧候选的逐样本冻结特征MSE
- checkpoint：`0/10/30/100/300`；仅按support robust risk与worst-class margin的硬Pareto条件选择，step0永久保留
- 真实checkpoint训练参数：`76,736/1,049,665=7.311%`，结构参数`76,224`；这是复盘报告放宽旧≤1%限制后的实测预算，不作≤1%声明
- source/clean输入`0`；query真值、角色、配额和反馈不进入predictor；query逐样本只读，不更新状态
- 系统技术停止：协议/row错误、输出碰撞、错误checkpoint/checkout、无合法prediction、scorer连接错误、确定性重复异常或进程归属不清；不得因低性能停止

## 最小预登记矩阵

- 配置：`configs/stage2b_apsta_p1_target5_s713101_20260824.json`
- 单seed=`713101`；receiver=`20-1、3-19、7-14、7-7、8-8`；`K5/new20`；3个LEO场景；共15个row
- 每row生成1份`DA1_REG0`prediction，并与既有同row`DA0_REG0`独立truth-last评分
- 旧类等权均值相对DA0_REG0至少`+1.0pp`且全矩阵floor至少`+0.5pp`才晋级Target25；否则记为`SCIENTIFIC_FAILURE_NO_PROMOTION`

## 本地证据

- RED→GREEN：APSTA核心、row绑定、矩阵与汇总测试从缺失实现失败转为通过
- 合并验证：APSTA聚焦测试和late-block/CAPTA邻近回归`48/48`通过；`git diff --check`通过
- 真实checkpoint无query smoke：strict load=`true`；support=`60`、source=`0`、query=`0`；backward=`1`；只有8个选中参数张量变化；非选中参数变化`0`、buffer变化`0`、原型不变
- smoke support选择：step0 robust risk=`7.9543`、worst margin=`-9.2059`；step1 risk=`7.2623`、margin=`-9.1824`，安全选择step1
- 独立P0/P1审查：`P0=0、P1=0、NO_P0_P1`；可进入真实实验发布

## 发布登记

- 本地工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\stage2b-lateblock-20260824`
- Git分支：`codex/apsta-p1-robust-20260824`
- 实现提交：`8bf9f752ff3cfa2a1a45c32313cbaf6e4a1a2b63`；GitHub远端分支OID独立回读一致
- release提交：`a4d4ee6ce9c161312cc402a76d48790cfcc417e3`；GitHub远端分支OID独立回读一致
- N607 Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- 不可覆盖输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_stage2b_apsta_p1_t5_s713101_20260824_v1`
- GPU：`GPU5`；启动前GPU0–4各有1个训练进程，GPU5–7空闲，选择GPU5避免叠加现有任务
- 正式命令：`run_stage2_apsta_target5_matrix.py --config <release>/configs/stage2b_apsta_p1_target5_s713101_20260824.json --release-root <release> --output-root <run-root>/results --device cuda:5`
- 预期artifact：release归档、真实checkpoint无query smoke、15份prediction、15份paired score、matrix summary、APSTA aggregate和日志
- release归档：`adv3b02_stage2b_apsta_p1_t5_s713101_20260824_v1_a4d4ee6c.zip`；本地与N607单次SHA256均为`5386ae481b0e76ebdd3302e8eae0a4c2a164d278d919c77316e40538b1841dc4`；远端编译通过
- N607真实checkpoint无query smoke：strict load=`true`，support=`30`、source=`0`、query=`0`、backward=`1`、非选中参数/buffer变化`0`、原型不变，独立JSON回读通过
- launcher实际PID=`2231723`，CWD、cmdline、GPU5子进程和事件增长已核对；提交SSH滞留时先只读确认作业已落地，再仅关闭本地SSH客户端，未重启或干预远端作业
- 实际artifact：15份prediction、15份paired score、matrix summary和APSTA aggregate完整；`ARTIFACTS_COMPLETE`、failed=`0`

## 结果

全矩阵旧类均值按15个row等权平均；全矩阵floor取15个row全部旧类准确率的最小值。

| 状态 | 旧类均值 | 相对DA0_REG0 | 全矩阵floor | 相对DA0_REG0 | 结论 |
|---|---:|---:|---:|---:|---|
| DA0_REG0 | 73.444% | N/A | 20.0% | N/A | 冻结基线 |
| DA1_REG0 | 70.000% | -3.444pp | 15.0% | -5.0pp | 未晋级 |

场景均值变化：`leo_clear_weak=-4.000pp`、`leo_low_elev_weak=-0.500pp`、`leo_rain_weak=-5.833pp`；场景floor分别为`20%→20%`、`25%→20%`、`25%→15%`。雨衰是主要失效场景。

90个row×旧类cell中，26个改善、18个不变、46个退化，单cell变化范围`-50pp至+40pp`。局部正收益row为`rx7_7_low=+4.167pp`、`rx8_8_clear=+3.333pp`和`rx8_8_low=+1.667pp`；最差row为`rx20_1_rain=-10.000pp`、`rx3_19_rain=-8.333pp`和`rx7_14_rain=-8.333pp`。

support-only安全选择为step300×12、step100×3、step0×0。所有15个row均完成300次真实反向传播，且协议边界、冻结头/原型、非选中参数/buffer和query只读审计全部通过。support robust risk与worst-class margin持续改善却未转化为query收益，说明当前support目标对接收机/场景外泛化的代理关系不足，较长更新出现系统性support过拟合；这不是工程故障，也不是协议泄漏。

最终判断：旧类均值与floor均未达到`+1.0pp/+0.5pp`联合门槛，本候选为`SCIENTIFIC_FAILURE_NO_PROMOTION`，不启动Target25或多seed。局部正收益不足以支持方法晋级。
