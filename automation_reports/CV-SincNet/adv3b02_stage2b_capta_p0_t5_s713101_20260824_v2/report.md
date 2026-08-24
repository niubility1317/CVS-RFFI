# ADV3B02 Stage2-B CAPTA-P0 Target5重试实验报告

## 结论状态

- run ID：`adv3b02_stage2b_capta_p0_t5_s713101_20260824_v2`
- 当前状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 重试原因：v1在首个prediction前因N607 NumPy2.2.5/PyTorch2.1.0数组桥技术故障自然退出，无prediction、无score；v1全部artifact原样保留
- 修复：所有CAPTA NumPy→Torch边界统一改为`tolist()`后的显式`torch.float32`值复制；新增禁止`torch.as_tensor`的完整adapt+query回归负测
- 冻结checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 协议：`p2_min_v1`、`VALIDATED_ONCE`，support/query在打开query IQ和prediction前严格核对`protocol_schema/phase2_data_status/capsule_id/split_id`

## 方法与预算

- A0：同row既有冻结`DA0_REG0`prediction
- A1：support类中心球面收缩；A2：类均衡共享平移加收缩；A3：target support类残差rank-4 SVD迁移加收缩；A6：support leave-one-out安全门控
- 不新增或训练协方差、LDA或持久分类头；训练参数`0`、反向传播`0`、适配更新`0`步
- source/clean输入`0`；query真值、角色、配额和反馈均不进入predictor；query只读且不更新状态

## 最小矩阵与晋级规则

- 配置：`configs/stage2b_capta_p0_target5_s713101_20260824.json`
- 单seed=`713101`；receiver=`20-1、3-19、7-14、7-7、8-8`；`K5/new20`；3个LEO场景；共15个row
- A1/A2/A3共45份`DA1_REG0`prediction，每份与同row A0独立truth-last评分
- 旧类等权均值相对A0至少`+1.0pp`且全矩阵floor至少`+0.5pp`才晋级；否则`SCIENTIFIC_FAILURE_NO_PROMOTION`，不扩大Target25或多seed

## 发布登记

- 本地工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\stage2b-lateblock-20260824`
- Git分支：`codex/stage2b-capta-p0-20260824`
- 基础实现提交：`dc707ef079c3690337fa8ff340f943610b3e390f`，本地与GitHub远端OID一致
- 修复提交：`af9a5b49857a61815649d3c6cef5ce74ca9f9f44`，本地与GitHub远端OID一致
- N607 Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- 不可覆盖输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_stage2b_capta_p0_t5_s713101_20260824_v2`
- release归档：`adv3b02_stage2b_capta_p0_t5_s713101_20260824_v2_af9a5b49.zip`；本地与远端单次SHA256均为`14bb2cd7201a9cf277c9c57e78d0fff5692dea62de1d059048478cd8d356a07c`；远端编译通过
- GPU：`GPU0`；启动前用户Python任务为0、显存`1/24576MiB`；launcher PID=`2147369`，CWD、cmdline、GPU子进程和事件增长均已核对，完成后正常退出
- 正式命令：`run_stage2_capta_target5_matrix.py --config <release>/configs/stage2b_capta_p0_target5_s713101_20260824.json --release-root <release> --output-root <run-root>/results --device cuda:0`
- 实际artifact：release、真实checkpoint无query smoke、45份prediction、45份paired score、matrix summary和日志均存在；summary=`ARTIFACTS_COMPLETE`、failed=`0`
- 系统技术停止：协议/row错误、输出碰撞、错误checkpoint/checkout、无合法prediction、scorer连接错误、确定性重复异常或进程归属不清；不得因低性能停止。若v2再现同一兼容错误则停止，不再盲目重试

## 本地证据

- 原兼容路径负测按预期失败于`torch.as_tensor`；显式值复制修复后该测试通过
- CAPTA聚焦、row绑定与late-block邻近回归合并`41/41`通过；Python编译和`git diff --check`通过
- v2 release的N607真实checkpoint无query smoke：strict load=`true`，support=`30`、source=`0`、query=`0`、rank=`4`、trainable=`0`、backward=`0`、model_changed=`false`
- 独立审查与唯一一次定点复审终局：`P0=0、P1=0`

## 结果

全矩阵旧类均值按15个row等权平均；全矩阵floor取15个row全部旧类准确率的最小值。

| 状态/候选 | 旧类均值 | 相对A0 | 全矩阵floor | 相对A0 | 结论 |
|---|---:|---:|---:|---:|---|
| DA0_REG0/A0 | 73.444% | N/A | 20.0% | N/A | 冻结基线 |
| DA1_REG0/A1 | 73.333% | -0.111pp | 20.0% | 0.0pp | 未晋级 |
| DA1_REG0/A2 | 73.556% | +0.111pp | 15.0% | -5.0pp | 未晋级 |
| DA1_REG0/A3 | 73.556% | +0.111pp | 15.0% | -5.0pp | 未晋级 |

场景诊断：A2/A3在`leo_low_elev_weak`均值提升`+1.167pp`，但在`leo_rain_weak`均值下降`-1.0pp`且场景floor从`25%`降至`15%`；A1在`leo_rain_weak`下降`-0.5pp`。局部收益不能抵消全矩阵均值不足和floor退化。

45份prediction均为`DA1_REG0/PREDICTIONS_COMPLETE`：`p2_min_v1`、`VALIDATED_ONCE`、source输入0、trainable 0、backward 0、model unchanged、query truth/role未加载、query状态未更新；每份520条query。45份score均在prediction完整性验证后连接520条truth，15个row的A0指标在三个候选间完全一致。

最终判断：没有候选同时达到`旧类均值+1.0pp`和`floor+0.5pp`，因此本轮为`SCIENTIFIC_FAILURE_NO_PROMOTION`。按预登记停止，不启动Target25或多seed。工程与协议闭合不等于科学正收益。
