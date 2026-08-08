# Phase1双读出deployment bundle v2最终修复发布报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase1_dualreadout_bundle_v2_20260808_v3`
- 状态：`LOCAL_VERIFIED / P0_P1_CLOSED / READY_FOR_N607`
- 证据等级：`TECHNICAL_BUNDLE_NOT_PERFORMANCE_PROMOTED`
- 时间：2026-08-08
- 实现commit：`8208c9370c48832c370e3569c9197d51cd21072c`
- 发布修复轮次：`2/2 FINAL`

## 1.目标与唯一修复

v3继承v2的全部方法、输入、校准、阈值、矩阵和设备锁，只修复TorchScript导出包含训练期`GradReverse`的问题。runtime不再调用或注册完整训练forward，而只封装`id_backbone`、`dom_backbone`、`identity_capacity`和`dom_enhancer`组成的部署必要子图，复现模型的`z_id/z_dom/tx_logits`两种representation分支。对抗域头、TX对抗头和gradient reversal均不进入runtime。

本地测试用一个forward必定抛错的训练模型证明部署wrapper不会调用完整forward，并验证eager与TorchScript三输出一致。`ssr-gpu`共13项focused tests、语法和`git diff --check`通过。

独立定向复审：`P0=0`、`P1=0`、`ALLOW_FINAL_REPAIR_V3=YES`；额外以真实Dual与`single_parameter_matched`模型验证精确parity、trace图无GradReverse且`torch.jit.save`成功。固定commit的module/script/test归档SHA256分别为`c177dc87d547bf2f74b11808cec31343805151e80c472744fe8e4e2440d55896`、`8c65c08617f0903681331fb4cac5c8191d22e632a0d8765e809b8dfde31dec56`、`c53ffae3dadbd497ccbac3f1ce06347441c1b1a2781e3c34bbb8a2561b147a60`。

## 2.冻结矩阵与设备

输入SHA、C `z_dom`导出参数、source-only build、2400条emit和两次score命令与v2完全相同。GPU0导出B runtime，GPU1导出C runtime，GPU2重新独立导出C `z_dom`；不复用v2 partial。三条子进程必须保持`CUDA_VISIBLE_DEVICES=<UNSET>`并分别传物理`cuda:0/1/2`，每卡1个本run进程。

## 3.N607不可覆盖路径

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_dualreadout_bundle_v2_20260808_v3_8208c937`。
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_dualreadout_bundle_v2_20260808_v3`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dualreadout_bundle_v2_20260808_v3`。
- CWD：`<release>/code`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；retry=`NO`。

## 4.终止边界

v3是最后一轮发布修复。任何runtime、parity、设备或子图导出错误都立即封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不再修补同一入口；后续只能冻结更小的独立one-shot exporter并使用全新候选/run ID。

成功和回收规则不变：两runtime parity、C `z_dom`逐行绑定、exact bundle/content root、source-only smoke、2400条无role/truth evidence和两份非部署诊断全部完成；只回收小artifact，不下载checkpoint、NPZ、runtime、calibration或完整evidence。

## 5.结果表（待回收）

| candidate | category | receiver/TX split | K-shot | seed | known/unknown | coverage/defer | bundle summary | verdict |
|---|---|---|---:|---:|---|---|---|---|
| P1-DUALREADOUT-BUNDLE-V2-FINAL-REPAIR | deployment-subgraph bundle | 4/1/1 | N/A | 7281105 | 待回收 | 待回收 | C class/domain+B continuous JS | `NO_PERFORMANCE_RESULT_YET` |

## 6.科学边界

proxy和held结果仅为`SOURCE_HELD_PROXY_NONDEPLOYMENT_DIAGNOSTIC`，不构成Phase3真实unknown、same-event多节点或真实在轨验证。
