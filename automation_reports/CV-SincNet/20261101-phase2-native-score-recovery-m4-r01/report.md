# 原生外部对比冻结预测的独立评分补全

GPU环境已恢复，8卡可识别且独立CUDA allocation/synchronization通过。本次为技术故障后的独立恢复，不更改原科学方法、矩阵、种子、数据权限或预算；已完成结果优先复用，不覆盖原root。

User2026-11-01 complete remainingcomparison and reusecompleted; independentCPUscore-onlyroot for alreadyfrozen originalnine nativeprofile predictions; originalowner/root read-only, no takeover/restart, no training/prediction

Originalr10 controller is verifieddead; this independentnewroot does notmodify,restart or take over itsowner. Source andprediction handles remainoriginal; no duplicatedprediction orsourcefit.

Beforeeachmissing score, recheck oldcomplete; ifsameprediction haslive oldscorer waitwithoutkill. Oldcompletepayload capsule/seed/rows matchfrozenprediction; otherwise independentoriginalscorer afterpredictionclosure.

Scorerconsumes noNNcheckpoint, soNNsmoke/sourcefit do not apply. PureCPUnegativeinput checks andoneP0/P1 concern only correctfrozenpredictionbinding/outputs/queue, no report-onlyextra gate.

ISSL userstop retained; Tweak/CSIL N/A; currentTriSSL PAMF baseline scores alreadycomplete andnotresubmitted; MoPCseparate recovery owner.

精确命令、环境、逐行/分组矩阵、数据/源模型来源、输出、GPU与停止规则见experiment.json。实际运行先执行当前阶段所需smoke，再自动补齐预测与独立评分，最后分析/资源表/登记与Git交付。评分已完成并经逐任务产物核实；Group1 性能比较仍等待 R42 与 MoPC。

启动后独立核实（2026-11-01T13:21:44.188663+08:00）：VERIFIED RUNNING: independentCPUscore-onlyroot,17397fixedoriginalpredictioninputs; no training/prediction/model orIQ loads; initialfixedpredictionvalidation and actual reusedscoreprogress/loggrowth observed; originalowner/root remainread-only.

控制器实际PID/CWD/argv与子进程、GPU、日志/预测/拟合产物增长已核实。源代码/config commit b4903b6928d88f755b6609a6160a0dde0acf22b0，双仓库push和remoteOID VERIFIED；单一归档传输SHA比对、远端compile通过。25项定向测试和每个恢复入口唯一一次P0/P1均通过，无额外审查/签名/成员hash。本评分矩阵已闭合；R42 与 MoPC 仍在执行，整体比较和 R42 的 +5pp 逐格结论待它们完成。

## 独立评分完成

核实时间：2026-11-01T15:29:03.424094+08:00。VERIFIED：17397／17397 个评分任务完成，覆盖 41328 条评分记录。复用 13219 个旧评分任务，独立补评 4178 个；本恢复 run 新训练、新预测均为 0。控制器正常退出。

逐任务读取全部 metrics.json 与原 predictions.json，核对 seed、capsule、split、注册类别、K/N、stage、arm、路径、prediction 行及完成标记，全部一致。没有重新读取 query 真值或 IQ。原 source/prediction 与 owner 保持不变。

| 原生 profile | 完成评分任务 | 指标行数 |
|---|---:|---:|
| protonet_cda | 1933 | 4368 |
| mrior_sda | 1933 | 4368 |
| dadda_sda | 1933 | 4368 |
| twostage | 1933 | 4368 |
| feature_separation_hu2024 | 1933 | 4368 |
| poster_ft | 1933 | 4368 |
| radionet_ada_knn | 1933 | 4368 |
| orthogonal | 1933 | 5376 |
| lora_triplet_knn | 1933 | 5376 |

完成状态为 ARTIFACTS_COMPLETE；同组 R42、MoPC 未闭合，整体比较分析仍待执行。ISSL 保持停止，Tweak/CSIL 保持 N/A，Hu scaled 不改标。汇总见 [score_completion_summary.json](score_completion_summary.json)。
