# FCR-V2完整矩阵预登记

- 状态：`LOCAL_VERIFIED`
- run ID：`phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1`
- 实现提交：`4fba1fd43be8580f12815e839ec4d5ffbc0d5604`
- 分支：`codex/adv3b02-fcr-r1r8-s392005-20260903`
- 环境：N607普通账户，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1`
- 数据：`Dataset_WigSig/ManySig.pkl`，equalized，split seed=`392005`；source receiver=`1,3,4,6,8`，day=`1,2,3`，TX=`0..5`；target receiver=`0,2,5,7,9,10,11`，day=`0,1,2,3`
- 初始化checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`
- checkpoint身份：seed=`392005`，epoch=`200`，candidate=`ADV3B02_CORE90_SOFT_E200`
- 固定基线：clean=`76.2268%`，LEO均值=`60.1397%`，四场景均值=`64.1615%`
- 矩阵：C0只评估；训练C1、C2、C3、S0、S1、S2、S3、S4、M1、M2、M3、M4、M5、M6，共14个E200 final-only行
- GPU：wave1的8行分别使用GPU0-7；wave2的6行分别使用GPU0-5；每张卡本矩阵同时只启动1行
- 启动命令：`RUN_ID=phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1 ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release-root> bash code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1`
- 预期artifact：14个`final.pth`、14份FCR诊断、15行四场景prediction、独立truth sidecar、15份独立score
- checkpoint选择：禁止使用target筛选，固定每行E200最后一个epoch的`final.pth`
- 评价边界：训练全部完成后才统一准备target输入；prediction先写出，独立scorer后连接truth；第二波发布不读取target结果
- 技术停止规则：仅在协议/query泄漏、错误seed/receiver/day/scenario、输出冲突、错误checkout、确定性重复异常、无prediction闭合、scorer连接错误或进程归属不清时停止精确run进程树；低性能不停止
- 机制解释边界：能力门控不能中止矩阵。M4/M6只有在真实pair、非零loss和非零梯度诊断同时成立时才能解释为机制已激活，否则报告`MECHANISM_NOT_ACTIVATED`
- 审查裁定：修复后定点复审的唯一遗留意见要求因ManySig无可验证fingerprint pair而禁止Task8；该要求与设计第140行“门控不足仍训练到E200”冲突，记录为`REJECTED_EXTRA_GATE`，不延迟发布
- 变更范围：新增FCR-V2metadata、pairing、factor、physics、loss、schedule、diagnostics模块；最小接入模型、训练、truth-last predictor/scorer和完整矩阵launcher
- 本地验证：`phase1_fcr`173项通过；训练图/诊断/launcher49项通过；checkpoint→predictor→scorer26项通过
- release映射：本地`release_archives/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v1.tar.gz`

## 启动结果

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 启动时间：2026-09-03 18:11 CST
- launcher PID：`838673`，已自行退出；检查时没有本run存活进程
- 失败范围：wave1的C1、C2、C3、S0、S1、S2、S3、S4均在训练开始前以相同指纹退出；wave2未发布
- 确定性指纹：`RuntimeError: locked mature identity checkpoint is incomplete: missing_mature_identity=42`
- 根因：launcher漏传基线的`branch_ablation=no_dac`和`domain_branch_ablation=no_stats`，错误构建额外DAC身份路径并把它判为旧checkpoint缺失
- 处置：保留全部日志和空训练目录，不原地修改、不覆盖、不复用run ID；修复提交`524f835536b5f467a72055975fd2d2b415b6843a`后改用v2重发

