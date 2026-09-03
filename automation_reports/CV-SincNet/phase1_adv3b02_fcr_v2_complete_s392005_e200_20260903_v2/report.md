# FCR-V2完整矩阵重发预登记

- 状态：`LOCAL_VERIFIED`
- run ID：`phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v2`
- 修复提交：`524f835536b5f467a72055975fd2d2b415b6843a`
- 分支：`codex/adv3b02-fcr-r1r8-s392005-20260903`
- 环境：N607普通账户，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v2`
- 数据：`Dataset_WigSig/ManySig.pkl`，equalized，split seed=`392005`；source receiver=`1,3,4,6,8`、day=`1,2,3`、TX=`0..5`；target receiver=`0,2,5,7,9,10,11`、day=`0,1,2,3`
- 初始化checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`
- checkpoint身份：seed=`392005`，epoch=`200`，candidate=`ADV3B02_CORE90_SOFT_E200`；架构锁=`no_dac/no_stats`
- 固定基线：clean=`76.2268%`，LEO均值=`60.1397%`，四场景均值=`64.1615%`
- 矩阵：C0只评估；训练C1、C2、C3、S0、S1、S2、S3、S4、M1、M2、M3、M4、M5、M6，共14个E200 final-only行
- GPU：wave1的8行分别使用GPU0-7；wave2的6行分别使用GPU0-5；每张卡本矩阵同时只启动1行
- 启动命令：`RUN_ID=phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v2 ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release-root> bash code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v2`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v2`
- 预期artifact：14个`final.pth`、14份FCR诊断、15行四场景prediction、独立truth sidecar、15份独立score
- checkpoint选择：禁止使用target筛选，固定每行E200最后一个epoch的`final.pth`
- 评价边界：训练全部完成后才统一准备target输入；prediction先写出，独立scorer后连接truth；第二波发布不读取target结果
- 技术停止规则：仅在协议/query泄漏、错误seed/receiver/day/scenario、输出冲突、错误checkout、确定性重复异常、无prediction闭合、scorer连接错误或进程归属不清时停止精确run进程树；低性能不停止
- 机制解释边界：能力门控不能中止矩阵。M4/M6只有在真实pair、非零loss和非零梯度诊断同时成立时才能解释为机制已激活，否则报告`MECHANISM_NOT_ACTIVATED`
- v1修复：恢复基线`no_dac/no_stats`架构锁；旧`id_backbone.*`同名同shape加载仍强制完整，seed/epoch/candidate检查不弱化；FCR-V2新增模块保留设计初始化
- 本地验证：独立运行`phase1_fcr`175项通过；修复前回归精确复现missing=42，修复后聚焦46项通过
- release映射：本地`release_archives/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v2.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v2.tar.gz`

## 运行结论

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 启动PID：`848020`；初始PID/CWD/cmdline/GPU/log增长绑定通过。
- 故障范围：C1已完成第1个epoch并继续运行；C2、C3、S0、S1、S2、S3、S4均在首个训练步触发同一确定性异常，wave2未发布。
- 故障指纹：`route_formal_identity_outputs -> select_identity_logits -> ValueError: formal FCR identity output has an incompatible feature schema`。
- 处置：触发“至少两行出现同一确定性预prediction异常”的预登记停止规则。复核进程组`848018`内成员均绑定本run后，仅向该进程组发送`SIGTERM`；随后按run ID独立查询无残留进程。未触碰其他任务。
- artifact：V2全部日志及部分训练产物原位保留；没有形成完整prediction，因此不得评分，也没有性能结果。
- 下一步：本地复现各ablation row输出schema差异，加入定点回归并最小修复；通过聚焦验证后使用新提交、新release和新run ID发布V3，禁止复用V2输出根。
