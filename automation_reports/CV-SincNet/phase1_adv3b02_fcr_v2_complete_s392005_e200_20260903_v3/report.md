# FCR-V2完整矩阵V3预登记

- 状态：`LOCAL_VERIFIED`
- run ID：`phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3`
- 代码修复提交：`0fd1a02a652b79284e4a522f6d4080ea30d965ac`
- V2失败记录提交：`ac5b437e2302d107aafa5fdb6ba837cd370bb45b`
- 分支：`codex/adv3b02-fcr-r1r8-s392005-20260903`
- 环境：N607普通账户，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3`
- 数据：`Dataset_WigSig/ManySig.pkl`，equalized，split seed=`392005`；source receiver=`1,3,4,6,8`、day=`1,2,3`、TX=`0..5`、pool=`90000`，训练只消费`L_s=6300`和`U_s=56700`；target receiver=`0,2,5,7,9,10,11`、day=`0,1,2,3`
- 初始化checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`
- checkpoint身份：seed=`392005`，epoch=`200`，candidate=`ADV3B02_CORE90_SOFT_E200`；架构锁=`no_dac/no_stats`
- 固定基线：clean=`76.2268%`，LEO均值=`60.1397%`，四场景均值=`64.1615%`
- 矩阵：C0只评估；训练C1、C2、C3、S0、S1、S2、S3、S4、M1、M2、M3、M4、M5、M6，共14个E200 final-only行
- GPU：wave1的8行分别使用GPU0-7；wave2的6行分别使用GPU0-5；用户明确允许在现有任务上增加本矩阵
- 启动命令：`RUN_ID=phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3 ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release-root> bash code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3`
- 预期artifact：14个`final.pth`、14份FCR诊断、15行四场景prediction、独立truth sidecar、15份独立score
- checkpoint选择：禁止使用target筛选，固定每行E200最后一个epoch的`final.pth`
- 评价边界：训练全部完成后才统一准备target输入；prediction先写出，独立scorer后连接truth；第二波发布不读取target结果
- 技术停止规则：仅在协议/query泄漏、错误seed/receiver/day/scenario、输出冲突、错误checkout、确定性重复异常、无prediction闭合、scorer连接错误或进程归属不清时停止精确run进程树；低性能不停止
- 机制解释边界：能力门控不能中止矩阵。M4/M6只有在真实pair、非零loss和非零梯度诊断同时成立时才能解释为机制已激活，否则报告`MECHANISM_NOT_ACTIVATED`
- V1修复：恢复基线`no_dac/no_stats`架构锁，真实checkpoint完整加载。
- V2修复：formal identity无模型路由只接受已定义的V1/V2正式schema，未知schema继续拒绝；不改变模型、损失、数据或矩阵。
- 本地验证：修复者聚焦45项及完整`phase1_fcr`183项通过；主流程独立运行V2训练集成38项通过。
- Git/release：运行提交`bb9e60ba3416dfded81f4c61444133f6dbaaf1e9`已推送且远端OID一致；release归档本地→远端SHA256均为`5689442a1762488dca531b70b1844728bddb548a8b141933675b8ea6696f5610`，远端编译通过。
- 真实checkpoint无query smoke：`loaded=195`、`skipped=0`、`incompatible_source_mature_identity=0`，seed=`392005`、epoch=`200`、candidate=`ADV3B02_CORE90_SOFT_E200`，V2 schema与`(2,6)`正式logits路由通过。

## 启动状态

- 当前状态：`RUNNING`
- 提交shell PID：`858714`；正式launcher PID/PGID：`858715`；wave1训练PID：C1=`859501`、C2=`859511`、C3=`859520`、S0=`859533`、S1=`859547`、S2=`859561`、S3=`859570`、S4=`859583`。
- 初始绑定：8个训练进程的CWD/代码路径/run root/seed/E200/final-only/checkpoint均与预登记一致，GPU0-7各绑定1行；日志均已创建并增长。
- 初始健康：C1已连续完成多个epoch；C2、C3、S0-S4均已越过V2的schema崩溃点，无`Traceback/RuntimeError/ValueError/OOM/Killed`。各行在AMP初始动态缩放期出现3次有限unsafe-step跳过，进程CPU/GPU持续活跃；后续监控需确认转入有效optimizer step与epoch推进。
- 监控策略：每30分钟短连接只读检查进程归属、行数/epoch、日志增长、GPU和确定性故障指纹；状态无变化时保持安静，完成、失败或需要用户处置时通知。
