# ADV3B02 CORE90 CVS-HSID最小实验报告v2

状态：`LOCAL_VERIFIED / P0P1_REVIEW_PASS / N607_PREFLIGHT_PASS / RELEASE_PENDING`

## 1.预登记

- run ID：`phase1_advb02_hsid_minimal_s392002_20260823_v2`。
- 基线：`ADV3B02_CORE90_SOFT_E200`；checkpoint=`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 数据角色：Phase1 source-only `L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；mask、训练和选模均不得读取target/query。
- 矩阵：`S0_CORE90`、`R3_SPEC_PROTO`、`X0_HIER_PROTO`、`F0_HIER_FUSION`、`X2_RX_ROBUST`；seed=`392002`；训练行200epoch。
- LEO_WEAK：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；每行训练完成后必须保留clean及三个逐场景结果。
- GPU：GPU0/GPU1；02:55 CST预检时各有1个既有训练进程，本run每卡最多再增加1个，保持每卡不超过2个。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_hsid_minimal_s392002_20260823_v2/`。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_hsid_minimal_s392002_20260823_v2/`。
- 技术停止：仅协议/query泄漏、错误checkpoint/checkout/row、输出冲突、无prediction闭合、确定性重复异常、OOM/NaN或进程归属不清；低性能不停止。
- 预期artifact：P0统计与分层mask、首步真实checkpoint无query smoke、每行checkpoint/训练日志、clean与三种LEO结果、same-row `y/raw/spec/fused` prediction、margin/gate/RX/day/scenario/质量字段。
- 结果边界：prediction完整并由独立scorer连接truth前，不声明性能提升。

## 2.实现与验证

- 主实现提交：`445a966b2f53fadcc9a807c625a776d295e93590`。
- 双根定点修复提交：`45fd122d0d6d25d78fe2fb7b368eb64f486e5013`。
- 相关回归：68项通过；10个Python入口编译、launcher`bash -n`、五行dry-run和`git diff --check`通过。
- 真实checkpoint无query smoke：`VERIFIED`；`query_input_count=0`、`target_input_count=0`、Raw可训练参数0、HSID可训练参数14,570、Raw主输出零漂移、输出有限。
- 独立审查：初审0个P0、5个P1；五项修复后的唯一一次定点复审`5/5 PASS`，剩余P0/P1为无。
- v1失败边界：P0在训练前因源码根/项目根混用失败，无训练、prediction或性能结果；失败产物保留，不复用v1。

## 3.发布命令

- release源码根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_hsid_minimal_s392002_20260823_v2-<release-commit8>`。
- 项目数据根：`/home/szu2070436088/2510044040/CV-SincNet`。
- P0准备：`cd <release> && env ROOT=<release> PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet RUN_ID=phase1_advb02_hsid_minimal_s392002_20260823_v2 GPU_0=0 GPU_1=1 MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_advb02_hsid_20260823.sh --prepare-p0 --only=R3,X0,F0,X2`。
- 正式启动：`cd <release> && nohup env ROOT=<release> PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet RUN_ID=phase1_advb02_hsid_minimal_s392002_20260823_v2 GPU_0=0 GPU_1=1 MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_advb02_hsid_20260823.sh --only=S0,R3,X0,F0,X2 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_hsid_minimal_s392002_20260823_v2/driver.out 2>&1 < /dev/null &`。
- release传输只对一个Git归档做一次本地/远端SHA-256比较；不增加成员hash、seal或receipt。

## 4.额外gate处理

除项目八项白名单外不增加审核、seal、receipt或逐文件哈希；旧要求若形成额外gate，记录`REJECTED_EXTRA_GATE`并继续最小流程。
