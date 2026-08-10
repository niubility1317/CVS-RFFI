# Phase1 P1-RECTE后冻结42步实验报告

## 1.状态与目标

- 实验ID：phase1_recte_postfreeze_20260810_v1
- 日期：2026-08-10
- 当前状态：LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT
- 操作边界：主代理冻结评价合同、矩阵和判定门；唯一N607 Runner只负责release落地、唯一启动、技术监控和小工件回收。
- 训练输入：phase1_recte12_20260810_v1，已技术闭合12/12臂；训练报告SHA256=013216a56da310ea4ae0b082904719255099fcc54985724870bdb8fc4c4a85bf，Git镜像commit=cd8daa75。
- 目标：对同fold C/G执行固定clean、三LEO、fixed400 proxy和连续Gaussian-NLL公平评价，产出6份pair JSON及F6矩阵聚合。
- 声明边界：技术完成不等于性能通过；任一非补偿门失败即REJECT_P1_RECTE_PERMANENT，全部通过也只能PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW。

## 2.冻结版本、本地文件与独立审查

- Git仓库：E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt
- 后冻结实现commit：b95aac57b82f623f729c2ac24c1793664c112ca1
- 独立actual-diff终裁：P0=0 / P1=0 / ALLOW
- 审查边界：ALLOW只允许技术发布与Runner交接，不包含性能结果、方法晋级或N607已执行声明。

|文件|工作树SHA256|用途|
|---|---|---|
|analysis/phase1_recte_postfreeze_design_20260810.md|47c24a0e0cf45082e7849fd39bbad9f4d3804a87d8f1caea1d6a22f2076c1865|后冻结设计、追踪和证据边界|
|code/export_phase1_recte_features.py|75eb27705fa479b923c8322cc791c9d3723dd47197edde87ae20a35a10e04d58|clean L/V/proxy专用导出|
|code/export_phase1_recte_leo_features.py|37b248a0c80e05538d941ee8e51df8f3aaed6d49ecefeb55cc85d0aa8d6359bb|三LEO导出与物理绑定|
|code/evaluate_phase1_recte_postfreeze_pair.py|69a4cd941094196c99e8c78d10c9bbca9a43742b46a00ffc9e2463c725ba7364|同fold C/G评分与F6原始工件聚合|
|code/tests/test_phase1_recte_postfreeze.py|fb4ad2aa28ee5b8f6873ff93f5df5c49a8bbc748d95ac37ccbb6aa740f8dd01d|receipt、物理绑定、篡改与门测试|
|code/scripts/launch_phase1_recte_postfreeze_20260810.sh|4aa54708540be5e4eeae5d42165e9ef1ca35ad4218981a7c33c49a2f4f8a1abe|冻结42步launcher，Git mode100755|

本地ssr-gpu串行验证：

- 四个Python文件py_compile通过。
- RECTE后冻结专项：33 passed。
- RCAT+RCRMD共享公平核回归：60 passed。
- bash -n通过。
- dry-run精确42步=12 clean+12 LEO/binding+12 proxy+6 pair，旧ICMT/RCAT/RCRMD/CAGM运行身份为0。
- git diff --check通过。
- 定向负测覆盖旧身份注入、source/LEO/proxy篡改、1-row proxy、F6 summary/raw tamper、非有限值和非补偿拒绝。

## 3.冻结训练原件

ManySig预期SHA256=2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f。训练root固定为/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte12_20260810_v1。

|候选|final checkpoint SHA256|
|---|---|
|F1C_RECTE12|4b8c24733196644b0cf0839efc7a165ba79a268086eb1457920633ae9e2afd49|
|F1G_RECTE12|fe0688d39d42b1746b3f2137e58764167395f09e14fd0bb397f8e4bc8b8e3d3a|
|F2C_RECTE12|5cc84ad5984a4c45627228734caff0f6b8c9e0cb36830bf93bcbd33122c3b1bc|
|F2G_RECTE12|d9971a30d7718fe75effb43cad0cc0995c660eaf0dc6e822cdbb9a6641d12a92|
|F3C_RECTE12|1b143700a15e378c6961f4a232d2a25151caa448c3fb3300ae29e2555668577c|
|F3G_RECTE12|53dc14892610ca9908170ab2d939a32beba7b3a2a3408f3fe4086b0b9b14c4d9|
|F4C_RECTE12|e8940f6b1366b6ebfe7f00f2f0b5a6fb1c5d8dff41a7fdfdd7b1a05e2fd25ac7|
|F4G_RECTE12|117f58eca2277c9dc212e1dbe8ffd07c879d9411fa4eed9080eb3f0e783d7172|
|F5C_RECTE12|5e4df2aea0065e08a401826dc977094349a16958d3aaa5021c4c496efd67f98a|
|F5G_RECTE12|fcb5406a56b23a65627bf556cc00f31371ff931a927aa78d3a06bed533f0d4a9|
|F6C_RECTE12|9055408c27b034cf02b4a67b6ddaae1d9bf7dea36c27b6289fe99e6d946478c8|
|F6G_RECTE12|728bea80024a758fe473ace53196233448553165091571b7d236140b51428416|

Runner必须逐项重验ManySig、12个checkpoint SHA和RECTE terminal receipt，不下载checkpoint。

## 4.冻结评价合同与非补偿门

- clean只以source-L的feat_joint拟合float64 totalized-L2对角Gaussian；精确zero映射0并保留，nonfinite fatal；V/proxy零fit。
- 每类方差使用ddof=1，按0.9×class+0.1×class-equal pooled收缩，方差下限1e-6；评分使用完整Gaussian-NLL和稳定logsumexp连续u。
- fixed proxy：days=2021_03_01,2021_03_08；RX=1-1,1-19,14-7,18-2,19-2,2-1；selection seed=7281148；max/TX=400；total=400。
- LEO绑定封存ManySig path/SHA、selection、physical ID及每scene TX/RX/day完整覆盖。
- pair重开C/G训练receipt并验证Rs0..6、28格、fixed378、lambda、functional equality、三scene positive-tail/VJP和共同训练投影。
- F6必须重开F1--F5的clean NPZ、LEO NPZ、binding、proxy JSON/CSV与当前checkpoint，在当前SHA下重算；不得信任prior自报summary。
- clean要求6/6折overall、min-class、min-RX、min-day均不低于C−2pp。
- LEO要求18/18场景格四floor均不低于C−2pp。
- 每fold三场景等权overall增量及全18格等权overall增量均不得为负。
- proxy要求每foldΔAUROC>0且Δ(mean u_proxy−mean u_V)>0，必须6/6。
- clean、LEO、四floor、fold/global overall与proxy双门互不补偿。

## 5.N607路径、资源与唯一命令

- Python：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
- 项目根：/home/szu2070436088/2510044040/CV-SincNet
- 预计release：/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_b95aac57
- 精确CWD：<release>/code
- 训练root：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte12_20260810_v1
- 后冻结root：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte_postfreeze_20260810_v1
- log root：/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1
- outer：/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1_launcher.out

GPU映射沿用训练：F1C+F5G/0，F1G+F5C/1，F2C+F6G/2，F2G+F6C/3，F3C/4，F3G/5，F4C/6，F4G/7；pair按F1至F6 CPU串行。启动前只读记录现有GPU任务，不干预无关进程；每GPU总实验数不得超过2。

唯一启动命令：

    cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_b95aac57/code && nohup env POSTFREEZE_RUN_ID=phase1_recte_postfreeze_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_b95aac57/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte12_20260810_v1 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte_postfreeze_20260810_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_b95aac57/code/scripts/launch_phase1_recte_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1_launcher.out 2>&1 < /dev/null &

调用只能1次，retry=NO。调用端超时后先清理本地SSH/TCP22，再只读核run/log/PID/CWD/cmdline是否已landed，严禁重发。

## 6.技术健康、停止规则与预期工件

启动前必须通过direct preflight、archive/member/mode、release静态py_compile/help/bash-n/dry-run42、ManySig/12 checkpoint SHA、run/log/outer ABSENT和GPU资源核验。完整Git archive必须无prefix、LF-only、code/code=0；不得直接复制Windows mixed-EOL工作树。

预期工件：12 clean NPZ、12 LEO NPZ、12 LEO binding、12 proxy JSON、12 proxy CSV、6 pair JSON、18阶段日志、PID表和outer。技术门检查schema、root、matrix、training-root、checkpoint、RECTE receipt、common/proxy/physical binding及F6 aggregate。

仅在错误checkout/hash、覆盖风险、协议/P0违反、launcher-wide确定性故障，或至少两个distinct candidate在产出目标工件前出现相同确定性异常时停止。停止前精确核run-owned PID/CWD/cmdline，只停止本run并保留partial。不得读取accuracy、floor、AUROC、u-gap等性能字段决定是否停止。

Runner只回收小JSON/CSV/binding/log/PID/manifest，不下载checkpoint或NPZ；技术交接先标记PAIR_JSON_READY / NO_PERFORMANCE_INTERPRETATION。主代理在6/6 pair和原始工件闭合后才读取性能并作最终判定。

## 7.预注册交接状态

- 本地实现：已提交。
- 本地验证：已完成。
- 独立P0/P1：已通过。
- 根报告与Git镜像：本次预注册后应逐字一致。
- N607 release/42步：尚未执行。
- 性能分析：尚未开始。
