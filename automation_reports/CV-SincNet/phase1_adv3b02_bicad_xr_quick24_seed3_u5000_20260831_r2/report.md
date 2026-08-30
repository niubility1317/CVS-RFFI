# Phase1 ADV3B02-BiCAD-XR Quick24正式实验报告（r2）

## 预登记

- 状态：`LOCAL_VERIFIED`，待N607 release与启动。
- run ID：`phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r2`。
- Git代码提交：`a81845cb95e0c4d303c592a32082c4ef4fc998b6`。
- 新run理由：r1的24行在模型构造前因`sample_rate_hz=0`一致技术失败，无训练和性能结果；r1所有partial artifact保留且不重启。
- r2定点修复：from-scratch WiSig入口解析`sample_rate_hz=25e6`，正式命令显式传`--sample_rate_hz 25000000`；真实双骨干模型构造测试通过。
- 方法：`D0`、`D5`、`E1`、`ADV3B02-BiCAD-XDC-V1`。
- 矩阵：4候选×fold1/fold8×seed392001/392002/392003，共24行。
- 训练：day1/2/3，5000 optimizer updates，200epochs上限，source-only，`concat_sat_ce_only+leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- fold1 source receivers：`3,4,6,8`；fold8 source receivers：`1,3,4,6`。
- 调度：GPU0–7，每卡3个本run训练进程；启动前若出现无关占用，则不得让总训练进程数超过该卡实时安全容量，也不得影响无关进程。
- 本地release归档：`E:\type10-7\local_artifacts\phase1_bicad_xr_quick24_20260831_r2.tar.gz`。
- N607 release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_bicad_xr_quick24_20260831_r2`。
- N607 run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r2`。
- N607 dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r2.dispatcher.log`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，仅Phase1 source数据。
- 启动命令：`/bin/bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_bicad_xr_quick24_20260831_r2/code/scripts/launch_phase1_bicad_xr_quick24_n607_20260831.sh`。
- 启动CWD：release根；环境：普通账户`szu2070436088`，禁止管理员账户。
- 预期每行artifact：`bicad_xr_final.pth`、checkpoint runtime、训练日志、clean JSON、三种LEO_WEAK JSON、严格重建结果和`ARTIFACTS_COMPLETE.json`。
- 允许停止：数据越权、错误candidate/fold/receiver/day/seed/update、输出冲突、错误release/CWD、命令无法运行、同一确定性pre-prediction异常至少重复2行、无final checkpoint/四场景闭合或进程归属不清。
- 禁止停止：中间或最终性能低、单seed差、缺少额外形式化receipt/hash/seal或报告字段。
- Phase2、target receiver、support、query、truth：全部禁止访问。

## 本地验证

- BiCAD-XR测试：241项通过；3条既有AMP弃用警告。
- 相邻HCF-DG/ADV3B03回归：159项通过。
- 真实ADV3B02历史checkpoint无query smoke：`PASS`。
- 真实from-scratch双骨干模型构造：`PASS`，`sample_rate_hz=25000000`。
- 独立P0/P1审查与原问题定点复审：`CLEAN`。
- 24行干跑：组合唯一，GPU0–7各3行，train days均为day1/2/3，updates均为5000。

## 发布与运行记录

- release SHA256：本地与N607均为`f8d7a98750944d8983f88113c9909e599e427d58e39f90466710d5c285dc7329`。
- 远端编译：通过。
- 远端真实checkpoint无query smoke：`PASS`。
- 远端环境未安装pytest；未安装任何包。from-scratch模型构造修复由本地`ssr-gpu`真实双骨干构造测试覆盖。
- 启动前GPU状态：GPU0–7均无compute process，显存占用均为1MiB；r2 run根不存在。
- dispatcher PID：`2518098`，PPID1，CWD严格绑定r2 release根。
- 启动后直属主训练进程：24个，全部命令绑定r2 release、r2 row根、正确candidate/fold/seed/day1/2/3、`sample_rate_hz=25000000`和source-only参数。
- GPU0–7均开始计算，首次读回利用率为`68/19/32/7/70/30/87/2%`，显存为`1934/1346/1308/1414/1874/1702/2152/1220MiB`。
- 24个`train.log`已创建；首次健康读回时仍为0字节，训练进程和GPU计算持续，不能据此停止。
- 启动后约64秒延迟复核：24/24直属训练进程持续存活；GPU0–7利用率为`99/95/96/97/91/99/99/96%`，显存为`2950/2974/3050/2966/3102/2928/2976/3222MiB`；`TECHNICAL_FAILURE=0`、`ARTIFACTS_COMPLETE=0`。
- 低频监控：Codex heartbeat `bicad-xr-quick24-r2`，每小时短连接只读检查；完成后执行全接收机day1–4严格零适配测试并发布报告。
- 当前状态：`RUNNING`。
- 低性能属于科学结果，不属于技术失败。


## 2026-08-31终态与技术失败分析

- 最终状态：部分闭合，不能标记为`ARTIFACTS_COMPLETE`或性能完成。
- dispatcher PID`2518098`及24个worker均已退出；GPU0–7空闲，无本run残留进程。
- D0：6/6行达到5000 updates，final checkpoint、严格重建以及clean和三种LEO_WEAK评估完整，行级`ARTIFACTS_COMPLETE.json`已保留。
- D5：6/6行达到5000 updates并保存final checkpoint，但最终闭合器将多元素factorized-head张量调用`.item()`，统一报`RuntimeError: a Tensor with 3840 elements cannot be converted to Scalar`；6行均保存`TECHNICAL_FAILURE.json`，未覆盖或删除。
- E1与`ADV3B02-BiCAD-XDC-V1`：12/12行达到5000 updates；`train_loss`有限，但`xdc_donor_query_matrix`包含合法未求值cell的`NaN`占位符，严格JSON遥测写入统一报`ValueError: Out of range float values are not JSON compliant`。这些行未写final checkpoint，均保存日志、CSV、JSONL partial artifact和`TECHNICAL_FAILURE.json`。
- 科学结论边界：r2仅D0闭合，不能进行四候选比较、不能冻结候选或seed，也不能进入全接收机day1–4目标测试。
- 根因修复提交：`0284b80288418ba6eb342a42741bc66e9e6a08de`。合法XDC矩阵占位符转为JSON`null`；真实非有限`train_loss`仍显式技术失败；多元素训练头张量转为嵌套列表。
- 回归：BiCAD-XR245项通过，相邻HCF-DG/ADV3B03共159项通过，定点模块编译通过。
- 后续：r2保持不可变；仅使用新release和新run ID`phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r3`重新运行完整24行，保持相同候选、fold、seed、训练预算和每GPU 3任务。
