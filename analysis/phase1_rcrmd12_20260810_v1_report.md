# Phase1 P1-RCRMD 12臂训练实验报告

## 1.状态与目标

- 实验ID：`phase1_rcrmd12_20260810_v1`
- 日期：2026-08-10
- 操作角色：主代理冻结方法与矩阵；唯一N607 Runner负责落地、启动、监控和工件回收
- 当前状态：`LOCAL_VERIFIED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- 目标：以GeoSat-C final-only checkpoint为共同起点，比较严格同折的C控制臂与仅增加P1-RCRMD辅助损失的G实验臂。
- 假设：source-L中receiver×class等权的正clean→LEO margin-drop二阶矩，可能减少接收机条件退化。
- 声明边界：该方法不是分位数tail估计；不得声称已修复RX/day、proxy、真实unknown或Phase3。即使完整门通过也只能进入`pending-main`复核。

## 2.冻结版本与本地验证

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`d99eb9391cf5d0de33f17722bba04f9d0bb3fe57`
- 独立实际diff复审：初审唯一P1为float32批loss与Python cell账本的过严对账；最小修复后终裁`P0=0/P1=0`。
- 修改文件：

|文件|SHA256|用途|
|---|---|---|
|`analysis/phase1_rcrmd_design_20260810.md`|`0affa6fed13769f56369270574411528fce7579c0dea05d1cb1b443c313a1954`|冻结设计、权限和证据边界|
|`code/cvsrffi/phase1_rcrmd.py`|`3968eaa1228621d607be7f4cc6008afeff82cc88fdf983e6ff6ae424f2d7e64b`|固定1/28损失、收据、VJP和终态合同|
|`code/SSDG/train_ssdg.py`|`4e076f044cabc23ec66e5e6866d54f977837b79723838cf331aaafd8aa8fc5c6`|共同训练路径接入|
|`code/tests/test_phase1_rcrmd.py`|`a4753b5629cb0fc157e25214721932a77c489218b6fe7edecb8275f129f61996`|公式、权限、账本、lite_d和launcher测试|
|`code/scripts/launch_phase1_rcrmd12_20260810.sh`|`ee5e91f39737beabfe054aebbdbf8ad94832f6c32d0bc2a7d25e55c89a7200cb`|冻结12臂launcher，Git mode100755|

本地`ssr-gpu`串行验证：

- `py_compile`：通过。
- RCRMD+CAGM+ICMT+GD+CB+CP聚焦回归：`60 passed`。
- `bash -n`：通过。
- launcher dry-run：12臂，6C+6G，旧候选激活数0。
- `git diff --check`：通过。
- float32账本回归：`75.0000057220459`与cell账本`75.0`合法闭合；`+1`实质漂移仍fail-closed。

## 3.方法与冻结配置

[
m_i^v=ell_{i,y_i}^v-operatorname{logsumexp}_{k
e y_i}ell_{i,k}^v,quad
q_i=[operatorname{sg}(m_i^{clean})-m_i^{leo}]_+^2
]

对冻结source receiver集合`R_s={0,1,2,3,4,5,6}`与local4 TX类：

[
g_{rc}=0;(n_{rc}=0),quad g_{rc}=rac{1}{n_{rc}}sum_{iin I_{rc}}q_i;(n_{rc}>0)
]

[
L_{RCRMD}=rac{1}{4|R_s|}sum_{r,c}g_{rc}=rac{1}{28}sum_{r,c}g_{rc},quad
L_G=L_{base}+0.02L_{RCRMD}
]

- C/G共同：同一`training_final_only`warm-start、物理L样本和批顺序、seed/sampler、clean+单次LEO forward、三场景轮转、40 epochs、新AdamW/AMP初态和`L_base`。
- C：RCRMD关闭，`lambda=0`，仅封存共同coverage；aux/active/loss/VJP为N/A或0。
- G：RCRMD开启，`lambda=0.02`；只读source-L的TX标签与physical-ID绑定RX；不增加forward、样本、epoch、state或重采样。
- 非有限logit/margin/q/g/loss均fail-closed；q=0逐行合法；不删行、不加eps。
- 每场景28格，三场景终态84格；至少一个active q；首个active batch要求共享encoder和exact head raw VJP均finite/nonzero。
- 优化loss不变；账本对账冻结为`32×float32 eps×max(1,|batch|,|cell|)`。

共同超参：epochs=40、batch=128、lr=2e-4、weight_decay=1e-4、label_smoothing=0.01、seed=7281105、sat_seed=9281105、`lambda_sat_cons=0.10`。

## 4.冻结矩阵与GPU

|GPU|候选1|候选2|
|---:|---|---|
|0|F1C_RCRMD12|F5G_RCRMD12|
|1|F1G_RCRMD12|F5C_RCRMD12|
|2|F2C_RCRMD12|F6G_RCRMD12|
|3|F2G_RCRMD12|F6C_RCRMD12|
|4|F3C_RCRMD12|—|
|5|F3G_RCRMD12|—|
|6|F4C_RCRMD12|—|
|7|F4G_RCRMD12|SCB v4既有进程，合计不超过2个训练进程|

每fold C/G使用相同GeoSat-C checkpoint、source TX、known-validation TX和proxy TX。Runner不得改fold、receiver、TX、seed、λ、场景、训练长度或GPU映射。

## 5.N607冻结路径与命令

- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 预计release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd12_20260810_v1_d99eb939`
- 精确CWD：`<release>/code`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- GeoSat-C根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcrmd12_20260810_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd12_20260810_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd12_20260810_v1_launcher.out`

唯一启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd12_20260810_v1_d99eb939/code && nohup env RUN_ID=phase1_rcrmd12_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd12_20260810_v1_d99eb939/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd12_20260810_v1_d99eb939/code/scripts/launch_phase1_rcrmd12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd12_20260810_v1_launcher.out 2>&1 < /dev/null &
```

Runner只能调用一次；调用端超时先只读确认是否landed，严禁盲目重发。

## 6.预期工件与技术停止规则

每臂应生成：

- `final_ssdg.pth`
- `metrics_epoch.csv`和JSONL
- config、training completion、terminal、heldout、resource和RCRMD receipt
- C臂共同coverage合同通过且aux为N/A/0；G臂84格、active、VJP和终态合同通过
- 由于P0 promotion默认禁用，预期终态为`NON_PROMOTABLE_P0_DISABLED`/exit8；若工件和合同闭合，该退出语义不视为技术失败。

仅在下列情况停止本run：错误checkout/hash、覆盖风险、协议/P0违反、launcher-wide确定性故障，或至少两个distinct arm在产生终态工件前出现相同确定性异常指纹。停止前必须核CWD/cmdline/PID树，只停止本run并保留partial。不得按accuracy、loss或任何性能值早停。retry默认`NO`。

Runner完成后回收小工件与日志，不下载checkpoint/NPZ，不读性能；状态先到`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。

## 7.后冻结判定（训练完成后另行执行）

固定42步：12 clean export+12 LEO/binding+12 proxy+6 same-fold pair。Gaussian只用L拟合，V/proxy零fit；totalized-L2保留零向量，nonfinite fatal；proxy days/RX/seed/400和ManySig SHA/physical keys固定。

非补偿门：

- clean 6/6四floor不低于C−2pp；
- LEO 18/18四floor不低于C−2pp；
- 每fold三场景overall与全18格overall均不低于C；
- 每foldproxy AUROC增量>0且proxy−V的mean-u gap增量>0，必须6/6。

任一完整门失败即`REJECT_P1_RCRMD_PERMANENT`；均值不得补偿floor，分类端点与proxy端点不得互相补偿。
