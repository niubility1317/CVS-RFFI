# FastTrust-RC4-QB伪标签质量预算E200实验报告

## 当前状态

`RUNNING / STARTUP_HEALTH_VERIFIED / NO_PERFORMANCE_RESULT`

- run_id：`phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1`
- 实际代码与配置提交：`de0b0a5d6cf58232aac99deb65efde4f41fce627`
- 分支：`work/cvs-active`
- 证据边界：当前只完成本地实现、测试、真实checkpoint无query smoke和独立审查；尚未发布至N607，尚无性能结果。

## 目标与固定协议

本轮重点是提高`U_s`伪标签的有效利用率，不把星地增强当作主体。固定`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`、seed392002、Core90初始化、U batch256、E200和相同训练步数；`U_s`真实TX标签不得用于训练、路由、校准或选模。保留现有Core90的`LEO_WEAK`拼接增强。

## 最小同row矩阵

|GPU|候选|身份路由|总有效身份预算|用途|
|---:|---|---|---:|---|
|0|`E200_QB0_NO_U_ID_SAFE`|无H/P/N|0|无U身份反事实|
|1|`E200_QB1_STRICT_H_SAFE`|严格H，N关闭|0.15|与候选同预算的严格H控制|
|2|`E200_QB2_H_PRESID_B15`|严格H＋V_cal安全P填剩余预算，N关闭|0.15|优化伪标签候选|

三行共同固定`rc4_lambda_domain=0.16`、class×receiver平衡、非有限批次技术保护和相同source-only校准。QB2相对QB1的唯一身份路由增量是P填充H未占用的预算。

## 稳定性修复与加速

- 统一detach批次遥测张量，避免非有限跳步时计算图保留至epoch末。
- RC4非有限保护：同一epoch累计至少8批且比例达到5%时，仅停止该故障行并保留产物。
- source-val重型星地/几何评估由每轮一次改为E1–E180每5轮一次、E181–E200逐轮一次，共56次；最终clean与三LEO评估不裁减。
- 训练和最终评估batch size为512；每GPU仅1个本矩阵训练进程。
- 保留AMP、TF32、pinned memory、persistent workers、prefetch和融合teacher/student视图；本轮不启用未经正式验证的`torch.compile`或fused AdamW。
- 预期矩阵墙钟8–10小时；目标相对旧P3约11.94小时缩短至少25%。该时间仅为资源允许时的工程估计。

## 本地验证

- 测试先行：聚焦RED先捕获缺失的质量预算、稳定性参数和launcher。
- 发布前回归：11个相关测试文件共143项通过；Python编译、两个launcher语法、JSON预算向量和diff检查均通过。
- 真实Core90无query smoke：checkpoint严格重建`missing=0/unexpected=0`；`query_input_count=0`、`target_truth_read_count=0`、`target_eval_count=0`；63组梯度有限，遥测均无计算图；技术覆盖分支有效权重`2.067908<=6.3`。
- 独立P0/P1审查首轮发现QB1预算未与QB2对齐；经定点RED→修复→GREEN后，预算向量固定为`[0.0,0.15,0.15]`，定点复审结论`P0=0、P1=0`。
- 真实smoke仅验证权限、重建、数值和路由可执行性，不是性能结果。

## 发布路径与命令

- 本地release归档：`E:/type10-7/release_artifacts/phase1_fasttrust_rc4_qb_de0b0a5d.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_fasttrust_rc4_qb_de0b0a5d.tar.gz`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_qb_de0b0a5d`
- CWD：上述release root
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 输入数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- Core90 checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1`
- launcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1/launcher.log`

预登记启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_qb_de0b0a5d
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_qb_de0b0a5d PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python CONTROL_PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ID=phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1 RESOURCE_SLOT_LIMIT=1 bash code/scripts/launch_phase1_adv3b02_fasttrust_rc4_qb_e200_20260823.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1/launcher.log 2>&1 < /dev/null &
```

## 技术停止规则与预期制品

仅因协议/query泄漏、错误seed/split/epoch、输出覆盖、错误checkout、无prediction闭合、进程归属不清，或预登记RC4非有限批次保护触发而停止对应任务；不得因中期准确率低停止。

每行训练必须保存`status.txt`、完整`train.log`、完整`metrics_epoch.jsonl`和`final_ssdg.pth`，并完成`metrics_clean.json`、`metrics_joint.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`、`metrics_leo_rain_weak.json`。启动后只做一次PID/CWD/cmdline/GPU/log增长核验。

## 科学判定

QB2相对QB1的预登记晋级条件：三LEO均值至少`+0.30pp`、receiver-cell floor至少`+0.30pp`、clean下降不超过`0.50pp`。低性能不属于技术失败；必须等待三行E200训练与clean＋三LEO评估完整闭合后再分析。

## 2026-08-23 20:14 CST发布与启动读回

### 状态

`RUNNING / STARTUP_HEALTH_VERIFIED / NO_PERFORMANCE_RESULT`

- 20:11:46 CST普通账号只读preflight通过；GPU0–7均无compute app，目标run/release/archive路径均不存在。
- 数据集大小2359341461字节；Core90 checkpoint大小8582116字节；远端PyTorch2.1.0＋cu121识别8张GPU。
- 唯一release归档大小35267753字节；本地/远端唯一SHA-256均为`b4395f6623d772b6a5d737d89414320fdc055c04473a836c3a8e0b722f05b5da`。
- release已落到`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_rc4_qb_de0b0a5d`；远端Python编译、两个launcher语法和预算向量`[0.0,0.15,0.15]`读回均为`VERIFIED`。
- dispatcher PID=`1707450`；CWD严格绑定上述release，cmdline为`bash code/scripts/launch_phase1_adv3b02_fasttrust_rc4_qb_e200_20260823.sh`。

### 一次启动健康核验

|GPU|候选|训练PID|显存/MiB|利用率|启动判定|
|---:|---|---:|---:|---:|---|
|0|`E200_QB0_NO_U_ID_SAFE`|1707492|1880|23%|RUNNING|
|1|`E200_QB1_STRICT_H_SAFE`|1707497|1878|38%|RUNNING|
|2|`E200_QB2_H_PRESID_B15`|1707500|1888|23%|RUNNING|

- 每张GPU只有1个本矩阵compute app；GPU3–7继续空闲，没有干预其他run。
- 15秒窗口内，三份candidate dispatcher日志已存在；三份`train.log`由尚未创建变为6935/6943/6935字节，证明训练子进程已产生新日志。
- 进程树、训练cmdline、`--run_id`、`--candidate_id`、`--epochs 200`、U batch256、预算0/0.15/0.15和`rc4_lambda_domain=0.16`均与预登记一致。
- 错误扫描未输出Traceback、CUDA OOM、RC4非有限保护、TRAIN_FAILED或segmentation fault。健康脚本最终退出码1来自`pipefail`下`grep`零匹配；其前置PID/CWD/cmdline/GPU/log读回均已完成，因此该退出码不代表远端训练失败。
- 当前没有完整epoch、final checkpoint或clean/三LEO结果，严禁把`RUNNING`称为实验性能完成。

### ETA

按加速设计和旧P3约11.94小时基线，预计三行矩阵训练加四场景评估总墙钟约8–10小时，即约在2026-08-24 04:15–06:15 CST闭合；受服务器瞬时I/O和评估耗时影响，该区间仅为工程估计。
