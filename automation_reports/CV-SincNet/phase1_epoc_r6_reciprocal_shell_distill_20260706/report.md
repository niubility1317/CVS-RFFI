# Phase1 EPOC R6 Reciprocal Shell Distill

|字段|值|
|---|---|
|实验ID|`phase1_epoc_r6_reciprocal_shell_distill_20260706`|
|记录时间|2026-07-06 02:35 CST|
|目标|在OSPR-CI++复评继续失败后，转入底层source-only特征分离/蒸馏；用`ADV3B02_CORE90_SOFT_E200`教师约束旧类，同时用源侧proxy/virtual unknown和reciprocal shell损失把星地信道扰动后的未知样式推离已知类特征邻域|
|协议边界|Phase1/source-only地面训练修复，不是Stage2-C成功；真实`Y_unknown`、ManyTx target-new/target-unknown、目标接收机样本均不进入训练或阈值拟合|
|比较基线|R4 teacher tail quarantine、R5 proxy accept crush、OSPR-CI++ retry1负证据|
|后续验收|训练完成后必须重新运行真实Stage2-C qknn8/OSPR-CI或等价M=1..5复评；同一行满足旧类/seen-new/unknown联合目标后才可讨论成功|

## 设计动机

OSPR-CI++复评显示，当前EPOC B特征包上决策层协同可以提高未知拒识，但会把旧类和seen-new压到不可用水平。R6不继续调阈值，而是在地面source-only训练阶段引入更强的ADV3B02教师锁定、紧类内半径、低密度排斥、source proxy virtual unknown和reciprocal shell约束。该路线仍然不能接触真实未知类；真实`Y_unknown`只允许后续Stage2-C评估使用。

## 本地文件与hash

|文件|目的|SHA256|
|---|---|---|
|`code/scripts/launch_phase1_epoc_r6_reciprocal_shell_distill_20260706.sh`|R6两候选启动器，默认GPU0/GPU1，source-only，ADV3B02教师，reciprocal shell/低密度排斥|`771AA65E6B44A9EAD32EEB3F7514491B5CEE91D273CB9363AE1B07023CDFA21B`|
|`code/tests/test_phase1_epoc_r6_reciprocal_shell_distill_launcher.py`|TDD验证R6启动器声明协议边界、禁止ManyTx/真实unknown训练、使用GPU0/1|`168530E95D392A54880478B89D32BA511FE279B16DB4EC92D426BF8C1F3AA558`|
|`code/evaluation/collaborative_open_set_qknn_eval.py`|补充OSPR-CI++机器可读非部署/非成功审计字段|`31615A1076CEEFA07C3EDC1B153E766BB24D87EEFEA28E36E27E9FC3FCBA3F5B`|

## 本地验证

|命令|结果|
|---|---|
|`ssr-gpu pytest code\tests\test_phase1_epoc_r6_reciprocal_shell_distill_launcher.py -q`|RED：启动器缺失时2 failed；GREEN：实现后`2 passed`，仅`.pytest_cache`权限warning|
|`bash -n code/scripts/launch_phase1_epoc_r6_reciprocal_shell_distill_20260706.sh`|PASS|
|`ssr-gpu py_compile code\tests\test_phase1_epoc_r6_reciprocal_shell_distill_launcher.py`|PASS|
|`ssr-gpu pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase1_epoc_r6_reciprocal_shell_distill_launcher.py -q`|PASS，`124 passed`，仅`.pytest_cache`权限warning|

## 候选设置

|候选|GPU|目的|关键差异|
|---|---:|---|---|
|`EPOC_R6_RECIPROCAL_SHELL_KD`|0|强负原型壳层排斥|更大proxy/OW权重、`proxy_virtual_count=128`、`radius_inter_ratio_target=0.18`、`bridge_accept_target=0.00`|
|`EPOC_R6_KNOWN_FLOOR_SHELL_KD`|1|更强旧类保持的壳层蒸馏|更强teacher clean KL与z_id MSE、较温和proxy权重、目标是避免旧类floor下降|

## 计划N607动作

1. 运行N607 preflight并记录GPU0/GPU1占用。
2. 同步启动器、测试、报告和`code/SYNC_MANIFEST.txt`。
3. 远端执行hash、`bash -n`和dry-run；若通过且GPU0/GPU1仍低占用，启动两个R6候选。
4. 启动后做短时健康检查：进程存在、日志出现`CONFIG-LOSS`/`CONFIG-TEACHER`/`CONFIG-SAT`/`EPOCH-BEGIN`、无`Traceback`/`RuntimeError`/`OOM`/`unrecognized arguments`。

## 风险

|风险|处理|
|---|---|
|继续强拒识但旧类floor受损|R6保留`joint_safe`、teacher KL、z_id MSE和PAIC guard；后续必须看Phase1 old/source验证和Stage2-C同row结果|
|source proxy unknown仍不能代表真实星地未知类|仅作为训练约束；真实ManyTx unknown仍只在后续评估中使用|
|M=5协同口径被误读|后续Stage2-C报告必须继续区分预算上限与严格同事件5 receiver协同|

## N607启动记录

启动时间：2026-07-06 02:34 CST。  
N607 preflight：PASS，GPU0/GPU1均`10MiB/24576MiB`，无计算进程；GPU2-7有既有训练进程，保持monitor-only。  
远端验证：hash一致，`py_compile` PASS，`bash -n` PASS，R6 dry-run PASS，远端直接运行`test_phase1_epoc_r6_reciprocal_shell_distill_launcher.py` PASS。  
启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && RUN_ID=phase1_epoc_r6_reciprocal_shell_distill_20260706 MAX_ACTIVE_PER_GPU=2 LAUNCH_SETTLE_SECONDS=10 bash code/scripts/launch_phase1_epoc_r6_reciprocal_shell_distill_20260706.sh
```

|候选|GPU|PID|日志|启动健康|
|---|---:|---:|---|---|
|`EPOC_R6_RECIPROCAL_SHELL_KD`|0|3136637|`logs/phase1_epoc_r6_reciprocal_shell_distill_20260706/EPOC_R6_RECIPROCAL_SHELL_KD.out`|运行中；约54秒时到E003；GPU显存约2224MiB；`JOINT-GUARD safe=1`；有限范围错误扫描无`Traceback`/`RuntimeError`/`CUDA out of memory`/`unrecognized arguments`/`Killed`|
|`EPOC_R6_KNOWN_FLOOR_SHELL_KD`|1|3137045|`logs/phase1_epoc_r6_reciprocal_shell_distill_20260706/EPOC_R6_KNOWN_FLOOR_SHELL_KD.out`|运行中；约44秒时到E002；GPU显存约2232MiB；`JOINT-GUARD safe=1`；有限范围错误扫描无`Traceback`/`RuntimeError`/`CUDA out of memory`/`unrecognized arguments`/`Killed`|

健康检查观察：两个候选均出现`EPOCH-BEGIN`、`LOSS-*`、`OW-FEAT`、`PROXY-UNK`、`ZID-FEATURE-SPACE`和checkpoint写入。当前proxy/OW/source episode在前几轮尚未激活，符合`start_epoch`设置；这不是拒识效果证据，后续需在proxy激活后监控`proxy_unknown_auc_proxy`、`virtual_accept_rate`、`soft_unknown_mixup_virtual_accept_rate`以及源验证旧类保持。

SSH清理：一次错误扫描命令因远端输出/管道行为超时，留下本地`ssh.exe` PID`42072`连接N607:22；已按规则关闭并复查，最终`ssh.exe processes: none`且`N607/bridge established connections: none`。

## 早期监控记录

监控时间：2026-07-06 02:48 CST。  
状态：两个R6候选仍在运行，metrics持续写入，最近日志未发现`Traceback`、`RuntimeError`、`CUDA out of memory`、`unrecognized arguments`或`Killed`。本轮只做monitor-only，不新增启动、不终止进程。

|候选|最新epoch|val_tx_acc|proxy_auc|virtual_accept|proxy_accept|bridge_accept|soft_virtual_accept|radius_inter_ratio|早期判断|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`EPOC_R6_RECIPROCAL_SHELL_KD`|25|98.51|0.3638|0.8160|0.0161|1.0000|1.0000|1.0143|旧类源验证稳定，但proxy unknown分离更弱；虚拟未知仍大量落入已知邻域|
|`EPOC_R6_KNOWN_FLOOR_SHELL_KD`|25|98.52|0.3811|0.8179|0.0167|1.0000|0.9997|0.9638|旧类源验证稳定，但未知拒识训练信号仍未改善|

解释：R6实现了“转入底层source-only蒸馏”的目标动作，但早期proxy信号仍为负趋势。当前不能声明R6已改善开集拒识；也不能用proxy单项拒识指标替代后续Stage2-C qknn8 M=1..5复评。下一次应在E35-E45附近再次检查`proxy_auc`、`virtual_accept`、`soft_virtual_accept`和源验证旧类保持；若仍无改善，应将R6降级为负证据，并设计更明确的feature-space上限诊断或target-old-only上限诊断。

## E35窗口监控与路线判定

监控时间：2026-07-06 02:59 CST。  
N607 preflight：PASS；本轮远端操作只读，不新增启动、不终止进程。GPU0/GPU1分别约`2529MiB/24576MiB`、`2435MiB/24576MiB`，R6两个候选仍在低显存运行。日志尾部错误扫描未发现`Traceback`、`RuntimeError`、`CUDA out of memory`、`unrecognized arguments`、`Killed`或`NaN`。SSH清理复查：本地`ssh.exe`为`none`，N607/bridge `ESTABLISHED`连接为`none`。

|候选|最新epoch|val_tx_acc|proxy_auc|virtual_accept|proxy_accept|bridge_accept|soft_virtual_accept|radius_inter_ratio|E35-E37判断|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`EPOC_R6_RECIPROCAL_SHELL_KD`|37|98.51|0.3655|0.8219|0.0111|1.0000|1.0000|0.9759|旧类source验证稳定，但proxy分离仍显著低于0.5；虚拟未知仍大量被已知邻域接受|
|`EPOC_R6_KNOWN_FLOOR_SHELL_KD`|37|98.52|0.3776|0.8171|0.0216|1.0000|0.9997|0.9557|旧类source验证稳定，但未知壳层/软未知约束没有形成可用开放边界|

路线判定：R6保留运行以获得完整训练轨迹，但不再作为主路线扩展同类超参。当前证据满足“`proxy_auc<0.55`且`virtual_accept>0.5`、`soft_virtual_accept≈1.0`”的降级条件，应把R6记为source-only底层蒸馏负趋势证据。下一步主线转入只读`feature-space upper-bound`诊断和`target-old-only`上限诊断：前者冻结ADV3B02/EPOC/R6特征，检查old、seen-new、unknown在`z_id`空间的半径、最近原型混淆、kNN margin、energy/radius分布；后者只用`R_t`内`Y_old` support/query评估目标域旧类上限。两者都不得使用真实`Y_unknown`训练、阈值拟合、早停或模型选择。

子agent审计结论：完成监督子agent确认当前不能标记`complete`，因为尚无同一候选达到旧类99%、旧类每类95%、seen-new 97%、seen-new每类93%、unknown reject 99%，且R6训练完成后仍需Stage2-C qknn8 M=1..全接收机复评；也不能标记`blocked`，因为仍有可执行的feature-space upper-bound和target-old-only诊断。方法review子agent指出OSPR-CI++已证明决策级强拒识不是当前主解，R3/R4/R5/R6的proxy趋势说明特征空间尚未形成可用开放边界，协同推理应保留为部署层低成本确认和冲突仲裁，不应继续承担主要表征修复。
