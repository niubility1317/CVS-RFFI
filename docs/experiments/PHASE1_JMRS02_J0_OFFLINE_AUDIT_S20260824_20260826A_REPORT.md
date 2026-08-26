# PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A预登记与追踪报告

## 一、目标与边界

本轮只执行JMRS02的J0离线联合审计。输入为已闭合的JMRS01 428064条prediction和对应truth；不训练模型、不调用GPU、不改写旧run、不重新生成prediction。J0只能判断旧分支错误集合是否存在独有rescue与组合协同，不能证明角色重构后的RC-X、RC-Z、稳健谱残差或phase nuisance有效，也不能声明target receiver DG。

## 二、不可覆盖运行定义

- run ID：`PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A`
- 本地代码分支：`codex/phase1-jmrs02-j0-20260826`
- 实验代码commit：`ad2e756b803849315a77785e7d8a7b86462c92f6`
- 本地验证环境：`ssr-gpu`
- N607运行解释器：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；服务器无`ssr-gpu`，沿用原JMRS01已验证环境，不安装新环境
- release Git状态：`3f73806b5fbc51131f68910eb6424e210e2633d6`
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_jmrs02_j0_20260826/3f73806b5fbc51131f68910eb6424e210e2633d6`
- release归档：本地`E:\type10-7\release_archives\PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A_3f73806b.zip`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_jmrs02_j0_20260826/PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A_3f73806b.zip`
- release SHA-256：`8605265bdc83a762187212ad07b939de31fd9653d17cc35cb6321a4d0b5f34d5`
- 输入prediction：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A/predictions.jsonl`
- 输入truth：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A/truth.jsonl`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs02_j0_20260826/PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A`
- GPU：不使用
- bootstrap：按`receiver×day×scenario`分组，2000次，seed=20260826
- 系统技术停止：输入闭合失败、缺少预登记row/scenario、输出根已存在、非有限结果或无法产生全部J0 JSON时保留现场并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

精确运行命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/score_phase1_jmrs02_j0.py --predictions /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A/predictions.jsonl --truth /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A/truth.jsonl --output_dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs02_j0_20260826/PHASE1_JMRS02_J0_OFFLINE_AUDIT_S20260824_20260826A --bootstrap_resamples 2000 --seed 20260826
```

## 三、预登记组合与晋级规则

组合固定为`R1+D1`、`R1+P2`、`R2+D1`、`D1+P2`、`R1+D1+P2`。对每个组合计算oracle gain、相对最佳单机制的synergy、分组bootstrap 95%CI、rescue Jaccard、相对S1的独有rescue及成员独有rescue。

只有`synergy>0`且95%CI下界>0的组合可形成J0协同信号。J0通过只允许进入角色正确的J1单模块设计，不直接授权联合训练；若没有组合通过，则停止JMRS02联合路线，不启动J1/J2。

## 四、设计追踪

|ID|原报告要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|
|P0-1|`nondegraded`同时相对M0和family sham|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|S1明确标为历史共享容量控制，不伪装成family-specific sham|
|P0-2|breadth相对Core90 rescue|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|只统计M0错误且候选正确|
|P0-3|safe gate排除`alpha=0`虚假通过|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|零采用记为不可评价且不通过|
|P0-4|J0 pairwise/triple unique rescue|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|固定五个组合和分组bootstrap|
|P0-5|统一身份margin|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|使用fold-local Fisher ratio|
|P0-6|成本改名为incremental runtime|`code/cvsrffi/jmrs02_j0.py`|verified|`test_jmrs02_j0.py`|显式标注不含Core90|
|P0-7|核对day3唯一日期|协议artifact与报告|verified|`protocol_and_smoke.json`为`2021_03_23`|修复旧报告的`2021_03_22`|
|P1|角色正确的RZ0/RZ1/RX1/D1′/P0|未来J1|deferred|等待J0|不得先训练联合|
|P2|只训练有J0证据的组合|未来J1后|deferred|等待J0/J1|J0不直接授权联合|
|P3|正式target DG一次性确认|未来J2|deferred|等待J1|只允许1—2个冻结候选|

## 五、当前状态

聚焦测试：JMRS02-J0 6项通过；JMRS01 scorer回归3项通过。`py_compile`通过。一次P0/P1审查发现bootstrap逐样本展开会造成约5亿次索引操作，已定点改为`receiver×day×scenario`组计数预聚合并通过RED→GREEN复测。

N607 preflight：prediction/truth各428064行，大小约460MB/29MB；输出根不存在；可用内存493GiB，`/home`可用7.3TiB；没有既有JMRS02 scorer进程。J0为CPU离线审计，不占用GPU。

`LOCAL_VERIFIED / RELEASE_SYNC_PENDING / NO_EXPERIMENT_LAUNCHED`
