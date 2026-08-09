# phase1_cp_sfce12_20260809_v2实验报告

## 1. 预注册

- 状态：`LOCAL_VERIFIED / READY_FOR_N607_RELEASE / NO_PERFORMANCE_RESULT`
- 日期：2026-08-09
- 负责人：`/root`；唯一N607 runner：`/root/n607_geosat_lite_runner`
- 目标：在不改变CP-SFCE科学机制、损失、数据、采样和12臂矩阵的前提下，修复v1的C臂遥测初始化和G臂AMP缩放伪溢出判别，完成40轮正式C/G配对训练。
- v1边界：`phase1_cp_sfce12_20260809_v1`已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得恢复、覆盖或重标性能结果。
- 来源：三轮回顾后唯一允许候选；不调`lambda=0.10`、`gamma=1`、场景、采样、数据、optimizer或矩阵。
- 科学边界：source-only；无RX/day/domain条件、teacher、表示对齐、新head、拒识阈值、proxy/held训练或选参。

## 2. 冻结机制与v2修复

C保持GeoSat-C共同base续训。G保留CB-SFCE损失，仅在共享identity encoder与精确classifier head上处理新增梯度：若`dot(a,b)<0`且`||b||²>0`，则`a'=a-dot(a,b)/||b||²*b`；否则`a'=a`。其中`b`是共同base梯度，`a`是`0.10*L_SFCE`梯度。

v2只增加执行合同：

- C每batch初始化CP遥测为空字典，终态固定`CONTROL_ARM_NOT_APPLICABLE`。
- base scaled/unscale梯度非有限时，只追加一次all-trainable raw base VJP；raw非有限或S内断图立即fail-closed，raw全有限才允许GradScaler跳过该batch并降低scale。
- aux scaled VJP非有限时，只追加一次raw aux VJP；raw非有限、S内断图或scope外非零立即fail-closed，raw全有限才整batch跳过并按公开backoff降低scale。
- skip批次不投影、不重试、不更新optimizer、EMA或prototype，也不计入applied；没有连续overflow阈值。
- 终态强制`attempted=applied+base_skips+aux_skips`，projection/outside-audit/optimizer-state-step均等于`applied>0`；raw故障与no-step为0；每次skip均需scale下降且optimizer state不变；local4×3每格必须有applied证据，encoder/head冲突只按applied累计且各至少一次。

## 3. 本地版本与验证

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`6bfaa335aea87e69c3dc0b6dc8b85048161383d9`
- 独立实际diff复核：`P0=0，P1=0，ALLOW`

|文件|SHA256|
|---|---|
|`code/SSDG/train_ssdg.py`|`77f72372ffe73fe5120d730078909d8c89bd6b416e93028b68f640f8ae8b703b`|
|`code/cvsrffi/phase1_cp_sfce.py`|`77e0a8bba146b049e2315a4ee05525a82be7dbfcd8516c845b6b412ac764dcda`|
|`code/tests/test_phase1_cp_sfce.py`|`1100009a9ffd0d509af03ab048730863170e8ca0a06ddccc1074c5d613be1278`|
|`code/scripts/launch_phase1_cp_sfce12_20260809.sh`|`b622c32949ffa5ff328ca98356b45d503d0faa310d66966fbc5fa9232aeb0f61`|
|`analysis/phase1_cp_sfce_design_20260809.md`|`d457dc75ce4158286929a2fccfc7189d269814d8402c35ce33e32a652fcc1be3`|

验证：`py_compile`通过；CP+CB focused共19项通过；真实CUDA GradScaler base/aux伪溢出均从65536降至32768且optimizer state不推进；raw非有限、断图和scope外异常负测通过；真实`lite_d`无query冒烟通过；`bash -n`、dry-run=12和`git diff --check`通过。

## 4. N607发布

- run ID：`phase1_cp_sfce12_20260809_v2`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce12_20260809_v2_6bfaa335`
- run/log：`/home/szu2070436088/2510044040/CV-SincNet/{runs,logs}/phase1_cp_sfce12_20260809_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cp_sfce12_20260809_v2.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 基线：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1/F1C...F6C/final_ssdg.pth`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce12_20260809_v2_6bfaa335/code && nohup setsid env RUN_ID=phase1_cp_sfce12_20260809_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce12_20260809_v2_6bfaa335/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce12_20260809_v2_6bfaa335/code/scripts/launch_phase1_cp_sfce12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cp_sfce12_20260809_v2.launch.out 2>&1 < /dev/null & echo $!
```

GPU0=`F1C+F5G`，GPU1=`F1G+F5C`，GPU2=`F2C+F6G`，GPU3=`F2G+F6C`，GPU4=`F3C`，GPU5=`F3G`，GPU6=`F4C`，GPU7=`F4G`；每卡不超过2个。

## 5. 健康、产物与判据

- 12臂40E、final-only；不增加1E审计任务。
- 启动后核PID、CWD、cmdline、run-root、GPU映射、CONFIG/EPOCH和日志增长。
- 仅P0、执行异常、无进展或至少两个arm同一确定性异常停止；不按任何性能数值停止。
- 重点技术字段：C=`CONTROL_ARM_NOT_APPLICABLE`；G的attempted/applied/base_skip/aux_skip、scale下降、state不推进、12格applied和双scope conflict。
- expected：12套metrics/final checkpoint/config/terminal/resource/heldout receipt；G另有CP终态合同。
- `NON_PROMOTABLE_P0_DISABLED/exit8`是预期终态，不是技术失败。
- 训练技术闭环后才允许建立新的不可覆盖postfreeze run；复用冻结42步和同一非补偿门，不调参。
- fresh-run retry：`NO`。若本run系统性技术失败，封存partial并返回主控；不得修补远端或重启。

## 6. 运行回填

- 状态：`PENDING_N607`
