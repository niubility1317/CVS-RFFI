# P1-PAMR六折C/G完整训练报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`ARTIFACTS_COMPLETE / NON_PROMOTABLE_P0_DISABLED_EXPECTED / NO_PERFORMANCE_RESULT`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.目标与假设

实验ID：`phase1_pamr12_20260809_v1`。时间：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

P1-PAMR以同physical clean观测的detached raw-cosine分类margin为标量边界，仅恢复LEO观测相对正确类与最难异类的角margin；按TX等权，固定`lambda_pamr=0.05`。不做显式z对齐、RX/domain对齐、GRL/MMD/CORAL，不加EMA/外部teacher、新head或阈值。proxy、held与LEO评估行零fit、零校准、零选参。

技术审计`phase1_pamr_audit6_20260809_v2`已6/6通过：每折local4/head4/class-order绑定闭合，raw PAMR梯度nonzero=1/nonfinite=0，每TX anchor/hinge覆盖，全部性能评估跳过。正式40epoch路径不执行额外`autograd.grad`，只计coverage并按正常主loss反传。

## 2.冻结矩阵

|Fold|train TX|known-validation TX|proxy TX|C/G GPU|
|---|---|---|---|---|
|F1|20-15,20-19,6-15,8-20|14-7|14-10|0/1|
|F2|14-10,20-19,6-15,8-20|20-15|14-7|2/3|
|F3|14-10,14-7,6-15,8-20|20-19|20-15|4/5|
|F4|14-10,14-7,20-15,8-20|6-15|20-19|6/7|
|F5|14-10,14-7,20-15,20-19|8-20|6-15|1/0|
|F6|14-7,20-15,20-19,6-15|14-10|8-20|3/2|

GPU0=F1C+F5G，GPU1=F1G+F5C，GPU2=F2C+F6G，GPU3=F2G+F6C，GPU4=F3C，GPU5=F3G，GPU6=F4C，GPU7=F4G；每卡最多2任务。C/G同fold从同一GeoSat-C checkpoint做weights-only strict warm-start，新建optimizer/AMP/RNG。固定40epoch、seed、sampler、final-only；C为原GeoSat-C续训，G只增加PAMR。

## 3.版本与本地验证

implementation commit：`ee31aa3345f9e5e3251bacf7de17098377b67bc0`；技术审计成功报告commit：`e2e109cd`。独立复核：`P0=0 / P1=0 / ALLOW`。代码hash与验证详见审计v2报告；本地py_compile通过、focused pytest 43 passed、launcher `bash -n`通过、dry-run精确12行。

## 4.N607路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_20260809_v1_ee31aa33`
- run/log：`/home/szu2070436088/2510044040/CV-SincNet/{runs,logs}/phase1_pamr12_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_20260809_v1_ee31aa33/code && nohup setsid env RUN_ID=phase1_pamr12_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_20260809_v1_ee31aa33/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GEOSAT_CKPT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_20260809_v1_ee31aa33/code/scripts/launch_phase1_pamr12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.健康、artifact与晋级门

启动后核launcher/CWD/cmdline、12 child/GPU、日志增长、CONFIG/E001。仅P0/路径/hash/覆盖、OOM/CUDA、至少2任务同一确定性异常、PAMR绑定/coverage/total-loss异常或无进展触发停止；不因性能停止，retry=`NO`。成功要求12×E040、final/metrics/config/terminal/heldout/resource receipt；G每个source TX均valid anchor和active hinge>0，正式路径无额外梯度审计。

训练完整后另行运行冻结postfreeze，不在本run内选模。五项非补偿门：技术健康；clean known保护；18个fold×LEO场景overall与min-class/min-RX/min-day不退化；18格angular margin不退化；真实checkpoint/artifact闭环。proxy和paired cosine仅诊断，不能补偿。任一失败即`REJECT_P1_PAMR_PERMANENT`，不调lambda、不改sampler、不挑fold、不进入Phase3。

## 6.N607训练终态与artifact（2026-08-09）

训练已按冻结命令唯一启动一次。首次caller SSH在等待窗口超时；随后清理本地残留`ssh.exe` PID 10192，并以短连接只读确认落地，因此没有重复启动。远端release为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_20260809_v1_ee31aa33`，CWD为`<release>/code`；归档commit为`ee31aa3345f9e5e3251bacf7de17098377b67bc0`，无prefix归档SHA256=`043cd042ef8c3ce89538a812ee953dfee2a44841bea85a6d3bf1b9e00f5940b5`（261468160 bytes）。远端代码以git-archive LF字节核验，未做远端代码编辑；成员hash与CRLF/LF口径、输入ManySig hash均记录在manifest。

精确命令、Python、run/log/outer路径见§4；launcher PID=`4161623`，12 child PID/GPU映射为`F1C 4161626/GPU0; F5G 4161628/GPU0; F1G 4161630/GPU1; F5C 4161632/GPU1; F2C 4161636/GPU2; F6G 4161640/GPU2; F2G 4161642/GPU3; F6C 4161644/GPU3; F3C 4161652/GPU4; F3G 4161657/GPU5; F4C 4161661/GPU6; F4G 4161663/GPU7`。12臂均完成E040/40并产生final、metrics、PAMR config/terminal、training、resource、heldout receipts；child exit均为8，终态`NON_PROMOTABLE_P0_DISABLED`，这是当前Phase1 final gate的预期技术终态。launcher脚本聚合非零child后退出码为1；错误指纹计数为0；GPU计算进程、run-owned进程与SSH/TCP22均已清理。

|arm|PAMR|GPU|child exit|PAMR batches/rows|valid anchors|active hinges|gradient contract|
|---|---:|---:|---:|---:|---:|---:|---|
|F1C|off|0|8|0/0|0|0|CONTROL_ARM_NOT_APPLICABLE / pass|
|F1G|on|1|8|1200/153600|153200|105860|FORMAL_PER_TX_ANCHOR_HINGE_COVERAGE / pass|
|F2C|off|2|8|0/0|0|0|CONTROL_ARM_NOT_APPLICABLE / pass|
|F2G|on|3|8|1200/153600|152997|120007|FORMAL_PER_TX_ANCHOR_HINGE_COVERAGE / pass|
|F3C|off|4|8|0/0|0|0|CONTROL_ARM_NOT_APPLICABLE / pass|
|F3G|on|5|8|1200/153600|153128|121836|FORMAL_PER_TX_ANCHOR_HINGE_COVERAGE / pass|
|F4C|off|6|8|0/0|0|0|CONTROL_ARM_NOT_APPLICABLE / pass|
|F4G|on|7|8|1200/153600|153264|117546|FORMAL_PER_TX_ANCHOR_HINGE_COVERAGE / pass|
|F5C|off|1|8|0/0|0|0|CONTROL_ARM_NOT_APPLICABLE / pass|
|F5G|on|0|8|1200/153600|151385|133576|FORMAL_PER_TX_ANCHOR_HINGE_COVERAGE / pass|
|F6C|off|3|8|0/0|0|0|CONTROL_ARM_NOT_APPLICABLE / pass|
|F6G|on|2|8|1200/153600|151932|129345|FORMAL_PER_TX_ANCHOR_HINGE_COVERAGE / pass|

所有臂`class_count=local_data_class_count=live_head_class_count=4`且source role count=4；G臂每TX anchor/hinge均为正，正式路径`gradient_audit_mode=NOT_REQUIRED_FORMAL`且shared relation计数为0；C臂合同不适用且通过。artifact已回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_pamr12_20260809_v1\artifacts`：124个文件（123条小文件manifest条目+manifest），manifest SHA256=`b8d2c9b379439e47f4f17d263f015359cbe18a97d0d6ca82eac12f9b56bc97e0`，completion.tsv SHA256=`60dbc6394fcf9dbeb8f64066caff1b12188c30e792a459b29e7aca264751b9c7`；逐项bytes/SHA匹配，未下载checkpoint/NPZ。传输包SHA256=`85c742a8b90eb256e6a9085554a822b643497a114bafcb347bbb54cbabe8a360`；远端临时包已删除。本run未启动postfreeze，retry=`NO`，不作性能结论。
