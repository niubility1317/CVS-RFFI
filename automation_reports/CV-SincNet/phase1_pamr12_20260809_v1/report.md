# P1-PAMR六折C/G完整训练报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LAUNCHED`

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
