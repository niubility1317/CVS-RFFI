# Phase1 CCPC-LEO六折C/G实验报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.实验目标与假设

实验ID：`phase1_ccpc_leo12_20260809_v1`。时间：2026-08-09。主Agent负责方法、矩阵和最终裁决；唯一N607 runner只负责落地、运行、监控和artifact回收。

前三轮表明失败根因是类条件几何在TX、RX和LEO扰动下漂移，而非缺少更精细的拒识阈值。本轮以每fold既有GeoSat-C checkpoint为共同warm-start，C/G均固定续训40epoch；G相对C只增加`P1-CCPC-LEO`：LEO anchor对detached clean bank执行class-conditional paired contrastive，同TX clean为正例、batch全部TX clean为分母。冻结`T=0.12、lambda=0.02`，不读取RX/domain标签，不使用GRL、MMD、CORAL、proxy/held训练、拒识head、阈值或扫参。

假设：直接优化source-known同physical clean↔单次LEO观测的类条件局部结构，并显式保留异类竞争，可改善跨RX和LEO floor，同时不损害clean known分类。

## 2.冻结矩阵与资源

| Fold | train TX | known-validation TX | proxy TX | C/G GPU |
|---|---|---|---|---|
| F1 | 20-15,20-19,6-15,8-20 | 14-7 | 14-10 | 0/1 |
| F2 | 14-10,20-19,6-15,8-20 | 20-15 | 14-7 | 2/3 |
| F3 | 14-10,14-7,6-15,8-20 | 20-19 | 20-15 | 4/5 |
| F4 | 14-10,14-7,20-15,8-20 | 6-15 | 20-19 | 6/7 |
| F5 | 14-10,14-7,20-15,20-19 | 8-20 | 6-15 | 1/0 |
| F6 | 14-7,20-15,20-19,6-15 | 14-10 | 8-20 | 3/2 |

实际并发映射为GPU0:F1C+F5G、GPU1:F1G+F5C、GPU2:F2C+F6G、GPU3:F2G+F6C、GPU4:F3C、GPU5:F3G、GPU6:F4C、GPU7:F4G；每卡一至两个任务。所有任务使用seed `7281105`、sat seed `9281105`、batch128、AdamW新状态、40epoch、`final_only`。warm-start只严格加载模型键，不恢复optimizer、AMP scaler或RNG状态；C/G同fold使用同一checkpoint和相同新优化器语义。

## 3.版本与本地验证

Git工作树：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`；release commit：`e999a6c526dc676dfa0ce193b00ce11cac3d308c`。

| 文件 | SHA256 |
|---|---|
| `code/SSDG/train_ssdg.py` | `324049158cf44d60ebbbdade77d0544c261104479c9f298d44643d6a6938c134` |
| `code/cvsrffi/phase1_ccpc_leo.py` | `7e7fa87abe080e2cb302cf2c975912a08509f897e1c32c84ac6209924ad9c8c9` |
| `code/tests/test_phase1_ccpc_leo.py` | `8ef19862e90651a30d4cfb323063382e6f5ade56d809853c3e4b605591eff8a2` |
| `code/scripts/launch_phase1_ccpc_leo12_20260809.sh` | `e4d39695f171e1449cddf090e91b47f30b28bc692c4a23eb89aac9c39bf4469e` |
| `analysis/phase1_ccpc_leo_design_20260809.md` | `b42b1e51cefef094479248a2637ef1fc520f08cc51af3ee598b35c902d375bee` |

`ssr-gpu`本地验证：py_compile通过；focused pytest 10/10；bash-n通过；dry-run精确12条，v1 checkpoint根12/12、G lambda 6/6、C lambda 6/6。独立代码复核最终为`APPROVE / Critical=0 / Important=0`；此前发现的默认checkpoint路径、非严格warm-start和teacher-loss漏禁用均已定点修复。

## 4.N607冻结路径与命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v1_e999a6c5`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v1.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- checkpoint根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v1_e999a6c5/code && nohup setsid env RUN_ID=phase1_ccpc_leo12_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v1_e999a6c5/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GEOSAT_CKPT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v1_e999a6c5/code/scripts/launch_phase1_ccpc_leo12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.预期artifact、健康门与性能门

每任务预期`final_ssdg.pth`、metrics CSV/JSONL、`phase1_ccpc_leo_config_receipt.json`、terminal/heldout/resource等receipt；log根预期`pids.tsv`和`completion.tsv`。启动后立即核对launcher PID、CWD/cmdline、12 child、GPU映射、日志增长、CONFIG-CCPC-LEO和首轮epoch；后续只用短连接监控。

技术停止只允许路径/hash/覆盖/协议错误、模型键不匹配、P0泄漏、非有限数、确定性异常、OOM、无进展或至少两个任务同一异常；不得读取中途性能停止。失败不重试、不远端改码，保留partial并标`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

训练完整后一次性做final-only clean、source proxy和三场景同physical LEO评估；proxy/held零fit、零校准、零选参。五项非补偿门：12任务健康；6/6 clean known overall/min-class/min-RX/min-day的G-C均≥−2pp；18/18 LEO原子格四项G-C均≥−2pp且总体有明确改善；source proxy连续排序相对C同向；真实checkpoint与deployment bundle闭环。任一门失败即`REJECT_CCPC_LEO_NO_RETRY`，不进入Phase3。

## 6.风险与完成后检查

最高技术风险是N607真实GeoSat-C checkpoint模型键或checkpoint metadata与当前代码不匹配；实现会在任何训练前fail-closed。最高科学风险是CCPC缩短clean↔LEO同类距离但同时压缩异类margin，或把收益集中到少数TX/RX。完成后必须同时检查同类距离、最近异类margin、clean floor、18格LEO floor和proxy方向，不能以均值补偿任何fold或原子格失败。
