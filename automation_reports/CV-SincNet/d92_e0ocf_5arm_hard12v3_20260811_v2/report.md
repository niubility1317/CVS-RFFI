# D92-E0OCF五臂Hard12-v3实验报告（v2一次性发布）

## 1.状态与目标

|字段|内容|
|---|---|
|run ID|`d92_e0ocf_5arm_hard12v3_20260811_v2`|
|日期|2026-08-11|
|当前状态|`LOCAL_VERIFIED / RELEASE_READY / NOT_LAUNCHED`|
|operator|Codex primary；N607唯一runner待交接|
|协议|`p2_min_v1`；复用`VALIDATED_ONCE`数据|
|证据范围|`DEVELOPMENT_ONLY_PSEUDO_BLIND_DISJOINT_STRESS_SCREEN`|
|目标|验证同after状态旧类contrast融合能否同时提高D92性能并降低注册计算量|
|唯一晋级候选|`E0_OCF25`|

本次以v2新run ID发布，因为v1只完成本地预注册，未同步、未启动、未产生性能artifact；v2冻结为实验专用one-shot runner，不覆盖任何旧路径。

## 2.冻结方法与比较

|arm|机制|角色|K5/K10 two-state fit|
|---|---|---|---:|
|`D92_FULL`|原D92：full/block LOO-soft fusion＋Fisher|原方法对照|48/88|
|`E0_FULL_ONLY`|关闭Fisher，仅after full主几何|低成本对照|2/2|
|`E0_FIXED50`|full/block全类固定0.5/0.5|因果对照|4/4|
|`E0_OCF25`|只向FULL_ONLY旧类contrast融入25% block contrast|唯一primary|4/4|
|`E0_OCF50`|同上，固定50%|diagnostic-only|4/4|

OCF只使用同一个`DA0_REG1` support上的full/block head；新类行byte-exact，旧类组均值守恒，不使用query truth/role/quota/fit/update/selection/global reassignment。K1/K2五臂严格别名为`D92_FULL`。旧类数冻结为6；OCF support MAC为K5`112350`、K10`216030`，query MAC和永久state与`E0_FULL_ONLY`相同。

## 3.冻结Hard12-v3矩阵

|outer|role|Hard|
|---|---|---:|
|`rx_20_1__seed_713104__k_5__new_20`|performance|0.629334677419|
|`rx_20_1__seed_713106__k_10__new_20`|performance|0.520866935484|
|`rx_20_1__seed_713106__k_1__new_20`|liveness|0.910584677419|
|`rx_3_19__seed_713102__k_10__new_5`|performance|0.429435483871|
|`rx_3_19__seed_713103__k_10__new_20`|performance|0.720463709677|
|`rx_3_19__seed_713105__k_10__new_5`|performance|0.454032258065|
|`rx_7_14__seed_713102__k_10__new_10`|performance|0.412600806452|
|`rx_7_14__seed_713105__k_1__new_20`|liveness|0.875403225806|
|`rx_7_7__seed_713104__k_10__new_10`|performance|0.297479838710|
|`rx_7_7__seed_713106__k_5__new_20`|performance|0.521471774194|
|`rx_8_8__seed_713103__k_10__new_20`|performance|0.456451612903|
|`rx_8_8__seed_713104__k_5__new_20`|performance|0.590826612903|

selection SHA=`20edc97b914c1031d9f917c63dee8e8ddb94223b31750151a05aabf0375d65f1`。矩阵为12outer×5arm=60job；每job固定`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，共180scene-arm；与Hard12-v1/v2零交集。真实checkpoint truth-free smoke固定为`rx_20_1__seed_713106__k_1__new_20`的`D92_FULL`。

## 4.本地版本与验证

|项目|结果|
|---|---|
|冻结代码commit|`7f2255ea0bb0f112ac17b83a78491f2f86b93549`|
|主代理新鲜聚焦回归|153/153通过，D92-BE/E0D/E0OCF相关集|
|静态验证|7个模块`py_compile`、配置身份、runner/analyzer CLI、`git diff --check`全部通过|
|独立核心复审|`APPROVE，P0=0，P1=0，P2=0`|
|独立Task2复审|`APPROVE，P0=0，P1=0，P2=0`|
|全分支复审|`APPROVE，P0=0，P1=0，P2=0`；独立12项通过|

## 5.交付物与N607映射

|本地文件|bytes|SHA256|远端目标|
|---|---:|---|---|
|`E:\type10-7\code\snapshots\d92_e0ocf_runtime_closure_c55f4241.tar.gz`|4978871|`c55f42412c3f8750457bd78bd56134cb7d3c51f9e66417d6e621fbec5b87c988`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0ocf_source_snapshot_20260811_v2/d92_e0ocf_runtime_closure_c55f4241.tar.gz`|
|`configs/stage2_d92_e0ocf_5arm_hard12v3_v1.json`|3045|`f16b21e03969b7a1710594a76aac03fe40d1f6b1b4fd361b15e4e45153f0a3cc`|`.../d92_e0ocf_source_snapshot_20260811_v2/configs/stage2_d92_e0ocf_5arm_hard12v3_v1.json`|
|`automation_reports/CV-SincNet/d92_e0ocf_5arm_hard12v3_20260811_v2/launch.sh`|3603|`d1bbb6ff2c2f573df9b8e73a91faca9a1a6c46406d15091da8066f899aa55474`|`.../d92_e0ocf_source_snapshot_20260811_v2/launch.sh`|

远端Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0ocf_source_snapshot_20260811_v2`。冻结启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0ocf_source_snapshot_20260811_v2 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

output=`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0ocf_5arm_hard12v3_20260811_v2`；logs=`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0ocf_5arm_hard12v3_20260811_v2`；smoke=`$output/smoke`。GPU0–7各运行一个shard，`CUDA_VISIBLE_DEVICES`与shard一一对应，每进程`--device cuda:0 --cpu-threads 2`。

## 6.健康门、预期artifact与晋级门

先执行真实sealed-checkpoint truth-free smoke，再启动8shard。只在query协议违规、输出覆盖风险、launcher级故障，或两个不同outer在prediction前产生同一确定性异常指纹时停止；禁止按中途准确率停止或调参。fresh retry未授权。

健康完成预期：60个job receipt、120个prediction、120个COMMIT、60个score、120个fit-audit文件（每文件3scene，共360行）、120个resource-audit、8个shard summary，且8/8为PASS、failed=0、stderr无异常。

`E0_OCF25`相对`E0_FULL_ONLY`必须提高mean old floor且至少8/10行floor不降，同时H、old BA、seen-new不降、forgetting不增；相对`D92_FULL`必须mean ΔH≥0.5pp、至少8/10行H不降，old BA/floor/seen-new不降、forgetting不增，median wall至少下降60%、peak不增，fit/query/state精确。`E0_OCF50`不得改变裁决。

## 7.结果

尚未启动、尚无性能结果。artifact返回后在本报告追加同排指标、old→old/old→new/new→old、资源表、异常与最终`PROMOTE/NO_PROMOTION`裁决。
