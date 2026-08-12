# D92 E0 FULL CSOAS Hard9+K1实验报告

状态：`LOCAL_VERIFIED / READY_FOR_N607_HANDOFF`

## 1.目标与冻结身份

|字段|值|
|---|---|
|run ID|`d92_e0_full_csoas_hard9k1_20260812_v1`|
|科学commit|`b8ebd4f4522fcc3e9e6b7dd18d722c329021f181`|
|机械commit|`ac811820`；P1修复`7a434080`|
|候选|`E0_FULL_CSOAS`；candidate=`d92_e0_full_csoas`；mode=`csoas_full`|
|目标|仅在与G0不重叠的9个最难performance outer上对比同outer`E0_FULL_ONLY`；另保留1个K1 liveness|
|声明|development-only Hard9；完整artifact返回前不作性能结论|
|retry|`false`；同一run ID只允许唯一detached launch|

CSOAS G0 v2已通过：三场景active、无fallback，wall P90=`14.268484ms`，query禁用与peak门闭合。Hard9才读取独立truth-side score并做性能裁决。

## 2.矩阵与裁决

- `p2_min_v1`，沿用`VALIDATED_ONCE`sealed inputs，不重验数据。
- 9个performance outer+1个K1 liveness，3个`leo_*_weak`场景，10 jobs、30 scene rows、8 shards；明确排除G0 outer`rx_7_7__seed_713106__k_10__new_5`。
- selection SHA256=`a851590bc6d502ddbe326a936096d95f5bb382e4cb235b61b0121d98c0b87b5d`。
- K>2必须active、无fallback、candidate/reference fit=`1/0`、actual FULL=`1`、two-state total=`2`；任何数值fallback或codec retry直接拒绝。K1必须精确alias=`K1_K2_EXACT_D92_FULL_ALIAS`。
- 八项总体均值严格优于E0：`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new accuracy升高；average forgetting、new→old、old→new降低。任一持平/反向或稳定性/资源硬门失败即`REJECT_ROUTE`。
- 大胆目标：`+1.0pp,+1.5pp,+1.0pp,+4.0pp,+0.5pp,-1.5pp,-0.5pp,-0.5pp`。资源硬门：wall P90≤150ms、paired wall ratio median≤1.50、peak delta≤512KiB、query/state逐row等同E0；120ms/1.25为目标门。

## 3.本地实现与验证

新增config、matrix、runner、analyzer、两个CLI和三组测试共8文件。TDD初始RED为缺模块ImportError；GREEN：focused 17 passed、邻接29 passed，py_compile、JSON、两个CLI `--help`和diff-check通过。动态C=11/16/26资源字段按`registered_class_count×288`核验并逐row对照E0。

独立审查发现并修复两个P1：并发shard不再递归重写全run root，只处理本shard job/summary与定点shared failure evidence；smoke恢复truth SHA、expected command、before/after prediction、COMMIT、fit audit哈希与完整prediction closure绑定。修复后runner focused 11 passed、py_compile/diff-check通过。

## 4.交付与服务器

|交付|size|SHA256|
|---|---:|---|
|`d92_csoas_hard9_runtime_7a434080.tar.gz`|6,184,379|`cf8d14a2ae3bc7f66c0dc9ce00132ee7114dd261785e5c53cd08b6713ed0211c`|
|`stage2_d92_csoas_hard10_v1.json`|6,293|`6fcd29dfab77c99745df336f32425dfdc0a0a0a99469c92766a4751fa92e427e`|
|`launch.sh`|3,717|`13e6a99479b17455bb2da6a911232a0eb4ae2e11cee93d9ba191c4bf2ff7c615`|

archive共1466 members，必需入口齐全且无`code/code`嵌套；launch `bash -n`通过。

|用途|远端路径|
|---|---|
|source|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_hard9_source_7a434080_20260812_v1`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_csoas_hard9k1_20260812_v1`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_csoas_hard9k1_20260812_v1`|
|context|`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json`|

唯一命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_hard9_source_7a434080_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

环境为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；8 shards分别映射GPU0–7并在可见域使用`cuda:0`。先prepare与真实K5/new20 truth-free smoke，通过后才启动8 shards。

## 5.健康、停止与期望artifact

仅wrong hash/CWD、覆盖风险、协议/安全违规、launcher确定性异常或两个distinct outer在prediction前出现同指纹时停止；不得按accuracy/H/floor等性能值停止。期望10 job receipts、正式before/after prediction/COMMIT/fit/resource/execution各20、score 10、shard summary 8；K1不进入性能均值。完成后完整取回source/output/logs和10份manifest引用truth sidecar，再离线运行冻结analyzer。

## 6.运行与分析结果

待sole runner回填。
