# D106真实588条RCMR-G0 r1预登记报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 1.身份、目标与假设

- run ID：`d106_rcmr_g0_real_20260801_121040_r1`
- 时间：2026-08-01
- operator：主agent完成本地冻结；唯一N607 Terra Max runner负责落地、启动、健康检查和artifact回收
- release source commit：`2af0b9158efb76373a6d26715c63f591f43075f2`
- candidate：`D106-RCMR-2V-qKNN/r1.1`，配套D106 RDCE/GTSM-r3训练面
- 目标：在真实588条Phase1 strict tap上执行K1/K5/K10机械G0，确认HEAD是否真实改变feature、neighbor/margin或argmax。
- 假设：双视图跨类秩可靠度与双侧拥挤度在真实28 receiver-day fold上对K1/K5/K10均产生至少1个top1 prediction改变。
- 比较目标：同fold冻结Student-t qKNN机械baseline；本run不访问accuracy、H、floor、source-held truth或Target truth。

判定固定：任一K的`argmax_changed_count=0`即`REJECT_REVISION_NO_FUNCTION`，停止该revision并恢复方法研发；三个K均非零则直接进入G1。不得根据本run扫描参数、调阈值或修改HEAD。

## 2.本地实现、验证与版本

|文件面|目的|
|---|---|
|`stage2_d106_rcmr_g0.py`|真实strict tap→fresh K1/K5/K10 locks→28fold RCMR机械执行与production结果|
|`stage2_d106_train_only_predecessor_lock.py`|同一无跟随FD内完成hash和parse|
|`run_d106_rcmr_g0_production.py`|POSIX不可覆盖父runner，双clean child后最后发布completion|
|`d106_rcmr_g0_clean_child.py`|受manifest约束的execute/verify隔离入口|
|`build_d106_rcmr_g0_release_manifest.py`|从5个根模块AST固定点生成29模块canonical release manifest|
|`tests/test_*d106*rcmr*g0*`|协议、功能、production、manifest及失败边界|

独立Sol High终审为`P0=0/P1=0/P2=1 / LOCAL IMPLEMENTATION GO`；完整相邻闭包169通过、1项真实fixture预期跳过。P2只涉及未来`cvsrffi/__init__.py`新增静态导入时的提前提示，不影响当前空init与29模块闭包，按用户要求不阻塞实验。工作树已提交且无未提交代码改动；不push、不上传GitHub。

## 3.Release资产与同步目标

|本地资产|SHA256|远端目标|
|---|---|---|
|`d106_rcmr_g0_source_2af0b915.zip`|`652a1237f6699670620d868d03046150b6334d409bdea5f362f8954586b181bb`|`<run-root>/input/release/source.zip`|
|`d106_rcmr_g0_release_manifest.json`|`a4d76d4522dab5999f69e2596c3af72aced696bbf89eda925a46a84245a6599c`|`<run-root>/input/d106_rcmr_g0_release_manifest.json`|

复用且只读的r7实际输入：

|远端资产|SHA256|
|---|---|
|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|

不得修改、移动或覆盖r7目录。

## 4.N607预登记

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_121040_r1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<run-root>/source/code`
- GPU：记录物理GPU0占用；`CUDA_VISIBLE_DEVICES=0`，但本机械G0预期主要为CPU/NumPy
- log：`<run-root>/logs/run.out`
- PID：启动后由唯一runner记录
- output：`<run-root>/output`，启动前必须`ABSENT`
- expected：`d106_rcmr_g0_production_result.json`、`d106_rcmr_g0_production_execution_manifest.json`、`COMPLETED.json`
- retry：`NOT_AUTHORIZED`
- 参数扫描：0
- Target/source-held truth：禁止

唯一实验命令：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_121040_r1/source/code/scripts/run_d106_rcmr_g0_production.py --archive /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.receipt.json --receipt-sha256 24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665 --release-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_121040_r1/input/d106_rcmr_g0_release_manifest.json --expected-release-manifest-sha256 a4d76d4522dab5999f69e2596c3af72aced696bbf89eda925a46a84245a6599c --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_121040_r1/output
```

## 5.健康、停止与成功标准

启动后只检查PID/CWD/cmdline/run-root、log增长、异常指纹、输出文件数、CPU/GPU占用和completion。不得读取accuracy、H、floor或其他性能指标。

- P0协议/安全错误、wrong hash/path、输出覆盖风险或确定性异常：停止且仅停止本run进程，保留partial，状态`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 正常完成：result、manifest、completion全部canonical且SHA闭合，28fold×3K覆盖588query/K，无query truth/selection/update，资源未超限。
- 科学功能门：K1/K5/K10每个`argmax_changed_count>0`。
- 本run只有一个job，不因中间changed count或任何指标提前停止。
- 完成后确认run-owned进程=0、GPU资源释放、本地SSH/TCP22连接清理。

## 6.完成后填写

|K|fold|query count|feature changed|neighbor/margin changed|argmax changed|结论|
|---|---:|---:|---:|---:|---:|---|
|K1|待填|待填|待填|待填|待填|待填|
|K5|待填|待填|待填|待填|待填|待填|
|K10|待填|待填|待填|待填|待填|待填|

当前状态严格为`NO_NEW_PERFORMANCE_RESULT`。

## 7.唯一N607 runner执行结论

- 直连预检：通过。项目根可见；GPU0—7均为0%利用率、1MiB显存占用；无活动compute作业。
- 落地：新run root在落地前为`ABSENT`；r7 tap与receipt SHA均匹配；冻结source zip和release manifest完成远端SHA、canonical JSON、解包及关键脚本编译复核。
- 启动：2026-08-01 12:21:35 CST；PID=`3191403`；`CWD=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_121040_r1/source/code`；`CUDA_VISIBLE_DEVICES=0`；命令与§4逐字一致。
- 首次3秒健康检查：主PID已退出，`output=ABSENT`，没有result、execution manifest或`COMPLETED.json`。
- 确定性异常指纹：实际clean child导入链到`cvsrffi.dual_feature_forward`时抛出`ModuleNotFoundError: No module named 'model_dual_cvsincnet'`。这是启动/导入闭包错误，不是性能结果。
- 清理：随后核验`run_owned_process_count=0`、GPU compute表为空；本地`ssh.exe=0`，N607/bridge TCP22连接=0。未终止任何非本run进程。

|工件/状态|结果|
|---|---|
|远端`logs/run.out`|4315bytes；SHA256=`b8aedbcf5e5eb64c7257e665ba455898526d8607b2f6bd5273c7ef5cb6869240`；已回收至root report的`artifacts/remote/run.out`|
|远端`logs/launch.pid`|`3191403`；SHA256=`093bcb16a383bc1006058e6fc3b9935939bf063aa76200395ce03fefa5f9af58`；已回收至root report的`artifacts/remote/launch.pid`|
|prediction/feature/neighbor/margin/argmax统计|未产生；三个K均未进入执行完成阶段|
|性能指标|未读取、未产生；`NO_PERFORMANCE_RESULT`|
|重试|`NOT_AUTHORIZED`；唯一runner未改方法、代码或输入，也未重启该run|

本revision的本次run结论为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。应由主agent在本地修复具体导入闭包、重新独立审查并用新的不可覆盖run ID发布；不得覆盖或恢复本run。
