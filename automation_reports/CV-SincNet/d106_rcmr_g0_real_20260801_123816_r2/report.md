# D106-RCMR-2V-qKNN真实G0发布报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 1.身份与目标

- run ID：`d106_rcmr_g0_real_20260801_123816_r2`
- 时间：2026-08-01
- operator：主agent冻结发布；唯一N607 Terra Max runner负责落地、启动、健康检查和artifact回收
- release commit：`83bb3f9deef8338251b9e4cd5c0854911e9f0197`
- candidate：`D106-RCMR-2V-qKNN/r1.1`
- 目标：在既有真实588条Phase1 strict tap上运行K1/K5/K10机械G0，仅确认feature、neighbor/margin和argmax是否发生变化。
- 前序失败：`d106_rcmr_g0_real_20260801_121040_r1`因clean child缺少顶层`model_dual_cvsincnet`而启动期退出，未产生output或性能结果。本run只修复该导入闭包，方法、参数、数据和G0判据不变。

## 2.最小发布门

| 项目 | 结果 |
|---|---|
|协议负测/query隔离|既有冻结检查通过；本修复未触碰协议或query路径|
|真实checkpoint无query入口|既有入口保持不变|
|导入缺陷聚焦测试|`16 passed`；3个生产文件`py_compile`通过；两个CLI入口`--help`通过|
|独立复核|`P0=0、P1=0、P2=0 / RELEASE`|
|Git|`83bb3f9deef8338251b9e4cd5c0854911e9f0197`，本地提交，不push|
|不可覆盖|新run ID；远端run-root与output必须在落地前为`ABSENT`|
|数据验证|复用`VALIDATED_ONCE`真实tap；候选修复不触发重复数据验证|

## 3.冻结输入与release资产

| 工件 | SHA256 |
|---|---|
|`d106_rcmr_g0_source_83bb3f9d.zip`|`7e94c1dc4d0c2194332a63444a7fab5a5ba40104969e61173dcb6c121f6e7a4d`|
|`d106_rcmr_g0_release_manifest.json`|`cff6f692f1b4c36ff49f918fa02e05a71b33496e53805559e12ac1800e83e2f4`|
|真实tap archive|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|真实tap receipt|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|

manifest保持29个`cvsrffi`闭包模块，并显式加入已哈希的`model.py`和`model_dual_cvsincnet.py`。注册类固定为`14-10、14-7、20-15、20-19、6-15、8-20`。

## 4.N607发布面

- 远端run-root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_123816_r2`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<run-root>/source/code`
- 预登记GPU：`CUDA_VISIBLE_DEVICES=0`；runner须先做只读preflight和占用检查
- 日志：`<run-root>/logs/run.out`
- 输出：`<run-root>/output`
- 重试授权：无；任何新启动必须使用新的不可覆盖run ID

精确命令：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_123816_r2/source/code/scripts/run_d106_rcmr_g0_production.py --archive /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.receipt.json --receipt-sha256 24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665 --release-manifest /home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_123816_r2/input/d106_rcmr_g0_release_manifest.json --expected-release-manifest-sha256 cff6f692f1b4c36ff49f918fa02e05a71b33496e53805559e12ac1800e83e2f4 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_real_20260801_123816_r2/output
```

## 5.G0判据与停止规则

G0不得读取accuracy、H、floor或Target truth。只记录K1/K5/K10的feature变化、neighbor/margin变化、argmax changed count、资源与协议/执行错误。

- 任一K的argmax changed count为0：`REJECT_REVISION_NO_FUNCTION`，停止该revision并回到HEAD/DA方法研发。
- 三个K均非零：`G0_PASS_PROCEED_G1`，立即进入冻结四臂`M0/M_DA/M_HEAD/M_JOINT`。
- 仅P0协议/安全或确定性执行错误可停止技术run；不得按性能停止。

## 6.实际运行与回收

|项目|实际证据|
|---|---|
|N607直连预检|通过；8张3090在启动前均无计算任务，GPU0预登记可用|
|远端落地|run-root初始为`ABSENT`；两项冻结资产上传后SHA256逐字匹配，source解压和入口`py_compile`通过|
|启动|以预登记命令在GPU0启动，PID=`3204843`|
|启动健康|3秒内退出，`exit=1`；`output`未创建|
|唯一错误指纹|`D106Phase1TapError: D106 tap receipt semantic closure drift`，经clean child封装为`strict D106 tap loader rejected production inputs`|
|性能/真值读取|未读取accuracy、H、floor或Target truth；未形成prediction、result、completion或score artifact|
|回收与清理|已回收远端`logs/run.out`和`launch.pid`至`artifacts/remote/`；PID已退出、GPU无该PID、本地`ssh.exe`与N607 TCP22连接均已清理|

本run属于启动期确定性技术失败，结论为`NO_PERFORMANCE_RESULT`，不构成G0通过、G0拒绝或任何性能结论。按冻结规则，不在本run重试、不调参、不修改方法；如主agent决定修复，必须先在本地聚焦处理tap receipt语义闭包问题、使用新的不可覆盖run ID，并重新交由唯一runner发布。
