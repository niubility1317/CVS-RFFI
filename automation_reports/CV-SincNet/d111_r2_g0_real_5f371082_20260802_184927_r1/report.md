# D111-r2真实G0实验报告

状态：`LOCAL_RELEASE_READY / NOT_LANDED / NO_PERFORMANCE_RESULT`

## 1.实验登记

|字段|值|
|---|---|
|run ID|`d111_r2_g0_real_5f371082_20260802_184927_r1`|
|时间|2026-08-02 18:49:27（Asia/Hong_Kong）|
|operator|主agent研发与整合；唯一N607 runner待交接|
|目标|用真实588条Phase1 strict tap验证D111-r2在K1/K5/K10是否真正改变prediction|
|证据边界|G0只读feature、neighbor、anchor、score、margin、argmax变化；不读或输出accuracy、H、floor和truth-side指标|

## 2.假设与冻结判据

D111-r2用Phase1 int8聚合类锚和rank-3共享域子空间，从其他5类support估计当前类的LOO共享位移，并以单位质量Student-t anchor混合修改旧类评分。比较目标为同fold、同query、同qKNN M0。

冻结K为`K1/K5/K10`。任一K的`argmax_changed_count=0`即`REJECT_REVISION_NO_FUNCTION`并关闭当前revision；三K均非零才允许进入G1。feature和support-neighbor预期保持不变，但必须如实输出identity receipt。不得读取性能决定停止或重跑。

## 3.本地版本与验证

|项目|证据|
|---|---|
|实现提交|`5f37108213e7bb70b09097f019dc8a69c2b4474f`|
|设计提交|`cc798133`|
|focused suite|`ssr-gpu`下32项通过|
|真实checkpoint无query smoke|checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；tap SHA256=`6626afbf5d5987b2944b53f9b4bddbb6c9397f4c577accb95cea5e0039b24578`；`p=160/d_eff=12`；6条support；`query_rows_used_for_fit=0`|
|独立复审|`P0=0/P1=0/P2=0 / GO_FOR_MINIMAL_G0_RELEASE`|

实验实际使用的变更文件为：

- `code/cvsrffi/stage2_d111_g0_source_bundle.py`：一次性588行Phase1 int8聚合；
- `code/cvsrffi/stage2_d111_loo_gat_bundle.py`：量化误差运行时字段；
- `code/cvsrffi/stage2_d111_loo_gat_score.py`：r2运输、稳定界和G0-only fit；
- `code/scripts/run_d111_r2_g0_one_shot.py`：28fold×三K无truth功能入口；
- 依赖的D106 G0和qKNN模块由上述Git提交完整归档，不做远端overlay编辑。

## 4.输入与N607路径

|项目|冻结值|
|---|---|
|strict tap|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|
|tap SHA256|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|tap receipt SHA256|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|tap completion SHA256|`c50412f0601a8c9135ff4d743ac71c2e6438ada328fe3c16a3fcf34e15578655`|
|远端run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1`|
|release CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1/release/code`|
|bundle|`/home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1/input/d111_g0_bundle`|
|log|`/home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1/logs/runner.log`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1/output/result.json`|
|GPU|无；该G0消费已生成feature，只运行CPU/Numpy评分；preflight仍记录GPU占用但不占卡|

runner必须先确认`/home/szu2070436088/.conda/envs/ssr-gpu/bin/python`存在；若实际解释器路径不同，只允许把命令机械替换为preflight确认的`ssr-gpu`解释器，并在本报告记录，不得换环境。

## 5.冻结命令

在release CWD中先一次性生成不可覆盖bundle：

```bash
/home/szu2070436088/.conda/envs/ssr-gpu/bin/python -c "from cvsrffi.stage2_d111_g0_source_bundle import build_d111_g0_source_bundle; build_d111_g0_source_bundle('/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz','/home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1/input/d111_g0_bundle',expected_tap_sha256='48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f')"
```

然后唯一launch：

```bash
nohup /home/szu2070436088/.conda/envs/ssr-gpu/bin/python scripts/run_d111_r2_g0_one_shot.py --archive /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --bundle /home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1/input/d111_g0_bundle --registered-class 14-10 --registered-class 14-7 --registered-class 20-15 --registered-class 20-19 --registered-class 6-15 --registered-class 8-20 --run-id d111_r2_g0_real_5f371082_20260802_184927_r1 --output /home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1/output/result.json > /home/szu2070436088/2510044040/CV-SincNet/runs/d111_r2_g0_real_5f371082_20260802_184927_r1/logs/runner.log 2>&1 &
```

## 6.健康、成功与停止语义

- launch后核对PID、CWD/cmdline、run root、log增长和输出路径绑定；
- 技术成功：进程exit 0，结果JSON存在且不覆盖，K1/K5/K10各有28fold和588query，`performance_metrics_emitted=false`、`query_label_read_for_scoring=false`；
- 科学功能裁决只读三K的`argmax_changed_count`；零值是合法负结果，不是技术故障；
- 仅协议/安全错误、错误提交或输入SHA、异常退出、缺失结果、覆盖风险可停止；不得按中间性能停止；
- 不授权retry。若技术失败，保留原run全部artifact，修复后必须使用新run ID。

## 7.完成后必须回收与分析

回收`result.json`、完整log、PID/exit证据、bundle manifest/payload SHA和release commit/hash。更新本报告的状态、每K功能表、资源表、异常、裁决和下一步。若三K均改变prediction，直接设计冻结四臂G1；若任一K为0，关闭D111-r2并研发下一机制，不调rank、`rho`、`eta`、包络或核参数。
