# D106-RCMR-2V-qKNN真实G0 one-shot发布报告

状态：`ARTIFACTS_COMPLETE / G0_PASS_PROCEED_G1 / NO_PERFORMANCE_RESULT`

## 1.身份与纠偏

- run ID：`d106_rcmr_g0_oneshot_20260801_132809_r3`
- 时间：2026-08-01
- operator：主agent冻结；唯一N607 Terra Max runner负责服务器执行与artifact回收
- code commit：`e20cb251a5585556dc24d26230e59e5e1c769d5b`
- candidate：`D106-RCMR-2V-qKNN/r1.1`
- 目标：在真实588条Phase1 tap上运行K1/K5/K10 G0，仅输出feature、neighbor、margin、argmax机械变化与资源摘要，不读取或输出性能truth。
- 纠偏原因：r1导入闭包失败，r2 receipt semantic closure失败；两轮release缺陷后停止修补通用manifest/receipt链，改用本次单文件one-shot入口。

## 2.最小发布门

| 项目 | 结果 |
|---|---|
|协议/query边界|固定8成员，无truth/accuracy/H/floor；held cell排除后才读取support label；query label不用于score|
|真实输入|外部固定真实tap SHA；6类registry由CLI预注册，不从全体标签推导|
|功能入口|直接执行冻结`g0._execute_fold`，审计argmax须与core逐query一致|
|机械证据|每K输出feature、neighbor、margin、argmax changed count和roots|
|证据契约|`g0_decision_consumption_allowed=true`；仅`functional_gate_pass=true`时`g1_entry_allowed=true`|
|本地验证|`py_compile`通过；唯一588行无truth smoke：`1 passed`|
|独立复核|两项原P1已修；最终仅状态字符串P1，两行修复后同一测试通过；其余`P0=0、P1=0`|
|Git|本地commit`e20cb251`；不push|
|不可覆盖|新run ID；远端root和结果文件落地前必须为`ABSENT`|

## 3.冻结资产与输入

| 工件 | SHA256 |
|---|---|
|`d106_rcmr_g0_oneshot_source_e20cb251.zip`|`55c4dd8fe5fc8957836a6b67ef1b376cc4846ae3e271252498edb6126f031e83`|
|`run_d106_rcmr_g0_one_shot.py`|`a2d83454a767ff2f52334e960968fa08a8729f3404d2577325b0b94ed9b5e383`|
|真实tap archive|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|

真实tap路径：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`。

## 4.N607命令

- remote root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_oneshot_20260801_132809_r3`
- CWD：`<run-root>/source/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：预登记`CUDA_VISIBLE_DEVICES=0`，启动前只读检查
- log：`<run-root>/logs/run.out`
- output：`<run-root>/output/g0_result.json`，必须预先不存在
- retry：无

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_oneshot_20260801_132809_r3/source/code/scripts/run_d106_rcmr_g0_one_shot.py --archive /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --registered-class 14-10 --registered-class 14-7 --registered-class 20-15 --registered-class 20-19 --registered-class 6-15 --registered-class 8-20 --run-id d106_rcmr_g0_oneshot_20260801_132809_r3 --output /home/szu2070436088/2510044040/CV-SincNet/runs/d106_rcmr_g0_oneshot_20260801_132809_r3/output/g0_result.json
```

## 5.判据

- 任一K的`argmax_changed_count=0`：`REJECT_REVISION_NO_FUNCTION`，停止当前revision并返回HEAD/DA研发。
- 三个K均非零：`G0_PASS_PROCEED_G1`，立即进入冻结四臂`M0/M_DA/M_HEAD/M_JOINT`。
- G0不计算accuracy、H、floor或Target truth；不得用中间性能停止。

## 6.实际执行与功能证据

N607 direct preflight通过；8张RTX 3090在启动前均为0%利用率、1MiB显存占用。远端run-root初始为`ABSENT`，冻结zip上传后SHA256匹配，解压和唯一入口`py_compile`通过。冻结命令以PID=`3225766`在GPU0启动，3秒健康检查时进程为`LIVE`，CWD与完整命令行均绑定本run；其后自然退出并形成结果JSON。`run.out`为7153字节，未发现Traceback、Error、Exception、RuntimeError、OOM或NaN指纹。

结果契约核验：`schema=cvs.phase1.d106.rcmr_2v_g0.one_shot.v2`，`run_id`和真实tap SHA逐字匹配，`row_count=588`，`real_archive_g0_executed=true`，`g0_decision_consumption_allowed=true`，`g1_entry_allowed=true`，`formal_performance_claim=false`，`performance_metrics_emitted=false`，`query_label_read_for_scoring=false`。

|K|query数|feature变化数／bitmap roots root|neighbor变化数／bitmap roots root|margin变化数／bitmap roots root|argmax变化数／bitmap roots root|
|---:|---:|---|---|---|---|
|1|588|588／`5861f9b47759c73459175610e69705ba8986f5e9ced13261bd339382fc45b140`|20／`81951aa84c6fb2441249f6f2b29d4649e422c2702c1bf79c36ca0a979469e5fc`|588／`f91c611960dd315f5035e38abb6c7a571d434d1f85f2b566bfa8c3fd69890bed`|20／`1be8e913da18a4ebc8134890034f3115ecb121ef7e24a106a64e246f0ea4ac9b`|
|5|588|588／`5861f9b47759c73459175610e69705ba8986f5e9ced13261bd339382fc45b140`|193／`63dd19dd099f94c9ad8b90754caf9737a345fc0294b2f99dd405258c2ff6cf96`|588／`f91c611960dd315f5035e38abb6c7a571d434d1f85f2b566bfa8c3fd69890bed`|28／`54a4316a9b5ae85bf638a98a4db35f257ebdc00a94b2c22756696bcd8e7c4e30`|
|10|588|588／`5861f9b47759c73459175610e69705ba8986f5e9ced13261bd339382fc45b140`|262／`0c0cabfa4dddcb9e23f8a35e1abe28bf9af1b6fd3c5d1d4247247c16e2150231`|588／`f91c611960dd315f5035e38abb6c7a571d434d1f85f2b566bfa8c3fd69890bed`|87／`47a364b8a5f21e7319c9dc660e547808532388d8a82f4dcc0f53abb4288b7520`|

|资源字段|结果|
|---|---:|
|analysis numeric array budget|1048576B|
|incremental numeric array peak estimate|820880B|
|parameter scan count|0|
|query state updates|0|
|analysis budget is process RSS cap|false|
|process RSS measured|false|

三个K的`argmax_changed_count`分别为20、28、87，均非零；`zero_changed_k_values=[]`。因此本run最终分类为`G0_PASS_PROCEED_G1`，允许主agent直接准备冻结G1四臂。该结论只证明真实功能路径生效，不是held性能或Target性能。

小工件已回收到`artifacts/remote/g0_result.json`、`artifacts/remote/run.out`和`artifacts/remote/launch.pid`。PID已退出，本地`ssh.exe`与N607 TCP22连接均已清理。
