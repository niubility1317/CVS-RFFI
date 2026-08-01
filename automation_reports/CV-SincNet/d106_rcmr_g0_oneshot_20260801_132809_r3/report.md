# D106-RCMR-2V-qKNN真实G0 one-shot发布报告

状态：`LOCAL_COMMITTED / ONE_SHOT_ASSET_BUILT / N607_NOT_LANDED / NO_NEW_PERFORMANCE_RESULT`

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

