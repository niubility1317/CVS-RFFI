# D110-SCPM真实G0 one-shot发布报告

状态：`LOCAL_VERIFIED / NOT_LANDED / NO_PERFORMANCE_RESULT`

## 1.身份与目标

- run ID：`d110_scpm_g0_oneshot_20260802_033047_r1`
- 时间：2026-08-02
- operator：主agent冻结；唯一N607 Terra Max runner负责服务器落地、执行与回收
- source commit：`179c997d`
- candidate：`D110-SCPM/r3-CONDITIONAL-VAR01`
- 目标：在已封存的真实588条Phase1 tap上运行K1/K5/K10机械功能门，只比较冻结qKNN与D110-SCPM的feature、neighbor、margin和argmax变化。
- 性能边界：不读取或输出accuracy、H、floor、Target truth或source-held性能；本run不能产生性能结论。

## 2.假设与判据

D110用D106量化闭合rank-3基、168个TX×receiver×day单元等权条件方差和12B量化先验构造预测度量。若该机制在真实tap上真正改变邻序和分类决策，三个K的`argmax_changed_count`都应非零。

- 任一K为0：`REJECT_REVISION_NO_FUNCTION`，关闭D110，不修runner、不调参数，直接研发下一机制。
- K1/K5/K10均非零：`G0_PASS_PROCEED_G1`，进入冻结四臂G1。
- 技术停止仅限协议泄漏、错误checkout/hash、输出覆盖风险、确定性异常或零prediction；不得按性能停止。

## 3.本地版本与验证

|项目|结果|
|---|---|
|核心实现commit|`1951f387`|
|G0入口commit|`179c997d`|
|入口|`code/scripts/run_d110_scpm_g0_one_shot.py`|
|测试|`tests/test_run_d110_scpm_g0_one_shot.py`|
|D110专项|`13 passed`|
|D106＋D110 one-shot联合回归|`2 passed`|
|编译／diff|`py_compile PASS`；`git diff --check PASS`|
|独立复审|`P0=0 / P1=0 / G0 RELEASE-READY`|
|协议|`query fit=0`、`query update=0`、`parameter scan=0`；scoring接口不接query label|
|资源|`12B`新增prior；rank=3；d=160；约960MAC投影回写；1MiB增量数组预算内；RSS不宣称实测|
|source archive|`d110_scpm_g0_source_179c997d.zip`；7,866,481B；SHA256=`920aa871a52db0e77855de2a5a4c41f1b79903aed07ce243d2d5be85c108b931`|

合成588行smoke能够完整执行，但其K1/K5/K10均未改变argmax；该结果只反映合成样本过易，不参与真实G0裁决，也未触发参数修改。

## 4.冻结真实输入

- tap archive：
  `/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`
- archive SHA256：`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`
- registered classes：`14-10,14-7,20-15,20-19,6-15,8-20`
- 数据状态：复用D106 r7已经完成的真实checkpoint→tap技术闭环；不重验Phase2数据。

## 5.N607预登记

- remote root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d110_scpm_g0_oneshot_20260802_033047_r1`
- source archive：`<run-root>/input/d110_scpm_g0_source_179c997d.zip`
- CWD：`<run-root>/source/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：预定物理GPU0；以preflight实时占用为准，若不适合则只更新实际分配，不改变方法
- log：`<run-root>/logs/run.out`
- PID：`<run-root>/logs/launch.pid`
- output：`<run-root>/output/g0_result.json`，落地前必须不存在
- expected artifacts：`run.out`、`launch.pid`、`g0_result.json`
- fresh-run retry：未授权；技术故障须保留r1并返回本地修复

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d110_scpm_g0_oneshot_20260802_033047_r1/source/code/scripts/run_d110_scpm_g0_one_shot.py --archive /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --registered-class 14-10 --registered-class 14-7 --registered-class 20-15 --registered-class 20-19 --registered-class 6-15 --registered-class 8-20 --run-id d110_scpm_g0_oneshot_20260802_033047_r1 --output /home/szu2070436088/2510044040/CV-SincNet/runs/d110_scpm_g0_oneshot_20260802_033047_r1/output/g0_result.json
```

## 6.回收后只检查

- schema、run ID、tap SHA和588行／28折／K1/K5/K10闭合；
- 每K的feature、neighbor、margin、argmax changed count与roots；
- `zero_changed_k_values`及唯一功能裁决；
- 资源预算、query零fit/update、无性能字段；
- PID自然退出、GPU释放和SSH连接清理。

本节之后由唯一runner补充LANDED／RUNNING／ARTIFACTS_COMPLETE证据；主agent只依据完整真实G0结果作G1或拒绝决策。

## 7.唯一runner实际执行与回收

最终状态：`ARTIFACTS_COMPLETE / REAL_ARCHIVE_G0_EXECUTED_NON_FORMAL_FUNCTIONAL_EVIDENCE / NO_PERFORMANCE_RESULT`

### 7.1落地与运行

|检查项|实际证据|
|---|---|
|direct preflight|通过；N607项目根可见，GPU0—7均为0%利用率、约1MiB显存，无用户训练进程|
|落地前run root|`ABSENT`；随后仅创建本run的`input/`、`logs/`、`output/`|
|真实tap|存在；SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`匹配冻结值|
|源码归档|已同步并在N607复核SHA256=`920aa871a52db0e77855de2a5a4c41f1b79903aed07ce243d2d5be85c108b931`；入口`py_compile`通过|
|GPU／PID|GPU0；PID=`3654115`；启动即时存活，随后自然退出|
|CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d110_scpm_g0_oneshot_20260802_033047_r1/source/code`|
|实际命令|冻结命令以`CUDA_VISIBLE_DEVICES=0`、预登记Python、archive SHA、6个registered class、固定run ID和不可覆盖output执行；使用`nohup`仅用于脱离SSH|
|运行后GPU／进程|PID已`EXITED`；`nvidia-smi`无compute app；未干预任何其他任务|
|SSH|每次短连接后本机`ssh.exe=NONE`且N607 TCP22连接为`NONE`|

启动通道末尾出现`base64: invalid input`提示；随后立即核验该次唯一运行的PID、完整日志、完整结果、远端／本地SHA和JSON字段均闭合，因此未重启、未重跑、未改方法。

### 7.2完整G0结果核验

已回收并解析完整`g0_result.json`，而非仅依据日志尾部。字段核验：`schema=cvs.phase1.d110.scpm_g0.one_shot.v1`；run ID和tap SHA匹配；`row_count=588`；`fold_count=28`；`K_values=[1,5,10]`；每K`query_count=588`。

|K|feature changed|neighbor changed|margin changed|argmax changed|结论|
|---:|---:|---:|---:|---:|---|
|1|588|23|588|23|非零|
|5|588|54|588|40|非零|
|10|588|116|588|96|非零|

- `argmax_changed_count=159`，且`zero_changed_k_values=[]`。
- `functional_gate_pass=true`，`functional_gate_status=G0_PASS_PROCEED_G1`，`g1_entry_allowed=true`。
- `query_rows_used_for_fit=0`、`query_state_updates=0`、`query_label_read_for_scoring=false`、`parameter_scan_count=0`。
- `performance_metrics_emitted=false`、`formal_performance_claim=false`；本run没有accuracy、H、floor或任何Target truth性能结论。
- `resource_budget_exceeded=false`；峰值增量数值数组估计`133408B`，低于`1048576B`预算。

### 7.3回收工件

|工件|远端SHA256|字节数|本地回收路径|
|---|---|---:|---|
|`g0_result.json`|`219925bd277c07b1b0d850b7aae367ebe5c76ecbd70578cff60406d13d6e3afa`|8368|`artifacts/remote/g0_result.json`|
|`run.out`|`186b51830855db4eb502e2f0d0787d121f2fcb09d23d2e6efa64238a76d01760`|8278|`artifacts/remote/run.out`|
|`launch.pid`|`7366f45abb07e68f1eeb0cb5b4e0715cd4e04fd9ceeb4cbcc06f90f3afc0c6cd`|8|`artifacts/remote/launch.pid`|

唯一runner交接：真实588条G0已完成且三种K的argmax变化均为非零。该结论仅允许主agent按冻结规则进入G1；它不构成任何held或Target性能收益声明。
