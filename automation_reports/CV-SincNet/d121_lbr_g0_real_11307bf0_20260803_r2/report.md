# D121-LBR-qKNN真实archive G0第1次release repair报告

状态：`ARTIFACTS_COMPLETE / ANALYZED / G0_PASS / NO_PERFORMANCE_CLAIM`

## 1.身份、目标与修复边界

- 实验ID：`d121_lbr_g0_real_11307bf0_20260803_r2`
- 时间：2026-08-03
- operator：主agent集成；唯一Terra Max runner落地、启动、健康检查和回收
- 目标：在固定588行真实checkpoint strict tap上运行K1/K5/K10无truth G0，检验LBR是否在每个K改变至少一个M0 argmax。
- 假设与比较：`M_HEAD(LBR)`对同fold、同query的`M0`；任一K的argmax changed为0即关闭当前revision。
- r1状态：`STOPPED_BEFORE_LANDING / NO_PERFORMANCE_RESULT`。唯一技术缺陷是N607不存在报告锁定的`ssr-gpu`解释器路径；没有创建run root、同步或启动。
- r2只把远端Python修正为已只读验证存在的`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。方法、实现、输入、K、类集合、source archive、输出结构、功能门和stop rule完全不变。这是第1次release repair。

## 2.协议、输入与版本

- `protocol_schema=p2_min_v1`；query零fit、零update、零selection，无truth/role/quota/global assignment。
- K：`1,5,10`；每K 28fold、588query。
- 注册类：`14-10,14-7,20-15,20-19,6-15,8-20`。
- strict archive：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`
- archive SHA256：`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`
- checkpoint SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- 分支：`codex/stage2-da25-r1`
- 设计commit：`114038ea`
- 实现commit：`11307bf0c5d43ea45088d63c284d585a42456931`
- 禁止push。

根目录`E:\type10-7`非Git仓库。本报告同步到Git承载面：
`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d121_lbr_g0_real_11307bf0_20260803_r2\report.md`。

|发布文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d121_lbr_qknn.py`|`97836fe59dcfcc0649f77a0463067eccd537b0b0bec7e5485fe966759ff518bc`|
|`code/scripts/run_d121_lbr_g0_one_shot.py`|`3ab095cb9404bb58f1783c22c06d7ed4d7c580adf638f781a0be7343cf78b201`|
|`d121_source_11307bf0.zip`|`c8becddb961bc01b543f47df08f1d6d43914622e5f2cf8e08ca21aad8dd9ec6e`|

source archive复用r1本地不可变文件：
`E:\type10-7\automation_reports\CV-SincNet\d121_lbr_g0_real_11307bf0_20260803_r1\input\d121_source_11307bf0.zip`，大小7,977,276bytes。

## 3.已通过验证

- `ssr-gpu`本地语法与聚焦测试：8 passed。
- 独立审查：`MERGE / P0=0 / P1=0 / P2=0`。
- 同一真实strict tap本地无truth smoke：K1/K5/K10的score changed均588、margin changed均588、argmax changed分别1/3/2；`performance_metrics_emitted=false`。
- r1 N607预检：直连通过、普通账户、项目根可见、8GPU空闲、archive hash匹配、r1 run root ABSENT。
- r1补充环境收据：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`可执行，Python3.10.19，`import numpy`成功。

不重复方法测试、数据验证或独立审查。

## 4.N607不可覆盖预登记

远端run root：
`/home/szu2070436088/2510044040/CV-SincNet/runs/d121_lbr_g0_real_11307bf0_20260803_r2`

- source：`<run-root>/source`
- CWD：`<run-root>/source`
- log：`<run-root>/logs/runner.log`
- PID：`<run-root>/logs/launch.pid`
- exit：`<run-root>/logs/runner.exit`
- output：`<run-root>/output/result.json`，创建run root前及启动前都必须ABSENT。
- GPU：`CUDA_VISIBLE_DEVICES=0`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

精确child command：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d121_lbr_g0_one_shot.py --archive /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --registered-class 14-10 --registered-class 14-7 --registered-class 20-15 --registered-class 20-19 --registered-class 6-15 --registered-class 8-20 --run-id d121_lbr_g0_real_11307bf0_20260803_r2 --output /home/szu2070436088/2510044040/CV-SincNet/runs/d121_lbr_g0_real_11307bf0_20260803_r2/output/result.json
```

## 5.runner、stop与artifact

唯一runner须重新执行短只读preflight；确认r2 root ABSENT、GPU/进程、source/archive hash后，才同步、解包、编译和detached启动。每次SSH/SCP结束都核对`ssh.exe=0`及N607/lab TCP22清零。

技术停止：wrong hash/path、output存在、query leakage、非零exit、零prediction或确定性异常；保留partial并记`NO_PERFORMANCE_RESULT`，不复用r2、不擅自重试。不得根据changed count中途停止。

完整结果只读取结构性字段：

|字段|要求|
|---|---|
|`K_values`|`[1,5,10]`|
|`query_count_per_k`|588|
|`performance_metrics_emitted`|false|
|`query_label_read_for_scoring`|false|
|`argmax_changed_count_by_k`|每K完整；每项大于0才允许G1|
|`output/result.json`|不可覆盖、hash回收|
|`logs/runner.log`、PID、exit|完整回收|

若三个K均非零，主agent立即进入冻结四臂G1；任一K为0则关闭D121当前revision，不调参、不扩矩阵。

## 6.N607完成结果

- 直连preflight通过；r2 root落地前为`ABSENT`，zip、核心、runner和strict archive SHA256全部匹配，远端`py_compile`通过。
- 固定child command使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`、`CUDA_VISIBLE_DEVICES=0`和`<run-root>/source`，PID=`39601`。
- G0在首次5秒健康轮询前自然结束，exit=`0`，log异常指纹计数0，result包含完整K1/K5/K10、28fold和588query/K。
- 由于进程结束很快，没有取得存活时的`/proc` CWD/cmdline快照；未为补观测而重跑。wrapper的`cd`由exit0、正确schema/run ID和完整result闭合验证。
- 最终无活PID，8张GPU均0%利用率、约1MiB显存占用；本地`ssh.exe=0`，N607/lab TCP22的`ESTABLISHED=0`。

|K|support-kernel changed|margin changed|argmax changed|判定|
|---:|---:|---:|---:|---|
|1|588|588|1|非零|
|5|588|588|3|非零|
|10|588|588|2|非零|

- `zero_changed_k_values=[]`
- `functional_gate_status=G0_PASS_PROCEED_G1`
- `g1_entry_allowed=true`
- `performance_metrics_emitted=false`
- `query_label_read_for_scoring=false`
- `formal_performance_claim=false`

## 7.完成artifact

|artifact|SHA256|
|---|---|
|`d121_source_11307bf0.zip`|`c8becddb961bc01b543f47df08f1d6d43914622e5f2cf8e08ca21aad8dd9ec6e`|
|`result.json`|`0c5d97d5196afdce1ed3a034e7a4f997b583c337af03a18fb322c74e72547a84`|
|`runner.log`|`c5d3940a7504c9bde58fd94ed71ffd0f6457c93daf6a6185ca85cab9864cee54`|
|`launch.pid`|`465dd53aea0d5ff0f45cad4a030e73084070fed2dd4aa3cfc3563f52913fd419`|
|`runner.exit`|`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`|
|`final_cleanup_receipt.txt`|`0129b2669d0548ee5538f3cac93c471cd0228265c5a1834d2b6bb7856f7e3edf`|

最终解释：D121-LBR在三个K上均能改变真实checkpoint archive的决策，因此通过功能证伪门。该结果不包含准确率或收益，不能回答性能是否更强；下一步只运行冻结四臂source-held G1，弱则立即关闭并转入下一原理候选。
