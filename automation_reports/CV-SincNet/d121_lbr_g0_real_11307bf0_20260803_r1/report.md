# D121-LBR-qKNN真实archive G0发布报告

状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_YET_LANDED / NO_PERFORMANCE_RESULT`

## 1.实验身份与目标

- 实验ID：`d121_lbr_g0_real_11307bf0_20260803_r1`
- 时间：2026-08-03 02:27:19+08:00
- operator：主agent负责方法冻结、集成和判定；唯一Terra Max N607 runner负责落地、启动、健康检查和artifact回收
- 目标：在固定588行真实checkpoint strict tap上，对K1/K5/K10运行D121-LBR-qKNN的无truth功能证伪；不得读取或输出accuracy、H、BA、floor等性能。
- 假设：固定异类最近support的单跳log-sigmoid竞争能在三个K上都改变至少一个M0 argmax；任一K为0即关闭当前revision。
- 比较对象：同fold、同query、同M0 Student-t qKNN的`M_HEAD(LBR)`对`M0`；本G0不运行RDCE、不运行四臂、不作性能排序。

## 2.科学与协议锁

- `protocol_schema=p2_min_v1`。
- 输入为D106冻结的588条Phase1可见严格tap；每个query只作机械前向，不读query truth，不更新状态，不选择query。
- 固定K：`1,5,10`；每K为28fold、588query。
- 注册类固定为：`14-10,14-7,20-15,20-19,6-15,8-20`。
- 原始archive SHA256：`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`。
- checkpoint SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`，由既有strict tap receipt绑定；D121不重新导出特征。
- 参数扫描数：0；无epoch、温度、阈值、rank或邻居数调节。

## 3.本地版本与文件

根目录`E:\type10-7`不是Git仓库。本报告同步镜像到Git工作树：
`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt\automation_reports\CV-SincNet\d121_lbr_g0_real_11307bf0_20260803_r1\report.md`。

- 分支：`codex/stage2-da25-r1`
- 设计冻结commit：`114038ea`
- 实现commit：`11307bf0c5d43ea45088d63c284d585a42456931`
- 禁止push。

|本地文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d121_lbr_qknn.py`|LBR固定support图与score|`97836fe59dcfcc0649f77a0463067eccd537b0b0bec7e5485fe966759ff518bc`|
|`code/scripts/run_d121_lbr_g0_one_shot.py`|588行K1/K5/K10无truth入口|`3ab095cb9404bb58f1783c22c06d7ed4d7c580adf638f781a0be7343cf78b201`|
|`tests/test_stage2_d121_lbr_qknn.py`|公式、tie、K、置换、零状态测试|`7ea5a94b192001eb9d771ac95dc0259506a7b9d57941e95d57d5333d42577743`|
|`tests/test_run_d121_lbr_g0_one_shot.py`|G0输出与禁用truth测试|`532769cb220b7d13fd6b73708507dd19ca4f7dfd666405278f171a497f7173de`|

发布源archive：

- 本地：`E:\type10-7\automation_reports\CV-SincNet\d121_lbr_g0_real_11307bf0_20260803_r1\input\d121_source_11307bf0.zip`
- 大小：7,977,276bytes
- SHA256：`c8becddb961bc01b543f47df08f1d6d43914622e5f2cf8e08ca21aad8dd9ec6e`
- 内容：`git archive 11307bf0... code`；远端不得修改。

## 4.本地验证

环境：`ssr-gpu`。

```text
python -m py_compile <4个D121文件>
python -m pytest -q tests/test_stage2_d121_lbr_qknn.py tests/test_run_d121_lbr_g0_one_shot.py tests/test_run_d106_rcmr_g0_one_shot.py
结果：8 passed
```

实现agent独立聚焦测试：`7 passed`。独立Terra Max发布审查：`MERGE / P0=0 / P1=0 / P2=0`。

同一真实strict tap的本地无truth smoke：

|K|score changed|margin changed|argmax changed|功能判定|
|---:|---:|---:|---:|---|
|1|588|588|1|非零|
|5|588|588|3|非零|
|10|588|588|2|非零|

- smoke输出：`C:\Users\lh594\AppData\Local\Temp\d121_real_checkpoint_no_truth_smoke_20260803.json`
- SHA256：`19dab781355a009e7183495fbcc0648a704fb2dc5d4ec85826c58b01010bbeea`
- `performance_metrics_emitted=false`
- `query_label_read_for_scoring=false`
- 当前只证明真实输入路径和功能门可闭合，不构成性能收益。

## 5.N607预登记

远端run root：
`/home/szu2070436088/2510044040/CV-SincNet/runs/d121_lbr_g0_real_11307bf0_20260803_r1`

- source：`<run-root>/source`
- log：`<run-root>/logs/runner.log`
- PID：`<run-root>/logs/launch.pid`
- exit：`<run-root>/logs/runner.exit`
- output：`<run-root>/output/result.json`，落地前必须不存在。
- GPU：`CUDA_VISIBLE_DEVICES=0`；本任务主要为CPU前向，预检仍须记录GPU占用。
- Python：`/home/szu2070436088/.conda/envs/ssr-gpu/bin/python`
- CWD：`<run-root>/source`

精确child command：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/ssr-gpu/bin/python code/scripts/run_d121_lbr_g0_one_shot.py --archive /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --registered-class 14-10 --registered-class 14-7 --registered-class 20-15 --registered-class 20-19 --registered-class 6-15 --registered-class 8-20 --run-id d121_lbr_g0_real_11307bf0_20260803_r1 --output /home/szu2070436088/2510044040/CV-SincNet/runs/d121_lbr_g0_real_11307bf0_20260803_r1/output/result.json
```

## 6.唯一runner职责与健康规则

runner必须：

1.先运行本地只读`tools\n607_ssh_preflight.ps1`，直连`N607`优先；只有直连TCP/SSH不可达且身份无歧义时才用已验证lab bridge。
2.只读检查服务器时间、项目根、GPU、现有任务和run root不存在。
3.同步上述Git archive到新run root，核对zip SHA、解包后两个D121代码SHA并编译；不得remote-only编辑。
4.以不可覆盖路径detached启动；立即核对PID、CWD、cmdline、run-root绑定和log增长。
5.短连接读取completion、exit、异常指纹和result结构；不得查看或计算性能。
6.回收`result.json`、log、PID、exit和必要hash收据，随后验证本地SSH进程与两条TCP22连接均为0。

技术停止条件：

- 任一协议/路径/hash/checkout错误、输出已存在、query leakage、非零exit、零prediction或确定性异常即`NO_PERFORMANCE_RESULT`。
- 不得根据argmax changed多少停止进程；功能判定只在完整三个K闭合后读取。
- G0成功条件仅为三个K的`argmax_changed_count>0`；任一K为0则科学关闭revision，不修参数、不重跑。

## 7.预期artifact与完成后分析

|artifact|要求|
|---|---|
|`output/result.json`|完整三个K；无accuracy/H/BA/floor/truth字段；不可覆盖|
|`logs/runner.log`|命令与异常证据，不含性能|
|`logs/launch.pid`、`logs/runner.exit`|run绑定和结束状态|
|远端hash收据|source zip、核心、runner、archive均匹配|

完成后把本报告状态更新为`ARTIFACTS_COMPLETE / ANALYZED`或`NO_PERFORMANCE_RESULT`。若G0通过，下一步只实现和发布冻结四臂G1；若G0失败，立即关闭D121当前revision并研发下一个原理候选。
