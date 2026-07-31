# D106真实集成r4技术失败报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 1.身份与目的

- run ID：`d106_real_integration_deefd57c_20260801_r4`
- 时间：2026-08-01
- operator：N607专属Terra Max runner
- artifact commit：`63cf0236d37915bb9fbf521eaa34f4c9c82dbfc0`
- release source commit：`deefd57c4185a5343f87772be78b5038c37e6217`
- 目的：在不打开source-held、Target或query truth的条件下，执行588条D104 L_s received-IQ→冻结checkpoint双视图tap→D106 RDCE asset/wire真实闭环。

## 2.落地与启动

- N607 root：`/home/szu2070436088/2510044040/CV-SincNet`
- run root：`runs/d106_real_integration_deefd57c_20260801_r4`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<run-root>/source`
- GPU：物理GPU0，`CUDA_VISIBLE_DEVICES=0`，进程内`cuda:0`
- PID：`3027186`
- log：`<run-root>/logs/run.out`
- log SHA256：`043ef9b131bf0d962ee3ee57b3e1851f558112f50639948e6ab15b627b5629a8`

实际子命令：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d106_real_integration.py --fixture ../input/d106_real_integration_fixture_deefd57c.json --output-dir ../output --device cuda:0
```

启动前direct preflight通过；四份传输SHA、6个source关键entry、D104 split关键entry、fixture canonical bytes/22字段/9组路径SHA绑定、198个Python文件编译均通过。`output=ABSENT`，GPU0无compute process。

## 3.失败证据

进程在3秒检查前退出，完整异常为：

```text
D106RealIntegrationError: fixture must be an absolute regular file
```

规范化异常指纹SHA256为`a4aacc5c2067ed79b056ecdfeb81722e68d873051a900eab602715aceba23789`。根因是交接命令给出相对`--fixture`，而release入口要求绝对正规文件。该故障发生在`load_fixture`入口，早于IQ读取、checkpoint forward、asset构建和任何性能路径。

|证据|结果|
|---|---|
|exit code|`UNKNOWN_NOT_CAPTURED`；nohup交接未封存exit code|
|output|`ABSENT`|
|result/completion|均`ABSENT`|
|run-owned process|0|
|GPU0退出后|0%/1MiB，无compute app|
|失败receipt SHA|`7b74a5c67512330e0f7e6f36b5cbd4082b5ee683beb8b5eaf59045c36dbd07ca`|
|SHA清单SHA|`b702a297d94dff0f923da356fee4119ba7a0a50913a7700e7aa1b99e0493843a`|
|SSH清理|`SSH_PROCESSES=NONE`、`N607_OR_BRIDGE_TCP22=NONE`|

小型证据封存在方法报告的`artifacts/remote_deefd57c/`。未读取accuracy、H、BA、floor、source-held、Target或query truth；没有性能结果，禁止进入分析或晋级。

## 4.处置

r4不得重启、续写、覆盖或删除。修复只发生在本地handoff、fixture和测试：新命令必须把`--fixture`及`--output-dir`都写成r5 run root内的绝对路径，并机械负测禁止`../`。完成独立审查和Git提交后，只能以新run ID`d106_real_integration_deefd57c_20260801_r5`发布。
