# D92 E0 FULL CSOAS K10 G0实验报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 1.实验身份

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_csoas_g0_k10_20260812_v1`|
|科学commit|`b8ebd4f4522fcc3e9e6b7dd18d722c329021f181`|
|候选|`E0_FULL_CSOAS`；candidate=`d92_e0_full_csoas`；mode=`csoas_full`|
|目标|在最难固定K10 outer的三场景做truth-free机制与资源G0；不读取性能|
|比较目标|同一outer的不可变`E0_FULL_ONLY`D42 state/resource收据|
|声明|`DEVELOPMENT_ONLY_SUPPORT_G0 / NO_PERFORMANCE_RESULT`|
|fresh retry|`false`；同一run ID不得覆盖或重试|

## 2.方法与假设

CSOAS复用同一D81变换和Cauchy权重，分类均值保持E0非加权均值；仅将每类FULL协方差改为独立Cauchy加权中心、effective-DOF闭式OAS，再按old/new组内类均衡和固定`0.5/0.5`合成一次FULL公共协方差。成功路径实际FULL fit恰为1，query仍是一条288D D42仿射头，永久state与query MAC等同E0。

G0假设：三场景均active且无fallback，D42部署state相对同场景E0至少跨越一个持久化存储量子，wall P90不超过150ms，并且增量peak不超过同场景E0加512KiB。G0不证明准确率提升。

## 3.冻结输入与边界

|字段|值|
|---|---|
|outer|`rx_7_7__seed_713106__k_10__new_5`|
|scenario|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|protocol|`p2_min_v1`；沿用`VALIDATED_ONCE`sealed package，不重验数据|
|sealed job|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5`|
|ground|`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`|
|ground SHA|`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|
|query禁用|fit/update/selection/truth/role oracle/class quota/global reassignment全部false|
|禁止|clean/source/query truth/scorer/accuracy/H/BA/floor/forgetting读取|

四份seal SHA：before enrollment=`e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9`；before apply=`736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473`；after enrollment=`2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286`；after apply=`afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a`。

## 4.同outer不可变E0基线

|scene|E0 after-state fingerprint|E0 incremental peak bytes|
|---|---:|---:|
|clear|`f68f4dba37fc89d475d9a5d9444c6e314b11ebdc0063c8cff5cddfb657016bc9`|1,327,104|
|low_elev|`091154fd6b5c3786f7097865b9fd1b8ba4ebed0ab81ab12f787e6ee0d5037d11`|1,060,864|
|rain|`db62c36f8e88a7dac344c03fd7793f514423e77fb725809003bfdefe6e9abb2a`|57,344|

来源：已取回完整E0_FULL_ONLY target125同outer fit audit。G0只比较artifact identity/resource，不读取该run性能。

## 5.本地实现与验证

科学提交包含CSOAS core、D92 probe、E0D slim/query及对应测试共8文件。验证：四聚焦文件`96 passed`；四生产文件`py_compile`通过；`git diff --check`通过；独立审查`P0=0、P1=0、APPROVE`。

runtime archive：`d92_csoas_g0_runtime_b8ebd4f4.tar.gz`，6,137,499 bytes，1458 members，SHA256=`4b0b434a26b47511cb0ddeb9f2455bc81964d8fcef312e75e57879547b631ca5`；`launch.sh`为8,902 bytes，SHA256=`c49450d7e82b9fb3c927493dbb5e1f5a9935d3cf0cbd38a4e4e055efcb2b7374`。必需入口存在且无`code/code`嵌套。archive采用与既有单GPU G0相同的代码闭包边界；不含数据、checkpoint或truth sidecar。

## 6.N607路径与唯一命令

|用途|路径|
|---|---|
|source|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v1`|
|archive|`<source>/d92_csoas_g0_runtime_b8ebd4f4.tar.gz`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_csoas_g0_k10_20260812_v1`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_csoas_g0_k10_20260812_v1`|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU0；`OMP_NUM_THREADS=2`；`MKL_NUM_THREADS=2`|

唯一detached命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

## 7.G0硬门与停止规则

- 三场景均`active=true`、`fallback=false`、reason为空；candidate/reference fit=`1/0`。
- after actual FULL=`1`、total=`2`，无BLOCK/LOO/Fisher；K≤2边界由本地测试证明alias，本G0为K10。
- candidate D42 state fingerprint逐场景不等于不可变E0 fingerprint；final coefficient codec error有限且大于0，记录为持久化D42量子变化。这里不声称support-margin性能提升。
- state bytes=`8583`、query MAC=`11×288`，query所有禁用字段为false。
- wall P90 nearest-rank`≤150ms`（目标`≤120ms`）；每场景incremental peak`≤paired E0+512KiB`。
- 任一失败、wrong hash/CWD、输出覆盖风险、缺artifact或技术异常均停止并保留partial evidence，标记`NO_PERFORMANCE_RESULT`；不因性能值停止。

期望artifact：before/after的prediction、COMMIT、fit/resource/execution receipt，`logs/g0_validation.json`及driver/import/prediction日志。G0不运行scorer。

## 8.运行结果（待runner回填）

当前无PID、无远端同步、无性能结果。sole runner须回填preflight、SCP/hashes、CWD/cmdline/PID/GPU、三场景机制/资源表、artifact tree hash、SSH清理与最终状态。

## 9.Runner回填（2026-08-12）

### 9.1最终状态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。冻结exact detached命令只执行一次并自然退出；没有重试、重启或换run ID，`fresh_run_retry=false`。`prediction.out`报告truth-free prediction entry完成，但随后launch closure在生成`g0_validation.json`时抛出：`KeyError: 'final_coefficient_quantization_error_max'`。因此本run没有G0成功marker，也不构成性能结果；未运行scorer/analyzer，未读取accuracy、H、BA、floor或forgetting。

### 9.2Precheck与落地

- direct普通账号preflight通过：`N607`→`szu2070436088@172.31.111.215`，项目根可见，8张RTX3090可见且空闲；未使用admin或bridge。
- release仓`E:\type10-7\code\snapshots\d92_125wt`在启动前tracked clean，HEAD=`10c1107645d61c696cd8ac9894152da787bca1fd`。
- 远端source创建前确认source/output/logs及同run进程均不存在；只创建source root，未预创建output、logs或`source/code`。
- archive与launch依次SCP；远端大小/SHA256分别为`6137499/4b0b434a26b47511cb0ddeb9f2455bc81964d8fcef312e75e57879547b631ca5`与`8902/c49450d7e82b9fb3c927493dbb5e1f5a9935d3cf0cbd38a4e4e055efcb2b7374`；tar成员`1458`，绝对路径、`../`和`code/code`均为0；`bash -n`通过。
- 远端Python=`3.10.19`，torch=`2.1.0+cu121`，CUDA可用、8卡；四份seal和ground manifest SHA256全部匹配。

### 9.3唯一命令与即时健康

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

SSH返回0。命令自然完成过快，未捕获可持续主PID；即时及终态进程列表均无该run的bash/Python进程，GPU0回到`0%/1MiB`。`launch_driver.err`唯一异常为上述closure `KeyError`；`prediction.err`为空，`prediction.out`为399 bytes。

### 9.4G0技术收据（非性能）

|场景|active|fallback|candidate/ref fit|FULL actual/total|wall(ms)|incremental peak(bytes)|query禁用字段|备注|
|---|---:|---:|---:|---:|---:|---:|---|---|
|`leo_clear_weak`|true|false|1/0|1/2|13.129212|868352|全部false|收据存在，closure未封存marker|
|`leo_low_elev_weak`|true|false|1/0|1/2|12.966224|692224|全部false|收据存在，closure未封存marker|
|`leo_rain_weak`|true|false|1/0|1/2|14.481507|425984|全部false|收据存在，closure未封存marker|

三场景wall P90（nearest-rank=max）为`14.481507ms`；query MAC为`3168=11×288`，`query_decision_policy=per_sample_all_registered_classes`，query truth/fit/update/selection/role/quota/global-reassignment均为false，source/clean runtime access均为false。上述仅为运行健康、协议与资源收据，不是准确率或任何性能声明。

### 9.5完整取回与tree hash

本地取回根：`E:\type10-7\local_artifacts\d92_e0_full_csoas_g0_k10_20260812_v1`；远端保留未删除。canonical tree hash按相对路径、文件size及SHA256排序后计算，远端=本地：

|root|files|bytes|tree SHA256|
|---|---:|---:|---|
|source|1429|73628803|`3966300b82263e01401ec0c905d6506f01947a9bdfd848835706fcb11a81aef1`|
|output|10|1018139|`ee10d95dbfc2ebac906a1575068f474a5724105f18cc58db3dbafa07ba3d62e6`|
|logs|4|399|`c590aaa74575d482edeca90561a2ff7202fc5c98210c5ca98aea6dd45e193094`|

output包含before/after各自`COMMIT.json`、`execution_receipt.json`、`fit_audit.json`、`resource_audit.json`及prediction artifact；未生成`g0_validation.json`。

### 9.6清理与后续

每次SSH/SCP后本地`ssh.exe`/`scp.exe`均已退出，N607及bridge TCP22均无ESTABLISHED连接；本任务未使用bridge。远端run进程为0，GPU已释放。不得在此run上修复、重启、重试或覆盖；如需修复closure，必须由主agent本地修复并另行完成独立门禁、commit和全新run ID。
