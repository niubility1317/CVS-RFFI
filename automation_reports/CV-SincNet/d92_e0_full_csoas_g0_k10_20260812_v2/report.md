# D92 E0 FULL CSOAS K10 G0实验报告

状态：`ARTIFACTS_COMPLETE / G0_MECHANISM_RESOURCE_PASS / NO_PERFORMANCE_RESULT`

## 1.实验身份

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_csoas_g0_k10_20260812_v2`|
|科学commit|`b8ebd4f4522fcc3e9e6b7dd18d722c329021f181`|
|候选|`E0_FULL_CSOAS`；candidate=`d92_e0_full_csoas`；mode=`csoas_full`|
|目标|在最难固定K10 outer的三场景做truth-free机制与资源G0；不读取性能|
|比较目标|同一outer的不可变`E0_FULL_ONLY`D42 state/resource收据与truth-free prediction identity|
|声明|`DEVELOPMENT_ONLY_SUPPORT_G0 / NO_PERFORMANCE_RESULT`|
|fresh retry|`false`；同一run ID不得覆盖或重试|

## 2.方法与假设

CSOAS复用同一D81变换和Cauchy权重，分类均值保持E0非加权均值；仅将每类FULL协方差改为独立Cauchy加权中心、effective-DOF闭式OAS，再按old/new组内类均衡和固定`0.5/0.5`合成一次FULL公共协方差。成功路径实际FULL fit恰为1，query仍是一条288D D42仿射头，永久state与query MAC等同E0。

G0假设：三场景均active且无fallback；相同660个query token/scenario顺序下，CSOAS部署预测不与E0逐字节相同；wall P90不超过150ms，并且增量peak不超过同场景E0加512KiB。该非同一性门只证明部署方法实际生效，不读取truth也不证明准确率提升。

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

不可变E0预测identity：query tokens SHA256=`12c37dacd13d14afce9df59d34ee134bb2b7e965eac2447e4362604ee8249d40`；scenarios SHA256=`79dc3dcb37d5324ccb45128fadefcab180d9460de8d6d9009cbec368302d50d3`；predicted class handles SHA256=`d539001ece0319b967023ea05dea7764264a731daa44bf7882a45660ba183cc0`。哈希基于按原顺序转成字符串列表后的canonical JSON；不打开truth。

## 5.本地实现与验证

科学提交包含CSOAS core、D92 probe、E0D slim/query及对应测试共8文件。验证：四聚焦文件`96 passed`；四生产文件`py_compile`通过；`git diff --check`通过；独立审查`P0=0、P1=0、APPROVE`。

runtime archive：`d92_csoas_g0_runtime_b8ebd4f4.tar.gz`，6,137,499 bytes，1458 members，SHA256=`4b0b434a26b47511cb0ddeb9f2455bc81964d8fcef312e75e57879547b631ca5`；`launch.sh`为10,309 bytes，SHA256=`d60520af5c0e9d7019894361cc8ab81fade8c0f3ec4957fa5bb01b7f09dff3f7`。必需入口存在且无`code/code`嵌套。archive采用与既有单GPU G0相同的代码闭包边界；不含数据、checkpoint或truth sidecar。

发布验证器使用v1完整取回artifact完成RED→GREEN：旧验证器稳定复现`KeyError: final_coefficient_quantization_error_max`；v2验证器在同一artifact上生成`D92_CSOAS_G0_ACTIVE_NON_E0_RESOURCE_PASS`，并在替换为E0 prediction artifact的负例中按预期拒绝`deployed predictions are byte-identical to paired E0`。该修复不改变科学代码、方法参数、outer或资源门。

## 6.N607路径与唯一命令

|用途|路径|
|---|---|
|source|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v2`|
|archive|`<source>/d92_csoas_g0_runtime_b8ebd4f4.tar.gz`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_csoas_g0_k10_20260812_v2`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_csoas_g0_k10_20260812_v2`|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU0；`OMP_NUM_THREADS=2`；`MKL_NUM_THREADS=2`|

唯一detached命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v2 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

## 7.G0硬门与停止规则

- 三场景均`active=true`、`fallback=false`、reason为空；candidate/reference fit=`1/0`。
- after actual FULL=`1`、total=`2`，无BLOCK/LOO/Fisher；K≤2边界由本地测试证明alias，本G0为K10。
- candidate D42 state fingerprint逐场景不等于不可变E0 fingerprint；同时query token/scenario identity必须与E0一致，而predicted class handles canonical SHA必须不同。后者证明部署路径产生可观察决策变化，但不读取truth、不声称性能提升。
- state bytes=`8583`、query MAC=`11×288`，query所有禁用字段为false。
- wall P90 nearest-rank`≤150ms`（目标`≤120ms`）；每场景incremental peak`≤paired E0+512KiB`。
- 任一失败、wrong hash/CWD、输出覆盖风险、缺artifact或技术异常均停止并保留partial evidence，标记`NO_PERFORMANCE_RESULT`；不因性能值停止。

期望artifact：before/after的prediction、COMMIT、fit/resource/execution receipt，`logs/g0_validation.json`及driver/import/prediction日志。G0不运行scorer。

## 8.运行结果

v1已因发布验证器读取未持久化量化字段而技术停止；本v2只修复该证据源，科学commit、方法、outer和门槛均不变。本次唯一launch完成G0机制/资源闭环，明确不运行scorer、不读取性能指标，结论仍为`NO_PERFORMANCE_RESULT`。

### 8.1Preflight、落地与唯一launch

- 2026-08-12 19:56:45 CST direct preflight通过：普通账户`szu2070436088`、项目根目录可见，8张RTX3090均为0%利用率、1MiB占用；无本地残留`ssh.exe`或TCP22连接。
- launch前远端source/output/logs三root均`ABSENT`、无同run进程；仅创建source目录，按`archive→launch.sh`顺序SCP。archive与launch远端size/SHA分别为`6137499/4b0b434a26b47511cb0ddeb9f2455bc81964d8fcef312e75e57879547b631ca5`和`10309/d60520af5c0e9d7019894361cc8ab81fade8c0f3ec4957fa5bb01b7f09dff3f7`。
- 远端landing核验通过：archive为1458 members、无绝对路径、`..`路径或link成员，required entries齐全，`bash -n`通过；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`为Python 3.10.19，`CUDA_AVAILABLE=True`且device_count=8；四份seal和ground manifest SHA全部匹配。
- 唯一detached命令按冻结值执行一次，SSH返回`SSH_EXIT=0`，未重试、未重启、未换命令。任务在首次完整探针前已结束；因此没有可持久读取的run PID。固定CWD/cmdline由上节命令确定，最终`RUN_ACTIVE=0`且`nvidia-smi --query-compute-apps`为空，GPU已释放。探针曾遇到一次进程退出竞态`/proc/<pid>/cmdline: No such file`，仅为读取已结束进程的诊断竞态，不是run异常；`launch_driver.err`、`import_closure.err`和`prediction.err`均为空。

### 8.2三场景G0技术收据

`logs/g0_validation.json`的schema为`cvs.phase2.d92_csoas.truth_free_g0_validation.v2`，status为`D92_CSOAS_G0_ACTIVE_NON_E0_RESOURCE_PASS`。三场景的14个query禁用字段（fit/update/selection/truth/role-oracle/class-quota/global-reassignment及其`d92_csoas_*`镜像字段）均为`false`。

|scene|active|fallback|candidate/reference fit|FULL actual/total|candidate state SHA256|wall|incremental peak|paired E0 peak|peak gate|
|---|---|---|---:|---:|---|---:|---:|---:|---|
|`leo_clear_weak`|true|false|1/0|1/2|`d967d57e8b47fcb602d63f4b10c8c21c61c95f5e5d28fecad5e4697a29475b5c`|12.965436ms|876,544|1,327,104|PASS|
|`leo_low_elev_weak`|true|false|1/0|1/2|`e2c83efeac79e01d678c465b22fad699a9c78b2e9f05dd64e25266fae301f9bb`|12.726085ms|581,632|1,060,864|PASS|
|`leo_rain_weak`|true|false|1/0|1/2|`454df5a8ca58324c3c29d7c3b9035ef29bd1c020757b29bddac842fa9a5db7ac`|14.268484ms|458,752|57,344|PASS|

wall P90 nearest-rank=`14.268484ms`，目标`120ms`和硬门`150ms`均通过；三场景candidate state均不等于paired E0 state，且paired query-token/scenario SHA匹配、candidate prediction SHA=`dd25a86a6b080eb1f30b7d3bf5b19857c23e0870e9f0b121e95e24925738eb05`不等于E0 prediction SHA=`d539001ece0319b967023ea05dea7764264a731daa44bf7882a45660ba183cc0`。这些是机制/资源和artifact identity证据，不是准确率、H、BA、floor、forgetting或unknown性能。

### 8.3完整artifact取回与树校验

远端artifact未删除，完整取回至`E:\type10-7\local_artifacts\d92_e0_full_csoas_g0_k10_20260812_v2`。tree SHA按远端`find -type f -printf '%P\\n' | sort`顺序，对每行使用`relative_path  file_sha256\\n`后SHA256；远端与本地count、bytes、逐文件SHA及tree SHA一致。

|root|远端文件数|远端bytes|远端tree SHA256|本地文件数|本地bytes|本地tree SHA256|
|---|---:|---:|---|---:|---:|---|
|source|1429|73,632,132|`5814b80c56e397da21ff51626be482c4f2f2fede20dfbfe902461f9f22ac8f5e`|1429|73,632,132|`5814b80c56e397da21ff51626be482c4f2f2fede20dfbfe902461f9f22ac8f5e`|
|output|10|1,018,135|`afb51a16be729a46639a7b6d20d2cd383de89b88ecbcde49cf20310df68d774a`|10|1,018,135|`afb51a16be729a46639a7b6d20d2cd383de89b88ecbcde49cf20310df68d774a`|
|logs|5|2,597|`b5243b2a679fae2f014e3c47b05362cf2ff333d3f5a920e695fbe87f82eae1a6`|5|2,597|`b5243b2a679fae2f014e3c47b05362cf2ff333d3f5a920e695fbe87f82eae1a6`|

### 8.4最终边界

- 本run只证明冻结E0_FULL_CSOAS路径在K10、三LEO弱场景下完成active/non-fallback、paired-E0非同一性、query禁用和wall/peak资源门；不证明性能提升、未知拒识、Phase3协同或真实在轨能力。
- 无异常指纹、无P0/P1、无输出覆盖、无retry；远端source/output/logs保留，最终run/GPU/SSH/SCP/TCP22均清零。
- 下一步仅由主agent决定是否将该G0技术收据纳入后续分析；runner不做方法、阈值、矩阵或性能晋级决策。
