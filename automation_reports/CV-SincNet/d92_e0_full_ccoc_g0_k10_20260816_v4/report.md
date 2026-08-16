# D92 E0 FULL CCOC K10 G0预登记与发布报告

状态：LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT

本报告记录Task3 v4的冻结机械入口、局部验证和不可覆盖发布物。当前run未执行N607、SSH/SCP或G0 runtime，因此不包含G0运行结果、性能结果、准确率、H、BA、floor、forgetting或unknown结论。

v4是唯一获授权的最小registration resource gate修复：candidate registration_incremental_peak_working_set_bytes硬上限由512KiB调整为1MiB，以容纳729088边界值；target512KiB、wall hard150ms、candidate/reference ratio 1.50、query/state/MAC exact、outer、arms、三scene、rho、量子公式、GPU0、exact command与health-stop均保持不变。v3保留为旧512KiB门下的G0_RESOURCE_GATE_REJECT，不追溯改判。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|d92_e0_full_ccoc_g0_k10_20260816_v4|
|预登记完成时间|2026-08-16（Asia/Hong_Kong）|
|operator|primary agent；sole N607 runner待交接|
|scientific/G0 entry commit|2a77d164|
|release提交消息|chore: prepare D92 CCOC G0 v4 release|
|outer|rx_7_7__seed_713106__k_10__new_5|
|reference arm|E0_FULL_ONLY|
|candidate arm|E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS|
|scene|leo_clear_weak、leo_low_elev_weak、leo_rain_weak|
|协议|p2_min_v1；沿用VALIDATED_ONCE sealed package|
|expected_marker|D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS|
|fresh-run retry|false|
|GPU|GPU0；one-shot truth-free technical G0|
|环境|ssr-gpu；发布launch使用固定/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python|

entry commit只包含本轮G0峰值门和对应回归测试，以及source manifest的entry对象SHA更新；科学/runner/validator/query代码、矩阵和发布历史均未扩大：

|文件|entry commit对象SHA256|
|---|---|
|code/cvsrffi/stage2_d92_ccoc_g0.py|ed0d73b206c03903c66560e70b0b4069260f4f25d529db800988cd315ae54886|
|tests/test_stage2_d92_ccoc_g0.py|6f781ce60220cd1971fd188ea9992fa361d7c8d22f55a52f2b5b7ccf5b8a9a64|
|code/CCOC_G0_SOURCE_MANIFEST.sha256|72ea5ad5be32bd3d91fa353bca1b64386d4ba2fa7ece2a3c386fc19e0ff65755|

## 2.G0边界与本轮最小修复

验证器仍只接收reference/candidate的support receipt与最终D42 state/resource audit，按canonical support identity、class registry、scene和row handle逐项连接；本轮只改变candidate registration incremental peak hard cap。

- candidate registration_incremental_peak_working_set_bytes <= 1MiB；reference资源不抵消candidate峰值；target512KiB语义保留。
- wall hard limit仍为150ms，candidate/reference wall ratio hard limit仍为1.50。
- query MAC、persistent state和所有query/truth禁用字段仍为exact gate；三scene集合、active/no fallback、state非E0、actual candidate FULL=1、量子与support identity门均未改变。
- validator不写raw support、query或truth artifact；CLI仍只执行冻结outer、reference arm和candidate arm。
- v3仍代表旧512KiB gate下的G0_RESOURCE_GATE_REJECT，不将其运行前状态改写为pass；v4仅修复注册峰值资源边界。

## 3.冻结输入与输出路径

```text
outer=rx_7_7__seed_713106__k_10__new_5
reference_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v4/reference_e0
candidate_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v4/candidate_ccoc
validation=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v4/g0_validation.json
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260816_v4
sealed_job=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5
ground_component=/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component
ground_manifest_sha256=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c
local_retrieval=E:/type10-7/local_artifacts/d92_e0_full_ccoc_g0_k10_20260816_v4
```

发布archive不携带数据、checkpoint、raw support或truth-sidecar。

## 4.RED/GREEN与静态验证

TDD先以旧512KiB常量运行真实RED，命令串行使用ssr-gpu：

```bash
export PYTHONPATH=code
conda run --no-capture-output -n ssr-gpu python -m pytest -q -ra tests/test_stage2_d92_ccoc_g0.py tests/test_run_d92_ccoc_g0.py
```

RED为1项预期失败、27项通过：`test_candidate_peak_729088_passes_new_one_mib_hard_cap`因旧512KiB gate返回false；失败发生在行为断言，不是收集或环境错误。随后仅将`PEAK_DELTA_LIMIT_BYTES`改为`1024 * 1024`，同一聚焦套件GREEN为28 passed。

新增行为回归同时证明candidate峰值729088通过、1048577失败，且wall>150ms、ratio>1.50、query MAC不精确和state不精确仍分别拒绝。相邻回归串行结果为40 passed与70 passed。ssr-gpu解释器、py_compile和CLI `--help`均通过；未执行launch、N607或任何runtime/性能读取。

## 5.源封存与archive闭合

runtime archive由entry commit对象`2a77d164`直接使用`git -c core.autocrlf=false archive`生成，未使用属性转换或工作树文件。source manifest列出archive内全部34个源文件（manifest自身除外），抽取后逐项`sha256sum -c`并与Git blob核验34/34；`code/cvsrffi/__init__.py`包含在archive内并保持Git blob SHA `13cc5247133854c79ed160269ee8fa9816cb8dae3d162e724ad86d0ad8fad7a2`。

```text
source_root=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_2a77d164_20260816_v4
source_archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_2a77d164_20260816_v4.tar.gz
```

|archive metadata|值|
|---|---:|
|size|215978 bytes|
|member count|39（35 files、4 directory entries）|
|SHA256|9c80cdd7829ffac7fe336bb228f05391c4dba909f559179ea0229ba195c07a20|
|source manifest entries|34/34|
|archive与Git blob|34/34|
|tar safety|无绝对路径、..路径或code/code嵌套|
|required entries|CCOC G0、E0D query、双执行entry、`code/cvsrffi/__init__.py`与source manifest齐全|

抽取后竞争import回归将v4 source/code置于路径首位、现有regular `cvsrffi`包置于路径第二位，`cvsrffi.stage2_d92_ccoc_g0`、E0D query和双执行entry均解析到v4 source，结果为COMPETITION_IMPORT_OK。

## 6.launch边界

launch只执行archive存在性、精确SHA/size/member count、tar path safety、source manifest逐成员校验、解包后py_compile/import/package-boundary closure、三个不可覆盖检查、唯一固定双执行CLI命令和预期marker检查。source、archive、launch、output和logs绝对路径均已展开为entry短SHA与v4 run ID，不带score/truth参数，不调用外部scorer，不读取性能结果，不执行第二次尝试。

冻结落地与启动路径：

```text
remote_launch=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_2a77d164_20260816_v4.sh
launch_cwd=/home/szu2070436088/2510044040/CV-SincNet/runs
launch_driver_out=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_2a77d164_20260816_v4.out
launch_driver_err=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_2a77d164_20260816_v4.err
```

唯一detached命令固定为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_g0_launch_2a77d164_20260816_v4.sh >./d92_ccoc_g0_launch_2a77d164_20260816_v4.out 2>./d92_ccoc_g0_launch_2a77d164_20260816_v4.err </dev/null &
```

启动前必须由普通N607账户确认archive、launch的size/SHA与本报告一致；source、output、logs、driver out/err和本地取回根均不存在；同run进程为0；GPU0可用。启动后用短连接核验shell/child PID、精确CWD/cmdline/run-root绑定、GPU0进程和日志增长。G0为一次性技术执行，不以accuracy、H、BA、floor、forgetting或任何性能值停止或重试。

成功条件仅为三个scene全部技术门通过并产生预期marker；错误hash、输出覆盖风险、query/truth协议违规、launcher-wide确定性异常、无prediction closure或非零退出均标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，保留并取回现有artifact，fresh-run retry=false。

## 7.发布镜像与concerns

repo release根为`E:/type10-7/code/snapshots/d92_125wt/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260816_v4/`。外部根`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260816_v4/`非Git，仅承载repo release的逐字节镜像（report、launch、DELIVERY_MANIFEST和runtime archive）；由repo release commit承载traceability。

v1、v2、v3 artifact均未修改；v3仍为旧512KiB gate下的G0_RESOURCE_GATE_REJECT。v4的唯一concern是尚未执行N607，故当前仍为LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT；expected marker只表示launch预期成功条件，不代表实际运行结果。
