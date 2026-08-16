# D92 E0 FULL CCOC K10 G0预登记与发布报告

状态：LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT

本报告记录Task3 v3的冻结机械入口、局部验证和不可覆盖发布物。当前run未执行N607、SSH/SCP或G0 runtime，因此不包含G0运行结果、性能结果、准确率、H、BA、floor、forgetting或unknown结论。

本v3 repair修复v2在archive与Git blob字节不一致后暴露的package boundary：补入tracked code/cvsrffi/__init__.py并锁定其冻结字节SHA 13cc5247133854c79ed160269ee8fa9816cb8dae3d162e724ad86d0ad8fad7a2；outer、arms、三scene、公式、阈值、GPU0、exact command与health-stop全部保持不变。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|d92_e0_full_ccoc_g0_k10_20260813_v3|
|预登记完成时间|2026-08-16（Asia/Hong_Kong）|
|operator|primary agent；sole N607 runner待交接|
|scientific/G0 entry commit|f05e25c4|
|release提交消息|chore: prepare D92 CCOC G0 release|
|outer|rx_7_7__seed_713106__k_10__new_5|
|reference arm|E0_FULL_ONLY|
|candidate arm|E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS|
|scene|leo_clear_weak、leo_low_elev_weak、leo_rain_weak|
|协议|p2_min_v1；沿用VALIDATED_ONCE sealed package|
|expected_marker|D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS|
|fresh-run retry|false|
|GPU|GPU0；one-shot truth-free technical G0|
|环境|ssr-gpu；发布launch使用固定/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python|

v3 repair entry commit只包含source manifest与Git-blob package-boundary regression test；科学/runner/validator/query代码均未改动：

|文件|SHA256（entry commit对象）|
|---|---|
|code/cvsrffi/stage2_d92_ccoc_g0.py|237ce1a89cf17dbf4f80263ba6d0b719510e299731c96c1c5a81555ea480165e|
|code/scripts/run_d92_ccoc_g0.py|ac0bbfee8c59b0d022ea381a9f8c453a240dbcc2cdf6c93316d91201fd1113c8|
|tests/test_stage2_d92_ccoc_g0.py|b808d170af2e6fc6b4f21fa0c5604a7fd4d0178bd13c7ae3aa0644fd5993f2a8|
|tests/test_run_d92_ccoc_g0.py|948925e8f5852a4cc516aaddd49392784476dd019fc20b2d75803f87dd66db5f|
|code/CCOC_G0_SOURCE_MANIFEST.sha256|f62a0d93557b1df371a7a9e993c30d1a73f9fc8739c164bbc65f262d5cd744ce|

## 2.G0边界与本轮机械修复

验证器仍只接收reference/candidate的support receipt与最终D42 state/resource audit，按canonical support identity、class registry、scene和row handle逐项连接；本轮没有改变outer、arm、scene、rho、量子公式、阈值、资源门或run ID。

- CLI删除E0到CCOC内部集合及私有callback的monkeypatch；两次执行都直接使用Task2正式technical_support_receipt_sink。
- peak门改为candidate自身registration_incremental_peak_working_set_bytes <= 512*1024，reference不再抵消candidate峰值。
- validator严格要求scene集合等于三项冻结G0_SCENES；reference/candidate两侧8个query/truth禁用字段全部进入门；输出candidate wall P90和candidate/reference ratio P90，使用三场景nearest-rank（即最大值）。
- prereg状态固定为LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT；只记录预期marker，不预填运行结论。

验证器不写raw support、query或truth artifact。reference资源不并入candidate资源门。CLI对两个输出subroot和最终验证文件使用exclusive-create，已存在时拒绝覆盖。

## 3.冻结输入与输出路径

outer=rx_7_7__seed_713106__k_10__new_5
reference_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v3/reference_e0
candidate_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v3/candidate_ccoc
validation=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v3/g0_validation.json
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v3

sealed_job=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5
ground_component=/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component
ground_manifest_sha256=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c
发布archive不携带数据、checkpoint、raw support或truth-sidecar。

## 4.源封存与archive闭合

runtime archive由entry commit对象f05e25c4直接使用git -c core.autocrlf=false archive生成，未使用属性转换或工作树文件。source manifest列出archive内全部34个源文件（manifest自身除外），抽取后逐项sha256sum -c并与Git blob核验34/34。

source_root=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_f05e25c4_20260813_v3
source_archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_f05e25c4_20260813_v3.tar.gz

|archive metadata|值|
|---|---:|
|size|215965 bytes|
|member count|39（35 files、4 directory entries）|
|SHA256|679ced78603fb69a4efe5ea85392fea5ea857c9143986cd5453ea6ac14462b12|
|source manifest entries|34/34|
|tar safety|无绝对路径、..路径或code/code嵌套|
|required entries|CCOC G0、E0D query、双执行entry与source manifest齐全|

v1的df17c06e archive因缺少code/cvsrffi/__init__.py在prediction前技术失败；v1 release及其远端artifact保持不变。v3仅修复archive package boundary字节一致性，未触碰v1或其他artifact。

## 5.launch边界

launch.sh只执行archive存在性、精确SHA/size/member count、tar path safety、source manifest逐成员校验、解包后import/compile closure、三个不可覆盖检查、唯一固定双执行CLI命令和预期marker检查。source绝对路径已展开为f05e25c4，没有占位符或变量化source/archive路径；不带score/truth参数，不调用外部scorer，不读取性能结果，不执行第二次尝试。

冻结落地与启动路径：

```text
remote_launch=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_f05e25c4_20260813_v3.sh
launch_cwd=/home/szu2070436088/2510044040/CV-SincNet/runs
launch_driver_out=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_f05e25c4_20260813_v3.out
launch_driver_err=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_f05e25c4_20260813_v3.err
local_retrieval=E:/type10-7/local_artifacts/d92_e0_full_ccoc_g0_k10_20260813_v3
```

唯一detached命令固定为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_g0_launch_f05e25c4_20260813_v3.sh >./d92_ccoc_g0_launch_f05e25c4_20260813_v3.out 2>./d92_ccoc_g0_launch_f05e25c4_20260813_v3.err </dev/null &
```

启动前必须由普通N607账户确认archive、launch的size/SHA与本报告一致；source、output、logs、driver out/err和本地取回根均不存在；同run进程为0；GPU0可用。启动后用短连接核验shell/child PID、精确CWD/cmdline/run-root绑定、GPU0进程和日志增长。G0为一次性技术执行，不以accuracy、H、BA、floor、forgetting或任何性能值停止或重试。

预期artifact包括reference_e0与candidate_ccoc各自的before/after prediction closure、fit/resource/execution audit，最终g0_validation.json，以及logs根中的source_manifest_check、import_closure、g0_driver、marker_check四组out/err。成功条件仅为三个scene全部技术门通过并产生预期marker；错误hash、输出覆盖风险、query/truth协议违规、launcher-wide确定性异常、无prediction closure或非零退出均标记STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT，保留并取回现有artifact，fresh-run retry=false。

## 6.RED/GREEN与静态验证

真实RED先于实现执行，命令串行使用conda run --no-capture-output -n ssr-gpu：

export PYTHONPATH=code
conda run --no-capture-output -n ssr-gpu python -m pytest -q tests/test_stage2_d92_ccoc_g0.py tests/test_run_d92_ccoc_g0.py

结果为6项预期失败：candidate peak绝对门、严格scene集合、reference query禁用字段、P90输出、正式sink调用、prereg状态各有一项失败；不是把预期缺失误报为测试收集失败，也不是仅检查finally恢复。

实现后同一聚焦套件为21 passed。相邻Task2/Query回归为56 passed与66 passed。其余静态闭合：py_compile、CLI --help、bash -n launch.sh、JSON/UTF-8语义重读、archive path safety、required entries、source manifest 34/34、抽取后required imports均完成。未执行launch、N607或任何runtime/性能读取。

## 7.发布镜像与concerns

repo release根为E:/type10-7/code/snapshots/d92_125wt/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v3/。外部根E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v3/非Git，仅承载repo release的逐字节镜像（report、launch、DELIVERY_MANIFEST和runtime archive）。最终文件SHA、size、镜像cmp结果和第二阶段release commit记录在Task3完整报告与code/SYNC_MANIFEST.txt。

v1与v2的技术失败concern已由v3纯Git blob package boundary修复；本轮没有扩大Task2科学代码接口。v3 archive逐成员与Git blob及source manifest均34/34一致。

traceability=LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT；release message=chore: prepare D92 CCOC G0 v3 release。

当前结论仅为本地机械发布准备完成，仍为NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT；预期marker不代表实际运行结果。
