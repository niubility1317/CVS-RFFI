# D92 E0 FULL CCOC K10 G0预登记与发布报告

状态：LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT

本报告记录Task3的冻结机械入口、局部验证和不可覆盖发布物。当前run未执行N607、SSH/SCP或G0 runtime，因此不包含G0运行结果、性能结果、准确率、H、BA、floor、forgetting或unknown结论。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|d92_e0_full_ccoc_g0_k10_20260813_v1|
|scientific/G0 entry commit|df17c06e|
|release提交消息|chore: prepare D92 CCOC G0 release|
|outer|rx_7_7__seed_713106__k_10__new_5|
|reference arm|E0_FULL_ONLY|
|candidate arm|E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS|
|scene|leo_clear_weak、leo_low_elev_weak、leo_rain_weak|
|协议|p2_min_v1；沿用VALIDATED_ONCE sealed package|
|expected_marker|D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS|
|fresh-run retry|false|
|环境|ssr-gpu；发布launch使用固定/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python|

第一阶段entry commit只包含Task3四个code/test文件与source manifest：

|文件|SHA256（entry commit对象）|
|---|---|
|code/cvsrffi/stage2_d92_ccoc_g0.py|237ce1a89cf17dbf4f80263ba6d0b719510e299731c96c1c5a81555ea480165e|
|code/scripts/run_d92_ccoc_g0.py|ac0bbfee8c59b0d022ea381a9f8c453a240dbcc2cdf6c93316d91201fd1113c8|
|tests/test_stage2_d92_ccoc_g0.py|b808d170af2e6fc6b4f21fa0c5604a7fd4d0178bd13c7ae3aa0644fd5993f2a8|
|tests/test_run_d92_ccoc_g0.py|39812abe095e4e054413888d85f566fb4e359ca505bae6354a81fee0627f2628|
|code/CCOC_G0_SOURCE_MANIFEST.sha256|527cc9b102567c74a36ec19599e8ec6ac07f013f2a94ca7753a73ce1f3ba10a5|

## 2.G0边界与本轮机械修复

验证器仍只接收reference/candidate的support receipt与最终D42 state/resource audit，按canonical support identity、class registry、scene和row handle逐项连接；本轮没有改变outer、arm、scene、rho、量子公式、阈值、资源门或run ID。

- CLI删除E0到CCOC内部集合及私有callback的monkeypatch；两次执行都直接使用Task2正式technical_support_receipt_sink。
- peak门改为candidate自身registration_incremental_peak_working_set_bytes <= 512*1024，reference不再抵消candidate峰值。
- validator严格要求scene集合等于三项冻结G0_SCENES；reference/candidate两侧8个query/truth禁用字段全部进入门；输出candidate wall P90和candidate/reference ratio P90，使用三场景nearest-rank（即最大值）。
- prereg状态固定为LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT；只记录预期marker，不预填运行结论。

验证器不写raw support、query或truth artifact。reference资源不并入candidate资源门。CLI对两个输出subroot和最终验证文件使用exclusive-create，已存在时拒绝覆盖。

## 3.冻结输入与输出路径

outer=rx_7_7__seed_713106__k_10__new_5
reference_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/reference_e0
candidate_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/candidate_ccoc
validation=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/g0_validation.json
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1

sealed_job=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5
ground_component=/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component
ground_manifest_sha256=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c
发布archive不携带数据、checkpoint、raw support或truth-sidecar。

## 4.源封存与archive闭合

runtime archive由git archive从entry commit对象生成，明确使用git -c core.autocrlf=false archive保留index blob的LF字节；未从工作树打包。source manifest列出archive内全部33个Git源文件（manifest自身除外），并在抽取目录中逐项sha256sum -c核验33/33。

source_root=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1
source_archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1.tar.gz

|archive metadata|值|
|---|---:|
|size|215746 bytes|
|member count|38（34 files、4 directory entries）|
|SHA256|af9bcc8d75e12010ff8fa18ea4aad75d40ee2886744384bafe4ca23738e0f538|
|source manifest entries|33/33|
|tar safety|无绝对路径、..路径或code/code嵌套|
|required entries|CCOC G0、E0D query、双执行entry与source manifest齐全|

旧archive d92_ccoc_g0_source_ce7973a5_20260813_v1.tar.gz仅在本release的runtime目录被新entry archive替换，旧版本仍可从Git历史恢复；本轮没有触碰其他artifact，报告状态为prelaunch superseded。

## 5.launch边界

launch.sh只执行archive存在性、精确SHA/size/member count、tar path safety、source manifest逐成员校验、解包后import/compile closure、三个不可覆盖检查、唯一固定双执行CLI命令和预期marker检查。source绝对路径已展开为df17c06e，没有占位符或变量化source/archive路径；不带score/truth参数，不调用外部scorer，不读取性能结果，不执行第二次尝试。

## 6.RED/GREEN与静态验证

真实RED先于实现执行，命令串行使用conda run --no-capture-output -n ssr-gpu：

export PYTHONPATH=code
conda run --no-capture-output -n ssr-gpu python -m pytest -q tests/test_stage2_d92_ccoc_g0.py tests/test_run_d92_ccoc_g0.py

结果为6项预期失败：candidate peak绝对门、严格scene集合、reference query禁用字段、P90输出、正式sink调用、prereg状态各有一项失败；不是把预期缺失误报为测试收集失败，也不是仅检查finally恢复。

实现后同一聚焦套件为21 passed。相邻Task2/Query回归为56 passed与66 passed。其余静态闭合：py_compile、CLI --help、bash -n launch.sh、JSON/UTF-8语义重读、archive path safety、required entries、source manifest 33/33、抽取后required imports均完成。未执行launch、N607或任何runtime/性能读取。

## 7.发布镜像与concerns

repo release根为E:/type10-7/code/snapshots/d92_125wt/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v1/。外部根E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v1/非Git，仅承载repo release的逐字节镜像（report、launch、DELIVERY_MANIFEST和runtime archive）。最终文件SHA、size、镜像cmp结果和第二阶段release commit记录在Task3完整报告与code/SYNC_MANIFEST.txt。

唯一接口concern已由Task2正式sink闭合；本轮没有扩大Task2科学代码接口。archive生成时发现仓库core.autocrlf=true会把未显式关闭转换的archive源文件变为CRLF，已使用core.autocrlf=false重新从Git对象生成并完成33/33字节核验。

当前结论仅为本地机械发布准备完成，仍为NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT；预期marker不代表实际运行结果。
