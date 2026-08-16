# D92 E0 FULL CCOC K10 G0预登记与发布报告

状态：ARTIFACTS_COMPLETE / G0_MECHANISM_RESOURCE_PASS / NO_PERFORMANCE_RESULT

本报告记录Task3 v6的独立one-shot纯本地发布修复。v5 launch已通过seal校验后进入prediction前的persisted JSON检查，但错误读取不存在的顶层`marker`字段，因此v5技术失败、未执行runtime、无性能结果。v6只修正launch对已持久化JSON字段的检查并使用全新不可覆盖路径；scientific entry、方法、矩阵、所有门和v1-v5 artifact均保持不变。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|d92_e0_full_ccoc_g0_k10_20260816_v6|
|预登记完成时间|2026-08-16（Asia/Hong_Kong）|
|operator|primary agent；sole N607 runner待交接|
|scientific/G0 entry commit|2a77d164|
|release提交消息|chore: prepare D92 CCOC G0 v6 release|
|outer|rx_7_7__seed_713106__k_10__new_5|
|reference arm|E0_FULL_ONLY|
|candidate arm|E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS|
|scene|leo_clear_weak、leo_low_elev_weak、leo_rain_weak|
|协议|p2_min_v1；沿用VALIDATED_ONCE sealed package|
|expected_marker|D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS|
|fresh-run retry|false|
|GPU|GPU0；one-shot truth-free technical G0|
|环境|ssr-gpu；发布launch使用固定/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python|

v6不创建新的scientific entry commit。archive由entry `2a77d164`的Git对象生成；v1-v5 repo artifact、external mirror和runner handoff均未修改、覆盖或重标。

## 2.v5根因与v6最小修复

`run_d92_ccoc_g0.run()`先把artifact持久化，再返回对象并补加顶层`marker`。持久化JSON的冻结成功形状是：

```text
value.status == D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS
value.validation.marker == D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS
value.validation.pass == true
value.validation.scenes == {leo_clear_weak, leo_low_elev_weak, leo_rain_weak}
value.validation.scene_gates == 同组三scene且全部true
```

v5检查`value.get("marker")`，该字段在持久化artifact中不存在，导致唯一launch技术失败。v6 launch只检查上述persisted `status`、内部`validation.marker`、`validation.pass`、三scene集合和`scene_gates`全true，明确不检查top-level marker。四个actual seal SHA preflight及完整64位`after_apply`值保留。

outer、arms、三scene、rho、量子公式、GPU0、candidate registration peak target512KiB/hard1MiB、wall hard150ms、ratio1.50、query/state/MAC exact及其他G0门均不变。v6是第二个非科学发布缺陷后的更小独立one-shot入口，不预填运行通过。

## 3.冻结路径与sealed inputs

```text
source_root=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_2a77d164_20260816_v6
source_archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_2a77d164_20260816_v6.tar.gz
remote_launch=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_2a77d164_20260816_v6.sh
reference_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v6/reference_e0
candidate_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v6/candidate_ccoc
validation=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v6/g0_validation.json
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260816_v6
local_retrieval=E:/type10-7/local_artifacts/d92_e0_full_ccoc_g0_k10_20260816_v6
```

四个actual seal SHA：

```text
before_enrollment=e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9
before_apply=736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473
after_enrollment=2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286
after_apply=afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a
```

launch在解包、import和prediction前直接对sealed_job中的四个actual seal文件执行SHA等值检查；不带score/truth参数、不调用外部scorer、不读取性能结果、不执行第二次尝试。

## 4.TDD与验证

主代理预先加入的窄测试先在v6 launch不存在时运行，真实RED为1 failed，失败为明确的missing v6 launch artifact断言。生成launch后，窄测试GREEN为1 passed；完整G0聚焦套件为30 passed。

静态闭合均通过：`bash -n`、四seal 64位小写hex和精确值、persisted marker shape（含三scene及scene_gates全true）、py_compile、CLI `--help`、UTF-8/LF语义重读、tar安全、required entries、source manifest和竞争import。未执行launch、SSH、N607或任何runtime/性能读取。

## 5.源封存与archive闭合

archive由`git -c core.autocrlf=false archive`从entry `2a77d164` Git对象生成，未使用工作树文件或属性转换。source manifest列出34个源文件（manifest自身除外），抽取后`sha256sum -c`为34/34，逐成员与entry Git blob为34/34；archive包含LF `code/cvsrffi/__init__.py`。

|archive metadata|值|
|---|---:|
|size|215970 bytes|
|member count|39（35 files、4 directory entries）|
|SHA256|03721e4e082592dca6d8faf9716d2f2f70e9b6c14ad48bfef7c1ebd1bd699a38|
|source manifest SHA256|72ea5ad5be32bd3d91fa353bca1b64386d4ba2fa7ece2a3c386fc19e0ff65755|
|source manifest entries|34/34|
|archive与Git blob|34/34|
|init Git blob SHA256|13cc5247133854c79ed160269ee8fa9816cb8dae3d162e724ad86d0ad8fad7a2|
|tar safety|无绝对路径、..路径或code/code嵌套|

竞争import将v6 source/code置于路径首位、既有regular `cvsrffi`包置于第二位，`cvsrffi.stage2_d92_ccoc_g0`、E0D query和双执行entry均解析到v6 source。

## 6.launch与one-shot边界

唯一detached命令固定为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_g0_launch_2a77d164_20260816_v6.sh >./d92_ccoc_g0_launch_2a77d164_20260816_v6.out 2>./d92_ccoc_g0_launch_2a77d164_20260816_v6.err </dev/null &
```

启动前必须确认archive、launch、四个actual seal的size/SHA与本报告一致；source、output、logs、driver out/err和local retrieval根均不存在；同run进程为0；GPU0可用。启动后核验精确CWD/cmdline/run-root绑定、GPU0进程和日志增长。若错误hash、输出覆盖、query/truth协议违规、launcher-wide确定性异常、无prediction closure或非零退出，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`并保留artifact，fresh-run retry=false。

## 7.发布镜像与concerns

repo release根：`E:/type10-7/code/snapshots/d92_125wt/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260816_v6/`。

外部根：`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260816_v6/`，非Git，仅承载repo release的report、launch、DELIVERY_MANIFEST和runtime archive逐字节镜像，由repo release commit承载traceability。

v1-v5均未修改或覆盖；v5 persisted marker assertion技术失败仅保留为历史边界，无性能结果。预登记阶段的未执行concern已由本次唯一v6 runtime artifact闭合；expected marker现已由persisted JSON、driver与独立取回证据验证，但不构成性能结果。

## 8.Luna/max N607运行闭环

最终状态：ARTIFACTS_COMPLETE / G0_MECHANISM_RESOURCE_PASS / NO_PERFORMANCE_RESULT。

本节由Luna/max sole N607 runner在2026-08-16执行。唯一detached launch=1，fresh-run retry=false；没有第二次launch、没有远端写入或删除、没有方法/矩阵/门修改。启动SSH通道未返回可采信exit，记录为SSH exit=UNKNOWN；独立远端artifact闭合证明唯一命令已经落地并完成。

### 8.1PRECHECK与SYNC

|检查|结果|
|---|---|
|直连与身份|VERIFIED；普通账户szu2070436088、项目根与固定远端环境可见|
|同run启动前|source、archive、launch、output、logs、launcher driver out/err全部ABSENT；同run PID=0|
|GPU0启动前|RTX3090，compute apps=0，显存1MiB/24576MiB|
|archive落地|215970 bytes；SHA256=03721e4e082592dca6d8faf9716d2f2f70e9b6c14ad48bfef7c1ebd1bd699a38|
|launch落地|9322 bytes；SHA256=fe7a8a366540d1b992fb793e8982ac7b6e5273f3875d3f06cf01e1544e3194b4|
|archive与launch顺序|archive→launch；每次远端size/SHA复核通过|
|四actual seals|四个冻结SHA通过，after_apply=afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a|
|远端启动门|tar安全、39 members、required entries、bash -n、Python3.10.19、Torch2.1.0+cu121、CUDA8卡全部通过|

### 8.2唯一COMMAND、PID与GPU

唯一执行命令：

~~~bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_g0_launch_2a77d164_20260816_v6.sh >./d92_ccoc_g0_launch_2a77d164_20260816_v6.out 2>./d92_ccoc_g0_launch_2a77d164_20260816_v6.err </dev/null &
~~~

远端driver开始时间约为2026-08-16 20:29:46 CST。完成态对账显示g0_validation.json、source_manifest_check、import_closure、marker_check及两组arm输出均已生成；精确pgrep核验run/source均无活进程，GPU0=0%/1MiB，compute apps=0。

### 8.3G0技术artifact

persisted JSON核验结果：

|字段|值|
|---|---|
|status|D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS|
|validation.marker|D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS|
|validation.pass|true|
|scene集合|leo_clear_weak、leo_low_elev_weak、leo_rain_weak|
|scene_gates|三scene全部true|
|g0_driver.out/.err|6881 bytes / 0 bytes|
|launcher.out/.err|41 bytes / 0 bytes|

逐scene技术字段：

|scene|active/fallback|rho/state/state_bytes|margin/quantum|wall|ratio|registration peak bytes|target512KiB|hard1MiB|
|---|---|---|---|---:|---:|---|
|leo_clear_weak|true/true|true/true/true|4.6970918607/true|70.463259ms|1.126558|729088|false|true|
|leo_low_elev_weak|true/true|true/true/true|5.5597268259/true|69.751364ms|1.129636|98304|true|true|
|leo_rain_weak|true/true|true/true/true|4.1071622442/true|70.339019ms|1.142053|81920|true|true|

candidate的candidate_wall_p90=70.463259ms，candidate/reference ratio p90=1.142053，均通过wall<=150ms、ratio<=1.50。canonical candidate after fit_audit的registration_incremental_peak_working_set_bytes为clear=729088B、low_elev=98304B、rain=81920B，max=729088B；512KiB target在clear scene未通过，1MiB hard在三scene均通过，因此G0技术状态按hard resource gate成立。resource_audit中的persistent state字段不作为registration peak证据。artifact只暴露state/state_bytes布尔门，没有单独的逐scene state hash字段；冻结四actual seal SHA已在8.1记录。

query/state/MAC访问核验（candidate/reference、before/after均一致）：

|字段|值|
|---|---|
|query_decision_policy|per_sample_all_registered_classes|
|query_batch_global_assignment、query_class_quota_access、query_role_oracle_access|false|
|query_true_batch_class_count_access、query_query_graph_used、query_truth_present_in_predictor、query_truth_used_for_fit|false|
|query_fit/update/selection access|false|
|query_extra_macs_for_ground_component|0|
|ground_component_update_access|false|
|state、state_bytes、query_macs及各scene技术门|true|

本runner未读取或解释accuracy、H、BA、floor、forgetting、truth内容或scorer结果。

### 8.4RETRIEVAL与remote/local闭合

冻结local retrieval根：E:/type10-7/local_artifacts/d92_e0_full_ccoc_g0_k10_20260816_v6。canonical组件路径为source目录、output/d92_e0_full_ccoc_g0_k10_20260816_v6、logs/d92_e0_full_ccoc_g0_k10_20260816_v6和drivers目录；远端artifact保持原样。

manifest定义为按相对路径、文件size、文件SHA256排序后的LF行流；source使用显式LC_ALL=C规范化。逐行source manifest diff=0，四组件remote/local闭合如下：

|组件|files|bytes|remote/local tree SHA256|
|---|---:|---:|---|
|source|69|1956312|89d5c6b606b754bc3cb18a332e933e7bc60f2e2dde5312edb0a9df5f02a86877|
|output|21|1451638|a058caa3167e129d5cd8a64fbac7b656ce7b513a5ba0d3d799530b9f48271e44|
|logs|8|8657|1d0e9492a934fb6f95f31c350f59d1cd76f152a0a688a52b69216626a71f2126|
|drivers|2|41|7aa8c9fdc385a03b4bc73c6de02e6bef58855eb7b3843d4f9b88ac9b6453d8f7|

因output与logs的远端basename相同，首次递归SCP形成的合并副本保留在local retrieval根下，但不纳入上述canonical manifest；未删除任何本地或远端artifact。

### 8.5SSH清理与下一步

每次SSH/SCP后均检查本地ssh.exe与N607/bridge TCP22；最终无ssh.exe、无ESTABLISHED连接，仅可能存在TIME_WAIT。下一步仅由主代理基于本技术artifact决定是否进入分析；本run不构成性能结果或晋级结论。
