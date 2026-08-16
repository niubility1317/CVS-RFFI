# D92 E0 FULL CCOC K10 G0预登记与发布报告

状态：LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT

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

v1-v5均未修改或覆盖；v5 persisted marker assertion技术失败仅保留为历史边界，无性能结果。当前唯一concern是尚未执行N607，状态保持LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT；expected marker不代表实际运行结果。
