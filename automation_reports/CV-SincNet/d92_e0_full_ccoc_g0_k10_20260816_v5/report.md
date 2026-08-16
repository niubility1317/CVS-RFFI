# D92 E0 FULL CCOC K10 G0预登记与发布报告

状态：LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT

本报告记录Task3 v5的不可覆盖纯本地发布修复。v4在真正prediction前由launch preflight正确停止：`after_apply` seal参数少了末尾字符`a`，因此没有执行N607、SSH/SCP或G0 runtime，也没有性能结果。v5只修正该发布边界并新增四个remote actual seal文件的SHA preflight，不改变scientific entry、方法、矩阵、阈值或运行ID历史。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|d92_e0_full_ccoc_g0_k10_20260816_v5|
|预登记完成时间|2026-08-16（Asia/Hong_Kong）|
|operator|primary agent；sole N607 runner待交接|
|scientific/G0 entry commit|2a77d164|
|release提交消息|chore: prepare D92 CCOC G0 v5 release|
|outer|rx_7_7__seed_713106__k_10__new_5|
|reference arm|E0_FULL_ONLY|
|candidate arm|E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS|
|scene|leo_clear_weak、leo_low_elev_weak、leo_rain_weak|
|协议|p2_min_v1；沿用VALIDATED_ONCE sealed package|
|expected_marker|D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS|
|fresh-run retry|false|
|GPU|GPU0；one-shot truth-free technical G0|
|环境|ssr-gpu；发布launch使用固定/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python|

v5不创建新的scientific entry commit。source archive继续由entry commit `2a77d164`的Git对象生成，v1-v4 release及其外部镜像保持不变。

## 2.本轮唯一发布修复

v4 launch在prediction前正确执行了seal校验，但其`--after-apply-seal-sha256`文字值缺少末尾`a`，为63字符，故v4未启动runtime。v5将该值固定为完整64位小写hex，并在任何解包或prediction前直接对四个remote actual seal文件执行`sha256sum`等值检查：

|参数|冻结SHA256|
|---|---|
|`--before-enrollment-seal-sha256`|`e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9`|
|`--before-apply-seal-sha256`|`736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473`|
|`--after-enrollment-seal-sha256`|`2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286`|
|`--after-apply-seal-sha256`|`afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a`|

outer、arms、三scene、rho、量子公式、GPU0、candidate registration 1MiB hard cap、target512KiB、wall hard150ms、ratio1.50、query/state/MAC exact及其他G0门均不变。v3仍为旧512KiB gate下的G0_RESOURCE_GATE_REJECT，v4保持其既有发布历史，不追溯改判。

## 3.冻结输入与输出路径

```text
reference_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v5/reference_e0
candidate_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v5/candidate_ccoc
validation=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260816_v5/g0_validation.json
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260816_v5
sealed_job=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5
ground_component=/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component
ground_manifest_sha256=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c
local_retrieval=E:/type10-7/local_artifacts/d92_e0_full_ccoc_g0_k10_20260816_v5
```

发布archive不携带数据、checkpoint、raw support或truth-sidecar。

## 4.TDD与静态验证

先新增窄测试读取v5 launch artifact并要求四个`--*-seal-sha256`参数均为64位小写hex且匹配冻结值。v5 artifact尚不存在时真实RED为1 failed，失败原因为明确的missing v5 launch artifact断言；不是收集、环境或方法失败。

生成v5 launch后同一窄测试GREEN为1 passed；完整G0聚焦套件为29 passed。`py_compile`、CLI `--help`、launch `bash -n`、四seal文本匹配、UTF-8/LF语义重读、tar安全、required entries、source manifest和竞争import闭合均通过。未执行launch、N607或任何runtime/性能读取。

## 5.源封存与archive闭合

runtime archive由entry commit对象`2a77d164`直接使用`git -c core.autocrlf=false archive`生成，未使用属性转换或工作树文件。source manifest列出archive内全部34个源文件（manifest自身除外），抽取后逐项`sha256sum -c`并与Git blob核验34/34；`code/cvsrffi/__init__.py`包含在archive内并保持Git blob SHA `13cc5247133854c79ed160269ee8fa9816cb8dae3d162e724ad86d0ad8fad7a2`。

```text
source_root=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_2a77d164_20260816_v5
source_archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_2a77d164_20260816_v5.tar.gz
```

|archive metadata|值|
|---|---:|
|size|215974 bytes|
|member count|39（35 files、4 directory entries）|
|SHA256|52e1488451222875e8ee71acc0d7d0f714af29af776760567d6312164fd80a95|
|source manifest SHA256|72ea5ad5be32bd3d91fa353bca1b64386d4ba2fa7ece2a3c386fc19e0ff65755|
|source manifest entries|34/34|
|archive与Git blob|34/34|
|tar safety|无绝对路径、..路径或code/code嵌套|
|required entries|CCOC G0、E0D query、双执行entry、`code/cvsrffi/__init__.py`与source manifest齐全|

抽取后竞争import回归将v5 source/code置于路径首位、现有regular `cvsrffi`包置于路径第二位，`cvsrffi.stage2_d92_ccoc_g0`、E0D query和双执行entry均解析到v5 source。

## 6.launch边界

launch只执行archive存在性、精确SHA/size/member count、tar path safety、四个remote actual seal SHA preflight、source manifest逐成员校验、解包后py_compile/import/package-boundary closure、三个不可覆盖检查、唯一固定双执行CLI命令和预期marker检查。source、archive、launch、output和logs绝对路径均已展开为entry短SHA与v5 run ID，不带score/truth参数，不调用外部scorer，不读取性能结果，不执行第二次尝试。

```text
remote_launch=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_2a77d164_20260816_v5.sh
launch_cwd=/home/szu2070436088/2510044040/CV-SincNet/runs
launch_driver_out=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_2a77d164_20260816_v5.out
launch_driver_err=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_launch_2a77d164_20260816_v5.err
```

唯一detached命令固定为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_g0_launch_2a77d164_20260816_v5.sh >./d92_ccoc_g0_launch_2a77d164_20260816_v5.out 2>./d92_ccoc_g0_launch_2a77d164_20260816_v5.err </dev/null &
```

启动前必须由普通N607账户确认archive、launch、四个actual seal的size/SHA与本报告一致；source、output、logs、driver out/err和本地取回根均不存在；同run进程为0；GPU0可用。启动后用短连接核验shell/child PID、精确CWD/cmdline/run-root绑定、GPU0进程和日志增长。G0为一次性技术执行，不以accuracy、H、BA、floor、forgetting或任何性能值停止或重试。

成功条件仅为三个scene全部技术门通过并产生预期marker；错误hash、输出覆盖风险、query/truth协议违规、launcher-wide确定性异常、无prediction closure或非零退出均标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，保留并取回现有artifact，fresh-run retry=false。

## 7.发布镜像与concerns

repo release根为`E:/type10-7/code/snapshots/d92_125wt/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260816_v5/`。外部根`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260816_v5/`非Git，仅承载repo release的逐字节镜像（report、launch、DELIVERY_MANIFEST和runtime archive）；由repo release commit承载traceability。

v1-v4 artifact均未修改；v4的63字符seal错误已由其自身prelaunch stop记录，v5仅修复launch文字值并增加actual seal preflight。当前concern仍是尚未执行N607，故状态保持LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT；expected marker只表示launch预期成功条件，不代表实际运行结果。
