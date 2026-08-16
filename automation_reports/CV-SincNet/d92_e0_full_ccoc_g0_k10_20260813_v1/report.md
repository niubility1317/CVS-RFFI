# D92 E0 FULL CCOC K10 G0发布报告

状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / G0_MECHANISM_RESOURCE_PASS / NO_PERFORMANCE_RESULT`

本报告是Task3的机械发布记录。它只证明固定outer下三场景的support-side量子、状态、部署和资源门闭合，不构成性能、准确率、H、BA、floor、forgetting或unknown结论。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|`d92_e0_full_ccoc_g0_k10_20260813_v1`|
|科学/G0 entry commit|`ce7973a5`（完整commit：`ce7973a5`）|
|release提交消息|`chore: prepare D92 CCOC G0 release`；commit由本地release提交产生并记录于Task3完整报告|
|outer|`rx_7_7__seed_713106__k_10__new_5`|
|reference arm|`E0_FULL_ONLY`|
|candidate arm|`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`|
|scene|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|协议|`p2_min_v1`；沿用`VALIDATED_ONCE`sealed package|
|marker|`D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS`|
|fresh-run retry|`false`|
|GPU/环境|GPU0；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|

科学入口只包含四个新文件：

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d92_ccoc_g0.py`|`d08d87fc9e0babfd9e1d87f49fe171a2d7537d9aca83acb4c9ed046ad5740c2b`|
|`code/scripts/run_d92_ccoc_g0.py`|`f3b9552ce399967cf05086ea4a10f129b6980a9ace8a7c660eaa8899587e47da`|
|`tests/test_stage2_d92_ccoc_g0.py`|`f24cf44180e613b31cfff3c38aac4c96e05a6f0afafa6d810b8655e5e261746f`|
|`tests/test_run_d92_ccoc_g0.py`|`4c28cac2cb093eb547355ab8c23ac148f7436a4f57af287dae2bfe07820b9fb9`|

## 2.G0边界与实现

验证器只接收两次执行产生的support receipt和最终D42 state/resource audit，按canonical support identity、class registry、scene和row handle逐项连接。每个scene按

`M_j = score_true - max_opposite_group_score`

计算margin，再以四份scale maxima与support block幅度形成

`q = max_block(A_b × max(e0_scale1,e0_scale2,ccoc_scale1,ccoc_scale2))`。

`max|Delta M| >= q`按闭边界通过；`q<=0`、两个rho均在端点、状态SHA相同、fallback、actual candidate FULL fit非1、任一query访问布尔为true、wall/ratio/peak超限或support identity不一致均拒绝。reference资源不并入candidate资源门。验证器不写raw support、query或label artifact。

CLI严格固定outer和两arm，先执行reference，再执行candidate；两个输出subroot为：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/reference_e0
/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/candidate_ccoc
```

最终验证收据固定写到：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/g0_validation.json
```

CLI对三个输出根和验证文件均使用exclusive-create，存在即拒绝，不覆盖既有artifact。

## 3.输入sealed package与ground

sealed job：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5
```

四个package子目录为该job下`offline/predictor/before/enrollment_only`、`offline/predictor/before/apply_only_staging`、`offline/predictor/after/enrollment_only`、`offline/predictor/after/apply_only_staging`。四份seal SHA保持既有同outer收据：

|seal|SHA256|
|---|---|
|before enrollment|`e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9`|
|before apply|`736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473`|
|after enrollment|`2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286`|
|after apply|`afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a`|

ground component目录为：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component
```

其`manifest.json`SHA256为`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。这些路径只在运行时由sealed package接口读取；发布archive不携带数据、checkpoint、truth-sidecar或raw support。

## 4.源封存与发布路径

runtime archive由科学entry commit`ce7973a5`直接通过Git archive生成，未从工作树混入未提交文件。源目录和archive远端路径已展开为：

```text
source=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_ce7973a5_20260813_v1
archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_ce7973a5_20260813_v1.tar.gz
output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1
```

|archive metadata|值|
|---|---:|
|size|`214489` bytes|
|member count|`37`|
|SHA256|`bdf9ddddcc9fb08d754e9ca07cc50b411622f0fd1fb36dd9586b6d6b18037c44`|
|required entries|CCOC G0、E0D query、双执行entry齐全|
|tar安全|无绝对路径、`..`路径或`code/code`嵌套|

## 5.唯一launch命令

`launch.sh`只执行archive SHA/size/member-count与tar路径闭合、source import/compile闭合、三个新root不可覆盖检查、一次双执行G0命令和精确marker验证。它不接受变量化source路径，不调用scorer，不带性能读取参数，不执行第二次尝试。

唯一运行命令由固定CLI执行：reference=`E0_FULL_ONLY`，candidate=`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`，device=`cuda:0`，输出分别落到`reference_e0/`和`candidate_ccoc/`。日志路径为：

```text
/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/import_closure.out
/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/g0_driver.out
/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/marker_check.out
```

## 6.RED/GREEN与静态验证

真实RED先于实现执行：

```text
PYTHONPATH=code conda run --no-capture-output -n ssr-gpu python -m pytest -q tests/test_stage2_d92_ccoc_g0.py tests/test_run_d92_ccoc_g0.py
```

collection失败，分别报告`ModuleNotFoundError: cvsrffi.stage2_d92_ccoc_g0`和`ImportError: cannot import name run_d92_ccoc_g0`；未把预期缺失误报为测试失败。

GREEN及静态检查：

|检查|结果|
|---|---|
|G0模块/CLI聚焦测试|`15 passed`|
|Task2相邻CCOC/E0D测试|`56 passed`与`66 passed`|
|`py_compile`|通过|
|CLI`--help`|通过；帮助面不暴露truth/score参数|
|`git diff --check`|通过|
|archive抽取import closure|通过；导入路径均在冻结source root内|
|archive tar安全|通过；37成员且required entries齐全|
|`bash -n launch.sh`|通过|

## 7.接口说明与concerns

Task2的`technical_support_receipt_sink`原生只准许CCOC arm。为满足本Task3“reference与candidate均从最终registered D42 state取得同类support receipt”的接口，G0 CLI在reference单次调用期间对Task2的内部arm admission和support-receipt callback做了进程内、可恢复的窄适配：reference callback只返回空的CCOC扩展收据，调用结束立即恢复原对象；未修改Task2源码、科学公式、arm、outer、scene、rho、阈值或资源门。该点是唯一接口concern，若主代理要求Task2提供正式reference sink，应由主代理决定是否扩大接口；本机械实现不自行改变Task2科学代码。

当前未做SSH/SCP/N607 launch；外部根已由repo release文件创建四项逐字节镜像，并完成逐SHA核验。第二阶段release commit承载repo release文件，外部根本身非Git。任何后续执行若root已存在、archive SHA不符、import逃逸、marker缺失或任一技术门失败，必须非零退出并保留partial evidence，不能覆盖或重试同run ID。
