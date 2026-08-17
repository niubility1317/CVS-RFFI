# D92 E0 FULL D42 AFCP K10 G0本地发布报告

状态：`LOCAL_VERIFIED / NOT_RUN / NO_PERFORMANCE_RESULT`

本报告只登记一次固定`rx_7_7/seed713106/K10/new5`、三scene、truth-free技术G0的本地发布包。基于P0修复后的runtime commit`ae4a2a2a6b74eb26b68d28091f0df7eb1c65079a`封装；未执行SSH、SCP、N607 launch、analyzer、truth/scorer或性能读取。

独立复核：`P0=0 / P1=0 / APPROVE`（主代理转发的定点复核结果）。该复核不等于本run已执行或G0性能通过。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|`d92_e0_full_d42_afcp_g0_k10_20260817_v1`|
|runtime commit|`ae4a2a2a6b74eb26b68d28091f0df7eb1c65079a`|
|candidate|`d92_e0_full_d42_allclass_fold_consensus_plane`|
|reference arm|`E0_FULL_ONLY`|
|candidate arm|`E0_FULL_D42_ALLCLASS_FOLD_CONSENSUS_PLANE`|
|config source|`configs/stage2_d92_afcp_g0_v1.json`|
|runner source|`code/scripts/run_d92_afcp_g0.py`|
|outer|`rx_7_7__seed_713106__k_10__new_5`|
|persisted receipt binding|`receiver=7-7`、`seed=713106`、`k_shot=10`、`registered_class_count=11`、两臂同一`row_handle`、三scene support/query列表完全一致|
|scene|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|协议|`p2_min_v1`；复用已验证sealed package|
|GPU|GPU0|
|expected marker|`D92_AFCP_G0_ACTIVE_RESOURCE_PASS`|
|fresh-run retry|`false`；stop/no-retry|

## 2.本地发布包

repo release root：`E:/type10-7/code/snapshots/d92_125wt/automation_reports/CV-SincNet/d92_e0_full_d42_afcp_g0_k10_20260817_v1`

external mirror root：`E:/type10-7/automation_reports/CV-SincNet/d92_e0_full_d42_afcp_g0_k10_20260817_v1`

|artifact|relative path|size|SHA256|
|---|---|---:|---|
|runtime archive|`runtime/d92_afcp_g0_source_ae4a2a2a_20260817_v1.tar.gz`|244907 bytes|`10791446976987f57a058ca87d0782154e66d2ee16887a36d2070654aefb5e95`|
|launch|`launch.sh`|6618 bytes|`d92ea8b91a2830105863b91cfbdbf06856bee1ce10d34e8db1d23773f646464b`|

archive包含41个tar成员目录/文件项，其中37个唯一source member；仅记录archive与launch整体SHA/size，不建立逐成员SHA体系。完整source manifest见`DELIVERY_MANIFEST.txt`的`[source_manifest]`。

最小runtime import closure包含AFCP runner及其实际静态/动态依赖；`probe_d81`链所需的D80、D66、D62、D61、D46、D45、D44、D43探针和D80/D81 core均已纳入，未纳入数据、checkpoint、truth、scorer或大型输出。

## 3.冻结sealed输入与远端路径登记

|字段|值|
|---|---|
|source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_afcp_g0_source_ae4a2a2a_20260817_v1`|
|source archive|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_afcp_g0_source_ae4a2a2a_20260817_v1.tar.gz`|
|remote launch|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_afcp_g0_launch_ae4a2a2a_20260817_v1.sh`|
|output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_afcp_g0_k10_20260817_v1`|
|logs root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_afcp_g0_k10_20260817_v1`|
|local retrieval|`E:/type10-7/local_artifacts/d92_e0_full_d42_afcp_g0_k10_20260817_v1`|
|before enrollment seal|`e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9`|
|before apply seal|`736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473`|
|after enrollment seal|`2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286`|
|after apply seal|`afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a`|
|ground manifest|`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|

## 4.唯一detached command（仅登记，未执行）

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.sh >./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.out 2>./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.err </dev/null &
```

远端环境登记为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，`CUDA_VISIBLE_DEVICES=0`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`。预期生成source root、reference/candidate output、三scene`fit_audit.json`、两臂`after/execution_receipt.json`、`g0_validation.json`及日志；当前PID为`NOT RUN`。

## 5.本地验证

|检查|结果|
|---|---|
|固定HEAD与source member存在性|通过；`ae4a2a2a`，37个唯一source member|
|archive解包|通过；41个tar成员，含4个目录项|
|tar safety|通过；无绝对路径、`..`路径或`code/code`嵌套|
|extracted import closure|通过；AFCP core、Slim、Query、runner导入路径均位于解包root|
|extracted runner`--help`|通过；help面不含truth/scorer参数|
|config JSON|通过；schema、candidate、outer、K10、三scene、marker匹配|
|`bash -n launch.sh`|通过|
|external mirror cmp|通过；四项artifact逐字节一致|
|AFCP11 focused tests|未执行；按主代理“最小验证”指令不扩展测试面|

## 6.停止与证据边界

launch只允许一次固定driver；任一入口、协议、artifact闭合或固定marker技术失败即停止并保持`NO_PERFORMANCE_RESULT`，`fresh_run_retry=false`。不以accuracy、H、BA、floor、forgetting或任意性能值作停止/重试依据。未完成N607执行前，不声明G0技术通过、性能贡献、Target125结果或任何truth/scorer结论。
