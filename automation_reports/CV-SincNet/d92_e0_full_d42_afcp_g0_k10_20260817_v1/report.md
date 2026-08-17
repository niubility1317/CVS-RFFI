# D92 E0 FULL D42 AFCP K10 G0本地发布报告

状态：`REJECT_AFCP / NO_HARD9 / NO_TARGET125 / NO_PERFORMANCE_RESULT`

本报告记录一次固定`rx_7_7/seed713106/K10/new5`、三scene、truth-free技术G0。基于P0修复后的runtime commit`ae4a2a2a6b74eb26b68d28091f0df7eb1c65079a`封装并同步到N607；exact detached command仅执行一次，随后因AFCP support margin delta非有限而拒绝候选。未运行Hard9、Target125、analyzer、truth/scorer，未以任何性能字段作判定。

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

## 4.唯一detached command（已执行一次）

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.sh >./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.out 2>./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.err </dev/null &
```

远端环境为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，`CUDA_VISIBLE_DEVICES=0`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`。命令执行一次后进程退出；source root、reference/candidate output、三scene`fit_audit.json`、两臂`after/execution_receipt.json`和日志已生成，但`g0_validation.json`未生成。

## 5.执行与失败归类

|字段|值|
|---|---|
|launch owner|Luna/max；ordinary N607账号`szu2070436088`|
|远端启动状态|命令已执行一次；同run PID=0|
|GPU0终态|`memory.used=1MiB`、`utilization=0%`；已释放|
|marker|`D92_AFCP_G0_ACTIVE_RESOURCE_PASS`：missing|
|g0 validation|missing；未进入marker_check|
|技术错误|`D92 AFCP G0 failed: afcp support margin delta must be finite`|
|错误来源|`logs/g0_driver.err`，61 bytes；无Python traceback|
|停止判定|`REJECT_AFCP / NO_HARD9 / NO_TARGET125 / NO_PERFORMANCE_RESULT`|
|重试|`fresh-run retry=false`；不修复、不重启、不重试|

### 5.1三scene已有技术字段

以下只记录candidate after审计中的active/fallback/reason、guard、resource和query字段；`null`或`missing`按artifact原值记录，不补造未生成的字段。

|scene|active|fallback/reason|three-block|class guard|cross guard|support guard|margin delta/quantum|incremental peak B|wall ns|support transient B|support MACs|288-square B|query禁用字段|query MAC delta|state delta|
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
|`leo_clear_weak`|false|true / `positive_selection_boundary_tie`|false|false|false|false|`null` / false|475136|66537709|398112|696960|0|truth/fit/update/selection/role/quota/global均false|0|0|
|`leo_low_elev_weak`|false|true / `negative_selection_boundary_tie`|false|false|false|false|`null` / false|81920|65747030|398112|696960|0|truth/fit/update/selection/role/quota/global均false|0|0|
|`leo_rain_weak`|false|true / `positive_selection_boundary_tie`|false|false|false|false|`null` / false|61440|66670225|398112|696960|0|truth/fit/update/selection/role/quota/global均false|0|0|

三scene均为E0状态回退：`final_state_non_e0=false`、`modified_state_field_names=[]`、`persistent_state_bytes_delta=0`、`additional_full_fit_count=0`；query字段均保持false，未发现query真值/拟合/更新/选择/角色Oracle/配额/全局重排访问。reference arm的AFCP专属字段按schema为`missing`。未生成`g0_validation.json`，因此marker、统一scene_gates和最终outer binding均为`missing`，不能声明G0通过。

## 6.取回与完整性

远端artifact保留原位；已取回至`E:/type10-7/local_artifacts/d92_e0_full_d42_afcp_g0_k10_20260817_v1`：`source/`、`output/`、`logs/`、`launch.out`、`launch.err`。

|artifact|file count|total bytes|规范化内容树摘要|
|---|---:|---:|---|
|source|71|2207936|`c258d7923495610d00806a7a6d8461b1ea840dd69399c854228dbb88faa303c5`|
|output|20|1442736|`1b12af0c83dfb20761e36eb4d387e5c9dc6ee6589bfa4ac68a6d4195c15e5e7b`|
|logs|4|61|`055e4a317f1a344188847e9a4cdb5889ea1f5323bc4dfb66a988c71b3a24e445`|
|`launch.out`|1|0|`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`|
|`launch.err`|1|0|`e3b0c44298fc1c149afaf4c8996fb92427ae41e4649b934ca495991b7852b855`|

远端与本地的file count、total bytes和规范化`relative_path<TAB>sha256`内容树摘要一致。Windows/MSYS与Linux原生`sha256sum`的二进制标记差异已在对账中归一化，不影响内容一致性。

## 7.清理与后续

每次SSH/SCP后均确认本地`ssh.exe/scp.exe`进程为0、N607/bridge TCP22无ESTABLISHED；最终同run PID=0且GPU0释放。远端source/output/logs及launch out/err不删除、不覆盖。该候选不得进入Hard9或Target125；下一步只能由主代理在本地另行修复/设计并分配新的immutable run ID。

## 8.本地发布包与版本验证

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

## 9.停止与证据边界

launch只允许一次固定driver；任一入口、协议、artifact闭合或固定marker技术失败即停止并保持`NO_PERFORMANCE_RESULT`，`fresh_run_retry=false`。不以accuracy、H、BA、floor、forgetting或任意性能值作停止/重试依据。未完成N607执行前，不声明G0技术通过、性能贡献、Target125结果或任何truth/scorer结论。
