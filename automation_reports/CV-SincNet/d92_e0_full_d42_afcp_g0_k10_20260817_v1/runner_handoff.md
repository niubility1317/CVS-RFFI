# AFCP G0 runner handoff

## Final status

`REJECT_AFCP / NO_HARD9 / NO_TARGET125 / NO_PERFORMANCE_RESULT`

Immutable run：`d92_e0_full_d42_afcp_g0_k10_20260817_v1`。本handoff由唯一N607 runner记录；不构成性能、truth/scorer或Target125结论。

## Precheck and sync

- ordinary账号：`szu2070436088`；直连N607；项目根可见。
- GPU0预检：1MiB/24576MiB、0%；同run进程0；7个冻结路径与本地取回根均ABSENT。
- release commit：`9847ec8c`；runtime commit：`ae4a2a2a6b74eb26b68d28091f0df7eb1c65079a`。
- archive：244907B，SHA256=`10791446976987f57a058ca87d0782154e66d2ee16887a36d2070654aefb5e95`，41成员且路径安全。
- launch：6618B，SHA256=`d92ea8b91a2830105863b91cfbdbf06856bee1ce10d34e8db1d23773f646464b`，本地/远端`bash -n`通过。
- archive和launch已按顺序SCP；远端保留，未覆盖既有artifact。

## Command and terminal state

exact detached command仅执行一次：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.sh >./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.out 2>./d92_afcp_g0_launch_ae4a2a2a_20260817_v1.err </dev/null &
```

- 远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU：GPU0。
- 同run最终PID=0；GPU0最终1MiB/0%。
- `D92_AFCP_G0_ACTIVE_RESOURCE_PASS`：missing；`g0_validation.json`：missing。
- `logs/g0_driver.err`：`D92 AFCP G0 failed: afcp support margin delta must be finite`；61B，无Python traceback。
- 不修复、不重启、不重试；未运行Hard9、Target125、analyzer、truth/scorer，未读取性能字段作判定。

## Candidate technical fields

|scene|active|fallback/reason|three-block|class guard|cross guard|support guard|margin delta/quantum|incremental peak B|wall ns|support transient B|support MACs|288-square B|query禁用字段|state delta|
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
|`leo_clear_weak`|false|true / `positive_selection_boundary_tie`|false|false|false|false|`null` / false|475136|66537709|398112|696960|0|truth/fit/update/selection/role/quota/global均false|0|
|`leo_low_elev_weak`|false|true / `negative_selection_boundary_tie`|false|false|false|false|`null` / false|81920|65747030|398112|696960|0|truth/fit/update/selection/role/quota/global均false|0|
|`leo_rain_weak`|false|true / `positive_selection_boundary_tie`|false|false|false|false|`null` / false|61440|66670225|398112|696960|0|truth/fit/update/selection/role/quota/global均false|0|

三scene均`final_state_non_e0=false`、`modified_state_field_names=[]`、`persistent_state_bytes_delta=0`、`additional_full_fit_count=0`；candidate query访问禁用字段均false，query MAC delta为0。未生成统一G0 validation，不能声明G0通过。

## Artifact retrieval

本地取回根：`E:/type10-7/local_artifacts/d92_e0_full_d42_afcp_g0_k10_20260817_v1`。远端source/output/logs及launch out/err均保留。

|artifact|count|bytes|normalized tree digest|
|---|---:|---:|---|
|source|71|2207936|`c258d7923495610d00806a7a6d8461b1ea840dd69399c854228dbb88faa303c5`|
|output|20|1442736|`1b12af0c83dfb20761e36eb4d387e5c9dc6ee6589bfa4ac68a6d4195c15e5e7b`|
|logs|4|61|`055e4a317f1a344188847e9a4cdb5889ea1f5323bc4dfb66a988c71b3a24e445`|
|launch.out|1|0|`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`|
|launch.err|1|0|`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`|

远端与本地的file count、total bytes和规范化`relative_path<TAB>sha256`内容树摘要一致。所有SSH/SCP连接均已退出；本地SSH/SCP进程为0，N607/bridge TCP22无ESTABLISHED。

## Next action

AFCP不得进入Hard9或Target125。若继续研发，主代理须在本地另行修复或设计新候选并分配新的immutable run ID；本run不可重用、不可重标、不可覆盖。
