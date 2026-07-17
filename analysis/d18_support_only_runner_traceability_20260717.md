# D18-CMRAE strict-K10 support-only runner追溯表

日期：2026-07-17

范围：签名授权的三场景密封`enrollment_only` package上的D18-CMRAE strict-K10 support adaptation runner；不开放query、truth、scorer或125确认矩阵。产物可以作为formal support adaptation state，但不具有formal metric或performance claim权限。本轮只使用本地合成LEO_weak fixture验证控制流，不产生真实性能结论。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D18-R01 | `AGENTS.md`、`项目.md`7.1/7.1.1 | actual fixed received IQ逐row SHA绑定；不回源、不读取clean/source | `code/scripts/run_d18_support_only_cmrae.py` | verified | 合成E2E＋authority负测 | runner只通过authority-bound verified materializer接收IQ；仅复用D14的`_payload_rows/_base_feature`支持处理，不再调用D14旧`_load_enrollment` |
| D18-R02 | `项目.md`7.1.1 | 每scenario每类exact-K10独立physical sample；Before/After旧support精确复用；三scene physical ID、parent received-IQ SHA与overlay provenance两两互斥 | runner、runner tests | verified | `test_cross_scene_disjointness_covers_physical_parent_and_overlay`；合成E2E | signed preauthorized token/overlay/IQ/seed/assignment roots必须和actual materialization roots逐项一致 |
| D18-R03 | D18预登记设计 | 三scene统一候选仅`D18_Z0`、`D18_CMRAE_L0125`、`D18_CMRAE_L0250`；old/new等权；逐fold逐类、floor、H、joint、forgetting门；任一失败原子回退true Z0 | runner、runner tests | verified | `test_candidate_surface_is_exact_and_three_scene_atomic`；core selector测试 | runner直接调用core原子选择器，不重写门逻辑 |
| D18-R04 | `项目.md`10.3 | 仅对已经接收的单一LEO_weak IQ做确定性CMRAE变换；physical-batch-one backbone；派生变换不增加K、不生成另一LEO状态 | runner、runner tests | verified | 合成E2E；core聚焦测试 | 每个physical support只计一个view；重复资源计时是确定性复执行，不形成额外support |
| D18-R05 | D18 core state contract | Before/After状态真实序列化、外部SHA、allowlist、语义round-trip；实测serialized bytes执行adapter/full-state硬门 | runner、runner tests | verified | `test_state_payload_roundtrip_uses_actual_bytes_and_external_commit` | 使用core真实serialized payload字节，不以估算值替代256KiB硬门 |
| D18-R06 | `AGENTS.md`Phase2 authority | before/after各由atomic pinned-authority入口完成签名preflight与同FD IQ物化；只有两套token-sealed evidence均经finalizer返回formal launch=true、metric=false、`SUPPORT_ONLY_NO_QUERY_CLAIM`后才可selection/anchor/output | runner、runner tests | verified | `test_post_materialization_gate_rejects_metric_or_query_disjointness_claim`；`test_synthetic_signer_cannot_open_production_atomic_iq_or_output`；合成E2E | 调用顺序为before/after atomic materialize→before/after finalizer→post gate→selection；`formal_metric_claim_allowed=true`或伪造query disjointness=`PASS`均fail closed |
| D18-R07 | 用户资源要求 | 0参数、0epoch、无dense query图；输出MAC、host latency、CUDA/Python峰值、状态，并与identity-only单qKNN同row比较 | runner、runner tests | verified | 合成E2E resource/report闭环 | 仅证明资源审计实现可运行；合成数字不得用作真实Pareto结论，qKNN缺matched完整serializer时继续禁止直接状态Pareto结论 |
| D18-R08 | 完整证据要求 | 输出receiver/class/support清单、逐class与三scene Before/After support-held结果、完整selection日志、状态审计、自动报告与receipt；每个fixed-received-IQ view显式绑定parent SHA、CMRAE operator、固定无随机`view_seed=0`及不增加K | runner、runner tests | verified | `test_inventory_records_and_validates_every_fixed_received_iq_view`；合成E2E artifact闭环 | Z0逐row`post_reception_view_used=false`，正candidate为true；两者view count均为1且physical ID保持唯一 |
| D18-R09 | 当前真实D8b authority | 没有当前signed formal policy/authorization envelope的历史D8b package不得IQ-open、不得运行D18、不得开放query或125 | runner、runner tests | blocked | fail-before-IQ负测；未重跑真实D8b | 历史D8b在当前协议下属于`UNVERIFIED_UNDER_CURRENT_PROTOCOL`；不能用development-only或manifest自声明替代独立签名授权 |
| D18-R10 | Phase2 clean-unreachability修复 | runner API/CLI不接收authority bundle root或expected authority commit；仅接收path-free v2 authorization及其signed envelope；v1、旧API和path/build_spec/raw-member授权在IQ/output前fail closed | runner、runner tests | verified | `test_runner_old_authority_api_fails_closed`；`test_runner_rejects_legacy_or_pathful_authorization_before_iq_and_output`；源码静态审计 | runner不调用offline lineage verifier；authority commit只从已验证的签名授权audit取得并写入最终证据 |
| D18-R11 | P0 atomic authority API | 删除runner侧独立preflight、audit/capability handoff和test-key verifier注入；唯一入口为`materialize_somph_enrollment_with_signed_authority`；旧API及synthetic signer在IQ/output前拒绝 | runner、runner tests | verified | `test_runner_old_authority_api_fails_closed`；`test_synthetic_signer_cannot_open_production_atomic_iq_or_output`；源码静态审计 | 正向编排测试只mock完整atomic入口并返回测试内token-sealed evidence；不绕生产pinned key |
| D18-R12 | support-only claim边界 | selection anchor、support audit、state `COMMIT`与`RECEIPT.json`统一声明formal support adaptation state；metric=false、query opened=false、performance claim=false，且不得把未打开的support/query disjointness伪报为`PASS` | runner、runner tests | verified | claim-boundary gate负测；合成E2E逐state COMMIT/receipt审计 | support内部LOO数值只用于adapter选择，不是formal query metric、正式性能或support/query无交叠证据 |

## Authority接线

runner的唯一合法路径为：

1. before/after分别调用`materialize_somph_enrollment_with_signed_authority`，传入package、detached seal、formal policy、path-free authorization v2、signed envelope及外部SHA。原子入口内部先用pinned Ed25519 trust完成全部preflight，再在同一受控路径完成archive SHA、allowlist、`np.load`、payload/provenance、actual IQ/token/overlay/seed/assignment roots和exact-K检查。
2. `finalize_somph_enrollment_authority_after_materialization`分别一次性消费两套内部token-sealed evidence；普通dict、copy、重复消费、旧audit/capability handoff及self receipt调用面均拒绝。
3. 只有两套final audit均为`CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS`、`formal_launch_authority=true`、`formal_metric_claim_allowed=false`且`support_query_disjointness_status=SUPPORT_ONLY_NO_QUERY_CLAIM`后，runner才创建artifact、selection authority anchor和输出目录。该launch权限只授权生成formal support adaptation state，不授权formal metric或performance claim。

## 单一观测边界

- 合成fixture为三场景各自独立的固定received-IQ row，不从一个clean样本派生多个LEO场景，不读取clean/source及其衍生信号。
- Before为6个旧类×K10；After精确复用旧类并新增5个新类×K10；每个support token只对应一个physical sample、一个LEO_weak场景和一个post-reception view。
- CMRAE只是对已接收IQ的确定性变换；`additional_leo_channel_state_generation=false`，`post_reception_view_counts_as_additional_physical_sample=false`。
- 以上只验证协议与软件路径，不是项目真实数据结果，不提供old/new准确率或部署Pareto claim。

## 验证记录

```text
conda run -n ssr-gpu python -m py_compile code/scripts/run_d18_support_only_cmrae.py tests/test_run_d18_support_only_cmrae.py
conda run -n ssr-gpu python -m pytest -q tests/test_run_d18_support_only_cmrae.py
conda run -n ssr-gpu python -m pytest -q tests/test_stage2_cmrae.py tests/test_run_d18_support_only_cmrae.py tests/test_run_d17_support_only_sprtdr.py tests/test_run_d14_support_only_pairwise_fisher_guard.py
conda run -n ssr-gpu python -m pytest -q tests/test_somph_lineage_authority.py tests/test_somph_predictor_bundle.py
git diff --check -- code/cvsrffi/somph_predictor_bundle.py tests/test_somph_predictor_bundle.py code/scripts/run_d18_support_only_cmrae.py tests/test_run_d18_support_only_cmrae.py analysis/d8b_authority_preflight_traceability_20260717.md analysis/d18_support_only_runner_traceability_20260717.md
```

本轮path-free v2＋atomic authority＋support-only claim边界同步结果：runner focused=`20 passed`；`py_compile` PASS；`git diff --check` PASS，仅有既存LF/CRLF提示。下列既往联合结果未在本轮重跑：CMRAE＋D18＋D17＋D14=`49 passed`；lineage authority＋predictor=`66 passed`。

- D18 runner SHA256：`7c4c52b6bdf3ab8bce4e56a718dee93ea16d1204cb553c1544574f816a4a165b`
- D18 runner test SHA256：`c8bd92fa71a269a6a7645ce753ad8269076d8da98bc6ff5d6eeac4c9480f3430`

atomic authority rebase已完成：runner不再接收preflight返回值或capability；最终可持久化authority内容只来自finalizer返回的普通audit mapping。selection anchor、support audit、state COMMIT与receipt均把claim边界持久化为formal support adaptation only、metric=false、no query opened、no performance claim。

反向审计计数：verified=11，implemented=0，pending=0，blocked=1，deferred=0，rejected=0。当前代码已收口到可供独立红队审查的合成验证状态；真实D18仍受D18-R09阻断，直到独立授权方在Phase2外为合法真实package签发path-free v2 policy authorization envelope。即使完成合法support materialization，仍不得据此自动开放query、125或formal metric claim。
