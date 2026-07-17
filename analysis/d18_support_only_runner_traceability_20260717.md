# D18-CMRAE strict-K10 support-only runner追溯表

日期：2026-07-17

范围：签名授权的三场景密封`enrollment_only` package上的D18-CMRAE strict-K10开发选择runner；不开放query、truth、scorer或125确认矩阵。本轮只使用本地合成LEO_weak fixture验证控制流，不产生真实性能结论。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D18-R01 | `AGENTS.md`、`项目.md`7.1/7.1.1 | actual fixed received IQ逐row SHA绑定；不回源、不读取clean/source | `code/scripts/run_d18_support_only_cmrae.py` | verified | 合成E2E＋authority负测 | runner只通过authority-bound verified materializer接收IQ；仅复用D14的`_payload_rows/_base_feature`支持处理，不再调用D14旧`_load_enrollment` |
| D18-R02 | `项目.md`7.1.1 | 每scenario每类exact-K10独立physical sample；Before/After旧support精确复用；三scene physical ID、parent received-IQ SHA与overlay provenance两两互斥 | runner、runner tests | verified | `test_cross_scene_disjointness_covers_physical_parent_and_overlay`；合成E2E | signed preauthorized token/overlay/IQ/seed/assignment roots必须和actual materialization roots逐项一致 |
| D18-R03 | D18预登记设计 | 三scene统一候选仅`D18_Z0`、`D18_CMRAE_L0125`、`D18_CMRAE_L0250`；old/new等权；逐fold逐类、floor、H、joint、forgetting门；任一失败原子回退true Z0 | runner、runner tests | verified | `test_candidate_surface_is_exact_and_three_scene_atomic`；core selector测试 | runner直接调用core原子选择器，不重写门逻辑 |
| D18-R04 | `项目.md`10.3 | 仅对已经接收的单一LEO_weak IQ做确定性CMRAE变换；physical-batch-one backbone；派生变换不增加K、不生成另一LEO状态 | runner、runner tests | verified | 合成E2E；core聚焦测试 | 每个physical support只计一个view；重复资源计时是确定性复执行，不形成额外support |
| D18-R05 | D18 core state contract | Before/After状态真实序列化、外部SHA、allowlist、语义round-trip；实测serialized bytes执行adapter/full-state硬门 | runner、runner tests | verified | `test_state_payload_roundtrip_uses_actual_bytes_and_external_commit` | 使用core真实serialized payload字节，不以估算值替代256KiB硬门 |
| D18-R06 | `AGENTS.md`Phase2 authority | IQ前完成signed policy preflight；只允许verified materializer同FD物化；只有一次性capability finalizer返回formal PASS后才可selection/anchor/output | runner、runner tests | verified | `test_authority_gate_precedes_any_iq_materialization`；`test_authority_failure_never_opens_or_crc_checks_iq_archive`；合成E2E | 调用顺序为before/after preflight→before/after materializer/finalizer→selection；旧payload dict＋self receipt接口已移除 |
| D18-R07 | 用户资源要求 | 0参数、0epoch、无dense query图；输出MAC、host latency、CUDA/Python峰值、状态，并与identity-only单qKNN同row比较 | runner、runner tests | verified | 合成E2E resource/report闭环 | 仅证明资源审计实现可运行；合成数字不得用作真实Pareto结论，qKNN缺matched完整serializer时继续禁止直接状态Pareto结论 |
| D18-R08 | 完整证据要求 | 输出receiver/class/support清单、逐class与三scene Before/After support-held结果、完整selection日志、状态审计、自动报告与receipt；每个fixed-received-IQ view显式绑定parent SHA、CMRAE operator、固定无随机`view_seed=0`及不增加K | runner、runner tests | verified | `test_inventory_records_and_validates_every_fixed_received_iq_view`；合成E2E artifact闭环 | Z0逐row`post_reception_view_used=false`，正candidate为true；两者view count均为1且physical ID保持唯一 |
| D18-R09 | 当前真实D8b authority | 没有当前signed formal policy/authorization envelope的历史D8b package不得IQ-open、不得运行D18、不得开放query或125 | runner、runner tests | blocked | fail-before-IQ负测；未重跑真实D8b | 历史D8b在当前协议下属于`UNVERIFIED_UNDER_CURRENT_PROTOCOL`；不能用development-only或manifest自声明替代独立签名授权 |

## Authority接线

runner的唯一合法路径为：

1. `preflight_somph_predictor_bundle_with_authority`验证detached seal、外部lineage authority、实际formal policy文件、authorization及Ed25519 envelope，只返回`AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED`，此时formal权限仍为false。
2. `materialize_somph_enrollment_with_authority`重新绑定package/seal/code closure，在同一文件描述符完成archive SHA、allowlist、`np.load`、payload/provenance、actual IQ/token/overlay/seed/assignment roots和exact-K检查。
3. `finalize_somph_enrollment_authority_after_materialization`一次性消费内部签发的evidence；普通dict、copy、重复消费及旧self receipt调用面均拒绝。
4. 只有final audit为`CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS`且formal/metric权限均为true后，runner才创建artifact、selection authority anchor和输出目录。

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

结果：runner focused=`12 passed`；CMRAE＋D18＋D17＋D14联合=`49 passed`；lineage authority＋predictor=`66 passed`；`py_compile` PASS；`git diff --check` PASS，仅有既存LF/CRLF提示。

- authority module SHA256：`9b6fddb305645cb016660d934651e86886e76af61dff3192ad844e578aa6e6ac`
- D18 runner SHA256：`da66a52b3b1c9f9e23fb817af0330369918670dfad1e8550504be9c73c2db240`
- D18 runner test SHA256：`8ad7702d4b7f700cf1ba728c8e1279dab7375b0f5af7db0a200ac848ac00c4b8`

反向审计计数：verified=8，implemented=0，pending=0，blocked=1，deferred=0，rejected=0。当前代码已收口到可供独立红队审查的合成验证状态；真实D18仍受D18-R09阻断，直到独立授权方为合法真实package签发当前policy/authorization envelope，并且该授权不得据此自动开放query或125。
