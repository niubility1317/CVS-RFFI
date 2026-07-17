# D18-CMRAE固定接收IQ公共模幅度等化追踪

日期：2026-07-17

范围：最小support-only算法原语与合成验证；不接真实runner、query或正式确认矩阵。

声明边界：development-only；本轮不提交Git。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D18-01 | `项目.md`7.1/7.1.1 | runner authority证明sealed package、pre-open allowlist、clean不可达及不可改名pre-overlay物理根 | 后续真实runner/authority evidence | deferred | core module不能证明文件/loader/control-flow clean不可达，也不能证明调用方没有伪造或改名physical root | 当前core不得自称获得sealed-package/lineage authority；只验证调用方提供的实际IQ SHA、token集合与runtime/checkpoint/code绑定。任意伪K10输入或物理根改名必须由runner pre-open authority anchor阻断 |
| D18-01a | D18 core边界 | 普通数组不能绕过，core只接受内部token构造且实际IQ SHA绑定的固定接收观测artifact；逐row绑定overlay token、canonical overlay provenance SHA、source LEO provenance SHA、source LEO cache SHA和satellite seed | `code/cvsrffi/stage2_cmrae.py`; `tests/test_stage2_cmrae.py` | verified | `test_received_iq_artifact_is_internal_sha_bound_and_ordinary_array_fails` | overlay token是普通非空token而不是伪SHA；其余SHA字段执行hex语义校验。core验证调用方字段及外部anchor一致性，这不是runner authority替代品 |
| D18-02 | D15-D17回顾/设计红队 | 共享、类平衡、old-only拟合的低频模幅度DCT8等化状态 | 同上 | verified | `test_dct8_is_class_balanced_scale_equivariant_and_zero_safe` | 频率median中心；类内median后跨旧类median；正交DCT-II索引1-8 |
| D18-03 | D18预登记设计 | 候选固定为Z0、`lambda=0.125`、`lambda=0.25`，`tau=log(1.10)` | 同上 | verified | `test_preregistered_surface_is_exact`及恶意hyperparameter测试 | 不开放任意lambda、tau或DCT rank |
| D18-04 | `项目.md`7.1.1/设计红队 | 不直接旋转非零FFT bin相位，不估计/去旋CFO，不产生第二LEO状态，计算view不增加K | 同上 | verified | `test_equalizer_preserves_fft_bin_phase_and_gain_is_really_bounded` | 固定`fftshift→DCT-II→ifftshift`；逐row恢复输入RMS；不声明时域CFO不变 |
| D18-05 | D18预登记设计/runner资源红队 | Before只用old support拟合共享equalizer；After冻结equalizer和旧prototype，只对new mask执行backbone并追加new prototype | 同上 | verified | `test_after_freezes_equalizer_and_old_prototypes_new_only_changes_new`; `test_enrollment_forwards_each_unique_physical_support_exactly_once` | old/new prototype均使用同一“变换后feature均值再L2归一”规则；旧prototype与旧score列逐位相同；最终enrollment每个unique physical support恰好一次backbone，After不重复计算old support |
| D18-06 | 统一K-shot协议/独立红队 | K1强制effective true Z0；K2-K4拒绝；K5严格exact-K；K10使用5折outer L2O；K1/K5不得独立选参或独立取support | 同上 | verified | `test_k1_is_bitwise_z0_k2_to_k4_closed_k5_exact`; `test_k1_positive_selector_lock_preserves_lock_identity_but_executes_z0`; `test_redteam_independent_k5_without_k10_lock_is_rejected`; `test_k10_lock_certificate_direct_sign_copy_replace_and_tamper_are_rejected` | 证书类型与公开K10 selector由一次性factory共同创建，factory随后从模块命名空间删除；token和issuer只存在selector闭包内，模块无token/issuer属性。证书不可直接构造、copy/deepcopy或dataclass replace。K10证书绑定runner提供的`selection_authority_anchor_sha256`；K1/K5必须显式提供并匹配同一expected anchor，同时绑定K10 selected candidate及自身rank前1/前5 SHA。证书不披露第6-10条记录。该core闭包token和外部anchor一致性检查都不能替代sealed runner authority。K1 state/trace同时记录`locked_k10_candidate_id`与effective`D18_Z0` |
| D18-07 | 用户目标/回顾门/独立红队 | K10三scene统一lambda；跨scene物理复用判据仅为physical ID、parent received-IQ SHA、overlay token、canonical per-row overlay provenance SHA；逐fold逐class记录完整门；old/new同等重要 | 同上 | verified | `test_three_scene_atomic_selector_uses_one_lambda_and_any_missing_scene_fails`; `test_redteam_cross_scene_reused_physical_id_is_rejected`; `test_redteam_cross_scene_parent_or_overlay_reuse_is_rejected`; `test_shared_source_authority_cache_and_satellite_seed_bind_but_do_not_define_physical_reuse`; `test_candidate_ranking_is_invariant_to_swapping_old_and_new_floors` | source LEO provenance/cache SHA与satellite seed纳入support/prefix/outer-train selection SHA和状态一致性绑定，但允许跨scene共享，不能把它们误当成物理样本复用判据。core只验证输入token集合的互斥，不能证明pre-overlay根未被改名；该authority留D18-01。性能门要求每fold逐类、floor、H、joint、forgetting不得退化；每scene聚合10个held后old floor、new floor、H分别严格改善；三scene全过才准正候选，否则统一Z0。eligible排序首先使用`min(worst_old_floor,worst_new_floor)`，随后依次使用跨scene最差H、平均H、joint、负forgetting，最后才以较小lambda定序；互换old/new floor不改变ranking key |
| D18-08 | `AGENTS.md`query边界 | 预测逐样本在全部注册类自主决策；无query标签、角色、quota、batch assignment接口 | 同上 | verified | `test_single_sample_prediction_scores_all_registered_without_truth_surface` | `predict_scores`只收state、sealed IQ artifact和sealed backbone；state authority固定`development_diagnostic_only` |
| D18-09 | 星上资源目标/独立红队/runner资源红队 | 0参数、0epoch、无dense query图；CMRAE额外状态小于16KiB；完整真实serialized state硬门不超过256KiB；MAC/FFT/状态口径诚实 | 同上 | verified | `test_state_self_seals_roundtrips_and_malicious_resource_cannot_reseal`; `test_full_serialized_state_over_256k_is_rejected_on_write_and_load`; `test_enrollment_forwards_each_unique_physical_support_exactly_once` | adapter与完整state分列；serialize和load都检查真实payload字节。资源分别报告1次forward FFT和1次inverse IFFT；最终enrollment按unique support计数且旧support零重复。outer L2O多fold/多candidate重复forward明确标为development selection cost、`deployment_resource_evidence=false`；真实时延/显存仍留runner |
| D18-10 | 状态安全/设计红队 | 状态内容SHA自封存、只读、外部serialized SHA、exact-schema round-trip；`OPERATOR_ID`、selection authority anchor与全部SHA语义校验；恶意候选、资源或绑定漂移必须拒绝 | 同上 | verified | 同上；序列化损坏、伪operator、非hex SHA、anchor不匹配、content SHA与resource自报攻击均拒绝 | support/prefix/outer-train selection SHA绑定label、rank、physical ID、parent IQ SHA、overlay token、canonical overlay provenance SHA、source provenance SHA、source cache SHA和satellite seed。outer fold state SHA同时绑定train support selection清单SHA、train old equalizer和train old/new prototype；仅改变held IQ及其parent SHA不改train selection/state SHA |
| D18-11 | 本轮边界 | 不接真实runner/query，不写正式性能或部署成功声明，不提交 | 本trace | verified | 本子任务只修改/新增3个D18所属文件；共享工作树存在其他agent或既有改动；未调用runner、N607或Git commit | 仅合成support与单样本接口验证 |
| D18-12 | 父任务明确边界 | D8b真实三scene strict-K10 support-only runner、实际性能和实测Pareto | 后续runner/report | deferred | 本轮禁止接真实runner | 这是下一步，不属于本最小原语任务 |

## 遗漏风险

- 仅实现等化函数但仍允许普通`ndarray`直接进入fit/predict，会绕过单观测与实际IQ SHA绑定。
- After重新拟合equalizer或重算旧prototype，会把新类注册变成旧模型漂移并掩盖遗忘。
- 把同一接收IQ的base/equalized结果作为两条support计数，会非法增加K。
- 只报告aggregate均值会掩盖scene、fold或floor类退化；本原语只能提供单scene完整fold证据，跨scene原子门必须由后续runner完成。
- FFT/IFFT不是零成本；资源报告必须把变换MAC近似、两次FFT和非MAC标量函数分开列出。

## 验证结果

```text
conda run -n ssr-gpu python -m pytest -q tests/test_stage2_cmrae.py
21 passed

conda run -n ssr-gpu python -m py_compile code/cvsrffi/stage2_cmrae.py tests/test_stage2_cmrae.py
PASS

git diff --no-index --check -- NUL <each-new-D18-file>
PASS
```

反向审计计数：verified=11，deferred=2，rejected=0，blocked=0。当前实现与“D18最小support-only算法原语”严格对齐；它不是完整D18真实实验实现，也没有取得sealed-package/clean不可达runner authority。最高风险项是D18-01与D18-12：后续runner必须先证明package authority与clean不可达，再在D8b真实三scene strict-K10 support-only数据验证floor和Pareto；未通过前不得打开query或125矩阵。
