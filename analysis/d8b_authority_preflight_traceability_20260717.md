# D8b authority-aware non-materializing preflight追踪

| ID | Requirement | Status | Verification | Claim boundary |
|---|---|---|---|---|
| AUTH-PF-01 | IQ archive/`np.load`前验证detached seal、manifest、allowlist与非IQ控制成员 | verified | 全局archive-open及`np.load`sentinel负测通过 | 预授权阶段不读取、CRC检查或物化IQ |
| AUTH-PF-02 | 使用真实lineage authority verifier及外部expected commit | verified | lineage+predictor联合59项通过 | lineage authority本身不授予formal权限 |
| AUTH-PF-03 | formal receiver、ManySig-old、ManyTx-new、TX集合、seed/cache scope一致 | verified | formal正测及D8b receiver=`1-20`负测通过 | D8b在IQ打开前稳定拒绝 |
| AUTH-PF-04 | 独立formal policy必须来自实际文件且其SHA被签名 | verified | policy内容漂移负测通过 | 不接受caller自写SHA代替实际文件hash |
| AUTH-PF-05 | policy authorization使用独立Ed25519域、pinned公钥和外部expected envelope SHA | verified | 真签名正测；签名、authorization root漂移负测通过 | 测试生成临时密钥但不绕过真实验签器 |
| AUTH-PF-06 | 签名绑定authorization、actual policy、package/seal、authority commit、formal row与code closure | verified | signature/binding drift四组负测通过 | 五文件代码闭包覆盖predictor、lineage、bundle、matrix、LEO cache |
| AUTH-PF-07 | 在IQ前从manifest+safe provenance重算class/role/physical/overlay/assignment roots | verified | package-control-root绑定正负测试通过 | 这些是预物化控制根，不冒充实际IQ根 |
| AUTH-PF-08 | 预授权只能返回`AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED` | verified | 正测断言formal/metric均为false | 仅`iq_open_authorized=true`，不能报告正式结果 |
| AUTH-PF-09 | 仅后物化finalizer可核验actual IQ、exact-K、scenario isolation、runtime binding并晋级 | verified | evidence-only晋级正测；普通dict/旧payload＋receipt接口、重复finalize均拒绝 | 成功后才返回`CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS`和formal true |
| AUTH-PF-10 | 不签生产证据、不建cache、不访问N607/私钥、不读取真实IQ | verified | 本轮仅本地合成fixture与单元测试 | 真实生产bundle/policy/receipt仍deferred |
| AUTH-PF-11 | preflight封存package class/physical/overlay/assignment及逐场景预授权根 | verified | IQ/token/overlay/seed/materialized-assignment逐场景root正测；内部伪造materializer负测 | actual roots逐项等于signed preauthorized roots，不只在finalizer内自洽 |
| AUTH-PF-12 | 仅preflight-bound verified materializer可same-FD打开三份support NPZ并生成不可复制evidence | verified | archive替换四组负测；copy/unissued/ordinary-dict evidence负测 | 私有闭包capability＋issued instance registry；payload使用bytes backing只读数组 |
| AUTH-PF-13 | finalizer仅接受内部token-sealed evidence，并复核package/seal/preflight/code/runtime全部绑定 | verified | evidence digest、一次性消费、root/runtime/code复核测试 | finalizer不再接受`materialized_payloads`或`materialization_receipt`调用面 |
| AUTH-PF-14 | pinned policy key由闭包/default immutable bytes捕获；测试只能使用显式test-verifier factory | verified | 修改生产`PINNED_*` globals仍不能使test-key envelope通过default verifier | 测试通过`_make_test_authority_preflight(public_key)`注入真实验签闭包 |
| AUTH-PF-15 | package root及成员拒绝parent目录symlink/junction/reparse路径 | verified | Windows真实parent symlink负测＋parent reparse marker负测 | 每次SOMP-H root打开均检查从root到volume anchor的ancestor chain |
| AUTH-PF-16 | D18严格按preflight→verified materializer→finalizer→selection/anchor接线 | verified | D18聚焦12项、CMRAE＋D18＋D17＋D14联合49项通过 | finalizer前禁止output mkdir、artifact、candidate selection和anchor；旧`_load_enrollment`及caller自签receipt已移除 |

## 稳定API

```python
preflight_somph_predictor_bundle_with_authority(
    package_root,
    *,
    detached_seal_path,
    expected_seal_sha256,
    authority_bundle_root,
    expected_authority_commit_sha256,
    formal_policy_path,
    formal_policy_authorization_path,
    signed_policy_authorization_envelope_path,
    expected_signed_policy_authorization_envelope_sha256,
) -> (manifest, seal, audit)
```

预授权成功状态为`AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED`，其中`iq_open_authorized=true`、`iq_payload_materialized=false`、`formal_launch_authority=false`、`formal_metric_claim_allowed=false`。

```python
materialize_somph_enrollment_with_authority(
    package_root,
    *,
    detached_seal_path,
    expected_seal_sha256,
    authority_preflight_audit,
) -> SomphMaterializedEnrollmentEvidence

finalize_somph_enrollment_authority_after_materialization(evidence) -> audit
```

verified materializer重新核验package/seal/preflight/code closure，随后在同一文件描述符上完成每个support NPZ的SHA、ZIP allowlist、`np.load`和payload/provenance交叉检查。它只在actual IQ/token/overlay/seed/assignment roots逐项等于signed preauthorized roots后签发闭包登记、只读、不可由普通dict或copy复制的evidence。finalizer一次性消费该evidence，复核package/seal/preflight/code/runtime及全部roots后才返回`CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS`。

## 验证记录

- `python -m py_compile ...`：PASS。
- `tests/test_somph_predictor_bundle.py`：49项PASS。
- lineage authority+predictor联合回归：66项PASS。
- D18 runner聚焦回归：12项PASS。
- CMRAE core＋D18＋D17＋D14 runner联合回归：49项PASS。
- authority module SHA256：`9b6fddb305645cb016660d934651e86886e76af61dff3192ad844e578aa6e6ac`。
- D18 runner SHA256：`da66a52b3b1c9f9e23fb817af0330369918670dfad1e8550504be9c73c2db240`。
- `git diff --check`：PASS，仅有既存LF/CRLF提示。

## 结论

D8b仍是development-only且无法在新gate下获得IQ打开授权。真实正式矩阵必须由离线builder生成正确receiver/ManySig-old/ManyTx-new package，并由独立授权方签署实际policy与authorization envelope；在实际IQ与exact-K审计完成前不得生成正式指标或运行声明。
