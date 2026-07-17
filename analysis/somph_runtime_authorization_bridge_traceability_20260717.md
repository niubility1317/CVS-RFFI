# SOMP-H离线v2授权签名桥追踪

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| B01 | 用户release Important修复 | 验证actual 30-cell manifest及对应cell authority bundle/外部commit | `code/scripts/sign_cvs_somph_runtime_authorization.py` | verified | manifest/commit/cache-root negative tests | verified cache文件或本地镜像均经完整loader和authority roots绑定 |
| B02 | 用户release Important修复 | 验证before/after sealed package、seal、stage、registration、K及同一receiver/seed | 同上 | verified | package-root及membership tests | 离线打开support control arrays，不向Phase2输出路径 |
| B03 | 用户release Important修复 | 验证formal policy v2和实际runtime code closure | 同上 | verified | 与predictor `_code_closure()`逐项相等 | closure只hash当前实际import执行三文件，无任意root参数 |
| B04 | 用户release Important修复 | 生成递归path-free authorization v2及signed policy envelope v2 | 同上 | verified | predictor shape/full binding tests | membership四字段进入签名authorization；support-only metric claim=false |
| B05 | 用户release Important修复 | 使用离线controller Ed25519私钥签名且私钥不进入输出 | 同上 | verified | wrong-private/no-output test | production signer闭包捕获pinned issuer/key/public key/verifier，无注入参数 |
| B06 | 用户release Important修复 | 错manifest、错包/root、错私钥全部fail closed且无输出 | `tests/test_sign_cvs_somph_runtime_authorization.py` | verified | 21项focused tests | 覆盖rename与parent fsync故障回滚、staging清理 |
| B07 | 独立release review | verified cache逐row唯一membership、IQ字节/SHA、seed、TX/class、role、exact K/rank、三scene物理ID互斥 | bridge及tests | verified | 9类membership negative controls及old-stability test | 真实cache overlay/seed/TX/rx/scenario被验证；opaque sid/oid仅作sealed内部一致性补充 |
| B08 | 独立release review | before/after锁feature runtime、method lock、Phase1及old registry前缀 | bridge及tests | verified | 4字段参数化negative tests | before/after旧类还要求同scene/class/rank真实sample_id一致 |

## Verification

- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\sign_cvs_somph_runtime_authorization.py`
- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\sign_cvs_somph_runtime_authorization.py --help`
- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q tests\test_sign_cvs_somph_runtime_authorization.py tests\test_sign_cvs_somph_authority_lock.py tests\test_somph_lineage_authority.py`→50 PASS
- 修改full-binding断言后单文件复跑→5 PASS
- 独立review修复后单文件复跑→21 PASS
- `test_sign_cvs_somph_runtime_authorization.py`、旧signer/lineage、predictor、D18 runner联合回归→153 PASS
- `git diff --check`→PASS

Reverse audit：8/8项verified；0项deferred、rejected、blocked。正式authorization明确`SUPPORT_ONLY_NO_QUERY_CLAIM`且`formal_metric_claim_allowed=false`，没有把未验证的support-query disjointness包装成正式指标权限。最高风险剩余项是尚未用真实30-cell artifact及离线生产私钥执行首次正式签发；本次按要求未读取真实私钥。
