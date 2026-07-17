# SOMP-H offline manifest migration traceability

Source of truth: `E:\type10-7\项目.md`（2026-07-17）及本任务的offline迁移要求。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| OM-1 | 任务要求；项目.md 7.1、7.4 | cache matrix接受显式`cache_output_root`，默认仍兼容旧N607输出根，并保持5 receiver×6 seed的30-cell exact manifest | `code/cvsrffi/somph_cache_build_matrix.py`; `code/scripts/build_cvs_somph_cache_specs.py`; `tests/test_somph_cache_build_matrix.py` | verified | 79项合并聚焦pytest中的matrix/CLI/validator用例通过 | 只改变Phase1/offline规划，不扩大Phase2输入 |
| OM-2 | 任务要求；项目.md 7.1、12 | authority lock builder不再把旧manifest SHA作为生产唯一authority；必须读取并严格验证实际manifest exact schema、30-cell覆盖、cell/spec字节与canonical SHA、output root绑定 | `code/cvsrffi/somph_authority_lock_builder.py`; `tests/test_somph_authority_lock_builder.py` | verified | actual dynamic SHA/root通过；spec tamper fail closed | 旧固定root manifest仍由同一exact validator离线接受，不再作为唯一hash |
| OM-3 | 任务要求；项目.md 7.1、12 | production signer必须验证实际manifest及receipt/lock/cell/spec绑定，并把实际manifest SHA写入Ed25519 envelope；未签名的新manifest不能进入正式lineage bundle | `code/scripts/sign_cvs_somph_authority_lock.py`; `tests/test_sign_cvs_somph_authority_lock.py` | verified | actual manifest binding、Ed25519 message、tamper负测通过 | 私钥/openssl身份规则不变 |
| OM-4 | 任务要求；项目.md 7.1、12 | offline lineage writer不再硬编码旧manifest SHA；必须验证实际manifest并要求signed envelope、build receipt、manifest三方SHA一致 | `code/cvsrffi/somph_lineage_authority.py`; `tests/test_somph_lineage_authority.py` | verified | dynamic manifest SHA及错SHA/错签名负测通过 | 本任务未修改Phase2 predictor或D18 runner |
| OM-5 | AGENTS.md Version Management；技能reverse audit | 不覆盖共享脏树、不提交Git，完成后记录diff与聚焦验证 | 本文件及上述目标文件 | verified | `git status -sb`; `git diff --check`; 79 passed | 未运行真实IQ、未执行N607动作、未提交Git |

## Omission traps

- 仅增加CLI参数但未把值传入每个cell spec和manifest validator。
- builder接受任意自报SHA而没有验证30-cell exact schema和spec文件绑定。
- signer把实际SHA写进receipt但Ed25519 message仍使用旧固定SHA。
- lineage writer验证签名但未要求envelope中的manifest SHA等于实际manifest和build receipt。
- 为兼容旧manifest而保留“固定hash或跳过exact validation”的旁路。

## Verification record

- `python -m py_compile`：5个修改后的Python实现文件通过。
- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q tests/test_somph_cache_build_matrix.py tests/test_somph_authority_lock_builder.py tests/test_sign_cvs_somph_authority_lock.py tests/test_somph_lineage_authority.py`：79 passed。
- `git diff --check`：通过，仅有工作区既有LF/CRLF提示。
- 额外兼容套件`tests/test_somph_offline_target_package.py`在收集阶段被共享脏树中的`somph_predictor_bundle.py`未定义`lineage_authority`阻断；该文件不在本任务授权范围，未修改。
