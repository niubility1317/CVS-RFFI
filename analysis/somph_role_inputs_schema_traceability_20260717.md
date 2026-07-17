# SOMP-H真实role_inputs schema修复追踪

| ID | Requirement | Target | Status | Verification |
|---|---|---|---|---|
| R01 | exact schema加入4个真实exporter字段 | `somph_lineage_authority.py` | complete | 38项focused tests通过 |
| R02 | 校验assigned scene/policy/assigned count/reference excluded count | authority及builder | complete | negative tests通过 |
| R03 | role-input root绑定三场景真实shape而非忽略场景字段 | authority及fixtures | complete | bundle roundtrip tests通过 |
| R04 | builder严格传入build spec和required count并验证full/assigned counts | `somph_authority_lock_builder.py` | complete | builder tests通过 |
| R05 | 真实embedded manifest fixture回归，旧缺字段fail closed | 两个test文件 | complete | `python -m pytest -q tests/test_somph_lineage_authority.py tests/test_somph_authority_lock_builder.py`：38 passed |
