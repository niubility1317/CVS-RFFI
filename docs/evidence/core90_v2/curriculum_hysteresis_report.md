# V2课程hysteresis修复

修复前：配置定义identity_exit/margin_exit/next_identity_exit，但V2更新只检查enter，没有持久ready，阈值中间带每次清空确认。

修复后：首次满足全部enter时ready置真，之后处于enter/exit之间的有效新观测保持ready并累积确认；任一identity/margin/next_identity跌破exit时清空ready和streak。next_margin与next_worst_tx没有独立exit配置，保留已有硬保护下限，不添加新source阈值。invalid/stale/hold及policy不匹配均清空状态。每次实际升级后ready/streak重置。

保持原有三次新证据、250步cooldown及每次最大0.1增量；重复同观测不积累也不提交。ready写入checkpoint，缺少该状态的旧V2checkpoint显式拒绝，避免恢复时重解释hysteresis。仅有效satellite_policy变化触发reset_optimistic/invalidate_feature_cache。

修改仅`code/cvsrffi/game_tracking/curriculum.py`与`code/tests/test_game_tracking_curriculum_v2.py`，没有修改runtime/step_context。

TDD：新增5个用例初次失败（缺少ready字段）；修复后执行以下命令10项全部通过：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 -m pytest code/tests/test_game_tracking_curriculum_v2.py -q
```

覆盖中间带不能初次进入、已进入保持、三次新证据、exit重置、恢复一致、cooldown与重复观测不提交、有效policy变化触发原接口。
