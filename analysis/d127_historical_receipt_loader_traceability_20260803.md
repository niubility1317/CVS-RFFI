# D127历史selected-IQ receipt兼容追踪

| ID | 来源要求 | 目标文件 | 状态 | 验证 |
|---|---|---|---|---|
| H1 | 固定历史sealed archive/receipt可读，不依赖当前运行时closure | `stage2_d127_phase1_release.py` | verified | `test_stage2_d127_phase1_release.py`14项通过 |
| H2 | completion、哈希、成员、数组、内容根、协议与访问字段严格闭合 | `stage2_d127_phase1_release.py` | verified | 6类语义及SHA/member/marker/canonical负例通过 |
| H3 | receipt内部execution结构闭合，唯独不等同当前代码重算 | module/tests | verified | r7冻结callable键；current键增删后历史fixture仍通过 |
